import torch
import json
import os
import argparse
import numpy as np
from tgcn_model import GCN_muti_att
from sign_dataset import read_pose_file
from configs import Config
from segment import load_keypoints_from_session


def load_model(checkpoint_path, config_path):
    """Load the trained TGCN model"""
    # Load config to get model parameters
    configs = Config(config_path, mode='test')
    
    # Initialize model with same parameters as training
    model = GCN_muti_att(
        input_feature=configs.num_samples * 2,  # 2 for x,y coordinates
        hidden_feature=configs.hidden_size,
        num_class=configs.num_classes,
        p_dropout=configs.drop_p,
        num_stage=configs.num_stages
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model, configs


def load_keypoints_for_segment(session_dir, start_frame, end_frame, num_samples=50):
    """Load and process keypoints for a specific segment"""
    keypoints_list = []
    
    # Get all keypoint files for the session
    keypoint_files = []
    for filename in os.listdir(session_dir):
        if filename.endswith('_keypoints.json'):
            keypoint_files.append(filename)
    
    keypoint_files.sort()
    
    # Extract frames for this segment
    segment_frames = keypoint_files[start_frame:end_frame + 1]
    
    if len(segment_frames) == 0:
        return None
    
    # Sample frames if we have more than num_samples
    if len(segment_frames) > num_samples:
        indices = np.linspace(0, len(segment_frames) - 1, num_samples, dtype=int)
        segment_frames = [segment_frames[i] for i in indices]
    
    # Load keypoints for selected frames
    for filename in segment_frames:
        filepath = os.path.join(session_dir, filename)
        try:
            pose_data = read_pose_file(filepath)
            if pose_data is not None:
                # Ensure pose_data is the correct size (75, 2)
                if pose_data.shape[0] < 75:
                    # Pad with zeros
                    padded = torch.zeros(75, 2)
                    padded[:pose_data.shape[0], :] = pose_data
                    pose_data = padded
                elif pose_data.shape[0] > 75:
                    # Truncate to 75
                    pose_data = pose_data[:75, :]
                keypoints_list.append(pose_data)
            else:
                # If pose data is missing, use zeros
                keypoints_list.append(torch.zeros(75, 2))
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")
            keypoints_list.append(torch.zeros(75, 2))
    
    # Pad or truncate to num_samples
    while len(keypoints_list) < num_samples:
        keypoints_list.append(torch.zeros(75, 2))
    
    if len(keypoints_list) > num_samples:
        keypoints_list = keypoints_list[:num_samples]
    
    # Stack all keypoints into a tensor
    if keypoints_list:
        keypoints_tensor = torch.stack(keypoints_list, dim=0)  # Shape: (num_samples, 75, 2)
        
        # The model expects (batch_size, 55, input_feature) where input_feature = num_samples * 2
        # We need to reshape to (1, 55, num_samples * 2)
        # First, flatten the coordinates: (num_samples, 75, 2) -> (num_samples, 150)
        keypoints_flat = keypoints_tensor.reshape(num_samples, -1)  # (num_samples, 150)
        
        # Take only the first 55 keypoints (first 110 values) and reshape to (55, num_samples * 2)
        # We need to transpose and reshape carefully
        keypoints_55 = keypoints_flat[:, :110]  # Take first 55 keypoints * 2 coordinates = 110 values
        
        # Reshape to (num_samples, 55, 2) then transpose to (55, num_samples, 2)
        keypoints_reshaped = keypoints_55.reshape(num_samples, 55, 2).transpose(0, 1)  # (55, num_samples, 2)
        
        # Flatten the last two dimensions: (55, num_samples, 2) -> (55, num_samples * 2)
        keypoints_final = keypoints_reshaped.reshape(55, num_samples * 2)
        
        # Add batch dimension: (1, 55, num_samples * 2)
        keypoints_final = keypoints_final.unsqueeze(0)
        
        return keypoints_final
    else:
        return torch.zeros(1, 55, num_samples * 2)


def run_inference_on_segment(model, keypoints, top_k=5):
    """Run inference on a single segment"""
    with torch.no_grad():
        # Forward pass
        output = model(keypoints)
        
        # Get probabilities
        probabilities = torch.softmax(output, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, k=top_k)
        
    return top_indices[0], top_probs[0]


def load_label_mapping():
    """Load the mapping from index to gloss words"""
    # Read the asl100.json file to get gloss order
    with open('../../data/splits/asl100.json', 'r') as f:
        data = json.load(f)
    
    # Create index to gloss mapping
    label_mapping = {}
    for idx, item in enumerate(data):
        label_mapping[idx] = item['gloss']
    
    return label_mapping


def process_segments(session_dir, segments_file, model, configs, label_mapping, top_k=5):
    """Process all segments and run inference on each"""
    # Load segmentation results
    with open(segments_file, 'r') as f:
        segmentation_results = json.load(f)
    
    segments = segmentation_results['segments']
    results = []
    
    print(f"Processing {len(segments)} segments...")
    
    for segment in segments:
        segment_id = segment['segment_id']
        start_frame = segment['start_frame']
        end_frame = segment['end_frame']
        
        print(f"Processing segment {segment_id}: frames {start_frame}-{end_frame}")
        
        # Load keypoints for this segment
        keypoints = load_keypoints_for_segment(
            session_dir, start_frame, end_frame, configs.num_samples
        )
        
        if keypoints is None:
            print(f"Warning: No keypoints found for segment {segment_id}")
            continue
        
        # Run inference
        top_indices, top_probs = run_inference_on_segment(model, keypoints, top_k)
        
        # Convert to predictions
        predictions = []
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            gloss = label_mapping[idx.item()]
            predictions.append({
                "rank": i + 1,
                "sign": gloss,
                "probability": prob.item()
            })
        
        # Create segment result
        segment_result = {
            "segment_id": segment_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_frames": end_frame - start_frame + 1,
            "avg_motion_score": segment.get('avg_motion_score', 0.0),
            "predictions": predictions
        }
        
        results.append(segment_result)
        
        # Print top prediction
        top_pred = predictions[0]
        print(f"  Top prediction: {top_pred['sign']} ({top_pred['probability']:.3f})")
    
    return results


def save_results(results, output_file):
    """Save inference results to JSON file"""
    output_data = {
        "timestamp": str(np.datetime64('now')),
        "total_segments": len(results),
        "segments": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Run multi-segment inference with TGCN model')
    parser.add_argument('--session_dir', type=str, required=True,
                       help='Path to session directory with keypoints')
    parser.add_argument('--segments_file', type=str, default=None,
                       help='Path to segments.json file (default: session_dir/segments.json)')
    parser.add_argument('--config', type=str, default='archived/asl100/asl100.ini',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='archived/asl100/ckpt.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for results (default: session_dir/inference_results.json)')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions per segment')
    
    args = parser.parse_args()
    
    # Set default segments file if not specified
    if args.segments_file is None:
        args.segments_file = os.path.join(args.session_dir, "segments.json")
    
    # Set default output file if not specified
    if args.output_file is None:
        args.output_file = os.path.join(args.session_dir, "inference_results.json")
    
    # Check if files exist
    if not os.path.exists(args.session_dir):
        print(f"Error: Session directory not found: {args.session_dir}")
        return
    
    if not os.path.exists(args.segments_file):
        print(f"Error: Segments file not found: {args.segments_file}")
        return
    
    try:
        print(f"Loading model from {args.checkpoint}...")
        model, configs = load_model(args.checkpoint, args.config)
        print("Model loaded successfully!")
        
        print("Loading label mapping...")
        label_mapping = load_label_mapping()
        print("Label mapping loaded!")
        
        print(f"Processing segments from {args.segments_file}...")
        results = process_segments(
            args.session_dir, args.segments_file, model, configs, label_mapping, args.top_k
        )
        
        print(f"\nInference complete! Processed {len(results)} segments")
        
        # Save results
        save_results(results, args.output_file)
        
        # Print summary
        print("\nSummary:")
        for result in results:
            top_pred = result['predictions'][0]
            print(f"  Segment {result['segment_id']}: {top_pred['sign']} ({top_pred['probability']:.3f})")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return


if __name__ == '__main__':
    main() 