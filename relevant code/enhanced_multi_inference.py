#!/usr/bin/env python3

import torch
import json
import os
import numpy as np
from tgcn_model import GCN_muti_att
from configs import Config
import argparse

def load_model(checkpoint_path, config_path):
    """Load TGCN model from checkpoint."""
    configs = Config(config_path, mode='test')
    
    model = GCN_muti_att(
        input_feature=configs.num_samples * 2,  # 2 for x,y coordinates
        hidden_feature=configs.hidden_size,
        num_class=configs.num_classes,
        p_dropout=configs.drop_p,
        num_stage=configs.num_stages
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model, configs

def load_keypoints_for_segment(session_dir, start_frame, end_frame, num_samples):
    """Load and process keypoints for a specific segment."""
    from sign_dataset import read_pose_file
    
    pose_data = []
    frame_indices = list(range(start_frame, end_frame))
    
    # Sample frames if needed
    if len(frame_indices) > num_samples:
        step = len(frame_indices) / num_samples
        frame_indices = [frame_indices[int(i * step)] for i in range(num_samples)]
    elif len(frame_indices) < num_samples:
        # Pad by repeating the last frame
        last_frame = frame_indices[-1] if frame_indices else 0
        frame_indices.extend([last_frame] * (num_samples - len(frame_indices)))
    
    for frame_idx in frame_indices:
        json_file = f"{frame_idx:06d}_keypoints.json"
        json_path = os.path.join(session_dir, json_file)
        
        if os.path.exists(json_path):
            try:
                pose_tensor = read_pose_file(json_path)
                if pose_tensor is not None:
                    # Ensure consistent size (75, 2)
                    if pose_tensor.shape[0] < 75:
                        # Pad with zeros
                        padding = torch.zeros(75 - pose_tensor.shape[0], 2)
                        pose_tensor = torch.cat([pose_tensor, padding], dim=0)
                    elif pose_tensor.shape[0] > 75:
                        # Truncate to first 75 keypoints
                        pose_tensor = pose_tensor[:75]
                    
                    pose_data.append(pose_tensor)
                else:
                    # Use zero tensor if file is invalid
                    pose_data.append(torch.zeros(75, 2))
            except Exception as e:
                print(f"Error loading {json_path}: {e}")
                pose_data.append(torch.zeros(75, 2))
        else:
            # Use zero tensor if file doesn't exist
            pose_data.append(torch.zeros(75, 2))
    
    if not pose_data:
        return None
    
    # Stack all pose data
    try:
        pose_tensor = torch.stack(pose_data)  # Shape: (num_samples, 75, 2)
        
        # Reshape for model input: (1, 55, num_samples * 2)
        # Flatten to (num_samples, 150)
        pose_flat = pose_tensor.reshape(pose_tensor.shape[0], -1)
        
        # Take first 55 keypoints (110 values)
        pose_55 = pose_flat[:, :110]
        
        # Reshape to (num_samples, 55, 2)
        pose_reshaped = pose_55.reshape(pose_tensor.shape[0], 55, 2)
        
        # Transpose to (55, num_samples, 2)
        pose_transposed = pose_reshaped.permute(1, 0, 2)
        
        # Flatten to (55, num_samples * 2)
        pose_final = pose_transposed.reshape(55, -1)
        
        # Add batch dimension: (1, 55, num_samples * 2)
        keypoints_reshaped = pose_final.unsqueeze(0)
        
        return keypoints_reshaped
        
    except Exception as e:
        print(f"Error processing keypoints: {e}")
        return None

def run_inference_on_segment(model, keypoints, top_k=3):
    """Run inference on a single segment and return top-k predictions."""
    try:
        with torch.no_grad():
            output = model(keypoints)
            probabilities = torch.softmax(output, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
            
            predictions = []
            for i in range(top_k):
                predictions.append({
                    'index': int(top_indices[0][i]),
                    'probability': float(top_probs[0][i])
                })
            
            return predictions
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

def load_label_mapping():
    """Load label mapping from asl100.json."""
    try:
        with open('../../data/splits/asl100.json', 'r') as f:
            data = json.load(f)
        
        label_mapping = {}
        for idx, item in enumerate(data):
            label_mapping[idx] = item['gloss']
        
        return label_mapping
    except Exception as e:
        print(f"Error loading label mapping: {e}")
        return {}

def process_segments(session_dir, segments_file, model, configs, label_mapping, top_k=3):
    """Process all segments and return predictions."""
    try:
        with open(segments_file, 'r') as f:
            segments_data = json.load(f)
        
        results = {
            'session_dir': session_dir,
            'segments': [],
            'total_segments': len(segments_data['segments'])
        }
        
        for segment_info in segments_data['segments']:
            segment_id = segment_info['segment_id']
            start_frame = segment_info['start_frame']
            end_frame = segment_info['end_frame']
            
            print(f"Processing segment {segment_id}: frames {start_frame}-{end_frame}")
            
            # Load keypoints for this segment
            keypoints = load_keypoints_for_segment(
                session_dir, start_frame, end_frame, configs.num_samples
            )
            
            if keypoints is None:
                print(f"Failed to load keypoints for segment {segment_id}")
                continue
            
            # Run inference
            predictions = run_inference_on_segment(model, keypoints, top_k)
            
            if predictions is None:
                print(f"Failed to run inference for segment {segment_id}")
                continue
            
            # Map indices to labels
            segment_result = {
                'segment_id': segment_id,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'num_frames': end_frame - start_frame,
                'predictions': []
            }
            
            for pred in predictions:
                label_index = pred['index']
                if label_index in label_mapping:
                    segment_result['predictions'].append({
                        'label': label_mapping[label_index],
                        'probability': pred['probability']
                    })
                else:
                    segment_result['predictions'].append({
                        'label': f'unknown_{label_index}',
                        'probability': pred['probability']
                    })
            
            results['segments'].append(segment_result)
        
        return results
        
    except Exception as e:
        print(f"Error processing segments: {e}")
        return None

def save_results(results, output_file):
    """Save results to JSON file."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced multi-segment inference')
    parser.add_argument('--session_dir', required=True, help='Directory containing keypoint JSON files')
    parser.add_argument('--segments_file', required=True, help='JSON file with segment information')
    parser.add_argument('--config', default='configs/asl100.ini', help='Configuration file path')
    parser.add_argument('--checkpoint', default='archived/asl100/ckpt.pth', help='Model checkpoint path')
    parser.add_argument('--output_file', default=None, help='Output JSON file path')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top predictions per segment')
    
    args = parser.parse_args()
    
    if args.output_file is None:
        args.output_file = os.path.join(args.session_dir, 'enhanced_inference_results.json')
    
    print("Loading model...")
    model, configs = load_model(args.checkpoint, args.config)
    
    print("Loading label mapping...")
    label_mapping = load_label_mapping()
    
    print("Processing segments...")
    results = process_segments(
        args.session_dir, args.segments_file, model, configs, label_mapping, args.top_k
    )
    
    if results:
        save_results(results, args.output_file)
        
        # Print summary
        print(f"\n=== INFERENCE SUMMARY ===")
        print(f"Total segments processed: {len(results['segments'])}")
        
        for segment in results['segments']:
            print(f"\nSegment {segment['segment_id']} (frames {segment['start_frame']}-{segment['end_frame']}):")
            for i, pred in enumerate(segment['predictions']):
                print(f"  {i+1}. {pred['label']} ({pred['probability']:.3f})")
    else:
        print("Failed to process segments")

if __name__ == "__main__":
    main() 