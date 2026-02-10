import torch
import json
import os
import argparse
import numpy as np
from tgcn_model import GCN_muti_att
from sign_dataset import read_pose_file
from configs import Config


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


def load_keypoints_from_json(keypoints_dir, video_id, frame_start, frame_end, num_samples=50):
    """Load and process keypoints from JSON files"""
    keypoints_list = []
    
    # Sample frames (similar to training)
    frame_indices = np.linspace(frame_start, frame_end, num_samples, dtype=int)
    
    for frame_idx in frame_indices:
        # Format frame number with leading zeros
        frame_name = f"image_{frame_idx:05d}_keypoints.json"
        keypoint_file = os.path.join(keypoints_dir, str(video_id), frame_name)
        
        if os.path.exists(keypoint_file):
            pose_data = read_pose_file(keypoint_file)
            if pose_data is not None:
                keypoints_list.append(pose_data)
            else:
                # If pose data is missing, use zeros
                keypoints_list.append(torch.zeros(75, 2))  # 75 keypoints, 2 coordinates
        else:
            # If file doesn't exist, use zeros
            keypoints_list.append(torch.zeros(75, 2))
    
    # Stack all keypoints into a tensor
    if keypoints_list:
        keypoints_tensor = torch.stack(keypoints_list, dim=0)  # Shape: (num_samples, 75, 2)
        
        # The model expects (batch_size, 55, input_feature) where input_feature = num_samples * 2
        # We need to reshape to (1, 55, num_samples * 2)
        # First, flatten the coordinates: (num_samples, 75, 2) -> (num_samples, 150)
        keypoints_flat = keypoints_tensor.reshape(num_samples, -1)  # (num_samples, 150)
        
        # Transpose to get (150, num_samples) and reshape to (55, num_samples * 2)
        # Since we have 75 keypoints * 2 coordinates = 150, we need to map to 55 nodes
        # We'll take the first 55 keypoints and their coordinates
        keypoints_reshaped = keypoints_flat[:55, :].transpose(0, 1)  # (150, 55) -> (55, 150)
        
        # Reshape to (55, num_samples * 2)
        keypoints_final = keypoints_reshaped.reshape(55, num_samples * 2)
        
        # Add batch dimension: (1, 55, num_samples * 2)
        keypoints_final = keypoints_final.unsqueeze(0)
        
        return keypoints_final
    else:
        return torch.zeros(1, 55, num_samples * 2)


def run_inference(model, keypoints, top_k=5):
    """Run inference on keypoints data"""
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


def main():
    parser = argparse.ArgumentParser(description='Run inference with TGCN model')
    parser.add_argument('--config', type=str, default='archived/asl100/asl100.ini',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='archived/asl100/ckpt.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--keypoints_dir', type=str, default='../../data/pose_per_individual_videos',
                       help='Directory containing pose keypoints')
    parser.add_argument('--video_id', type=str, required=True,
                       help='Video ID to run inference on')
    parser.add_argument('--frame_start', type=int, default=1,
                       help='Starting frame number')
    parser.add_argument('--frame_end', type=int, default=50,
                       help='Ending frame number')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    model, configs = load_model(args.checkpoint, args.config)
    print("Model loaded successfully!")
    
    print(f"Loading keypoints for video {args.video_id}...")
    keypoints = load_keypoints_from_json(
        args.keypoints_dir, 
        args.video_id, 
        args.frame_start, 
        args.frame_end, 
        configs.num_samples
    )
    print(f"Keypoints loaded! Shape: {keypoints.shape}")
    
    print("Running inference...")
    top_indices, top_probs = run_inference(model, keypoints, args.top_k)
    
    # Load label mapping
    label_mapping = load_label_mapping()
    
    print(f"\nTop {args.top_k} predictions:")
    print("-" * 40)
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
        gloss = label_mapping[idx.item()]
        print(f"{i+1}. {gloss} ({prob.item():.3f})")
    
    return top_indices, top_probs, label_mapping


if __name__ == '__main__':
    main() 