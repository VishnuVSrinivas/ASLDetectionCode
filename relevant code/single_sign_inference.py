#!/usr/bin/env python3

import torch
import json
import os
import numpy as np
from tgcn_model import GCN_muti_att
from configs import Config
import argparse
from sign_dataset import read_pose_file

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

def load_keypoints_from_session(session_dir, num_samples=25):
    """Load all keypoints from a session directory."""
    pose_data = []
    
    # Get all keypoint files
    keypoint_files = [f for f in os.listdir(session_dir) if f.endswith('_keypoints.json')]
    keypoint_files.sort()  # Sort by frame number
    
    if not keypoint_files:
        return None
    
    # Sample frames if needed
    if len(keypoint_files) > num_samples:
        step = len(keypoint_files) / num_samples
        indices = [int(i * step) for i in range(num_samples)]
        keypoint_files = [keypoint_files[i] for i in indices]
    elif len(keypoint_files) < num_samples:
        # Pad by repeating the last frame
        last_file = keypoint_files[-1] if keypoint_files else None
        keypoint_files.extend([last_file] * (num_samples - len(keypoint_files)))
    
    for json_file in keypoint_files:
        json_path = os.path.join(session_dir, json_file)
        
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

def run_inference(model, keypoints, top_k=5):
    """Run inference on keypoints and return top-k predictions."""
    with torch.no_grad():
        outputs = model(keypoints)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        
        predictions = []
        for i in range(top_k):
            predictions.append({
                'label': int(top_indices[0][i].item()),
                'probability': float(top_probs[0][i].item())
            })
        
        return predictions

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

def process_single_sign(session_dir, model, configs, label_mapping, top_k=5):
    """Process a single sign from session directory."""
    # Load keypoints
    keypoints = load_keypoints_from_session(session_dir, configs.num_samples)
    if keypoints is None:
        return None
    
    # Run inference
    predictions = run_inference(model, keypoints, top_k)
    
    # Map labels to words
    for pred in predictions:
        pred['label'] = label_mapping.get(pred['label'], f"unknown_{pred['label']}")
    
    return {
        'session_dir': session_dir,
        'predictions': predictions
    }

def save_results(results, output_file):
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Single Sign Inference')
    parser.add_argument('--session_dir', required=True, help='Session directory with keypoints')
    parser.add_argument('--output_file', required=True, help='Output JSON file')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions')
    parser.add_argument('--config', default='configs/asl100.ini', help='Config file path')
    parser.add_argument('--checkpoint', default='archived/asl100/ckpt.pth', help='Model checkpoint path')
    
    args = parser.parse_args()
    
    # Load model
    model, configs = load_model(args.checkpoint, args.config)
    
    # Load label mapping
    label_mapping = load_label_mapping()
    
    # Process sign
    results = process_single_sign(args.session_dir, model, configs, label_mapping, args.top_k)
    
    if results:
        # Add segments structure for compatibility
        results['segments'] = [{
            'segment_id': 1,
            'start_frame': 0,
            'end_frame': 0,
            'num_frames': 0,
            'predictions': results['predictions']
        }]
        
        save_results(results, args.output_file)
        print(f"Results saved to {args.output_file}")
    else:
        print("Failed to process sign")

if __name__ == "__main__":
    main() 