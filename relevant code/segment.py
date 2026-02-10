##!/usr/bin/env python3

import numpy as np
import json
import os

def load_keypoints_from_session(session_dir):
    """Load all keypoint JSON files from a session directory."""
    keypoints_list = []
    if not os.path.exists(session_dir):
        print(f"Session directory {session_dir} does not exist!")
        return keypoints_list
    
    json_files = [f for f in os.listdir(session_dir) if f.endswith('_keypoints.json')]
    json_files.sort()
    
    for json_file in json_files:
        file_path = os.path.join(session_dir, json_file)
        try:
            with open(file_path, 'r') as f:
                keypoint_data = json.load(f)
                keypoints_list.append(keypoint_data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return keypoints_list

def extract_keypoints_with_confidence(keypoint_data):
    """Extract keypoints with confidence scores."""
    try:
        if 'people' in keypoint_data and len(keypoint_data['people']) > 0:
            person = keypoint_data['people'][0]
            if 'pose_keypoints_2d' in person:
                keypoints = person['pose_keypoints_2d']
                # Extract x, y, confidence for each keypoint
                keypoint_list = []
                for i in range(0, len(keypoints), 3):
                    if i + 2 < len(keypoints):
                        x, y, conf = keypoints[i], keypoints[i+1], keypoints[i+2]
                        keypoint_list.append((x, y, conf))
                return keypoint_list
        return []
    except Exception as e:
        print(f"Error extracting keypoints: {e}")
        return []

def compute_motion_energy(keypoints1, keypoints2, confidence_threshold=0.3):
    """Compute motion energy between two consecutive frames."""
    if not keypoints1 or not keypoints2:
        return 0.0
    
    total_distance = 0.0
    valid_points = 0
    
    # Use the minimum number of keypoints
    num_keypoints = min(len(keypoints1), len(keypoints2))
    
    for i in range(num_keypoints):
        x1, y1, conf1 = keypoints1[i]
        x2, y2, conf2 = keypoints2[i]
        
        # Only use keypoints with confidence > threshold
        if conf1 > confidence_threshold and conf2 > confidence_threshold:
            # Skip if either point is at origin (likely invalid)
            if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
                continue
            
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            total_distance += distance
            valid_points += 1
    
    return total_distance / max(valid_points, 1)

def compute_motion_energy_sequence(keypoints_list, confidence_threshold=0.3):
    """Compute motion energy for all consecutive frame pairs."""
    motion_energy = []
    
    for i in range(len(keypoints_list) - 1):
        kp1 = extract_keypoints_with_confidence(keypoints_list[i])
        kp2 = extract_keypoints_with_confidence(keypoints_list[i + 1])
        energy = compute_motion_energy(kp1, kp2, confidence_threshold)
        motion_energy.append(energy)
    
    return motion_energy

def smooth_motion_energy(motion_energy, window_size=5):
    """Apply moving average filter to smooth motion energy."""
    if len(motion_energy) < window_size:
        return motion_energy
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(motion_energy)):
        start = max(0, i - half_window)
        end = min(len(motion_energy), i + half_window + 1)
        window = motion_energy[start:end]
        smoothed.append(np.mean(window))
    
    return smoothed

def find_active_segments(motion_energy, threshold=5.0, min_frames=20, max_frames=60, padding=5):
    """Find active segments based on motion energy threshold."""
    if not motion_energy:
        return []
    
    # Mark frames as active or idle
    active_frames = [energy > threshold for energy in motion_energy]
    
    segments = []
    start_frame = None
    
    for i, is_active in enumerate(active_frames):
        if is_active:
            if start_frame is None:
                start_frame = i
        else:
            if start_frame is not None:
                end_frame = i
                segment_length = end_frame - start_frame
                
                # Only keep segments within the desired length range
                if min_frames <= segment_length <= max_frames:
                    # Add padding
                    padded_start = max(0, start_frame - padding)
                    padded_end = min(len(motion_energy), end_frame + padding)
                    segments.append((padded_start, padded_end))
                
                start_frame = None
    
    # Handle case where video ends with active frames
    if start_frame is not None:
        end_frame = len(motion_energy)
        segment_length = end_frame - start_frame
        if min_frames <= segment_length <= max_frames:
            padded_start = max(0, start_frame - padding)
            padded_end = min(len(motion_energy), end_frame + padding)
            segments.append((padded_start, padded_end))
    
    return segments

def segment_video(session_dir, confidence_threshold=0.3, motion_threshold=5.0, 
                 min_frames=20, max_frames=60, padding=5, smoothing_window=5):
    """
    Segment video based on pose activity and motion energy.
    """
    print(f"Loading keypoints from {session_dir}...")
    keypoints_list = load_keypoints_from_session(session_dir)
    
    if not keypoints_list:
        print("No keypoint files found!")
        return None
    
    print(f"Computing motion energy for {len(keypoints_list)} frames...")
    motion_energy = compute_motion_energy_sequence(keypoints_list, confidence_threshold)
    
    print("Smoothing motion energy...")
    smoothed_energy = smooth_motion_energy(motion_energy, smoothing_window)
    
    print(f"Finding active segments with threshold {motion_threshold}...")
    segments = find_active_segments(smoothed_energy, motion_threshold, min_frames, max_frames, padding)
    
    # Calculate average motion energy for each segment
    segment_info = []
    for i, (start, end) in enumerate(segments):
        if start < len(smoothed_energy):
            segment_energy = smoothed_energy[start:end]
            avg_energy = sum(segment_energy) / len(segment_energy) if segment_energy else 0
            segment_info.append({
                'segment_id': i + 1,
                'start_frame': start,
                'end_frame': end,
                'num_frames': end - start,
                'avg_motion_energy': avg_energy
            })
    
    results = {
        'session_dir': session_dir,
        'total_frames': len(keypoints_list),
        'segments': segment_info,
        'motion_energy': motion_energy,
        'smoothed_energy': smoothed_energy,
        'parameters': {
            'confidence_threshold': confidence_threshold,
            'motion_threshold': motion_threshold,
            'min_frames': min_frames,
            'max_frames': max_frames,
            'padding': padding,
            'smoothing_window': smoothing_window
        }
    }
    
    print(f"Found {len(segments)} active segments:")
    for segment in segment_info:
        print(f"  Segment {segment['segment_id']}: frames {segment['start_frame']}-{segment['end_frame']} "
              f"({segment['num_frames']} frames, avg energy: {segment['avg_motion_energy']:.3f})")
    
    return results

def save_segmentation_results(results, output_path):
    """Save segmentation results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Segmentation results saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Segment video based on pose activity')
    parser.add_argument('--session_dir', required=True, help='Directory containing keypoint JSON files')
    parser.add_argument('--output_file', default=None, help='Output JSON file path')
    parser.add_argument('--confidence_threshold', type=float, default=0.3, help='Minimum confidence for keypoints')
    parser.add_argument('--motion_threshold', type=float, default=5.0, help='Motion energy threshold for active frames')
    parser.add_argument('--min_frames', type=int, default=20, help='Minimum frames for a segment')
    parser.add_argument('--max_frames', type=int, default=60, help='Maximum frames for a segment')
    parser.add_argument('--padding', type=int, default=5, help='Frames to add before/after each segment')
    parser.add_argument('--smoothing_window', type=int, default=5, help='Window size for motion smoothing')
    
    args = parser.parse_args()
    
    if args.output_file is None:
        args.output_file = os.path.join(args.session_dir, 'segments.json')
    
    results = segment_video(
        args.session_dir,
        confidence_threshold=args.confidence_threshold,
        motion_threshold=args.motion_threshold,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        padding=args.padding,
        smoothing_window=args.smoothing_window
    )
    
    if results:
        save_segmentation_results(results, args.output_file)
