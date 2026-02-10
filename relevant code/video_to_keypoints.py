#!/usr/bin/env python3

import cv2
import mediapipe as mp
import json
import os
import numpy as np
import torch

def extract_keypoints_from_video(video_path, output_dir):
    """
    Extract pose keypoints from video using MediaPipe.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save keypoint JSON files
        
    Returns:
        Number of frames processed
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    left_hand = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    right_hand = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose
        pose_results = pose.process(rgb_frame)
        
        # Process hands
        left_hand_results = left_hand.process(rgb_frame)
        right_hand_results = right_hand.process(rgb_frame)
        
        # Extract keypoints
        keypoints_data = {
            "people": [{
                "pose_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": []
            }]
        }
        
        # Extract pose keypoints (33 points)
        if pose_results.pose_landmarks:
            pose_landmarks = pose_results.pose_landmarks.landmark
            for landmark in pose_landmarks:
                # Convert to normalized coordinates (0-256 range as expected by the model)
                x = landmark.x * 256.0
                y = landmark.y * 256.0
                confidence = landmark.visibility
                keypoints_data["people"][0]["pose_keypoints_2d"].extend([x, y, confidence])
        else:
            # Fill with zeros if no pose detected
            keypoints_data["people"][0]["pose_keypoints_2d"] = [0.0, 0.0, 0.0] * 33
        
        # Extract left hand keypoints (21 points)
        if left_hand_results.multi_hand_landmarks:
            hand_landmarks = left_hand_results.multi_hand_landmarks[0].landmark
            for landmark in hand_landmarks:
                x = landmark.x * 256.0
                y = landmark.y * 256.0
                confidence = 1.0  # MediaPipe doesn't provide confidence for hands
                keypoints_data["people"][0]["hand_left_keypoints_2d"].extend([x, y, confidence])
        else:
            keypoints_data["people"][0]["hand_left_keypoints_2d"] = [0.0, 0.0, 0.0] * 21
        
        # Extract right hand keypoints (21 points)
        if right_hand_results.multi_hand_landmarks:
            hand_landmarks = right_hand_results.multi_hand_landmarks[0].landmark
            for landmark in hand_landmarks:
                x = landmark.x * 256.0
                y = landmark.y * 256.0
                confidence = 1.0
                keypoints_data["people"][0]["hand_right_keypoints_2d"].extend([x, y, confidence])
        else:
            keypoints_data["people"][0]["hand_right_keypoints_2d"] = [0.0, 0.0, 0.0] * 21
        
        # Save keypoints to JSON file
        output_file = os.path.join(output_dir, f"image_{frame_count:06d}_keypoints.json")
        with open(output_file, 'w') as f:
            json.dump(keypoints_data, f, indent=2)
        
        frame_count += 1
        
        # Print progress every 10 frames
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    pose.close()
    left_hand.close()
    right_hand.close()
    
    print(f"Extracted keypoints from {frame_count} frames")
    return frame_count

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract keypoints from video')
    parser.add_argument('--video_path', required=True, help='Path to input video')
    parser.add_argument('--output_dir', required=True, help='Output directory for keypoints')
    
    args = parser.parse_args()
    
    extract_keypoints_from_video(args.video_path, args.output_dir) 