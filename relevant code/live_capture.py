import cv2
import time
import os
import json
import subprocess
import tempfile
from datetime import datetime
import uuid
import mediapipe as mp

def capture_sign(duration=1.5, output_path=None, show_preview=True):
    """Capture video from webcam for specified duration with MediaPipe tracking."""
    if output_path is None:
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        session_id = str(uuid.uuid4())[:8]
        output_path = os.path.join(temp_dir, f"captured_sign_{session_id}.mp4")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam")
    
    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20.0
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hands = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2
    )
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Recording for {duration} seconds...")
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)
        hands_results = hands.process(rgb_frame)
        
        # Draw landmarks
        annotated_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Add recording indicator
        cv2.putText(annotated_frame, "RECORDING", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.circle(annotated_frame, (20, 20), 10, (0, 0, 255), -1)
        
        # Save original frame (without annotations) for processing
        out.write(frame)
        
        if show_preview:
            cv2.imshow('Recording ASL Sign...', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
    
    cap.release()
    out.release()
    
    if show_preview:
        cv2.destroyAllWindows()
    
    # Clean up MediaPipe
    pose.close()
    hands.close()
    
    print(f"Captured {frame_count} frames")
    return output_path, frame_count

def process_captured_sign(video_path):
    """Process captured video through the full pipeline."""
    try:
        # Create session directory
        session_id = str(uuid.uuid4())[:8]
        session_dir = os.path.join('../../data/pose_per_individual_videos', f'live_session_{session_id}')
        os.makedirs(session_dir, exist_ok=True)
        
        # Step 1: Extract keypoints from video
        from video_to_keypoints import extract_keypoints_from_video
        frame_count = extract_keypoints_from_video(video_path, session_dir)
        
        if frame_count == 0:
            raise Exception("No frames were processed from the video")
        
        # Step 2: Run inference directly (no segmentation needed for single sign)
        inference_file = os.path.join(session_dir, 'inference_results.json')
        inference_cmd = [
            'python', 'single_sign_inference.py',
            '--session_dir', session_dir,
            '--output_file', inference_file,
            '--top_k', '5'
        ]
        
        result = subprocess.run(inference_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Inference failed: {result.stderr}")
        
        # Load results
        with open(inference_file, 'r') as f:
            results = json.load(f)
        
        # Clean up temporary video
        os.remove(video_path)
        
        return results
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(video_path):
            os.remove(video_path)
        raise e

def capture_and_process_sign(duration=1.5):
    """Main function to capture and process a sign."""
    video_path = None
    try:
        # Capture video
        video_path, frame_count = capture_sign(duration)
        
        # Process through pipeline
        results = process_captured_sign(video_path)
        
        return {
            'success': True,
            'frame_count': frame_count,
            'predictions': results['segments'][0]['predictions'] if results['segments'] else [],
            'session_id': results.get('session_dir', '').split('_')[-1]
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'frame_count': 0,
            'predictions': []
        }

if __name__ == "__main__":
    # Test the capture function
    result = capture_and_process_sign(1.5)
    print(json.dumps(result, indent=2)) 