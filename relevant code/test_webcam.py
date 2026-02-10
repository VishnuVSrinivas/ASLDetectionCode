#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np

def show_webcam_with_tracking():
    """Show live webcam feed with MediaPipe pose and hand tracking."""
    
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
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam started! You should see:")
    print("- Pose tracking (body landmarks)")
    print("- Hand tracking (hand landmarks)")
    print("- Press 'r' to record a sign")
    print("- Press 'q' to quit")
    
    recording = False
    recording_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)
        hands_results = hands.process(rgb_frame)
        
        # Draw landmarks
        annotated_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # Draw pose landmarks
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw hand landmarks
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
        if recording:
            cv2.putText(annotated_frame, "RECORDING", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.circle(annotated_frame, (20, 20), 10, (0, 0, 255), -1)
            recording_frames.append(frame.copy())
        
        # Show frame
        cv2.imshow('ASL Recognition - MediaPipe Tracking', annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            if not recording:
                print("Starting recording...")
                recording = True
                recording_frames = []
            else:
                print("Stopping recording...")
                recording = False
                print(f"Recorded {len(recording_frames)} frames")
                # Here you would process the frames through the ASL recognition pipeline
    
    # Cleanup
    cap.release()
    pose.close()
    hands.close()
    cv2.destroyAllWindows()
    print("Webcam closed")

if __name__ == "__main__":
    show_webcam_with_tracking() 