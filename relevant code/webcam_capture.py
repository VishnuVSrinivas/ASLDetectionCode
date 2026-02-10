#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import queue
from typing import Optional, Tuple

class WebcamCapture:
    def __init__(self, camera_id=0):
        """Initialize webcam capture with MediaPipe pose tracking."""
        self.camera_id = camera_id
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.recording_frames = []
        self.is_recording = False
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose and hands
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
    
    def start_camera(self):
        """Start the webcam capture."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame with MediaPipe
            processed_frame = self._process_frame(frame)
            
            # Add to queue for display
            if not self.frame_queue.full():
                self.frame_queue.put(processed_frame)
            
            # Record frame if recording
            if self.is_recording:
                self.recording_frames.append(frame.copy())
    
    def _process_frame(self, frame):
        """Process frame with MediaPipe pose and hands tracking."""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose
        pose_results = self.pose.process(rgb_frame)
        
        # Process hands
        hands_results = self.hands.process(rgb_frame)
        
        # Convert back to BGR for OpenCV
        annotated_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # Draw pose landmarks
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw hand landmarks
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Add recording indicator
        if self.is_recording:
            cv2.putText(annotated_frame, "RECORDING", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.circle(annotated_frame, (20, 20), 10, (0, 0, 255), -1)
        
        return annotated_frame
    
    def start_recording(self, duration=1.5):
        """Start recording for specified duration."""
        self.recording_frames = []
        self.is_recording = True
        time.sleep(duration)
        self.is_recording = False
        return len(self.recording_frames)
    
    def get_latest_frame(self):
        """Get the latest processed frame."""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop_camera(self):
        """Stop the webcam capture."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        if self.pose:
            self.pose.close()
        if self.hands:
            self.hands.close()
    
    def get_recorded_frames(self):
        """Get the recorded frames."""
        return self.recording_frames.copy()

def show_webcam_interface():
    """Show live webcam interface with MediaPipe tracking."""
    capture = WebcamCapture()
    
    try:
        capture.start_camera()
        
        print("Webcam started! Press 'r' to record a sign (1.5s), 'q' to quit")
        
        while True:
            frame = capture.get_latest_frame()
            if frame is not None:
                cv2.imshow('ASL Recognition - Press R to Record', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Recording sign...")
                frame_count = capture.start_recording(1.5)
                print(f"Recorded {frame_count} frames")
                
                # Here you would process the recorded frames
                recorded_frames = capture.get_recorded_frames()
                print(f"Got {len(recorded_frames)} frames for processing")
                
                # TODO: Process frames through the ASL recognition pipeline
                
    except KeyboardInterrupt:
        print("Stopping webcam...")
    finally:
        capture.stop_camera()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    show_webcam_interface() 