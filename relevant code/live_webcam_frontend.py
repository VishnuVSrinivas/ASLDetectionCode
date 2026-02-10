#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import base64
from flask import Flask, render_template, Response, request, jsonify
import json
import uuid
import os
import tempfile
from video_to_keypoints import extract_keypoints_from_video
from single_sign_inference import process_single_sign, load_model, load_label_mapping

app = Flask(__name__)

# Global state
current_session = {
    'signs': [],
    'predictions_history': [],
    'session_id': None
}

# MediaPipe setup
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

# Webcam setup
cap = None
is_recording = False
recording_frames = []
model = None
configs = None
label_mapping = None

def init_model():
    """Initialize the TGCN model."""
    global model, configs, label_mapping
    try:
        from configs import Config
        configs = Config('configs/asl100.ini', mode='test')
        model, configs = load_model('archived/asl100/ckpt.pth', 'configs/asl100.ini')
        label_mapping = load_label_mapping()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

def process_frame(frame):
    """Process frame with MediaPipe tracking."""
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process pose and hands
    pose_results = pose.process(rgb_frame)
    hands_results = hands.process(rgb_frame)
    
    # Convert back to BGR for OpenCV
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
    if is_recording:
        cv2.putText(annotated_frame, "RECORDING", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.circle(annotated_frame, (20, 20), 10, (0, 0, 255), -1)
        recording_frames.append(frame.copy())
    
    return annotated_frame

def generate_frames():
    """Generate video frames for streaming."""
    global cap, is_recording, recording_frames
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame with MediaPipe
        processed_frame = process_frame(frame)
        
        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Main page with live webcam."""
    return render_template('live_webcam.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """Start recording frames."""
    global is_recording, recording_frames
    is_recording = True
    recording_frames = []
    return jsonify({'success': True, 'message': 'Recording started'})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """Stop recording and process the sign."""
    global is_recording, recording_frames, model, configs, label_mapping
    
    is_recording = False
    frames = recording_frames.copy()
    recording_frames = []
    
    if len(frames) == 0:
        return jsonify({'success': False, 'error': 'No frames recorded'})
    
    try:
        # Save frames to temporary video
        temp_dir = tempfile.gettempdir()
        session_id = str(uuid.uuid4())[:8]
        video_path = os.path.join(temp_dir, f"captured_sign_{session_id}.mp4")
        
        # Create video from frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        # Create session directory
        session_dir = os.path.join('../../data/pose_per_individual_videos', f'live_session_{session_id}')
        os.makedirs(session_dir, exist_ok=True)
        
        # Extract keypoints
        frame_count = extract_keypoints_from_video(video_path, session_dir)
        
        if frame_count == 0:
            return jsonify({'success': False, 'error': 'No keypoints extracted'})
        
        # Run inference
        if model is None:
            init_model()
        
        results = process_single_sign(session_dir, model, configs, label_mapping, 5)
        
        if results:
            predictions = results['predictions']
            current_session['predictions_history'].append({
                'predictions': predictions,
                'frame_count': frame_count
            })
            
            return jsonify({
                'success': True,
                'predictions': predictions,
                'frame_count': frame_count
            })
        else:
            return jsonify({'success': False, 'error': 'Inference failed'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/confirm_sign', methods=['POST'])
def confirm_sign():
    """Confirm a sign selection."""
    data = request.get_json()
    sign_index = data.get('sign_index', 0)
    
    # Always use the most recent prediction
    if current_session['predictions_history']:
        predictions = current_session['predictions_history'][-1]['predictions']
        if sign_index < len(predictions):
            selected_sign = predictions[sign_index]['label']
            current_session['signs'].append(selected_sign)
            
            return jsonify({
                'success': True,
                'selected_sign': selected_sign,
                'all_signs': current_session['signs']
            })
    
    return jsonify({
        'success': False,
        'error': 'Invalid sign selection'
    }), 400

@app.route('/process_sentence', methods=['POST'])
def process_sentence():
    """Process the complete sentence with LLM."""
    if not current_session['signs']:
        return jsonify({
            'success': False,
            'error': 'No signs to process'
        }), 400
    
    try:
        print(f"Processing signs: {current_session['signs']}")
        from ollama_processor import OllamaProcessor
        processor = OllamaProcessor()
        result = processor.process_signs_to_sentence(current_session['signs'])
        print(f"LLM result: {result}")
        
        response_data = {
            'success': True,
            'sentence': result.get('sentence', ' '.join(current_session['signs'])),
            'emoji': result.get('emoji', '😐'),
            'sentiment': result.get('sentiment', 'neutral'),
            'signs': current_session['signs']
        }
        print(f"Sending response: {response_data}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in process_sentence: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/clear_session', methods=['POST'])
def clear_session():
    """Clear the current session."""
    global current_session
    current_session = {
        'signs': [],
        'predictions_history': [],
        'session_id': str(uuid.uuid4())[:8]
    }
    
    return jsonify({
        'success': True,
        'message': 'Session cleared'
    })

@app.route('/get_current_state', methods=['GET'])
def get_current_state():
    """Get the current session state."""
    return jsonify({
        'signs': current_session['signs'],
        'predictions_history': current_session['predictions_history']
    })

if __name__ == '__main__':
    # Initialize model
    init_model()
    
    # Initialize session
    current_session['session_id'] = str(uuid.uuid4())[:8]
    
    app.run(debug=True, host='0.0.0.0', port=8082) 