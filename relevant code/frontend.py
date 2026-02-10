#!/usr/bin/env python3

import os
import json
import subprocess
import tempfile
import shutil
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import threading
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for processing status
processing_status = {}
session_results = {}

def allowed_file(filename):
    """Check if file extension is allowed."""
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_pipeline(video_path, session_id):
    """Process video through the complete pipeline."""
    try:
        processing_status[session_id] = {
            'status': 'processing',
            'step': 'Starting processing...',
            'progress': 0
        }
        
        # Step 1: Create session directory
        session_dir = os.path.join('../../data/pose_per_individual_videos', f'session_{session_id}')
        os.makedirs(session_dir, exist_ok=True)
        
        processing_status[session_id]['step'] = 'Extracting keypoints from video...'
        processing_status[session_id]['progress'] = 20
        
        # Step 2: Extract keypoints from the uploaded video
        from video_to_keypoints import extract_keypoints_from_video
        
        print(f"Extracting keypoints from {video_path} to {session_dir}")
        frame_count = extract_keypoints_from_video(video_path, session_dir)
        
        if frame_count == 0:
            raise Exception("No frames were processed from the video. Please check the video file.")
        
        print(f"Successfully extracted keypoints from {frame_count} frames")
        
        processing_status[session_id]['step'] = 'Segmenting video...'
        processing_status[session_id]['progress'] = 40
        
        # Step 3: Run segmentation
        segments_file = os.path.join(session_dir, 'segments.json')
        segment_cmd = [
            'python', 'segment.py',
            '--session_dir', session_dir,
            '--output_file', segments_file,
            '--motion_threshold', '0.8',  # Lowered threshold to detect more motion
            '--min_frames', '15',
            '--max_frames', '80'
        ]
        
        result = subprocess.run(segment_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Segmentation failed: {result.stderr}")
        
        processing_status[session_id]['step'] = 'Running inference...'
        processing_status[session_id]['progress'] = 60
        
        # Step 4: Run enhanced inference
        inference_file = os.path.join(session_dir, 'enhanced_inference_results.json')
        inference_cmd = [
            'python', 'enhanced_multi_inference.py',
            '--session_dir', session_dir,
            '--segments_file', segments_file,
            '--output_file', inference_file,
            '--top_k', '3'
        ]
        
        result = subprocess.run(inference_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Inference failed: {result.stderr}")
        
        processing_status[session_id]['step'] = 'Processing with LLM...'
        processing_status[session_id]['progress'] = 80
        
        # Step 5: Process with LLM
        final_file = os.path.join(session_dir, 'final_results.json')
        try:
            from ollama_processor import process_inference_results
            final_results = process_inference_results(inference_file, final_file)
        except Exception as e:
            print(f"LLM processing failed (using fallback): {e}")
            # Continue with fallback processing
        
        # Load final results
        if os.path.exists(final_file):
            with open(final_file, 'r') as f:
                final_results = json.load(f)
        else:
            # Fallback: load inference results directly
            with open(inference_file, 'r') as f:
                inference_results = json.load(f)
            
            # Extract signs
            signs = []
            for segment in inference_results['segments']:
                if segment['predictions']:
                    signs.append(segment['predictions'][0]['label'])
            
            final_results = {
                'signs': signs,
                'sentence': ' '.join(signs).capitalize(),
                'emoji': '😐',
                'sentiment': 'neutral',
                'segments': inference_results['segments']
            }
        
        session_results[session_id] = final_results
        processing_status[session_id] = {
            'status': 'completed',
            'step': 'Processing complete!',
            'progress': 100
        }
        
    except Exception as e:
        processing_status[session_id] = {
            'status': 'error',
            'step': f'Error: {str(e)}',
            'progress': 0
        }

@app.route('/')
def index():
    """Main upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle video upload and start processing."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Generate session ID
    session_id = str(int(time.time()))
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_{filename}')
    file.save(video_path)
    
    # Start processing in background thread
    thread = threading.Thread(target=process_video_pipeline, args=(video_path, session_id))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'session_id': session_id,
        'message': 'Video uploaded successfully. Processing started.'
    })

@app.route('/status/<session_id>')
def get_status(session_id):
    """Get processing status for a session."""
    if session_id not in processing_status:
        return jsonify({'error': 'Session not found'}), 404
    
    return jsonify(processing_status[session_id])

@app.route('/results/<session_id>')
def get_results(session_id):
    """Get results for a completed session."""
    if session_id not in session_results:
        return jsonify({'error': 'Results not found'}), 404
    
    return jsonify(session_results[session_id])

@app.route('/update_prediction', methods=['POST'])
def update_prediction():
    """Update a prediction for a segment."""
    data = request.json
    session_id = data.get('session_id')
    segment_id = data.get('segment_id')
    new_prediction = data.get('prediction')
    
    if session_id not in session_results:
        return jsonify({'error': 'Session not found'}), 404
    
    # Update the prediction
    results = session_results[session_id]
    for segment in results['segments']:
        if segment['segment_id'] == segment_id:
            # Update top prediction
            if segment['predictions']:
                segment['predictions'][0]['label'] = new_prediction
            break
    
    # Rebuild signs list
    signs = []
    for segment in results['segments']:
        if segment['predictions']:
            signs.append(segment['predictions'][0]['label'])
    
    results['signs'] = signs
    results['sentence'] = ' '.join(signs).capitalize()
    
    # Save updated results
    session_dir = os.path.join('../../data/pose_per_individual_videos', f'session_{session_id}')
    final_file = os.path.join(session_dir, 'final_results.json')
    with open(final_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return jsonify({'message': 'Prediction updated successfully'})

@app.route('/reprocess_sentence', methods=['POST'])
def reprocess_sentence():
    """Reprocess sentence with LLM."""
    data = request.json
    session_id = data.get('session_id')
    custom_sentence = data.get('sentence')
    
    if session_id not in session_results:
        return jsonify({'error': 'Session not found'}), 404
    
    try:
        # Import LLM processor
        from llm_processor import LLMProcessor
        
        llm_processor = LLMProcessor()
        result = llm_processor.reprocess_sentence(custom_sentence)
        
        # Update session results
        session_results[session_id]['sentence'] = result['sentence']
        session_results[session_id]['emoji'] = result['emoji']
        session_results[session_id]['sentiment'] = result['sentiment']
        
        # Save updated results
        session_dir = os.path.join('../../data/pose_per_individual_videos', f'session_{session_id}')
        final_file = os.path.join(session_dir, 'final_results.json')
        with open(final_file, 'w') as f:
            json.dump(session_results[session_id], f, indent=2)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Reprocessing failed: {str(e)}'}), 500

@app.route('/export/<session_id>')
def export_results(session_id):
    """Export results as JSON file."""
    if session_id not in session_results:
        return jsonify({'error': 'Results not found'}), 404
    
    results = session_results[session_id]
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(results, temp_file, indent=2)
    temp_file.close()
    
    return send_file(
        temp_file.name,
        as_attachment=True,
        download_name=f'asl_results_{session_id}.json',
        mimetype='application/json'
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 