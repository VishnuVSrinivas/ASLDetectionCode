#!/usr/bin/env python3

import os
import json
import threading
from flask import Flask, render_template, request, jsonify, session
from live_capture import capture_and_process_sign
from ollama_processor import process_inference_results
import uuid

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Global state for the current session
current_session = {
    'signs': [],
    'predictions_history': [],
    'session_id': None
}

@app.route('/')
def index():
    """Main page with webcam interface."""
    return render_template('live_index.html')

@app.route('/capture_sign', methods=['POST'])
def capture_sign():
    """Capture a sign and return predictions."""
    try:
        # Capture and process sign
        result = capture_and_process_sign(duration=1.5)
        
        if result['success']:
            # Add to session history
            current_session['predictions_history'].append(result)
            
            return jsonify({
                'success': True,
                'predictions': result['predictions'],
                'frame_count': result['frame_count']
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/confirm_sign', methods=['POST'])
def confirm_sign():
    """Confirm a sign selection."""
    data = request.get_json()
    sign_index = data.get('sign_index', 0)
    prediction_index = data.get('prediction_index', 0)
    
    if prediction_index < len(current_session['predictions_history']):
        predictions = current_session['predictions_history'][prediction_index]['predictions']
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

@app.route('/undo_last_sign', methods=['POST'])
def undo_last_sign():
    """Remove the last confirmed sign."""
    if current_session['signs']:
        removed_sign = current_session['signs'].pop()
        return jsonify({
            'success': True,
            'removed_sign': removed_sign,
            'all_signs': current_session['signs']
        })
    
    return jsonify({
        'success': False,
        'error': 'No signs to undo'
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
        # Create a mock inference result for LLM processing
        mock_inference = {
            'segments': [{
                'predictions': [{'label': sign} for sign in current_session['signs']]
            }]
        }
        
        # Process with LLM
        try:
            llm_result = process_inference_results(mock_inference)
            return jsonify({
                'success': True,
                'sentence': llm_result.get('sentence', ' '.join(current_session['signs'])),
                'emoji': llm_result.get('emoji', '😐'),
                'sentiment': llm_result.get('sentiment', 'neutral'),
                'signs': current_session['signs'],
                'llm_status': 'processed',
                'message': 'Processed with Ollama LLM'
            })
        except Exception as e:
            # Fallback processing
            signs_text = ' '.join(current_session['signs'])
            return jsonify({
                'success': True,
                'sentence': signs_text.capitalize(),
                'emoji': '😐',
                'sentiment': 'neutral',
                'signs': current_session['signs'],
                'llm_status': 'fallback',
                'message': 'Using fallback processing (Ollama not available)'
            })
        
    except Exception as e:
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
    # Initialize session
    current_session['session_id'] = str(uuid.uuid4())[:8]
    app.run(debug=True, host='0.0.0.0', port=8081) 