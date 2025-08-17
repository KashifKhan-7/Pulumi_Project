"""
Fatigue Detection Web Application

This is the main Flask application that provides a web interface and REST API
for the multi-modal fatigue detection system.
"""

import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import sqlite3
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fusion_system import FatigueDetectionSystem
from ai_models import ModelTrainer

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'fatigue_detection_secret_key_2024'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize fatigue detection system
fatigue_system = None
model_trainer = None

# Global state
system_thread = None
is_system_running = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_system():
    """Initialize the fatigue detection system"""
    global fatigue_system, model_trainer
    
    try:
        # Initialize the system
        fatigue_system = FatigueDetectionSystem()
        model_trainer = ModelTrainer()
        
        # Try to load existing models, if not available, train new ones
        model_trainer.load_all_models()
        
        # Check if any models are loaded
        models_loaded = any(model.is_trained for model in model_trainer.models.values())
        
        if not models_loaded:
            logger.info("No trained models found. Training new models...")
            model_trainer.train_all_models()
        
        logger.info("Fatigue detection system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        return False

def system_monitoring_loop():
    """Background thread for system monitoring and real-time updates"""
    global is_system_running
    
    while is_system_running and fatigue_system:
        try:
            # Get latest result
            latest_result = fatigue_system.get_latest_result()
            
            if latest_result:
                # Emit real-time update to connected clients
                socketio.emit('fatigue_update', {
                    'timestamp': latest_result.timestamp.isoformat(),
                    'overall_score': latest_result.overall_fatigue_score,
                    'fatigue_level': latest_result.fatigue_level,
                    'typing_score': latest_result.typing_score,
                    'facial_score': latest_result.facial_score,
                    'wearable_score': latest_result.wearable_score,
                    'confidence': latest_result.confidence,
                    'alert_level': latest_result.alert_level,
                    'recommendations': latest_result.recommendations[:3]  # Limit for UI
                })
            
            # Get system status
            status = fatigue_system.get_current_status()
            socketio.emit('system_status', {
                'is_active': status.is_active,
                'typing_active': status.typing_active,
                'facial_active': status.facial_active,
                'wearable_active': status.wearable_active,
                'session_duration': status.current_session_duration
            })
            
            time.sleep(2)  # Update every 2 seconds
            
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            time.sleep(5)

# Web Routes
@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/settings')
def settings():
    """Settings page"""
    return render_template('settings.html')

@app.route('/history')
def history():
    """Historical data page"""
    return render_template('history.html')

@app.route('/models')
def models():
    """ML models management page"""
    return render_template('models.html')

# API Routes
@app.route('/api/status')
def api_status():
    """Get current system status"""
    try:
        if fatigue_system:
            status = fatigue_system.get_current_status()
            return jsonify({
                'success': True,
                'data': {
                    'is_active': status.is_active,
                    'typing_active': status.typing_active,
                    'facial_active': status.facial_active,
                    'wearable_active': status.wearable_active,
                    'last_update': status.last_update.isoformat(),
                    'total_sessions': status.total_sessions,
                    'current_session_duration': status.current_session_duration,
                    'alerts_count': status.alerts_count
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'System not initialized'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/start', methods=['POST'])
def api_start_monitoring():
    """Start fatigue monitoring"""
    global system_thread, is_system_running
    
    try:
        if not fatigue_system:
            return jsonify({
                'success': False,
                'error': 'System not initialized'
            }), 500
        
        if fatigue_system.is_running:
            return jsonify({
                'success': False,
                'error': 'System is already running'
            }), 400
        
        # Start the fatigue detection system
        fatigue_system.start_monitoring()
        
        # Start background monitoring thread
        is_system_running = True
        system_thread = threading.Thread(target=system_monitoring_loop)
        system_thread.daemon = True
        system_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Fatigue monitoring started'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stop', methods=['POST'])
def api_stop_monitoring():
    """Stop fatigue monitoring"""
    global is_system_running
    
    try:
        if not fatigue_system:
            return jsonify({
                'success': False,
                'error': 'System not initialized'
            }), 500
        
        if not fatigue_system.is_running:
            return jsonify({
                'success': False,
                'error': 'System is not running'
            }), 400
        
        # Stop the system
        is_system_running = False
        fatigue_system.stop_monitoring()
        
        return jsonify({
            'success': True,
            'message': 'Fatigue monitoring stopped'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/current')
def api_current_data():
    """Get current fatigue data"""
    try:
        if not fatigue_system:
            return jsonify({
                'success': False,
                'error': 'System not initialized'
            }), 500
        
        latest_result = fatigue_system.get_latest_result()
        
        if not latest_result:
            return jsonify({
                'success': True,
                'data': None,
                'message': 'No data available yet'
            })
        
        return jsonify({
            'success': True,
            'data': latest_result.to_dict()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/recent')
def api_recent_data():
    """Get recent fatigue data"""
    try:
        minutes = request.args.get('minutes', 30, type=int)
        
        if not fatigue_system:
            return jsonify({
                'success': False,
                'error': 'System not initialized'
            }), 500
        
        recent_results = fatigue_system.get_recent_results(minutes)
        
        data = [result.to_dict() for result in recent_results]
        
        return jsonify({
            'success': True,
            'data': data,
            'count': len(data)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/session/summary')
def api_session_summary():
    """Get current session summary"""
    try:
        if not fatigue_system:
            return jsonify({
                'success': False,
                'error': 'System not initialized'
            }), 500
        
        summary = fatigue_system.get_session_summary()
        
        return jsonify({
            'success': True,
            'data': summary
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/history')
def api_history():
    """Get historical data from database"""
    try:
        # Get query parameters
        days = request.args.get('days', 7, type=int)
        limit = request.args.get('limit', 1000, type=int)
        
        # Connect to database
        conn = sqlite3.connect('data/fatigue_data.db')
        cursor = conn.cursor()
        
        # Query measurements
        cursor.execute('''
            SELECT timestamp, overall_score, fatigue_level, typing_score, 
                   facial_score, wearable_score, confidence, alert_level
            FROM fatigue_measurements 
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC 
            LIMIT ?
        '''.format(days), (limit,))
        
        measurements = []
        for row in cursor.fetchall():
            measurements.append({
                'timestamp': row[0],
                'overall_score': row[1],
                'fatigue_level': row[2],
                'typing_score': row[3],
                'facial_score': row[4],
                'wearable_score': row[5],
                'confidence': row[6],
                'alert_level': row[7]
            })
        
        # Query sessions
        cursor.execute('''
            SELECT session_start, session_end, duration, avg_fatigue_score, 
                   max_fatigue_score, alerts_count
            FROM fatigue_sessions 
            WHERE session_start >= datetime('now', '-{} days')
            ORDER BY session_start DESC
        '''.format(days))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'session_start': row[0],
                'session_end': row[1],
                'duration': row[2],
                'avg_fatigue_score': row[3],
                'max_fatigue_score': row[4],
                'alerts_count': row[5]
            })
        
        conn.close()
        
        return jsonify({
            'success': True,
            'data': {
                'measurements': measurements,
                'sessions': sessions
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/status')
def api_models_status():
    """Get ML models status"""
    try:
        if not model_trainer:
            return jsonify({
                'success': False,
                'error': 'Model trainer not initialized'
            }), 500
        
        models_status = {}
        for name, model in model_trainer.models.items():
            models_status[name] = {
                'is_trained': model.is_trained,
                'model_name': model.model_name,
                'feature_count': len(model.feature_names) if model.feature_names else 0,
                'performance': model.performance_metrics.__dict__ if model.performance_metrics else None
            }
        
        return jsonify({
            'success': True,
            'data': models_status
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/train', methods=['POST'])
def api_train_models():
    """Train ML models"""
    try:
        if not model_trainer:
            return jsonify({
                'success': False,
                'error': 'Model trainer not initialized'
            }), 500
        
        # Train models in background
        def train_models():
            try:
                results = model_trainer.train_all_models()
                logger.info("Model training completed")
                
                # Emit training completion event
                socketio.emit('model_training_complete', {
                    'success': True,
                    'results': {name: result.__dict__ if result else None 
                              for name, result in results.items()}
                })
                
            except Exception as e:
                logger.error(f"Model training failed: {e}")
                socketio.emit('model_training_complete', {
                    'success': False,
                    'error': str(e)
                })
        
        training_thread = threading.Thread(target=train_models)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Model training started in background'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Make fatigue prediction with provided features"""
    try:
        if not model_trainer:
            return jsonify({
                'success': False,
                'error': 'Model trainer not initialized'
            }), 500
        
        # Get features from request
        features = request.get_json()
        
        if not features:
            return jsonify({
                'success': False,
                'error': 'No features provided'
            }), 400
        
        # Make predictions
        predictions = model_trainer.predict_with_all_models(features)
        
        # Convert to serializable format
        result = {}
        for model_name, prediction in predictions.items():
            result[model_name] = {
                'fatigue_probability': prediction.fatigue_probability,
                'fatigue_level': prediction.fatigue_level,
                'confidence': prediction.confidence,
                'timestamp': prediction.timestamp.isoformat()
            }
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Socket.IO Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected')
    
    # Send current status to newly connected client
    if fatigue_system:
        latest_result = fatigue_system.get_latest_result()
        if latest_result:
            emit('fatigue_update', latest_result.to_dict())
        
        status = fatigue_system.get_current_status()
        emit('system_status', {
            'is_active': status.is_active,
            'typing_active': status.typing_active,
            'facial_active': status.facial_active,
            'wearable_active': status.wearable_active,
            'session_duration': status.current_session_duration
        })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected')

@socketio.on('request_update')
def handle_request_update():
    """Handle client request for immediate update"""
    if fatigue_system:
        latest_result = fatigue_system.get_latest_result()
        if latest_result:
            emit('fatigue_update', latest_result.to_dict())

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# Initialize system on startup
@app.before_first_request
def startup():
    """Initialize system on first request"""
    logger.info("Initializing fatigue detection system...")
    success = initialize_system()
    
    if not success:
        logger.error("Failed to initialize system")
    else:
        logger.info("System initialized successfully")

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Initialize system
    logger.info("Starting fatigue detection web application...")
    success = initialize_system()
    
    if success:
        logger.info("System initialized successfully")
        
        # Run the Flask app with SocketIO
        socketio.run(app, 
                    host='0.0.0.0', 
                    port=5000, 
                    debug=False,
                    allow_unsafe_werkzeug=True)
    else:
        logger.error("Failed to initialize system. Exiting.")
        sys.exit(1)