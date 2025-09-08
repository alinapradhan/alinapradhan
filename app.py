"""
Flask web application for crop disease detection system
Provides web interface for image upload, real-time detection, and disease mapping
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
import json
from datetime import datetime

# Import our modules
from src.inference import CropDiseaseInference
from src.geolocation import DiseaseLocationTracker, DiseaseMapGenerator, get_user_location

app = Flask(__name__)
app.secret_key = 'crop_disease_detection_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Global variables for models and trackers
inference_engine = None
location_tracker = None
map_generator = None

def init_app():
    """Initialize the application with models and services"""
    global inference_engine, location_tracker, map_generator
    
    try:
        # Initialize inference engine (try TFLite first, then Keras)
        model_paths = [
            'models/crop_disease_model.tflite',
            'models/crop_disease_model.h5'
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                model_type = 'tflite' if model_path.endswith('.tflite') else 'keras'
                inference_engine = CropDiseaseInference(model_path, model_type)
                print(f"Loaded model: {model_path}")
                break
        
        if inference_engine is None:
            print("Warning: No trained model found. Please train a model first.")
        
        # Initialize location tracking
        location_tracker = DiseaseLocationTracker()
        map_generator = DiseaseMapGenerator(location_tracker)
        
        print("Application initialized successfully")
        
    except Exception as e:
        print(f"Error initializing application: {e}")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and disease detection"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get user location for tracking
            latitude = request.form.get('latitude', type=float)
            longitude = request.form.get('longitude', type=float)
            
            # Perform disease detection
            if inference_engine:
                result = inference_engine.predict(filepath)
                
                # Record detection with location if available
                if latitude and longitude and not result['is_healthy']:
                    location_tracker.record_detection(
                        latitude, longitude,
                        result['plant'], result['disease'],
                        result['confidence'],
                        user_id=request.remote_addr,
                        image_path=filepath
                    )
                
                return render_template('result.html', 
                                     result=result, 
                                     image_path=filepath,
                                     has_location=bool(latitude and longitude))
            else:
                flash('Model not available. Please train a model first.')
                return redirect(url_for('index'))
    
    return render_template('upload.html')

@app.route('/camera')
def camera_detection():
    """Camera-based detection page"""
    return render_template('camera.html')

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Handle base64 image prediction from camera"""
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        if inference_engine:
            result = inference_engine.predict(image)
            
            # Record detection with location if available
            if latitude and longitude and not result['is_healthy']:
                location_tracker.record_detection(
                    latitude, longitude,
                    result['plant'], result['disease'],
                    result['confidence'],
                    user_id=request.remote_addr
                )
            
            return jsonify({
                'success': True,
                'result': result
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Model not available'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/map')
def disease_map():
    """Disease hotspot map page"""
    try:
        # Generate updated map
        if map_generator:
            disease_map = map_generator.create_disease_map(
                save_path="static/disease_map.html"
            )
            
            # Get hotspot statistics
            hotspots = location_tracker.get_disease_hotspots()
            
            return render_template('map.html', 
                                 hotspots=hotspots,
                                 map_available=disease_map is not None)
        else:
            return render_template('map.html', 
                                 hotspots=[],
                                 map_available=False)
    except Exception as e:
        flash(f'Error generating map: {e}')
        return render_template('map.html', hotspots=[], map_available=False)

@app.route('/api/nearby_detections')
def nearby_detections():
    """API endpoint for getting nearby disease detections"""
    try:
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)
        radius = request.args.get('radius', default=10, type=float)
        
        if lat is None or lng is None:
            return jsonify({'error': 'Latitude and longitude required'})
        
        detections = location_tracker.get_detections_in_radius(lat, lng, radius)
        
        return jsonify({
            'detections': detections,
            'count': len(detections)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/disease_stats')
def disease_stats():
    """API endpoint for disease statistics"""
    try:
        import sqlite3
        
        conn = sqlite3.connect(location_tracker.db_path)
        
        # Get disease counts
        disease_query = """
            SELECT disease_name, COUNT(*) as count 
            FROM disease_detections 
            WHERE disease_name != 'healthy'
            GROUP BY disease_name 
            ORDER BY count DESC
        """
        
        # Get plant counts
        plant_query = """
            SELECT plant_type, COUNT(*) as count 
            FROM disease_detections 
            GROUP BY plant_type 
            ORDER BY count DESC
        """
        
        # Get recent detections
        recent_query = """
            SELECT * FROM disease_detections 
            ORDER BY timestamp DESC 
            LIMIT 10
        """
        
        disease_counts = conn.execute(disease_query).fetchall()
        plant_counts = conn.execute(plant_query).fetchall()
        recent_detections = conn.execute(recent_query).fetchall()
        
        conn.close()
        
        return jsonify({
            'disease_counts': [{'name': d[0], 'count': d[1]} for d in disease_counts],
            'plant_counts': [{'name': p[0], 'count': p[1]} for p in plant_counts],
            'recent_detections': len(recent_detections),
            'total_detections': sum(d[1] for d in disease_counts) + 
                              len([d for d in recent_detections if 'healthy' in str(d)])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large. Maximum size is 16MB.')
    return redirect(url_for('upload_file'))

if __name__ == '__main__':
    init_app()
    app.run(debug=True, host='0.0.0.0', port=5000)