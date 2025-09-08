"""
Simplified Flask web application for crop disease detection system demo
This version provides the web interface without requiring heavy ML dependencies
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
import random

app = Flask(__name__)
app.secret_key = 'crop_disease_detection_demo_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Demo data for plant diseases (PlantVillage dataset classes)
PLANT_DISEASE_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def parse_class_name(class_name):
    """Parse class name to extract plant and disease information"""
    parts = class_name.split('___')
    plant_name = parts[0].replace('_', ' ')
    disease_name = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
    return plant_name, disease_name

def simulate_ai_prediction():
    """Simulate AI model prediction for demo purposes"""
    # Randomly select a disease class
    predicted_class = random.choice(PLANT_DISEASE_CLASSES)
    plant, disease = parse_class_name(predicted_class)
    
    # Generate realistic confidence score
    confidence = random.uniform(0.75, 0.98) if 'healthy' not in disease.lower() else random.uniform(0.85, 0.99)
    
    # Generate top 5 predictions
    top5_classes = random.sample(PLANT_DISEASE_CLASSES, 5)
    if predicted_class not in top5_classes:
        top5_classes[0] = predicted_class
    
    top5_predictions = []
    for i, cls in enumerate(top5_classes):
        plant_pred, disease_pred = parse_class_name(cls)
        conf = confidence if i == 0 else random.uniform(0.05, confidence - 0.1)
        top5_predictions.append({
            'plant': plant_pred,
            'disease': disease_pred,
            'confidence': conf,
            'class_name': cls
        })
    
    # Sort by confidence
    top5_predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    result = {
        'plant': plant,
        'disease': disease,
        'confidence': confidence,
        'predicted_class': predicted_class,
        'inference_time': random.uniform(0.1, 0.5),
        'top5_predictions': top5_predictions,
        'is_healthy': 'healthy' in disease.lower()
    }
    
    return result

# Simple in-memory storage for demo
demo_detections = []

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
            
            # Simulate disease detection
            result = simulate_ai_prediction()
            
            # Record detection with location if available
            if latitude and longitude and not result['is_healthy']:
                demo_detections.append({
                    'timestamp': datetime.now().isoformat(),
                    'latitude': latitude,
                    'longitude': longitude,
                    'plant_type': result['plant'],
                    'disease_name': result['disease'],
                    'confidence': result['confidence'],
                    'user_id': request.remote_addr,
                    'image_path': filepath
                })
            
            return render_template('result.html', 
                                 result=result, 
                                 image_path=filepath,
                                 has_location=bool(latitude and longitude))
    
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
        
        # Decode base64 image (for demo, we just simulate processing)
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Simulate disease detection
        result = simulate_ai_prediction()
        
        # Record detection with location if available
        if latitude and longitude and not result['is_healthy']:
            demo_detections.append({
                'timestamp': datetime.now().isoformat(),
                'latitude': latitude,
                'longitude': longitude,
                'plant_type': result['plant'],
                'disease_name': result['disease'],
                'confidence': result['confidence'],
                'user_id': request.remote_addr
            })
        
        return jsonify({
            'success': True,
            'result': result
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
        # For demo, create some sample hotspots
        sample_hotspots = [
            {
                'location_name': 'Sample Farm Area 1',
                'total_detections': len([d for d in demo_detections if 'blight' in d.get('disease_name', '').lower()]),
                'most_common_disease': 'Late_blight',
                'most_common_plant': 'Tomato'
            }
        ] if demo_detections else []
        
        return render_template('map.html', 
                             hotspots=sample_hotspots,
                             map_available=len(demo_detections) > 0)
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
        
        # For demo, return sample data
        detections = demo_detections[:5]  # Return first 5 detections
        
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
        disease_counts = {}
        plant_counts = {}
        
        for detection in demo_detections:
            disease = detection.get('disease_name', 'Unknown')
            plant = detection.get('plant_type', 'Unknown')
            
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
            plant_counts[plant] = plant_counts.get(plant, 0) + 1
        
        return jsonify({
            'disease_counts': [{'name': d, 'count': c} for d, c in disease_counts.items()],
            'plant_counts': [{'name': p, 'count': c} for p, c in plant_counts.items()],
            'recent_detections': len(demo_detections),
            'total_detections': len(demo_detections)
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
    print("🌾 CropGuard AI Demo - Starting web application...")
    print("📝 Note: This is a demo version with simulated AI predictions")
    print("🌐 Open http://localhost:5000 in your browser")
    print("⚠️  For full AI functionality, install complete dependencies and train the model")
    app.run(debug=True, host='0.0.0.0', port=5000)