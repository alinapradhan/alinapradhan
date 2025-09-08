#!/usr/bin/env python3
"""
Test script for the crop disease detection inference engine
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from inference import CropDiseaseInference, CameraInference

def test_inference():
    """Test the inference engine with sample images"""
    
    # Check for model files
    model_paths = [
        'models/crop_disease_model.tflite',
        'models/crop_disease_model.h5'
    ]
    
    model_path = None
    model_type = None
    
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            model_type = 'tflite' if path.endswith('.tflite') else 'keras'
            break
    
    if not model_path:
        print("No trained model found. Please train a model first:")
        print("python scripts/train_model.py --download-info")
        return 1
    
    print(f"Testing inference with model: {model_path}")
    
    try:
        # Initialize inference engine
        inference_engine = CropDiseaseInference(model_path, model_type)
        print("✅ Inference engine initialized successfully")
        
        # Test with sample images if available
        sample_images_dir = "data/samples"
        if os.path.exists(sample_images_dir):
            import glob
            
            image_files = glob.glob(os.path.join(sample_images_dir, "*.jpg")) + \
                         glob.glob(os.path.join(sample_images_dir, "*.png"))
            
            if image_files:
                print(f"Testing with {len(image_files)} sample images...")
                
                for img_path in image_files[:3]:  # Test first 3 images
                    print(f"\nTesting: {os.path.basename(img_path)}")
                    result = inference_engine.predict(img_path)
                    
                    print(f"  Plant: {result['plant']}")
                    print(f"  Disease: {result['disease']}")
                    print(f"  Confidence: {result['confidence']:.2%}")
                    print(f"  Healthy: {result['is_healthy']}")
                    print(f"  Inference time: {result['inference_time']:.3f}s")
            else:
                print("No sample images found in data/samples/")
        else:
            print("Sample images directory not found. Create data/samples/ and add test images.")
        
        print("\n✅ Inference testing completed successfully")
        
        # Test camera interface (without actually starting camera)
        print("\nTesting camera interface initialization...")
        camera_interface = CameraInference(inference_engine, camera_id=0)
        print("✅ Camera interface initialized successfully")
        
        return 0
        
    except Exception as e:
        print(f"❌ Inference testing failed: {e}")
        return 1

def test_geolocation():
    """Test the geolocation module"""
    
    try:
        from geolocation import DiseaseLocationTracker, DiseaseMapGenerator
        
        print("Testing geolocation module...")
        
        # Initialize tracker
        tracker = DiseaseLocationTracker("data/test_disease_locations.db")
        print("✅ Disease location tracker initialized")
        
        # Test recording detection
        record_id = tracker.record_detection(
            latitude=40.7128,
            longitude=-74.0060,
            plant_type="Tomato",
            disease_name="Late_blight",
            confidence=0.85,
            user_id="test_user"
        )
        print(f"✅ Test detection recorded with ID: {record_id}")
        
        # Test nearby detections
        nearby = tracker.get_detections_in_radius(40.7128, -74.0060, 10)
        print(f"✅ Found {len(nearby)} nearby detections")
        
        # Test map generation
        map_generator = DiseaseMapGenerator(tracker)
        disease_map = map_generator.create_disease_map(
            save_path="static/test_disease_map.html"
        )
        print("✅ Disease map generated successfully")
        
        # Clean up test database
        os.remove("data/test_disease_locations.db")
        print("✅ Test database cleaned up")
        
        return 0
        
    except Exception as e:
        print(f"❌ Geolocation testing failed: {e}")
        return 1

def test_web_app():
    """Test web application components"""
    
    try:
        from app import app
        
        print("Testing web application...")
        
        # Test app configuration
        assert app.config['UPLOAD_FOLDER'] == 'static/uploads'
        assert app.config['MAX_CONTENT_LENGTH'] == 16 * 1024 * 1024
        print("✅ App configuration correct")
        
        # Test routes exist
        with app.test_client() as client:
            # Test main routes
            routes_to_test = ['/', '/upload', '/camera', '/map', '/about']
            
            for route in routes_to_test:
                response = client.get(route)
                assert response.status_code in [200, 302], f"Route {route} failed"
                print(f"✅ Route {route} accessible")
        
        print("✅ Web application testing completed")
        return 0
        
    except Exception as e:
        print(f"❌ Web application testing failed: {e}")
        return 1

def main():
    """Run all tests"""
    
    print("🧪 Running CropGuard AI System Tests\n")
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/samples', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/uploads', exist_ok=True)
    
    tests = [
        ("Inference Engine", test_inference),
        ("Geolocation Module", test_geolocation),
        ("Web Application", test_web_app)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Testing: {test_name}")
        print('='*50)
        
        result = test_func()
        results.append((test_name, result == 0))
        
        if result == 0:
            print(f"✅ {test_name} tests passed")
        else:
            print(f"❌ {test_name} tests failed")
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())