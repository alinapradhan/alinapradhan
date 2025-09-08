"""
Real-time inference engine for crop disease detection
Supports both TensorFlow and TensorFlow Lite models
"""

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time
from model import PLANT_DISEASE_CLASSES

class CropDiseaseInference:
    def __init__(self, model_path, model_type='tflite'):
        """
        Initialize inference engine
        
        Args:
            model_path (str): Path to the model file (.h5 or .tflite)
            model_type (str): Type of model ('tflite' or 'keras')
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_shape = (224, 224, 3)
        
        self.load_model()
        
    def load_model(self):
        """Load the model based on type"""
        if self.model_type == 'tflite':
            self.load_tflite_model()
        elif self.model_type == 'keras':
            self.load_keras_model()
        else:
            raise ValueError("Model type must be 'tflite' or 'keras'")
            
    def load_tflite_model(self):
        """Load TensorFlow Lite model"""
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape
        self.input_shape = self.input_details[0]['shape'][1:4]
        print(f"TFLite model loaded. Input shape: {self.input_shape}")
        
    def load_keras_model(self):
        """Load Keras model"""
        self.model = tf.keras.models.load_model(self.model_path)
        print(f"Keras model loaded. Input shape: {self.model.input_shape[1:4]}")
        
    def preprocess_image(self, image):
        """
        Preprocess image for inference
        
        Args:
            image: Input image (PIL Image, numpy array, or file path)
            
        Returns:
            Preprocessed image ready for inference
        """
        # Handle different input types
        if isinstance(image, str):
            # File path
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
            
        # Resize image
        image = image.resize((self.input_shape[1], self.input_shape[0]))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1]
        image_array = image_array / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def predict(self, image):
        """
        Make prediction on a single image
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Make prediction
        if self.model_type == 'tflite':
            predictions = self.predict_tflite(processed_image)
        else:
            predictions = self.predict_keras(processed_image)
            
        # Process results
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        predicted_class = PLANT_DISEASE_CLASSES[predicted_class_idx]
        
        # Parse plant and disease
        plant_name, disease_name = self.parse_class_name(predicted_class)
        
        inference_time = time.time() - start_time
        
        # Get top 5 predictions
        top5_indices = np.argsort(predictions[0])[-5:][::-1]
        top5_predictions = []
        
        for idx in top5_indices:
            class_name = PLANT_DISEASE_CLASSES[idx]
            plant, disease = self.parse_class_name(class_name)
            top5_predictions.append({
                'plant': plant,
                'disease': disease,
                'confidence': float(predictions[0][idx]),
                'class_name': class_name
            })
        
        result = {
            'plant': plant_name,
            'disease': disease_name,
            'confidence': confidence,
            'predicted_class': predicted_class,
            'inference_time': inference_time,
            'top5_predictions': top5_predictions,
            'is_healthy': 'healthy' in disease_name.lower()
        }
        
        return result
    
    def predict_tflite(self, image):
        """Make prediction using TensorFlow Lite model"""
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
        return predictions
    
    def predict_keras(self, image):
        """Make prediction using Keras model"""
        predictions = self.model.predict(image, verbose=0)
        return predictions
    
    def parse_class_name(self, class_name):
        """
        Parse class name to extract plant and disease information
        
        Args:
            class_name (str): Class name in format 'Plant___Disease'
            
        Returns:
            tuple: (plant_name, disease_name)
        """
        parts = class_name.split('___')
        plant_name = parts[0].replace('_', ' ')
        disease_name = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
        
        return plant_name, disease_name

class CameraInference:
    def __init__(self, inference_engine, camera_id=0):
        """
        Initialize camera-based inference
        
        Args:
            inference_engine (CropDiseaseInference): Inference engine
            camera_id (int): Camera device ID
        """
        self.inference_engine = inference_engine
        self.camera_id = camera_id
        self.cap = None
        
    def start_camera(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
            
        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera started successfully")
        
    def stop_camera(self):
        """Stop camera capture"""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
            
    def run_realtime_detection(self, confidence_threshold=0.5):
        """
        Run real-time disease detection from camera
        
        Args:
            confidence_threshold (float): Minimum confidence for predictions
        """
        if not self.cap:
            self.start_camera()
            
        print("Starting real-time crop disease detection...")
        print("Press 'c' to capture and analyze, 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read from camera")
                break
                
            # Display frame
            cv2.imshow('Crop Disease Detection - Press C to Capture', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                # Capture and analyze
                print("Analyzing image...")
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Make prediction
                result = self.inference_engine.predict(rgb_frame)
                
                # Display results
                self.display_prediction_results(result, confidence_threshold)
                
            elif key == ord('q'):
                break
                
        self.stop_camera()
        
    def display_prediction_results(self, result, confidence_threshold):
        """Display prediction results"""
        print("\n" + "="*50)
        print("CROP DISEASE DETECTION RESULTS")
        print("="*50)
        print(f"Plant: {result['plant']}")
        print(f"Disease: {result['disease']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Health Status: {'Healthy' if result['is_healthy'] else 'Disease Detected'}")
        print(f"Inference Time: {result['inference_time']:.3f} seconds")
        
        if result['confidence'] >= confidence_threshold:
            if result['is_healthy']:
                print("✅ Plant appears healthy!")
            else:
                print("⚠️  Disease detected! Consider taking preventive action.")
        else:
            print("❓ Low confidence prediction. Try capturing a clearer image.")
            
        print("\nTop 5 Predictions:")
        for i, pred in enumerate(result['top5_predictions'], 1):
            print(f"{i}. {pred['plant']} - {pred['disease']} ({pred['confidence']:.2%})")
        print("="*50)

def main():
    """Main function for testing inference"""
    print("Crop Disease Detection Inference Engine")
    print("Please ensure you have a trained model available")
    
    # Example usage
    model_path = "models/crop_disease_model.tflite"  # or .h5 for Keras
    
    try:
        # Initialize inference engine
        inference_engine = CropDiseaseInference(model_path, model_type='tflite')
        
        # Initialize camera inference
        camera_inference = CameraInference(inference_engine)
        
        # Run real-time detection
        camera_inference.run_realtime_detection()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. A trained model file")
        print("2. A working camera")
        print("3. All required dependencies installed")

if __name__ == "__main__":
    main()