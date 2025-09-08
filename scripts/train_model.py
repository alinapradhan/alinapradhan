#!/usr/bin/env python3
"""
Training script for the crop disease detection model
Run this script to train the CNN model on the PlantVillage dataset
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from model import CropDiseaseModel
from train import DataProcessor, ModelTrainer, download_plantvillage_dataset

def main():
    parser = argparse.ArgumentParser(description='Train crop disease detection model')
    parser.add_argument('--data-dir', type=str, default='data/raw/PlantVillage',
                        help='Path to PlantVillage dataset directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate for training')
    parser.add_argument('--model-save-path', type=str, default='models/crop_disease_model.h5',
                        help='Path to save the trained model')
    parser.add_argument('--convert-tflite', action='store_true',
                        help='Convert model to TensorFlow Lite format')
    parser.add_argument('--download-info', action='store_true',
                        help='Show dataset download information')
    
    args = parser.parse_args()
    
    # Show download information if requested
    if args.download_info:
        download_plantvillage_dataset()
        return
    
    # Check if dataset exists
    if not os.path.exists(args.data_dir):
        print(f"Dataset directory not found: {args.data_dir}")
        print("Run with --download-info to get dataset download instructions")
        return 1
    
    # Create models directory
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    
    try:
        print("Initializing crop disease detection model...")
        
        # Initialize model
        model = CropDiseaseModel(num_classes=38, input_shape=(224, 224, 3))
        model.build_model()
        model.compile_model(learning_rate=args.learning_rate)
        
        print("Model architecture:")
        model.get_model_summary()
        
        # Initialize data processor
        print("Preparing datasets...")
        data_processor = DataProcessor(
            data_dir=args.data_dir,
            img_size=(224, 224),
            batch_size=args.batch_size
        )
        
        # Prepare datasets
        train_ds, val_ds, test_ds = data_processor.prepare_datasets()
        
        # Initialize trainer
        trainer = ModelTrainer(model, data_processor)
        
        # Train model
        print(f"Starting training for {args.epochs} epochs...")
        history = trainer.train(
            epochs=args.epochs,
            model_save_path=args.model_save_path
        )
        
        # Plot training history
        print("Generating training plots...")
        trainer.plot_training_history()
        
        # Evaluate model
        print("Evaluating model on test set...")
        test_accuracy, test_top5_accuracy, test_loss = trainer.evaluate_model()
        
        print(f"\nTraining completed!")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Top-5 Accuracy: {test_top5_accuracy:.4f}")
        print(f"Model saved to: {args.model_save_path}")
        
        # Convert to TensorFlow Lite if requested
        if args.convert_tflite:
            print("Converting model to TensorFlow Lite...")
            tflite_path = args.model_save_path.replace('.h5', '.tflite')
            model.convert_to_tflite(args.model_save_path, tflite_path)
            print(f"TFLite model saved to: {tflite_path}")
        
        return 0
        
    except Exception as e:
        print(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())