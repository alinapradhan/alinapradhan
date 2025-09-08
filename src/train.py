"""
Data preprocessing and training utilities for crop disease detection
"""

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from model import CropDiseaseModel, PLANT_DISEASE_CLASSES

class DataProcessor:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        """
        Initialize data processor
        
        Args:
            data_dir (str): Path to PlantVillage dataset directory
            img_size (tuple): Target image size
            batch_size (int): Batch size for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        
    def prepare_datasets(self, validation_split=0.2, test_split=0.1):
        """Prepare train, validation, and test datasets"""
        
        # Create training dataset
        self.train_ds = keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=validation_split + test_split,
            subset="training",
            seed=123,
            image_size=self.img_size,
            batch_size=self.batch_size
        )
        
        # Create validation and test datasets
        val_test_ds = keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=validation_split + test_split,
            subset="validation",
            seed=123,
            image_size=self.img_size,
            batch_size=self.batch_size
        )
        
        # Split validation and test
        val_batches = tf.data.experimental.cardinality(val_test_ds)
        test_batches = int(val_batches * test_split / (validation_split + test_split))
        
        self.test_ds = val_test_ds.take(test_batches)
        self.val_ds = val_test_ds.skip(test_batches)
        
        # Optimize datasets for performance
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.test_ds = self.test_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        print(f"Training batches: {tf.data.experimental.cardinality(self.train_ds)}")
        print(f"Validation batches: {tf.data.experimental.cardinality(self.val_ds)}")
        print(f"Test batches: {tf.data.experimental.cardinality(self.test_ds)}")
        
        return self.train_ds, self.val_ds, self.test_ds

class ModelTrainer:
    def __init__(self, model, data_processor):
        """
        Initialize model trainer
        
        Args:
            model (CropDiseaseModel): The CNN model to train
            data_processor (DataProcessor): Data processor with prepared datasets
        """
        self.model = model
        self.data_processor = data_processor
        self.history = None
        
    def setup_callbacks(self, model_save_path, patience=10):
        """Setup training callbacks"""
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, epochs=50, model_save_path='models/crop_disease_model.h5'):
        """Train the model"""
        
        if self.data_processor.train_ds is None:
            raise ValueError("Datasets must be prepared before training")
            
        # Setup callbacks
        callbacks = self.setup_callbacks(model_save_path)
        
        # Train the model
        self.history = self.model.model.fit(
            self.data_processor.train_ds,
            validation_data=self.data_processor.val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def plot_training_history(self, save_path='static/training_history.png'):
        """Plot training history"""
        
        if self.history is None:
            raise ValueError("Model must be trained first")
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def evaluate_model(self, save_confusion_matrix=True):
        """Evaluate model on test dataset"""
        
        if self.data_processor.test_ds is None:
            raise ValueError("Test dataset must be prepared first")
            
        # Evaluate on test set
        test_loss, test_accuracy, test_top5_accuracy = self.model.model.evaluate(
            self.data_processor.test_ds, verbose=1
        )
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Top-5 Accuracy: {test_top5_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Generate predictions for confusion matrix
        if save_confusion_matrix:
            y_pred = []
            y_true = []
            
            for images, labels in self.data_processor.test_ds:
                predictions = self.model.model.predict(images, verbose=0)
                y_pred.extend(np.argmax(predictions, axis=1))
                y_true.extend(np.argmax(labels, axis=1))
            
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(15, 12))
            sns.heatmap(cm, annot=False, cmap='Blues', 
                       xticklabels=[cls.split('___')[1] for cls in PLANT_DISEASE_CLASSES],
                       yticklabels=[cls.split('___')[1] for cls in PLANT_DISEASE_CLASSES])
            plt.title('Confusion Matrix - Crop Disease Detection')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('static/confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, 
                                      target_names=[cls.split('___')[1] for cls in PLANT_DISEASE_CLASSES]))
        
        return test_accuracy, test_top5_accuracy, test_loss

def download_plantvillage_dataset():
    """
    Instructions for downloading PlantVillage dataset
    Note: This function provides instructions since automatic download 
    requires agreement to dataset terms
    """
    instructions = """
    To use this crop disease detection system, please download the PlantVillage dataset:
    
    1. Visit: https://www.kaggle.com/datasets/arjuntejaswi/plant-village
    2. Download the dataset (requires Kaggle account)
    3. Extract to ./data/raw/PlantVillage/
    4. The dataset should have the following structure:
       data/raw/PlantVillage/
       ├── Apple___Apple_scab/
       ├── Apple___Black_rot/
       ├── Apple___Cedar_apple_rust/
       └── ... (38 total classes)
    
    Alternative: You can also use tensorflow_datasets:
    pip install tensorflow-datasets
    
    Then use:
    import tensorflow_datasets as tfds
    ds = tfds.load('plant_village', split='train', as_supervised=True)
    """
    
    print(instructions)
    return instructions