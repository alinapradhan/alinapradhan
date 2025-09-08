# CropGuard AI - Crop Disease Detection System

An AI-powered crop disease detection system using computer vision technology to help farmers identify plant diseases and track disease outbreaks.

## 🌾 Features

- **AI-Powered Detection**: Convolutional Neural Network trained on PlantVillage dataset
- **High Accuracy**: Identifies 38 different plant diseases across 14 plant species
- **Mobile Optimized**: TensorFlow Lite optimization for smartphone deployment
- **Real-time Predictions**: Camera-based disease detection
- **Disease Mapping**: Geolocation tracking and hotspot visualization
- **Web Interface**: User-friendly web application
- **Early Intervention**: Helps farmers take preventive action

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Camera (optional, for real-time detection)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/alinapradhan/alinapradhan.git
   cd alinapradhan
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the PlantVillage dataset**
   ```bash
   python scripts/train_model.py --download-info
   ```
   Follow the instructions to download and extract the dataset to `data/raw/PlantVillage/`

4. **Train the model** (optional - for development)
   ```bash
   python scripts/train_model.py --epochs 50 --convert-tflite
   ```

5. **Test the system**
   ```bash
   python scripts/test_system.py
   ```

6. **Run the web application**
   ```bash
   python app.py
   ```

7. **Open your browser** and navigate to `http://localhost:5000`

## 📱 Usage

### Web Interface

1. **Upload Image Detection**
   - Go to the Upload page
   - Select a plant image
   - Enable location tracking (optional)
   - Click "Analyze Plant Disease"

2. **Camera Detection**
   - Go to the Camera page
   - Allow camera access
   - Position plant leaf in frame
   - Click "Capture & Analyze"

3. **Disease Map**
   - View disease hotspots on interactive map
   - Search for diseases in specific areas
   - Track disease outbreak patterns

### Command Line Interface

```bash
# Train model
python scripts/train_model.py --data-dir data/raw/PlantVillage --epochs 50

# Test inference
python scripts/test_system.py

# Run web app
python app.py
```

## 🧠 Model Architecture

- **Type**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow 2.13+
- **Input**: 224x224 RGB images
- **Classes**: 38 disease categories
- **Optimization**: TensorFlow Lite for mobile deployment
- **Accuracy**: High accuracy on PlantVillage dataset

### Supported Plants and Diseases

**Plants**: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

**Diseases**: Various including blights, rusts, spots, scabs, mildews, and healthy classifications

## 🗺️ Disease Mapping

The system includes advanced geolocation features:

- **GPS Tracking**: Records disease locations automatically
- **Hotspot Detection**: Identifies areas with multiple disease occurrences
- **Interactive Maps**: Visualize disease patterns on OpenStreetMap
- **Early Warning**: Helps predict and prevent disease spread

## 📁 Project Structure

```
alinapradhan/
├── src/                     # Source code
│   ├── model.py            # CNN model architecture
│   ├── train.py            # Training utilities
│   ├── inference.py        # Inference engine
│   └── geolocation.py      # Mapping and tracking
├── templates/              # HTML templates
├── static/                 # CSS, JS, and assets
├── scripts/                # Utility scripts
├── data/                   # Dataset storage
├── models/                 # Trained models
├── app.py                  # Flask web application
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 👩‍💻 About the Developer

Hi! I'm **Alina Pradhan** 👋 - an AI engineer enthusiast with curiosity to integrate algorithms for the goodness of mankind.

### 🏆 Achievements
- **VizQuest 3.0**: Secured All India Rank 5th in data visualization project by IIM Nagpur
- **Research Publication**: Presented paper on "ANNs in prediction and detection of cyberattacks in smartgrids"

### 📫 Contact
- Email: [alinapradhan15021707@gmail.com](mailto:alinapradhan15021707@gmail.com)
- Interests: Data Science, AI/ML, Volleyball, Swimming, Athletics, Deadlifting

### 🌱 Philosophy
*"Integrating algorithms for the goodness of mankind"* - Using AI to solve real-world problems and make a positive impact.

## 🔧 Configuration

### Environment Variables

- `FLASK_ENV`: Set to `development` for debug mode
- `MODEL_PATH`: Path to trained model file
- `DATABASE_PATH`: Path to SQLite database

### Model Training Parameters

- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Training batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)
- `--convert-tflite`: Convert to TensorFlow Lite format

## 📊 Performance

- **Inference Time**: < 1 second on modern hardware
- **Model Size**: ~50MB (full), ~15MB (TFLite)
- **Accuracy**: High accuracy on test dataset
- **Mobile Support**: Optimized for smartphone deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## ⚠️ Disclaimer

This AI system is designed to assist in crop disease identification but should not be the sole basis for agricultural decisions. Always consult with agricultural experts and plant pathologists for critical crop management decisions.

## 🙏 Acknowledgments

- **PlantVillage Dataset**: For providing comprehensive plant disease images
- **TensorFlow**: For the machine learning framework
- **OpenStreetMap**: For mapping services
- **Flask**: For the web framework
- **Bootstrap**: For responsive UI components

---

**CropGuard AI** - Protecting crops with artificial intelligence 🌱🤖