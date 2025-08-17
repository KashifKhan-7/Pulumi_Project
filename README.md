# Multi-Modal Fatigue Detection System

A comprehensive fatigue detection system that analyzes typing patterns, facial expressions, and wearable data using AI models to provide real-time fatigue monitoring and personalized recommendations.

## 🚀 Features

### Multi-Modal Analysis
- **Typing Pattern Analysis**: Keystroke dynamics, timing patterns, error rates, and typing rhythm
- **Facial Expression Analysis**: Eye aspect ratio, blink detection, yawn detection, head pose estimation
- **Wearable Data Integration**: Heart rate variability, activity levels, sleep quality, stress indicators

### AI-Powered Detection
- **Machine Learning Models**: Random Forest, Gradient Boosting, and Neural Networks
- **Multi-Modal Fusion**: Advanced fusion strategies combining all data sources
- **Real-Time Prediction**: Continuous monitoring with immediate feedback

### Web Interface
- **Real-Time Dashboard**: Live monitoring with interactive charts and alerts
- **Historical Analysis**: Session tracking and trend analysis
- **REST API**: Full API access for integration with other systems
- **WebSocket Support**: Real-time updates and notifications

### Intelligent Recommendations
- **Personalized Alerts**: Context-aware fatigue alerts
- **Actionable Recommendations**: Specific suggestions based on fatigue source
- **Break Scheduling**: Smart break recommendations

## 📋 System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Camera**: Webcam for facial analysis (optional)
- **Wearable Devices**: Heart rate monitors, fitness trackers (optional)

### Software Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows 10+, macOS 10.15+, or Linux
- **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/fatigue-detection-system.git
cd fatigue-detection-system
```

### 2. Create Virtual Environment
```bash
python -m venv fatigue_env
source fatigue_env/bin/activate  # On Windows: fatigue_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Required Models (Optional)
For facial analysis, download the dlib shape predictor:
```bash
# Create models directory
mkdir -p models

# Download shape predictor (68 facial landmarks)
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat models/
```

### 5. Initialize the System
```bash
python main.py train  # Train ML models
python main.py test   # Run system tests
```

## 🚀 Usage

### Web Interface (Recommended)
Start the web server:
```bash
python main.py web
```

Then open your browser and navigate to: `http://localhost:5000`

### Command Line Interface
For terminal-based monitoring:
```bash
python main.py cli --duration 300  # Monitor for 5 minutes
```

### Training Models
To retrain the AI models:
```bash
python main.py train
```

### Running Tests
To verify system functionality:
```bash
python main.py test
```

## 🎛️ Configuration

Edit `config.yaml` to customize system behavior:

```yaml
# System Settings
system:
  debug: false
  log_level: "INFO"
  data_collection_interval: 1.0
  fatigue_threshold: 0.7

# Enable/disable modalities
typing:
  enabled: true
  window_size: 30

facial:
  enabled: true
  camera_index: 0
  fps: 10

wearable:
  enabled: true

# Fusion strategy
fusion:
  method: "weighted_ensemble"  # Options: weighted_ensemble, confidence_weighted, adaptive
  weights:
    typing: 0.3
    facial: 0.4
    wearable: 0.3

# Alerts
alerts:
  enabled: true
  sound_alerts: true
  break_suggestions: true
```

## 📊 API Documentation

### REST API Endpoints

#### System Control
- `GET /api/status` - Get system status
- `POST /api/start` - Start monitoring
- `POST /api/stop` - Stop monitoring

#### Data Access
- `GET /api/current` - Get current fatigue data
- `GET /api/recent?minutes=30` - Get recent data
- `GET /api/session/summary` - Get session summary
- `GET /api/history?days=7` - Get historical data

#### Machine Learning
- `GET /api/models/status` - Get model status
- `POST /api/models/train` - Train models
- `POST /api/predict` - Make prediction with custom features

### WebSocket Events

#### Client → Server
- `connect` - Establish connection
- `request_update` - Request immediate update

#### Server → Client
- `fatigue_update` - Real-time fatigue data
- `system_status` - System status update
- `model_training_complete` - Model training finished

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Typing         │    │  Facial         │    │  Wearable       │
│  Analyzer       │    │  Analyzer       │    │  Analyzer       │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼──────────────┐
                    │     Fusion System          │
                    │  - Weighted Average        │
                    │  - Confidence Weighted     │
                    │  - Adaptive Fusion         │
                    └─────────────┬──────────────┘
                                 │
                    ┌─────────────▼──────────────┐
                    │     AI Models              │
                    │  - Random Forest           │
                    │  - Gradient Boosting       │
                    │  - Neural Network          │
                    └─────────────┬──────────────┘
                                 │
                    ┌─────────────▼──────────────┐
                    │  Web Interface & API       │
                    │  - Real-time Dashboard     │
                    │  - REST API                │
                    │  - WebSocket Updates       │
                    └────────────────────────────┘
```

### Data Flow

1. **Data Collection**: Each analyzer collects data from its respective source
2. **Feature Extraction**: Raw data is processed into meaningful features
3. **Fusion**: Features from all modalities are combined using fusion strategies
4. **AI Prediction**: Machine learning models make fatigue predictions
5. **Decision Making**: Final fatigue score and recommendations are generated
6. **User Interface**: Results are displayed via web dashboard or API

## 🔧 Development

### Project Structure
```
fatigue-detection-system/
├── src/                    # Source code
│   ├── typing_analyzer.py  # Typing pattern analysis
│   ├── facial_analyzer.py  # Facial expression analysis
│   ├── wearable_analyzer.py # Wearable data integration
│   ├── ai_models.py        # Machine learning models
│   └── fusion_system.py    # Multi-modal fusion
├── templates/              # HTML templates
│   ├── base.html          # Base template
│   └── dashboard.html     # Main dashboard
├── static/                # Static files (CSS, JS, images)
├── data/                  # Data storage
├── models/                # Trained ML models
├── tests/                 # Test files
├── app.py                 # Flask web application
├── main.py                # Main entry point
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

### Adding New Features

1. **New Analyzer**: Create a new analyzer class in `src/`
2. **Integration**: Add to fusion system in `fusion_system.py`
3. **Web Interface**: Update templates and add API endpoints
4. **Testing**: Add tests to verify functionality

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Commit changes: `git commit -am 'Add feature'`
5. Push to branch: `git push origin feature-name`
6. Submit a pull request

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/ -v
```

### Integration Tests
```bash
python main.py test
```

### Performance Tests
```bash
python -m pytest tests/performance/ -v
```

## 📈 Performance

### Benchmarks
- **Typing Analysis**: ~1ms per keystroke
- **Facial Analysis**: ~100ms per frame (10 FPS)
- **Wearable Analysis**: ~10ms per data point
- **Fusion**: ~5ms per update
- **Web Interface**: <100ms response time

### Optimization Tips
- Reduce facial analysis FPS for better performance
- Adjust data collection intervals based on needs
- Use confidence-weighted fusion for better accuracy
- Enable only required modalities

## 🔒 Privacy & Security

### Data Privacy
- All data processing is performed locally
- No personal data is transmitted to external servers
- Facial images are analyzed in real-time and not stored
- User can control which modalities are enabled

### Security Features
- Local data storage with SQLite
- Configurable data retention periods
- Option to disable data logging
- Secure WebSocket connections

## 🐛 Troubleshooting

### Common Issues

#### Camera Not Detected
```bash
# Check available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).read()[0]])"

# Update camera index in config.yaml
facial:
  camera_index: 1  # Try different values
```

#### Permission Errors
```bash
# On Linux/macOS, ensure camera permissions
sudo usermod -a -G video $USER

# On Windows, check camera privacy settings
```

#### Performance Issues
- Reduce facial analysis FPS in config
- Disable unused modalities
- Close other resource-intensive applications
- Ensure adequate RAM availability

#### Model Training Fails
- Ensure sufficient disk space
- Check Python environment
- Verify all dependencies are installed
- Try reducing training data size

### Getting Help
- Check the [Issues](https://github.com/your-username/fatigue-detection-system/issues) page
- Review system logs in `fatigue_detection.log`
- Run diagnostics: `python main.py test`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenCV** for computer vision capabilities
- **MediaPipe** for facial landmark detection
- **scikit-learn** for machine learning algorithms
- **Flask** for web framework
- **Chart.js** for data visualization
- **Bootstrap** for UI components

## 📞 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@your-username](https://github.com/your-username)
- **Project**: [Fatigue Detection System](https://github.com/your-username/fatigue-detection-system)

---

**⚠️ Important**: This system is designed for research and personal use. It should not be used as the sole method for safety-critical fatigue detection in professional environments without proper validation and testing.