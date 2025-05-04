# 🔍 Vision-Validator

[Deutsche Version](README-de.md) | **English Version**

Vision-Validator is a powerful web application for testing, comparing, and applying YOLO models for object detection and instance segmentation. With a user-friendly interface, you can load various models, analyze images, and process live video streams.

## ✨ Features

- 🎯 **Multi-Model Support**: Supports YOLO .pt, .onnx, and .engine models
- 📸 **Live Camera Detection**: Real-time object detection via webcam
- 🖼️ **Image Upload & Detection**: Upload images for batch processing
- 🎨 **Instance Segmentation**: Pixel-perfect object outlines
- ⚙️ **Configurable Settings**: Confidence thresholds, mask colors, and transparency
- 💾 **Automatic Saving**: Detected objects are saved locally
- 🔄 **Model Management**: Upload, rename, and delete models
- 🚀 **GPU Acceleration**: Automatic GPU utilization when available
- 🧹 **Memory Management**: Automatic cleanup for stable operation

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (optional, for live detection)
- CUDA-capable GPU (optional, for acceleration)

### Step 1: Clone Repository
```bash
git clone https://github.com/Loky-Coffee/Yolo-Vision.git
cd Yolo-Vision
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

For CPU-only installation:
```bash
pip install -r requirements.txt
```

For GPU acceleration (CUDA 11.8):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Step 4: Start Application
```bash
python app.py
```

The application is now available at http://localhost:5000

## 📁 Project Structure

```
Yolo-Vision/
├── app.py                  # Main application
├── requirements.txt        # Python dependencies
├── models/                 # Model storage
├── uploads/               # Uploaded images
├── results/               # Detection results
├── templates/
│   └── index.html         # Frontend template
└── README.md              # This file
```

## 🚦 Getting Started

### 1. Upload Model
- Click "Choose File" to select a YOLO model (.pt, .onnx, .engine)
- The model is automatically detected and loaded

### 2. Object Detection

**For Images:**
- Click "Choose Image File"
- Select an image
- Click "Detect"
- Results will be displayed

**For Live Video:**
- Ensure a webcam is connected
- Live feed starts automatically
- Detected objects are automatically marked and saved

### 3. Adjust Settings
- **Confidence Threshold**: Minimum probability for detections (0-1)
- **Camera Resolution**: Set webcam resolution
- **Mask Color**: Segmentation mask color
- **Mask Alpha**: Mask transparency
- **Model Input Size**: Input size for the model

## 🎮 Usage

### Model Management
```python
# Load model
POST /load_model
{
    "model_index": 0  # Index from model list
}

# Rename model
POST /rename_model
{
    "model_index": 0,
    "new_name": "my_model.pt"
}

# Delete model
POST /delete_model/0  # Model index
```

### Detection
```python
# Image detection
POST /detect_image
Files: {'image': file}

# Reset cooldown
POST /reset_cooldown
```

## ⚙️ Configuration

### Settings in app.py
```python
settings = {
    'conf_threshold': 0.5,      # Confidence threshold
    'camera_width': 1280,       # Camera width
    'camera_height': 720,       # Camera height
    'mask_alpha': 0.5,          # Mask transparency
    'mask_color': '#FF0000',    # Mask color (hex)
    'model_input_width': 640,   # Model input width
    'model_input_height': 640   # Model input height
}
```

## 🐛 Troubleshooting

### Common Issues

**Camera not detected:**
- Check permissions
- Ensure camera isn't used by another application
- Close other tabs that might use the camera
- Test with: `cv2.VideoCapture(0)`

**Model won't load:**
- Check format compatibility (YOLO format)
- Ensure sufficient RAM/VRAM

**GPU not detected:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Logs
Run with logging:
```bash
python app.py 2>&1 | tee app.log
```

## 🚀 Performance Optimization

- **GPU Usage**: Automatically when CUDA is available
- **Model Optimization**: TensorRT models (.engine) for best performance
- **Memory Management**: Automatic cleanup every 5 minutes
- **Frame Rate**: Optimized to 30 FPS for smooth display

## 🔒 Security

- Automatic cleanup of old files (max. 20)
- Secure file uploads
- Thread-safe operations
- Automatic memory leak fixes

## 📚 Technologies Used

- **Backend**: Flask, OpenCV, PyTorch, Ultralytics
- **Frontend**: HTML, CSS, JavaScript
- **Computer Vision**: YOLO, OpenCV
- **Threading**: Python threading, concurrent.futures
- **Fonts**: Google Fonts (Poppins)

## 🤝 Contributing

Contributions are welcome! Please create a fork and open a pull request.

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [OpenCV Community](https://opencv.org/)
- All contributors and testers

Developed with ❤️ by [Loky Coffee](https://github.com/Loky-Coffee)