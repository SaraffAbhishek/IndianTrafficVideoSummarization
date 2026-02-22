# 🚗 Indian Traffic Video Summarization using YOLO and Multi-Level Masking

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![React](https://img.shields.io/badge/React-19.1.0-61dafb.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-orange.svg)

**Video processing system for intelligent traffic object detection and background manipulation**

[🚀 Features](#-features) • [🛠️ Technology Stack](#️-technology-stack) • [📦 Installation](#-installation) • [🎯 Usage](#-usage) • [📊 Examples](#-examples)

</div>

---

## 🚀 Features

### 🎯 **Object Detection**
- **Custom YOLO Model**: Trained specifically for Indian traffic scenarios
- **14 Vehicle Classes**: Car, bike, bus, auto-rickshaw, e-rickshaw, tempo, truck, and more
- **Real-time Processing**: High-performance detection with configurable confidence thresholds

### 🎨 **7 Advanced Masking Techniques**

| Technique | Description | Use Case |
|-----------|-------------|----------|
| **Black & White** | Preserves road context while highlighting objects | Traffic analysis with context |
| **Complete Blackout** | Pure object focus with black background | Object isolation studies |
| **Blur Blackout** | Blurs non-objects while preserving context | Privacy-focused processing |
| **Complete Inpainting** | Removes background using advanced inpainting | Clean object extraction |
| **Gaussian Blur** | Smooth background blur for object focus | Professional presentations |
| **Object Inpainting** | Removes non-selected objects via Telea algorithm | Selective object retention |
| **Selective Blur** | Texture replacement for background elements | Enhanced visual focus |

### 🌐 **Modern Web Interface**
- **React Frontend**: Responsive, intuitive user interface
- **Real-time Progress**: Live processing status updates
- **Video Preview**: Before/after comparison capabilities
- **Drag & Drop**: Easy video upload functionality

### 🔧 **Robust Backend**
- **Flask API**: RESTful endpoints for video processing
- **CORS Support**: Cross-origin resource sharing enabled
- **File Management**: Automatic upload/output organization
- **Error Handling**: Comprehensive error management

---

## 🛠️ Technology Stack

### Backend
- **Python 3.8+** - Core programming language
- **Flask** - Web framework for API development
- **OpenCV** - Computer vision and image processing
- **Ultralytics YOLO** - Object detection model
- **NumPy** - Numerical computing
- **Flask-CORS** - Cross-origin resource sharing

### Frontend
- **React 19.1.0** - Modern JavaScript framework
- **Axios** - HTTP client for API communication
- **CSS3** - Styling and responsive design

### AI/ML
- **Custom YOLO Model** (`best_yolo11.pt`) - Trained on Indian traffic data
- **Multi-level Masking** - Advanced background manipulation techniques
- **Inpainting Algorithms** - Background removal and reconstruction

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Node.js 16+ and npm
- Git

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd -Indian-Traffic-Video-Summarization-using-YOLO-and-Multi-Level-Masking
   ```

2. **Install Python dependencies**
   ```bash
   pip install flask flask-cors opencv-python ultralytics numpy
   ```

3. **Download the YOLO model**
   ```bash
   # Place your custom YOLO model as 'best_yolo11.pt' in the root directory
   # The model should be trained on Indian traffic classes
   ```

4. **Start the Flask server**
   ```bash
   python process_video.py
   ```
   The API will be available at `http://127.0.0.1:5000`

### Frontend Setup

1. **Navigate to the React app directory**
   ```bash
   cd cvrc
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm start
   ```
   The web interface will be available at `http://localhost:3000`

---

## 🎯 Usage

### Web Interface

1. **Upload Video**: Drag and drop or select a traffic video file
2. **Select Object**: Choose from 14 available vehicle classes
3. **Choose Technique**: Pick from 7 masking techniques
4. **Adjust Confidence**: Set detection confidence threshold (0.1-1.0)
5. **Process**: Click "Process Video" to start analysis
6. **Download**: View and download the processed video

### API Usage

#### Upload Video
```bash
curl -X POST http://127.0.0.1:5000/api/upload \
  -F "file=@your_video.mp4"
```

#### Process Video
```bash
curl -X POST http://127.0.0.1:5000/api/process \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "your_file_id",
    "selected_object": "car",
    "selected_technique": "Black & White",
    "confidence_threshold": 0.5
  }'
```

---

### Object Classes
- `trak` (truck)
- `cyclist`
- `bike`
- `tempo`
- `car`
- `zeep`
- `toto`
- `e-rickshaw`
- `auto-rickshaw`
- `bus`
- `van`
- `cycle-rickshaw`
- `person`
- `taxi`

### Masking Techniques
1. **Black & White** - Grayscale background with colored objects
2. **Complete Blackout** - Black background, only objects visible
3. **Blur Blackout** - Blurred background with context preservation
4. **Complete Inpainting** - Background removal with inpainting
5. **Gaussian Blur** - Gaussian blur on background
6. **Object Inpainting** - Remove non-selected objects
7. **Selective Blur** - Texture replacement for background

---

## 📊 Examples

### Sample Output Files
The system generates various output formats:
- `BlacknWhite.mp4` - Black and white background technique
- `CompleteBlackout.mp4` - Complete blackout technique
- `GaussianBlur.mp4` - Gaussian blur technique
- `CompleteInpainting.mp4` - Inpainting technique
- And more...

---

## 🏗️ Project Structure

```
├── process_video.py          # Flask backend server
├── cvrc/                     # React frontend application
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── App.jsx          # Main application
│   │   └── apiService.js    # API communication
│   └── package.json         # Frontend dependencies
├── uploads/                  # Uploaded video storage
├── outputs/                  # Processed video storage
└── best_yolo11.pt           # Custom YOLO model
```

---

## 🚀 Performance

- **Processing Speed**: Real-time video processing capabilities
- **Accuracy**: High-precision object detection with custom YOLO model
- **Scalability**: Modular architecture for easy scaling
- **Memory Efficient**: Optimized for large video files

---

## 🙏 Acknowledgments

- **Ultralytics** for the YOLO framework
- **OpenCV** for computer vision capabilities
- **React** team for the frontend framework
- **Flask** team for the web framework

---

<div align="center">

**Made with ❤️ for Indian Traffic Analysis**

</div> 
