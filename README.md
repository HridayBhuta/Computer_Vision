# Computer Vision Toolkit

A comprehensive collection of computer vision applications and experiments developed during my time with a college Mars Rover team. This repository showcases practical implementations of fundamental CV algorithms using Python, OpenCV, and cutting-edge hardware like Intel RealSense cameras.

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Project Overview

This repository demonstrates a diverse range of computer vision applications, from real-time optical flow tracking to augmented reality effects. Each implementation focuses on practical applications that could be valuable in robotics, automation, and interactive systems.

### Key Features

- **Real-time Optical Flow Tracking** — Dense and sparse optical flow implementations with Lucas-Kanade and Farneback algorithms
- **Gesture-based Mouse Control** — Hand tracking with RealSense depth camera for touchless computer interaction  
- **Invisibility Cloak Effect** — Real-time color-based object removal using HSV masking
- **Stereo Depth Estimation** — Custom disparity map generation from stereo image pairs
- **Interactive Point Tracking** — Click-to-track feature point following across video frames

## Repository Structure

```
Computer_Vision/
├── src/
│   ├── invisibility_cloak.py
│   ├── opti_tracking.py
│   ├── optiflow_dense.py
│   ├── optiflow_sparse.py
│   ├── realsense.py
│   └── stereo_disparity.py
├── pyproject.toml
├── .gitignore
└── README.md
```

## Installation & Setup

### Prerequisites
- Python 3.12+
- Webcam (for most applications)
- Intel RealSense camera (for gesture mouse control)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/HridayBhuta/Computer_Vision.git
   cd Computer_Vision
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   # On Windows
   .venv/Scripts/activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   
   **Option A: Using pip**
   ```bash
   pip install numpy opencv-python matplotlib mediapipe pyrealsense2 pynput screeninfo
   ```
   
   **Option B: Using Poetry (recommended)**
   ```bash
   poetry install
   ```

## Applications & Usage

### 1. Invisibility Cloak
Creates a Harry Potter-style invisibility effect using color detection.

```bash
python src/invisibility_cloak.py
```

**Features:**
- Real-time background subtraction
- HSV-based red cloth detection
- Morphological operations for noise reduction
- Interactive setup with countdown timer

### 2. Gesture Mouse Control
Control your computer cursor using hand gestures and depth sensing.

```bash
python src/realsense.py
```

**Features:**
- MediaPipe hand tracking
- RealSense depth-based clicking (push forward to click)
- Smooth cursor movement with configurable sensitivity
- Visual depth indicator and click threshold display

### 3. Optical Flow Tracking

#### Dense Optical Flow
```bash
python src/optiflow_dense.py
```
- Farneback algorithm implementation
- HSV color visualization of motion vectors
- Real-time FPS monitoring

#### Sparse Optical Flow
```bash
python src/optiflow_sparse.py
```
- Lucas-Kanade tracking with feature detection
- Trajectory visualization
- Automatic feature point management

#### Interactive Point Tracking
```bash
python src/opti_tracking.py
```
- Click any point to start tracking
- Lucas-Kanade pyramid implementation
- Real-time feature point following

### 4. Stereo Disparity Mapping
Generate depth maps from stereo image pairs using custom block matching.

```bash
python src/stereo_disparity.py left_image.jpg right_image.jpg
```

**Features:**
- Custom block matching algorithm
- Gaussian smoothing for noise reduction
- Colorized disparity visualization with matplotlib
- Command-line interface for batch processing

## Technical Implementation Details

### Optical Flow Algorithms
- **Dense Flow**: Farneback method for pixel-level motion estimation
- **Sparse Flow**: Lucas-Kanade pyramid tracking for feature points
- **Visualization**: HSV color space mapping (hue = direction, value = magnitude)

### Hand Tracking Pipeline
This project was just for fun while I was tinkering with my depth camera
1. MediaPipe hand detection and landmark extraction
2. RealSense depth data acquisition and alignment
3. Coordinate mapping from camera space to screen space
4. Smoothing filters for stable cursor movement
5. Depth-based gesture recognition for clicking

### Color-based Object Detection
- HSV color space conversion for robust color detection
- Morphological operations (opening/dilation) for noise removal
- Binary masking for background replacement
- Real-time background capture and averaging
