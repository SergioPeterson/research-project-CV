# Advanced Lane Detection with Memory

This project implements an advanced lane detection pipeline capable of processing camera feeds in real-time to identify lanes, estimate their curvature, and determine the vehicle's position relative to the lane center. The system is designed for autonomous racing applications and includes features for lane curvature calculation and guidance for a Model Predictive Controller (MPC).

## Features

- **Real-Time Lane Detection**: Detects lane lines on road images using a combination of color and gradient thresholding.
- **Lane Curvature Calculation**: Computes the curvature of lanes in meters for left and right lane lines.
- **Vehicle Center Offset**: Determines the vehicle’s offset from the lane center in real-world measurements.
- **Perspective Transformations**: Generates a bird's-eye view of the lane for easier processing.
- **Memory Integration**: Uses historical data for smoother lane detection and continuity.
- **Visualization**: Outputs processed images with lane areas highlighted, curvature data overlaid, and hotspots for better understanding.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [How It Works](#how-it-works)
    - [Preprocessing](#preprocessing)
    - [Lane Detection](#lane-detection)
    - [Curvature and Offset Calculation](#curvature-and-offset-calculation)
    - [Visualization](#visualization)
4. [Parameters](#parameters)
5. [Examples](#examples)
6. [Dependencies](#dependencies)
7. [Contributing](#contributing)
8. [License](#license)

---

## Installation

### Prerequisites

Ensure you have Python 3.8+ installed along with the required dependencies listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Clone the Repository

```bash
git clone https://github.com/SergioPeterson/research-project-CV.git
cd advanced-lane-detection
```

## Usage

### Running the Pipeline

1. Place test images in the `test_img/` directory.
2. Modify the `test_img` path in the script to point to your input images.
3. Run the main script to process images:

```bash
python lane_detection.py
```

### Output

Processed images will be displayed and saved with lane detection visualizations, including lane lines, highlighted regions, curvature data, and vehicle offset.

## How It Works

### Preprocessing

1. **Color Thresholding**:
   - Uses HSV color space to isolate white and red regions representing lanes.
2. **Gradient Thresholding**:
   - Applies Sobel filters in x and y directions to highlight lane edges.
3. **Perspective Transformation**:
   - Warps images to a bird's-eye view for accurate lane detection.

### Lane Detection

1. **Sliding Window Search**:
   - Identifies potential lane pixels using a histogram-based sliding window approach.
2. **Polynomial Fitting**:
   - Fits a second-order polynomial to the detected lane pixels.

### Curvature and Offset Calculation

1. **Curvature**:
   - Computes left and right lane curvatures in meters using real-world scaling.
2. **Offset**:
   - Determines the vehicle’s lateral offset from the lane center.

### Visualization

- Combines color and gradient thresholds for binary images.
- Highlights detected lanes in green and displays curvature and offset data.

## Parameters

- **Perspective Transformation**:
  - `src_pts` and `dst_pts`: Source and destination points for perspective transformation.
- **Sliding Window**:
  - `sliding_windows_per_line`: Number of windows used for lane detection.
  - `sliding_window_half_width`: Window width for detecting lane pixels.
- **Real-World Scaling**:
  - `real_world_lane_size_meters`: Real-world lane dimensions in meters.

## Examples

### Input Image

![Input Image](assets/input.jpg)

### Processed Output

![Processed Output](assets/output.jpg)

- Left Curvature: 350.5m
- Right Curvature: 400.2m
- Center Offset: 0.15m Left

## Dependencies

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Contributing

We welcome contributions! Please submit issues and pull requests for any improvements or bug fixes.

### Steps to Contribute

1. Fork the repository.
2. Create a new branch for your feature.
3. Commit your changes with clear messages.
4. Submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.


