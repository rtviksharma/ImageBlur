# Face & License Plate Blurring

A Python GUI application that automatically detects and blurs faces and license plates in images using YOLO models. Built with PyQt5 for easy batch processing with a user-friendly interface.

## Features

- **Automatic Detection**: Uses YOLO models to detect faces and license plates in images
- **Selective Blurring**: Choose to blur faces, license plates, or both
- **Batch Processing**: Process entire folders of images at once
- **Real-time Progress**: Live progress bar and status updates
- **Multiple Formats**: Supports JPG, JPEG, PNG, BMP, TIF, TIFF, and WebP images
- **Tiled Processing**: Efficient processing using sliding window technique for large images
- **PyInstaller Ready**: Includes path handling for standalone executable distribution

## Requirements

- Python 3.8 or newer
- Required Python packages:
  - PyQt5
  - ultralytics (YOLO)
  - opencv-python
  - numpy

## Installation

1. Install the required dependencies:
```bash
pip install pyqt5 ultralytics opencv-python numpy
```

2. Create a `models/` directory in your project folder

3. Download and place the following YOLO model files in the `models/` directory:
   - `yolov11n-face.pt` (for face detection)
   - `yolov11x-license-plate.pt` (for license plate detection)

## Usage

1. Run the application:
```bash
python main.py
```

2. In the GUI window:
   - Click "Browse Input" to select the folder containing images to process
   - Click "Browse Output" to select where processed images will be saved
   - Check/uncheck "Blur Faces" and "Blur Plates" based on your needs
   - Click "Start" to begin processing

3. The application will:
   - Load the YOLO models
   - Process each image in the input folder
   - Save blurred images to the output folder with original filenames
   - Show progress and status updates

4. Click "Stop" at any time to halt processing

## Configuration

The following parameters can be adjusted in the code:

- `TILE = 1536`: Size of processing tiles for large images
- `OVERLAP = 0.25`: Overlap ratio between tiles
- `CONF = 0.05`: Confidence threshold for detections
- `IOU = 0.5`: IoU threshold for non-maximum suppression
- `k = 51`: Gaussian blur kernel size (must be odd)

## How It Works

1. **Model Loading**: Loads pre-trained YOLO models for face and license plate detection
2. **Image Processing**: Divides large images into overlapping tiles for efficient processing
3. **Detection**: Runs YOLO inference on each tile
4. **Coordinate Mapping**: Maps detection coordinates back to the full image
5. **Blurring**: Applies Gaussian blur to detected regions
6. **Saving**: Saves processed images to the output directory

## Project Structure
IMAGE_BLUR/
├── main.py                        # Main application file
├── models/
│   ├── yolov11n-face.pt          # Face detection model
│   └── yolov11x-license-plate.pt # License plate detection model
└── README.md                      # Project documentation



## Technical Details

- **Threading**: Uses QThread for non-blocking processing
- **Memory Efficient**: Processes images in tiles to handle large files
- **Error Handling**: Gracefully handles corrupted or unreadable images
- **Cross-platform**: Works on Windows, macOS, and Linux

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tif, .tiff)
- WebP (.webp)


## Author

- Name: Rtvik Anilkumar Sharma
- Email: sharmartvik@gmail.com
- GitHub: https://github.com/rtviksharma
- LinkedIn: https://www.linkedin.com/in/rtvik-sharma-1b97b1204/
- Location: Bengaluru, India

