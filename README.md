# Final-project

# Object Detection and Image Classification using YOLO, DeiT, and CCT

This script integrates Ultralytics' YOLO for object detection, Hugging Face's DeiT for image classification, and a Compact Convolutional Transformer (CCT) for feature extraction using a webcam feed.

## Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)
- PyTorch (`pip install torch torchvision`)
- Transformers (`pip install transformers`)
- Ultralytics (`pip install yolov5`) - Ensure yolov8n.pt is in the "yolo-Weights" directory.

## Description

The script captures video frames from the webcam and performs the following tasks:

1. **Object Detection (YOLO)**:
   - Utilizes YOLOv8n for real-time object detection.
   - Defines a range of classes for object detection.

2. **Image Classification (DeiT)**:
   - Loads a pre-trained DeiT model from Hugging Face's transformers for image classification.
   - Uses a collection of class names for classification.

3. **Feature Extraction (CCT)**:
   - Utilizes a CCT model for feature extraction within the Region of Interest (ROI) identified by YOLO.
   - Evaluates and retrieves predictions from the CCT model.

## Instructions

1. Ensure all required libraries are installed.
2. Download the `yolov8n.pt` file and place it in the "yolo-Weights" directory.
3. Run the script.
4. The webcam will display frames with bounding boxes around detected objects, along with labels obtained from YOLO and predictions from DeiT using CCT-extracted features.

## Usage

1. Connect your webcam or provide a specific camera index.
2. Execute the script.
3. Press 'Esc' key code to exit the webcam frame display.

## Credits

- YOLO: Ultralytics' YOLO (https://github.com/ultralytics/yolov5)
- DeiT: Hugging Face's Transformers (https://huggingface.co/models)

Please refer to the code comments for more details on the implementation.

