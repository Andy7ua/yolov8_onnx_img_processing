# YOLOv8 - OpenCV Integration

This repository provides a pipeline for object detection using the YOLOv8 model in ONNX format, integrated with OpenCV. It includes a standard set of YOLOv8 weights and a script that processes images to classify objects.

## Getting Started

### Prerequisites

To run this project, you'll need to set up Python and install all required dependencies:

1. **Clone the repository**:
   ```
   git clone https://github.com/Andy7ua/yolov8_onnx_img_processing.git
   cd yolov8_onnx_img_processing
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Export YOLOv8 model to ONNX** (if starting from scratch):
   ```bash
   yolo export model=yolov8n.pt imgsz=640 format=onnx opset=12
   ```
   _Make sure to include "opset=12" as it specifies the ONNX version to be used._

### Running the Object Detection

To perform object detection, navigate to the directory containing `main.py`, and run the following command:

```bash
python main.py --model yolov8n.onnx --img image.jpg
```

Replace `image.jpg` with the name of your image file. By default, the output image will be saved in the current directory. You can change the output location by modifying line 119 in `main.py`.

## Image Processing

To apply filters and further process the images, you can use the `img_processing.py` script. Before running, ensure to adjust the paths for the input/output images and set the desired number of processing iterations at the end of the file.

To run the script, execute:

```bash
python img_processing.py
```

This will apply predefined filters to the images and save the results.

## Video Processing

To apply filters and further process the video, you can use the `video_proc.py` script. Before running, ensure to adjust the paths for the input/output videos. Also, you can change saving and filtering or number of combination filters.

To run the script, execute:

```bash
python video_proc.py
```

This will apply predefined filters to the video and save the result.

## Further Customization

You can customize the processing behaviors by modifying the scripts according to your project needs. This might include changing the filters used, adjusting image paths, or modifying export parameters for the ONNX model.

## Troubleshooting

If you encounter issues with model exports or other steps in the process, review the setup instructions and ensure all dependencies are correctly installed. Check the official Ultralytics and OpenCV documentation for additional guidance.

---

This README file provides a comprehensive guide to setting up and running the YOLOv8 object detection pipeline, including both standard and custom image processing steps.
