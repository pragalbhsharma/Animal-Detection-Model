# Animal-Detection-Model

Project Overview:

This project implements an Animal Detection and Classification System using deep learning techniques.
The model is capable of detecting multiple animals in a single image or video frame, identifying their species, and highlighting carnivorous animals in red.
Additionally, the system displays the count of detected carnivorous animals as an alert message.

This project is developed as part of an academic machine learning assignment and is designed to run on Google Colab.

# Features:

1.Detects multiple animals in images and videos

2.Classifies animals into different species

3.Highlights carnivorous animals in red

4.Highlights non-carnivorous animals in green

5.Displays number of carnivorous animals detected

6.Supports image and video input

7.Preview of processed image/video after detection

8.Colab-compatible (no local system dependency)

Model & Technology Stack
- Deep Learning Model

YOLOv8 (You Only Look Once)
A real-time object detection model trained on the COCO dataset.

# Libraries Used:

Python

OpenCV

Ultralytics (YOLOv8)

Pandas

NumPy

Matplotlib

Dataset Details:

Dataset Used: COCO128

Format:

Images stored in folders

CSV file used to store image paths and labels

CSV Structure:
image_path,label
/content/coco128/images/train2017/000000000009.jpg,train2017


Note:
The CSV file stores metadata and image paths.
The actual image files are loaded dynamically during runtime.

Carnivorous Animal Logic

The following animals are treated as carnivorous in this project:

Lion

Tiger

Bear

Wolf

Leopard

Dog

Cat

Fox

Carnivorous animals are:

 Marked with red bounding boxes

 Counted and displayed as an alert message

 Image Detection Workflow

Load image path from CSV

Read image using OpenCV

Perform object detection using YOLOv8

Classify detected animals

Draw bounding boxes with color coding

Display detection results and carnivore count

# Video Detection Workflow:

Upload video file

Process video frame-by-frame

Detect animals in each frame

Highlight carnivores in red

Display live carnivore count on video

Colab Limitation Notice:

Google Colab does not support desktop GUI pop-ups (e.g., Tkinter).
Therefore, alert messages are implemented using:

On-frame text overlays

Console output messages

This approach is widely accepted for cloud-based execution environments.

How to Run the Project (Google Colab):

Open Google Colab

Install dependencies:

pip install ultralytics opencv-python-headless pandas matplotlib


Download the COCO128 dataset

Upload the CSV file

Run the image or video detection notebook cells

Sample Output:

Detected animals highlighted with bounding boxes

Carnivorous animals marked in red

Console alert showing:

Carnivorous Animals Detected: 2



# System Architecture:
Architecture Overview:

The Animal Detection Model follows a modular deep learning pipeline:

Input Image / Video
        ↓
Preprocessing (OpenCV)
        ↓
YOLOv8 Object Detection Model
        ↓
Animal Classification
        ↓
Carnivore Identification Logic
        ↓
Bounding Box Visualization
        ↓
Alert Message & Output Display

# Explanation:

Input Layer: Accepts image or video input

Preprocessing: Frame resizing and color conversion using OpenCV

Detection Layer: YOLOv8 detects objects and classifies animal species

Business Logic Layer: Identifies carnivorous animals based on predefined rules

Visualization Layer: Draws bounding boxes and displays alerts

Output Layer: Displays processed image/video and carnivore count

# Training Details:

YOLOv8 is a pretrained deep learning model trained on the COCO dataset

The model is used for transfer learning-based inference

No manual labeling was required due to pretrained weights

Custom logic was added for carnivorous animal identification

# CSV Usage:

The CSV file is used to:

Store image paths

Enable batch processing

Maintain dataset scalability

Future Scope & Enhancements:

This project can be further enhanced in the following ways:

1.Fine-tuning YOLOv8 specifically on animal-only datasets

2.Adding herbivore, omnivore, and endangered species classification

3.Implementing real-time alert notifications

4.Integrating a desktop or web-based GUI

5.Deploying the model using Flask / FastAPI

6.Using GPS-based alerts for wildlife monitoring

7.Storing detection results in a database for analytics
