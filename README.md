# SafeWalker
## Traffic Sign and Vehicle Detection using YOLOv8 and OpenCV

## Overview
This script utilizes YOLOv8 and OpenCV to detect traffic signs (pedestrian signals) and vehicles from a video feed. It analyzes the colors of pedestrian signals—red indicating "Do not cross the road" and green indicating "You may cross" — and provides voice alerts for both pedestrian signals and large vehicles passing in front of the client detected within the frame. The system intentionally does not analyze pedestrians, as this would be an inefficient use of resources and could lead to unnecessary disturbances.

## Members
- Eric Liu   eyl17@pitt.edu
- Thomas Tran  tdt17@pitt.edu  
- Lokesh Daita lod42@pitt.edu
- Abhinav Nath abn52@pitt.edu

## Features
- Detects traffic signals and vehicles in a video stream.
- Identifies the color of pedestrian signals using HSV color space.
- Provides real-time voice alerts for traffic signals and nearby vehicles.

## Requirements

To install the required dependencies, run the following command:
```bash
pip install opencv-python ultralytics numpy pyttsx3
```
## Libraries Used:

- OpenCV: For video capture and drawing bounding boxes, mostly used for detecting colors of signals.
- YOLOv8 (Ultralytics): For object detection mostly traffic related objects, cars, traffic lights, busses, etc.
- NumPy: For image processing.
- pyttsx3: For text-to-speech functionality.
  
## Usage
- Replace 0 with 'video_name' if you don't want webcam in line cap = cv2.VideoCapture(0)
- Run the script to start detection.
- Press control and c to quit the program.

## Citations
- This code was assisted by ChatGPT (OpenAI, 2024)
  https://chat.openai.com/
