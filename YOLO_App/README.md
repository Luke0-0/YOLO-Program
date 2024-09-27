# YOAT Video Player - README

## Introduction

This repository contains the code for a YOAT (Yet Another Object Tracker) video player application built using Python, customtkinter, and OpenCV. It allows users to:

* Load and play videos.
* Track objects using YOLOv8 object detection.
* Filter objects based on classes.
* Add manual bounding boxes.
* Edit and delete existing bounding boxes.
* Save the tracking data as a JSON file.

## Files

* **Application.py:** The main application file, responsible for the user interface and event handling.
* **VideoPlayer.py:** Contains the `VideoPlayer` class, which handles video playback, frame processing, and interaction with the UI.
* **yoatLogic.py:** Contains classes for object tracking (`ObjectTracker`), video management (`VideoManager`), and video processing (`VideoProcessor`).

## Dependencies

* Python 3.7+
* customtkinter
* OpenCV
* ultralytics
* torch
* torchvision

## Installation

1. Install the required dependencies using pip:

   ```bash
   pip install requirements.txt

## Usage
Download the YOLOv8 model file (yolov8n.pt) and place it in the same directory as the Python files.
Running the Application
Run Application.py using a Python interpreter:
python Application.py
Use code with caution.
Bash
The application window will appear.
Use the "Select File" button to load a video.
Use the controls to play, pause, rewind, fast forward, and seek through the video.
Use the "Add box" button to add manual bounding boxes.
Use the "Remove box" and "Edit box" buttons to modify bounding boxes.
Use the "Filter Settings" section to filter objects based on their classes.
Use the "Export JSON" button to save the tracking data to a tracked_objects.json file.
Usage
Keyboard Shortcuts:
Space: Play/Pause
Right Arrow: Skip to the next frame
Left Arrow: Skip to the previous frame
Up Arrow: Skip forward 60 frames
Down Arrow: Skip backward 60 frames
R: Remove box popup
A: Add box popup
E: Edit box popup
UI Elements:
Select File: Loads a video file.
Play/Pause: Plays or pauses the video.
Rewind/Fast Forward: Rewinds or fast-forwards the video by 60 frames.
Skip Frame: Skips forward or backward by one frame.
Progress Slider: Seeks to a specific frame.
Filter Settings: Filters objects based on class names.
Export JSON: Saves the tracking data to a JSON file.
Example: Tracking Objects in a Video
Load a video using the "Select File" button.
The video will start playing, and the YOLOv8 model will automatically detect objects.
Use the controls to navigate the video.
To add a manual bounding box:
Click the "Add box" button.
Draw a rectangle on the video window.
Enter the class name and the number of frames the box should remain.
The tracked objects and their IDs will be displayed on the video.
To export the tracking data:
Click the "Export JSON" button.
A tracked_objects.json file will be created in the same directory as the application.
## Notes
The requirements.txt and the YOLOv8 model file (yolov8n.pt) needs to be downloaded and placed in the same directory as the Python files.  
Link to GitLab Page: https://gitlab.cs.uct.ac.za/capstone3682133/capstone 
