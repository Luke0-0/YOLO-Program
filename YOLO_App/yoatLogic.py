"""
YOLO software detection program

Apply the YOLO model to an image or video and allow the user to move through the video and add, remove, and edit bounding boxes 
through keyboard and terminal commands.

Authors: Luca Biccari, Michael Haydam, Luke Reinbach
Date: 20/09/2024
"""

import json
import cv2
from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image


class BoundingBox:
    def __init__(self, coords):
        """
        Initialize BoundingBox object with coordinates.
        
        Args:
            This takes a coords tuple of four values representing the bounding box coordinates (x1, y1, x2, y2).
        """
        self.x1, self.y1, self.x2, self.y2 = coords

    def calculate_iou(self, other):
        """
        Calculate the Intersection over Union (IoU), or overlap, between this bounding box and another.
        
        Args:
            This takes another bounding box as an argument and calculates IoU. 
        
        Returns:
            Number between 0.0 (no overlap) and 1.0 (perfect overlap).
        """
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = (self.x2 - self.x1) * (self.y2 - self.y1)
        box2_area = (other.x2 - other.x1) * (other.y2 - other.y1)

        if box1_area + box2_area - inter_area == 0:
            return 0.0

        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou
    
    def get_coordinates(self):
        """
        Retrieve the coordinates of the bounding box.
        
        Returns:
             a tuple containing the coordinates (x1, y1, x2, y2).
        """
        return (self.x1, self.y1, self.x2, self.y2)
    
    # Convert to a dictionary (JSON-serializable)
    def to_dict(self):
        """
        Convert the bounding box coordinates to a dictionary format.
        
        Returns:
            dict: A dictionary containing the bounding box coordinates with keys 'x1', 'y1', 'x2', 'y2'.
        """
        return {'x1': int(self.x1), 'y1': int(self.y1), 'x2': int(self.x2), 'y2': int(self.y2)}
    

class ObjectTracker:
    def __init__(self):
        """
        Initialize the ObjectTracker with dictionaries for tracked objects, manual boxes, and deleted boxes, as well as other variables
        used for frame tracking.
        """
        self.tracked_objects = {}
        self.id_mapping = {}
        self.manualBoxes = {}
        self.deletedBoxes = []
        self.frameData = {}
        self.currentFrame = 0
        self.selected_box = None  # Holds the current box being moved
        self.offset_x = 0  # Offset from where the user clicked inside the box
        self.offset_y = 0

    def assign_unique_id(self, obj_class, bbox, previous_bboxes):
        """
        Assign a unique ID to a newly detected object based on IoU.
        
        Args:
            obj_class: class of the detected object.
            bbox: bounding box coordinates of the detected object.
            previous_bboxes: dictionary of previously detected bounding boxes.
        
        Returns:
            new ID
        """
        best_iou = 0
        best_id = None
        bbox = BoundingBox(bbox)
        for obj_id, obj_info in previous_bboxes.items():
            previous_bbox = BoundingBox(obj_info['bbox'])
            iou = bbox.calculate_iou(previous_bbox)
            if iou > best_iou and iou > 0.5:  # Threshold to match the same object
                best_iou = iou
                best_id = obj_id

        if best_id is not None:
            if best_id in self.id_mapping:
                best_id = self.id_mapping[best_id]  # Apply the new ID if mapped
            return best_id
        else:
            new_id = len(self.tracked_objects) + 1
            self.tracked_objects[new_id] = {'class': obj_class, 'bbox': bbox}
            return new_id

    def modify_object_id(self, old_id, new_id, current_bboxes, apply_globally=False):
        """
        Modify the ID of an object, with the option of applying changes to all future instances.
        
        Args:
            old_id: ID to be changed.
            new_id: New ID to be assigned.
            current_bboxes: Dictionary of bounding boxes for the current frame.
            apply_globally: If True, change the ID for future instances of this object.
        """
        if old_id in self.tracked_objects:
            # Update the in-memory dictionary
            self.tracked_objects[new_id] = self.tracked_objects.pop(old_id)
            self.id_mapping[old_id] = new_id
            print(f"Changed object ID {old_id} to {new_id}")

            if apply_globally:
                self.id_mapping[old_id] = new_id
                print(f"Future instances of object {old_id} will also be changed to {new_id}")

            if old_id in current_bboxes:
                current_bboxes[new_id] = current_bboxes.pop(old_id)
        else:
            print(f"Object ID {old_id} not found")
            
    # Change the id of an object
    def modify_id(self, current_bboxes, frame):
        """
        Prompt the user to enter an object ID to change the ID and display the new ID
        
        Args:
            current_bboxes: Dictionary of current bounding boxes.
            frame: The current frame being displayed.
        """
        print("Enter object ID to modify")
        
        # Check for valid inputs
        while True:
            try:
                old_id = int(input("Old ID: "))
                new_id = int(input("New ID: "))
                break
            except ValueError:
                print("Invalid input, please enter an integer.")
              
        if old_id not in self.tracked_objects:
            print (f"Object {old_id} does not exist.") 
                   
        else:
            apply_globally_input = input("Apply change to all future instances of this object? (y/n): ").lower()
            apply_globally = apply_globally_input == 'y'

            self.modify_object_id(old_id, new_id, current_bboxes, apply_globally)

            for obj_id, obj_info in current_bboxes.items():
                bbox = obj_info['bbox']
                obj_class = obj_info['class']

                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                cv2.putText(frame, f'{obj_class}{obj_id}', (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow('YOLOv8 Object Detection', frame)
            
    # Draw a new bounding box around on screen    
    def createBox(self, frame):
        """
        Allow the user to manually draw a new bounding box on the current frame.
        
        Args:
            frame: The current frame where the box will be drawn.
        """  
        obj_class = input("Enter the class of the object: ")     
        # Check for valid input
        while True:
            try:
                numFrames = int(input("Enter the number of frames you would like the bounding box to remain: "))
                break
            except ValueError:
                print("Invalid input, please enter an integer.")
        
        obj_id = len(self.tracked_objects) + 1
        
        # Get coordinates of bounding box from user input
        x, y, w, h = cv2.selectROI('YOLOv8 Object Detection', frame, fromCenter = False, showCrosshair = False)
        bbox = [x, y, (x + w), (y + h)]
        bbox = BoundingBox(bbox)
        print(self.manualBoxes)

        self.manualBoxes[obj_id] = {'bbox': bbox, 'start_frame': self.currentFrame, 'end_frame': self.currentFrame+numFrames}
        
        self.tracked_objects[obj_id] = {"class": obj_class, "bbox": bbox}
        
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'{obj_class}{obj_id}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        print(f"Created bounding box {bbox} with ID {obj_id}")
        cv2.imshow('YOLOv8 Object Detection', frame)
 
# Move specified bounding box
    def moveBox(self, frame):
        """
        Move a bounding box to a new position based on user input. Prompt user to enter ID of box to be moved, enter direction to be moved,
        and adjust bounding box coordinates of object
        
        Args:
            frame: current video frame where the bounding box will be drawn.
        """
        # Check for valid input
        while True:
            try:
                obj_id = int(input("Enter ID of bounding box to be moved: "))
                break
            except ValueError:
                print("Invalid ID, please enter an integer.")
            
        if int(obj_id) not in self.tracked_objects and int(obj_id) not in self.manualBoxes:
            print(f"Object {obj_id} does not exist.")
        
        else:
            bbox = self.tracked_objects[obj_id]['bbox']
            obj_class = self.tracked_objects[obj_id]['class']
            
                
            # Ensure bbox is a BoundingBox instance, otherwise handle accordingly
            if not isinstance(bbox, BoundingBox):
                bbox = BoundingBox(bbox)

            print(f"Current bounding box: {bbox.x1}, {bbox.y1}, {bbox.x2}, {bbox.y2}")
            
            # Ask for direction and number of pixels to move
            direction = input("Enter direction to move the box (up, down, left, right): ").lower()
            while direction not in ['up', 'down', 'left', 'right']:
                direction = input("Invalid direction. Enter again (up, down, left, right): ").lower()

            # Check for valid pixel input
            while True:
                try:
                    pixels = int(input("Enter number of pixels to move: "))
                    numFrames = int(input("Enter the number of frames you would like the bounding box to remain: "))
                    break
                except ValueError:
                    print("Invalid input, please enter an integer.")

            # Get the current bounding box
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

            # Adjust the bounding box based on the direction
            if direction == 'up':
                y1 -= pixels
                y2 -= pixels
            elif direction == 'down':
                y1 += pixels
                y2 += pixels
            elif direction == 'left':
                x1 -= pixels
                x2 -= pixels
            elif direction == 'right':
                x1 += pixels
                x2 += pixels

            # Make sure the bounding box stays within the frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # Update the bounding box
            self.deletedBoxes.append(int(obj_id))
            print(f"Bounding box {obj_id} deleted.", end =" ")
            obj_id = len(self.tracked_objects) + 1
            self.manualBoxes[obj_id] = {'bbox': [x1, y1, x2, y2], 'start_frame': self.currentFrame, "end_frame": self.currentFrame+numFrames}
            bbox = BoundingBox([x1, y1, x2, y2])
            self.tracked_objects[obj_id] = {"class": obj_class, "bbox": bbox}
            print(f"Moved new bounding box {obj_id} to {[x1, y1, x2, y2]}")
            
            # Redraw the updated bounding box on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f'{obj_class.capitalize()}{obj_id}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.imshow('YOLOv8 Object Detection', frame)
            
    # Delete specified bounding box
    def removeBox(self):
        """
        Delete a specified bounding box based on user input. Prompt user with ID of box to delete and then append box ID 
        to deleted boxes list.
        """
        # Check for valid input
        while True:
            try:
                obj_id = int(input("Enter ID of bounding box to be deleted: "))
                break
            except ValueError:
                print("Invalid ID, please enter an integer.")
        
        if int(obj_id) in self.deletedBoxes:
            print(f"Object {obj_id} has already been deleted.")
            
        elif int(obj_id) not in self.tracked_objects:
            print(f"Object {obj_id} does not exist.")
            
        else:
            self.deletedBoxes.append(int(obj_id))
            print(f"Bounding box {obj_id} deleted.")
        
    # Restore a previously deleted bounding box
    def restoreBox(self):
        """
        Restore a previously deleted bounding box based on user input. Remove object ID from deleted boxes list
        """
        print("Deleted IDs: ", end = "")
        print(*self.deletedBoxes, sep =", ")
        
        # Check for valid input
        while True:
            try:
                obj_id = int(input("Enter ID of bounding box to be restored: "))
                break
            except ValueError:
                print("Invalid ID, please enter an integer.")
        
        if int(obj_id) not in self.tracked_objects:
            print(f"Object {obj_id} does not exist.")
        
        elif int(obj_id) not in self.deletedBoxes:
            print(f"Object {obj_id} has not been deleted.")
        
        else:
            self.deletedBoxes.remove(int(obj_id))
            print(f"Bounding box {obj_id} restored.")
        

class VideoManager:
    """
    Read in the frames and video.
    """
    def __init__(self, video_path, output_path):
        """
        Initialize the video manager with given video paths.
        
        Args:
            video_path: Path to input video file.
            output_path: Path where output video will be saved.

        """
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'X264'), self.fps, (self.width, self.height))

    def release(self):
        """
        Releases video capture and closes any open OpenCV windows.
        """
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

    def write_frame(self, frame):
        """
        Write a frame to the output video file.

        Args:
            frame: current frame to be written.
        """
        self.out.write(frame)

    def read_frame(self):
        """
        Reads the next frame from the input video.

        Returns:
            A tuple containing the read status and the frame.
        """
        return self.cap.read()

class VideoProcessor:
    """
    Class for processing video frames, handling object detection, tracking, and class filters.
    """
    
    def __init__(self, model_path, video_manager, confidence_threshold=25):
        """
        Initializes the VideoProcessor with the YOLO model, video manager, and confidence threshold.

        Args:
            model_path: Path to the YOLO model file.
            video_manager: instance of VideoManager, handles video operations.
            confidence_threshold: how confient the user wants the detection software to be (default is 25).
        """
        self.model = YOLO(model_path)
        self.video_manager = video_manager
        self.tracker = ObjectTracker()
        self.previous_bboxes = {}
        self.confidence_threshold = confidence_threshold / float(100)
        self.coco_classes = self.load_coco_classes()
        self.filtered_classes = set()
        self.is_filtering = False 
        self.processed_frames = {}  # Format: {frame_number: (frame, bboxes)}

    def set_confidence_threshold(self, confidence_threshold):
        """
        Set a new confidence threshold for object detection.

        Args:
            confidence_threshold: New confidence threshold percentage to be added.
        """
        self.confidence_threshold = confidence_threshold / float(100)

    def load_coco_classes(self):
        """
        Loads COCO class names from 'coco.txt'.
        
        Returns:
            A list of COCO class names.
        """
        coco_classes = []
        try:
            with open('coco.txt', 'r') as file:
                coco_classes = [line.strip() for line in file]
        except FileNotFoundError:
            print("Warning: coco.txt file not found. Class filtering may not work correctly.")
        return coco_classes

    def set_filtered_classes(self, classes):
        """
        Sets the list of classes to filter, with the option to enable or disable

        Args:
            classes: list of classes to filter and keep.
        """
        self.filtered_classes = set(classes)
        self.is_filtering = bool(self.filtered_classes)

    def set_class_filter(self):
        """
        Allows the user to select specific COCO classes to filter.
        Prompts the user to enter class names and sets the filter.
        """
        print("Available classes:")
        for cls in self.coco_classes:
            print(cls)
        selected_classes = ''
        while selected_classes not in self.coco_classes:
         selected_classes = input("Enter the names of classes to keep (comma-separated): ").strip()
        
        filtered_classes = [cls.strip() for cls in selected_classes.split(',') if cls.strip() in self.coco_classes]
        
        if filtered_classes:
            self.set_filtered_classes(filtered_classes)
            print(f"Filtering set to: {', '.join(filtered_classes)}")
        else:
            print("No valid classes entered. Filter not applied.")

    def turn_off_filter(self):
        """
        Disables class filtering, makes all object classes shown.
        """
        self.set_filtered_classes(set())
        print("Filtering turned off. All classes will be shown.")   

    def navigate_frames(self, step, direction):
        """
        Navigate through video frames by rewinding or fast forwarding based on user input.

        Args:
            step: number of frames to move forward or backward.
            direction: direction to move, either 'rewind' to move backward or 'fast_forward' to move forward.

        If the direction is invalid, an error message is displayed.
        """
        if direction == 'rewind':
            # Decrease current frame by step
            self.tracker.currentFrame = max(self.tracker.currentFrame - step, 0)
        elif direction == 'fast_forward':
                # Increase current frame by step
                total_frames = int(self.video_manager.cap.get(cv2.CAP_PROP_FRAME_COUNT))

                self.tracker.currentFrame = min(self.tracker.currentFrame + step, total_frames - 1)
        else:
                print("Error: Invalid direction. Use 'rewind' or 'fast_forward'.")
                return
        
        # Set the video capture position to the new frame
        self.video_manager.cap.set(cv2.CAP_PROP_POS_FRAMES, self.tracker.currentFrame)

        # self.video_manager.cap.set(cv2.CAP_PROP_POS_FRAMES, self.processor.tracker.currentFrame)

        
        # Read the frame from the new position
        ret, frame = self.video_manager.cap.read()
        
        if ret:
            # Process the frame as needed
            frame, _ = self.process_frame(frame)
            
            # Display the frame
            cv2.imshow('YOLOv8 Object Detection', frame)
        else:
            print("Error: Could not read the frame. It might be out of bounds.")
        
        # Print status message
        if direction == 'rewind':
            print(f"Rewinded to frame {self.tracker.currentFrame}. Press 'c' to continue or 'f' to fast forward.")
        elif direction == 'fast_forward':
            print(f"Fast forwarded to frame {self.tracker.currentFrame}. Press 'r' to rewind or 'c' to continue.")
                
    def set_reprocess(self, frameNum):
        """
        Clears processed frames from a given frame number onwards, allowing frames to be reprocessed.

        Args:
            frameNum: frame number where frames should be cleared and reprocessed.
        """
        self.processed_frames = {key: value for key, value in self.processed_frames.items() if key < frameNum}

    def process_frame(self, frame):
        """
        Processes a frame for object detection, tracking, and handling of manual bounding boxes.

        Args:
            frame: current video frame to be processed.

        Returns:
            frame: processed frame with bounding boxes and labels drawn.
            current_bboxes: dictionary of currently detected bounding boxes.

        This function also updates the tracker with detected objects and handles both automatic YOLO detections
        and manually drawn bounding boxes.
        """
        frame_number = self.tracker.currentFrame

        # # Check if the frame has been processed before
        # if frame_number in self.processed_frames:
        #     print(f"Frame {frame_number} already processed. Retrieving from cache.")
        #     self.tracker.currentFrame += 1
        #     return self.processed_frames[frame_number]  # Return cached frame and bboxes
        
        print("Current Frame: ", self.tracker.currentFrame)
        results = self.model(frame)
        current_bboxes = {}
        confidenceLevel = 0

        for result in results:
            for box in result.boxes:
                if box.conf[0] < self.confidence_threshold:  # Filter boxes by confidence
                    continue
                obj_class_id = int(box.cls[0])
                obj_class = result.names[obj_class_id]

                if self.is_filtering and obj_class not in self.filtered_classes:
                            continue
                
                confidenceLevel = round(float(box.conf[0]), 2)
                bbox = box.xyxy[0].tolist()
                obj_id = self.tracker.assign_unique_id(obj_class, bbox, self.previous_bboxes)
                current_bboxes[obj_id] = {'class': obj_class, 'bbox': bbox}

                confidence = box.conf[0] * 100
                if obj_id not in self.tracker.deletedBoxes: # check if bounding box has been deleted, if not create bounding box
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    cv2.putText(frame, f'{obj_class}{obj_id}:{confidence:.1f}%', (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)    
                                   
        # Process manual bounding boxes
        for obj_id in list(self.tracker.manualBoxes.keys()):
            if obj_id not in self.tracker.deletedBoxes: # check if bounding box has been deleted, if not create bounding box        
                if self.tracker.manualBoxes[obj_id]['start_frame'] <= self.tracker.currentFrame <= self.tracker.manualBoxes[obj_id]['end_frame']: # check if box has reached its designated frame limit and does not appear before the frame it was created on

                    obj_class = self.tracker.tracked_objects[obj_id]["class"]

                    if self.is_filtering and obj_class not in self.filtered_classes:
                            continue
                    bbox = self.tracker.manualBoxes[obj_id]["bbox"]
                    # Ensure bbox is a BoundingBox instance
                    if not isinstance(bbox, BoundingBox):
                        bbox = BoundingBox(bbox)  # Convert list to BoundingBox
                    
                    confidenceLevel = 100 
                    x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{obj_class}{obj_id}: MANUAL', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                

        self.previous_bboxes = current_bboxes.copy()
        
        # Add specific frame info to list for JSON file
        for obj_id, obj_info in self.tracker.tracked_objects.items():
            if obj_id not in self.tracker.deletedBoxes: # check if bounding box has been deleted, if not create bounding box 
                # If the frame doesn't already exist in frameData, initialize it as a list
                if self.tracker.currentFrame not in self.tracker.frameData:
                    self.tracker.frameData[self.tracker.currentFrame] = []
                
                # Store the object ID and its associated data (class and bbox) in the frameData
                obj_class = obj_info["class"]
                bbox = obj_info["bbox"]

                # Ensure bbox is a dictionary for easy JSON serialization
                if isinstance(bbox, BoundingBox):
                    bbox = bbox.to_dict()

                # Append the object data for this frame
                self.tracker.frameData[self.tracker.currentFrame].append({
                    "ID": obj_id,
                    "class": obj_class,
                    "Confidence Level": confidenceLevel,
                    "bbox": bbox
                })
            
         # Store the processed frame and bounding boxes in the cache
        self.processed_frames[frame_number] = (frame, current_bboxes)

        # Move to the next frame
        self.tracker.currentFrame += 1
        return frame, current_bboxes

class Application:
    def __init__(self, model_path, video_path, output_path):
        """
        Main application class to handle video processing using YOLO model. 
        Initializes the video manager, video processor, and provides user input for interaction.

        Args:
            model_path: Path to the YOLO model file.
            video_path: Path to the input video file.
            output_path: Path to save the processed output video.
        """
        self.video_manager = VideoManager(video_path, output_path)
        self.processor = VideoProcessor(model_path, self.video_manager)
    
    def set_confidence_threshold(self):
        """
        Prompts the user to input the confidence threshold for object detection. 
        If there is no input the default is 0.25
        """
        confidence_threshold = float(input("Enter the confidence threshold (0.0 to 1.0) or 'd' for default(=0.25): ").strip())
        if confidence_threshold=='d':
            confidence_threshold=0.25
        self.processor.set_confidence_threshold(confidence_threshold)
        
    def run(self):
        """
        Runs the main video processing loop, with user ability to manipulate video playback and bounding box commands
        """
        self.set_confidence_threshold()  # Set confidence threshold at the start
        paused = False
        fast_forward_step = 20  # Number of frames to fast forward
        rewind_step = 20  # Number of frames to rewind
    
        total_frames = int(self.video_manager.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            if not paused:
                ret, frame = self.video_manager.read_frame()
                if not ret:
                    print("End of video")
                    break

                frame, current_bboxes = self.processor.process_frame(frame)
                self.video_manager.write_frame(frame)
                cv2.imshow('YOLOv8 Object Detection', frame)

            key = cv2.waitKey(30) & 0xFF
            # Press 'p' to pause the video
            if key == ord('p'):
                paused = True
                print("Video paused. Press 'c' to continue, 'b' to draw a bounding box, 'z' to edit an existing box, 'd' to delete a bounding box, 'v' to restore a bounding box, f to filter out objects, or 'm' to modify an object ID.")
            
            # Press 'c' to unpause the video 
            elif key == ord('c') and paused:
                paused = False
                
            #Filter out boxes    
            elif key == ord('f') and paused:     #This is to filter out boxes
                 self.processor.set_class_filter()

            #Remove filters that are applied
            elif key ==ord('e'):
                self.processor.turn_off_filter()

            # Fast forward by a certain number of frames
            elif key == ord('n') and paused:
                # Increase current frame by fast_forward_step
                self.processor.navigate_frames(rewind_step,"fast_forward")

                # self.processor.tracker.currentFrame = min(self.processor.tracker.currentFrame + fast_forward_step, total_frames)
                # self.video_manager.cap.set(cv2.CAP_PROP_POS_FRAMES, self.processor.tracker.currentFrame)
                # cv2.imshow('YOLOv8 Object Detection', frame)
                print(f"Fast forwarded to frame {self.processor.tracker.currentFrame}. Press 'c' to continue or 'r' to rewind.")

            # Rewind by a certain number of frames
            elif key == ord('r') and paused:
                # Decrease current frame by rewind_step
                self.processor.navigate_frames(rewind_step,"rewind")



                # self.processor.tracker.currentFrame = max(self.processor.tracker.currentFrame - rewind_step, 0)
                # cv2.imshow('YOLOv8 Object Detection', frame)
                # self.video_manager.cap.set(cv2.CAP_PROP_POS_FRAMES, self.processor.tracker.currentFrame)
                # print(f"Rewinded to frame {self.processor.tracker.currentFrame}. Press 'c' to continue or 'f' to fast forward.")    
                
            # Press 'm' to modify object ID
            elif key == ord('m') and paused:
                self.processor.tracker.modify_id(current_bboxes,frame)
                print("Video paused. Press 'c' to continue, 'b' to draw a bounding box, 'z' to edit an existing box, 'd' to delete a bounding box, 'v' to restore a bounding box, or 'm' to modify an object ID.")
           
            # Press 'b' to create a new boudning box
            elif key == ord('b') and paused:
                self.processor.tracker.createBox(frame)
                print("Video paused. Press 'c' to continue, 'b' to draw a bounding box, 'z' to edit an existing box, 'd' to delete a bounding box, 'v' to restore a bounding box, or 'm' to modify an object ID.")
           
           # Press 'p' to pause the video
            elif key == ord('z'):
               # self.processor.tracker.moveBox(frame)\
                self.processor.tracker.mouse_event_handler()
                print("Video paused. Press 'c' to continue, 'b' to draw a bounding box, 'z' to edit an existing box, 'd' to delete a bounding box, 'v' to restore a bounding box, or 'm' to modify an object ID.")

            # Press 'd' to remove a bounding box
            elif key == ord('d') and paused:
                self.processor.tracker.removeBox()
                print("Video paused. Press 'c' to continue, 'b' to draw a bounding box, 'z' to edit an existing box, 'd' to delete a bounding box, 'v' to restore a bounding box, or 'm' to modify an object ID.")
            
            # Press 'v' to restore a previously deleted bounding box
            elif key == ord('v') and paused:
                self.processor.tracker.restoreBox()
                print("Video paused. Press 'c' to continue, 'b' to draw a bounding box, 'z' to edit an existing box, 'd' to delete a bounding box, 'v' to restore a bounding box, or 'm' to modify an object ID.")
            
            # Press 'q' to exit the program
            elif key == ord('q'):
                break

        self.video_manager.release()
        
        # Create JSON file
        self.processor.tracker.frameData = dict(sorted(self.processor.tracker.frameData.items(), key=lambda item: int(item[0]))) # put list of frames into sequential order
        json_file = "tracked_objects.json"
        with open(json_file, "w") as f:
            json.dump(self.processor.tracker.frameData, f, indent=4)
        print(f"Tracked objects saved to {json_file}")
        
# Main function to run the program
if __name__ == "__main__":
    model_path = "yolov8n.pt"
    video_path = "video.mp4"
    output_video_path = "output_video.mp4"

    app = Application(model_path, video_path, output_video_path)
    app.run()
