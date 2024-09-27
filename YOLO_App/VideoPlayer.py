"""
YOLO Program Video Player

Initializes video playback, frame processing, and bounding box drawing

Authors: Luca Biccari, Michael Haydam, Luke Reinbach
Date: 19/09/2024
    
"""

from tkinter import*
import customtkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk
from PIL import Image as Img
import numpy as np
import cv2
import os
import copy
import yoatLogic as yolo

# inherit from CTKFrame to be used as CTK Widget
class VideoPlayer(tk.CTkFrame):

    def __init__(self, parent: tk.CTkFrame = None, **params: dict):
        """
        Initates VideoPlayer class, which inherits from CTkFrame 

        Args:
            parent (ttk.Frame, optional): Optional CTkFrame, and optional excess arguments. Defaults to None.
        """ 

        # init CTkFrame
        tk.CTkFrame.__init__(self, parent)

        self.__cap = None  # VideoCapture object from OpenCV
        self.__size = (1080, 720)  # Default video frame size
        self.__frames_numbers = 0  # Total frames in video
        self.__play = False  # Play status
        self.__frame = np.array  # Current frame
        self.__current_frame_number = 0
        
        self.model_path = "yolov8n.pt"
        self.videoManager = object
        self.processor = object
        self.tracker = object
        self.frame = np.array
        self.name = "json"
        
        # build widget
        self.build_ui_widget(parent)

        # Variables to store the start and end coordinates of the box
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        
        self.drawing = False
        self.drawing_complete_callback = None
        self.bbox = None
   
    @property
    def frame(self)->np.array:
        """
        Get the current frame of the video being processed.

        Returns:
            np.array: current video frame.
        """
        return self.__frame
    
    @property
    def currentFrameNumber(self)->int:
        """
        Get the current frame number being processed.
        
        Returns:
            int: current frame number.
        """
        return int(self.__current_frame_number)
    
    @frame.setter
    def frame(self, value: np.array):
        """
        Setter for the current frame of the video.

        Args:
            value (np.array): Change this frame to current frame
        """
        self.__frame = value

    def drawMode(self):
        """
        Activates drawing mode for creating a bounding box on the video frame using mouse events and mouse bindings.

        Returns:
            Bounding box coordinates [start_x, start_y, end_x, end_y] after user draws.
        """
        # Initialize bounding box coordinates
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        
        # Bind mouse events for drawing a bounding box
        self.display_panel.bind("<ButtonPress-1>", self.start_box)
        self.display_panel.bind("<B1-Motion>", self.update_box)
        self.display_panel.bind("<ButtonRelease-1>", self.complete_box)
        
        # Wait for drawing to be complete
        self.drawing = True
        self.bbox = None
        while self.drawing:
            self.update_idletasks()
            self.update()
        return self.bbox
        
    def exitDrawMode(self):
        """
        Exits the drawing mode by unbinding the mouse events used for drawing.
        """
        # Bind mouse events for drawing a bounding box
        self.display_panel.unbind("<ButtonPress-1>")
        self.display_panel.unbind("<B1-Motion>")
        self.display_panel.unbind("<ButtonRelease-1>")
        
    # Function to start drawing the box
    def start_box(self, event):
        """
        Handles the mouse press event to start drawing a bounding box.
        Initializes the starting coordinates of the bounding box.

        Args:
            event: mouse event that triggers this function.
        """
        if self.bbox is not None:
            self.display_panel.delete(self.bbox)
        self.start_x = event.x
        self.start_y = event.y
        # Create a rectangle on the canvas, we will modify it later
        self.rect_id = self.display_panel.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    # Function to update the rectangle as the user drags the mouse
    def update_box(self, event):
        """
        Handles the mouse drag event to dynamically update the bounding box as the user drags the mouse.

        Args:
            event: mouse movement that triggers this function.
        """
        current_x, current_y = (event.x, event.y)
        # Update the size of the rectangle based on the mouse drag
        self.display_panel.coords(self.rect_id, self.start_x, self.start_y, current_x, current_y)

    # Function to complete the drawing and store the coordinates
    def complete_box(self, event):
        """
        Handles the mouse release event to finish the bounding box.
        Stores the coordinates of the bounding box and exit drawing mode.

        Args:
            event: The mouse event that triggers this function.
        """
        # Final coordinates of the box
        self.end_x, self.end_y = (event.x, event.y)
        self.exitDrawMode()

    def create_box(self):
        """
        Converts the drawn bounding box coordinates from the display panel to the video frame's scale.
        Returns the bounding box coordinates so that they match the video.

        Returns:
            list: The bounding box coordinates scaled to the video frame dimensions.
        """
        scale_x = self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH) / self.display_panel.winfo_width()
        scale_y = self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / self.display_panel.winfo_height()
        self.bbox = [
            int(self.start_x * scale_x - 0), 
            int(self.start_y * scale_y), 
            int(self.end_x * scale_x - 0), 
            int(self.end_y * scale_y)
        ]
        print(f"Box drawn with coordinates: {self.bbox}")

        # Store the bounding box in the tracker (or wherever necessary)
        #obj_id = len(self.tracker.tracked_objects) + 1
       # self.tracker.manualBoxes[obj_id] = {'bbox': bbox, 'start_frame': self.__current_frame_number, 'end_frame': self.__current_frame_number + 10}
        #self.tracker.tracked_objects[obj_id] = {"class": "manual", "bbox": bbox}

        # Exit drawing mode
        self.drawing = False
        return self.bbox

    def display_frame(self, image: Img.Image):
        """
        Displays a video frame on the canvas after resizing it to fit the panel.

        Args:
            image (Img.Image): The image to display.
        """
        image.thumbnail(self.__size)
        self.photo = ImageTk.PhotoImage(image=image)
        self.display_panel.create_image(55, 0, image=self.photo, anchor=tk.NW)
        #? protect image from garbage collection?
        self.display_panel.image = self.photo

        self.display_panel.update()

    def load_video(self, filename):
        """
        Loads the video file and initializes the video manager and processor for video handling.

        Args:
            filename (str): name of the video file to load.
        """

        self.name = filename[:-4]
        self.output_path = self.name + "_YOAT_output" + ".mp4"
        self.model_menu.configure(state="disabled")

        if len(filename) != 0:
            self.__initialdir_movie = os.path.dirname(os.path.abspath(filename))
            self.videoManager = yolo.VideoManager(filename, self.output_path)
            self.processor = yolo.VideoProcessor(self.model_path, self.videoManager)
            self.tracker = self.processor.tracker
            self.play_video(filename)

        pass

    def play_video(self, filename: str):
        """
        Starts playing the video using OpenCV and sets up frame count and dimensions.

        Args:
            filename (str): The name of the video file to play.
        """
        self.__cap = cv2.VideoCapture(filename)
        self.__frames_numbers = int(self.__cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__image_ratio = self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        #TODO add in prgress bar stuff
        self.progress_slider.configure(number_of_steps=self.__frames_numbers, to=self.__frames_numbers)

        self.__play = True

        self.run_frames()

    def play_pause(self, setter:bool = None):
        """
        Toggles the play/pause state of the video.
        
        Args:
            setter (bool)
        """
        if self.__cap.isOpened():
            if setter == None:
                self.__play = not self.__play
            else:
                self.__play = setter

        if self.__play:
            self.run_frames()
            self.play_pauseButton.configure(image=self.pause_icon)
            print("Playing")
        elif not self.__play:
            self.play_pauseButton.configure(image=self.play_icon)
            print("Paused")

    def seek(self, seek_frame):
        """
        Jump to specific frame in the video.

        Args:
            seek_frame (int): The frame number to jump to in the video.
        """
        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)
        self.__current_frame_number = seek_frame

    def update_slider(self, val):
        """
        Update the position of the progress slider based on the current frame.

        Args:
            val (int): The value to set on the slider.
        """
        self.progress_slider.set(val)

    def skip(self, offset):
        """
        Skip forward or backward by a number of frames.

        Args:
            offset (int): The number of frames to skip. Positive for forward, negative for backward.
        """
        self.__current_frame_number = self.__current_frame_number + offset - 1
        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, self.__current_frame_number)

        # Display the new frame if paused
        if not self.__play:
            ret, self.frame = self.__cap.read()
            self.process_and_display(ret)
    
    def set_model(self, version):
        """
        Set object detection model
        """
        print("optionmenu dropdown clicked:", version)
        self.model_path = version

    def run_frames(self):
        """
        Play video frames in a loop if video is in play mode.
        
        This function reads the next frame and displays it every 30 ms.
        """
        if self.__play:
            # update the frame number
            ret, self.frame = self.__cap.read()
            # self.frame = image_matrix
            self.process_and_display(ret)
            self.after(30, self.run_frames)

    def process_and_display(self, ret):
        """
        Process and display a single frame, update trackers and the UI.

        Args:
            ret (bool): Whether a frame was successfully retrieved or end of video.
        """
        if ret:
            self.__current_frame_number = int(self.__cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.update_slider(self.__current_frame_number)
            self.tracker.currentFrame = self.__current_frame_number
            self.frame, self.current_bboxes = self.processor.process_frame(self.frame)
            self.videoManager.write_frame(self.frame)

            # convert matrix image to pillow image object
            self.frame = self.matrix_to_pillow(self.frame)
            self.display_frame(self.frame)
        else:
            self.__cap.release()
            cv2.destroyAllWindows()
    
    @staticmethod
    def matrix_to_pillow(frame: np.array):
        """
        Convert an OpenCV image to a Pillow image.

        Args:
            frame (np.array): The OpenCV image.

        Returns:
            PIL.Image: The Pillow image object.
        """
        # convert to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # convert matrix image to pillow image object
        frame_pillow = Img.fromarray(frame_bgr)
        return frame_pillow

    def on_closing(self):
        """
        Handles cleanup and closing of the video and the UI window.

        Releases video resources and destroys the main window.
        """
        # Cleanup code
        if self.__cap is not None:
            self.__cap.release()
        cv2.destroyAllWindows()
        self.master.destroy()
    
    def build_ui_widget(self, parent: tk.CTkFrame=None):
        """
        Build the main UI components such as play/pause, skip buttons, and video display.

        Args:
            parent (tk.CTkFrame): The parent frame for the UI. If None it creates a new main window.
        """

        primary_colour = "#383838"
        secondary_colour = "#9B9B9B"

        # If no parent frame is provided, create a new frame as the main panel
        if parent is None:
            self.master.geometry("1080x720+0+0")  # Set the window size and position
            self.main_panel = tk.CTkFrame(self.master) # Create a new frame
            self.main_panel.pack(expand=True, fill="both")  # Place it in the window
        else:
            self.main_panel = parent # Else Use the provided parent frame
        
        # Configure the main panel's background color
        self.main_panel.configure(bg_color="black")

        button_width = 45
        button_height = 50

        self.lower_panel = tk.CTkFrame(self.main_panel, width=200)
        self.lower_panel.pack(fill="y", side=BOTTOM)
        self.lower_panel.grid_columnconfigure((0,1,2,3,4), weight=1)

        self.display_panel = tk.CTkCanvas(self.main_panel)
        self.display_panel.pack(fill=BOTH, expand=True)

        # Get the absolute path to the "Icons" folder located in the parent directory.
        # os.pardir refers to the parent directory (".."), and 'Icons' is the folder name.
        # os.path.join() creates a relative path "../Icons", and os.path.abspath() converts it into an absolute path.
        # icons_path = os.path.abspath(os.path.join(os.pardir, 'VideoButtons'))

        self.icons_path = "VideoButtons"

        # Load the image using CTkImage instead of PhotoImage
        self.pause_icon = tk.CTkImage(Img.open(os.path.join(self.icons_path, "Pause.png")), size=(40, 40))
        self.play_icon = tk.CTkImage(Img.open(os.path.join(self.icons_path, "Play.png")), size=(40, 40))
        self.back_icon = tk.CTkImage(Img.open(os.path.join(self.icons_path, "Backward.png")), size=(40, 40))
        self.forward_icon = tk.CTkImage(Img.open(os.path.join(self.icons_path, "Forward.png")), size=(40, 40))
        self.frame_forward_icon = tk.CTkImage(Img.open(os.path.join(self.icons_path, "one_frame_forward.png")), size=(40, 40))
        self.frame_back_icon = tk.CTkImage(Img.open(os.path.join(self.icons_path, "one_frame_back.png")), size=(40, 40))


        self.progress_value = tk.IntVar(self.main_panel)
        self.progress_slider = tk.CTkSlider(self.main_panel, from_=0, to=100, command=self.seek)
        self.progress_slider.set(0)
        self.progress_slider.pack(side="left", fill="x", padx=10, pady=5,  expand=True)


        # Create button for playing/pausing the video
        self.play_pauseButton = tk.CTkButton(self.lower_panel, image= self.pause_icon, text="", fg_color=secondary_colour, 
                                             width=20, height=20, command=lambda:self.play_pause())
        self.play_pauseButton.grid(row=0, column=2, padx=10, pady=20)

        # Create button for skipping 60 frames backward
        self.backButton = tk.CTkButton(self.lower_panel, image=self.back_icon, text="", fg_color=secondary_colour,
                                       border_width=0, height=20, width=20, command=lambda: self.skip(-60))
        self.backButton.grid(row=0, column=0, padx=10, pady=20)

        # Create button for skipping 1 frames backward
        self.frame_backButton = tk.CTkButton(self.lower_panel, image=self.frame_back_icon, text="", fg_color=secondary_colour,
                                       border_width=0, height=20, width=20, command=lambda: self.skip(-1))
        self.frame_backButton.grid(row=0, column=1, padx=10, pady=20)

        # Create button for skipping 60 frames forward
        self.forwardButton = tk.CTkButton(self.lower_panel, image=self.forward_icon, text="", fg_color=secondary_colour,
                                       border_width=0, height=20, width=20, command=lambda: self.skip(60))
        self.forwardButton.grid(row=0, column=4, padx=10, pady=20)

        # Create button for skipping 1 frame forward
        self.frame_forwardButton = tk.CTkButton(self.lower_panel, image=self.frame_forward_icon, text="", fg_color=secondary_colour,
                                       border_width=0, height=20, width=20, command=lambda: self.skip(1))
        self.frame_forwardButton.grid(row=0, column=3, padx=10, pady=20)

        self.model_menu = tk.CTkOptionMenu(self.lower_panel, values=["yolov8n.pt", "yolov10n.pt"],
                                         command=self.set_model)
        self.model_menu.grid(row=0, column=5, padx=10, pady=20)
        self.model_menu.set("yolov8n.pt")


def main():
    """
    Run the application
    """
    vid = VideoPlayer()
    #! something
    vid.mainloop()

if __name__ == "__main__":
    main()
