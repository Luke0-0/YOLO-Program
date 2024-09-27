"""
Application that runs YOLO model

This program creates the application UI that hosts the YOLO object detection model. 
It allows users to add edits to the annotations made by YOLO and export the annotations in a JSON file.

Authors: Luke Reinbach (RNBLUK001), Michael Haydam (HYDMIC009), Gianluca Biccari (BCCGIA001)
Version: September 19, 2024
"""

from tkinter import *
import customtkinter as tk
from tkinter import filedialog
from VideoPlayer import VideoPlayer
import json

# Set up customtkinter appearance and theme
tk.set_appearance_mode("dark")
tk.set_default_color_theme("dark-blue")

primary_colour = "#383838"
secondary_colour = "#9B9B9B"
error_colour = "#800000"
custom_classes = []
filter_list = []

# Create root window with an initial size
# Lock window size to aim performance
root = tk.CTk()
root.title("YOAT Video Player")
root.resizable(width=False, height=False)

# Set initial dimensions for the window
# Set window position when it is created
initial_width = 1380
initial_height = 720
root.update_idletasks() 
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root_x = (screen_width/2) - (initial_width/2)
root_y = (screen_height/2) - (initial_height/2)
root.geometry(f"{initial_width}x{initial_height}+{root_x}+{root_y}")

# Variables for video dimensions and aspect ratio
video_width = initial_width
video_height = initial_height
aspect_ratio = video_width / video_height

# Make frame to hold option inputs
right_frame = tk.CTkFrame(master=root, width=300, bg_color=secondary_colour, fg_color=secondary_colour)
right_frame.pack(fill="y", side=RIGHT)

# Make frame that will host the VideoPlayer
video_frame = tk.CTkFrame(master=root)
video_frame.pack(expand=True, fill="both")

# Initialize the video player from the VideoPlayer class
video_player = VideoPlayer(video_frame)

def export():
    """
    Exports a json file containing information on the objects present
    in each frame. 

    Args:
        None
    """ 
    video_player.tracker.frameData = dict(sorted(video_player.tracker.frameData.items(), 
                                    key=lambda item: int(item[0]))) # put list of frames into sequential order
    json_file = video_player.name + "_tracked_objects.json"
    with open(json_file, "w") as f:
        json.dump(video_player.tracker.frameData, f, indent=4)
    print(f"Tracked objects saved to {json_file}")

    # Create label in UI to tell user the JSON file has been exported
    export_label = tk.CTkLabel(right_frame, text="JSON File Exported!", font=("04b03", 14), 
                           text_color=primary_colour, anchor="w", justify="center", fg_color="transparent")
    export_label.grid(row=16, columnspan=2, padx=(25,20), pady=(5,10), sticky="w")

def load_coco_classes():
    """
    Load the coco.txt file into an array. This file contains a list of all YOLO identifiable objects 

    Args:
        None
    """ 
    coco_classes = []
    try:
        with open('coco.txt', 'r') as file:
            coco_classes = [line.strip() for line in file]
    except FileNotFoundError:
        print("Warning: coco.txt file not found. Class filtering may not work correctly.")
    return coco_classes

coco_list = load_coco_classes()

def delete_items():
    """
    Remove items from the filter list, and in turn from showing in the frame
    Args:
        None
    """ 
    global filter_list
    selection = filter_listbox.curselection() # get items selected/highlighted in list

    # Iterate through the selected items in reverse order.
    # Remove item from list in UI and reset filter to exclude item
    for item in selection[::-1]:
        filter_list.pop(item) 
        filter_listbox.delete(item)
    video_player.processor.set_filtered_classes(filter_list) # set filter for video processor

def select_items():
    """
    Remove items from the filter list, and in turn from showing in the frame
    Args:
        None
    """ 
    global filter_list
    selection = filter_listbox.curselection() # get items selected/highlighted in list

    # wipe list
    filter_listbox.delete(0, END)
    selection_list = []

    # add all selected items to new list and repopulate selection
    # filter classes based on new list
    for i in selection: 
        item = filter_list[i]
        selection_list.append(item)
        print("item: "+item)
        filter_listbox.insert(END, item)
    video_player.processor.set_filtered_classes(selection_list) # set filter for video processor

def reset_items():
    """
    Remove items from the filter list, and in turn from showing in the frame
    Args:
        None
    """ 

    global coco_list, custom_classes, filter_list
    filter_list = []

    filter_listbox.delete(0, END) # wipe list
    filter_list = coco_list + custom_classes # make new list combining coco and custom classes

    # iterate through refreshed filter list and add to selection
    for item in filter_list:
        filter_listbox.insert(END, item)
    video_player.processor.set_filtered_classes(filter_list)

def update_conf(val):
    """
    Update confidence threshhold by setting it to slider value. This function is called by the slider when adjusted.
    
    Args:
        val = int
    """
    confidence_label.configure(text=f"Confidence: {int(val)}%") # display new confidence threshold
    video_player.processor.set_confidence_threshold(val) # set confidence threshold

def load_video():
    """
    Trigger the video player to load in a video selected in file manager. 
    This is prompted by the select file button.
    
    Args:
        None
    """
    file_path = filedialog.askopenfilename()
    video_player.load_video(file_path) # load video into video player
    confidence_slider.set(25) # set confidence to 25% by default
    confidence_label.configure(text="Confidence: 25%") # display confidence
    confidence_slider.configure(state="normal") # set confidence to 25% by default

    # Once the file has been loaded in all the user to export JSON file
    # Display the export button
    export_button = tk.CTkButton(right_frame ,text="Export JSON", bg_color=secondary_colour, 
                                  fg_color=primary_colour, font=("04b03", 12, "bold"), command=export)
    export_button.grid(row=15, columnspan=2, padx=20, pady=(0,5))

def removeBox(remove_popup, entry: tk.CTkEntry):
    """
    Get and send box ID to the video player tracker to remove it from the displaying boxes.
    This information is retrieved from the remove popup when enter is clicked.
    
    Args:
        remove_popup: Popup custom object that collects user input
        entry: tk.CTkEntry that is hosted in the Popup
    """

    box_id = int(entry.get())
    video_player.tracker.deletedBoxes.append(box_id) # add box_id to deleted boxes
    video_player.skip(1) # play the next frame to show the deletion
    remove_popup.destroy() # close Popup  

def removeBox_popup():
    """
    Create a Popup window that gets user info required to delete a box.
    
    Args:
        None: This popup is used to capture information.
    """

    video_player.play_pause(False) # Pause the video
    remove_popup = Popup(root, "Box Details", 300, 180)
    remove_popup.columnconfigure(0, weight=2)

    remove_label = tk.CTkLabel(remove_popup, text="Please enter box details:", font=("04b03", 16), 
                               text_color=primary_colour, anchor="w", justify = "center", fg_color="transparent")
    remove_label.grid(row=0, column=0, padx=0, pady=20, sticky = "")

    rid_label = tk.CTkLabel(remove_popup, text="Box ID:", font=("04b03", 16), text_color=primary_colour, 
                            anchor="w", justify = "left", fg_color="transparent")
    rid_label.grid(row=1, column=0, padx=10, pady=0, sticky = "W")

    rid_entry = tk.CTkEntry(remove_popup, width = 250, placeholder_text="Eg: 36")
    rid_entry.grid(row=1, column=0, padx=100, pady=0, sticky = "EW")

    rid_enter = tk.CTkButton(remove_popup, text="Enter", bg_color=secondary_colour, fg_color=primary_colour, 
                             font=("04b03", 12, "bold"), command=lambda:removeBox(remove_popup, rid_entry))
    rid_enter.grid(row=2, column=0,  padx=20, pady=20)

def addBox(add_popup, box_class, numFrames):
    """
    Triggered by addBox_popup to get video player to create a new box. Creates a new ID
    and adds the class to filter list if new class.
    
    Args:
        add_popup: This popup is used to capture information
        box_class: The class of the object entered by the user
        numFrames: The amount of frames the user wants the box to stay for
    """
    video_player.create_box() # call creat box in video player to get cv2 to draw box
    obj_id = len(video_player.tracker.tracked_objects) + 1 # create new id

    # create new class if class does not match existing list
    if box_class not in filter_list:
        custom_classes.append(box_class)
        filter_list.append(box_class)
        reset_items() # refresh list if new class added
    currentFrame = int(video_player.currentFrameNumber)

    # add to video player manual boxes
    video_player.tracker.manualBoxes[obj_id] = {'bbox': video_player.bbox, 'class': box_class, 
                                                'start_frame': currentFrame, 'end_frame': currentFrame + int(numFrames)}
    # add to video player tracker
    video_player.tracker.tracked_objects[obj_id] = {"class": box_class, "bbox": video_player.bbox}
    add_popup.destroy() # close popup

def addBox_popup():
    """
    Create a Popup window that gets user info required to add a box. 
    This also triggers the canvas to let the user draw.
    
    Args:
        None: This popup is used to capture information.
    """
    video_player.play_pause(False)
    add_popup = Popup(root, "Box Details", 300, 200)

    add_label = tk.CTkLabel(add_popup, text="Please enter box details:", font=("04b03", 16), 
                            text_color=primary_colour, anchor="w", justify = "center", fg_color="transparent")
    add_label.grid(row=0, columnspan=2, padx=(45,0), pady=20, sticky = "")

    acl_label = tk.CTkLabel(add_popup, text="Box Class:", font=("04b03", 16), text_color=primary_colour, 
                            anchor="w", justify = "left", fg_color="transparent")
    acl_label.grid(row=1, column=0, padx=10, pady=5, sticky = "W")

    acl_entry = tk.CTkEntry(add_popup, width = 100, placeholder_text="Eg: Person")
    acl_entry.grid(row=1, column=1, padx=0, pady=5, sticky = "EW")

    afr_label = tk.CTkLabel(add_popup, text="Num Frames:", font=("04b03", 16), text_color=primary_colour, 
                            anchor="w", justify = "left", fg_color="transparent")
    afr_label.grid(row=2, column=0, padx=10, pady=5, sticky = "W")

    afr_entry = tk.CTkEntry(add_popup, width = 100, placeholder_text="Eg: 36")
    afr_entry.grid(row=2, column=1, padx=0, pady=5, sticky = "EW")

    add_enter = tk.CTkButton(add_popup, text="Enter", bg_color=secondary_colour, 
                             fg_color=primary_colour, font=("04b03", 12, "bold"), command=lambda:addBox(add_popup, acl_entry.get(), afr_entry.get()))
    add_enter.grid(row=3, columnspan=2,  padx=(45,0), pady=20)

    video_player.drawMode() # turn draw mode on so user can draw boxes

def editBox(edit_popup, old_id, new_id, toAll):
    """
    Triggered by editBox_popup to get yolo code to edit box information. 
    This allows the user to change the id's of boxes.
    
    Args:
        edit_popup: This popup is used to capture information
        old_id: The ID to be replaced
        new_id: The ID chosen by the user
        toAll: bool that allows user to apply change to all occurences of that box/id
    """   
    old_id = int(old_id)
    new_id = int(new_id)
    if old_id in video_player.tracker.tracked_objects:
        video_player.tracker.tracked_objects[new_id] = video_player.tracker.tracked_objects.pop(old_id)
        video_player.tracker.id_mapping[old_id] = new_id
        print(f"Changed object ID {old_id} to {new_id}")
        if int(toAll) == 1:
            video_player.tracker.id_mapping[old_id] = new_id
            print(f"Future instances of object {old_id} will also be changed to {new_id}")
        if old_id in video_player.current_bboxes:
            video_player.current_bboxes[new_id] = video_player.current_bboxes.pop(old_id)
    else:
        print(f"Object ID {old_id} not found")
    edit_popup.destroy() # close popup

def editBox_popup():
    """
    Create a Popup window that gets user info required to edit a box. 
    
    Args:
        None: This popup is used to capture information.
    """
    video_player.play_pause(False)
    edit_popup = Popup(root, "Box Details", 300, 220)
    edit_popup.columnconfigure(0, weight=2)

    edit_label = tk.CTkLabel(edit_popup, text="Please enter box details:", font=("04b03", 16), 
                             text_color=primary_colour, anchor="w", justify = "center", fg_color="transparent")
    edit_label.grid(row=0, column=0, padx=0, pady=20, sticky = "")

    aid_label = tk.CTkLabel(edit_popup, text="Old ID:", font=("04b03", 16), text_color=primary_colour, 
                            anchor="w", justify = "left", fg_color="transparent")
    aid_label.grid(row=1, column=0, padx=10, pady=(0,10), sticky = W)

    oid_entry = tk.CTkEntry(edit_popup, width = 250, placeholder_text="Eg: 36")
    oid_entry.grid(row=1, column=0, padx=100, pady=(0,10), sticky = EW)

    nid_label = tk.CTkLabel(edit_popup, text="New ID:", font=("04b03", 16), text_color=primary_colour, 
                            anchor="w", justify = "left", fg_color="transparent")
    nid_label.grid(row=2, column=0, padx=10, pady=0, sticky = W)

    nid_entry = tk.CTkEntry(edit_popup, width = 250, placeholder_text="Eg: 23")
    nid_entry.grid(row=2, column=0, padx=100, pady=0, sticky = EW)

    toAll_checkbox = tk.CTkCheckBox(edit_popup, text="Apply globally", font=("04b03", 15), checkbox_width=18, 
                                    checkbox_height=18, text_color=primary_colour, border_color=primary_colour)
    toAll_checkbox.grid(row=3, column=0, padx=(15,0), pady=(10,0), sticky = W)

    edit_enter = tk.CTkButton(edit_popup, text="Enter", bg_color=secondary_colour, fg_color=primary_colour, 
                              font=("04b03", 12, "bold"), command=lambda:editBox(edit_popup, oid_entry.get(), nid_entry.get(), toAll_checkbox.get()))
    edit_enter.grid(row=4, column=0,  padx=20, pady=(10,0))

def on_close():
    """
    Function to handle the closing of the application safely
    
    Args:
        None
    """
    video_player.on_closing() # run video player closing fucntion
    root.destroy() # close root

# Bind the window close event to on_close function
root.protocol("WM_DELETE_WINDOW", on_close)

# Create button for loading a video file
load_button = tk.CTkButton(right_frame, text="Select File", bg_color=secondary_colour, 
                           fg_color=primary_colour, font=("04b03", 12, "bold"), command=lambda: load_video())
load_button.grid(row=0, columnspan=2, padx=20, pady=20)

# HEADING
edits_label = tk.CTkLabel(right_frame, text="Annotation Edits", font=("04b03", 18), 
                          text_color=primary_colour, anchor="e", fg_color="transparent")
edits_label.grid(row=1, columnspan=2, padx=0, pady=(0,10))

# Create button for loading adding boxes
add_button = tk.CTkButton(right_frame, text="Add box (A)", bg_color=secondary_colour, 
                          fg_color=primary_colour, font=("04b03", 12, "bold"), command=lambda:addBox_popup())
add_button.grid(row=2, columnspan=2,  padx=20, pady=(0,5))

# Create button for removing boxes
remove_button = tk.CTkButton(right_frame, text="Remove box (R)", bg_color=secondary_colour, 
                             fg_color=primary_colour, font=("04b03", 12, "bold"), command=lambda:removeBox_popup())
remove_button.grid(row=3, columnspan=2,  padx=20, pady=5)

# Create button for editing boxes
edit_button = tk.CTkButton(right_frame, text="Edit box (E)", bg_color=secondary_colour, 
                           fg_color=primary_colour, font=("04b03", 12, "bold"), command=lambda:editBox_popup())
edit_button.grid(row=4, columnspan=2,  padx=20, pady=5)

# HEADING
filter_label = tk.CTkLabel(right_frame, text="Filter Settings", font=("04b03", 18), 
                           text_color=primary_colour, anchor="e", fg_color="transparent")
filter_label.grid(row=5, columnspan=2, padx=0, pady=(20,10))

# List all classes
filter_listbox = Listbox(right_frame, width=15, bg=primary_colour, font=("04b03", 12, "bold"), 
                         selectmode="multiple")
filter_listbox .grid(row=6, columnspan=2, padx=2, pady=(0,5))

# Button to delete selected classes
delete_items_button = tk.CTkButton(right_frame, width=20,text="Delete", bg_color=secondary_colour, 
                                  fg_color=primary_colour, font=("04b03", 12, "bold"), command=delete_items)
delete_items_button.grid(row=7, column=0, padx=1, pady=(5,3), ipadx=8, sticky="e")

# Button to keep only selected classes
select_items_button = tk.CTkButton(right_frame, width=20,text="Select", bg_color=secondary_colour, 
                                  fg_color=primary_colour, font=("04b03", 12, "bold"), command=select_items)
select_items_button.grid(row=7, column=1, padx=1, pady=(5,3), ipadx=8, sticky="w")

# Button to reset selected classes
reset_items_button = tk.CTkButton(right_frame ,text="Reset Items", bg_color=secondary_colour, 
                                  fg_color=primary_colour, font=("04b03", 12, "bold"), command=reset_items)
reset_items_button.grid(row=8, columnspan=2, padx=20, pady=(0,5))

# Label to display confidence threshold
confidence_label = tk.CTkLabel(right_frame, text="Confidence: 25%", font=("04b03", 14), 
                           text_color=primary_colour, anchor="w", justify="left", fg_color="transparent")
confidence_label.grid(row=10, columnspan=2, padx=(40,0), pady=(5,10), sticky="w")

# Slider to set threshold between 0 and 100
confidence_slider = tk.CTkSlider(right_frame, width=150, from_=0, to=100, state="disabled" ,command=update_conf)
confidence_slider.grid(row=9, columnspan=2, padx=20, pady=(10,5))

# Set confidence to 25% by default
confidence_slider.set(25)

# populate filter list
filter_list = []
filter_listbox.delete(0, END)
filter_list = coco_list + custom_classes
for item in filter_list:
    filter_listbox.insert(END, item)

class Popup(tk.CTkToplevel):
    """
    A class that is used to make and display multiple popup windows that get user input.
    """
    def __init__(self, root:tk.CTk, title = "Popup", width = 300, height = 300):

        self.__root = root
        self.__width = width
        self.__height = height
        self.__x = 0
        self.__y = 0
        self.__title = title

        # Get the root window's current position and size
        self.__root.update_idletasks()  # Ensures we get the correct dimensions of the main window
        self.window_width = root.winfo_width()
        self.window_height = root.winfo_height()
        self.window_x = root.winfo_x()
        self.window_y = root.winfo_y()

        # Calculate position for popup in the center of the main window
        self.__x = self.window_x + (self.window_width // 2) - (self.__width // 2) - 80
        self.__y = self.window_y + (self.window_height // 2) - (self.__height // 2) - 20

        # Set the popup size and position
        tk.CTkToplevel.__init__(self, fg_color=secondary_colour)
        self.title(self.__title)
        self.geometry(f'{self.__width}x{self.__height}+{self.__x}+{self.__y}')
        self.resizable(width=False, height=False)
        self.attributes("-topmost", True)

# bind keys to buttons for usability
root.bind('<space>', lambda event: video_player.play_pause())
root.bind('<Right>', lambda event: video_player.skip(1))
root.bind('<Left>', lambda event: video_player.skip(-1))
root.bind('<Up>', lambda event: video_player.skip(60))
root.bind('<Down>', lambda event: video_player.skip(-60))
root.bind('<r>', lambda event: removeBox_popup())
root.bind('<a>', lambda event: addBox_popup())
root.bind('<e>', lambda event: editBox_popup())

# Start the Tkinter main loop
root.mainloop()
