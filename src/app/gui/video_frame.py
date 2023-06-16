import customtkinter as ctk
import sys 

sys.path.append('./')
from src.app.recording.recorder_video import VideoRecorder
from src.app.utils.config import Path
from PIL import Image

class VideoFrame(ctk.CTkFrame):
    """Second Page of the GUI"""
    def __init__(self, parent, controller):
        """
        Initialize the second page
        
        Args:
            parent (tk.Frame): Parent frame
            controller (App): Controller class for the GUI
        """
        ctk.CTkFrame.__init__(self, parent)
        self.controller = controller

        video_label = ctk.CTkLabel(self, text=Path.TEXT_GUI_VIDEO_0.value)
        video_label.pack(pady=20, padx=10)

        self.icon = ctk.CTkImage(
            light_image=Image.open(Path.IMAGE_GUI_VIDEO.value),
            dark_image=Image.open(Path.IMAGE_GUI_VIDEO.value),
            size=(200, 200)
            )
        
        self.video_label = ctk.CTkLabel(self, image=self.icon, text="")
        self.video_label.pack(pady=15, padx=10)

        self.video_capture = None

        self.video_button_start = ctk.CTkButton(
            self, 
            text=Path.TEXT_GUI_1.value, 
            command=self.video_start)
        self.video_button_start.pack(side=ctk.LEFT, pady=12, padx=10)
        self.video_button_start.place(relx=0.35, rely=0.87, anchor='center')
        
        self.video_button_stop = ctk.CTkButton(
            self, 
            text=Path.TEXT_GUI_2.value, 
            command=self.video_stop)
        self.video_button_stop.pack(side=ctk.LEFT, pady=12, padx=10)
        self.video_button_stop.place(relx=0.65, rely=0.87, anchor='center')
        self.video_button_stop.configure(state=ctk.DISABLED)
        
        self.video_button = ctk.CTkButton(
            self, 
            text=Path.TEXT_GUI_3.value, 
            command=self.video_go_back)
        self.video_button.pack(side=ctk.BOTTOM, pady=12, padx=10)
        # self.video_button.place(relx=0.5, rely=0.90, anchor='center')
        
    
    def video_start(self):
        """
        Function called when the "Start" button is pressed
        """
        self.video_button_start.configure(state=ctk.DISABLED)
        self.video_button_stop.configure(state=ctk.NORMAL)
        self.video_capture = VideoRecorder()
        self.update_video()
        
    
    def video_stop(self):
        """
        Function called when the "Stop" button is pressed
        """
        self.video_button_start.configure(state=ctk.NORMAL)
        self.video_button_stop.configure(state=ctk.DISABLED)
        self.video_capture.stop()
        self.video_capture = None
        self.video_label.configure(image=self.icon)
        self.video_label.image = self.icon
        
        
    def update_video(self):
        """
        Function called to update the video frame
        """
        if self.video_capture is None:
            return
        ret, frame = self.video_capture.get_frame()
        if frame is not None:
            image = Image.fromarray(frame)
            image = image.resize((400, 300))
            # photo = ImageTk.PhotoImage(image)
            photo = ctk.CTkImage(image, size=(400, 300))
            self.video_label.configure(image=photo)
            self.video_label.image = photo
        self.after(15, self.update_video)
    
    def video_go_back(self):
        """
        Function called when the "Go Back to Main Page" button is pressed
        """
        if self.video_capture is not None:
            self.stop()
        self.controller.show_frame("MainFrame")