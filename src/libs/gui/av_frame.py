import customtkinter as ctk
import sys 
import threading

sys.path.append('./')
from src.libs.recording.recorder_video import VideoRecorder
from src.libs.recording.recorder_audio import AudioRecorder
from src.libs.detection.loudness import Loudness
from PIL import Image

class AudioVideoFrame(ctk.CTkFrame):
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
        self.video_capture = None

        video_label = ctk.CTkLabel(self, text="Recording Video")
        video_label.pack(pady=20, padx=10)

        self.icon = ctk.CTkImage(
            light_image=Image.open("src/data/images/static/video-conference.png"),
            dark_image=Image.open("src/data/images/static/video-conference.png"),
            size=(200, 200)
            )
        
        self.video_label = ctk.CTkLabel(self, image=self.icon, text="")
        self.video_label.pack(pady=15, padx=10)
        
        ### Audio ###
        self.audio_loudness_label = ctk.CTkLabel(
            self,
            text="No Volume Recorded", 
            font=("Helvetica", 16), 
            text_color="yellow")
        self.audio_loudness_label.pack(pady=15, padx=10)
        
        self.audio_rcs_label = ctk.CTkLabel(self, text="Loudness: 0")
        self.audio_rcs_label.pack(pady=15, padx=10)
        #############
        
        ### Video ###
        self.video_button_start = ctk.CTkButton(self, text="Start", command=self.video_start)
        self.video_button_start.pack(side=ctk.LEFT, pady=12, padx=10)
        self.video_button_start.place(relx=0.35, rely=0.87, anchor='center')
        
        self.video_button_stop = ctk.CTkButton(self, text="Stop", command=self.video_stop)
        self.video_button_stop.pack(side=ctk.LEFT, pady=12, padx=10)
        self.video_button_stop.place(relx=0.65, rely=0.87, anchor='center')
        self.video_button_stop.configure(state=ctk.DISABLED)
        
        self.video_button = ctk.CTkButton(self, text="Go Back to Main Page", command=self.video_go_back)
        self.video_button.pack(side=ctk.BOTTOM, pady=12, padx=10)
        #self.video_button.place(relx=0.5, rely=0.90, anchor='center')
        #############
    
    def video_start(self):
        """
        Function called when the "Start" button is pressed
        """
        self.video_button_start.configure(state=ctk.DISABLED)
        self.video_button_stop.configure(state=ctk.NORMAL)
        
        self.video_capture = VideoRecorder()
        self.update_video()
        
        audio_thread = threading.Thread(target=self.record_audio)
        audio_thread.start()
    
    def video_stop(self):
        """
        Function called when the "Stop" button is pressed
        """
        self.video_button_start.configure(state=ctk.NORMAL)
        self.video_button_stop.configure(state=ctk.DISABLED)
        self.video_label.configure(image=self.icon)
        self.video_label.image = self.icon
        
        if self.video_capture is not None:
            self.video_capture.stop()
        self.video_capture = None
        
        if self.audio_capture is not None:
            self.audio_capture.stop()
        self.audio_capture = None
        
        
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
        
    
    def record_audio(self):
        self.audio_capture = AudioRecorder()
        loudness = Loudness()
        
        self.audio_capture.stream.start_stream()
        self.audio_loudness_label.configure(
            text="Normal Volume", 
            font=("Helvetica", 16), 
            text_color="green")
        
        while(self.audio_capture.open):
            try:
                data = self.audio_capture.stream.read(
                    self.audio_capture.frames_per_buffer
                ) 
                self.audio_capture.audio_frames.append(data)
                # Compute the loudness of the audio and display it if it is 
                # greater than 0.5
                rcs = loudness.compute_loudness(data)
                self.audio_rcs_label.configure(text="RCS: " + str(rcs))
                if rcs > 0.5:
                    self.audio_loudness_label.configure(
                        text="  Attention!!! Loud Noise:  " + str(rcs), 
                        text_color="red")
                
                if self.video_capture is None:
                    break
                if self.audio_capture.open==False:
                    break
            except  Exception as e:
                print(e)
                break
            
    
    def video_go_back(self):
        """
        Function called when the "Go Back to Main Page" button is pressed
        """
        if self.video_capture is not None:
            self.stop()
        self.controller.show_frame("MainFrame")