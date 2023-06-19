import customtkinter as ctk
import sys 
import threading

sys.path.append('./')
from src.app.recording.recorder_video import VideoRecorder
from src.app.recording.recorder_audio import AudioRecorder
from src.app.detection.loudness import Loudness
from src.app.utils.config import Path
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
        self.parent = parent   
        self.controller = controller
        self.video_capture = None
        

        video_label = ctk.CTkLabel(self, text=Path.TEXT_GUI_AV_1.value)
        video_label.pack(pady=20, padx=10)

        self.icon = ctk.CTkImage(
            light_image=Image.open(Path.IMAGE_GUI_VIDEO.value),
            dark_image=Image.open(Path.IMAGE_GUI_VIDEO.value),
            size=(200, 200)
            )
        
        self.video_label = ctk.CTkLabel(self, image=self.icon, text="")
        self.video_label.pack(pady=15, padx=10)
        
        ### Audio ###
        self.audio_loudness_label = ctk.CTkLabel(
            self,
            text=Path.TEXT_GUI_AUDIO_1.value, 
            font=("Helvetica", 16), 
            text_color="yellow")
        self.audio_loudness_label.pack(pady=15, padx=10)
        
        self.audio_rcs_label = ctk.CTkLabel(
            self, 
            text=Path.TEXT_GUI_AUDIO_2.value + str(0))
        self.audio_rcs_label.pack(pady=15, padx=10)
        #############
        
        ### Video ###
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
        ret, frame, blink = self.video_capture.get_frame()
        if frame is not None:
            image = Image.fromarray(frame)
            image = image.resize((400, 300))
            # photo = ImageTk.PhotoImage(image)
            photo = ctk.CTkImage(image, size=(400, 300))
            self.video_label.configure(image=photo)
            self.video_label.image = photo
            
            # if the alarm is activated, flash the background color
            if blink:
                threading.Thread(target=self.change_frame_color).start()
                # self.change_frame_color()
                
        self.after(15, self.update_video)
        
    
    def record_audio(self):
        self.audio_capture = AudioRecorder()
        loudness = Loudness()
        
        self.audio_capture.stream.start_stream()
        self.audio_loudness_label.configure(
            text=Path.TEXT_GUI_AUDIO_3.value,
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
                self.audio_rcs_label.configure(
                    text=Path.TEXT_GUI_AUDIO_4.value + str(rcs))
                if rcs > 0.5:
                    self.audio_loudness_label.configure(
                        text=Path.TEXT_GUI_AUDIO_5.value + str(rcs), 
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
    
    def change_frame_color(self, i=0):
        """
        Flashing the background color of the frame to indicate the activation
        of the alarm.
        """
        # Counter to stop the flashing after 10 iterations
        if i < 20:
            # default is 'gray17'
            current_color = self.cget("fg_color")
            next_color = "red" if current_color != "red" else "gray17"
            self.configure(fg_color=next_color)
            self.after(500, self.change_frame_color, i+1)
        else:
            self.configure(fg_color="gray17")
            return
