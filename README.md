# multimodal-interaction-project
Project for the Multimodal Interaction course, for the master's degree program at the Sapienza University of Rome.

## How to use

From the terminal, run the following command:

### CLI

```cli
python app.py -e CLI -w video -t 10
```

- to record only the video for 10 seconds

```cli
python app.py -e CLI -w audio -t 10
```

- to record only the audio for 10 seconds

```cli
python app.py -e CLI -w both -t 10
```

- to record both audio and video for 10 seconds

### GUI

```cli
python app.py -e GUI: to run the GUI
```

- to use the graphical interface to record audio and video

#### GUI Main

![GUI Main](data/gui_main.png)

#### GUI Video

![GUI Video](data/gui_video_1.png)

![GUI Video Execution](data/gui_video_2.png)

#### GUI Audio

![GUI Audio](data/gui_audio_1.png)

![GUI Audio Execution](data/gui_audio_2.png)

## BugFixing

If installation of requirements fail on PyAudio can'find portaudio.h do the following:

```cli
sudo apt install portaudio19-dev
```
