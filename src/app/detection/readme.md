# Detection

This folder contains the files and libraries for detecting if the driver is drowsy, if he is looking in the right direction, and finally if the vehicle interior is too noisy.

- `detector.py`: file containing the `Detector()` class which allows you to manage all three types of possible distractions previously listed. It also allows you to print warning text and video frame information.
- `drowsiness.py`: file containing the `Drowsiness()` class which manages the detection of drowsiness, doing all the related threshold checks after calling the `EyeAspectRatio()` class for calculating the EAR value, useful for understand how closed the eyes are.
- `looking_away.py`: file containing the `LookingAway()` class which manages the detection of the driver's gaze direction, doing all the related threshold checks after calling the `Gaze()` class for gaze calculation score.
- `loudness.py`: file containing the `Loudness()` class which manages the detection of the intensity of the noise inside the vehicle.
