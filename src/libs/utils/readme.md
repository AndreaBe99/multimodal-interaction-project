# Utils

This folder contains different types of classes and files:

- `config.py` contains the `Colors` and `Path` classes of type Enum for specifying colors and paths by names.
- `ear.py` contains the `EyeAspectRatio` class, useful for calculating the EAR value to detect if the driver is sleeping.
- `face_mesh.py` contains the `FaceMesh` class useful for calculating the driver's face landmarks via the Mediapipe library.
- `gaze.py` containing the `Gaze` class used to calculate the direction of the driver's gaze, and the gaze score, i.e. the average value between the left and right eye gazes. The gaze score is the Euclidean distance between the pupil and the center of the eye.
