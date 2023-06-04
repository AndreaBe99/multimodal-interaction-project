import cv2
import sys
import mediapipe as mp
import numpy as np

sys.path.append("./")
from src.libs.utils.colors import Colors


class Gaze:
    """Class to manage the gaze direction."""
    def __init__(self):
        self.relative = lambda landmark, shape: (
            int(landmark.x * shape[1]),
            int(landmark.y * shape[0]),
        )
        self.relativeT = lambda landmark, shape: (
            int(landmark.x * shape[1]),
            int(landmark.y * shape[0]),
            0,
        )
        # 2D image points.
        self.face_coordinates_2D = {
            "nose_tip": 4,  # Nose tip
            "chin": 152,  # Chin
            "left_eye": 263,  # Left eye left corner
            "right_eye": 33,  # Right eye right corner
            "left_mouth": 287,  # Left Mouth corner
            "right_mouth": 57,  # Right mouth corner
            "left_pupil": 468,  # Left pupil
            "right_pupil": 473,  # Right pupil
        }
        # 3D model points.
        self.face_coordinates_3D = {
            "nose_tip": (0.0, 0.0, 0.0),  # Nose tip
            "chin": (0, -63.6, -12.5),  # Chin
            "left_eye": (-43.3, 32.7, -26),  # Left eye left corner
            "right_eye": (43.3, 32.7, -26),  # Right eye right corner
            "left_mouth": (-28.9, -28.9, -24.1),  # Left Mouth corner
            "right_mouth": (28.9, -28.9, -24.1),  # Right mouth corner
            "eye_ball_center_left": (29.05, 32.7, -39.5),  # Left eye ball center
            "eye_ball_center_right": (-29.05, 32.7, -39.5),  # Right eye ball center
        }
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        self.rotation_vector = None
        self.translation_vector = None
        self.transformation = None
        self.camera_matrix = None

    
    def get_relative(
        self,
        frame: np.ndarray,
        points,
        function,
    ) -> np.ndarray:
        """
        2D image points. `relative` takes mediapipe points that is normalized 
        to [-1, 1] and returns image points at (x,y) format

        Args:
            frame: The frame to get the relative points from.
            points: The points to get the relative points from.
            function: The function to use to get the relative points. 
                `relative` or `relativeT`.

        Returns:
            The relative points.
        """
        image_points = np.array(
            [
                function(
                    points.landmark[self.face_coordinates_2D["nose_tip"]], 
                    frame.shape
                ),
                function(
                    points.landmark[self.face_coordinates_2D["chin"]], 
                    frame.shape
                ),
                function(
                    points.landmark[self.face_coordinates_2D["left_eye"]], 
                    frame.shape
                ),
                function(
                    points.landmark[self.face_coordinates_2D["right_eye"]], 
                    frame.shape
                ),
                function(
                    points.landmark[self.face_coordinates_2D["left_mouth"]], 
                    frame.shape
                ),
                function(
                    points.landmark[self.face_coordinates_2D["right_mouth"]],
                    frame.shape,
                ),
            ],
            dtype="double",
        )
        return image_points

    
    def get_camera_matrix(self, frame: np.ndarray) -> np.ndarray:
        """
        Camera matrix estimation.

        Args:
            frame: The frame to get the camera matrix from.

        Returns:
            The camera matrix.
        """
        focal_length = frame.shape[1]
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]], 
                [0, focal_length, center[1]], 
                [0, 0, 1]
            ],
            dtype="double",
        )
        return camera_matrix
    
    
    def get_gaze_3d(self, pupil: np.ndarray, eye_ball_center: np.ndarray) -> np.ndarray:
        """
        Compute 3D gaze point, i.e. point in real world coordinates where user
        is looking at.

        Args:
            pupil (np.ndarray): pupil image point
            eye_ball_center (np.ndarray): eye ball center in 3d world coordinates

        Returns:
            np.ndarray: _description_
        """
        # if estimateAffine3D secsseded project pupil image point into 3d world point
        pupil_world_cord = (
            self.transformation @ np.array(
                [[pupil[0], pupil[1], 0, 1]]
            ).T
        )
        # 3D gaze point (10 is arbitrary value denoting gaze distance)
        gaze_3d = eye_ball_center + (pupil_world_cord - eye_ball_center) * 10
        return gaze_3d, pupil_world_cord

    
    def project_points(self, coords) -> np.ndarray:
        """
        Project 3D points to 2D.

        Args:
            coords: The coordinates to project.

        Returns:
            The projected points.
        """
        return cv2.projectPoints(
            (int(coords[0]), int(coords[1]), int(coords[2])),
            self.rotation_vector,
            self.translation_vector,
            self.camera_matrix,
            self.dist_coeffs,
        )

    
    def correct_gaze(
        self, 
        pupil: np.ndarray, 
        eye_pupil2D: np.ndarray, 
        head_pose: np.ndarray
    ) -> np.ndarray:
        """
        Corrects the gaze direction by using the pupil location.

        Args:
            pupil: The pupil location.
            eye_pupil2D: The eye pupil location.
            head_pose: The head pose.

        Returns:
            The corrected gaze direction.
        """
        # correct gaze for head rotation
        gaze = pupil + (eye_pupil2D[0][0] - pupil) - (head_pose[0][0] - pupil)
        return gaze

    
    def plot_gaze(
        self,
        frame: np.ndarray,
        pupil: np.ndarray,
        gaze: np.ndarray,
    ) -> None:
        """
        Draw gaze line into screen.
        
        Args:
            frame: The frame to draw the gaze line into.
            pupil: The pupil location.
            gaze: The gaze direction.
        """
        p1 = (int(pupil[0]), int(pupil[1]))
        p2 = (int(gaze[0]), int(gaze[1]))
        cv2.line(frame, p1, p2, Colors.RED.value, 2)
    
    
    def compute_gaze(
        self,
        frame: np.ndarray,
        points,
    ) -> None:
        """
        The gaze function gets an image and face landmarks from mediapipe 
        framework, and draws the gaze direction into the frame.
        
        Args:
            frame: The frame to draw the gaze line into.
            points: The face landmarks.
        """
        image_points = self.get_relative(frame, points, self.relative)
        image_points1 = self.get_relative(frame, points, self.relativeT)
        
        # 3D model points.
        model_points = np.array(
            [
                self.face_coordinates_3D["nose_tip"],  # Nose tip
                self.face_coordinates_3D["chin"],  # Chin
                self.face_coordinates_3D["left_eye"],  # Left eye left corner
                self.face_coordinates_3D["right_eye"],  # Right eye right corner
                self.face_coordinates_3D["left_mouth"],  # Left Mouth corner
                self.face_coordinates_3D["right_mouth"],  # Right mouth corner
            ]
        )
        
        # 3D model eye points.
        eye_ball_center_left = np.array(
            [
                [self.face_coordinates_3D["eye_ball_center_left"][0]],
                [self.face_coordinates_3D["eye_ball_center_left"][1]],
                [self.face_coordinates_3D["eye_ball_center_left"][2]],
            ]
        )  # the center of the left eyeball as a vector.
        eye_ball_center_right = np.array(
            [
                [self.face_coordinates_3D["eye_ball_center_right"][0]],
                [self.face_coordinates_3D["eye_ball_center_right"][1]],
                [self.face_coordinates_3D["eye_ball_center_right"][2]],
            ]
        )  # the center of the right eyeball as a vector.
        
        self.camera_matrix = self.get_camera_matrix(frame)
        
        (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(
            model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        
        # 2D pupil location
        left_pupil = self.relative(
            points.landmark[self.face_coordinates_2D["left_pupil"]], 
            frame.shape
        )
        right_pupil = self.relative(
            points.landmark[self.face_coordinates_2D["right_pupil"]], 
            frame.shape
        )
        
        # Transformation between image point to world point
        _, self.transformation, _ = cv2.estimateAffine3D(
            image_points1, model_points
        )  # image to world transformation
        
        if self.transformation is None:
            return
        
        # if estimateAffine3D secsseded project pupil image point into 3d world point
        left_gaze_3d, left_pupil_world_cord = self.get_gaze_3d(
            left_pupil, eye_ball_center_left
        )
        right_gaze_3d, right_pupil_world_cord = self.get_gaze_3d(
            right_pupil, eye_ball_center_right
        )
        
        # Project a 3D gaze direction onto the image plane.
        (left_eye_pupil2D, _) = self.project_points(left_gaze_3d)
        (right_eye_pupil2D, _) = self.project_points(right_gaze_3d)
        
        # project 3D head pose into the image plane
        (head_pose, _) = self.project_points(
            (left_pupil_world_cord[0], left_pupil_world_cord[1], 40)
        )
        # NOTE! Maybe it is necessary to implement the head pose detection, also
        # for the right eye, and take the average of the two head poses.
        
        # correct gaze for head rotation
        left_gaze = self.correct_gaze(left_pupil, left_eye_pupil2D, head_pose)
        right_gaze = self.correct_gaze(right_pupil, right_eye_pupil2D, head_pose)
        
        # Draw gaze line into screen
        self.plot_gaze(frame, left_pupil, left_gaze)
        # self.plot_gaze(frame, right_pupil, right_gaze)
        
        return left_gaze, right_gaze





if __name__ == "__main__":
    mp_face_mesh = mp.solutions.face_mesh  # initialize the face mesh model

    # camera stream:
    cap = cv2.VideoCapture(-1)  # chose camera index (try 1, 2, 3)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # number of faces to track in each frame
        refine_landmarks=True,  # includes iris landmarks in the face mesh model
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        
        gaze = Gaze()  # initialize gaze estimation class
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:  # no frame input
                print("Ignoring empty camera frame.")
                continue
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(
                image, cv2.COLOR_BGR2RGB
            )  # frame to RGB for the face-mesh model
            results = face_mesh.process(image)
            image = cv2.cvtColor(
                image, cv2.COLOR_RGB2BGR
            )  # frame back to BGR for OpenCV

            if results.multi_face_landmarks:
                gaze.compute_gaze(image, results.multi_face_landmarks[0])  # gaze estimation
            cv2.imshow("output window", image)
            if cv2.waitKey(2) & 0xFF == 27:
                break
    cap.release()
