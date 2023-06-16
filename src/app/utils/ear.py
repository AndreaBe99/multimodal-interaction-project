import typing
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates

class EyeAspectRatio:
    """
    Class for eye aspect ratio calculation, usefull for driver drowsiness 
    detection
    """
    def __init__(self, landmarks, image_w, image_h):
        self.landmarks = landmarks
        self.image_w = image_w
        self.image_h = image_h
        # Eye landmarks indexes, see image in references folder
        # P1, P2, P3, P4, P5, P6
        # NOTE! If we flip the image we need to invert the two lists
        self.left_eye_idxs = [33, 160, 158, 133, 153, 144]
        self.right_eye_idxs = [362, 385, 387, 263, 373, 380]
        
    def distance(self, point_1, point_2)-> float:
        """
        Calculate l2-norm between two points
        
        Args:
            point_1 (list): First point
            point_2 (list): Second point
        
        Returns:
            float: Distance between two points
        """
        dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
        return dist


    def get_ear(self, refer_idxs)-> typing.Tuple[float, list]:
        """
        Calculate Eye Aspect Ratio for one eye.
        
        Args:
            refer_idxs: (list) Index positions of the chosen landmarks
                                in order P1, P2, P3, P4, P5, P6
            
        Returns:
            ear: (float) Eye aspect ratio
            coords_points: (list) List of eye landmarks coordinates
        """
        try:
            # Compute the euclidean distance between the horizontal
            coords_points = []
            for i in refer_idxs:
                lm = self.landmarks[i]
                coord = denormalize_coordinates(lm.x, 
                                                lm.y, 
                                                self.image_w, 
                                                self.image_h)
                coords_points.append(coord)
                
            # Eye landmark (x, y)-coordinates
            P2_P6 = self.distance(coords_points[1], coords_points[5])
            P3_P5 = self.distance(coords_points[2], coords_points[4])
            P1_P4 = self.distance(coords_points[0], coords_points[3])
            
            # Compute the eye aspect ratio
            ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)
            
        except:
            ear = 0.0
            coords_points = None
            
        return ear, coords_points


    def calculate_avg_ear(self)-> typing.Tuple[float, tuple]:
        """
        Calculate average eye aspect ratio
            
        Returns:
            avg_ear: (float) Average eye aspect ratio
            (left_lm_coordinates, right_lm_coordinates): (tuple) Tuple of left 
                and right eye landmarks coordinates
        """
        left_ear, left_lm_coordinates = self.get_ear(self.left_eye_idxs)
        right_ear, right_lm_coordinates = self.get_ear(self.right_eye_idxs)
        avg_ear = (left_ear + right_ear) / 2.0
        
        return avg_ear, (left_lm_coordinates, right_lm_coordinates)