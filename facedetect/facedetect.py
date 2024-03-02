import argparse
from math import atan2
from math import cos
from math import sin
import time
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
import numpy as np
import skimage.transform



BROW_INDICES = (54, 104, 69, 108, 151, 337, 299, 333, 284)
LEFT_EYE_CONTOUR_INDICES = [ 263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 386, 387, 388, 466, 263 ]
RIGHT_EYE_CONTOUR_INDICES = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246 ]
LIPS_CONTOUR_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185 ]


def convert_landmarks_to_pixel_coordinates(landmarks: List[Tuple[int, int]], image_shape: Tuple[int, int, int]) -> List[Tuple[int, int]]:
    """
    Parameters
    ----------
    landmarks : List[Tuple[int, int]]
        Array of landmarks, which are expressed in normalized image coordinates.
    image_shape : Tuple[int, int, int]
        Image shape: (height, width, channels).

    Returns
    -------
    List[Tuple[int, int]]
        A list of (y, x) pixel coordinates corresponding to each landmark.
    """
    return [ (int(image_shape[0] * landmark.y), int(image_shape[1] * landmark.x)) for landmark in landmarks ]

def detect_faces(image: np.ndarray) -> List[List[Tuple[int, int]]]:
    with mp_face_mesh.FaceMesh(
        max_num_faces = 8,
        refine_landmarks = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    ) as face_mesh:
        faces = []
        results = face_mesh.process(image)
        if results.multi_face_landmarks is None:  # no faces
            print("Warning: No face detected")
            return []
        if len(results.multi_face_landmarks) == 0:
            print("Warning: No face landmarks detected")
            return []
        for i in range(len(results.multi_face_landmarks)):
            landmarks = convert_landmarks_to_pixel_coordinates(landmarks = results.multi_face_landmarks[i].landmark, image_shape = image.shape)
            faces.append(landmarks)
        return faces

def convert_yx_to_xy(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Converts a list of (y, x) points into (x, y) format.

    Parameters
    ----------
    points : List[Tuple[int, int]]
        A list of (y, x) points.

    Returns
    -------
    List[Tuple[int, int]]
        A list of (x, y) points.
    """
    return [ (x, y) for (y, x) in points ]

def compute_face_bounding_box_from_landmarks(landmarks: List[Tuple[int,int]]) -> Tuple[int,int,int,int]:
    """
    Fits a bounding box to a face mesh.

    Parameters
    ----------
    landmarks : List[Tuple[int,int]]
        All face landmarks as (y,x) pixel coordinates.

    Returns
    -------
    [int, int, int, int]
        Box corners: (y_min, y_max, x_min, x_max).
    """
    box_corners = [ 1000000, 1000000, -1000000, -1000000 ]
    for landmark in landmarks:
        box_corners[0] = min(box_corners[0], landmark[0]) # y1
        box_corners[2] = max(box_corners[2], landmark[0]) # y2
        box_corners[1] = min(box_corners[1], landmark[1]) # x1
        box_corners[3] = max(box_corners[3], landmark[1]) # x2
    return box_corners

def compute_face_height_in_pixels(landmarks):
    """
    Parameters
    ----------
    landmarks : List[Tuple[int,int]]
        All face landmarks as (y,x) pixel coordinates.

    Returns
    -------
    int
        Characteristic height of face in pixels. This does not correspond exactly
        to the entire face (the bottom landmark is not the bottom-most one, for
        instance) but is useful nonetheless for normalizing hair line y-values to
        some characteristic face dimension.
    """
    return abs(landmarks[175][0] - landmarks[10][0])

def fix_rotation(image: np.ndarray, landmarks: List[Tuple[int,int]]) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    # Function to create a standard 2D rotation matrix
    def create_rotation_matrix(angle):
        theta = np.deg2rad(angle)
        return np.array([
            [ cos(theta), -sin(theta) ],
            [ sin(theta), cos(theta) ]
        ])

    # Function to rotate point of form (y,x) using standard 2D rotation matrix
    def rotate_point(point, matrix, pivot):
        pivot = np.array([ [ pivot[1] ] , [ pivot[0] ] ])
        point = np.array([ [ point[1] ] , [ point[0] ] ]) # column vector [x;y] to be compatible with rotation matrix
        rotated = np.round(np.matmul(matrix, point - pivot) + pivot).astype(int)
        return (rotated[1,0], rotated[0,0])

    # Compute angle based on vertical face center line (landmark 10 is at the top
    # of the face, 175 on the bottom)
    hor = landmarks[10][1] - landmarks[175][1]
    ver = landmarks[175][0] - landmarks[10][0]
    angle = np.rad2deg(atan2(hor, ver))

    # Rotate image about its center
    image = skimage.transform.rotate(image = image, angle = angle, preserve_range = True).astype(np.uint8)

    # Rotate all points
    rotation_matrix = create_rotation_matrix(angle = -angle)
    pivot_point = (image.shape[0] * 0.5 - 0.5, image.shape[1] * 0.5 - 0.5)  # rotate about center of image
    landmarks = [ rotate_point(point = landmark, matrix = rotation_matrix, pivot = pivot_point) for landmark in landmarks ]
    return image, landmarks

def process_image(filepath: str) -> List[np.ndarray]:
    # Load image and detect faces
    image = cv2.imread(filepath)  # loads apparently in RGB format (not BGR like from webcam)
    image.flags.writeable = False
    faces = detect_faces(image=image)
    
    # Create an output image for each face detect
    out_images = []
    normalized_faces = []
    for i in range(len(faces)):
        out_image, normalized_face = fix_rotation(image=image, landmarks=faces[i])

        # Draw bounding box
        box_corners = compute_face_bounding_box_from_landmarks(landmarks=normalized_face)
        out_image.flags.writeable = True
        color = (255, 255, 255)
        cv2.line(img = out_image, pt1 = (box_corners[1], box_corners[0]), pt2 = (box_corners[3], box_corners[0]), color = color, thickness = 2)
        cv2.line(img = out_image, pt1 = (box_corners[3], box_corners[0]), pt2 = (box_corners[3], box_corners[2]), color = color, thickness = 2)
        cv2.line(img = out_image, pt1 = (box_corners[3], box_corners[2]), pt2 = (box_corners[1], box_corners[2]), color = color, thickness = 2)
        cv2.line(img = out_image, pt1 = (box_corners[1], box_corners[2]), pt2 = (box_corners[1], box_corners[0]), color = color, thickness = 2)

        # Draw vertical center line of face
        y1, x1 = normalized_face[10]
        y2, x2 = normalized_face[175]
        cv2.line(img = out_image, pt1 = (x1, y1), pt2 = (x2, y2), color = (0, 0, 255), thickness = 2)

        # Store
        out_images.append(out_image)
        normalized_faces.append(normalized_face)
    
    return out_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser("facedetect")
    parser.add_argument("input", help = "Image file")
    options = parser.parse_args()

    t0 = time.perf_counter()
    images = process_image(filepath=options.input)
    t1 = time.perf_counter()
    print(f"Processing took {(t1 - t0):1.2f} seconds")
    for i in range(len(images)):
        outfile = f"out-{i}.png"
        cv2.imwrite(outfile, images[i])
        print(f"Wrote {outfile}")
    