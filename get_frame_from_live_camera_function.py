import cv2
import time


# Input:
def get_frame(cap: VideoCapture) -> UMat:
    # Read frame from video feed
    ret, frame = cap.read()
    if not ret:
        break
