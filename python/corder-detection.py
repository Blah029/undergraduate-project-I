"""Detect the corrners of the overlay provided by the deep learning model
to find the endpoints of detected lanelines using Harris corner detection
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

## Read image
inputdir = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Post Processing"
detection = plt.imread(f"{inputdir}\\frame.png")
## Preprocess
detection = 
gray = detection[:,:,1]
gray = np.float32(gray)
print(gray.shape)
# dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.cornerHarris(gray,50,1,0)
## Result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
## Threshold for an optimal value, it may vary depending on the image.
detection[dst>0.1*dst.max()]=[255,0,0]
plt.imshow(detection[:,:,0])
plt.show()
