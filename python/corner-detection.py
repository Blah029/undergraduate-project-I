"""Detect the corrners of the overlay provided by the deep learning model
to find the endpoints of detected lanelines using Harris corner detection
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

## Read image
inputdir = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Post Processing"
# src = plt.imread(f"{inputdir}\\frame_fitaverage.png")   ## 160x80
src = plt.imread(f"{inputdir}\\frame_laneresized.png")  ## Upscaled
## Preprocess
src = np.float32(np.digitize(src,[0.5,1]))
if len(src.shape) > 2:
    ## Make greyscale copy
    gray = src[:,:,1]
else:
    gray = src
    ## Convert geryscale to RGB
    src = np.dstack([np.zeros_like(src), src, np.zeros_like(src)])
gray = np.float32(gray)
# dst = cv2.cornerHarris(gray,2,3,0.04)   ## Default
# dst = cv2.cornerHarris(gray,2,3,0.2)   ## For 160x80
dst = cv2.cornerHarris(gray,10,31,0)     ## For upscaled
## Result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
## Threshold for an optimal value, it may vary depending on the image.
src[dst>0.1*dst.max()]=[255,0,0]
plt.imshow(src)
plt.show()
