"""Caculate road trajectory, plot road trajectory, and calculate 
deviation between road trajectory and ego trajectory

References
    - https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array
"""
import cv2
import logging
import math
import matplotlib.pyplot as plt
import numpy as np


class Calculator:
    def __init__(self, image:np.ndarray, coordinates:np.ndarray):
        """Initialise frame objecto with image and lane line coordinates"""
        self.image = image
        ## Scale coordinates form 80x160
        coordinates[:,:,0] = \
            np.array(coordinates[:,:,0]/80*self.image.shape[0])
        coordinates[:,:,1] = \
            np.array(coordinates[:,:,1]/160*self.image.shape[1])
        self.coordinates = coordinates.astype("int")
        self.framesize =  np.array([image.shape[1], image.shape[0]])
        self.findtrajectory()
        # self.plotoverlay_plt()
        self.plotoverlay_cv2()
        logger.debug(f"instance image shape: {self.image.shape}")

    def findtrajectory(self):
        """Calculate the angle between detected road trajectory and centerline
        of the frame and return a tuple containing thetrajectory as 
        a 2d array of 2 points, and the angle
        Further improved to be compatible with imperfect detections
        See 4th Year > Final Year Project > Calculation - Angle for notation
        """
        y0, x0 = tuple(self.coordinates[1,0])
        y1, x1 = tuple(self.coordinates[0,0])
        y2, x2 = tuple(self.coordinates[0,1])
        y3, x3 = tuple(self.coordinates[1,1])
        xf, yf = tuple(self.framesize)
        ## Calculate coordinates
        xt0 = xf/2
        yt0 = yf
        yt1 = (y1+y2)/2

        def xr(y):
            return x2 - (x2-x3)/(y2-y3)*(y2-y)
        
        def xl(y):
            return x1 - (x1-x0)/(y1-y0)*(y1-y)
        
        ## Standardise coordinates
        xr0 = xr(yt0)
        xr1 = xr(yt1)
        xl0 = xl(yt0)
        xl1 = xl(yt1)
        xt1 = (xt0*(xl1-xr1) - xl1*xr0 + xl0*xr1)/(xl0-xr0)
        self.trajectory = np.array([[xt0, yt0], [xt1, yt1]]).astype("int")
        ## Angle before perspective adjustment
        self.angle = math.degrees(math.atan((xt0-xt1)/(yt1-yt0)))


    def plotoverlay_cv2(self, colour_road:str=(255,0,0), 
                        colour_trajectory:tuple=(255,128,0), 
                        colour_centre:tuple=(255,255,0), 
                        thickness:int=10):
        """Plot calculated tracjetory, lane lines, and self trajectory
        onto input image
        """
        ## Ego trajectory
        # logger.debug(f"framesize: {self.framesize}")
        # logger.debug(f"framsize/2 as int: {np.array(self.framesize/2).astype('int')}")
        # logger.debug(f"frametype array as int: {np.array([self.framesize[0]/2, self.framesize[1]]).astype('int')}")
        cv2.line(self.image,
                 np.array(self.framesize/2).astype("int"),
                 np.array([self.framesize[0]/2, self.framesize[1]]).astype("int"),
                 colour_centre,thickness)
        ## Lane lines
        cv2.line(self.image,self.coordinates[0,0,::-1],self.coordinates[1,0,::-1], 
                 colour_road,thickness)
        cv2.line(self.image,self.coordinates[0,1,::-1],self.coordinates[1,1,::-1], 
                 colour_road,thickness)
        ## Road trajectory
        cv2.line(self.image,self.trajectory[0],self.trajectory[1],
                 colour_trajectory,thickness)


## Set up the logger
logging.basicConfig(format="[%(name)s][%(levelname)s] %(message)s")
logger = logging.getLogger("angle-calculation")
## Script
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    def process_single():
        """Load test image and coordinates pre-obtained coordinates"""
        image_4 = plt.imread(f"D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Post Processing\\frame_result.png")
        ## Original detection
        overlay_4 = np.array([[[ 40, 67], [ 40, 96]], 
                              [[ 59, 45], [ 59,132]]])
        ## Scaled detection
        # overlay_4 = np.array([[[1032, 719], [1591, 721]], 
        #                       [[ 680,1072], [2170,1069]]])
        # overlay_4 = overlay_4[:,:,::-1]
        frame_4 = Calculator(image_4,overlay_4)
        logger.info(f"deviation: {frame_4.angle}")
        plt.imshow(frame_4.image)
        plt.show()
        plt.close()

    def process_batch():
        """Temporary function to batch process images"""
        images = []
        for i in range(0,6):
            images.append(plt.imread(f"D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Datasets\\image-footage\\localDashcam_1080p_04_P{i+1}.png"))
        overlays = [np.array([[[ 734, 565], [1016, 484]],
                              [[ 249, 902], [1363, 900]]]),
                    np.array([[[ 361, 643], [ 853, 492]], 
                              [[  14, 846], [ 951, 909]]]),
                    np.array([[[948+105, 515], [1138+105, 535]], 
                              [[304+105, 715], [1015+105, 908]]]),
                    np.array([[[1010, 499], [1245, 481]], 
                              [[ 477, 868], [1811, 837]]]),
                    np.array([[[ 804, 450], [ 960, 446]], 
                              [[ 194, 907], [1697, 889]]]),
                    np.array([[[ 690, 450], [ 839, 455]], 
                              [[ 110, 847], [1506, 898]]])]
        for i in range(0,6):
            image = images[i]
            overlay = overlays[i]
            frame = Calculator(image,overlay)
            logger.info(f"localDashcam_1080p_04_P{i+1} Deviation: {frame.angle}°")
            plt.savefig(f"D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Algorithm Outputs\\localDashcam_1080p_04_P{i+1}.png", 
                        dpi = 330)
            # plt.show()
            plt.close(frame.figure)

    process_single()
    # process_batch()
    # chatgpt1()
    # chatgpt2()
