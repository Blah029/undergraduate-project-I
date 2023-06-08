import logging
import math
import matplotlib.pyplot as plt
import numpy as np


class Frame:
    def __init__(self, image:np.ndarray, overlay:np.ndarray):
        """Initialise frame objecto with image and lane line coordinates"""
        self.image = np.flipud(image)
        self.overlay = overlay
        self.framesize =  np.array([image.shape[1], image.shape[0]])
        self.calculate_angle()
        self.plot_overlay()
        logger.debug(f"instance image shape: {self.image.shape}")

    def calculate_angle(self):
        """Calculate the angle between detected road trajectory and centerline
        of the frame and return a tuple containing thetrajectory as 
        a 2d array of 2 points, and the angle
        Further improved to be compatible with imperfect detections
        See 4th Year > Final Year Project > Calculation - Angle for notation
        """
        ## Unpack coordinates
        x0, y0 = self.overlay[0,0], self.overlay[0,1]
        x1, y1 = self.overlay[1,0], self.overlay[1,1]
        x2, y2 = self.overlay[2,0], self.overlay[2,1]
        x3, y3 = self.overlay[3,0], self.overlay[3,1]
        xf, yf = self.framesize[0], self.framesize[1]
        ## Calculate coordinates
        xt0 = xf/2
        yt0 = 0
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
        self.trajectory = np.array([[xt0, yt0], [xt1, yt1]])
        ## Angle before perspective adjustment
        self.angle = math.degrees(math.atan((xt0-xt1)/(yt1-yt0)))

    def plot_overlay(self, colour_trajectory:str="yellow", colour_road:str="red"):
        self.figure = plt.figure("Road trajectory detection")
        ## Place frame boundary
        plt.plot(self.framesize[0],self.framesize[1],".", markersize=0)
        ## Plot detected road
        leftline_x = np.array([self.overlay[0,0], self.overlay[1,0]])
        leftline_y = np.array([self.overlay[0,1], self.overlay[1,1]])
        rightline_x = np.array([self.overlay[2,0], self.overlay[3,0]])
        rightline_y = np.array([self.overlay[2,1], self.overlay[3,1]])
        plot_leftline, = plt.plot(leftline_x,leftline_y,".",  markersize=0,
                linestyle="-", color=colour_road)
        plot_rightline, = plt.plot(rightline_x,rightline_y,".",  markersize=0,
                linestyle="-", color=colour_road)
        ## Plot centerline
        plot_centerline, = plt.plot([self.framesize[0]/2, self.framesize[0]/2],[0, self.framesize[1]], 
                 color="orange")
        ## Plot lane trajectory
        trajectory_x = np.array([self.trajectory[0,0], self.trajectory[1,0]])
        trajectory_y = np.array([self.trajectory[0,1], self.trajectory[1,1]])
        plot_trajectory, = plt.plot(trajectory_x,trajectory_y,".",  markersize=0,
                                linestyle="-", color=colour_trajectory)
        plt.legend([plot_centerline, plot_leftline, plot_trajectory],
                   ["Assumed vehicle trajectory", "Detected lane line","Calculated road trajectory"])
        plt.imshow(self.image, origin="lower")


def batchprocess():
    """Temporary function to batch process images"""
    images = []
    for i in range(0,6):
        images.append(plt.imread(f"D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Datasets\\image-footage\\localDashcam_1080p_04_P{i+1}.png"))
    overlays = [np.array([[249, 1080-902], [734, 1080-565], [1016, 1080-484], [1363, 1080-900]]),
                np.array([[14, 1080-846], [361, 1080-643], [853, 1080-492], [951, 1080-909]]),
                np.array([[304+105, 1080-715], [948+105, 1080-515], [1138+105, 1080-535], [1015+105, 1080-908]]),
                np.array([[477, 1080-868], [1010, 1080-499], [1245, 1080-481], [1811, 1080-837]]),
                np.array([[194, 1080-907], [804, 1080-450], [960, 1080-446], [1697, 1080-889]]),
                np.array([[110, 1080-847], [690, 1080-450], [839, 1080-455], [1506, 1080-898]])]
    for i in range(0,6):
        image = images[i]
        overlay = overlays[i]
        frame = Frame(image,overlay)
        logger.info(f"localDashcam_1080p_04_P{i+1} Deviation: {frame.angle}°")
        plt.savefig(f"D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Algorithm Outputs\\localDashcam_1080p_04_P{i+1}.png", 
                    dpi = 330)
        # plt.show()
        plt.close(frame.figure)


if __name__ == "__main__":
    ## Set up the logger
    logging.basicConfig(format="[%(name)s][%(levelname)s] %(message)s")
    logger = logging.getLogger("angle-calculation")
    logger.setLevel(logging.INFO)

    # ## Load test image and coordinates pre-obtained coordinates
    # image_4 = plt.imread(f"D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Datasets\\image-footage\\localDashcam_1080p_04_P4.png")
    # overlay_4 = np.array([[477, 1080-868], [1010, 1080-499], 
    #                       [1245, 1080-481], [1811, 1080-837]])
    # frame_4 = Frame(image_4,overlay_4)
    # plt.savefig()
    # logger.info(f"deviation: {frame_4.angle}")

    batchprocess()
