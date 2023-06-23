"""Detect the corrners of the overlay provided by the deep learning model
to find the endpoints of detected lanelines using Harris corner detection
"""
import logging
import matplotlib.pyplot as plt
import numpy as np

import angle_calculation


def linesearch(image:np.ndarray):
    # image = np.float32(np.digitize(image,[confidence*np.max(image)]))
    rows = np.array([40, 59])
    corners = np.zeros((2,2,2))
    # flag_lane = False
    for i,row in enumerate(rows):
        flag_left = False
        flag_right = False
        j = 0
        for k,pixel in enumerate(image[row]):
            # ## Left
            # if flag_lane == False and pixel > 0.5:
            #     corners[i,j] = np.array([row, k])
            #     flag_lane = True
            # ## Right
            # if flag_lane == True and pixel < 0.5:
            #     corners[i,j+1] = np.array([row, k])
            #     flag_lane = False
            #     j += 1
            #     break
            ## Left
            if flag_left == False and image[row,k] > 0.5:
                corners[i,j] = np.array([row, k])
                flag_left = True
            ## Right
            if flag_right == False and image[row,-k-1] > 0.5:
                corners[i,j+1] = np.array([row, 2*image.shape[0]-k])
                flag_right = True
            if flag_left and flag_right:
                j += 1
                break
    return corners


## Set up the logger
logging.basicConfig(format="[%(name)s][%(levelname)s] %(message)s")
logger = logging.getLogger("corner-detection")
## Script
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    ## Read image
    inputdir = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Post Processing"
    detectedlanes = plt.imread(f"{inputdir}\\frame_fitaverage.png")  ## 160x80
    image = plt.imread(f"{inputdir}\\frame_result.png") 

    corners = linesearch(detectedlanes)
    logger.debug(f"coordinates: \n{corners}")

    def test():
        """Test angle=calculation module compatibility"""
        calculator = angle_calculation.Calculator(image,corners)
        plt.imshow(calculator.image)
        plt.show()

    test()

