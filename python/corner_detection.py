"""Detect the corrners of the overlay provided by the deep learning model
to find the endpoints of detected lanelines using Harris corner detection
"""
import logging
import matplotlib.pyplot as plt
import numpy as np

import angle_calculation


def bruteforce(image:np.ndarray):
    image = np.float32(np.digitize(image,[0.5]))
    rows = np.array([40, 59])
    corners = np.zeros((2,2,2))
    flag_lane = False
    for i,row in enumerate(rows):
        j = 0
        for k,pixel in enumerate(image[row]):
            if flag_lane == False and pixel > 0.5:
                corners[i,j] = np.array([row, k])
                flag_lane = True
            if flag_lane == True and pixel < 0.5:
                corners[i,j+1] = np.array([row, k])
                flag_lane = False
                j += 1
                break
    return corners


## Set up the logger
logging.basicConfig(format="[%(name)s][%(levelname)s] %(message)s")
logger = logging.getLogger("corner-detection")
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    ## Read image
    inputdir = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Post Processing"
    detectedlanes = plt.imread(f"{inputdir}\\frame_fitaverage.png")  ## 160x80
    image = plt.imread(f"{inputdir}\\frame_result.png") 

    corners = bruteforce(detectedlanes)
    logger.debug(f"coordinates: \n{corners}")

    def test():
        """Test angle=calculation module compatibility"""
        calculator = angle_calculation.Calculator(image,corners)
        plt.imshow(calculator.image)
        plt.show()

    test()

