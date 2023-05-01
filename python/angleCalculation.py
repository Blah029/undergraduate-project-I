import math
import numpy as np
import matplotlib.pyplot as plt


def angleCalculation_v1(roadOverlay, frameSize):
    # Unpack x coordinates
    x0 = roadOverlay[0,0]
    x1 = roadOverlay[1,0]
    x2 = roadOverlay[2,0]
    x3 = roadOverlay[3,0]
    # Unpack y coordinates
    y0 = max(roadOverlay[0,1],roadOverlay[3,1])
    y1 = min(roadOverlay[1,1],roadOverlay[2,1])
    # Road trajectory x coordinates
    xt0 = 0.5*(0 + frameSize[0])   # viewport midpoint
    xt1 = (xt0*(x1-x2) - x1*x3 + x0*x2)/(x0-x3)
    # Angle before perspective adjustment 
    angle = math.acos((y1-y0)/math.sqrt(pow(xt1-xt0, 2) + pow(y1-y0, 2)))

    # Return trajectory and angle
    return np.array([[xt0, y0], [xt1, y1]]), math.degrees(angle)


def angleCalculation_v2(roadOverlay, frameSize):
    # Unpack coordinates
    x0,y0 = roadOverlay[0,0],roadOverlay[0,1]
    x1,y1 = roadOverlay[1,0],roadOverlay[1,1]
    x2,y2 = roadOverlay[2,0],roadOverlay[2,1]
    x3,y3 = roadOverlay[3,0],roadOverlay[3,1]
    xf,yf = frameSize[0],frameSize[1]
    # Calculate coordinates
    xt0 = xf/2
    yt0 = 0
    yt1 = (y1+y2)/2

    def xr(y):
        return x2 - (x2-x3)/(y2-y3)*(y2-y)
    
    def xl(y):
        return x1 - (x1-x0)/(y1-y0)*(y1-y)
    
    xr0 = xr(yt0)
    xr1 = xr(yt1)
    xl0 = xl(yt0)
    xl1 = xl(yt1)
    xt1 = (xt0*(xl1-xr1) - xl1*xr0 + xl0*xr1)/(xl0-xr0)
    # Angle before perspective adjustment
    angle = math.atan((xt0-xt1)/(yt1-yt0))

    # Return trajectory and angle
    return np.array([[xt0, yt0], [xt1, yt1]]), math.degrees(angle)


def plotOverlay(roadOverlay, frameSize,
                trajectory, trajectoryColour="tab:orange"):
    # Place frame boundary
    plt.plot(frameSize[0],frameSize[1],".", markersize=0)
    # Plot detected road
    xLeftLine = np.array([roadOverlay[0,0], roadOverlay[1,0]])
    yLeftLine = np.array([roadOverlay[0,1], roadOverlay[1,1]])
    xRightLine = np.array([roadOverlay[2,0], roadOverlay[3,0]])
    yRightLine = np.array([roadOverlay[2,1], roadOverlay[3,1]])
    plt.plot(xLeftLine,yLeftLine,".",  markersize=0,
             linestyle="-", color="tab:blue")
    plt.plot(xRightLine,yRightLine,".",  markersize=0,
             linestyle="-", color="tab:blue")
    # Plot lane trajectory
    xTrajectory = np.array([trajectory[0,0],trajectory[1,0]])
    yTrajectory = np.array([trajectory[0,1],trajectory[1,1]])
    trajectoryPlot, = plt.plot(xTrajectory,yTrajectory,".",
                               linestyle="-", color=trajectoryColour)

    return trajectoryPlot


if __name__ == "__main__":
    # Test perfect detection
    plt.figure("Figure 01: Perfectly Detected Road")
    handles = []
    # Coordinates detected from cv
    roadOverlay = np.array([[5,0], [10,15], [20,15], [30,0]])   # trapezium
    frameSize = np.array([40,30])   # point
    # Version 1
    trajectory, angle = angleCalculation_v1(roadOverlay,frameSize)
    print(f"perfect detection   - angleCalculation_v1 = {angle}")
    handles.append(plotOverlay(roadOverlay,frameSize,trajectory))
    # Version 2
    trajectory, angle = angleCalculation_v2(roadOverlay,frameSize)
    print(f"perfect detection   - angleCalculation_v2 = {angle}")
    handles.append(plotOverlay(roadOverlay,frameSize,trajectory,\
                               "tab:green"))

    plt.legend(handles,["angleCalculation_v1", "angleCalculation_v2"])
    plt.show()

    # Test imperfect detection
    plt.figure("Figure 02: Imerfectly Detected Road")
    handles = []
    # Coordinates detected from cv
    roadOverlay = np.array([[5,7], [10,15], [20,17], [30,0]])   # trapezium
    frameSize = np.array([40,30])   # point
    # Version 1
    trajectory, angle = angleCalculation_v1(roadOverlay,frameSize)
    print(f"imperfect detection - angleCalculation_v1 = {angle}")
    handles.append(plotOverlay(roadOverlay,frameSize,trajectory))
    # Version 2
    trajectory, angle = angleCalculation_v2(roadOverlay,frameSize)
    print(f"imperfect detection - angleCalculation_v2 = {angle}")
    handles.append(plotOverlay(roadOverlay,frameSize,trajectory,\
                               "tab:green"))

    plt.legend(handles,["angleCalculation_v1", "angleCalculation_v2"])
    plt.show()