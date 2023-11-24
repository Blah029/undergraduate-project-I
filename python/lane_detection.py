"""Run inference on a dashcam footage using the trained CNN model"""
import cv2
import logging
import numpy as np
from moviepy.editor import VideoFileClip
from tensorflow import keras

import angle_calculation as angle
import corner_detection as corner
import retinex


class Detector:
    def __init__(self, 
                 confidence:float=0.5, 
                 dumpframes:bool=False, 
                 runretinex:bool=False,
                 retinex_low_percent:int=50):
        """Initialise Detector class containig immediate and moving average 
        probabiltiies of each pixel be part of a lane"""
        ## Store average and recent predictions
        self.confidence = confidence
        self.dumpframes = dumpframes
        self.runretinex = runretinex
        self.retinex_low_percent = retinex_low_percent
        self.framecount = 0
        self.predection_recent = []
        self.corners_recent = []

    def detect(self, image:np.ndarray=None):
        """Run prediction and get lane probability of each pixel"""
        if image != None:
            self.image = image
        ## Perform retinex image enhancement
        if self.runretinex:
            self.image = retinex.msrcr(
                self.image,
                low_percent=self.retinex_low_percent
                )
        ## Strip alpha channel, resize image to 160x80, and add axis
        self.image = self.image[:,:,:3]
        self.img_resized = cv2.resize(self.image,(160, 80))
        self.img_resized = self.img_resized[None,:,:,:]
        ## Recent
        prediction = model.predict(self.img_resized)[0] * 255
        logger.debug(f"predection max: {np.max(prediction)}")
        self.predection_recent.append(prediction)
        ## Moving average
        if len(self.predection_recent) > 5:
            self.predection_recent = self.predection_recent[1:]
        self.prediction_average = \
            np.mean(np.array([i for i in self.predection_recent]), axis = 0)
        logger.debug(f"predection_average max: {np.max(self.prediction_average)}, shape: {self.prediction_average.shape}")
        ## Crop
        self.prediction_average[:40,:] = \
            np.zeros_like(self.prediction_average[:40,:])
        self.prediction_average[60:,:] = \
            np.zeros_like(self.prediction_average[60:,:])
        ## Quantise
        self.prediction_average = \
            np.digitize(self.prediction_average,
                        [self.confidence*np.max(self.prediction_average)])*255.0
        logger.debug(f"self.prediction_average max after quantising: {np.max(self.prediction_average)}")
        ## Resize
        predection_resized = \
            cv2.resize(self.prediction_average,(self.image.shape[1], 
                                                self.image.shape[0]))
        logger.debug(f"predection_resized max: {np.max(predection_resized)}")
        ## Quantise
        # predection_resized = \
        #     np.digitize(predection_resized,
        #                 [self.confidence*np.max(predection_resized)])*255
        # logger.debug(f"predection_resized max after quantising: {np.max(predection_resized)}")
        ## Create RGB image
        blanks = np.zeros_like(predection_resized).astype(np.uint8)
        self.predection_rgb = np.dstack((blanks, predection_resized, blanks))
        ## Dump detection
        if self.dumpframes == True:
            dir_dump = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Model Outputs\\framedump"
            cv2.imwrite(f"{dir_dump}\\{self.framecount}-1_average.png",self.prediction_average)
            cv2.imwrite(f"{dir_dump}\\{self.framecount}-2_resized.png",predection_resized)
            self.framecount += 1

    def overlay_lane(self,image):
        """Overlay detection onto the image"""
        self.image = image
        self.detect()
        self.result = cv2.addWeighted(self.image,1,self.predection_rgb,1,0, 
                                              dtype=cv2.CV_32F)
        self.result = np.clip(self.result,None,255)
        ## Dump result
        if self.dumpframes == True:
            dir_dump = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Model Outputs\\framedump"
            cv2.imwrite(f"{dir_dump}\\{self.framecount-1}-3_laneimage.png",self.result[:,:,::-1])
        return self.result
    
    def replace_lane(self,image):
        """Replace image with lane detection mask"""
        self.image = image
        self.detect()
        return self.predection_rgb
    
    def overlay_all(self,image):
        """Overlay lane detection, angle deviation, positional deviation and 
        departure warning onto the image
        """
        self.image = image
        self.detect()
        ## Recent
        corners = corner.linesearch(self.prediction_average)
        self.corners_recent.append(corners)
        ## Moving average
        if len(self.corners_recent) > 5:
            self.corners_recent = self.corners_recent[1:]
        self.corners_average = \
            np.mean(np.array([i for i in self.corners_recent]), axis = 0)
        # logger.debug(f"corners: \n{corners}")
        calculator = angle.Calculator(self.image,self.corners_average)
        self.image = calculator.image
        self.result = cv2.addWeighted(self.image,1,self.predection_rgb,0.25,0, 
                                              dtype=cv2.CV_32F)
        self.result = np.clip(self.result,None,255)
        ## Dump result
        if self.dumpframes == True:
            dir_dump = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Model Outputs\\framedump"
            cv2.imwrite(f"{dir_dump}\\{self.framecount-1}-4_trajectoryimage.png",self.result[:,:,::-1])
        return self.result

## Set up the logger
logging.basicConfig(format="[%(name)s][%(levelname)s] %(message)s")
logger = logging.getLogger("corner-detection")
## Script
if __name__ == "__main__":
    # logger.setLevel(logging.DEBUG)
    ## Set I/O
    dir_model = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Models\\FCNN"
    # dir_input = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Datasets\\video-footage\\local-dashcam"
    dir_input = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Datasets\\video-footage"
    dir_output = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Model Outputs"
    vid_name = "techodyssey_1080p.MP4"
    ## Original model
    # confidence = 0.05
    # model = keras.models.load_model(f"{dir_model}\\MLND-Capstone\\full_CNN_model.h5")
    # label = f"originalmodel_{confidence}"
    ## Duct tape fixed model trained on custom data set of 250 images
    # model_version = 0
    # confidence = 0.45
    ## Fixed model trained on custom data set of 250 images
    model_version = 3
    confidence = 0.35     ## optimal threshold for daytime
    ## Complie separately to fix weight_decay typeerror
    model = keras.models.load_model(f"{dir_model}\\test_model_v{model_version}.h5", compile=False)
    model.compile(optimizer='Adam', loss='mean_squared_error')
    label = f"custommodelv{model_version}_{confidence}_msrcr"
    ## Perform detection
    detector = Detector(confidence)
    ## Configure retinex
    detector.runretinex = True
    detector.retinex_low_percent = 50
    vid_input = VideoFileClip(f"{dir_input}\\{vid_name}", audio=False)
        
    def detect_clip(start:int, end:int):
        """Perform lane detection on a part of a video"""
        detector.dumpframes = True
        ## Read video and trim
        vid_trimmed = vid_input.subclip(start,end)
        ## Modify frames
        vid_output = vid_trimmed.fl_image(detector.overlay_lane)
        ## Save trimmed video
        vid_output.write_videofile(
            f"{dir_output}\\trimmed_{vid_name[:-4]}_detected_{label}.MP4"
            )
        
    def detect_full():
        """Perform lane detection on a full a video"""
        detector.dumpframes = True
        # Modify frames
        vid_output = vid_input.fl_image(detector.overlay_lane)
        # Save video
        vid_output.write_videofile(
            f"{dir_output}\\{vid_name[:-4]}_detected_{label}.MP4"
            )
    
    def detectncalculate_clip(start:int, end:int):
        """Test anlge_calculation module integration"""
        detector.dumpframes = True
        vid_trimmed = vid_input.subclip(start,end)
        vid_output = vid_trimmed.fl_image(detector.overlay_all)
        vid_output.write_videofile(
            f"{dir_output}\\trimmed_{vid_name[:-4]}_calculated_{label}.MP4"
            )
        
    def detectncalculate_full():
        """Perform lane detection and trajectory calculation on a full video"""
        detector.dumpframes = False
        vid_output = vid_input.fl_image(detector.overlay_all)
        vid_output.write_videofile(
            f"{dir_output}\\{vid_name[:-4]}_calculated_{label}.MP4"
            )
        
    # detect_clip(3,6)
    detect_full()
    # detectncalculate_clip(14,19)
    # detectncalculate_full()