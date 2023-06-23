"""Run inference on a dashcam footage using the trained CNN model"""
import cv2
import logging
import numpy as np
from moviepy.editor import VideoFileClip
from tensorflow import keras

import angle_calculation as angle
import corner_detection as corner


class Detector:
    def __init__(self, confidence:float=0.5, dumpframes:bool=False):
        """Initialise Detector class containig immediate and moving average 
        probabiltiies of each pixel be part of a lane"""
        ## Store average and recent predictions
        self.confidence = confidence
        self.dumpframes = dumpframes
        self.framecount = 0
        self.predection_recent = []
        self.corners_recent = []

    def detect(self, image:np.ndarray=None):
        """Run prediction and get lane probability of each pixel"""
        ## Strip alpha channel and resize image to 160x80
        if image != None:
            self.image = image
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
            dir_dump = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Model Outputs\\framedump"
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
            dir_dump = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Model Outputs\\framedump"
            cv2.imwrite(f"{dir_dump}\\{self.framecount-1}-3_laneimage.png",self.result[:,:,::-1])
        return self.result
    
    def replace_lane(self,image):
        self.image = image
        self.detect()
        return self.predection_rgb
    
    def overlay_all(self,image):
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
            dir_dump = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Model Outputs\\framedump"
            cv2.imwrite(f"{dir_dump}\\{self.framecount-1}-4_trajectoryimage.png",self.result[:,:,::-1])
        return self.result

## Set up the logger
logging.basicConfig(format="[%(name)s][%(levelname)s] %(message)s")
logger = logging.getLogger("corner-detection")
## Script
if __name__ == "__main__":
    ## Set up the logger
    logging.basicConfig(format="[%(name)s][%(levelname)s] %(message)s")
    logger = logging.getLogger("corner-detection")
    # logger.setLevel(logging.DEBUG)
    ## Set I/O
    dir_model = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Models"
    dir_input = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Datasets\\video-footage\\local-dashcam"
    dir_output = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Model Outputs"
    vid_name = "localDashcam_1440p_02.MP4"
    ## Original model
    # model = keras.models.load_model(f"{dir_model}\\MLND-Capstone\\full_CNN_model.h5")
    # confidence = 0.05
    # label = f"originalmodel_{confidence}"
    ## Duct tape fixed model trained on custom data set of 250 images
    # model = keras.models.load_model(f"{dir_model}\\Custom\\test_model_v0.h5")
    # confidence = 0.45
    # label = f"custommodelv0_{confidence}"
    ## Fixed model trained on custom data set of 250 images
    model = keras.models.load_model(f"{dir_model}\\Custom\\test_model_v1.h5")
    confidence = 0.35
    label = f"custommodelv1_{confidence}"
    ## Perform detection
    detector = Detector(confidence)
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
        detector.dumpframes = False
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
        
    # detect_clip(14,19)
    # detect_full()
    detectncalculate_clip(14,19)
    # detectncalculate_full()