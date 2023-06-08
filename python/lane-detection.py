import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from tensorflow import keras


class Lanes:
    def __init__(self, dump:bool=False):
        """Initialise Lanes class containig immediate and moving average 
        probabiltiies of each pixel be part of a lane"""
        ## Store average and recent predictions
        self.dump = dump
        self.framcount = 0
        self.fit_recent = []
        self.fit_average = []

    def detect(self, image:np.ndarray=None):
        """Run prediction and get lane probability of each pixel"""
        ## Strip alpha channel and resize image to 160x80
        if image != None:
            self.image = image
        self.image = self.image[:,:,:3]
        self.img_resized = cv2.resize(self.image,(160, 80))
        self.img_resized = self.img_resized[None,:,:,:]
        ## Recent
        self.prediction = model.predict(self.img_resized)[0] * 255
        self.fit_recent.append(self.prediction)
        ## Moving average
        if len(self.fit_recent) > 5:
            self.fit_recent = self.fit_recent[1:]
        self.fit_average = np.mean(np.array([i for i in self.fit_recent]), 
                                axis = 0)
        ## Clip top and bottom
        self.fit_average[:40,:] = np.zeros_like(self.fit_average[:40,:])
        self.fit_average[60:,:] = np.zeros_like(self.fit_average[60:,:])
        ## Create image
        ## Stack, resize, quantise - 150 frames, 1min 25s
        # blanks = np.zeros_like(self.fit_average).astype(np.uint8)
        # lane_drawn = np.dstack((blanks, self.fit_average, blanks))
        # self.lane_image = cv2.resize(lane_drawn,(self.image.shape[1],
        #                                     self.image.shape[0]))[:,:,:3]
        # self.lane_image = np.digitize(self.lane_image,[0.5,1])*255
        ## Resize, quantise, stack - 150 frames, 52s
        lane_resized = cv2.resize(self.fit_average,(self.image.shape[1],
                                              self.image.shape[0]))
        lane_resized = np.digitize(lane_resized,[0.5, 1])*255
        blanks = np.zeros_like(lane_resized).astype(np.uint8)
        self.lane_image = np.dstack((blanks, lane_resized, blanks))
        ## Dump detection
        if self.dump == True:
            dir_dump = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Model Outputs\\framedump"
            cv2.imwrite(f"{dir_dump}\\{self.framcount}-1_fitaverage.png",self.fit_average)
            cv2.imwrite(f"{dir_dump}\\{self.framcount}-2_laneresized.png",lane_resized)
            self.framcount += 1

    def overlay(self,image):
        """Overlay detection onto the image"""
        self.image = image
        self.detect()
        self.result = cv2.addWeighted(self.image,1,self.lane_image,1,0, 
                                              dtype=cv2.CV_32F)
        self.result = np.clip(self.result,None,255)
        ## Dump new frame
        if self.dump == True:
            dir_dump = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Model Outputs\\framedump"
            cv2.imwrite(f"{dir_dump}\\{self.framcount-1}-3_result.png",self.result[:,:,::-1])
        return self.result
    
    def replace(self,image):
        self.image = image
        self.detect()
        return self.lane_image
    

def detect_clip(video:str, start:int, end:int):
    """Perform detection on a part of a video"""
    ## Read video and trim
    dir_input = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Datasets\\video-footage\\local-dashcam"
    vid_input = VideoFileClip(f"{dir_input}\\{video}", audio=False)
    vid_trimmed = vid_input.subclip(start,end)
    ## Modify frames
    vid_processed = vid_trimmed.fl_image(lanes.overlay)
    ## Save trimmed video
    dir_output = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Model Outputs"
    vid_processed.write_videofile(
        f"{dir_output}\\trimmed_{video[:-4]}_processed.MP4"
        )
    

def detect_full(video:str):
    """Perform detection on a full a video"""    # Read video
    ## Read video
    dir_input = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Datasets\\video-footage\\local-dashcam"
    vid_input = VideoFileClip(f"{dir_input}\\{video}", audio=False)
    # Modify frames
    vid_processed = vid_input.fl_image(lanes.overlay)
    # Save video
    dir_output = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Model Outputs"
    vid_processed.write_videofile(
        f"{dir_output}\\{video[:-4]}_processed.MP4"
        )
    

if __name__ == "__main__":
    ## Load model
    dir_model = "D:\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Models\\MLND-Capstone"
    model = keras.models.load_model(f"{dir_model}\\full_CNN_model.h5")
    lanes = Lanes(dump=True)
    # detect_clip("localDashcam_1440p_02.MP4",15,20)
    # detect_clip("localDashcam_1440p_02.MP4",16,17)
    detect_full("localDashcam_1440p_02.MP4")