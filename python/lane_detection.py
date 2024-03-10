"""Run inference on a dashcam footage using the trained CNN model"""
import cv2
import logging
import numpy as np
from moviepy.editor import VideoFileClip
from tensorflow import keras

import angle_calculation as angle
import corner_detection as corner
import line_generation as line
import retinex


class Detector:
    def __init__(self, 
                 confidence:float=0.5, 
                 dumpframes:bool=False, 
                 runretinex:bool=False,
                 retinex_low_percent:int=50,
                 generate:bool=False):
        """Initialise Detector class containig immediate and moving average 
        probabiltiies of each pixel be part of a lane"""
        ## Store average and recent predictions
        self.confidence = confidence
        self.dumpframes = dumpframes
        self.runretinex = runretinex
        self.retinex_low_percent = retinex_low_percent
        self.generate = generate
        self.framecount = 0
        self.predection_recent = []
        self.generation_recent = []
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
        ## Strip alpha channel, resize image to 160x80, and add new axis
        self.image = self.image[:,:,:3]
        logger.debug(f"input shape: {self.image.shape}")
        ## Skip downscaling if using line generation. It already does that.
        if not self.generate:
            self.img_resized = cv2.resize(self.image,(160, 80))
            self.img_resized = self.img_resized[None,:,:,:]
            logger.debug(f"resized input shape: {self.img_resized.shape}")
        
        ## Lan line generation
        if self.generate:
            generator = line.LineGenerator(model_unet)
            lines = generator.generate(self.image)
            lines_rgb = generator.stack_rgb()
            lines_rgb_resized = cv2.resize(lines_rgb,(self.image.shape[1],
                                                      self.image.shape[0]))
            generation_processor.moving_average(lines)
            generation_processor.crop()
            generation_processor.quantise(self.confidence)
            generation_processor.resize(self.image.shape)
            generation_average = np.dstack((generation_processor.average,
                                             generation_processor.average,
                                             generation_processor.average))
            generation_average = generation_average[None,:,:,:]
            logger.debug(f"lines shape: {generation_average.shape}")
            logger.debug(f"resized shape: {generator.img_resized.shape}")
            ## Overlay generated lines
            self.img_resized = cv2.addWeighted(generator.img_resized.shape,1,
                                               generation_average,1,
                                               0, dtype=cv2.CV_32F)
            logger.debug(f"lines rgb resized shape: {lines_rgb_resized.shape}")
            self.image = cv2.addWeighted(self.image,1,
                                         lines_rgb_resized,0.75,
                                         0, dtype=cv2.CV_32F)
        ## ==============================
        # ## Generate lane lines
        # if self.generate:
        #     generator = line.LineGenerator(model_unet)
        #     # generator.generate(self.image)
        #     # generator.resize()
        #     # generator.stack_rgb()
        #     ## Recent generation
        #     lines = generator.generate(self.image)
        #     self.generation_recent.append(lines)
        #     ## Moving average generation
        #     if len(self.generation_recent) > 5:
        #         self.generation_recent = self.generation_recent[1:]
        #     self.generation_average = \
        #         np.mean(np.array([i for i in self.generation_recent]), 
        #                 axis = 0)
        #     ## Quantise generation

        #     # self.lines = generator.prediction_rgb
        #     # self.lines_resized = generator.prediction_resized_rgb
        #     # ## Add axis
        #     # self.lines = self.lines[None,:,:,:]
        #     # logger.debug(f"generated image shape: {self.lines.shape}")
        #     # logger.debug(f"resized generated image shape: {self.lines_resized.shape}")
        #     # ## Overlay generated lines (visualised output is not affected)
        #     # self.img_resized = cv2.addWeighted(generator.img_resized,1,
        #     #                                    self.lines,1,
        #     #                                    0, dtype=cv2.CV_32F)
        #     # self.image = cv2.addWeighted(self.image,1,
        #     #                              self.lines_resized,1,
        #     #                              0, dtype=cv2.CV_32F)
        ## ==============================
        ## Lane detection
        prediction = model_fcnn.predict(self.img_resized)[0] * 255
        logger.debug(f"predection max: {np.max(prediction)}")
        prediction_processor.moving_average(prediction)
        prediction_processor.crop()
        prediction_processor.quantise(self.confidence)
        prediction_processor.resize(self.image.shape)
        blanks = np.zeros_like(prediction_processor.resized).astype(np.uint8)
        self.prediction_average = prediction_processor.average
        self.predection_rgb = np.dstack((blanks,
                                         prediction_processor.resized,
                                         blanks))
        ## ============================== 
        # ## Recent detection
        # self.predection_recent.append(prediction)
        # ## Moving average detection
        # if len(self.predection_recent) > 5:
        #     self.predection_recent = self.predection_recent[1:]
        # self.prediction_average = \
        #     np.mean(np.array([i for i in self.predection_recent]), axis = 0)
        # logger.debug(f"predection_average max: {np.max(self.prediction_average)}, shape: {self.prediction_average.shape}")
        # ## Crop detection
        # self.prediction_average[:40,:] = \
        #     np.zeros_like(self.prediction_average[:40,:])
        # self.prediction_average[60:,:] = \
        #     np.zeros_like(self.prediction_average[60:,:])
        # ## Quantise detection
        # self.prediction_average = \
        #     np.digitize(self.prediction_average,
        #                 [self.confidence*np.max(self.prediction_average)])*255.0
        # logger.debug(f"self.prediction_average max after quantising: {np.max(self.prediction_average)}")
        # ## Resize
        # predection_resized = \
        #     cv2.resize(self.prediction_average,(self.image.shape[1], 
        #                                         self.image.shape[0]))
        # logger.debug(f"predection_resized max: {np.max(predection_resized)}")
        ## Create RGB image
        # blanks = np.zeros_like(predection_resized).astype(np.uint8)
        # self.predection_rgb = np.dstack((blanks, predection_resized, blanks))
        ## ==============================
        ## Dump detection
        if self.dumpframes == True:
            dir_dump = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Model Outputs\\framedump"
            cv2.imwrite(f"{dir_dump}\\{self.framecount}-1_average.png",self.prediction_average)
            cv2.imwrite(f"{dir_dump}\\{self.framecount}-2_resized.png",prediction_processor.resized)
            self.framecount += 1

    def overlay_lane(self, image:np.ndarray):
        """Overlay detection onto the image"""
        self.image = image
        self.detect()
        self.result = cv2.addWeighted(self.image,1,
                                      self.predection_rgb,1,
                                      0, dtype=cv2.CV_32F)
        self.result = np.clip(self.result,None,255)
        ## Dump result
        if self.dumpframes == True:
            dir_dump = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Model Outputs\\framedump"
            cv2.imwrite(f"{dir_dump}\\{self.framecount-1}-3_laneimage.png",self.result[:,:,::-1])
        return self.result
    
    def replace_lane(self, image:np.ndarray):
        """Replace image with lane detection mask"""
        self.image = image
        self.detect()
        return self.predection_rgb
    
    def overlay_all(self, image:np.ndarray):
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


class Postprocessor:
    def __init__(self, 
                 average_count:int=5):
        """Initialise postprocessor class for taking moving average, cropping,
        and upscaling binary mask of 0-255 range"""
        self.average_count = average_count
        self.recent = []
        
    def moving_average(self, mask:np.ndarray):
        """Take moving average of specified number of masks"""
        self.recent.append(mask)
        logger.debug(f"mask max: {np.max(mask)}")
        logger.debug(f"recents lenght: {len(self.recent)}")
        if len(self.recent) > self.average_count:
            self.recent = self.recent[1:]
        self.average = \
            np.mean(np.array([i for i in self.recent]), axis = 0)
        logger.debug(f"average max: {np.max(self.average)}, shape: {self.average.shape}")

    def crop(self, bottom:int=40, top:int=60):
        """Set elements below the specified bottom row and above the 
        specified top row to 0"""
        self.average[:bottom,:] = \
            np.zeros_like(self.average[:bottom,:])
        self.average[top:,:] = \
            np.zeros_like(self.average[top:,:])
        
    def quantise(self, confidence:float=0.5):
        """Quantise mask to 0 and 1"""
        self.average = \
            np.digitize(self.average,
                        [confidence*np.max(self.average)])*255.0
        logger.debug(f"average max after quantising: {np.max(self.average)}")

    def resize(self, resize_dimensions:tuple=None):
        """Resize mask"""
        self.resize_width = resize_dimensions[1]
        self.resize_height = resize_dimensions[0]
        self.resized = \
            cv2.resize(self.average,(self.resize_width,self.resize_height))
        logger.debug(f"resized max: {np.max(self.resized)}")
        


## Set up the logger
logging.basicConfig(format="[%(name)s][%(levelname)s] %(message)s")
logger = logging.getLogger("corner-detection")
## Script
if __name__ == "__main__":
    # logger.setLevel(logging.DEBUG)
    # line.logger.setLevel(logging.DEBUG)
    ## Set I/O
    dir_model_fcnn = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Models\\FCNN"
    dir_model_unet = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Models\\UNet\\Combined"
    # dir_input = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Datasets\\video-footage\\local-dashcam"
    dir_input = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Datasets\\video-footage"
    dir_output = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Model Outputs"
    vid_name = "techodyssey_1080p.MP4"
    ## Original model
    # confidence = 0.05
    # model_fcnn = keras.models.load_model(f"{dir_model_fcnn}\\MLND-Capstone\\full_CNN_model.h5")
    # label = f"originalmodel_{confidence}"
    ## Duct tape fixed model trained on custom data set of 250 images
    # model_version = 0
    # confidence = 0.45
    ## Fixed model trained on custom data set of 250 images
    model_version = 3
    confidence = 0.35     ## optimal threshold for daytime
    ## Complie separately to fix weight_decay typeerror
    model_fcnn = keras.models.load_model(f"{dir_model_fcnn}\\test_model_v{model_version}.h5", compile=False)
    model_fcnn.compile(optimizer='Adam', loss='mean_squared_error')
    model_unet = keras.models.load_model(f"{dir_model_unet}\\combined_model.h5", compile=False)
    model_unet.compile(optimizer='adam', 
                       loss=keras.losses.BinaryFocalCrossentropy())
    label = f"fcnn-{model_version}_conf-{confidence}_unet"
    ## Perform detection
    generation_processor = Postprocessor()
    prediction_processor = Postprocessor()
    detector = Detector(confidence=confidence,
                        runretinex=False,           ## Retinex enhancement
                        retinex_low_percent=50,
                        generate=True)             ## Generate lane lines
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
        
    # detect_clip(3,6)
    # detect_full()
    # detectncalculate_clip(14,14.17)
    detectncalculate_full()