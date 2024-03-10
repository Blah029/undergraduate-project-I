"""Redraw dashed or broken lane lines using a U-Net model"""
import cv2
import keras
import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class LineGenerator:
    def __init__ (self, model):
        """Initialise LineGenerator class"""
        self.model = model
        self.prediction = None
        self.prediction_resized = None

    def generate(self, image:np.ndarray):
        """Generate lane line mask for 160x80 image using unet"""
        ## Strip alpha channel, resize image to 160x80, and add axis
        if image.shape != (1, 80, 160, 3):
            self.image = image[:,:,:3]
            self.img_resized = cv2.resize(self.image,(160,80))
            self.img_resized = self.img_resized[None,:,:,:]
            logger.debug(f"image read: {self.image.shape}")
            logger.debug(f"image resized: {self.img_resized.shape}")
            logger.debug(f"added new axis: {self.img_resized.shape}")
        else:
            self.image = image
            self.img_resized = image
            logger.debug(f"image already processed to shape {image.shape}")
        self.prediction = self.model.predict(self.img_resized)
        ## Remove negative values and quantise
        self.prediction[self.prediction < 0] = 0.0
        self.prediction = (self.prediction >= 0.5).astype('uint8')
        self.prediction = self.prediction.squeeze()*255
        logger.debug(f"mask shape: {self.prediction.shape}")
        self.output_mask = self.prediction
        return self.output_mask

    def resize(self):
        """Rescale the output mask to match input resolution"""
        if self.image.shape != (1, 80, 160, 3):
            self.prediction_resized = cv2.resize(self.prediction, 
                                                    (self.image.shape[1],
                                                    self.image.shape[0]))
        else:
            self.prediction_resized = self.prediction
        logger.debug(f"resized shape: {self.prediction_resized.shape}")
        self.output_mask = self.prediction_resized
        return self.output_mask

    def stack_rgb(self, stack_160x80:bool=True, stack_resized:bool=True):
        """Depth stack output mask to create RGB image"""
        if stack_160x80:
            self.prediction_rgb = np.dstack((self.prediction,
                                             self.prediction,
                                             self.prediction))
            logger.debug(f"rgb mask shape: {self.prediction_rgb.shape}")
            self.output_mask = self.prediction_rgb
        if stack_resized and self.prediction_resized is not None:
            self.prediction_resized_rgb = np.dstack((self.prediction_resized,
                                                     self.prediction_resized,
                                                     self.prediction_resized))
            logger.debug(f"resized rgb mask shape: {self.prediction_resized_rgb.shape}")
            self.output_mask = self.prediction_resized_rgb
        return self.output_mask
    
    
## Set up the logger
logging.basicConfig(format="[%(name)s][%(levelname)s] %(message)s")
logger = logging.getLogger("line-generation")
## Script
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    ## Directory paths
    dir_model = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Models\\UNet"
    dir_input = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Datasets\\image-footage"
    # dir_input = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Datasets\\image-footage\\CULane Selected\\Image-Label Couples\\images"
    dir_output = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Model Outputs\\UNet"
    image = cv2.imread(f"{dir_input}\\techodyssey_1080p_1.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ## Load model
    # model = tf.keras.models.load_model(f"{dir_model}\\new_model.h5")
    model = tf.keras.models.load_model(f"{dir_model}\\Combined\\combined_model.h5", compile=False)
    model.compile(optimizer='adam', 
                  loss=keras.losses.BinaryFocalCrossentropy(), 
                  metrics=['accuracy'])
    generator = LineGenerator(model)
    generator.generate(image)
    generator.resize()
    generator.stack_rgb()
    fig, ax = plt.subplots(2,1)
    ax[0].imshow(generator.image)
    ax[0].set_title("Image")
    ax[1].imshow(generator.output_mask)
    ax[1].set_title("Label")
    fig.tight_layout()
    plt.show()


