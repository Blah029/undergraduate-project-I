"""Redraw dashed or broken lane lines using a U-Net model"""
import cv2
import keras
import logging
import numpy as np
import tensorflow as tf


class LineGenerator:
    def __init__ (self, model):
        """Initialise LineGenerator class"""
        self.model = model

    def generate(self, image:np.ndarray):
        ## Strip alpha channel, resize image to 320x256, and add axis
        self.image = image[:,:,:3]
        self.img_resized = cv2.resize(self.image,(320,256))
        self.img_resized = self.img_resized[None,:,:,:]
        logger.debug(f"added new axis: {self.img_resized.shape}")
        self.img_resized = self.img_resized.reshape(1,256,320,3)
        logger.debug(f"rehsaped: {self.img_resized.shape}")
        prediction = self.model.predict(image)
        ## Remove negative values and quantise
        prediction[prediction < 0] = 0.0
        prediction = (prediction >= 0.5).astype('int')
        prediction = prediction.squeeze()
        self.prediction_resized = cv2.resize(prediction,)
        return self.prediction


## Set up the logger
logging.basicConfig(format="[%(name)s][%(levelname)s] %(message)s")
logger = logging.getLogger("line-generation")
## Script
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    ## Directory paths
    dir_model = r"C:\Users\User Files\Documents\University\Misc\4th Year Work\Final Year Project\Models\UNet"
    dir_input = r"C:\Users\User Files\Documents\University\Misc\4th Year Work\Final Year Project\Datasets\image-footage"
    dir_output = r"C:\Users\User Files\Documents\University\Misc\4th Year Work\Final Year Project\Outputs\Model Outputs\UNet"
    ## Load model
    # model = tf.keras.models.load_model(f"{dir_model}\\new_model.h5")
    model = tf.keras.models.load_model(f"{dir_model}\\new_model.h5", compile=False)
    model.compile(optimizer='adam', loss=keras.losses.BinaryFocalCrossentropy(), metrics=['accuracy'])
    generator = LineGenerator(model)


