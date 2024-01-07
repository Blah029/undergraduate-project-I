# undergraduate-project-I

Final year EE405: Undergraduate Project I  

## Stage 1:

This project presents an innovative, deep learning-based approach for enhancing road safety by leveraging a Fully Convolutional Neural Network (FCNN). This self-deployable Advanced Driver Assistance System (ADAS) system detects and predicts lane departure using data from a dashboard camera. The FCNN efficiently detects the driveable area in real-time and the novel departure detection algorithm predicts lane departure, providing timely alerts to drivers about significant deviations from their intended trajectory. Robustness and generalisation are achieved through diverse and extensive dataset training, encompassing various road conditions. By focusing on real-time performance and user-friendly alerts, the proposed system aims to prevent accidents caused by lane departure. The proposed system holds significant potential in enhancing road safety by providing drivers with a self-deployable ADAS for vehicles that are not equipped with one by the manufacturer.

## Stage 2:

This project presents an innovative, deep learning-based approach for enhancing the lane detection performance of semantic segmentation models by leveraging Generative Adversarial Networks (GAN) to increase the visibility of lane lines in three key scenarios; missing or broken lane lines, night time, and rainy daytime by leveraging the robustness of the U-Net architecture. The proposed GAN was found to be more effective and computationally efficient than using Retinex for improving low light lane images and using filters for rain removal. The generalisation was achieved through combining diverse and extensive datasets for training, encompassing the key driving scenarios. By focusing on fast real-time performance, the proposed system aims to vastly improve the lane recognition ability of semantic segmentation models, and holds significant potential in the field of Advanced Driver Assistance Systems.

## Group Members

  - E/17/099
  - E/17/180
  - E/17/371

## Files

- jupyter
  - pickle_datset_processing.ipynb  
    Compile black and white labels from CVAT masks
  - pickle_datset_exporting.ipynb  
    Generate pickle files from images and labels
  - fcnn_model_training_v2.ipynb  
    Train semantic FCNN model for semantic segmentation
  - unet_night_model_training.ipynb  
    Train and test U-Net model for night time lane detection improvement
  - unet_rain_model_testing.ipynb  
    Test U-Net model for night time lane detection improvement
  - unet_rain_model_training.ipynb  
    Train U-Net model for lane detection improvement in rain
  - single_frame_detevtion.ipynb  
    Load FCNN model and perform semantic segmentation for a single frame
  - utils.ipynb  
    Utility functions use throughout the project
<br><br>
- python
  - lane_detection.py  
    Load FCNN model and perform semantic segmentation
  - corner_detection.py  
    Detect cornercoordinates of the segementation mask created by the model
  - angle_calculation.py  
    Calculation and angle deviation and the position offset using the detected coordinates


### To-do

- Convert and add colab notebooks

