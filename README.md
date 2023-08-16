# undergraduate-project-I

Final year EE405: Undergraduate Project I  

This project presents an innovative, deep learning-based approach for enhancing road safety by leveraging a Fully Convolutional Neural Network (FCNN). This self-deployable Advanced Driver Assistance System (ADAS) system detects and predicts lane departure using data from a dashboard camera. The FCNN efficiently detects the driveable area in real-time and the novel departure detection algorithm predicts lane departure, providing timely alerts to drivers about significant deviations from their intended trajectory. Robustness and generalisation are achieved through diverse and extensive dataset training, encompassing various road conditions. By focusing on real-time performance and user-friendly alerts, the proposed system aims to prevent accidents caused by lane departure. The proposed system holds significant potential in enhancing road safety by providing drivers with a self-deployable ADAS for vehicles that are not equipped with one by the manufacturer. 

- Group G8
  - E/17/099 - Gunathilaka M.T.V.
  - E/17/180 - Kumarasinghe V.N.
  - [E/17/371 - Warnakulasuriya R.](https://sites.google.com/eng.pdn.ac.lk/ee405-g08-e17371/home)
<br><br>
- python
  - lane_detection.py  
    Load FCNN model and perform semantic segmentation
  - corner_detection.py  
    Detect cornercoordinates of the segementation mask created by the model
  - angle_calculation.py  
    Calculation and angle deviation and the position offset using the detected coordinates
- reference-codes  
  Open source codes used for reference and testing
<br><br>
- To-do
  - Convert and add colab notebooks

