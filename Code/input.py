# import numpy as np
# # import random
# import cv2
# import os

"""
This is the input module. We import this module in main.py and DataPreprocess.py
as 
" from input import * "

In DataPreprocess we use,
file_path
and write_path
In main.py we use,
camera_path
and write_path+"/*"
to read the camera model and the undistorted images 
"""

path = input("Enter the root directory of your dataset: []")
file_path = path + "/Oxford_dataset/stereo/centre/1399381446204705.png"
camera_path = path + "/Oxford_dataset/model"
print("Image Path: ", file_path)
print("camera model path: ", camera_path)

# This path must contain the path to save the undistorted images
write_path = path + "visual-odometry-sfm/Code/Undistorted"

# img = cv2.imread(file_path)
# name = "img1"
# cv2.imwrite(os.path.join(write_path, name), img)
