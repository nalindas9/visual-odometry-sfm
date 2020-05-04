import numpy as np
import cv2
import glob
import ReadCameraModel as r
import UndistortImage as udi
import os

file_path = "/home/aditya/Oxford_dataset/stereo/centre/*"
camera_path = "/home/aditya/Oxford_dataset/model"
write_path = "/home/aditya/PycharmProjects/VisualOdometry/Undistorted"


fx, fy, cx, cy, G_camera_img, LUT = r.ReadCameraModel(camera_path)
Frame = 0
print("Images are loading.. ")
for fname in sorted(glob.glob(file_path)):
    img_name = fname.split("/")[-1]
    print("Loading:", img_name)
    raw = cv2.imread(fname, 0)
    color_img = cv2.cvtColor(raw, cv2.COLOR_BayerGR2BGR)
    undistort_img = udi.UndistortImage(color_img, LUT)
    write_name = "undistorted_"+img_name
    cv2.imwrite(os.path.join(write_path, write_name), undistort_img)
    Frame += 1
print("Images loaded !")

