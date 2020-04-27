import numpy as np
import cv2
import glob
import ReadCameraModel as r
import UndistortImage as udi

file_path = "/home/aditya/Oxford_dataset/stereo/centre/*"
camera_path = "/home/aditya/Oxford_dataset/model"

undistorted = []
fx, fy, cx, cy, G_camera_img, LUT = r.ReadCameraModel(camera_path)

for fname in sorted(glob.glob(file_path)):
    print(fname)
    raw = cv2.imread(fname, 0)
    color_img = cv2.cvtColor(raw, cv2.COLOR_BayerGR2BGR)
    undistort_img = udi.UndistortImage(color_img, LUT)
    img = cv2.resize(undistort_img, (0,0), fx=0.5, fy=0.5)
    undistorted.append(img)

CameraModel = [fx, fy, cx, cy, G_camera_img, LUT]
