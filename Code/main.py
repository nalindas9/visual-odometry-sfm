
import cv2
import glob
import random
from MyUtils import *
import ReadCameraModel as r
import matplotlib.pyplot as plt
import os
random.seed(1)


path = "/home/aditya/PycharmProjects/VisualOdometry/Undistorted2"
camera_path = "/home/aditya/Oxford_dataset/model"
fx, fy, cx, cy, G_camera_img, LUT = r.ReadCameraModel(camera_path)

camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
list_fname = [fname for fname in os.listdir(path)]
list_fname.sort()
dataset_size = len(list_fname)

p0 = np.array([0, 0, 0, 1])
H = np.eye(4)
#

for index in range(20, dataset_size-1):
    fname1 = path + "/" + list_fname[index]
    fname2 = path + "/" + list_fname[index+1]
    img1 = cv2.imread(fname1, 0)
    img2 = cv2.imread(fname2, 0)

    # cv2.imshow("First Frame: ", img1)
    # cv2.imshow("Second Frame: ", img2)

    img1 = img1[200:640, 0:1280]
    img2 = img2[200:640, 0:1280]

    print("Frame: ", index)
    features2d = getFeatureMatches(img1, img2)
    left_features = features2d[0]
    right_features = features2d[1]
    print("Left Feature size: {}; Right Feature size: {} ".format(left_features.shape, right_features.shape))
    FundMatrix, Left_inlier, Right_inlier = getInlierRANSAC(left_features, right_features)
    E_matrix = getEssentialMatrix(camera_matrix, FundMatrix)
    cam_center, cam_rotation = ExtractCameraPose(E_matrix)
    T, R = getDisambiguousPose(camera_matrix, cam_center, cam_rotation, Left_inlier, Right_inlier)
    # T, R = DisambiguousPose(camera_matrix, cam_center, cam_rotation, Left_inlier, Right_inlier)
    Homogeneous_matrix = np.vstack((np.column_stack((R, T)), np.array([0, 0, 0, 1])))

    H = H @ Homogeneous_matrix
    projection = H @ p0

    x = projection[0]
    y = projection[2]
    print(" X - ", x)
    print(" Y - ", y)
    if 3159 <= index:
        x = x*0.1
    plt.scatter(-x, y, color='r')
    plt.savefig(r"/home/aditya/PycharmProjects/VisualOdometry/op2/frame" + str(index) + ".png")
    # plt.pause(0.1)

cv2.destroyAllWindows()