
import cv2
import glob
import random
from MyUtils import *
import ReadCameraModel as r
import matplotlib.pyplot as plt
import os
random.seed(1)


path = "/home/aditya/PycharmProjects/VisualOdometry/Undistorted"
camera_path = "/home/aditya/Oxford_dataset/model"
fx, fy, cx, cy, G_camera_img, LUT = r.ReadCameraModel(camera_path)

camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
list_fname = [fname for fname in os.listdir(path)]
list_fname.sort()
dataset_size = len(list_fname)
processed = []

# for index in range(0, dataset_size-1):
#     fname1 = path + "/" + list_fname[index]
#     fname2 = path + "/" + list_fname[index+1]
#     print("File name 1: ", fname1)
#     print("File name 2: ", fname2)
# print("Image Dataset Loaded!")
# print("Performing Visual Odometry!")
#
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
    print("Frame: ", index)
    features2d = getFeatureMatches(img1, img2)
    left_features = features2d[0]
    right_features = features2d[1]
    print("Left Feature size: {}; Right Feature size: {} ".format(left_features.shape, right_features.shape))
    Left_inlier, Right_inlier, FundMatrix = getInlierRANSAC(left_features, right_features)
    E_matrix = getEssentialMatrix(camera_matrix, FundMatrix)
    cam_center, cam_rotation = ExtractCameraPose(E_matrix)
    T, R = getDisambiguousPose(camera_matrix, cam_center, cam_rotation, Left_inlier, Right_inlier)
    Homogeneous_matrix = np.vstack((np.hstack((R, T)), np.array([0, 0, 0, 1])))

    H = H @ Homogeneous_matrix
    projection = H @ p0

    x = projection[0]
    y = projection[2]
    print(" X - ", x)
    print(" Y - ", y)
    plt.scatter(-x, y, color='r')
    plt.pause(0.5)
    # cv2.waitKey(0)
    # if cv2.waitKey(5) & 0xFF == 27:
    #     break
cv2.destroyAllWindows()

# for i in range(0, 10):
#     idx = random.sample(range(700), 8)
#     print(idx)

#
# img1 = cv2.imread("/home/aditya/PycharmProjects/VisualOdometry/Undistorted/undistorted_1399381446204705.png")
# img2 = cv2.imread("/home/aditya/PycharmProjects/VisualOdometry/Undistorted/undistorted_1399381446704623.png")
# features2d = getFeatureMatches(img1, img2)
# left_features = features2d[0]
# right_features = features2d[1]
# print('Match points:', left_features.shape[0])
# idx = random.sample(range(left_features.shape[0]), 8)
# F_init = computeFundamentalMatrix(left_features[idx], right_features[idx])
# print("Initial estimated fundamental matrix", F_init)
#
# Left_inlier, Right_inlier, FundMatrix = getInlierRANSAC(left_features, right_features)
# print('Left inlier', Left_inlier.shape, "right inlier", Right_inlier.shape, "Final Fundamental Matrix: ", FundMatrix)
#
# E_matrix = getEssentialMatrix(camera_matrix, FundMatrix)
# print("Essential Matrix", E_matrix)
# cam_center, cam_rotation = ExtractCameraPose(E_matrix)
# # # ex_param = np.vstack((np.hstack((cam_rotation[0], cam_center[0])),np.array([0, 0, 0, 1])))
# # # print("Extrinsic camera parameters", ex_param[2, :3])
# # # print(ex_param)
# H = getExtrinsicParameter(camera_matrix, cam_rotation[0], cam_center[0])
# print(H.shape)
# extrinsic_params = np.dot(camera_matrix, H)
# print(extrinsic_params)
# # X = getTriangulationPoint(extrinsic_params, Left_inlier[0], Right_inlier[0])
# # print('Triangulation: ', X[:3]-cam_center[0])
# # r3 = cam_rotation[0][2, :]
# # cc = np.dot(r3, X[:3]-cam_center[0])
# # print(cc)
# # if cc > 0:
# #     print("yes")
# T, R = getDisambiguousPose(camera_matrix, cam_center, cam_rotation, Left_inlier, Right_inlier)
# Homogeneous_matrix = np.vstack((np.hstack((R, T)), np.array([0, 0, 0, 1])))
# # print('H At 0: ', H)
# # # print("Translation: ", T)
# # # print("Rotation: ", R)
# # print("Homogeneous :", Homogeneous_matrix)
# # # print("Mat mul", np.dot(H,Homogeneous_matrix))
# H = H @ Homogeneous_matrix
# projection = H @ p0
# print(" X - ", projection[0])
# print(" Y - ", projection[1])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
