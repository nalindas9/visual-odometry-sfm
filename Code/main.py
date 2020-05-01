# import DataPreprocess as pp
import cv2
import glob
import random
from MyUtils import *
import ReadCameraModel as r

random.seed(1)

# processed = pp.undistorted
# cam_model = pp.CameraModel

path = "/home/aditya/PycharmProjects/VisualOdometry/Undistorted/*"
camera_path = "/home/aditya/Oxford_dataset/model"
fx, fy, cx, cy, G_camera_img, LUT = r.ReadCameraModel(camera_path)

camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
# processed = []
# for fname in sorted(glob.glob(path)):
#     print(fname.split("/")[-1])
#     img = cv2.imread(fname)
#     processed.append(img)


img1 = cv2.imread("/home/aditya/PycharmProjects/VisualOdometry/Undistorted/undistorted_1399381446204705.png")
img2 = cv2.imread("/home/aditya/PycharmProjects/VisualOdometry/Undistorted/undistorted_1399381446704623.png")
features2d = getFeatureMatches(img1, img2)
left_features = features2d[0]
right_features = features2d[1]
print('Match points:', left_features.shape[0])
idx = random.sample(range(left_features.shape[0]), 8)
print(idx)
F = computeFundamentalMatrix(left_features[idx], right_features[idx])
print(F)
print(left_features[idx], right_features[idx])

Fit1 = []
Fit2 = []
for i in range(len(idx)):
    h_right = np.array([right_features[i, 0], right_features[i, 1], 1])
    h_left = np.array([left_features[i, 0], left_features[i, 1], 1])
    fit1 = np.squeeze(np.matmul((np.matmul(h_right, F)), h_left.T))
    fit2 = np.dot(h_right.T, np.dot(F, h_left))
    Fit1.append(fit1)
    Fit2.append(fit2)
# print(Fit1)
# print(Fit2)

Left_inlier, Right_inlier = getInlierRANSAC(left_features, right_features)
print('Left inlier', Left_inlier, "right inlier", Right_inlier)
newF = computeFundamentalMatrix(Left_inlier, Right_inlier)
print('Good Fundamental Matrix', newF)
Ess_mat = getEssentialMatrix(camera_matrix, newF)
print("Essential Matrix", Ess_mat)
cam_center, cam_rotation = ExtractCameraPose(Ess_mat)
# C1 = np.reshape(cam_center[0], (3, 1))
print("Camera center: ", len(cam_center))
print("Camera rotation: ", len(cam_rotation))

extrinsic_params = np.vstack((np.hstack((cam_rotation[0].reshape(3, 3), cam_center[0])), np.array([0, 0, 0, 1])))
print(extrinsic_params[:3, 2])
P = np.eye(4)
print(P)
print(P[0:3, 2])
print(len(Left_inlier))
# A = []
# for i in range(Left_inlier.shape[0]):
#     A.append(x[i]*extrinsic_params[:3, 2])
# print(A)
A = getTriangulationPoint(extrinsic_params, Left_inlier[0], Right_inlier[0])
r3 = cam_rotation[0][2, :]
print(cam_rotation[0])
print(np.dot(r3, A - cam_center[0]))
T, R = getDisambiguousPose(cam_center, cam_rotation, Left_inlier, Right_inlier)
print(T, R)
cv2.waitKey(0)
cv2.destroyAllWindows()
