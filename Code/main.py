# import DataPreprocess as pp
import cv2
import glob
import numpy as np
from cv2 import xfeatures2d
import random
random.seed(1)
# processed = pp.undistorted
# cam_model = pp.CameraModel

path = "/home/aditya/PycharmProjects/VisualOdometry/Undistorted/*"


# processed = []
# for fname in sorted(glob.glob(path)):
#     print(fname.split("/")[-1])
#     img = cv2.imread(fname)
#     processed.append(img)


def GetFeatureMatches(img1, img2):
    """
    Uses keypoint algorithm to extract feature points from the image and get point correspondences
    :param: img1 (Left image), img2 (Right image)
    :return: ndarray of (x,y)L and (x,y)R feature points
    """
    orb = xfeatures2d.SIFT_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, 2)
    Left_Pts = list()
    Right_Pts = list()

    # Ratio criteria according to Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            Left_Pts.append(kp2[m.trainIdx].pt)
            Right_Pts.append(kp1[m.queryIdx].pt)

    left = np.int32(Left_Pts)
    right = np.int32(Right_Pts)
    features = (left, right)
    return features


def ComputeFundamentalMatrix(pts1, pts2):
    """
    This function computes the fundamental matrix by computing the SVD of Ax = 0
    :param pts1: ndarray left feature points
    :param pts2: ndarray right feature points
    :return: F(3x3) matrix of rank 2
    """
    x1 = pts1[:, 0]
    y1 = pts1[:, 1]
    x2 = pts2[:, 0]
    y2 = pts2[:, 1]
    n = x1.shape[0]
    A = np.zeros((n, 9))
    for i in range(0, n):
        A[i] = [x1[i] * x2[i], x1[i] * y2[i], x1[i],
                y1[i] * x2[i], y1[i] * y2[i], y1[i],
                x2[i], y2[i], 1]
    # Compute F matrix by evaluating SVD
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # Constrain the F matrix to rank 2
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    return F / F[2, 2]


def GetBestfitFundamentalMatrix(pts1, pts2):
    """
    Leverages the 8-point algorithm and implement RANSAC algorithm to find the best F matrix
    :param pts1: ndarray of left features
    :param pts2: ndarray of right features
    :return: F(3x3) matrix
    """
    # global finalFundamentalMatrix
    n = 0
    iterations = 500
    threshold = 0.05
    max_count = 0
    for i in range(0, iterations):
        count = 0
        idx = random.sample(range(8), 8)
        left_pts = pts1[idx]
        right_pts = pts2[idx]
        F = ComputeFundamentalMatrix(left_pts, right_pts)
        left_feature_inlier = []
        right_feature_inlier = []
        for j in range(0, len(idx)):
            homogeneous_right = np.array([right_pts[j, 0], right_pts[j, 1], 1])
            homogeneous_left = np.array([left_pts[j, 0], left_pts[j, 1], 1])
            fit = np.dot(homogeneous_right.T, np.dot(F, homogeneous_left))
            if fit < threshold:
                left_feature_inlier.append(left_pts)
                right_feature_inlier.append(right_pts)
                count += 1
        print('Inlier count', count)
        if count > max_count:
            max_count = count
            finalFundamentalMatrix = F
            inlier_Left = left_feature_inlier
            inlier_Right = right_feature_inlier
    return finalFundamentalMatrix


def GetEssentialMatrix(K, F):
    """
    This function computes the essential matrix from the fundamental matrix. The E matrix is defined
    in normalized image coordinates
    :param K: camera calibration matrix
    :param F: best fitted Fundamental matrix
    :return: Essential Matrix
    """
    E = np.dot(K.T, np.dot(F, K))
    u, s, v = np.linalg.svd(E)

    # We correct the singular values of the E matrix
    s_new = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]).reshape(3,3)
    final_E = np.dot(u, np.dot(s_new, v.T))
    return final_E


def ExtractCameraPose(E):
    """
    Given the essential matrix, we derive the camera position and orientation
    :param E: Essential Matrix (3x3)
    :return: list(rotations), list(position)
    """
    u, s, v = np.linalg.svd(E)
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).reshape(3,3)
    c1 = u[:, 2]
    r1 = np.dot(u, np.dot(w, v.T))
    c2 = u[:, 2]
    r2 = np.dot(u, np.dot(w, v.T))
    c3 = u[:, 2]
    r3 = np.dot(u, np.dot(w.T, v.T))
    c4 = -u[:, 2]
    r4 = np.dot(u, np.dot(w.T, v.T))
    if np.linalg.det(r1) < 0:
        c1 = -c1
        r1 = -r1
    if np.linalg.det(r2) < 0:
        c2 = -c2
        r2 = -r2
    if np.linalg.det(r3) < 0:
        c3 = -c3
        r3 = -r3
    if np.linalg.det(r4) < 0:
        c4 = -c4
        r4 = -r4
    cam_pose = ([c1, c2, c3, c4], [r1, r2, r3, r4])
    return cam_pose



img1 = cv2.imread("/home/aditya/PycharmProjects/VisualOdometry/Undistorted/undistorted_1399381446204705.png")
img2 = cv2.imread("/home/aditya/PycharmProjects/VisualOdometry/Undistorted/undistorted_1399381446704623.png")
features2d = GetFeatureMatches(img1, img2)
left_features = features2d[0]
right_features = features2d[1]
print('Match points:', left_features[:8])
F = ComputeFundamentalMatrix(left_features[:8], right_features[:8])
print(np.linalg.matrix_rank(F))
idx = random.sample(range(8), 8)
print(idx)
print(left_features[idx], right_features[idx])
F2 = ComputeFundamentalMatrix(left_features[idx], right_features[idx])
print(F2)
left = left_features[idx]
right = right_features[idx]
nleft = np.array([left[0,0], left[0,1], 1])
nright = np.array([right[0,0], right[0,1], 1])
print(nleft, nright)
fit = np.dot(nright.T, np.dot(F, nleft))
print(fit)
Ff = GetBestfitFundamentalMatrix(left_features, right_features)
print(Ff)
# img1 = cv2.drawKeypoints(img1, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# print(len(kp))
# cv2.imshow("First Ref", img1)
# cv2.imshow("Second ref", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
