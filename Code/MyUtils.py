import numpy as np
from cv2 import xfeatures2d
import random
import cv2

#### THIS CODE FILE CONTAINS THE UTILITIES TO IMPLEMENT VISUAL ODOMETRY ####


def getFeatureMatches(img1, img2):
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
        if m.distance < 0.5 * n.distance:
            Left_Pts.append(kp2[m.trainIdx].pt)
            Right_Pts.append(kp1[m.queryIdx].pt)

    left = np.int32(Left_Pts)
    right = np.int32(Right_Pts)
    features = (left, right)
    return features


def computeFundamentalMatrix(pts1, pts2):
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
    # print('Old S', S)
    # S[2] = 0
    S2 = np.array([[S[0], 0, 0], [0, S[1], 0], [0, 0, 0]])
    # print('New S', S)
    F = np.dot(np.dot(U, S2), V)

    return F


def getInlierRANSAC(pts1, pts2):
    """
    Leverages the 8-point algorithm and implement RANSAC algorithm to find the inliers
    :param pts1: ndarray of left features
    :param pts2: ndarray of right features
    :return: left inliers and Right inliers
    """
    # global finalFundamentalMatrix
    iterations = 50
    threshold = 0.01
    max_count = 0
    n = len(pts1)
    for i in range(0, iterations):
        count = 0
        idx = random.sample(range(n), 8)
        left_pts = pts1[idx]
        right_pts = pts2[idx]
        F = computeFundamentalMatrix(left_pts, right_pts)
        left_feature_inlier = []
        right_feature_inlier = []
        # print("Sample index: ", len(idx))
        for j in range(0, n):
            homogeneous_right = np.array([pts2[j, 0], pts2[j, 1], 1])
            homogeneous_left = np.array([pts1[j, 0], pts1[j, 1], 1])
            fit = np.dot(homogeneous_right.T, np.dot(F, homogeneous_left))
            # print("Fit for iteration ", i," ", np.abs(fit))
            if np.abs(fit) < threshold:
                left_feature_inlier.append(pts1[j])
                right_feature_inlier.append(pts2[j])
                count += 1
        # print('Inlier count', count)
        inlier_Left = np.array(left_feature_inlier)
        inlier_Right = np.array(right_feature_inlier)
        # betterF = computeFundamentalMatrix(inlier_Left, inlier_Right)
        if count > max_count:
            max_count = count
            finalFundamentalMatrix = F
            final_inlier_Left = inlier_Left
            final_inlier_Right = inlier_Right
    return final_inlier_Left, final_inlier_Right, finalFundamentalMatrix


def getEssentialMatrix(K, F):
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
    final_E = np.dot(u, np.dot(s_new, v))
    return final_E


def ExtractCameraPose(E):
    """
    Given the essential matrix, we derive the camera position and orientation
    :param E: Essential Matrix (3x3)
    :return: list(rotations), list(position)
    """
    u, s, v = np.linalg.svd(E, full_matrices=True)
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).reshape(3, 3)
    c1 = u[:, 2].reshape(3, 1)
    r1 = np.dot(np.dot(u, w), v).reshape(3, 3)
    c2 = -u[:, 2].reshape(3, 1)
    r2 = np.dot(np.dot(u, w), v).reshape(3, 3)
    c3 = u[:, 2].reshape(3, 1)
    r3 = np.dot(np.dot(u, w.T), v).reshape(3, 3)
    c4 = -u[:, 2].reshape(3, 1)
    r4 = np.dot(np.dot(u, w.T), v).reshape(3, 3)
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
    cam_center = np.array([c1, c2, c3, c4])
    cam_rotation = np.array([r1, r2, r3, r4])
    return cam_center, cam_rotation


def getTriangulationPoint(K, M, left_point, right_point):
    """
    Find triangulation point
    :param K: camera intrinsic parameter
    :param M: camera extrinsic parameter
    :param pts1: left image point
    :param pts2: right image point
    :return: P (point in 3D plane)
    """
    R_origin = np.eye(3)
    C_origin = np.array([[0],[0],[0]])

    Cam_Origin = K @ np.hstack((R_origin, -R_origin @ C_origin))

    m3 = Cam_Origin[2, :]
    m1 = Cam_Origin[0, :]
    m2 = Cam_Origin[1, :]
    m3_dash = M[2, :]
    m1_dash = M[0, :]
    m2_dash = M[1, :]
    x, y = left_point[0], left_point[1]
    x_dash, y_dash = right_point[0], right_point[1]
    a1 = x*m3 - m1
    a2 = y*m3 - m2
    a3 = x_dash*m3_dash - m1_dash
    a4 = y_dash*m3_dash - m2_dash
    A = np.vstack((a1, a2, a3, a4))
    # print(A.shape)
    u, s, v = np.linalg.svd(A)

    X = v[3]
    X = X/X[-1]

    # print(X)
    return X.reshape((4, 1))


def getExtrinsicParameter(K, R, C):
    """
    This function returns the extrinsic parameter matrix
    :param K: Camera Intrinsic Matrix
    :param R: Rotation matrix 3x3
    :param C: Camera Center vector 3x1
    :return: 4x4 Extrinsic parameter
    """
    t = np.dot(-R, C)
    homogeneous_matrix = np.hstack((R.reshape(3, 3), t))
    extrinsic_parameter = np.dot(K, homogeneous_matrix)
    return extrinsic_parameter


def getDisambiguousPose(K, C, R, left_features, right_features):
    """
    Gets the translation vector and rotation matrix of the camera
    :param K: Camera intrinsic matrix
    :param C: Camera center ndarray
    :param R: Camera Rotation Matrix ndarray
    :param left_features: feature points from left image
    :param right_features: feature points from right image
    :return: Position, Rotation
    """
    check = 0
    for i in range(0, len(R)):
        count = 0
        # extrinsic_params = np.vstack((np.hstack((R[i].reshape(3, 3), C[i])), np.array([0, 0, 0, 1])))
        extrinsic_params = getExtrinsicParameter(K, R[i], C[i])
        for j in range(0, len(left_features)):
            X = getTriangulationPoint(K, extrinsic_params, left_features[j], right_features[j])
            r3 = R[i][2, :].reshape((1, 3))
            cheiralityCondition = np.dot(r3, X[:3] - C[i])
            if cheiralityCondition > 0 and X[2]>0:
                count += 1
        if count > check:
            check = count
            Translation = C[i]
            Rotation = R[i]
    if Translation[2] < 0:
        Translation = -Translation
    return Translation, Rotation
