"""
Visual Odomotry SFM

Authors:
Nalin Das (nalindas9@gmail.com)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""
import numpy as np
import os
import glob
import cv2
import ReadCameraModel
import UndistortImage
from matplotlib import pyplot as plt
from cv2 import xfeatures2d 
import random

# Specify the path for all the video frames here
IMAGES_PATH = "/home/nalindas9/Documents/courses/spring_2020/enpm673-perception/datasets/Oxford_dataset/stereo/centre"
# Specify the path for the camera parameters
MODELS_PATH = "/home/nalindas9/Documents/courses/spring_2020/enpm673-perception/datasets/Oxford_dataset/model"

# Function to find point correspondences between two successive frames
def featurematch(frame1, frame2):
  # Using SIFT to find keypoints and descriptors
  orb = xfeatures2d.SIFT_create()
  kp1, des1 = orb.detectAndCompute(frame1,None)
  kp2, des2 = orb.detectAndCompute(frame2,None)
  flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
  matches = flann.knnMatch(des1, des2, 2)
  left_pts = list()
  right_pts = list()
  
  # Need to draw only good matches, so create a mask
  matchesMask = [[0,0] for i in range(len(matches))]
  # Ratio criteria according to Lowe's paper
  for i, (m, n) in enumerate(matches):
    if m.distance < 0.75 * n.distance:
      left_pts.append(kp2[m.trainIdx].pt)
      right_pts.append(kp1[m.queryIdx].pt)
      matchesMask[i]=[1,0]
  
  draw_params = dict(matchColor = (0,255,0),
                     singlePointColor = (255,0,0),
                     matchesMask = matchesMask,
                     flags = 0)

  img3 = cv2.drawMatchesKnn(frame1,kp1,frame2,kp2,matches,None,**draw_params)

  #plt.imshow(img3,),plt.show()
  left_pts = np.array(left_pts)
  right_pts = np.array(right_pts)
  return left_pts, right_pts
  
# Function to compute fundamental matrix
def fundamental_matrix(pts1, pts2):
  x1 = pts1[:, 0]    
  y1 = pts1[:, 1]
  x2 = pts2[:, 0]
  y2 = pts2[:, 1]
  n = x1.shape[0]
  A = np.zeros((n, 9))
  # Find A matrix
  for i in range(0,n):
    A[i] = [x1[i] * x2[i], x1[i] * y2[i], x1[i], y1[i] * x2[i], y1[i] * y2[i], y1[i], x2[i], y2[i], 1]
  # Compute the F matrix by calculating the SVD
  U, S, Vh = np.linalg.svd(A)
  F = Vh[:, -1].reshape(3,3)
  # Impose Rank 2 constraint on the fundamental matrix
  U, S, Vh = np.linalg.svd(F)
  S[2] = 0
  F = np.dot(U, np.dot(np.diag(S),Vh))
  F = F/F[2,2]
  #print('F:', F)
  return F

def plot(points):
  for point in points:
    plt.plot(point[0], point[1])
    plt.waitKey(0)   
    plt.show()
# Function to implement RANSAC algorithm to reject outliers from SIFT
def ransac(pts1, pts2):
  iterations = 50
  threshold = 0.05
  max_count = 0
  for i in range(0, iterations):
    count = 0
    idx = random.sample(range(pts1.shape[0]), 8)
    left_pts = pts1[idx]
    right_pts = pts2[idx]
    F = fundamental_matrix(left_pts,right_pts)
    left_feature_inliers = []
    right_feature_inliers = []
    for j in range(0, len(pts1)):
      homogeneous_right = np.array([pts2[j, 0], pts2[j, 1], 1])
      homogeneous_left = np.array([pts1[j, 0], pts1[j, 1], 1])
      fit = np.dot(homogeneous_right.T, np.dot(F, homogeneous_left))  
      if np.abs(fit) < threshold:
        left_feature_inliers.append(pts1[j])
        right_feature_inliers.append(pts2[j])
        count += 1
    inliers_left = np.array(left_feature_inliers)
    inliers_right = np.array(right_feature_inliers)  
    if count > max_count:
      max_count = count
      estimated_F = F  
      final_inliers_left = inliers_left
      final_inliers_right = inliers_right
  #print('The estimated fundamental Matrix is:', estimated_F)
  return final_inliers_left, final_inliers_right, estimated_F
  
def essential_matrix(K, F):
  E = np.dot(K.T, np.dot(F, K))
  u, s, v = np.linalg.svd(E)

  # We correct the singular values of the E matrix
  s_new = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]).reshape(3,3)
  final_E = np.dot(u, np.dot(s_new, v))
  return final_E
  
# Main Function
def main():
  ax =  plt.gca()
  list_fname = [fname for fname in os.listdir(IMAGES_PATH)]
  list_fname.sort()
  H = np.eye(4)
  p0 = np.array([0, 0, 0, 1])
  # Extract the camera params
  fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel.ReadCameraModel(MODELS_PATH)
  K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
  print('The extracted camera parameters are fx = {:}, fy = {:}, cx = {:}, cy = {:}, G_camera_image = {:}, LUT = {:}:'.format(fx, fy, cx, cy, G_camera_image, LUT))
  successive_frames = []
  itr = 0
  points = []
  # Iterating through all frames in the video and doing some preprocessing
  for index in range(20, len(list_fname)):
    print('Image:', index)
    frame1 = IMAGES_PATH  + "/" + list_fname[index]
    frame2 = IMAGES_PATH + "/" + list_fname[index+1]
    img1 = cv2.imread(frame1,0)
    img2 = cv2.imread(frame2,0)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BayerGR2BGR)
    img1 = UndistortImage.UndistortImage(img1,LUT)
    img2 = UndistortImage.UndistortImage(img2,LUT)
    #img = cv2.resize(img, (0,0),fx=0.5,fy=0.5)
    # Get point matches 
    left_pts, right_pts = featurematch(img1, img2)
    #left_inliers, right_inliers, F = ransac(left_pts, right_pts)
    E1, _ = cv2.findEssentialMat(left_pts, right_pts, focal=fx, pp=(cx,cy), method=cv2.RANSAC, prob=0.999, threshold=0.05)
    #E = essential_matrix(K, F)
    #print('Essential matrix:', E)
    #print('Essential matrix from Opencv:', E1)
    _, R, T, mask = cv2.recoverPose(E1, left_pts, right_pts, focal=fx, pp=(cx,cy))
    H1 = np.hstack((R, T))
    H1 = np.vstack((H1,np.array([0,0,0,1])))
    H = np.dot(H, H1)
    #print('H:', H)
    x = H[0][3]
    y = H[2][3]
    ax.scatter(x, y, color='b')
    plt.draw()
    plt.pause(0.001)
    plt.savefig(r"/home/nalindas9/Desktop/op1/frame" + str(index) + ".png")
    #print("(x,y)", (x,y))
    points.append((x,y))
    successive_frames = []  
  plot(points)      
  #cv2.imshow('Color image', img)
  #cv2.waitKey(0)
  
  cv2.destroyAllWindows()
    
if __name__ == '__main__':
  main()
