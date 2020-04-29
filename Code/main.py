"""
Visual Odomotry SFM

Authors:
Nalin Das (nalindas9@gmail.com)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""
import glob
import cv2
import ReadCameraModel
import UndistortImage
from matplotlib import pyplot as plt 

# Specify the path for all the video frames here
IMAGES_PATH = "/home/nalindas9/Documents/Courses/Spring_2020_Semester_2/ENPM673_Perception_for_Autonomous_Robots/Github/enpm673/Oxford_dataset/stereo/centre"
# Specify the path for the camera parameters
MODELS_PATH = "/home/nalindas9/Documents/Courses/Spring_2020_Semester_2/ENPM673_Perception_for_Autonomous_Robots/Github/enpm673/Oxford_dataset/model"

def featurematch(frame1, frame2):
  # Using SIFT to find keypoints and descriptors
  orb = cv2.ORB_create()
  kp1, des1 = orb.detectAndCompute(frame1,None)
  kp2, des2 = orb.detectAndCompute(frame2,None)
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(des1,des2)
  matches = sorted(matches, key = lambda x:x.distance)
  img3 = cv2.drawMatches(frame1,kp1,frame2,kp2,matches[:20], None, flags=2)
  plt.imshow(img3),plt.show()
  
  
# Main Function
def main():
  # Extract the camera params
  fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel.ReadCameraModel(MODELS_PATH)
  print('The extracted camera parameters are fx = {:}, fy = {:}, cx = {:}, cy = {:}, G_camera_image = {:}, LUT = {:}:'.format(fx, fy, cx, cy, G_camera_image, LUT))
  successive_frames = []
  itr = 0
  # Iterating through all frames in the video and doing some preprocessing
  for frame in sorted(glob.glob(IMAGES_PATH + "/*")):
    print('Image:', frame.split("centre/", 1)[1])
    img = cv2.imread(frame,0)
    img = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    img = UndistortImage.UndistortImage(img,LUT)
    img = cv2.resize(img, (0,0),fx=0.5,fy=0.5)
    successive_frames.append(img)
    if itr != 0 and itr%2 == 0:
      featurematch(successive_frames[0], successive_frames[1])
      successive_frames = []        
    cv2.imshow('Color image', img)
    cv2.waitKey(0)
    itr = itr + 1
    
if __name__ == '__main__':
  main()
