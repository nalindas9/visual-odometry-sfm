"""
Main file for Visual Odomotry SFM

Authors:
Nalin Das (nalindas9@gmail.com)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""
import glob
import cv2
import ReadCameraModel

# Specify the path for all the video frames here
IMAGES_PATH = "/home/nalindas9/Documents/Courses/Spring_2020_Semester_2/ENPM673_Perception_for_Autonomous_Robots/Github/enpm673/Oxford_dataset/stereo/centre"
# Specify the path for the camera parameters
MODELS_PATH = "/home/nalindas9/Documents/Courses/Spring_2020_Semester_2/ENPM673_Perception_for_Autonomous_Robots/Github/enpm673/Oxford_dataset/model"

# Main Function
def main():
  fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel.ReadCameraModel(MODELS_PATH)
  print('The extracted camera parameters are fx = {:}, fy = {:}, cx = {:}, cy = {:}, G_camera_image = {:}, LUT = {:}:'.format(fx, fy, cx, cy, G_camera_image, LUT))
  for frame in sorted(glob.glob(IMAGES_PATH + "/*")):
    print('Image:', frame.split("centre/", 1)[1])
    img = cv2.imread(frame,0)
    img = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    img = cv2.resize(img, (0,0),fx=0.5,fy=0.5)
    cv2.imshow('Color image', img)
    cv2.waitKey(0)
    
if __name__ == '__main__':
  main()
