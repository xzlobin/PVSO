import numpy as np
import cv2
import yaml
import os

script_folder = os.path.dirname(os.path.realpath(__file__))

with open(f'{script_folder}/camera_parameters.yaml', 'r') as f:
    cam_parsed = yaml.safe_load(f)
    cam_params = {'camera_mat': np.array(cam_parsed['camera_mat']), 
                  'distortion': np.array(cam_parsed['distortion'])}
    

minDist = 20
param1 = 100
param2 = 100
minRadius = 0
maxRadius = 0

img = np.zeros((480,480,4), np.uint8)
cv2.namedWindow('Camera')

def callback_factory(global_var_name):
    def callback(new_value):
        globals()[global_var_name] = new_value
    return callback

# create trackbars for parameters change
cv2.createTrackbar('minDist  ' , 'Camera', 0, 100, callback_factory('minDist'))
cv2.createTrackbar('param1   ' , 'Camera', 0, 200, callback_factory('param1'))
cv2.createTrackbar('param2   ' , 'Camera', 0, 200, callback_factory('param2'))
cv2.createTrackbar('minRadius' , 'Camera', 0, 255, callback_factory('minRadius'))
cv2.createTrackbar('maxRadius' , 'Camera', 0, 255, callback_factory('maxRadius'))

# set trackbars' initial values
cv2.setTrackbarPos('minDist  ' , 'Camera', minDist)
cv2.setTrackbarPos('param1   ' , 'Camera', param1)
cv2.setTrackbarPos('param2   ' , 'Camera', param2)

while True:
    cv2.imshow('Camera', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    print(minDist)


cv2.waitKey(0)
