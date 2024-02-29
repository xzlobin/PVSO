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

try:
    from ximea import xiapi

    cam = xiapi.Camera()

    print('Opening first camera...')
    cam.open_device()
    # settings
    cam.set_exposure(10000)
    cam.set_param('imgdataformat', 'XI_RGB32')
    cam.set_param('auto_wb', 1)
    print('Exposure was set to %i us' % cam.get_exposure())

    img = xiapi.Image()
    print('Starting data acquisition...')
    cam.start_acquisition()

    while cv2.waitKey() != ord('q'):
        cam.get_image(img)
        image = img.get_image_data_numpy()

        circles = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,1,minDist,
                            param1=param1,param2=param2,minRadius=minRadius,maxRadius=minRadius)
        
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(image,(i[0],i[1]),2,(0,0,255),3)

        cv2.imshow("Camera", image)
except ImportError:
    print('xiapi not found')

