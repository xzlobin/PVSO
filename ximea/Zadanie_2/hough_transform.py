import numpy as np
import cv2
import yaml
import os
import time

# To increase buffer size use:
# sudo sh -c 'echo 1024 > /sys/module/usbcore/parameters/usbfs_memory_mb'
# works until reboot

script_folder = os.path.dirname(os.path.realpath(__file__))

with open(f'{script_folder}/camera_parameters.yaml', 'r') as f:
    cam_parsed = yaml.safe_load(f)
    cam_params = {'camera_mat':  np.array(cam_parsed['camera_mat']), 
                  'distortion':  np.array(cam_parsed['distortion']),
                  'image_shape': cam_parsed['image_shape']}

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam_params['camera_mat'], cam_params['distortion'], 
                                                  cam_params['image_shape'], 1, cam_params['image_shape'])

minDist = 62
param1 = 118
param2 = 45
minRadius = 0
maxRadius = 0

img = np.zeros((480,480,3), np.uint8)
cv2.namedWindow('Camera')

def callback_factory(global_var_name):
    def callback(new_value):
        globals()[global_var_name] = new_value + 1
    return callback

# create trackbars for parameters change
cv2.createTrackbar('minDist  ' , 'Camera', 0, 500, callback_factory('minDist'))
cv2.createTrackbar('param1   ' , 'Camera', 0, 200, callback_factory('param1'))
cv2.createTrackbar('param2   ' , 'Camera', 0, 200, callback_factory('param2'))
cv2.createTrackbar('minRadius' , 'Camera', 0, 3000, callback_factory('minRadius'))
cv2.createTrackbar('maxRadius' , 'Camera', 0, 3000, callback_factory('maxRadius'))

# set trackbars' initial values
cv2.setTrackbarPos('minDist  ' , 'Camera', minDist)
cv2.setTrackbarPos('param1   ' , 'Camera', param1)
cv2.setTrackbarPos('param2   ' , 'Camera', param2)

def win_resize(image):
    return cv2.resize(image, (480,480))
def hough_resize(image):
    return cv2.resize(image, (960,960))
try:
    from ximea import xiapi

    cam = xiapi.Camera()

    print('Opening first camera...')
    cam.open_device()
    # settings
    cam.set_exposure(15000)
    cam.set_param('imgdataformat', 'XI_RGB32')
    cam.set_param('auto_wb', 1)
    print('Exposure was set to %i us' % cam.get_exposure())

    img = xiapi.Image()
    print('Starting data acquisition...')
    cam.start_acquisition()

    stored_method = cam.get_image
    while cv2.waitKey(1) != ord('q'):
        cam.get_image(img)
        image = img.get_image_data_numpy()
        image = cv2.undistort(image, cam_params['camera_mat'], 
                              cam_params['distortion'], None, newcameramtx)
        image = hough_resize(image)
        
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
        circles = cv2.HoughCircles(image_gray,cv2.HOUGH_GRADIENT,1,minDist=minDist,circles=None,
                            param1=param1,param2=param2,minRadius=minRadius,maxRadius=minRadius)

        cv2.imshow("Camera", win_resize(image))

        if circles is None or circles.size == 0:
            print('No circles found')
            continue

        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(image,(i[0],i[1]),2,(0,0,255),3)

        cv2.imshow("Camera", win_resize(image))

        keyp = cv2.waitKey(1)
        if keyp == ord('p'):
            cam.get_image = lambda img: img
        elif keyp == ord('c'):
            cam.get_image = stored_method

except ImportError:
    print('xiapi not found')

cv2.destroyAllWindows()