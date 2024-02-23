import numpy as np
import cv2
import glob
import uuid
import os
import yaml
from operator import methodcaller

script_folder = os.path.dirname(os.path.realpath(__file__))

### runn this command first echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb  ###
# create instance for first connected camera
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
        image_uid = uuid.uuid1()
        cv2.imwrite(f'{script_folder}/img/img_{image_uid}.jpg', image)
        cv2.imshow("result", cv2.resize(image, (480, 480)))

    print('Data acquisition is done...')
    # stop data acquisition
    print('Stopping acquisition...')
    cam.stop_acquisition()

    # stop communication
    cam.close_device()
    print('Camera device has been closed.')
    cv2.destroyAllWindows()
except ImportError:
    print('xiapi not found, continue using ./img/*.jpg')

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# shape of the chessboard
shape = (7, 5)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((np.prod(shape), 3), np.float32)
objp[:, :2] = np.mgrid[0:shape[0], 0:shape[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
# 3d point in real world space
objpoints = []
# 2d points in image plane.
imgpoints = []

print('Starting calibration with ./img/*.jpg')
images = glob.glob(f'{script_folder}/img/*.jpg')
print(f'Detected files:')
for fname in images:
    print(f'- {fname}')

if not fname:
    print('No files found, terminating...')
    raise SystemExit

print(f'Starting detecting points in image plane...')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board cornersexi
    ret, corners = cv2.findChessboardCorners(gray, shape, None)
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, shape, corners2, ret)
        cv2.imshow('img', cv2.resize(img, (480, 480)))
        cv2.waitKey(100)

cv2.destroyAllWindows()

print(f'Points have been detected.\nStarting calibration...')
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

cam_params = {"camera_mat" : mtx.tolist(), "distortion": dist.tolist(), "rotation": [i.tolist() for i in rvecs], "translation": [i.tolist() for i in tvecs]}

print('Calibration is done, saving data in camera_parameters.yaml')
with open(f'{script_folder}/camera_parameters.yaml', 'w') as f:
    yaml.dump(cam_params, f)
