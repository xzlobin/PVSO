import numpy as np
import cv2
import os
import time
import edges

# To increase buffer size use:
# sudo sh -c 'echo 1024 > /sys/module/usbcore/parameters/usbfs_memory_mb'
# works until reboot

script_folder = os.path.dirname(os.path.realpath(__file__))

threshold =0.2
variance_thr = 0.07
background_extraction_thr = 0.05

img = np.zeros((800,800,3), np.uint8)
cv2.namedWindow('Camera')

def callback_factory(global_var_name):
    def callback(new_value):
        globals()[global_var_name] = new_value/1000
    return callback

# create trackbars for parameters change
cv2.createTrackbar('threshold  ' , 'Camera', 0, 1000, callback_factory('threshold'))
cv2.createTrackbar('variance_thr   ' , 'Camera', 0, 1000, callback_factory('variance_thr'))
cv2.createTrackbar('bkg_ext_thr   ' , 'Camera', 0, 1000, callback_factory('background_extraction_thr'))

# set trackbars' initial values
cv2.setTrackbarPos('minDist  ' , 'Camera', int(threshold*1000) )
cv2.setTrackbarPos('param1   ' , 'Camera', int(variance_thr*1000))
cv2.setTrackbarPos('param2   ' , 'Camera', int(background_extraction_thr*1000))

def resize(image):
    return cv2.resize(image, (800,800))
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
        try:
            cam.get_image(img)
        except Exception as e:
            print(f"Camera error: {e}")
            print("Skipping frame...")
            time.sleep(1/30)
            continue

        image = img.get_image_data_numpy()
        image = resize(image)
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
        res = edges.find_edges(image_gray, threshold=threshold, variance_thr=variance_thr, background_extraction_thr=background_extraction_thr)

        cv2.imshow("Camera", image)

        keyp = cv2.waitKey(1)
        if keyp == ord('p'):
            cam.get_image = lambda img: img
        elif keyp == ord('c'):
            cam.get_image = stored_method

except ImportError:
    print('xiapi not found')

cv2.destroyAllWindows()