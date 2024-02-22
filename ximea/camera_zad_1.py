import numpy as np
from ximea import xiapi
import cv2
### runn this command first echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb  ###
#create instance for first connected camera
cam = xiapi.Camera()

print('Opening first camera...')
cam.open_device()
#settings
cam.set_exposure(10000)
cam.set_param('imgdataformat','XI_RGB32')
cam.set_param('auto_wb', 1)
print('Exposure was set to %i us' %cam.get_exposure())

img = xiapi.Image()
print('Starting data acquisition...')
cam.start_acquisition()

def cont_imgs(imgs):
    line_1 = np.concatenate((imgs[0], imgs[1]), axis=1)
    line_2 = np.concatenate((imgs[2], imgs[3]), axis=1)
    result_img = np.concatenate((line_1, line_2), axis=0)
    return result_img

i = 0
result_img = None
imgs = [np.zeros((240, 240, 4), dtype=np.uint8)]*4;
cv2.imshow("result", cont_imgs(imgs))

def action_conv(image):
    custom_kernel = np.ones((3, 3), np.float32) / 3
    return cv2.filter2D(src=image, dst=image, ddepth=-1, kernel=custom_kernel)

def action_rotate(image):
    for i in range(240):
        for j in range(i):
            buff = image[j, i, :].copy()
            image[j, i, :] = image[i, j, :]
            image[i, j, :] = buff
    return image

def action_get_red(image):
    image[:, :, [0, 1, 3]] = 0
    return image
def nothing(image):
    return image

actions = [action_conv, action_rotate, action_get_red, nothing]

while cv2.waitKey() != ord('q'):
    if i < 4:
        cam.get_image(img)
        image = img.get_image_data_numpy()
        image = cv2.resize(image, (240, 240))
        imgs[i] = image
        cv2.imwrite(f'img_{i}.jpg', image)
        result_img = cont_imgs(imgs)
        cv2.imwrite(f'img_result.jpg', result_img)
        cv2.imshow("result", result_img)
    elif i < 8:
        i_sel = (i % 4) // 2
        i_start = i_sel*240
        j_sel = (i % 4) % 2
        j_start = j_sel*240

        actions[i % 4](result_img[i_start:(i_start + 240), j_start:(j_start + 240), :])
        cv2.imwrite(f'img_result_filtered.jpg', result_img)
        cv2.imshow("result", result_img)
    else:
        break
    i += 1

print('Data acquisition is done...')
print(f'Data type: {result_img.dtype}, '
      f'shape: {result_img.shape}, '
      f'size: {np.prod(result_img.shape)*result_img.dtype.itemsize} byte')

#stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()

#stop communication
cam.close_device()
print('Done.')