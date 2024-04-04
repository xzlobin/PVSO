import cv2
#from edges import waitUserExit

inputfile="./ximea/Zadanie_3/images/cheetah.jpg"

# reading image from file
img = cv2.imread(inputfile)

src = cv2.GaussianBlur(img, (7, 7), 2)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.Laplacian(img_gray, cv2.CV_16S, ksize=3)
abs_dst = cv2.convertScaleAbs(dst)
cv2.imshow("result", abs_dst)
cv2.waitKey()