import sys
import numpy as np
import getopt
import cv2
import LoG


def find_edges(img, threshold=0.2, variance_thr = 0.07, background_extraction_thr=0.05):
   """Function to detect edges
   Parameters
   ----------
   img          - gray scaled image 2D numpy array
   threshold    - from 0 to 1 - threshold for normalizied laplacian to consider it zero (if laplacian zero - pixel is the set of edge points)
   variance_thr - from 0 to 1 - threshold for normalizied square of pixel brightness variance to filter zero laplacian points.
                  point is considered to be an edge point only if [variance(x,y)]^2 > variance_thr
   background_extraction_thr - from 0 to 1 - threshold for normalized gauss difference ( img - gauss(img) ) of point to be considered 
                               as not backuround point. Background points are those points that are distorted by the bokeh effect.
   """
   # converting image to float64 type
   img = img.astype(np.float64)

   # building Kernels for Laplacian, Gaussian and Mean Blur to apply to an image.
   # realization is in the LoG.py module
   laplace = LoG.Laplacian(signature=(2,2)).build()
   gauss   = LoG.Gaussian(shape=(7,7), variance=1).build()
   mean    = LoG.Mean(shape=(5,5)).build()
   
   # applying gaussian to an image
   g_img = gauss.apply_to(img)
   # applying laplace operator to the result
   edges = laplace.apply_to(g_img)

   # applying mean blur (equals to math expectation of brightness around every pixel)
   m_img = mean.apply_to(img)
   # in the similar way math expectation for squared image
   m_img_sq = mean.apply_to(img*img)
   # looking for variance^2(x,y) using formula 
   # V = E(X-E(X))^2 = E(X^2 - 2X*E(X) + E^2(X)) = E(X^2) - E^2(X)
   variance_sq = m_img_sq - m_img*m_img
   # normalizing variance^2(x,y) to appy threshold later
   variance_sq = LoG.normalize(variance_sq)

   # normalazing laplacian of image to apply threshold
   edges = np.abs(edges)
   edges = LoG.normalize(edges)

   # calculating difference between an image and the blured image to separate the background
   gauss_diff = np.abs(img - g_img)
   # bluring to avoid high frequency noise
   gauss_diff = 255*mean.apply_to(LoG.normalize(gauss_diff))

   # array to store result image. Background will be gray 
   # the edges - white - the whiter the closer point's laplacian to zero
   strong_edges = np.zeros(edges.shape, dtype=np.float64) + 0.5

   # select only those edges where laplacian is close enough to zero, variance of brightness around is high enough
   # and the point isn't considered to be a backround one
   selector = (edges < threshold) & (variance_sq > variance_thr) & (gauss_diff > 256*background_extraction_thr)

   # set white color to the edges based on their strength, using the selector found
   strong_edges[selector] = 1-edges[selector]
   se_min = np.min(strong_edges[selector])
   se_max = np.max(strong_edges[selector])
   strong_edges[selector] = 0.5 + 0.5 * (strong_edges[selector] - se_min)/(se_max - se_min)

   # convert to 8 bit gray-scale image and return
   return (255*strong_edges).astype(np.uint8)


def waitUserExit(window_name="result"):
   """Function to wait window with name window_name to be closed. Either with a keyboard or by system UI.
   The only purpose of this function to exist is that after using cv2.waitKey() script stucks forever
   if user closed the window with X button.
   """
   while( cv2.getWindowProperty(window_name,cv2.WND_PROP_VISIBLE ) ):
      cv2.waitKey(100)
   cv2.destroyAllWindows()


def main(argv):
   """Main function.
   Thi script can be used from command line in the way
   > python3 edges.py -i image.jpg -o result.jpg
   if -o is not provided, then image will be shown in the opencv window
   if -i is not provided, then will be opened an image with relative path
   "./ximea/Zadanie_3/cheetah.jpg"
   """
   inputfile = ''
   outputfile = ''
   
   # just parsing arguments given to the script
   try:
      opts, _ = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print ('edges.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print ('edges.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   

   if not inputfile: inputfile="./ximea/Zadanie_3/images/cheetah.jpg"
   
   # reading image from file
   img = cv2.imread(inputfile)
   
   # converting to gray scale image
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   # detecting edges
   res = find_edges(gray)

   # write the result to a file or display with an opencv window
   if not outputfile:
      cv2.imshow('result', res)
      waitUserExit()
   else:
      cv2.imwrite(outputfile, res)

# runtime check if script is being run by user, not imported as a module
if __name__ == "__main__":
   main(sys.argv[1:])