import sys
import numpy as np
import getopt
import cv2
import LoG

def render_amplitude(imageLike):
   i = np.abs(imageLike)
   i = 255*cv2.normalize(i, None, 0.0, 1.0, cv2.NORM_MINMAX)
   return i.astype('uint8')

def find_edges(img, threshold=0.20, variance_thr = 0.07, background_extraction_thr=0.07):
   img = img.astype(np.float64)

   laplace = LoG.Laplacian(signature=(3,3)).build()
   gauss   = LoG.Gaussian(shape=(7,7), variance=2).build()
   mean    = LoG.Mean(shape=(5,5)).build()
   
   g_img = gauss.jit_apply_to(img)
   edges = laplace.jit_apply_to(g_img)

   m_img = mean.jit_apply_to(img)
   m_img_sq = mean.jit_apply_to(img*img)
   variance_sq = m_img_sq - m_img*m_img
   variance_sq = cv2.normalize(variance_sq, None, 0.0, 1.0, cv2.NORM_MINMAX)

   edges = np.abs(edges)
   edges = cv2.normalize(edges, None, 0.0, 1.0, cv2.NORM_MINMAX)

   gauss_diff = np.abs(img - g_img)
   gauss_diff = cv2.normalize(gauss_diff, None, 0.0, 1.0, cv2.NORM_MINMAX)
   gauss_diff = cv2.medianBlur((255*gauss_diff).astype(np.uint8), 5)

   strong_edges = np.zeros(edges.shape, dtype=np.float64) + 0.5
   selector = (edges < threshold) & (variance_sq > variance_thr) & (gauss_diff > 256*background_extraction_thr)
   strong_edges[selector] = 1-edges[selector]
   se_min = np.min(strong_edges[selector])
   se_max = np.max(strong_edges[selector])
   strong_edges[selector] = 0.5 + 0.5 * (strong_edges[selector] - se_min)/(se_max - se_min)
   return (255*strong_edges).astype(np.uint8)

def waitUserExit(window_name="result"):
   while( cv2.getWindowProperty(window_name,cv2.WND_PROP_VISIBLE ) ):
      cv2.waitKey(100)
   cv2.destroyAllWindows()

def main(argv):
   inputfile = ''
   outputfile = ''
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
   if not inputfile: inputfile="./ximea/Zadanie_3/cheetah.jpg"
   cv2.imshow('result', np.zeros((50,50)))
   
   img = cv2.imread(inputfile)
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   res = find_edges(gray)

   if not outputfile:
      cv2.imshow('result', res)
      waitUserExit()
   else:
      cv2.imwrite(outputfile, res)

if __name__ == "__main__":
   main(sys.argv[1:])