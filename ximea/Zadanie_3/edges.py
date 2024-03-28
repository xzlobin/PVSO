import sys
import getopt
import cv2
import LoG

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
   img = cv2.imread(inputfile)
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   kernel = LoG.Laplacian(signature=(1,1)).build()
   gauss  = LoG.Gaussian(shape=(5,5), variance=2).build()
   res_img = kernel.jit_apply_to(gray.astype('float'))
   res_img = gauss.jit_apply_to(res_img)
   res_img = cv2.normalize(res_img, None, 0.0, 1.0, cv2.NORM_MINMAX)
   if not outputfile:
      cv2.imshow('result', res_img)
      cv2.waitKey(0)
   else:
      cv2.imwrite(outputfile, (255*res_img).astype('uint8'))

if __name__ == "__main__":
   main(sys.argv[1:])