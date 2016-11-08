#!/usr/bin/env python

import cv2
import cv2.cv as cv
import argparse, sys, os
import numpy as np


LENGTH = 1536
WIDTH = 1536

##Errors
NO_ARGUMENTS = 1

LIGHT_GREEN = np.asarray([0,255,0])
DARK_GREEN = np.asarray([0,128,0])


def remove_green_roi(img):
   ##removing all the green and repainting
   mask = cv2.inRange(img,DARK_GREEN,LIGHT_GREEN)
   #print cv2.fitEllipse(cv2.cvtColor(mask,))
   #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) * ( 1 - mask )
   ## repair the image
   return cv2.inpaint(img, mask, inpaintRadius=1, flags=cv2.INPAINT_NS)
   #return mask

def resize(img):
   return cv2.resize(img, (img.shape[1]/2,img.shape[0]/2))

def blur(img, kernel_size):
   # ksize = 19
   #kernel_size = 51
   return cv2.GaussianBlur(img, (kernel_size,kernel_size),0)

def canny_edge(img):
   # canny high threshold should be between 2-3x the low threshold
   low_threshold = np.median(img) * .50
   high_threshold = np.median(img) * 1.40
   return cv2.Canny(img, low_threshold,high_threshold, apertureSize=5)

def open_image(path):
   return cv2.imread(path)



def split_grey(image, dimensions):
   l,w = dimensions
##opencv store images as [length][width][(b,g,r)]
##imageJ are usually [width/X][length/Y][(r,g,b)]
   return image[:l,:w,:], image[:l,w:,:] ###optic, optic_slice

def compare(img_obj, img_scene):
   surf_detector = cv2.FeatureDetector_create("SURF")
   surf_detector_extractor = cv2.DescriptorExtractor_create("SURF")
   
   keypoints = surf_detector.detect(img_obj)
   (keypoints, decriptors) = surf_detector_extractor.compute(img_obj,keypoints)
   
   keypoints2 = surf_detector.detect(img_scene)
   (keypoints2, descriptors2) = surf_detector_extractor.compute(img_scene,keypoints2)
   
   
   matches = match_flann(decriptors, descriptors2)
   ## get keypoints
   obj = np.asarray([ keypoints[m.queryIdx].pt for m in matches])
   scene = np.asarray([ keypoints2[m.trainIdx].pt for m in matches])
   
   
   img_obj = cv2.cvtColor(img_obj, cv2.COLOR_GRAY2BGR)
   img_scene = cv2.cvtColor(img_scene, cv2.COLOR_GRAY2BGR)
   for pt in obj: 
      cv2.circle(img_obj, (int(pt[0]),int(pt[1])) , 5, (0,255,255), -1)
   for pt in scene:
      cv2.circle(img_scene, (int(pt[0]),int(pt[1])) , 5, (0,255,255), -1)


   cv2.imshow("kitty", resize(img_obj))

   cv2.imshow("cat", resize(img_scene))
   cv2.waitKey(20000)
   mat_tuples = cv2.findHomography(obj, scene)
   mat = mat_tuples[0]
   
   ##get corners
   obj_corners = []
   obj_corners.append(np.zeros(2))
   obj_corners.append(np.asarray([img_obj.shape[0], 0 ]))
   obj_corners.append(np.asarray([0, img_obj.shape[1]]))
   obj_corners.append(np.asarray([img_obj.shape[0], img_obj.shape[1]]))
  
   return mat
   #output = cv2.perspectiveTransform( np.asarray(obj_corners), m=mat)
   #return cv2.warpPerspective( img_scene, mat , img_obj.shape )
   

def match_flann(descriptor1, descriptor2 ):
   FLANN_INDEX_KDTREE = 0
   index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
   search_params = dict(checks = 50)
   flann = cv2.FlannBasedMatcher(index_params, search_params)
   matches = flann.knnMatch(descriptor1, descriptor2, k=2 )
   good = []
   for m,n in matches:
      if m.distance < .7*n.distance:
         good.append(m)
   #matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
   return good

def main(argv):
   parser = argparse.ArgumentParser()
   parser.add_argument('-b','--black',dest='b')
   parser.add_argument('-c','--color',dest='c')
   
   args = parser.parse_args()
   
   if args.b is None:
      sys.exit(NO_ARGUMENTS)
   if args.c is None:
      sys.exit(NO_ARGUMENTS)
      
   grey_raw_image = open_image(args.b)
   color_raw_image = open_image(args.c)
   
   
   optic_retina, optic_slice = split_grey(grey_raw_image, ( LENGTH,WIDTH))
   ## remove green noise
   optic_slice_roi_mask = cv2.inRange(optic_retina,DARK_GREEN,LIGHT_GREEN)
   
   ## change color image to grey
   grey_color_image = cv2.cvtColor(color_raw_image, cv2.COLOR_BGR2GRAY)
   
   optic_retina = cv2.cvtColor(remove_green_roi(optic_retina ), cv2.COLOR_BGR2GRAY)
   
   #grey_vein_roi = canny_edge(blur(optic_retina,49))
   #color_vein_roi = canny_edge(blur(grey_color_image,15))
   
   output = cv2.Sobel(blur(optic_retina,27), -1, 1, 1, ksize=7)
   output2 = cv2.Sobel(blur(grey_color_image,13), -1, 1,1, ksize=7)
   #mat = compare(blur(color_vein_roi,7), blur(grey_vein_roi,7))
   
   
   #output = cv2.warpPerspective( color_raw_image, mat , optic_retina.shape )
   cv2.imshow("fish",resize(output))
   cv2.imshow("fish2",resize(output2))

   cv2.waitKey(15000)
   
   
   

   #cv2.imshow("fish",resize(grey_color_image))
   #cv2.waitKey(0)
   
if __name__ == "__main__":
   main(sys.argv[1:])

