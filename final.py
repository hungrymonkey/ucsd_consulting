#!/usr/bin/env python

import argparse, re

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt, pylab
import numpy as np
import xlrd
from  scipy import interpolate
import cv2.cv as cv, cv2
from matplotlib.nxutils import points_inside_poly
import sys, os
import csv
import re
from StringIO import StringIO

def readData(inputfile, ext):
    
    data = []
    if ext in '.xls':
      print inputfile
      wb = xlrd.open_workbook(inputfile)
      sh = wb.sheet_by_index(0)
      for rownum in range(sh.nrows):
         data.append(sh.row_values(rownum))
         if len(data[rownum]) is 201:
            del data[rownum][200]
      data = np.array(data).T
      
    elif ext in '.dat':
      inp = open(inputfile, 'r')
      for line in inp.readlines():   
         data.append([float(s) for s in line.strip().lstrip(',').rstrip(',').split(',')])
      data = np.array(data).T
      inp.close()
    elif ext in '.txt':
       #data = np.genfromtxt(inputfile, delimiter=',', skip_header=1)
       #data = data[6:]
       inp = open(inputfile,'r')
       #for line in inp.readlines():
          #data = np.array([float(s) for s in line.strip().lstrip(',').rstrip(',').split(',')[5:]])
       inp.readline()
       s = StringIO(inp.readline())
       data = np.genfromtxt(s,delimiter=',',dtype=np.float64)
       data = data[6:]
       inp.close()
    return data

def readRatio(fileName, whichOIS, ODorOS):
   f, ext = os.path.splitext(fileName)
   if ext in '.xls':
      wb = xlrd.open_workbook(fileName)
      sh = wb.sheet_by_index(0)
      for rownum in range(sh.nrows):
         if sh.row_values(rownum)[0].encode('utf8').upper() in whichOIS.upper():
            if sh.row_values(rownum)[1].encode('utf8').upper() in ODorOS.upper():
               try:
                  val = float(sh.row_values(rownum)[2])
               except ValueError, TypeError:
                  return 1
               else:
                  return val
   elif ext in '.csv':
      csv_reader = csv.reader(open(fileName, 'rU'))
      for row in csv_reader:
         if row[0] in whichOIS.upper():
            if row[1] in ODorOS.upper():
               try:
                  val = float(row[2])
               except ValueError, TypeError:
                  return 1
               else:
                  return val
   return 1
   #title = csv_reader.next()
   #header = ['' , '']
   #place = {}
   #value = []
   #for i, w in enumerate(title):
      #for s in header:
         #if s is w:
            #place[w]=i
      #if value is w:
         #value = {w: i}
   #name = os.path.basename(fileName).upper()
   #for row in csv_reader:
      #tmp = place.values()
      #for i in tmp:
         #if row[i] in name:
            #tmp.pop()
      #if len(tmp) is 0:
         #return row[y.values().pop()]
   
            
   
    
def getHeights(src, coord):
   height = []
   for i in range(len(coord[0])):
      height.append(src[int(coord[0][i])][int(coord[1][i])])
   return np.array(height)

def getContour(cvImg):
   lowerBound = cv.Scalar(0, 0, 0);
   upperBound = cv.Scalar(3, 3, 3);
   size =  cv.GetSize(cvImg)
   output = cv.CreateMat(size[0],size[1],cv.CV_8UC1)   
   cv.InRangeS(cvImg, lowerBound, upperBound,output)
   
   storage = cv.CreateMemStorage (0)
   seq = cv.FindContours(output, storage, cv.CV_RETR_LIST , cv.CV_CHAIN_APPROX_SIMPLE )
   mask = np.array(output)
   y,x = np.nonzero(mask)
   return (x,y)
   
   
def centerRadius(cvImg, ColorLower=None,ColorUpper=None):
   lowerBound = cv.Scalar(90, 0, 90);
   upperBound = cv.Scalar(171, 80, 171);
   size =  cv.GetSize(cvImg)
   output = cv.CreateMat(size[0],size[1],cv.CV_8UC1)
   cv.InRangeS(cvImg, lowerBound, upperBound,output)
   
   yi,xi = np.nonzero(output)
   #points = np.array([xi,yi]).T
   #coords = [tuple(points[pt]) for pt in range(len(points))]
   coords = zip(xi,yi)
   PointArray2D32f = cv.CreateMat(1, len(coords), cv.CV_32FC2)
   for (i, (x, y)) in enumerate(coords):
      PointArray2D32f[0, i] = (x, y)      
   (center, size, angle) = cv.FitEllipse2(PointArray2D32f)
   h,k =  center
   radius = np.sqrt(np.power((xi[0]-h),2)+np.power(yi[0]-k,2))
   return tuple(np.around([h,k])), round(radius)
   
def circumference(a,b):
   #return np.pi* (a+b)*(1+3*((a-b)/(b+a))**2/(10+np.sqrt(4-3*((a-b)/(b+a))**2)))
   return 2*np.pi*np.sqrt((np.power(a,2)+np.power(b,2))/2)
   
def mkellipseBox( circum, box2d):
   (cent, size, ang) = box2d
   w, l = size
   perimeter = circumference(w/2, l/2)
   np.multiply(size, circum/perimeter)
   
   
def cart2polar(x,y, origin=None, size=np.array([200,200])):
  ny, nx= size[0], size[1]
  if origin is None:
      origin_x, origin_y = nx//2, ny//2
  else:
      origin_x, origin_y = origin[0], origin[1]

  xi = x-origin_x
  yi = y-origin_y
  r = np.sqrt(np.power(xi,2) + np.power(yi,2))
  theta = np.arctan2(yi, xi)
  return (r, theta)
  
def polar2cart(r, theta, origin=None, size=np.array([200,200]) ):
  ny, nx= size[0], size[1]
  if origin is None:
      origin_x, origin_y = nx//2, ny//2
  else:
      origin_x, origin_y = origin[0], origin[1]
  x = r * np.cos(theta)
  y = r * np.sin(theta)
  x += origin_x
  y += origin_y
  return x, y
  
def uniquePts(pts):
   return list(set(pts))
   
def offsetGraph(height, SIDE):
   if SIDE == 1:
      return height
   else:# SIDE == 2:
      return  height[::-1]
      
      

      
def mkEllipsePts(theta, box2dorcenter, second=None, third=None):
   if type(box2dorcenter) is tuple:
      (center, size, ang) = box2dorcenter
   else:
      center = box2dorcenter
      size = second
      ang = third
   ang = ang*np.pi/180
   
   a,b = np.divide(size,2)
   h,k = center
   x,y = (a*(np.cos(theta-ang)), b*(np.sin(theta-ang)))
   return (np.cos(ang)*x-np.sin(ang)*y + h , np.sin(ang)*x+np.cos(ang)*y+k)
   
def mkPeriod(th, box, grid):
   x , y  = mkEllipsePts(th, box)
   cent, size, ang = box
   xy     = np.asarray(uniquePts(zip(np.floor(x),np.floor(y)))).T
   rad,th = cart2polar(xy[0],xy[1],origin=cent)
   index  = np.argsort(th)           
   xy, th = xy[:,index[:]], th[index[:]]
   validMask = xy.T < 200
   validMask = 1 - validMask.T[0,:] * validMask.T[1,:]
   th = np.delete(th, np.nonzero(validMask) ,0)
   xy = np.delete(xy.T, np.nonzero(validMask),0).T
   h = getHeights(grid,xy)
   return (th, h)
   
def mkResizeBox(circum, box):
   (cent,size,ang) = box
   w,l=size
   ratio = circum/circumference(l/2,w/2)
   
   return (cent,( w*ratio,l*ratio),ang)
   
def fmtCSVArr(arr):
   return ','.join([ '%.3f'%(s) for s in arr])
   
def main(argv):
   
   TITLE = ""
   PURPLE_RADIUS_MM = 3.46/2
   GROWTH_MM = 600
   RGB_Array = [ (0,255,0)  , (0,0,255), (50,100,0), (255,0,0),(0,100,50) ,(100,50,255)]
   GRAPHGROWTH = 1
   rescale = 1
   IMAGE_FORMAT = 'png'
   
   parser = argparse.ArgumentParser()
   parser.add_argument('-i','--image',required=True, dest='i')
   parser.add_argument('-d','--data',required=True, dest='d')
   parser.add_argument('-t','--tsnit', dest='t')
   parser.add_argument('-r','--rescale', dest='r')
   parser.add_argument('-m',action='store_true')
   parser.add_argument('-n',action='store_true')

   args = parser.parse_args()
   
   if args.i is  None:
      sys.exit(1)
   else:
      OPTICIMAGE= args.i
   if args.d is  None:
      sys.exit(1)
   else:
      DATAFILE = args.d
   if args.t is not None:
      fileName, extension = os.path.splitext(args.t)
      tsnitData = readData(args.t, extension)
   if args.m is True:
      GRAPHGROWTH = 0
      TITLE += 'By mic ' 
   else:
      GRAPHGROWTH = 1
      TITLE += 'By Percent '
   #if args.n is True:
      #GRAPHGROWTH = 1
      #TITLE += 'By Percent '
   fileName, extension = os.path.splitext(DATAFILE)
   heightData  = readData(DATAFILE, extension)
   if len(heightData) is 0:
      sys.exit(2)
      
   regex = re.compile('^[A-Z]+[0-9]+')
   basename = os.path.basename(fileName)
   m = regex.match(basename)
   
   whichOIS = m.group()
   SIDE = 0
   if any(s in basename.upper().replace(" ", "_") for s in ['_OD_', ' OD ']):
      SIDE = 1
      TITLE += "Right Eye Clockwise Graph"
      XLABEL = "Start at 9'o clock"
      SAVEFILE = whichOIS + '_OD'
      STARTANG= np.pi
   
   elif any( s in basename.upper().replace(" ","_")  for s in ['_OS_', ' OS ']):
      SIDE = 2
      STARTANG= 0
      SAVEFILE = whichOIS + '_OS'
      TITLE += "Left Eye Counter-Clockwise"
      XLABEL = "Start at 3'o clock"
   
   if args.r is not None:
      RATIOFILE = args.r
      if SIDE is 1:
         rescale = readRatio(RATIOFILE,whichOIS,'OD')
      else:
         rescale = readRatio(RATIOFILE,whichOIS,'OS')
   
   SUPRESS_ADJUSTED = False
   if rescale is 1:
      SUPRESS_ADJUSTED = True
   
   GRAPH_SAVE = SAVEFILE + "_GRAPH"
   IMAGE_SAVE = SAVEFILE + "_EYE"
   CSV_SAVE = SAVEFILE+'_DATA'
    # 0 nothing, 1 right, 2 left
   
     
   
   if GRAPHGROWTH is 1:
      
      GRAPH_SAVE = GRAPH_SAVE+'_PERCENT'
      IMAGE_SAVE = IMAGE_SAVE+'_PERCENT'
      CSV_SAVE = CSV_SAVE + '_PERCENT'
      
   else:
      GRAPH_SAVE = GRAPH_SAVE+'_MIC'
      IMAGE_SAVE = IMAGE_SAVE+'_MIC'
      CSV_SAVE = CSV_SAVE + '_MIC'
   
   dataFile = (open(CSV_SAVE+'.csv','w'))     
   imgColor       = cv.LoadImage(OPTICIMAGE, cv.CV_LOAD_IMAGE_UNCHANGED)  
   width, height      = cv.GetSize(imgColor)

   center, purpRad = centerRadius(imgColor, 2)
   xi, yi = getContour(imgColor)
   coords = zip(xi,yi)
   PointArray2D32f = cv.CreateMat(1, len(coords), cv.CV_32FC2)
   for (i, (x, y)) in enumerate(coords):
      PointArray2D32f[0, i] = (x, y)      
   #(centerCont, size, angle) = cv.FitEllipse2(PointArray2D32f)
   box2d = cv.FitEllipse2(PointArray2D32f)
   
   fig  = plt.figure(1)
   fig.subplots_adjust(hspace=.2)

   graph1 = fig.add_subplot(2,1,1)
   #graph2 = fig.add_subplot(3,1,2)
   
   #norm =  fig.add_subplot(2,1,2)
   tsnitGraph = fig.add_subplot(2,1,2)
   
   
   
   ratio = width/float(len(heightData))
   mmperpixel = PURPLE_RADIUS_MM/purpRad
   addedDist = (GROWTH_MM/1000. /mmperpixel)
   
   #dataFile.write(fmtCSVArr(np.linspace(1,360,360))+"\n")
   dataFile.write("%s,%s,%s,%s,%s\n"%('name','shape','dimensions (um)',"AUC (um)",fmtCSVArr(np.linspace(1,360,360))))
   writeTheta = np.linspace( STARTANG*180/np.pi,(STARTANG+ 2 *np.pi)*180/np.pi  , 361)
   #dataWriter.writerow(np.around(np.linspace(0,360,len(theta)),decimals=3))
   writeTheta = writeTheta*np.pi/180
   writeTheta = offsetGraph(writeTheta,SIDE)
   
   #np.savetxt(dataFile, np.linspace(0,360,len(theta)),fmt='%.3f',delimiter='\n',newline=',')
   def graphShape(ellipBox, grid, ratio, graph=None, label="", color=None, flags=0):
      # 1 = image
      # 2 = circle
      # 4 = ellipse
      # 8 = other graph
      ellipse= circle= image = writeOther =False
      if (flags&0x01) != 0:
         image=True
      if (flags&0x02) != 0:
         ellipse=True      
      if (flags&0x04) != 0:
         circle=True
      if (flags&0x08) != 0:
         writeOther = True

         
      (cent, size, angle) = ellipBox
      w, l = size
      circum = circumference(l/2,w/2)
      d = circum/np.pi
      
      excelEllipeBox = (tuple(np.divide(cent,ratio)), tuple(np.round(np.divide(size,ratio))), angle)
      circBox  = (tuple(np.divide(cent,ratio)),tuple(np.round((d/ratio,d/ratio))),angle)
      
      th = np.linspace( STARTANG,STARTANG+ 2 *np.pi  , np.round(circum/ratio*2))
      
      
      thet, h = mkPeriod(th, circBox, grid)
      f = interpolate.interp1d(np.concatenate((thet-2*np.pi,thet,thet+2*np.pi,thet+4*np.pi)),np.concatenate((h,h,h,h)))
      
      writeCircHeight = f(writeTheta)
      
      thet, h = mkPeriod(th, excelEllipeBox, grid)
      f = interpolate.interp1d(np.concatenate((thet-2*np.pi,thet,thet+2*np.pi,thet+4*np.pi)),np.concatenate((h,h,h,h)))
      
      writeEllipHeight = f(writeTheta)
      
      areaCirc  = circum * mmperpixel / len(writeCircHeight) * np.sum(writeCircHeight)
      areaEllip = circum * mmperpixel / len(writeEllipHeight) * np.sum(writeEllipHeight)
      
      
      r, g, b = color
      ovalR, ovalG, ovalB = 255-r,255-g,255-b
      if image is True:
         cv.Circle(imgColor, tuple(np.around(cent).astype('int')), int(d/2), color=cv.RGB(r, g, b))
         cv.EllipseBox(imgColor, ellipBox, cv.RGB(ovalR, ovalG, ovalB))
      
      #dataWriter.writerow(np.around(circHeight,decimals=3))
      #dataWriter.writerow(np.around(ellipHeight,decimals=3))
      #np.savetxt(dataFile,circHeight,fmt='%.4f',newline=',',delimiter='\n')
      #np.savetxt(dataFile,ellipHeight,fmt='%.4f',newline=',',delimiter='\n')
      
      #f = interpolate.interp1d(th,circHeight-ellipHeight)
      #diffHeight = f(th)
      dataFile.write("%s,%s,%.0fx%.0f,%.3f,%s\n"%(label, 'ellipse', l*mmperpixel*1000,w*mmperpixel*1000, areaEllip,fmtCSVArr(writeEllipHeight[1:])))
      dataFile.write("%s,%s,%.0fx%.0f,%.3f,%s\n"%(label, 'circle',  d*mmperpixel*1000,d*mmperpixel*1000, areaCirc,fmtCSVArr(writeCircHeight[1:])))
      if writeOther is True:
         tsnitGraph.plot(np.linspace(0,360,len(writeCircHeight)) , writeCircHeight, color=(r/255., g/255., b/255.) , label=(label+" cir AUC "+str(round(areaCirc/1000,3))+" mm^2"))
      if ellipse is True:
         #dataFile.write(label+','++','+fmtCSVArr(writeEllipHeight)+"\n")
         graph.plot(np.linspace(0,360,len(writeEllipHeight)) , writeEllipHeight, color=(ovalR/255., ovalG/255., ovalB/255.), label=(label+" ellip AUC "+str(round(areaEllip/1000,3))+" mm^2"))
      
      if circle is True:
         #dataFile.write(label+','+'%.0fx%.0f'%()+','+fmtCSVArr(writeCircHeight)+"\n")    
         graph.plot(np.linspace(0,360,len(writeCircHeight)) , writeCircHeight, color=(r/255., g/255., b/255.) , label=(label+" cir AUC "+str(round(areaCirc/1000,3))+" mm^2"))
      
      #norm.plot(np.linspace(0,360,len(ellipHeight)),diffHeight,  label=(label+" height diff"))
      
   if GRAPHGROWTH is 1:
      for i in [50,100]:
         (cent, size, ang) = box2d
         a,b = size 
         circum = circumference(a/2,b/2)*(1+i/100.)
         graphShape(mkResizeBox(circum, box2d), heightData, ratio*rescale, graph1,'%'+str(i)+' larger', color=RGB_Array.pop(),flags=4)
         
   if GRAPHGROWTH is 0:   
      for i in [1,2]:
         (cent, size, ang) = box2d
         graphShape((cent, tuple(np.add(size, addedDist*2*i)) , ang), heightData, ratio*rescale, graph1,str(int(i*GROWTH_MM))+' um', color=RGB_Array.pop(),flags=4)
      
   (cent, size, ang) = box2d
   circleBox = (tuple(center), size, ang)
   graphShape( mkResizeBox( 2*np.pi*purpRad, circleBox), heightData, ratio, graph1, color=RGB_Array.pop(), label=str(3.46),flags=15)
   if SUPRESS_ADJUSTED is False:
      graphShape( mkResizeBox( 2*np.pi*purpRad, circleBox), heightData, ratio*rescale, graph1, color=RGB_Array.pop(), label='3.46 adjusted',flags=15)
   
   
   
   cv.SaveImage(IMAGE_SAVE+'.' + IMAGE_FORMAT, imgColor)
   
   graph1.set_title(TITLE + '\nscaling factor: ' +str(round(rescale,3)))
   #norm.set_title('Circle and Oval Height Difference')
   #norm.set_xlabel(XLABEL +'(Degrees)')
   #norm.legend(bbox_to_anchor=(1.13,1), prop={'size':6})
   
   tsnitCircum = np.pi*PURPLE_RADIUS_MM*2/1000
   tsnitArea = tsnitCircum /len(tsnitData)* np.sum(tsnitData) 
   tsnitGraph.plot(np.linspace(0,360,len(tsnitData)),tsnitData , color=tuple(np.divide(RGB_Array.pop(),255.)), label=("3.46 TSNIT " + str(round(tsnitArea,3))+'mm^2'))
   dataFile.write("%s,%s,%.0fx%.0f,%.3f,%s\n"%("tsnit", 'circle',PURPLE_RADIUS_MM*2*1000,PURPLE_RADIUS_MM*2*1000,tsnitArea*1000,''))
   tsnitGraph.set_title('TSNIT GRAPH')
   tsnitGraph.set_xlabel(XLABEL +'(Degrees)')
   tsnitGraph.legend(bbox_to_anchor=(1.13,1), prop={'size':6})
   
   plt.ylabel('Heights (nm)')
   
   #graph.set_xlabel(XLABEL +'(Degrees)')
   graph1.legend(bbox_to_anchor=(1.13,1),prop={'size':6})
   #graph2.legend(bbox_to_anchor=(1.13,1),prop={'size':8})
   fontP = FontProperties()
   fontP.set_size('xx-small')
   
   plt.savefig(GRAPH_SAVE+ "." + IMAGE_FORMAT)
   dataFile.close()
   
if __name__ == "__main__":
   main(sys.argv[1:])

   
   