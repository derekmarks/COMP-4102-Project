#Derek Marks
#100 974 560
#14 Apr 2020

import cv2
import tifffile as tiff
import numpy as np
import os
import math
import imutils

#THRESH_VAL = 180 #Best for image 4
THRESH_VAL = 203 #Best for all other images
SCALE_VAL = 110

def scale_image(src,percent):
        width = int(src.shape[1] * percent / 100)
        height = int(src.shape[0] * percent / 100)
        dsize = (width, height)
        imS = cv2.resize(src, dsize)
        return imS

def loadSample_1():
    img = tiff.imread(os.getcwd()+"\\aerials\\aerial_1.tif")    #1. Load
    img = img[1750:2500,img.shape[0]-1500:img.shape[0]-750]     #2. Crop
    img = scale_image(img,SCALE_VAL)                            #3. Scale
    return img

def loadSample_2():
    img = tiff.imread(os.getcwd()+"\\aerials\\aerial_2.tif")    #1. Load
    img = img[4000:4750,2500:3250]                              #2. Crop
    img = scale_image(img,SCALE_VAL)                            #3. Scale
    return img

def loadSample_3():
    img = tiff.imread(os.getcwd()+"\\aerials\\aerial_3.tif")    #1. Load
    img = img[3100:3850,5100:5850]                              #2. Crop
    img = scale_image(img,SCALE_VAL)                            #3. Scale
    return img

def loadSample_4():
    img = tiff.imread(os.getcwd()+"\\aerials\\aerial_4.tif")    #1. Load
    img = img[5000:5750,8800:9550]                              #2. Crop
    img = scale_image(img,SCALE_VAL)                            #3. Scale
    return img

def threshold(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                     #1. Gray
    img = cv2.GaussianBlur(img,(5,5),0)                             #2. Smooth
    ret,img = cv2.threshold(img,THRESH_VAL,255,cv2.THRESH_BINARY)   #3. Threshold
    return img                                                      #4. Return

def runwayWidth(markerCount):
    return (markerCount*1.95)+(markerCount*1.8)


def getCentroid(c):
    M = cv2.moments(c)                   #Get the image Moment
    cX = int(0)                          #Declare x cord as int
    cY = int(0)                          #Decalre y cord as int
    if M["m00"] != 0:                    #Check for zero before division
        cX = int(M["m10"] / M["m00"])        #Calc X coord of centroid
        cY = int(M["m01"] / M["m00"])        #Calc Y coord of centroid
    else:                                #Avoid divsion by zero, default (0,0)
        cX, cY = 0, 0
    return (cX,cY)                       #Return tuple of (x,y coordinate)

def visCntsCorners(visual,thresh):                                                      #Draw contours and lable with number of corners
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)   #1. Calculate Contours
    cnts = imutils.grab_contours(cnts)                                                  #2. Extract contour list from data object
    for c in cnts:                                                                      #3. For each contour
        cv2.drawContours(visual, [c], -1, (0, 255, 0), 2)                               #Draw the Contour on the origonal image
    for c in cnts:
        point=getCentroid(c)
        cv2.putText(visual, "{}".format(c.size), point, cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 2)
    return visual

def rangeClassifier(visual,thresh):        #Draw contours and lable with number of corners
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)   #1. Calculate Contours
    cnts = imutils.grab_contours(cnts)                                                  #2. Extract contour list from data object
    for c in cnts:                                                                      #3. For each contour
        if c.size>300 and c.size<670:                                                   #Vertex range for contours denoting Arrows and width markers
            cv2.drawContours(visual, [c], -1, (0, 255, 0), 2)                           #Draw the Contour on the origonal image
        else:                                                                           #Draw non relevant contours in red
            cv2.drawContours(visual, [c], -1, (0, 0, 255), 2)                           #Draw the Contour on the origonal image
    for c in cnts:                                                                      #Each contour set
        point=getCentroid(c)                                                            #Get its center
        cv2.putText(visual, "{}".format(c.size), point, cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 2) #Write number of vertexes
    return visual

def rectArea(rect):
    return rect[1][0]*rect[1][1]

def bndBoxClassifier(visual,thresh):        #Draw contours and lable with number of corners
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)   #1. Calculate Contours
    cnts = imutils.grab_contours(cnts)                                                  #2. Extract contour list from data object

    rects = []                                                                          #an array to store all relevant bounding boxes
    widthMarkersCenter = []
    for c in cnts: #Draw The contours                                                   #3. For each contour
        if c.size>300 and c.size<670:
            cv2.drawContours(visual, [c], -1, (0, 255, 0), 2)                               #Draw the Contour on the origonal image
            rect = cv2.minAreaRect(c)
            rects.append(rect)
    mean=0
    for rect in rects:#Find the mean
        mean=mean+rectArea(rect) #add the area of the current box to the sum
    mean = mean / len(rects)
    for rect in rects:
        box = cv2.boxPoints(rect)   #generates an narray of points representing rect
        box = np.int0(box)  #ALias to 64 bit integer for iterating
        #Catagorize and draw
        if(rect[1][0]*rect[1][1])<mean*1.1:#Classified as width markers
            cv2.drawContours(visual,[box],0,(0,0,255),2)
            widthMarkersCenter.append(rect[0])#[0]is center, [1]is dimensions, [2]is angle
        else:#Classified as Arrows
            cv2.drawContours(visual,[box],0,(255,0,0),2)
            #rects.remove(rect)  #It isnt a widht marker, so get rid of it


    #Only width markers are left in rects
    widthCount=len(widthMarkersCenter)
    if(widthCount%2!=0):
        widthCount=widthCount+1 #Must be an even number of width markers, corrects for one not picked up due to intersection with other markings
    #print("There are {} width markers on this runway".format(widthCount))
    bRCorner = (50,visual.shape[1]-50)
    cv2.putText(visual, "{} markers, width is {} m".format(widthCount, runwayWidth(widthCount)), bRCorner, cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 2)     #Count them, write width at bottom of screen

    #__FIT_A_LINE_TO_RUNWAY_AXIS__#
    #print(widthMarkersCenter)
    #widthMarkersCenter=cv2.UMat(np.array(widthMarkersCenter, dtype=np.float32))
    #print(widthMarkersCenter)
    #Convert Width markers to a tr<cv::UMat>


    #[dx,dy,x,y] = cv2.fitLine(widthMarkersCenter,  cv2.DIST_L2, 0, 0.1, 0.1)
    #left = int((-x*dy/dx) + y)
    #right = int(((visual.shape[1]-x)*dy/dx)+y)
    #cv2.line(visual,(visual.shape[1]-1,right),(0,left),255,2)
    #Fit line to centers of rectangles
    #Draw Line, reps runway axis
    return visual

    #size
    #angle
