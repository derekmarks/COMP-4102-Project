#Derek Marks
#100 974 560
#14 Apr 2020

import cv2
import tifffile as tiff
import numpy as np
import os
import math
import imutils


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

def runwayWidth(markerCount):                            #Takes the number of markers
    return (markerCount*1.95)+(markerCount*1.8)          #Returns the width of the airefeild in meters


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

def rectArea(rect):                             #Takes a rect object, returns its area A=(WxH)
    return rect[1][0]*rect[1][1]

def bndBoxClassifier(visual,thresh):        #Classifies Chevrons from threshold markings
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)   #1. Calculate Contours
    cnts = imutils.grab_contours(cnts)                                                  #2. Extract contour list from data object

    rects = []                                                                          #an array to store all relevant bounding boxes
    widthMarkersCenter = []
    for c in cnts: #Draw The contours                                       #3. For each contour
        if c.size>300 and c.size<670:
            cv2.drawContours(visual, [c], -1, (0, 255, 0), 2)               #3.1 Draw the Contour on the origonal image
            rect = cv2.minAreaRect(c)                                       #3.2 Calclulate the min area bounding box
            rects.append(rect)
    mean=0
    for rect in rects:                                                      #4.1 Find the mean
        mean=mean+rectArea(rect)                                            #4.2 add the area of the current box to the sum
    mean = mean / len(rects)                                                #4.3 divide by num bounding boxes to find mean
    for rect in rects:
        box = cv2.boxPoints(rect)       #generates an narray of points representing rect
        box = np.int0(box)              #Alias to 64 bit integer for iterating
        #Catagorize and draw
        if(rect[1][0]*rect[1][1])<mean*1.1:     #Classified as threshold markers
            cv2.drawContours(visual,[box],0,(0,0,255),2)        #Draw the bounding box in red
            widthMarkersCenter.append(rect[0])                  #Save the centroid of the contour group
        else:                                   #Classified as chevrons
            cv2.drawContours(visual,[box],0,(255,0,0),2)        #Draw the bounding box in blue
   


    
    widthCount=len(widthMarkersCenter)  #Count the number of width markers in the width marker array
    if(widthCount%2!=0):                #If there is an odd number of width markers, then round up
        widthCount=widthCount+1 #Must be an even number of width markers, corrects for one not picked up due to intersection with other markings
    bRCorner = (50,visual.shape[1]-50)  #Get the bottom right corner of the image
    cv2.putText(visual, "{} markers, width is {} m".format(widthCount, runwayWidth(widthCount)), bRCorner, cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 2)     #Count them, write width at bottom of screen
    return visual       #write the runway width on the images bottom right corner, and return it
