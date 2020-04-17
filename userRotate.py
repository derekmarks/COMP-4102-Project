#Robin Wohlfarth
#100 847 725
from funct import scale_image
import numpy as np
import tifffile as tiff
import imutils
import time
import cv2

anglep = None


def click(event, x, y, flags, p):

    global anglep

    if event == cv2.EVENT_LBUTTONDOWN:
        print("Leftclick", p)
        anglep = p
        print("anglep", anglep)

# 
def rotateit(imagePath, scaleFactor):


    image = tiff.imread(imagePath)
    angle = 0
    image = scale_image(image,scaleFactor)  
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #ret, gray = cv2.threshold(gray,203,255,cv2.THRESH_BINARY)

    #Rotates the image 
    while True:
        rotated = imutils.rotate_bound(image, angle)
        cv2.imshow("Rotate", rotated)
        cv2.setMouseCallback("Rotate", click, angle)
        cv2.waitKey(0)

        if angle == 360:
            angle = 0
        angle = angle + 1 

        if not(anglep is None):
            break

    cv2.destroyWindow("Rotate")
    return anglep