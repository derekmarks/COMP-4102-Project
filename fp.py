from funct import *
from findScale import *

def visContours(windowName,img):
    thresh = threshold(img)    #Threshold the image
    contour = bndBoxClassifier(img,thresh)      #The final method for counting threshold markings

    #contour = rangeClassifier(img,thresh)      #Shows how contour set length can be used to filter out undesriable markings
    #contour = visCntsCorners(img,thresh)       #Shows contour detection after thresholding and number of vertex in each contour
    cv2.imshow(windowName+"_Thresh",thresh)    #Show the thresholded image
    cv2.imshow(windowName, contour)             #Show the contour image

img_1 = loadSample_1()
img_2 = loadSample_2()
img_3 = loadSample_3()
img_4 = loadSample_4()

visContours("Img_1",img_1)
visContours("Img_2",img_2)
visContours("Img_3",img_3)
visContours("Img_4",img_4)


img_5, startX, startY, endX, endY = matchit(os.getcwd()+"\\aerials\\aerial_1-r.tif", False)
print("Y: ", (endY +200) - (startY -200) , " X: ", (endX +200) - (startX -200) )
img_5 = img_5[startY -200 :endY +200 , startX -200 :endX +200]
img_5 = imutils.rotate_bound(img_5, 60)

cv2.imshow("img_5", img_5)
visContours("Img_5",img_5) #This is broken because a div by zero occurs in funct line 108

cv2.waitKey(0)
