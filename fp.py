from funct import *
from findScale import *

def visContours(windowName,img):
    thresh = threshold(img)    #Threshold the image
    contour = bndBoxClassifier(img,thresh)      #The final method for counting threshold markings

    #contour = rangeClassifier(img,thresh)      #Shows how contour set length can be used to filter out undesriable markings
    #contour = visCntsCorners(img,thresh)       #Shows contour detection after thresholding and number of vertex in each contour
    cv2.imshow(windowName+"_Thresh",thresh)    #Show the thresholded image
    cv2.imshow(windowName, contour)             #Show the contour image

img_1 = loadSample_1()  #Quickly load some image samples wihout using the template finder, for sake of time
img_2 = loadSample_2()
img_3 = loadSample_3()
img_4 = loadSample_4()

visContours("Img_1",img_1) #Display the output of the classifier on sampel images
visContours("Img_2",img_2)
visContours("Img_3",img_3)
visContours("Img_4",img_4)


img_5, startX, startY, endX, endY = matchit(os.getcwd()+"\\aerials\\aerial_1-r.tif", False, 0) # Runs the matchit function. This template matches and returns the cordinates of a box that best correlates to the templates
img_5 = img_5[startY -200 :endY +200 , startX -200 :endX +200] # Takes the box of best correlated, expands its size to include more context in the image and then crops out the rest
img_5 = imutils.rotate_bound(img_5, 60) # Rotates it 60 degrees to match the original aerial_1

visContours("Img_5",img_5) # Runs the croped and scaled image through the Countours to find its shape

cv2.waitKey(0)
