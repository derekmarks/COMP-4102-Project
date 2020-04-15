from funct import *

def visContours(windowName,img):
    thresh = threshold(img)    #Threshold the image
    contour = bndBoxClassifier(img,thresh)      #The final method for counting threshold markings

    #contour = rangeClassifier(img,thresh)      #Shows how contour set length can be used to filter out undesriable markings
    #contour = visCntsCorners(img,thresh)       #Shows contour detection after thresholding and number of vertex in each contour
    #cv2.imshow(windowName+"_Thresh",thresh)    #Show the thresholded image
    cv2.imshow(windowName, contour)             #Show the contour image

img_1 = loadSample_1()
img_2 = loadSample_2()
img_3 = loadSample_3()
img_4 = loadSample_4()


visContours("Img_1",img_1)  #Run marking detection on each image
visContours("Img_2",img_2)
visContours("Img_3",img_3)
visContours("Img_4",img_4)


cv2.waitKey(0)
