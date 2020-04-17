#Robin Wohlfarth
#100 847 725

import numpy as np
import tifffile as tiff
import imutils
import glob
import cv2

# Function matchit takes in An image path, A flag to turn visuals on and off, and a angle
def matchit(imagePath, vis, angle):

	found = None																		#Varaible used to hold highest correlation 

	image = tiff.imread(imagePath)														#Read image in
	rot = imutils.rotate_bound(image, angle)											#Rotation of the image if required

	gray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)										#Color Space Conversions
	ret, gray = cv2.threshold(gray,203,255,cv2.THRESH_BINARY)							#Binary threshold of the image to seperate the markings from the runway

	# 

	for templatePath in glob.glob("template" + "/*.png"): 								#For every png in the template folder 

		template = cv2.imread(templatePath)												#read in the template
		tump,template = cv2.threshold(template,240,255,cv2.THRESH_BINARY)				#Binary threshold the template
		template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) 							#Color Space Conversions
		(tH, tW) = template.shape[:2]													#Defining tH and TW

		if vis:																			#Visual output if the flag is set to true
			cv2.imshow("Template", template)
			cv2.waitKey(0)
	


		
		for scale in np.linspace(0.2, 1.0, 20)[::-1]:									# Scales the image 20 times between 0.2-1.0
			# 
			resized = imutils.resize(gray, width = int(gray.shape[1] * scale))			#Define resized
			r = gray.shape[1] / float(resized.shape[1])									#defining r

			# 
			if resized.shape[0] < tH or resized.shape[1] < tW:							
				break


			result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF)				#Template matching to the rezied img
			(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)								#Defines the highest correlation and its location 

			if vis:
				cv2.imshow("result", result)
				cv2.waitKey(0)

			if vis:																		#Visual output if the flag is set to true
				clone = np.dstack([resized, resized, resized])
				cv2.rectangle(clone, (maxLoc[0] , maxLoc[1]) , (maxLoc[0] + tW , maxLoc[1] + tH) , (0, 0, 255), 2)
				clone = clone[maxLoc[1]  :(maxLoc[1] + tH)  , maxLoc[0]  :(maxLoc[0] + tW) ]  
				cv2.imshow("Visualize", clone)
				cv2.waitKey(0)

			# 
			if found is None or maxVal > found[0]:										#Correlation compared to previous best 
				found = (maxVal, maxLoc, r)

			# ..

	(_, maxLoc, r) = found
	(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))							#Defines the X and Y of the box
	(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))				#Defines the other  X and Y of the box

	# draw
	if vis: 
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)			#Visual output if the flag is set to true
		cv2.imshow( "image" , image)
		cv2.waitKey(0)

	return image, startX, startY, endX, endY											#Returns image and location of template with best Correlation