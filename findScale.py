#Robin Wohlfarth
#100 847 725

import numpy as np
import tifffile as tiff
import imutils
import glob
import cv2

# 
def matchit(imagePath, vis):

	found = None

	# 
	for templatePath in glob.glob("template" + "/*.png"):

		template = cv2.imread(templatePath)
		tump,template = cv2.threshold(template,240,255,cv2.THRESH_BINARY)
		template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		(tH, tW) = template.shape[:2]

		if vis:
			cv2.imshow("Template", template)
			#cv2.waitKey(0)
		

		
		# 
		image = tiff.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		ret, gray = cv2.threshold(gray,203,255,cv2.THRESH_BINARY)

		#Rotates the image 360/24=15 times
		#for angle in np.arange(0, 360, 24):
			#rotated = imutils.rotate_bound(gray, angle)
			#cv2.imshow("Visualize", rotated)
			#cv2.waitKey(0)

		# Scales the image 20 times
		for scale in np.linspace(0.2, 1.0, 20)[::-1]:
			# 
			resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
			r = gray.shape[1] / float(resized.shape[1])

			# 
			if resized.shape[0] < tH or resized.shape[1] < tW:
				break


			result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF)
			(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

			if vis:
				cv2.imshow("result", result)
				#cv2.waitKey(0)

			if vis:
				clone = np.dstack([resized, resized, resized])
				cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
				clone = clone[maxLoc[1]  :(maxLoc[1] + tH)  , maxLoc[0]  :(maxLoc[0] + tW) ]  
				cv2.imshow("Visualize", clone)
				cv2.waitKey(0)

			# 
			if found is None or maxVal > found[0]:
				found = (maxVal, maxLoc, r)

			# ..

	(_, maxLoc, r) = found
	(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
	(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

	# draw
	if vis: 
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
		cv2.imshow( "image" , image)
		cv2.waitKey(0)

	return image, startX, startY, endX, endY