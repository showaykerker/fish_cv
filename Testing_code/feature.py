import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(r'D:\Users\ASUS\Desktop\fish\fish.mpeg')

def maskByROI_(frame, x, y, w, h):
	#x, y = 190, 220
	#w, h = 190, 200
	
	#x, y = 128, 96
	#w, h = 512, 384
	
	# Create the basic black image 
	mask = np.zeros(frame.shape, dtype = "uint8")
	# Draw a white, filled rectangle on the mask image
	cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)
	# Apply the mask and display the result
	#maskedImg = cv2.bitwise_and(frame, mask)
	maskedImg = frame[y:y+h+1,x:x+w+1]
	return maskedImg

def doLaplacian(frame):
	gray_lap=cv2.Laplacian(frame ,cv2.CV_16S,ksize=3) #拉式算子  
	frame=cv2.convertScaleAbs(gray_lap)  
	return frame

	
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
	
i = 0

k25 = cv2.getStructuringElement(cv2.MORPH_RECT,(25, 25))	
k5 = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))	
k3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
k2 = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
k1 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 1))

while(1):
	ret ,frame = cap.read()
	if ret == True:
		
	
		
		cvt_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		adj_gamma = adjust_gamma(cvt_img, gamma=1)
		#img = cv2.threshold(img, 15, 255, cv2.THRESH_TOZERO)[1]
		
		hist,bins = np.histogram(adj_gamma.flatten(),256,[0,256])
		cdf = hist.cumsum()
		#plt.hist(adj_gamma.flatten(),256,[0,256], color = 'r')
		#plt.show()
		cdf_normalized = cdf * hist.max() / cdf.max()
		
		cdf_m = np.ma.masked_equal(cdf,0)
		cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
		cdf = np.ma.filled(cdf_m,0).astype('uint8')
		
		cdf_after = cdf[adj_gamma]
		
		x, y, w, h = 128, 96, 500, 384
		
		roi = maskByROI_(cdf_after, x, y, w, h)
		adj_gamma_2 = adjust_gamma(roi, gamma=0.8)
		
		#Gau_blur = cv2.GaussianBlur(adj_gamma_2, (7,7), 15, 15)
		Med_blur = cv2.medianBlur(adj_gamma_2, 7)
		
		'''
		roi_lap = doLaplacian(roi)
		#roi_lap = cv2.Canny(roi,100,200)
		roi_lap = cv2.threshold(roi_lap, 10, 255, cv2.THRESH_BINARY)[1]
		roi_lap_2 = cv2.dilate(roi_lap, k2)
		roi_lap_2 = cv2.erode(roi_lap_2, k2)
		roi_lap_2 = cv2.dilate(roi_lap_2, k1)
		roi_lap_2 = cv2.erode(roi_lap_2, k2)
		roi_lap_2 = cv2.bilateralFilter(roi_lap_2, 9, 100, 100)

		roi_lap_2 = cv2.morphologyEx(roi_lap_2, cv2.MORPH_OPEN, k1)
		'''
		
		k = cv2.waitKey(60) & 0xff
		if k == 27: break
		else: 
			#cv2.imshow('origin', frame)
			#cv2.imshow('cvtColor', cvt_img)
			#cv2.imshow('adjust_gamma', adj_gamma)
			#cv2.imshow('cdf_normalized', cdf_after)
			cv2.imshow('adj_gamma_2', adj_gamma_2)
			#cv2.imshow('Gau_blur', Gau_blur)
			cv2.imshow('Med_blur', Med_blur)
			
			#cv2.imshow('roi', roi)
			#cv2.imshow('roi_lap', roi_lap)
			#cv2.imshow('roi_lap2', roi_lap_2)

	else: break
	
cv2.destroyAllWindows()
cap.release()