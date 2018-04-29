import numpy as np
import cv2

# http://lab.toutiao.com/index.php/2017/04/04/farneback-guangliusuanfaxiangjieyu-calcopticalflowfarneback-yuanmafenxi.html

input('Press Enter to Start')
Boundary = 0, 20, 620, 460
Bx, By, Bw, Bh = Boundary
B_TopLeft = (By, Bx)
B_BottomRight = (By+Bw, Bx+Bh)

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

cap = cv2.VideoCapture(r'D:\Users\ASUS\Desktop\fish\vid2\fish2-0-10.mpeg')
ret, frame1 = cap.read()

frame1 = maskByROI_(frame1, Bx, By, Bw, Bh)

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 127

fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter(r'D:\Users\ASUS\Desktop\fish\fish-out.avi',fourcc, 20.0, (480, 384))



while(1):
	ret,frame2_ = cap.read()
	
	empty = np.zeros(frame2_.shape, dtype=frame2_.dtype)
	
	frame2 =  maskByROI_(frame2_, 0, 20, 620, 460)
	next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
	
	flow = cv2.calcOpticalFlowFarneback(prvs, next, 
										None, # pyr_scale
										0.5, # levels 0.5
										3, # winsize 3
										15, # iterations 15
										7, # poly_n 3
										1, # poly_sigma 5
										1.2, 0) # flags 1.2, 0
	#print(flow.shape)
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
	bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	empty[Boundary[0]:Boundary[0]+Boundary[3], Boundary[1]-1:Boundary[2]+Boundary[1], :] = bgr
	
	output = cv2.addWeighted(frame2_, 1, empty, 0.8, 0)

	#output = maskByROI_(output, x, y, w, h)
	cv2.rectangle(output,B_TopLeft,B_BottomRight,(0,255,0),3)

	#cv2.imshow('origin', frame2_)
	#cv2.imshow('frame2', bgr)
	cv2.imshow('output', output)
	#out.write(output)
	
	k = cv2.waitKey(1) & 0xff
	if k == 27: break
	prvs = next
	

	
cv2.destroyAllWindows()
cap.release()