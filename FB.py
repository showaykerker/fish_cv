import numpy as np
import cv2

# http://lab.toutiao.com/index.php/2017/04/04/farneback-guangliusuanfaxiangjieyu-calcopticalflowfarneback-yuanmafenxi.html




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
	#maskedImg = frame[y:y+h+1,x:x+w+1]
	maskedImg = frame[x:x+w, y:y+h]
	return maskedImg

cap = cv2.VideoCapture(r'D:\Users\ASUS\Desktop\fish\vid2\fish2-0-10.mpeg')

########################################## Select Frame to Mark ##########################################
ret, frame = cap.read()
frame_ = frame.copy()
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(frame_,'Press \'n\' to go next frame, \'k\' to select frame',(10,20), font, 0.4,(255,255,255), 1,cv2.LINE_AA)
while(1):

	cv2.imshow('select',frame_)
	key = cv2.waitKey(0) & 0xFF
	if key == ord('n'): 
		ret, frame = cap.read()
		frame_ = frame.copy()
		cv2.putText(frame_,'Press \'n\' to go next frame, \'k\' to select frame',(10,20), font, 0.4,(255,255,255), 1,cv2.LINE_AA)
		continue
	elif key == ord('k'): break

cv2.destroyWindow('select')
########################################## Mark Selected Frame  ##########################################
frame = frame[:,:,0]
frame_ = frame.copy()
cv2.putText(frame_,'Press space to continue.',(10,20), font, 0.4,(255,255,255), 1,cv2.LINE_AA)
bbox = cv2.selectROI(frame_, False)

cv2.destroyAllWindows()

########################################## Convert ROI position ##########################################
Boundary = bbox 
Bx, By, Bw, Bh = Boundary
B_TopLeft = (Bx, By)
B_BottomRight = (Bx+Bw, By+Bh)



ret, frame1 = cap.read()
frame1 = maskByROI_(frame1, By, Bx, Bh, Bw)
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 127


# Start timer
#timer = cv2.getTickCount()


# Calculate Frames per second (FPS)
#fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);


while(1):

	# Start timer
	timer = cv2.getTickCount()
 

	ret,frame2_ = cap.read()
	try: 
		if frame2_ == None: break
	except: pass
	
	empty = np.zeros(frame2_.shape, dtype=frame2_.dtype)
	
	frame2 = maskByROI_(frame2_, By, Bx, Bh, Bw)
	next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
	
	#      cv2.calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags) → flow
	flow = cv2.calcOpticalFlowFarneback(prvs, next, None,       0.5,      3,      15,          3,      7,        1.5,     0) 
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
	bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	empty[Boundary[1]:Boundary[1]+Boundary[3], Boundary[0]:Boundary[0]+Boundary[2], :] = bgr
	
	output = cv2.addWeighted(frame2_, 1, empty, 0.8, 0)

	
	
	
	# Calculate Frames per second (FPS)
	fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
	
	
	cv2.rectangle(output,B_TopLeft,B_BottomRight,(255,0,0),1)
	cv2.putText(output, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 1);
	
	cv2.imshow('output', output)
	cv2.moveWindow("output", 0, 40)
	k = cv2.waitKey(1) & 0xff
	if k == 27: break
	prvs = next
	
	
cv2.destroyAllWindows()
cap.release()