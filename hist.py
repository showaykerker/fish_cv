import numpy as np
import cv2
import time
import copy
import math

px, py = 0, 0

kernel_size = ks = 5
w = h = ks
if ks%2 != 1 or ks<=1 : raise ValueError('Kernel Size must be 2n+3, where n is a positive integer.')

cap = cv2.VideoCapture(r'D:\Users\ASUS\Desktop\fish\0423-1.avi')

def process_frame_(frame_):
	#try:
	#	equ = cv2.equalizeHist(frame_[:,:,0])
	#except:
	#	equ = cv2.equalizeHist(frame_)
	blur = cv2.GaussianBlur(frame_, (15,15), 0)
	#blur = cv2.equalizeHist(blur)
	return blur

def getK(frame, px, py):
	ret = np.array(frame[ px-int(ks/2):px+int(ks+1/2) , py-int(ks/2):py+int(ks+1/2) ], dtype=np.float32)
	
	#if ret.shape != (ks+2, ks+2):
	#	raise RuntimeError('frame.shape=%s, ret.shape=%s, px=%s, py=%s, int(ks/2)=%s, int(ks+1/2)=%s'%(str(frame.shape), str(ret.shape), str(px), str(py), str(int(ks/2)), str(int(ks+1/2))))
	return ret

def Optical_Flow_Calculation(K, last_K):
	# https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
	
	print('K.shape=', K.shape)

	fx = np.zeros((ks-2, ks-2), dtype=np.float32)
	fy = np.zeros((ks-2, ks-2), dtype=np.float32)
	ft = np.zeros((ks-2, ks-2), dtype=np.float32)
	
	for i in range(1, ks-1):
		for j in range(1, ks-1):
			try:
				fx[i-1][j-1] = (K[i][j+1]-K[i][j-1])/2 
				fy[i-1][j-1] = (K[i+1][j]-K[i-1][j])/2 
				ft[i-1][j-1] = K[i][j]-last_K[i][j]
			except:
				
				raise RuntimeError('Out of Range.')

	Sumfx2 = sum([xi**2 for xi in fx.flatten()])
	Sumfy2 = sum([yi**2 for yi in fy.flatten()])
	Sumfxfy = sum([xi*yi for xi, yi in zip(fx.flatten(), fy.flatten())])
	n_Sumfxft = -1 * sum([xi*ti for xi, ti in zip(fx.flatten(), ft.flatten())])
	n_Sumfyft = -1 * sum([yi*ti for yi, ti in zip(fy.flatten(), ft.flatten())])
	
	T1 = (np.array([[Sumfx2, Sumfxfy],[Sumfxfy, Sumfy2]]))**-1
	T2 = np.array([[n_Sumfxft],[n_Sumfyft]])
	u, v = T1.dot(T2).flatten()
	u_, v_ = np.clip(u, -2*ks, 2*ks), np.clip(v, -2*ks, 2*ks)
	#u_, v_ = np.clip(u, -2, 2), np.clip(v, -2, 2)
	if math.isnan(u_): u_ = 0
	if math.isnan(v_): v_ = 0
	if math.isnan(u): u = 0
	if math.isnan(v): v = 0
	if math.isinf(u): u = 9999
	if math.isinf(v): v = 9999
	print('unclipped:', int(u), int(v),' clipped:', int(u_), int(v_))
	return (int(u_), int(v_))



########################################## Select Frame to Mark ##########################################
ret, frame = cap.read()
frame_ = process_frame_(copy.deepcopy(frame))
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(frame_,'Press \'n\' to go next frame, \'k\' to select frame',(10,20), font, 0.4,(255,255,255), 1,cv2.LINE_AA)
while(1):

	cv2.imshow('select',frame_)
	key = cv2.waitKey(0) & 0xFF
	if key == ord('n'): 
		ret, frame = cap.read()
		frame_ = process_frame_(copy.deepcopy(frame))
		cv2.putText(frame_,'Press \'n\' to go next frame, \'k\' to select frame',(10,20), font, 0.4,(255,255,255), 1,cv2.LINE_AA)
		continue
	elif key == ord('k'): break

########################################## Mark Selected Frame  ##########################################
frame = frame[:,:,0]
frame_ = process_frame_(copy.deepcopy(frame))
cv2.putText(frame_,'Press space to continue.',(10,20), font, 0.4,(255,255,255), 1,cv2.LINE_AA)
bbox = cv2.selectROI(frame_, False)
px, py = int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2)

########################################## Make Kernel          ##########################################
last_K = np.zeros((kernel_size, kernel_size))
K = np.zeros((kernel_size, kernel_size))
last_K = getK(frame_, px, py)

#input('Press Enter to Start Tracking')

i = 0

while(True):
	ret, frame = cap.read()
	frame = frame[:,:,0]
	frame_ = copy.deepcopy(frame)
	#blur = process_frame_(frame_)
	blur = frame_

	K = getK(blur, px, py)
	u, v = Optical_Flow_Calculation(K, last_K)
	px = np.clip(px+u, 0+int(w/2), blur.shape[0]-int(w/2))
	py = np.clip(py+v, 0+int(w/2), blur.shape[1]-int(w/2))
	#input((px, py))
	last_K = copy.deepcopy(K)
	
	cv2.rectangle(frame_,(int(px-w/2), int(py-w/2)),(int(px+w/2), int(py+w/2)),(255,255,255),1)
	cv2.rectangle(blur,(int(px-w/2), int(py-w/2)),(int(px+w/2), int(py+w/2)),(255,255,255),1)

	res = np.hstack((frame_, blur))
	#res = np.hstack((equ, edges))
	#res = np.hstack((res, lap))
	cv2.imshow('frame',res)
	time.sleep(0.04)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		input()
	elif key == ord('a'):
		name = input('name: ')
		cv2.imwrite(r'C:\Users\ASUS\.spyder-py3\shoe\pic_0402\%s.png'%name, frame)
	
	

cap.release()
cv2.destroyAllWindows()