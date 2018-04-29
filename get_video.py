import numpy as np
import cv2

cap = cv2.VideoCapture(1)
try:
	while(True):
		ret, frame = cap.read()
		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	 
		cv2.imshow('frame',frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break
		elif key == ord('a'):
			name = input('name: ')
			cv2.imwrite(r'C:\Users\ASUS\.spyder-py3\shoe\pic_0402\%s.png'%name, frame)
except:
	cap.release()
	cv2.destroyAllWindows()