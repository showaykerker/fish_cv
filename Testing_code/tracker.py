import numpy as np
import cv2
import matplotlib.pyplot as plt

def maskByROI_(frame, x, y, w, h):

	# Create the basic black image 
	mask = np.zeros(frame.shape, dtype = "uint8")
	# Draw a white, filled rectangle on the mask image
	cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)
	# Apply the mask and display the result
	#maskedImg = cv2.bitwise_and(frame, mask)
	maskedImg = frame[y:y+h+1,x:x+w+1]
	return maskedImg




(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.') 
 
if __name__ == '__main__' :
 
	# Set up tracker.
	# Instead of MIL, you can also use
 
	tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
	tracker_type = tracker_types[1]
 
	if int(minor_ver) < 3:
		tracker = cv2.Tracker_create(tracker_type)
	else:
		if tracker_type == 'BOOSTING':
			tracker = cv2.TrackerBoosting_create()
		if tracker_type == 'MIL':
			tracker = cv2.TrackerMIL_create()
		if tracker_type == 'KCF':
			tracker = cv2.TrackerKCF_create()
		if tracker_type == 'TLD':
			tracker = cv2.TrackerTLD_create()
		if tracker_type == 'MEDIANFLOW':
			tracker = cv2.TrackerMedianFlow_create()
		if tracker_type == 'GOTURN':
			tracker = cv2.TrackerGOTURN_create()
 
	# Read video
	video = cv2.VideoCapture(r'D:\Users\ASUS\Desktop\fish\paul-1.mpeg')
 
	# Exit if video not opened.
	if not video.isOpened():
		print("Could not open video")
		sys.exit()
 
	# Read first frame.
	ok, frame = video.read()
	
	equ = cv2.equalizeHist(frame[:,:,0])
	frame = np.hstack((frame[:,:,0], equ))
	
	if not ok:
		print('Cannot read video file')
		sys.exit()
	 
	# Define an initial bounding box
	bbox = (287, 23, 86, 320)
 
	# Uncomment the line below to select a different bounding box
	bbox = cv2.selectROI(frame, False)
 
	# Initialize tracker with first frame and bounding box
	ok = tracker.init(frame, bbox)
 
	while True:
		# Read a new frame
		ok, frame = video.read()

		equ = cv2.equalizeHist(frame[:,:,0])
		frame = np.hstack((frame[:,:,0], equ))
		if not ok:
			break
		 
		# Start timer
		timer = cv2.getTickCount()
 
		# Update tracker
		ok, bbox = tracker.update(frame)
 
		# Calculate Frames per second (FPS)
		fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
		# Draw bounding box
		if ok:
			# Tracking success
			p1 = (int(bbox[0]), int(bbox[1]))
			p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
			cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
		else :
			# Tracking failure
			cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
		# Display tracker type on frame
		cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
	 
		# Display FPS on frame
		cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
		# Display result
		cv2.imshow("Tracking", frame)
 
		# Exit if ESC pressed
		k = cv2.waitKey(1) & 0xff
		if k == 27 : break