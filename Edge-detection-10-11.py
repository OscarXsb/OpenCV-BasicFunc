import cv2

import numpy as np

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# 0代表默认摄像头

cap.open(0)

while cap.isOpened():
	flag, frame = cap.read()
	#返回一帧画面
	if not flag:
		break

	key_pressed = cv2.waitKey(60)
	print('The key you pressed is: ', key_pressed)

	frame = cv2.Canny(frame,100,200)#边缘检测

	#摞成三通道图像
	frame = np.dstack((frame,frame,frame))

	cv2.imshow('my_window',frame)
	if key_pressed == 27:
		break

cap.release()

cv2.destroyAllWindows()