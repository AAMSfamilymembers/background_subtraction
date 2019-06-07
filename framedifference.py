import cv2
import numpy

video = cv2.VideoCapture(0)
while(1):
	ret,frame2 = video.read()
	cv2.waitKey(10)
	ret3,frame3 = video.read()
	if(ret == False):
		break
	grayframe2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
	grayframe2 = cv2.GaussianBlur(grayframe2,(5,5),0)

	grayframe3 = cv2.cvtColor(frame3,cv2.COLOR_BGR2GRAY)
	grayframe3 = cv2.GaussianBlur(grayframe3,(5,5),0)
	framediff = cv2.absdiff(grayframe2,grayframe3)

	ret1,thresh = cv2.threshold(framediff,20,255,cv2.THRESH_BINARY)
	thresh = cv2.erode(thresh,(5,5),iterations=5)						
	res = cv2.bitwise_and(frame2,frame2,mask = thresh)
	
	cv2.imshow("thresh",thresh)
	cv2.imshow("framediffrgb",res)
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break
video.release()
cv2.destroyAllWindows()



