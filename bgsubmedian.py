import cv2
import numpy as np
 
vid = cv2.VideoCapture(0)
_,fr = vid.read()
i = 0
avg1 = np.zeros([fr.shape[0],fr.shape[1],3],dtype = 'uint8')
bg = np.zeros([fr.shape[0],fr.shape[1],3],dtype = 'uint8')
while(1):
	_,f = vid.read()
	
	inc = np.where(f>bg)
	bg[inc] = bg[inc] +  1
	dec = np.where(f<bg)
	bg[dec] = bg[dec] -  1
	avg1 =  cv2.absdiff(f,bg) 
	ret,thresh1 = cv2.threshold(cv2.cvtColor(avg1,cv2.COLOR_BGR2GRAY),20,255,cv2.THRESH_BINARY)
	avg2 = cv2.bitwise_and(f,f,mask = thresh1)	
	i = i+1
	cv2.imshow('img',f)
	cv2.imshow('mean',bg)
	cv2.imshow('threshold',thresh1)
	cv2.imshow('avg2',avg2) 
	if cv2.waitKey(1) & 0xFF == ord('q'):
      		   break 
 
cv2.destroyAllWindows()
vid.release() 

