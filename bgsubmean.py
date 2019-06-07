import cv2
import numpy as np
 
vid = cv2.VideoCapture(0)
_,fr = vid.read()
avg1 = np.zeros([fr.shape[0],fr.shape[1]],dtype = 'uint8')
i = 0
avg1 = np.zeros([fr.shape[0],fr.shape[1]],dtype = 'uint8')
alpha =0.06
mean = np.zeros([fr.shape[0],fr.shape[1]],dtype = 'uint8')
while(1):
	_,f = vid.read()
	gray = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)	
	mean = ((1-alpha)*mean) + (alpha*gray)
	mean = mean.astype(np.uint8)
	avg1 =  cv2.absdiff(gray,mean) 
	thresh1 = cv2.adaptiveThreshold(avg1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
	avg2 = cv2.bitwise_and(gray,gray,mask = thresh1)	
	i = i+1
	cv2.imshow('img',f)
	cv2.imshow('mean',mean)
	cv2.imshow('threshold',thresh1)
	cv2.imshow('avg2',avg2) 
	if cv2.waitKey(1) & 0xFF == ord('q'):
      		   break 
 
cv2.destroyAllWindows()
vid.release() 

