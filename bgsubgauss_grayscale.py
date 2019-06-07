import cv2
import numpy as np
 
vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FPS, 30)
_,f = vid.read()
f = cv2.resize(f,(320,240), interpolation = cv2.INTER_CUBIC)
global result,diff,mean,var,gray,square,power
result = np.zeros([f.shape[0],f.shape[1]])
square = np.zeros([f.shape[0],f.shape[1]])
diff = np.zeros([f.shape[0],f.shape[1]])
power = np.zeros([f.shape[0],f.shape[1]])
acc = np.zeros([f.shape[0],f.shape[1],60],dtype = 'uint8')
mean = np.zeros([f.shape[0],f.shape[1]])
var = np.zeros([f.shape[0],f.shape[1]])
bg = np.zeros([f.shape[0],f.shape[1]])
fg = np.zeros([f.shape[0],f.shape[1]])
global pi
pi=3.14
def gaussian():	
	global result,diff,mean,var,gray,square,power,pi	
	diff = gray-mean
	
	square = (-1)*np.square(diff)			
	variance2 = 2*var
	power = np.divide(square,variance2)
	e = np.exp(power)
	
	variance2 = 2*pi * variance2
	div = np.sqrt(variance2)
	result = np.divide(e,div)
	power = np.divide(np.abs(diff),np.sqrt(var))


avg1 = np.zeros([f.shape[0],f.shape[1]],dtype = 'uint8')

alpha = 0.04
for i in range(60):
	_,f = vid.read()
	f = cv2.resize(f,(320,240), interpolation = cv2.INTER_CUBIC)
	gray = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)	
	acc[:,:,i] = gray
mean = np.mean(acc , axis = 2)
var = np.var(acc , axis = 2)
gaussian()	
while(1):
	bgind = np.where(power<=2.5)
	fgind = np.where(power>2.5)
	mean[bgind] = alpha*(gray[bgind]) + ((1-alpha) * mean[bgind])
	
	var[bgind] = ((-1)*alpha*(square[bgind])) + ((1-alpha)*var[bgind])	
	gray =np.uint8(gray)	
	bg[bgind] = gray[bgind]
	
	fg[fgind] = gray[fgind]
	fg =np.uint8(fg)
	bg =np.uint8(bg)
	_,f = vid.read()	
	f = cv2.resize(f,(320,240), interpolation = cv2.INTER_CUBIC)
	
	gray = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
	
	gaussian()		
	gray =np.uint8(gray)
	
	cv2.imshow('img',gray)
	cv2.imshow('fg',fg)
	
	cv2.imshow('bg',bg)
	fg = np.zeros([f.shape[0],f.shape[1]]) 
	if cv2.waitKey(1) & 0xFF == ord('q'):
      		   break 
 
cv2.destroyAllWindows()
vid.release() 

