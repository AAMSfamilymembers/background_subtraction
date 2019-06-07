import cv2
import numpy as np
 
vid = cv2.VideoCapture(0)
_,f = vid.read()
f = cv2.resize(f,(320,240), interpolation = cv2.INTER_CUBIC)

acc = np.zeros([f.shape[0],f.shape[1],3,100],dtype = 'uint8')
mean = np.zeros([f.shape[0],f.shape[1],3])
var = np.zeros([f.shape[0],f.shape[1],3])
power = np.zeros([f.shape[0],f.shape[1],3])
std = np.zeros([f.shape[0],f.shape[1],3])
diff = np.zeros([f.shape[0],f.shape[1],3])

bg = np.zeros([f.shape[0],f.shape[1],3])
fg = np.zeros([f.shape[0],f.shape[1],3])

	
alpha = 0.01

for i in range(100):
	_,f = vid.read()
	f = cv2.resize(f,(320,240), interpolation = cv2.INTER_CUBIC)
	acc[:,:,:,i] = f[:,:,:]

mean = np.mean(acc , axis = 3)
var = 1125.100*np.var(acc , axis = 3)
std = np.sqrt(var)
diff = cv2.absdiff(np.float64(f),mean)
square = np.square(diff)
power = np.divide(diff,std)

k=0
while(1):
	bgind = np.where(power<=2.5)
	fgind = np.where(power>2.5)
	if k>=100:	
		k=0
		mean = np.mean(acc , axis = 3)
		var = 1125.100*np.var(acc , axis = 3)	
	else:
		mean[bgind] = alpha*(f[bgind]) + ((1-alpha) * mean[bgind])
		var[bgind] = (alpha*(square[bgind])) + ((1-alpha)*var[bgind])			
	f = np.uint8(f)		
	
	bg[bgind] = f[bgind]
	fg[fgind] = f[fgind]
	
	fg =np.uint8(fg)
	bg =np.uint8(bg)
	
	
	_,f = vid.read()	
	f = cv2.GaussianBlur(f,(5,5),0)
	f = cv2.resize(f,(320,240), interpolation = cv2.INTER_CUBIC)
	acc[:,:,:,k] = f[:,:,:]
	k = k+1
	std = np.sqrt(var)
	ind = np.where(std==0)
	std[ind] = 0.01
	diff = cv2.absdiff(np.float64(f),mean)
	square = np.square(diff)
	power = np.divide(diff,std)
	cv2.imshow('img',f)
	cv2.imshow('fg',fg)
	
	cv2.imshow('bg',bg)
	fg = np.zeros([f.shape[0],f.shape[1],3]) 
	if cv2.waitKey(1) & 0xFF == ord('q'):
      		   break 
 
cv2.destroyAllWindows()
vid.release() 

