import cv2
import numpy as np

vid = cv2.VideoCapture(0)
_, f = vid.read()
f = cv2.resize(f, (320, 240), interpolation=cv2.INTER_CUBIC)

acc = np.zeros([f.shape[0], f.shape[1], 3], dtype='uint8')
mean = np.zeros([f.shape[0], f.shape[1], 3])
var = 100 * np.ones([f.shape[0], f.shape[1], 3])
power = np.zeros([f.shape[0], f.shape[1], 3])
std = np.zeros([f.shape[0], f.shape[1], 3])
diff = np.zeros([f.shape[0], f.shape[1], 3])
x_mu = np.zeros((f.shape[0], f.shape[1], 3))
bg = np.zeros([f.shape[0], f.shape[1], 3])
fg = np.zeros([f.shape[0], f.shape[1], 3])

alpha = 0.001
N = 1000
'''for i in range(100):
    _, f = vid.read()
    f = cv2.resize(f, (320, 240), interpolation=cv2.INTER_CUBIC)
    acc[:, :, :, i] = f[:, :, :]
'''
mean = f
std = np.sqrt(var)
diff = np.abs(np.float64(f) - mean)
square = np.square(diff)
#power = np.divide(diff, std)

beta = (N - 1) / N

def calculate_x_mu_square(x_mu):
    x_mu[:, :, 0] = beta * np.square(np.subtract(f[:, :, 0], mean[:, :, 0])) + (1 - beta) * x_mu[:, :, 0]
    x_mu[:, :, 1] = beta * np.square(np.subtract(f[:, :, 1], mean[:, :, 1])) + (1 - beta) * x_mu[:, :, 1]
    x_mu[:, :, 2] = beta * np.square(np.subtract(f[:, :, 2], mean[:, :, 2])) + (1 - beta) * x_mu[:, :, 2]

k = 0
while (1):
    bgind = np.where(diff <= 2 * std)
    fgind = np.where(diff > 2 * std)
    if k >= N:
        k = 0
        mean = acc
        var = 100 * np.ones([f.shape[0], f.shape[1], 3])
    else:
        mean[bgind] = alpha * (f[bgind]) + ((1 - alpha) * mean[bgind])
        var[bgind] = (alpha * (x_mu[bgind])) + ((1 - alpha) * var[bgind])
    f = np.uint8(f)

    bg[bgind] = f[bgind]
    fg[fgind] = f[fgind]

    fg = np.uint8(fg)
    bg = np.uint8(bg)

    _, f = vid.read()
    #f = cv2.GaussianBlur(f, (5, 5), 0)
    f = cv2.resize(f, (320, 240), interpolation=cv2.INTER_CUBIC)
    acc =  beta * f + (1-beta) * f
    k = k + 1
    std = np.sqrt(var)
    #ind = np.where(std == 0)
    #std[ind] = 1
    diff = np.abs(np.float64(f) - mean)
    cv2.imshow('img', f)
    cv2.imshow('fg', fg)

    cv2.imshow('bg', bg)
    fg = np.zeros([f.shape[0], f.shape[1], 3])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vid.release()