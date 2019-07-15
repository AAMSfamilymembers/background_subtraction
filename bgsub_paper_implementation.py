import numpy as np
import cv2

vid = cv2.VideoCapture(0)
_, f = vid.read()
f = cv2.resize(f, (320, 240), interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

mu = np.zeros((gray.shape[0], gray.shape[1], 3))

variance = 200 * np.ones((gray.shape[0], gray.shape[1], 3))

x_mu = np.zeros((gray.shape[0], gray.shape[1], 3))

M = np.zeros((gray.shape[0], gray.shape[1], 3))
rho = np.zeros((gray.shape[0], gray.shape[1], 3))

w = np.zeros((gray.shape[0], gray.shape[1], 3))
w[:, :, 2] = 1
sigma = np.zeros((gray.shape[0], gray.shape[1], 3))

# mu[:,:,0] = gray

# x_mu_index = 0

N = 100
T = 0.7
learning_rate = 0.0007
constant = learning_rate / (2 * (3.141 ** (N / 2)))

for i in range(10):
    _, f = vid.read()
    f = cv2.resize(f, (320, 240), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    mu[:, :, 2] = mu[:, :, 2] + gray
mu[:, :, 2] = mu[:, :, 2] / 10

beta = (N - 1) / N


def calculate_x_mu_square(x_mu):
    x_mu[:, :, 0] = beta * np.square(np.subtract(gray, mu[:, :, 0])) + (1 - beta) * x_mu[:, :, 0]
    x_mu[:, :, 1] = beta * np.square(np.subtract(gray, mu[:, :, 1])) + (1 - beta) * x_mu[:, :, 1]
    x_mu[:, :, 2] = beta * np.square(np.subtract(gray, mu[:, :, 2])) + (1 - beta) * x_mu[:, :, 2]


'''mu[:, :, 1] = gray
mu[:, :, 2] = gray'''
background = np.zeros(gray.shape, dtype = np.uint8)
foreground = np.zeros(gray.shape, dtype = np.uint8)
while 1:
    _, f = vid.read()
    f = cv2.resize(f, (320, 240), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

    calculate_x_mu_square(x_mu)

    matchindex1 = np.where(np.abs(gray - mu[:, :, 0]) <= 2.5 * sigma[:, :, 0])
    matchindex2 = np.where(np.abs(gray - mu[:, :, 1]) <= 2.5 * sigma[:, :, 1])
    matchindex3 = np.where(np.abs(gray - mu[:, :, 2]) <= 2.5 * sigma[:, :, 2])

    (M[:, :, 0])[matchindex1] = 1
    (M[:, :, 1])[matchindex2] = 1
    (M[:, :, 2])[matchindex3] = 1

    w = w + learning_rate * (M - w)

    # normalizing weights
    s = np.sum(w, axis=2)
    w[:, :, 0] = w[:, :, 0] / s
    w[:, :, 1] = w[:, :, 1] / s
    w[:, :, 2] = w[:, :, 2] / s

    # Calculating Gaussians : rho

    (rho[:, :, 0])[matchindex1] = constant * (
        np.exp(-np.divide((x_mu[:, :, 0])[matchindex1], 2 * (variance[:, :, 0])[matchindex1])))

    (rho[:, :, 1])[matchindex2] = constant * (
        np.exp(-np.divide((x_mu[:, :, 1])[matchindex2], 2 * (variance[:, :, 1])[matchindex2])))

    (rho[:, :, 2])[matchindex3] = constant * (
        np.exp(-np.divide((x_mu[:, :, 2])[matchindex3], 2 * (variance[:, :, 2])[matchindex3])))

    # Updating Params : mean

    (mu[:, :, 0])[matchindex1] = (1 - (rho[:, :, 0])[matchindex1]) * (mu[:, :, 0])[matchindex1] + (rho[:, :, 0])[
        matchindex1] * gray[
                                     matchindex1]

    (mu[:, :, 1])[matchindex2] = (1 - (rho[:, :, 1])[matchindex2]) * (mu[:, :, 1])[matchindex2] + (rho[:, :, 1])[
        matchindex2] * gray[
                                     matchindex2]

    (mu[:, :, 2])[matchindex3] = (1 - (rho[:, :, 2])[matchindex3]) * (mu[:, :, 2])[matchindex3] + (rho[:, :, 2])[
        matchindex3] * gray[
                                     matchindex3]

    # Updating Params : variance

    (variance[:, :, 0])[matchindex1] = (1 - (rho[:, :, 0])[matchindex1]) * (variance[:, :, 0])[matchindex1] + \
                                       (rho[:, :, 0])[matchindex1] * (x_mu[:, :, 2])[matchindex1]

    (variance[:, :, 1])[matchindex2] = (1 - (rho[:, :, 1])[matchindex2]) * (variance[:, :, 1])[matchindex2] + \
                                       (rho[:, :, 1])[matchindex2] * (x_mu[:, :, 2])[matchindex2]

    (variance[:, :, 2])[matchindex3] = (1 - (rho[:, :, 2])[matchindex3]) * (variance[:, :, 2])[matchindex3] + \
                                       (rho[:, :, 2])[matchindex3] * (x_mu[:, :, 2])[matchindex3]

    # Sorting
    sigma = np.sqrt(variance)

    w_by_sigma = np.divide(w, sigma)

    index = np.argsort(w, axis=2)

    w_by_sigma = np.take_along_axis(w_by_sigma, index, axis=2)
    mu = np.take_along_axis(mu, index, axis=2)
    variance = np.take_along_axis(variance, index, axis=2)
    w = np.take_along_axis(w, index, axis=2)
    x_mu = np.take_along_axis(x_mu, index, axis=2)
    M = np.take_along_axis(M, index, axis=2)

    # assigning values to foreground and background

    back_index1 = np.where((w[:, :, 2] > T) & (M[:, :, 2] == 1))
    back_index2 = np.where(((w[:, :, 2] + w[:, :, 1]) > T) & (w[:, :, 2] < T) & (M[:, :, 1] == 1))
    back_index3 = np.where((w[:, :, 2] + w[:, :, 1] < T) & (M[:, :, 0] == 1))
    temp = np.zeros(gray.shape)
    temp[back_index1] = 1
    temp[back_index2] = 1
    temp[back_index3] = 1

    background[back_index1] = (mu[:, :, 2])[back_index1]
    background[back_index2] = ((mu[:, :, 1])[back_index2] + (mu[:, :, 2])[back_index2])/2
    background[back_index3] = ((mu[:, :, 0])[back_index3] + (mu[:, :, 0])[back_index3] + (mu[:, :, 0])[back_index3])/3

    foreground[np.where(temp == 0)] = gray[np.where(temp == 0)]

    cv2.imshow('background', background.astype(np.uint8))
    cv2.imshow('foreground', foreground.astype(np.uint8))
    cv2.imshow('frame', gray.astype(np.uint8))

    # replacing gaussians
    replace_index = np.where(M[:, :, 0] == 0)
    (mu[:, :, 0])[replace_index] = gray[replace_index]
    (variance[:, :, 0])[replace_index] = 200
    (x_mu[:, :, 0])[replace_index] = 0
    foreground = np.zeros(gray.shape)

    M = np.zeros((gray.shape[0], gray.shape[1], 3))

    if cv2.waitKey(1) & 0xFF == 27:
        break

vid.release()
cv2.destroyAllWindows()
