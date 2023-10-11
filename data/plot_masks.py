import os
import glob
import numpy as np
import cv2
import random 
import sys 
import pickle as pkl
import json
import shutil
import matplotlib.pyplot as plt
from skimage.transform import rotate
import imageio as iio
from scipy.spatial.transform import Rotation as R
from datetime import datetime

directory = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/wrong'
category = 'camera'
split = 'test'


bottle_test = np.load(os.path.join(directory, '{}_test.npy'.format(category))) # load
bottle_train = np.load(os.path.join(directory, '{}_train.npy'.format(category))) # load



fig = plt.figure()
# max_dist = bottle_test.max()
# print(max_dist)

max_value = max(bottle_test.max(), bottle_train.max())
bins = np.arange(max_value + 1, step = max_value//50)
plt.hist(bottle_train, bins=bins, alpha=1, label='{} train set'.format(category), color='tomato')
plt.hist(bottle_test, bins=bins, alpha=1, label='{} test set'.format(category), color='skyblue')

plt.legend()


hist, bins = np.histogram(bottle_test, bins)
print(hist[0])
print(bins[0])
hist, bins = np.histogram(bottle_train, bins)
print(hist[0])
print(bins[0])
# plt.hist(bottle_test, bins = np.arange(max_dist, step = max_dist//50) )
# plt.axvline(x = 9365, color = 'r')
# plt.axvline(x = 63682, color = 'r')
plt.title("Mask Pixel Size Distribution in {} Category".format(category))
plt.xlabel('Mask Pixel Size')
plt.ylabel('Density')
plt.savefig("/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/wrong/{}.png".format(category))
