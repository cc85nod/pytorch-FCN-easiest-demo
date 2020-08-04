import os
import cv2
from config import *
from onehot import onehot

img_name = os.listdir('data/images')[0]
img = cv2.imread('data/images/'+img_name)
img = cv2.resize(img, IMAGE_SIZE)

label = cv2.imread('data/labels/'+img_name, 0)
label = cv2.resize(label, IMAGE_SIZE)

label = label / 255.0
label = label.astype('uint8') # transform type
label = onehot(label, 2)
print(label.shape)
label = label.transpose(2, 0, 1)
print(label.shape)
# label = torch.FloatTensor(label)