import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from data_loader import test_dataloader
from multiprocessing import set_start_method
from FCN import FCN8s, VGGNet
from config import *
from torchvision import transforms

vgg_model = VGGNet(requires_grad=True, show_params=False)
fcn_model = FCN8s(vgg_model, 2)

# Load pytorch model
fcn_model.load_state_dict(torch.load('model'))
fcn_model.eval()

try:
    set_start_method('spawn')
except:
    pass

for idx, (img, label) in enumerate(test_dataloader):

    output = fcn_model(img)
    output_np = output.cpu().detach().numpy().copy()
    output_np = np.argmin(output_np, axis=1)

    # Origin image
    plt.subplot(1,2,1)
    plt.imshow(np.transpose(img[0].cpu().detach().numpy(), (1, 2, 0)))

    # Output image
    plt.subplot(1,2,2)
    # 因為 array element 非 1 即 0, 用 gray 即可顯示出來
    plt.imshow(np.squeeze(output_np[0, ...]), 'gray') # squeeze: 刪除指定的單維度
    
    plt.pause(10)