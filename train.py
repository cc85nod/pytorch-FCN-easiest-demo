from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom

from multiprocessing import set_start_method
from data_loader import test_dataloader, train_dataloader
from FCN import FCN8s, VGGNet
from config import *



"""
1. define network (in FCN.py)
    - VGG16 + deconv
2. compile network (loss function, optimizer and metrics)
    - optimizer: SGD
    - loss function: binary cross-entropy
3. fit network (epoch, validate, set batch_size and start trainning)
    - epoch: EPOCH_NUM in config.py
    - batch_size: BATCH_SIZE in config.py
4. evaluate network (calculate result)
5. make prediction
"""
def train(show_vgg_params=False):

    vis = visdom.Visdom(env='fcn')

    """
    torch.device: an object representing the device on which a torch.Tensor is or will be allocated
    """

    # Use cuda training if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    fcn_model = FCN8s(pretrained_net=vgg_model, n_class=2)
    # Copy data to GPU and run on GPU
    fcn_model = fcn_model.to(device)

    # Binary cross entropy loss function
    criterion = nn.BCELoss().to(device)
    # Stochastic Gradient Descent optimizer
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)

    all_train_iter_loss = []
    all_test_iter_loss = []

    # Start timing
    stime = datetime.now()

    try:
        set_start_method('spawn')
    except:
        pass

    for epo in range(EPOCH_NUM):
        
        """
        Training part
        """
        train_loss = 0
        fcn_model.train()
        for idx, (img, label) in enumerate(train_dataloader):
            
            img = img.to(device)
            label = label.to(device)

            """
            Init grad to zero
            We need to set the gradients to zero before starting to do backpropagation
            because pytorch accumulates the gradients on subsequent backward passes
            """
            optimizer.zero_grad()
            output = fcn_model(img)
            output = torch.sigmoid(output) # Get probability

            # Calc loss and backpropagation
            loss = criterion(output, label)
            loss.backward()

            # Extract loss value to echo
            iter_loss = loss.item()
            train_loss += iter_loss
            all_train_iter_loss.append(iter_loss)
            
            # Update all parameter
            optimizer.step()

            """
            Get the training result of every epoch

            cpu(): put data in cpu
            detach(): return a tensor disallowed backpropagation
            numpy(): cast tensor to numpy
            copy(): copy
            """

            """
            output.shape: torch.Size([BATCH_SIZE, 2, *IMAGE_SIZE])
            output_np.shape: (BATCH_SIZE, 2, *IMAGE_SIZE)
            np.argmin(output_np): (BATCH_SIZE, *IMAGE_SIZE)
            np.squeeze(output_np[0, ...]).shape: (*IMAGE_SIZE)
            """
            output_np = output.cpu().detach().numpy().copy()
            output_np = np.argmin(output_np, axis=1) # Get indice of smallest value of row
            label_np = label.cpu().detach().numpy().copy()
            label_np = np.argmin(label_np, axis=1)

            # 每 15 步紀錄一次
            if np.mod(idx, 15) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epo, idx, len(train_dataloader), iter_loss)) # Log

                # output_np[:, None, ...] == (batch_size, 1, height, width), batch_size 決定顯示圖片的數量
                # visdom:
                #   - win: windows
                #   - opts: 設定 visualization 的 config
                vis.images(output_np[:, None, :, :], win='train_pred', opts=dict(title='train prediction')) 
                vis.images(label_np[:, None, :, :], win='train_label', opts=dict(title='label'))
                vis.line(all_train_iter_loss, win='train_iter_loss',opts=dict(title='train iter loss'))


        """
        Testing part
        """
        test_loss = 0
        fcn_model.eval()
        for idx, (img, label) in enumerate(test_dataloader):

            img = img.to(device)
            label = label.to(device)

            loss = criterion(output, label)
            optimizer.zero_grad()
            
            output = fcn_model(img)
            output = torch.sigmoid(output)

            iter_loss = loss.item()
            test_loss += iter_loss
            all_test_iter_loss.append(iter_loss)

            output_np = output.cpu().detach().numpy().copy()
            output_np = np.argmin(output_np, axis=1)

            label_np = label.cpu().detach().numpy().copy()
            label_np = np.argmin(label_np, axis=1)
    
            if np.mod(idx, 15) == 0:
                print(r'Testing... Open http://localhost:8097/ to see test result.')
                vis.images(output_np[:, None, :, :], win='test_pred', opts=dict(title='test prediction')) 
                vis.images(label_np[:, None, :, :], win='test_label', opts=dict(title='label'))
                vis.line(all_test_iter_loss, win='test_iter_loss', opts=dict(title='test iter loss'))

        etime = datetime.now() # End time
        h, remainder = divmod((etime - stime).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        stime = etime

        # Log
        print('epoch train loss = %f, epoch test loss = %f, %s'
                % (train_loss / len(train_dataloader), test_loss / len(test_dataloader), time_str))

    # Save trained model
    torch.save(fcn_model.state_dict(), 'model')


if __name__ == "__main__":
    train(show_vgg_params=False)