from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom

from multiprocessing import set_start_method
from data_loader import test_dataloader, train_dataloader
from FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet
from config import *

def train(show_vgg_params=False):

    vis = visdom.Visdom(env='fcn')

    # Use cuda training if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=2)
    # Copy data to GPU and run on GPU
    fcn_model = fcn_model.to(device)

    # Binary cross entropy loss function
    criterion = nn.BCELoss().to(device)
    # Stochastic Gradient Descent optimizer
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)

    all_train_iter_loss = []
    all_test_iter_loss = []

    # Start timing
    prev_time = datetime.now()

    try:
        set_start_method('spawn')
    except:
        pass

    for epo in range(EPOCH_NUM):
        
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

            # Calc loss
            loss = criterion(output, label)
            # Backpropagation
            loss.backward()

            # Extract loss value to echo
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            
            # Update all parameter
            optimizer.step()

            """
            Get the training result of every epoch

            cpu(): put data in cpu
            detach(): return a tensor disallowed backpropagation
            numpy(): cast tensor to numpy
            copy(): copy
            """
            output_np = output.cpu().detach().numpy().copy()
            output_np = np.argmin(output_np, axis=1) # Get indice of smallest value of row
            label_np = label.cpu().detach().numpy().copy() # label_np.shape = (4, 2, 160, 160) 
            label_np = np.argmin(label_np, axis=1)

            if np.mod(idx, 15) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epo, idx, len(train_dataloader), iter_loss))
                # vis.close()
                vis.images(output_np[:, None, :, :], win='train_pred', opts=dict(title='train prediction')) 
                vis.images(label_np[:, None, :, :], win='train_label', opts=dict(title='label'))
                vis.line(all_train_iter_loss, win='train_iter_loss',opts=dict(title='train iter loss'))

            # plt.subplot(1, 2, 1) 
            # plt.imshow(np.squeeze(label_np[0, ...]), 'gray')
            # plt.subplot(1, 2, 2) 
            # plt.imshow(np.squeeze(output_np[0, ...]), 'gray')
            # plt.pause(0.5)

        
        test_loss = 0
        fcn_model.eval()
        with torch.no_grad():
            for idx, (img, label) in enumerate(test_dataloader):

                img = img.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                output = fcn_model(img)
                output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
                loss = criterion(output, label)
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

                output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)  
                output_np = np.argmin(output_np, axis=1)
                label_np = label.cpu().detach().numpy().copy() # label_np.shape = (4, 2, 160, 160) 
                label_np = np.argmin(label_np, axis=1)
        
                if np.mod(idx, 15) == 0:
                    print(r'Testing... Open http://localhost:8097/ to see test result.')
                    # vis.close()
                    vis.images(output_np[:, None, :, :], win='test_pred', opts=dict(title='test prediction')) 
                    vis.images(label_np[:, None, :, :], win='test_label', opts=dict(title='label'))
                    vis.line(all_test_iter_loss, win='test_iter_loss', opts=dict(title='test iter loss'))

                # plt.subplot(1, 2, 1) 
                # plt.imshow(np.squeeze(label_np[0, ...]), 'gray')
                # plt.subplot(1, 2, 2) 
                # plt.imshow(np.squeeze(output_np[0, ...]), 'gray')
                # plt.pause(0.5)


        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print('epoch train loss = %f, epoch test loss = %f, %s'
                %(train_loss/len(train_dataloader), test_loss/len(test_dataloader), time_str))
        
        if np.mod(epo, 5) == 0:
            torch.save(fcn_model, 'checkpoints/fcn_model_{}.pt'.format(epo))
            print('saveing checkpoints/fcn_model_{}.pt'.format(epo))

    torch.save(fcn_model.state_dict(), 'model')

if __name__ == "__main__":
    train(show_vgg_params=False)