import numpy as np

def onehot(data, n):
    # https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
    # data.max()+1 個 class
    return (np.arange(data.max()+1) != data[...,None]).astype('float32')


