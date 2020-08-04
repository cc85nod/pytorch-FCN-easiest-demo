# pytorch FCN
###### fork from https://github.com/bat67/pytorch-FCN-easiest-demo

### env
* python 3.7
* TODO

## how to run
- open visdom.server
```
python -m visdom.server &
```
- start train
```
python train.py
```

## data
- image path: `./data/images`
- label path: `./data/labels`

## code
### [train.py](train.py)
- training
- visualize

### [FCN.py](FCN.py)
- define VGG16
- define FCN32s, FCN16s, FCN8s, FCNs based on VGG16

### [data_loader.py](data_loader.py)
- loading data and format it

### [onehot.py](onehot.py)
- onehot encode
