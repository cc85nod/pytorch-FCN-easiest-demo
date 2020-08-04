import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG
from data_loader import test_dataloader, train_dataloader

ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# VGG layout
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



# make layers using Vgg-Net config(cfg)
# 由cfg构建vgg-Net
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGGNet(VGG):
    """
    pretrained: Pre-train with TORCH.UTILS.MODEL_ZOO
    model: Inherit with torchvision.models.vgg16
    required_grad: Allow backpropagation
    remove_fc: Remove fully connected layer
    """
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        # Load pretrain model parameters
        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        # Remove all grad parameter
        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  
            del self.classifier

        # Print all parameters
        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        # Get the output of each maxpooling layer (5 maxpool in VGG net)
        # VGG16 ranges: ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31))
        for idx, (begin, end) in enumerate(self.ranges):
            for layer in range(begin, end):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x
            """
            For example:
            output = {
                "x1": x1_feature,
                "x2": x2_feature,
                "x3": x3_feature,
                ...
            }
            """
        return output