import torch.nn as nn
import math
import torch


class VGG(nn.Module):

    def __init__(self, builder, features):
        super(VGG, self).__init__()
        self.features = features
        num_classes = 10 if parser_args.set == "CIFAR10" else 100
        self.linear = builder.conv1x1(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.linear(x)
        return x.squeeze()


def make_layers(cfg, builder, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = builder.conv3x3(in_channels, v) #nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v, eps=1e-5), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(cfg, batch_norm, builder):
    model = VGG(builder, make_layers(cfgs[cfg], builder, batch_norm=batch_norm))
    return model

def vgg19_bn_new_fc(num_classes):
    return _vgg('E', True, get_builder())