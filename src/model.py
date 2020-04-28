"""
Definition of neural model.

Consits of CNN and RNN.

"""
import torchvision
import torch
import torch.nn as nn

"""
The CNN portion of the model 
In the paper a pretrained vgg16 net is used.
Available in pytorch.
"""
cnn = torchvision.models.vgg16(pretrained=True)
