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

class ImageCpationer(nn.Module):
	def __init__(self):
		super(ImageCpationer, self).__init__()
		self.cnn = torchvision.models.vgg16_bn(pretrained=True)

	def forward(self, x):
		return self.cnn(x)

