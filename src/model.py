"""
Definition of neural model.

Consits of CNN and RNN.

"""
import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable

"""
The CNN portion of the model 
In the paper a pretrained vgg16 net is used.
Available in pytorch.
"""


class LockedDropout(nn.Module):
	def __init__(self, p=0.5):
		super(LockedDropout,self).__init__()
		self.m = None
		self.p = p

	def reset_state(self):
		self.m = None

	def forward(self, x, train_sess=True):
		if train_sess==False:
			return x
		if(self.m is None):
			self.m = x.data.new(x.size()).bernoulli_(1 - self.p)
		mask = Variable(self.m, requires_grad=False) / (1 - self.p)

		return mask * x

class ImageCaptioner(nn.Module):
	def __init__(self, config):
		super(ImageCaptioner, self).__init__()
		self.config = config
		self.vgg16 = self.create_encoder()
		# Får inputen från tidigare LSTM, behövs inte på första eftersom då får vi info från encodern.
		self.embedd = nn.Embedding(self.config.vocabulary_size, self.config.dim_embedding)
		# The LSTM cell
		self.rnn = nn.LSTM(input_size=config.dim_embedding, hidden_size=config.num_lstm_units, batch_first=True)
		# Mappar outputen från tidigare LSTM cell till ett ord.
		self.decoder = nn.Linear(self.config.num_lstm_units, self.config.vocabulary_size)
		self.dropout = LockedDropout(self.config.lstm_drop_rate)

	def forward(self, x, captions=None):
		if not self.training:
			return self.evaluate(x)
		x = self.vgg16(x)
		captions = self.embedd(captions)

		x = torch.cat((x.unsqueeze(1), captions[:, 1:]), dim=1)
		out, _ = self.rnn(x)
		out = self.decoder(out)
		return out


	def evaluate(self, x):
		predictions = []
		inputs = self.vgg16(x).unsqueeze(1)
		state = None
		for i in range(self.config.max_caption_length):
			out, state = self.rnn(inputs, state)
			out = out.squeeze(1)
			out = self.decoder(out)
			pred = torch.argmax(out, dim=1)
			predictions.append(pred)
			inputs = self.embedd(pred).unsqueeze(1)

		return torch.stack(predictions)

	def create_encoder(self):
		model = torchvision.models.vgg16_bn(pretrained=True)
		for param in model.parameters():
			param.requires_grad = False
		### Ändrar sista lagret så att det passar med input size av RNN.
		### Sista lagret måste omtränas eftersom det init random
		model.classifier._modules['6'] = nn.Linear(model.classifier._modules['6'].in_features, self.config.dim_embedding)
		return model

### NOTE: "The above loss is minimized w.r.t. all the parameters of theLSTM, the top layer of the image embedder CNN and word embedding We"
