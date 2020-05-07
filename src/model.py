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
		self.rnn_model = nn.LSTMCell( input_size = self.config.dim_embedding,
								 hidden_size = self.config.num_lstm_units)
	
		# Mappar outputen från tidigare LSTM cell till ett ord.
		self.decoder = nn.Linear(self.config.dim_embedding, self.config.vocabulary_size)
		self.dropout = LockedDropout(self.config.lstm_drop_rate)

	def forward(self, x, captions=None):
		if not self.training:
			return self.evaluate(x)
		
		predictions = []
		scores = []
		for i in range(self.config.max_caption_length):
			
			if i == 0:
				h = torch.zeros(self.config.batch_size, self.config.num_lstm_units).cuda()
				c = torch.zeros(self.config.batch_size, self.config.num_lstm_units).cuda()

				x = self.vgg16(x)
				x = x.reshape(self.config.batch_size, 8, self.config.dim_embedding)
				x = torch.mean(x, 1)
				
			else:
				x = self.embedd(captions[:, i])

			h,c = self.rnn_model(x, (h, c))
			h = self.dropout(h)
			score = self.decoder(h)
			scores.append(score)
		return torch.stack(scores)
	
	def evaluate(self, x):
		predictions = []
		for i in range(self.config.max_caption_length):
			
			if i == 0:
				h = torch.zeros(self.config.batch_size, self.config.num_lstm_units).cuda()
				c = torch.zeros(self.config.batch_size, self.config.num_lstm_units).cuda()

				x = self.vgg16(x)
				x = x.reshape(self.config.batch_size, 8, self.config.dim_embedding)
				x = torch.mean(x, 1)
			else: 
				x = self.embedd(pred_word)
				
			h,c = self.rnn_model(x, (h, c))
			score = self.decoder(h)
			pred_word = torch.argmax(score, dim = 1)	

			predictions.append(pred_word)
				
		return torch.stack(predictions)

	def create_encoder(self):
		model = torchvision.models.vgg16_bn(pretrained=True)
		for param in model.parameters():
			param.requires_grad = False
		### Ändrar sista lagret så att det passar med input size av RNN.
		### Sista lagret måste omtränas eftersom det init random
		model.classifier._modules['6'] = nn.Linear(model.classifier._modules['6'].in_features, 8*self.config.dim_embedding)
		return model

	def train(self):
		pass

### NOTE: "The above loss is minimized w.r.t. all the parameters of theLSTM, the top layer of the image embedder CNN and word embedding We"