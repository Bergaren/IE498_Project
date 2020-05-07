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
		self.fc = nn.Linear(self.config.dim_embedding, self.config.vocabulary_size)

	def forward(self, x, captions):
		sentence = []
		for i in range(self.config.max_caption_length):
			
			if i == 0:
				### Osäker om det "self.config.batch_size" dim
				# Memory
				h = torch.zeros(self.config.batch_size, self.config.num_lstm_units).cuda()
				c = torch.zeros(self.config.batch_size, self.config.num_lstm_units).cuda()

				x = self.vgg16(x)
				x =x.resize(self.config.batch_size, 8, self.config.dim_embedding)
				x = torch.mean(x, 1)
				
			else:
				x = self.embedd(captions[:, i])
			#x = x.view(-1, x.shape[0], x.shape[1])
			h,c = self.rnn_model(x, (h, c))
			#print(f"H: {h.shape}, C:{c.shape}, X:{x.shape}")
			x = self.fc(h)
			sentence.append( torch.max(x, dim=1) )
		return sentence

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