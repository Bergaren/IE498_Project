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

class ImageCaptioner(nn.Module):
	def __init__(self, config):
		super(ImageCaptioner, self).__init__()
		self.config = config
		self.vgg16 = self.create_encoder()
		self.embedd = nn.Embedding(self.config.vocabulary_size, self.config.dim_embedding)
		self.rnn = nn.LSTM(input_size=config.dim_embedding, hidden_size=config.num_lstm_units, batch_first=True)
		self.decoder = nn.Linear(self.config.num_lstm_units, self.config.vocabulary_size)

	def forward(self, x, captions=None):
		if not self.training:
			return self.evaluate(x)
		# Run the image through the CNN 
		x = self.vgg16(x)

		# Concatenate the representation of the image with the embeddings of the tokens in the caption
		captions = self.embedd(captions[:, :-1])
		x = torch.cat((x.unsqueeze(1), captions), dim=1)

		out, _ = self.rnn(x)
		out = self.decoder(out)
		return out


	def evaluate(self, x):
		predictions = []
		inputs = self.vgg16(x).unsqueeze(1)
		state = None

		# For sampling new captions, the representation of the image is first fed through the RNN
		# For each step the predicited word is the one with highest probability
		# The predicted word is then used as input for the following iteration 
		for i in range(self.config.max_caption_length):
			out, state = self.rnn(inputs, state)
			out = out.squeeze(1)
			out = self.decoder(out)

			pred = torch.argmax(out, dim=1)
			predictions.append(pred)
			inputs = self.embedd(pred).unsqueeze(1)

		return torch.stack(predictions) # Return the generated caption

	def create_encoder(self):
		# A pretrained vgg16 net with batch normalization is loaded 
		# The parameters are locked
		model = torchvision.models.vgg16_bn(pretrained=True)
		for param in model.parameters():
			param.requires_grad = False
		
		# The last layer is modified in order to accomodate for the desired output dimensions
		# As the last layer is changed it also unlocked and will be affected by training.
		model.classifier._modules['6'] = nn.Linear(model.classifier._modules['6'].in_features, self.config.dim_embedding)
		return model

# NOTE: "The above loss is minimized w.r.t. all the parameters of theLSTM, the final layer of the CNN, the decoding layer ad the embedding layer"
