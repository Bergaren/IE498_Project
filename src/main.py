import torch
from model import ImageCaptioner
from config import Config
from tqdm import tqdm 
from dataset import CaptionDataset, prepare_train_data
from torch.utils.data import DataLoader
from torch.autograd import Variable

device = "cuda" if torch.cuda.is_available() else "cpu"
config = Config()


model = ImageCaptioner(config)
model = model.to(device)


train_data = prepare_train_data(config)

loader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size, num_workers=8)

"""
train loop
TODO: load batches from dataset and feed to network
"""

for e in range(config.num_epochs):
	for b in tqdm(loader):
		images = b["images"]
		images = Variable(torch.FloatTensor(images.float()))
		images = images.permute(0,3,1,2)

		images = images.to(device)

		captions = b["captions"]
		captions = Variable(torch.LongTensor(captions.long())).to(device)
		pred = model(images, captions)


			
	