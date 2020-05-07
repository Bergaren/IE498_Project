import torch
from model import ImageCaptioner
from config import Config
from tqdm import tqdm 
from dataset import CaptionDataset, prepare_train_data
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
config = Config()


model = ImageCaptioner(config)
model = model.to(device)


train_data = prepare_train_data(config)
loader = DataLoader(train_data, shuffle=True, batch_size=4, num_workers=1)

"""
train loop
TODO: load batches from dataset and feed to network
"""

upSample = nn.Upsample(scale_factor=7, mode="bilinear")

for e in tqdm(range(config.num_epochs)):
	for i, b in enumerate(loader):
		print(b)

	