import torch
import torch.nn as nn
from model import ImageCaptioner
from config import Config
from tqdm import tqdm 
from dataset import CaptionDataset, prepare_train_data, prepare_eval_data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.coco.pycocoevalcap.eval import COCOEvalCap
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
config = Config()


model = ImageCaptioner(config)
model = model.to(device)


train_data = prepare_train_data(config)
coco, eval_data, vocabulary = prepare_eval_data(config)

train_loader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size, num_workers=8)
test_loader = DataLoader(eval_data, shuffle=False, batch_size=config.batch_size, num_workers=8)


"""
train loop
TODO: load batches from dataset and feed to network
"""

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr = config.initial_learning_rate)
""" model.train()
for e in range(config.num_epochs):
	hits = 0
	for b in tqdm(train_loader):
		optimizer.zero_grad()

		images = b["images"]
		images = Variable(torch.FloatTensor(images.float()))
		images = images.permute(0,3,1,2)
		images = images.to(device)

		captions = b["captions"]
		captions = Variable(torch.LongTensor(captions.long())).to(device)
		scores = model(images, captions, train = True)

		loss = criterion(scores.permute(1, 2, 0), captions)
		
		loss.backward()
		optimizer.step()

		masks = b["masks"]
		pred = torch.argmax(scores, dim  = 2)
		hits += pred.permute(1, 0).eq(captions).int().sum()
	
	print("Epoch %d achieved training accuracy of: %f" % (e+1,hits.item()*100.0/len(train_loader))) """

with torch.no_grad():
	model.training = False
	results = []
	for b in tqdm(test_loader):
		images = b["images"]
		images = Variable(torch.FloatTensor(images.float()))
		images = images.permute(0,3,1,2)
		images = images.to(device)
		
		pred = model(images, captions = None)
		
		for i in range(pred.shape[1]):
			results.append({
				"caption": str(vocabulary.get_sentence(pred[:, i])), 
				"image_id": int(b["image_ids"][i].item())
			})

	# Write generated captions to result file		
	with open(config.eval_result_file, "w") as resfile:
		json.dump(results, resfile)
		
	# Evaluate these captions
	eval_result_coco = coco.loadRes(config.eval_result_file)
	scorer = COCOEvalCap(coco, eval_result_coco)
	scorer.evaluate()
	print("Evaluation complete.")


		



			