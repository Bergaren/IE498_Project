import torch
import torch.nn as nn
from model import ImageCaptioner
from config import Config
from tqdm import tqdm 
from dataset import CaptionDataset, prepare_train_data, prepare_eval_data, prepare_sample_data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.coco.pycocoevalcap.eval import COCOEvalCap
import json
from sys import argv

"""
Main program containing training and eval loops
"""


if len(argv) < 3:
	print("Supply arguments: load|new train|eval")

device = "cuda" if torch.cuda.is_available() else "cpu"
config = Config()


if argv[1] == "new":
	model = ImageCaptioner(config)
	model = model.to(device)
else:
	model = torch.load(config.save_dir)
	

train_data = prepare_train_data(config)
coco, eval_data, vocabulary = prepare_eval_data(config)
sample_data = prepare_sample_data(config)

train_loader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size, num_workers=8)
test_loader = DataLoader(eval_data, shuffle=False, batch_size=config.batch_size, num_workers=8)
sample_loader = DataLoader(sample_data, shuffle=False, batch_size = config.batch_size, num_workers=8)


"""
train loop
"""
def train():
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.RMSprop(model.parameters(), lr = config.initial_learning_rate)

	model.train()
	for e in range(config.num_epochs):
		hits = 0
		total_loss = 0.0
		for batch in tqdm(train_loader):
			optimizer.zero_grad()

			images = batch["images"]
			images = Variable(torch.FloatTensor(images.float()))
			images = images.permute(0,3,1,2)
			images = images.to(device)

			captions = batch["captions"]
			captions = Variable(torch.LongTensor(captions.long())).to(device)
			scores = model(images, captions)

			loss = criterion(scores.view(-1, config.vocabulary_size), captions.view(-1))
			total_loss += loss.item()

			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_gradients)

			optimizer.step()
		
		print("Epoch %d - average loss: %f" % (e+1, total_loss/len(train_loader)))
		# SAVE MODEL
		
		if (e+1) % config.save_period == 0:
			torch.save(model, config.save_dir)


""" 
Eval loop
"""

def evaulate():
	model.eval()
	results = []
	for batch in tqdm(test_loader):
		images = batch["images"]
		images = Variable(torch.FloatTensor(images.float()))
		images = images.permute(0,3,1,2)
		images = images.to(device)
			
		pred = model(images, captions = None)
			
		for i in range(pred.shape[1]):
			result = {
				"caption": str(vocabulary.get_sentence(pred[:, i])), 
				"image_id": int(batch["image_ids"][i].item())
			}
		#	print(result)
			results.append(result)

	# Write generated captions to result file		
	with open(config.eval_result_file, "w") as resfile:
		json.dump(results, resfile)
			
	# Evaluate these captions
	eval_result_coco = coco.loadRes(config.eval_result_file)
	scorer = COCOEvalCap(coco, eval_result_coco)
	scorer.evaluate()
	print("Evaluation complete.")

def test():
	model.eval()
	for batch in tqdm(sample_loader):
		images = batch["images"]
		images = Variable(torch.FloatTensor(images.float()))
		images = images.permute(0,3,1,2)
		images = images.to(device)

		pred = model(images, captions=None)

		for i in range(pred.shape[1]):
			result = {
				"caption": str(vocabulary.get_sentence(pred[:, i])),
				"img": batch["image_files"][i]
			}

			print(result)


if argv[2] == "eval":
	evaulate()
elif argv[2] == "train":
	train()
else:
	test()
