from utils.coco.coco import COCO
from utils.vocabulary import Vocabulary
from utils.misc import ImageLoader
import numpy as np
import os
import pandas as pd
import nltk
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch

"""
Adopted from https://github.com/nikhilmaram/Show_and_Tell.git
Reimplemented for usage with pytorch Dataloader
"""


"""
Class used for the training data in conjuction with a dataloader
"""
class CaptionDataset(Dataset):
	def __init__(self,
		image_ids,
		image_files,
		word_idx=None,
		transform=None):

		self.image_ids = image_ids
		self.image_files = image_files
		self.word_idx = word_idx
		self.transform = transform
		self.imageloader = ImageLoader("./utils/ilsvrc_2012_mean.npy")


	def __len__(self):
		return len(self.image_files)

	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.toList()
		#print(self.image_files[idx])
		images = self.imageloader.load_image(self.image_files[idx])
		captions = self.word_idx[idx]
		sample = {"images": images, "captions": captions}
		return sample
		
""" 
Class used for the eval data in conjuction with a dataloader
"""
class CaptionEvalDataset(Dataset):
	def __init__(self,
		image_ids,
		image_files,
		transform=None):

		self.image_ids = image_ids
		self.image_files = image_files
		self.transform = transform
		self.imageloader = ImageLoader("./utils/ilsvrc_2012_mean.npy")

	def __len__(self):
		return len(self.image_files)

	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.toList()
		#print(self.image_files[idx])
		images = self.imageloader.load_image(self.image_files[idx])
		sample = {"images": images, "image_ids": self.image_ids[idx]}
		return sample

"""
Class used for testing images outside of MSCOCO eval set
"""
class CaptionTestDataset(Dataset):
	def __init__(self, image_files, transforms=None):
		self.image_files = image_files
		self.transforms = transforms
		self.imageloader = ImageLoader("./utils/ilsvrc_2012_mean.npy")
	
	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.toList()
		images = self.imageloader.load_image(self.image_files[idx])
		sample = {"images": images, "image_files": self.image_files[idx]}
		return sample


def prepare_train_data(config):
	"""
		If all is done => Load directly and return dataset
	"""

	if os.path.exists(config.vocabulary_file) and os.path.exists(config.temp_annotation_file) and os.path.exists(config.temp_data_file):

		print("Loading all data...")
		annotations = pd.read_csv(config.temp_annotation_file)
		captions = annotations['caption'].values
		image_ids = annotations['image_id'].values
		image_files = annotations['image_file'].values

		data = np.load(config.temp_data_file, allow_pickle=True).item()
		word_idxs = data['word_idxs']

		print("Building dataset...")
		dataset = CaptionDataset(image_ids, image_files, word_idxs, None)
		return dataset

	
	""" Prepare the data for training the model. """
	coco = COCO(config.train_caption_file)
	coco.filter_by_cap_len(config.max_caption_length)

	print("Building the vocabulary...")
	vocabulary = Vocabulary(config.vocabulary_size)
	if not os.path.exists(config.vocabulary_file):
		vocabulary.build(coco.all_captions())
		vocabulary.save(config.vocabulary_file)
	else:
		vocabulary.load(config.vocabulary_file)
	print("Vocabulary built.")
	print("Number of words = %d" %(vocabulary.size))

	coco.filter_by_words(set(vocabulary.words))

	print("Processing the captions...")
	if not os.path.exists(config.temp_annotation_file):
		captions = [coco.anns[ann_id]['caption'] for ann_id in coco.anns]
		image_ids = [coco.anns[ann_id]['image_id'] for ann_id in coco.anns]
		image_files = [os.path.join(config.train_image_dir,
									coco.imgs[image_id]['file_name'])
									for image_id in image_ids]
		annotations = pd.DataFrame({'image_id': image_ids,
									'image_file': image_files,
									'caption': captions})
		annotations.to_csv(config.temp_annotation_file)
	else:
		annotations = pd.read_csv(config.temp_annotation_file)
		captions = annotations['caption'].values
		image_ids = annotations['image_id'].values
		image_files = annotations['image_file'].values

	if not os.path.exists(config.temp_data_file):
		word_idxs = []
		for caption in tqdm(captions):
			current_word_idxs_ = vocabulary.process_sentence(caption)
			current_num_words = len(current_word_idxs_)
			current_word_idxs = np.zeros(config.max_caption_length,
										 dtype = np.int32)
			current_word_idxs[:current_num_words] = np.array(current_word_idxs_)
			word_idxs.append(current_word_idxs)
		word_idxs = np.array(word_idxs)
		data = {'word_idxs': word_idxs}
		np.save(config.temp_data_file, data)
	else:
		data = np.load(config.temp_data_file, allow_pickle=True).item()
		word_idxs = data['word_idxs']
	print("Captions processed.")
	print("Number of captions = %d" %(len(captions)))

	print("Building the dataset...")
	dataset = CaptionDataset(image_ids, image_files, word_idxs, None)
	print("Dataset built.")
	return dataset


def prepare_eval_data(config):
	""" Prepare the data for evaluating the model. """
	coco = COCO(config.eval_caption_file)
	image_ids = list(coco.imgs.keys())
	image_files = [os.path.join(config.eval_image_dir,
								coco.imgs[image_id]['file_name'])
								for image_id in image_ids]

	print("Building the vocabulary...")
	if os.path.exists(config.vocabulary_file):
		vocabulary = Vocabulary(config.vocabulary_size,
								config.vocabulary_file)
	else:
		vocabulary = build_vocabulary(config)
	print("Vocabulary built.")
	print("Number of words = %d" %(vocabulary.size))

	print("Building the dataset...")
	dataset = CaptionEvalDataset(image_ids, image_files)
	print("Dataset built.")
	return coco, dataset, vocabulary

def prepare_sample_data(config):
	image_files = os.listdir(config.test_image_dir)
	image_files = [os.path.join(config.test_image_dir, filename) for filename in image_files]
	dataset = CaptionTestDataset(image_files)
	return dataset

def build_vocabulary(config):
	""" Build the vocabulary from the training data and save it to a file. """
	coco = COCO(config.train_caption_file)
	coco.filter_by_cap_len(config.max_caption_length)

	vocabulary = Vocabulary(config.vocabulary_size)
	vocabulary.build(coco.all_captions())
	vocabulary.save(config.vocabulary_file)
	return vocabulary

if __name__ == "__main__":
	from config import Config
	print("running...")
	c = Config()
	d = prepare_sample_data(c)
