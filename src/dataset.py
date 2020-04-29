from utils.coco.coco import COCO
from utils.vocabulary import Vocabulary
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import torch

"""
Adopted from https://github.com/nikhilmaram/Show_and_Tell.git
"""

# TODO: Implement transforms and load images

class CaptionDataset(Dataset):
	def __init__(self,
		image_ids,
		image_files,
		word_idx,
		transform=None):

		self.image_ids = image_ids
		self.image_files = image_files
		self.word_idx = word_idx
		self.transform = transform


	def __len__(self):
		return len(self.image_files)

	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.toList()
		images = self.image_files[idx]
		captions = self.word_idx[idx]
		sample = {"images": images, "captions": captions}

		return sample
		



# TODO: Save and retrieve preporcessed directly if avaiable

def prepare_train_data(config):
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
		masks = []
		for caption in tqdm(captions):
			current_word_idxs_ = vocabulary.process_sentence(caption)
			current_num_words = len(current_word_idxs_)
			current_word_idxs = np.zeros(config.max_caption_length,
										 dtype = np.int32)
			current_masks = np.zeros(config.max_caption_length)
			current_word_idxs[:current_num_words] = np.array(current_word_idxs_)
			current_masks[:current_num_words] = 1.0
			word_idxs.append(current_word_idxs)
			masks.append(current_masks)
		word_idxs = np.array(word_idxs)
		masks = np.array(masks)
		data = {'word_idxs': word_idxs, 'masks': masks}
		np.save(config.temp_data_file, data)
	else:
		data = np.load(config.temp_data_file, allow_pickle=True).item()
		word_idxs = data['word_idxs']
		masks = data['masks']
	print("Captions processed.")
	print("Number of captions = %d" %(len(captions)))

	print("Building the dataset...")
	dataset = CaptionDataset(image_ids, image_files, word_idxs, None)
	print("Dataset built.")
	return dataset


if __name__ == "__main__":
	from config import Config

	c = Config()
	d = preparte_train_data(c)