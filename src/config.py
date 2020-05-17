"""
Adopted from: https://github.com/nikhilmaram/Show_and_Tell.git
"""


DATA_BASE_DIR = "../data/"

class Config(object):
	""" Wrapper class for various (hyper)parameters. """
	def __init__(self):
		# about the model architecture
		self.max_caption_length = 20
		self.dim_embedding = 512
		self.num_lstm_units = 512

		# about the weight initialization and regularization
		self.lstm_drop_rate = 0.3

		# about the optimization
		self.num_epochs = 100
		self.batch_size = 64
		self.initial_learning_rate = 0.0001
		self.clip_gradients = 5.0

		# about the saver
		self.save_period = 2
		self.save_dir = './models/rnn_adam.model'

		# about the vocabulary
		self.vocabulary_file = DATA_BASE_DIR + 'vocabulary.csv'
		self.vocabulary_size = 5000

		# about the training
		self.train_image_dir = DATA_BASE_DIR + 'train/images/'
		self.train_caption_file = DATA_BASE_DIR + '/train/captions_train2014.json'
		self.temp_annotation_file = DATA_BASE_DIR + '/train/anns.csv'
		self.temp_data_file = DATA_BASE_DIR + 'train/data.npy'
		self.mean_file = "./utils/ilsvrc_2012_mean.npy"

		# about the evaluation
		self.eval_image_dir = DATA_BASE_DIR + 'val/images/'
		self.eval_caption_file = DATA_BASE_DIR + 'val/captions_val2014.json'
		self.eval_result_dir = DATA_BASE_DIR + 'val/results/'
		self.eval_result_file = DATA_BASE_DIR + 'val/results.json'
		self.save_eval_result_as_image = False

		# about the testing
		self.test_image_dir = DATA_BASE_DIR + 'test/images/'
		self.test_result_dir = DATA_BASE_DIR + 'test/results/'
		self.test_result_file = DATA_BASE_DIR + 'test/results.csv'
