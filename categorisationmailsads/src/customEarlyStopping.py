from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
import warnings
import numpy as np


class CustomEarlyStopping(Callback):
	"""
	Custom Early stopping :
		Stop training if (train_loss / val_loss) < ratio,
		for minimum n epochs ( with n = patience)
	
	
	"""
	def __init__(self, ratio=0.0, patience=0, verbose=0):
		super(EarlyStopping).__init__()
		self.ratio = ratio
		self.patience = patience
		self.verbose = verbose
		self.wait = 0
		self.stopped_epoch = 0
		self.monitor_op = np.greater

	def on_train_begin(self, logs=None):
		self.wait = 0  # Allow instances to be re-used

	def on_epoch_end(self, epoch, logs=None):
		current_val = logs.get('val_loss')
		current_train = logs.get('loss')
		if current_val is None:
			warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        # If ratio current_loss / current_val_loss > self.ratio
		if self.monitor_op(np.divide(current_train,current_val),self.ratio):
			self.wait = 0
		else:
			if self.wait >= self.patience:
				self.stopped_epoch = epoch
				self.model.stop_training = True
			self.wait += 1

	def on_train_end(self, logs=None):
		if self.stopped_epoch > 0 and self.verbose > 0:
			print('Epoch %05d: early stopping' % (self.stopped_epoch))