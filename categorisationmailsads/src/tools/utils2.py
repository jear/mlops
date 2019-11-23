import re
import string
from collections import Counter
import numpy as np



	
	
def encode_ST_messages_lstm(messages, vocab_to_int):
	messages_encoded = []
	for message in messages :
		toAppend = []
		for word in message.split() :
			if word in vocab_to_int :
				toAppend.append(vocab_to_int[word])
		messages_encoded.append(toAppend)
		
	return np.array(messages_encoded)

	
def encode_cat_labels(labels):
	listLabels = []
	CogitoEncodedCat = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
	CogitoDecodedCat = ["I02", "I03", "I04", "I05", "I06", "I07", "I08", "I09", "I10", "P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10", "P11", "TZ1", "TZ0"]
	for cat in labels :
		for i in range (0,len(CogitoDecodedCat)):
			if cat == CogitoDecodedCat[i] :
				listLabels.append(CogitoEncodedCat[i])
	encodedLabels = np.array(listLabels)
	return encodedLabels

def encode_labels_INDEMN_PLACEMENT(labels) : # todo
    listLabels = []
    for lab in labels :
        if lab.startswith("P") == True :
            listLabels.append(1)
        elif lab.startswith("I") == True :
            listLabels.append(2)
        else :
            listLabels.append(3)
    encodedLabels = np.array(listLabels)
    return encodedLabels

	
def decode_cat_labels(labels):
    listLabels = []
    CogitoEncodedCat = [i for i in range(1,23)]
    CogitoDecodedCat = ["I02", "I03", "I04", "I05", "I06", "I07", "I08", "I09", "I10", "P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10", "P11", "TZ1", "TZ0"]
    for cat in labels :
        for i in range (0,len(CogitoEncodedCat)) :
            if cat == CogitoEncodedCat[i] :
                 listLabels.append(CogitoDecodedCat[i])
    encodedLabels = np.array(listLabels)
    return encodedLabels	
	
def drop_empty_messages(messages, labels):
    """
    Drop messages that are left empty after preprocessing
    :param messages: list of encoded messages
    :return: tuple of arrays. First array is non-empty messages, second array is non-empty labels
    """
    non_zero_idx = [ii for ii, message in enumerate(messages) if len(message) != 0]
    messages_non_zero = np.array([messages[ii] for ii in non_zero_idx])
    labels_non_zero = np.array([labels[ii] for ii in non_zero_idx])
    return messages_non_zero, labels_non_zero

def zero_pad_messages(messages, seq_len):
    """
    Zero Pad input messages
    :param messages: Input list of encoded messages
    :param seq_ken: Input int, maximum sequence input length
    :return: numpy array.  The encoded labels
    """
    messages_padded = np.zeros((len(messages), seq_len), dtype=int)
    for i, row in enumerate(messages):
        messages_padded[i, -len(row):] = np.array(row)[:seq_len]

    return np.array(messages_padded)

def train_val_test_split(messages, labels, split_frac, random_seed=None):
    """
    Zero Pad input messages
    :param messages: Input list of encoded messages
    :param labels: Input list of encoded labels
    :param split_frac: Input float, training split percentage
    :return: tuple of arrays train_x, val_x, test_x, train_y, val_y, test_y
    """
    # make sure that number of messages and labels allign
	
    assert len(messages) == len(labels)
    # random shuffle data
    if random_seed:
        np.random.seed(random_seed)
    shuf_idx = np.random.permutation(len(messages))
    messages_shuf = np.array(messages)[shuf_idx] 
    labels_shuf = np.array(labels)[shuf_idx]

    #make splits
    split_idx = int(len(messages_shuf)*split_frac)
    train_x, val_x = messages_shuf[:split_idx], messages_shuf[split_idx:]
    train_y, val_y = labels_shuf[:split_idx], labels_shuf[split_idx:]

    test_idx = int(len(val_x)*0.5)
    val_x, test_x = val_x[:test_idx], val_x[test_idx:]
    val_y, test_y = val_y[:test_idx], val_y[test_idx:]

    return train_x, val_x, test_x, train_y, val_y, test_y
	
	
def train_val_split(messages, labels, split_frac, random_seed=None):
    """
    Zero Pad input messages
    :param messages: Input list of encoded messages
    :param labels: Input list of encoded labels
    :param split_frac: Input float, training split percentage
    :return: tuple of arrays train_x, val_x, test_x, train_y, val_y, test_y
    """
    # make sure that number of messages and labels allign
	
    assert len(messages) == len(labels)
    # random shuffle data
    if random_seed:
        np.random.seed(random_seed)
    shuf_idx = np.random.permutation(len(messages))
    messages_shuf = np.array(messages)[shuf_idx] 
    labels_shuf = np.array(labels)[shuf_idx]

    #make splits
    split_idx = int(len(messages_shuf)*split_frac)
    train_x, val_x = messages_shuf[:split_idx], messages_shuf[split_idx:]
    train_y, val_y = labels_shuf[:split_idx], labels_shuf[split_idx:]


    return train_x, val_x, train_y, val_y
    
def get_batches(x, y, batch_size=100):
    """
    Batch Generator for Training
    :param x: Input array of x data
    :param y: Input array of y data
    :param batch_size: Input int, size of batch
    :return: generator that returns a tuple of our x batch and y batch
    """
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]
		
		
		
		
def cut_label_categories(labels) :
	"""
	Only keep first word of the label description
	:param labels: List of labels
	:return: labels with only first word ( "P07 rendez vous placement" become "P07" )
	"""
	Result = labels
	for i in range(0,len(labels)) :
		labelCut = labels[i].split()[0]
		Result[i] = labelCut
	return Result