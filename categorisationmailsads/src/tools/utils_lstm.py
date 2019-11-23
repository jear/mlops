# -*- coding: utf-8 -*-

import numpy as np

import utils2 as utl


def predict_topn_element(model, inputMessage, n = 3) :
    prediction = model.predict(inputMessage, verbose = 3)
    #print("prediction :", prediction)
    predictCat = []
    if n > 14 : # We have 21 catégories, we will give maximum top15 predictions
        n = 14
    for k in range(0,n) :
        #max_value = np.amax(prediction)
        max_indice = np.argmax(prediction)
        #print("max_indice :", max_indice)
        cat = utl.decode_cat_labels([max_indice + 1])
        #print("cat :", cat)
        predictCat.append(cat)
        prediction[0][max_indice] = -1
    topCategories = np.array(predictCat)
    return topCategories


def predict_topn_element_v2(model, inputMessage, n = 3) :
    prediction = model.predict(inputMessage, verbose = 3)
    predictCat = []
    predictValues = []
    if n > 22 : # We have 21 catégories, we will give maximum top15 predictions
        n = 22
    for k in range(0,n) :
        max_indice = np.argmax(prediction)
        cat = utl.decode_cat_labels([max_indice + 1])
        predictCat.append(cat)
        predictValues.append(prediction[0][max_indice])
        prediction[0][max_indice] = -1
    topCategories = np.array(predictCat)
    topValues = np.array(predictValues)
    return (topCategories, topValues)


def encode_ST_messages_lstm(messages, vocab_to_int):
    """
    Encode ST Sentiment Labels
    :param messages: list of list of strings. List of message tokens
    :param vocab : the whole vocabulary from word embedding phase
    :param vocab_to_int: mapping of vocab to idx
    :return: list of ints. Lists of encoded messages
    """
    messages_encoded = []
    for message in messages:
        toAppend = []
        for word in message.split() :
            if word in vocab_to_int :
                toAppend.append(vocab_to_int[word])
        messages_encoded.append(toAppend)
    if len(messages_encoded[0]) <= 0:
        return np.array([[0]])
    return np.array(messages_encoded)


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
