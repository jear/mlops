# -*- coding: utf-8 -*-

import sys, os


homebase = '/home/test/categorisationmailsads'
modelVersion = "v1"

os.chdir(homebase)
sys.path.append(os.path.abspath(os.path.join('..','tools')))
sys.path.append(os.path.abspath(os.path.join('..')))

from tools import utils
import pandas as pd
import numpy as np
from tools import basic as bas
from tools import api as ap
from keras.models import model_from_json
from sklearn import preprocessing
from tools import utils_lstm
from tools import utils2 as utl


os.chdir(homebase)

csvToPredict =  homebase + '/data/sample_test.csv' #"../data/sample_test.csv"

file_encoding = 'utf8'        # set file_encoding to the file encoding (utf8, latin1, etc.)
input_fd = open(csvToPredict, encoding=file_encoding, errors = 'backslashreplace')
pandaTSCE = pd.read_csv(input_fd, sep=';', header=0)

pandaTSCE = bas.notnull(pandaTSCE)

    
df = pd.DataFrame(columns=['LSTM_predict'])

pandaTSCEPrediction = pd.concat([pandaTSCE, df], axis = 1)
pandaTSCEPrediction = pd.concat([pandaTSCE, df], axis = 1)


FileForModelWeights = homebase + '/models/lstm/lstmModelH5_' + modelVersion + '.h5'
FileForArchitecture = homebase + '/models/lstm/lstmModelJSON_' + modelVersion + '.json'

#Loading the model
with open(FileForArchitecture, 'r') as f:
    modelOpenned = model_from_json(f.read())

# Load weights of the model
modelOpenned.load_weights(FileForModelWeights)
DictionaryFile = homebase + '/src/tools/dictionary_v3.npy'

# FBerque (start): to support numpy 1.17.3
#dictionary = np.load(DictionaryFile).item()

np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
dictionary = np.load(DictionaryFile).item()

# restore np.load for future normal usage
np.load = np_load_old

# FBerque (end)


sequencePadding = 250

lb = preprocessing.LabelBinarizer()
lb.fit([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])



for i in range(0, len(pandaTSCEPrediction)) : #11370
    print(i)

    mess = pandaTSCEPrediction['Texte'][i]
    mess = pd.Series(mess)
    mess = utils_lstm.encode_ST_messages_lstm(mess, dictionary)

    mess = utils_lstm.zero_pad_messages(mess, seq_len=250)
    pred = modelOpenned.predict(mess)
    (pred21cat, predictIndex) = utils_lstm.predict_topn_element_v2(modelOpenned, mess, 22)
    myDict = {}
    for j in range(0,22) :
        key = pred21cat[j][0]
        value = predictIndex[j]
        myDict[key] = value

    res = {}
    indexCat = lb.inverse_transform(pred)
    stringCat = utl.decode_cat_labels(indexCat)
    pandaTSCEPrediction['LSTM_predict'][i] = pred21cat[0][0] #pred21cat[j][0] myDict



pathtoSave = homebase + '/data/'
nameOfFile = "sample_test_predict.csv"
pandaTSCEPrediction.to_csv(pathtoSave + nameOfFile, sep=';', index = False, encoding='utf-8')