# -*- coding: utf-8 -*-

#This is the main script to train our Deep Learning model
#Note that in this script we used data already preprocessed
import sys, os

homebase = '/home/test/categorisationmailsads'
print(homebase)

os.chdir(homebase)
sys.path.append(os.path.abspath(os.path.join('..','tools')))
sys.path.append(os.path.abspath(os.path.join('..')))


import numpy as np
import pandas as pd
from sklearn import preprocessing
from tools import utils2 as utl
import neural_architecture as deep
import customEarlyStopping as myEarlyStop
from tools import utl_plot
import matplotlib.style as style

style.use('ggplot')

#Load data
# Note that this is a data already preprocessed :
# The csv contains 2 columns : Texte | Categorie
pandaFinal = pd.read_csv( homebase + '/data/sample_train.csv', sep=';', header=0, encoding='latin-1')

messages = pandaFinal['Texte'][0:].values
labels = pandaFinal['Categorie'].values


labels = utl.cut_label_categories(labels)

#Little check if the data is 'ok'
for i in range(0,2):
    print("Messages: {}...".format(messages[i]),
          "Sentiment: {}".format(labels[i]))


#Load Dictionary
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


# Encode Messages and Labels
messages = utl.encode_ST_messages_lstm(messages, dictionary)
labels = utl.encode_cat_labels(labels)

print(len(messages))
print(len(labels))


# Pad Messages
messages = utl.zero_pad_messages(messages, seq_len=250)


# Using sklearn libraries to transform our labels into an array of binarize value
# Example :In a classification over 5 labels, label "3" become [0 0 1 0 0]
lb = preprocessing.LabelBinarizer()

lb.fit([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
print("Classes :", lb.classes_)
labelsCategoricals = lb.transform(labels)


#Split the data into 2 groups :
# - Training set : Generally 80-95% of the data. The data used to train the model
# - Validation : Generally 5-20% of the data . The data used to check the efficiency of the model over the epochs.
#  - Also this dataset is used to stop the training under some constraints ( to avoid overfiiting for example)
# Note that in this script we only split into training and validation set. We will test the accuracy of the model in another script
train_x, val_x, train_y, val_y = utl.train_val_split(messages, labelsCategoricals, split_frac=0.9)

print("Data Set Size")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set: \t{}".format(val_x.shape))

#We visualize the repartition of labels among the 21 categories
print("Statistiques sur le dataset de train :")
utl_plot.plot_data(utl_plot.get_insight_of_data(train_y, nbLabels = 22), nbLabels = 22)
print("Statistiques sur le dataset de validation :")
utl_plot.plot_data(utl_plot.get_insight_of_data(val_y, nbLabels = 22), nbLabels = 22)

# Embedding parameter , according to the Word2Vec we trained in a previous task
embed_size = 100
seq_len = 250

#Load the word2Vec embedding matrix
FileToEmbedding = homebase + '/src/tools/W2VEmbedding_v3.npy'
final_embeddings = np.load(FileToEmbedding)
vocabulary_size = 50000

#Build the Neural network
model = deep.ltsm_model(final_embeddings, vocabulary_size, embed_size, seq_len)
modelVersion = "v1"

# Define the callback :
# We choose a customCallback
# Stop if train_loss/val_loss < ratio, during at least n epoch ( n = patience )
callbacks = [myEarlyStop.CustomEarlyStopping(ratio=0.75, patience=4, verbose=1)]

#Train the neural network
history = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=20,
                    batch_size=100, callbacks=callbacks)


#Plot the evolution of the metrics during the training
utl_plot.plot_model_training_metrics(history, metric = 'accuracy', save = True, folderWithName = homebase + '/img/accuracyLSTM_'+ modelVersion +'.png')
utl_plot.plot_model_training_metrics(history, metric = 'loss', save = True, folderWithName = homebase + '/img/lossLSTM_'+ modelVersion +'.png')



# Save the weights
FileForModelWeights = homebase + '/models/lstm/lstmModelH5_'+ modelVersion +'.h5'
model.save_weights(FileForModelWeights)

# Save the model architecture
FileForArchitecture = homebase + '/models/lstm/lstmModelJSON_'+ modelVersion +'.json'
with open(FileForArchitecture, 'w') as f:
    f.write(model.to_json())

#Save both weights and architecture in one file
FileForFullModel = homebase + '/models/lstm/lstmModelFullH5_'+ modelVersion +'.h5'
model.save(FileForFullModel)
