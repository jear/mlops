# -*- coding: utf-8 -*-

import sys, os

path = '/home/test/' + os.path.dirname(__file__)
homebase = '/home/test/categorisationmailsads/'

os.chdir(homebase)
sys.path.append(os.path.abspath(os.path.join('..','tools')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import itertools
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from tools import utils2 as utl

import math


CogitoDecodedCat = ["I02", "I03", "I04", "I05", "I06", "I07", "I08", "I09", "I10", "P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10", "P11","TZ0"]#, "TZ1"]


def create_list_categories_from_sample(labels_averes, labels_predis) :
    list_categories = []
    # On gère ces deux catégories à part car l'ordre n'est pas alphabétique dans notre binarizer : TZ1 arrive avant TZ0
    append_TZ1 = False
    append_TZ0 = False
    for el in labels_averes :
        if el not in list_categories :
            if el == "TZ1" :
                append_TZ1 = True
            elif el == "TZ0" :
                append_TZ0 = True
            else :
                list_categories.append(el)
    for el in labels_predis :
        if el not in list_categories :
            if el == "TZ1":
                append_TZ1 = True
            elif el == "TZ0":
                append_TZ0 = True
            else:
                list_categories.append(el)
    list_categories.sort()
    if append_TZ1 :
        list_categories.append("TZ1")
    if append_TZ0 :
        list_categories.append("TZ0")
    return list_categories


# Using sklearn libraries to transform our labels into an array of binarize value
# Example :In a classification over 5 labels, label "3" become [0 0 1 0 0]
lb = preprocessing.LabelBinarizer()
lb.fit([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])

#Load data used for test
    
testPath = homebase + "/data/sample_test_predict.csv"
pandaApplet=pd.read_csv(testPath, sep=';', header=0, encoding='latin-1')

messages = pandaApplet['Texte'][0:].values
labels = pandaApplet['Categorie'].values

# Prediction
Y_pred = pandaApplet['LSTM_predict'].values

currentDecodedCat = create_list_categories_from_sample(labels,  Y_pred)

def plot_confusion_matrix2(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #♠classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=currentDecodedCat, yticklabels=currentDecodedCat,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fig = plt.gcf()
    fig.set_size_inches(16, 12)
    
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap,aspect='auto') #aspect='auto'  extent=[-0.5,0.5,-0.5,0.5]
    #forceAspect(img,aspect=1)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, currentDecodedCat, rotation=45)
    plt.yticks(tick_marks, currentDecodedCat)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




def get_precision_rappel_f1score(confusion_matrix) :
    Recall = []
    Precision = []
    F1Score = []
    for i in range(0,len(confusion_mtx)) :
        numerateur = 0
        denominateurRecall = 0
        denominateurPrecision = 0
        for j in range(0,len(confusion_mtx)) :
            if i == j :
                numerateur += confusion_mtx[i][j]
                denominateurRecall += confusion_mtx[i][j]
                denominateurPrecision += confusion_mtx[i][j]
            else :
                 denominateurRecall += confusion_mtx[i][j]
                 denominateurPrecision += confusion_mtx[j][i]

                 
        Recall.append(numerateur/denominateurRecall)
        Precision.append(numerateur/denominateurPrecision)
    #Computing F1-Score
    for i in range(0, len(Recall)) :
        if (Precision[i] > 0) and (Recall[i] > 0) :
            F1Scorei = 2*(Recall[i] * Precision[i]) / (Recall[i] + Precision[i])
            F1Score.append(F1Scorei)
        elif (math.isnan(Precision[i])) or (math.isnan(Recall[i])) :
            F1Score.append(None)
        else :
            F1Score.append(0)
          
    
 
    #Assembling Recall matrix
    matriceRecall = pd.DataFrame(Recall, columns = ["Recall"])
    matriceRecall.index = currentDecodedCat
    #Assembling Precision matrix
    matricePrecision = pd.DataFrame(Precision, columns = ["Precision"])
    matricePrecision.index = currentDecodedCat
    #Assembling F1Score matrix
    matriceF1Score = pd.DataFrame(F1Score, columns = ["F1-Score"])
    matriceF1Score.index = currentDecodedCat
    
    res = pd.concat([matriceRecall, matricePrecision, matriceF1Score], join = 'outer', axis = 1)
    
    return res


def get_macro_micro_recall(confusion_matrix) :
    macro_recall = 0
    micro_recall = 0
    
    macro_precision = 0
    micro_precision = 0
    
    numerateurMicroRecall = 0
    denominateurMicroRecall = 0
    
    numerateurMicroPrecision = 0
    denominateurMicroPrecision = 0
    for i in range(0,len(confusion_mtx)) :
        numerateurMacroRecall = 0
        numerateurMacroPrecision = 0
        denominateurMacroPrecision = 0
        denominateurMacroRecall = 0
        for j in range(0,len(confusion_mtx)) :
            if i == j :
                numerateurMacroPrecision += confusion_mtx[i][j]
                denominateurMacroPrecision += confusion_mtx[i][j]
                
                numerateurMacroRecall += confusion_mtx[i][j]
                denominateurMacroRecall += confusion_mtx[i][j]
                
                numerateurMicroRecall += confusion_mtx[i][j]
                denominateurMicroRecall += confusion_mtx[i][j]
                
                numerateurMicroPrecision += confusion_mtx[i][j]
                denominateurMicroPrecision += confusion_mtx[i][j]
                
            else :
                denominateurMacroRecall += confusion_mtx[i][j]
                
                denominateurMacroPrecision += confusion_mtx[j][i]
                
                denominateurMicroRecall += confusion_mtx[i][j]
                denominateurMicroPrecision += confusion_mtx[j][i]
                
        macro_recall += numerateurMacroRecall/denominateurMacroRecall
        macro_precision += numerateurMacroPrecision / denominateurMacroPrecision
        
    macro_recall = macro_recall / len(confusion_mtx)
    micro_recall = numerateurMicroRecall / denominateurMicroRecall
    
    macro_precision = macro_precision / len(confusion_mtx)
    micro_precision = numerateurMicroPrecision / denominateurMicroPrecision
    return (macro_recall, micro_recall, macro_precision, micro_precision)



#Delete additionnal informations on labels
labels = utl.cut_label_categories(labels)

# Encode Labels
labels = utl.encode_cat_labels(labels)

labelsCategoricals = lb.transform(labels)

print(labelsCategoricals)



# Convert predictions classes to one hot vectors 
Y_pred_classes = utl.encode_cat_labels(Y_pred)


#print(Y_pred_classes)

# Convert validation observations to one hot vectors
Y_true = np.argmax(labelsCategoricals,axis = 1) 
#Y_true start from 0 to 21 , we just change it : 1 to 22
Y_true = Y_true + 1
#print(Y_true)


confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
plot_confusion_matrix2(Y_true, Y_pred_classes,  classes = np.array(range(22)))


res = get_precision_rappel_f1score(confusion_mtx)

(macroRecall, microRecall, macroPrecision, microPrecision) = get_macro_micro_recall(confusion_mtx)
print(res)

IndexMatriceOverall = ['macroRecall', 'microRecall', 'macroPrecision', 'microPrecision']
matriceOverall = pd.DataFrame([macroRecall, microRecall, macroPrecision, microPrecision], columns = ["Métriques Globales"])
matriceOverall.index = IndexMatriceOverall
print("-----------------------")
print(matriceOverall)