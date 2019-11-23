# -*- coding: utf-8 -*-


# But : séparer le dataset global csv en deux autres dataset cvs ( train + validation / Test)

import numpy as np
import pandas as pd
import random

homebase = '/home/test/categorisationmailsads'

pandaToSplit = pd.read_csv( homebase + '/data/sample.csv', sep=';', header=0, encoding='latin-1')
pathtoSave = homebase + "/data/"
nameOutputTrain = "sample_train.csv"
nameOutputTest = "sample_test.csv"

CogitoDecodedCat = ["I02", "I03", "I04", "I05", "I06", "I07", "I08", "I09", "I10", "P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10", "P11", "TZ1", "TZ0"]

def get_total_amount(pandaF) :
    L = np.zeros(22)

    for i in range(0,len(pandaF['Texte'])) :
        realCat = pandaF['Categorie'][i]
        #partie comptage simple
        for k in range(0,len(CogitoDecodedCat)) :
            if CogitoDecodedCat[k] == realCat :
                L[k] += 1
    L2 = pd.DataFrame(L, columns=['TotalMail Test in categorie'])
    L2.index = CogitoDecodedCat
    return L2



def get_amount_train_val(pandaF, percent) :
    total = get_total_amount(pandaF)
    L = [[0]*2 for i in range(len(CogitoDecodedCat))]
    for i in range (0, len(CogitoDecodedCat)) :
        amountForTest = int(percent * total['TotalMail Test in categorie'].iloc[i])
        if amountForTest > 0 :
            L[i][0] = amountForTest
        else :
            L[i][0] = 1
    for i in range (0, len(CogitoDecodedCat)) :
        L[i][1] = total['TotalMail Test in categorie'].iloc[i] - L[i][0]
    L2 = pd.DataFrame(L, columns=['Total mail for test', 'Total mail for train'])
    L2.index = CogitoDecodedCat
    total = pd.concat([total, L2], join = 'outer', axis = 1)
    return total



def create_train_and_val(pandaF, percent) :
    """
    Split a DataFrame into 2 dataframe : 1 for the train and one for the test
    Respect the distribution of the classes of the initial dataset

    :param pandaF :
        input pandaFrame
    :param percent :
        the percentage of mails to put in the test dataFrame

    """
    pandaTrain = pd.DataFrame(columns=['Texte', 'Categorie'])
    pandaTest = pd.DataFrame(columns=['Texte', 'Categorie'])
    total = get_amount_train_val(pandaF, percent)
    nbMailTest = [[0]*2 for i in range(len(CogitoDecodedCat))]
    for i in range(0, len(pandaF['Texte'])) :
        realCat = pandaF['Categorie'][i]
        for k in range(0,len(CogitoDecodedCat)) :
            if CogitoDecodedCat[k] == realCat :
                indiceCat = k
                break;
        if nbMailTest[indiceCat][0] <  total['Total mail for test'].iloc[indiceCat]:
            if nbMailTest[indiceCat][1] < total['Total mail for train'].iloc[indiceCat] :
                rand = random.randint(1,5)
                if rand == 1 :
                    nbMailTest[indiceCat][0] += 1
                    pandaTest = pandaTest.append(pandaF.iloc[i], ignore_index=True)
                else :
                    nbMailTest[indiceCat][1] += 1
                    pandaTrain = pandaTrain.append(pandaF.iloc[i], ignore_index=True)
            else :
                nbMailTest[indiceCat][0] += 1
                pandaTest = pandaTest.append(pandaF.iloc[i], ignore_index=True)
        else :
            nbMailTest[indiceCat][1] += 1
            pandaTrain = pandaTrain.append(pandaF.iloc[i], ignore_index=True)
    return pandaTrain, pandaTest

(pandaTrain, pandaTest) = create_train_and_val(pandaToSplit, 0.05)

pandaTrain.to_csv(pathtoSave + nameOutputTrain, sep=';', index = False, encoding='latin-1')
pandaTest.to_csv(pathtoSave + nameOutputTest, sep=';', index = False, encoding='latin-1')
print("File Created under ",  pathtoSave)



