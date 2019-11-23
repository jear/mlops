# -*- coding: utf-8 -*-
# Code used to Preprocess entire Dataset of mails
# The pipeline defines what functions we want to use in the preprocess step
# Output a new csv File with 2 column : Texte and Categorie :
# - The categorie column is identical as the origin
# - The Texte is preprocessed

import sys, os
import numpy as np
import pandas as pd
from tools import basic as bas
from tools import api as ap

homebase = '/home/test/categorisationmailsads'

print(homebase)
os.chdir(homebase)
sys.path.append(os.path.abspath(os.path.join('..','tools')))
sys.path.append(os.path.abspath(os.path.join('..')))

os.chdir(homebase)

#Parameter you have to set before running program  :
#Csv you want to preprocess :
csvToPreprocess = homebase + '/data/sample.csv'
# Location and name of the preprocessed file :
pathtoSave = homebase + '/data/'
nameOfFile = "sample_preprocessed.csv"


f = open("/bd-fs-mnt/extData/categorisationmailsads/data/sample.csv", "r", encoding='latin-1')
pandaFinal = pd.read_csv(f, sep=';', header=0, encoding='latin-1')

#pandaFinal = pd.read_csv(csvToPreprocess, sep=';', header=0, encoding='latin-1')

#Using step to do the preprocess .
# Why ? Because Lemmatisation struggle where 1000+ mails are given at the same time
# You can choose your functions used to preprocess. For example, a basic pipeline would be :
#
# ADS uses a pipeline of 13 preprocessing functions, the most important one being the lemmatization
#


default_pipeline = ['remove_numeric','to_lower',  'remove_punct']



#Preprocess the csv batch by batch
# :param step: size of the batchs
# :param pipeline: contains each function applied to the pandaframe
# :return: pandaFrame preprocessed
def preprocess_csv(pandaf, pipeline, step) :
    if step > len(pandaf['Texte']) :
        print("Carefull : you must specify a step parameter lower than the size of your pandaFrame texte column")
        print("Step is set at the size of the pandaframe 'Texte' column")
        step = len(pandaf['Texte'])
        pandaPartial =  pandaf['Texte'][0:step]
        pandaPreprocess = ap.preprocess_pipeline(pandaPartial, pipeline)
    else :
        #First step
        pandaPartial =  pandaf['Texte'][0:step]
        pandaPreprocess = ap.preprocess_pipeline(pandaPartial, pipeline)
        #Next n Step
        a = step
        while (a + step) < len(pandaf['Texte']) :
            pandaPartial = pandaf['Texte'][a:a + step]
            pandaPartialPreprocessed = ap.preprocess_pipeline(pandaPartial, pipeline)
            pandaPreprocess = pd.concat([pandaPreprocess, pandaPartialPreprocessed], join = 'outer', ignore_index= True)
            a += step
        # Final step
        pandaPartial = pandaf['Texte'][a:len(pandaf['Texte'])]
        pandaPartialPreprocessed = ap.preprocess_pipeline(pandaPartial, pipeline)
        pandaPreprocess = pd.concat([pandaPreprocess, pandaPartialPreprocessed], join = 'outer', ignore_index= True)

    pandaFinal = pd.concat([pandaf['Texte'][:], pandaPreprocess, pandaf['Categorie'][:]], axis=1)
    pandaFinal.columns = ['Texte', 'Texte_preprocessed', 'Categorie']
    return pandaFinal



pandaPreprocessed = preprocess_csv(pandaFinal, default_pipeline, 500)
#pandaPreprocessed.to_csv(pathtoSave + nameOfFile, sep=';', index = False, encoding='latin-1')

g = open("/bd-fs-mnt/extData/categorisationmailsads/data/sample_preprocessed.csv", "w", encoding='latin-1')
pandaPreprocessed.to_csv( g, sep=';', index = False)
