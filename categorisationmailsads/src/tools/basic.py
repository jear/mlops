#import stopwords
import re
import pandas as pd
#import lemmatizer
# ----------------------------------------------------------------------------------------------------------------------
# FONCTIONS preprocessing light
# ----------------------------------------------------------------------------------------------------------------------




def to_lower(docs):
    return docs.apply(lambda x: " ".join(x.lower() for x in x.split()))


def remove_punct(docs, del_parenthesis=True):
    docs = docs.str.replace('[\',;:.!\*-?]', ' ')  # On ne retire NI le / NI les parenthèses dans un premier temps # Careful cette fonction supprime tout les caractères ascii entre "*" et "?" il semblerait
    if del_parenthesis:
        docs = docs.str.replace('[()\/]', ' ')
    return docs


def trim_string(docs):
    return docs.str.replace('\s{2,}', ' ')


def remove_numeric(docs):
    return docs.str.replace('([0-9]+)', ' ')
	
	
## Enlève les éléments null des documents
#
# Remplace les éléments null de docs par des chaines vides
#
# @param docs   Document à modifier
# @return Document modifié avec les null remplacés par des chaines vides
def notnull(docs):
    docs_notnull = docs.fillna('')
    return docs_notnull


