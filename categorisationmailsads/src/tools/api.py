import sys, os
os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join('..','tools')))
sys.path.append(os.path.abspath(os.path.join('..')))


import utils
import basic
#import synonym_malefemale_replacement as syn
#import lemmatizer as lem
#import stopwords

from nltk.stem.snowball import FrenchStemmer
import pandas as pd
import collections
# ---------------------------------------------------------------------------------------------------------------------------
# FONCTION PREPROCESS FINALE
# ---------------------------------------------------------------------------------------------------------------------------


## Nettoyage du texte
#
# Effectue l'ensemble des actions de preprocessing suivantes sur tous les textes d'un corpus :
# 1) Remplacement des éléments null de docs par des blancs
# 2) Passage en minuscules + Suppression de la ponctuation + Suppression des blancs multiples
# 3) Remplacement des mots masculin/féminin par le mot masculin (ex : serveur/serveuse remplacé par serveur)
# 4) Suppression des / et des () :
# 5) Lemmatisation
# 6) Complément préprocessing : suppression des valeurs numériques + accents + blancs multiples
# 7) Suppression des mots qui n'apparaissent qu'une seule fois dans tout le corpus
#
# @param docs   Le corpus à nettoyer
# @return Le corpus une fois ses textes nettoyés
def preprocess(docs):
    # 1) On remplace les éléments null de docs par des blancs
    docs = basic.notnull(docs)

    # 2) Passage en minuscules + Suppression de la ponctuation + Suppression des blancs multiples
    docs = preprocess_fundation(docs)

    # 3) Remplacement des mots masculin/féminin par le mot masculin (ex : serveur/serveuse remplacé par serveur)
    d = syn.malefemale_listing()
    liste_syn = d.listing_synonym(docs)  # On commence par définir la liste des synonymes
    d = syn.replace_synonym()
    docs = d.doc_replace_synonym(docs, liste_syn)

    # 4) Avant la lemmatisation on supprime les / et les () :
    docs = pd.Series(docs).str.replace('[()\/]', ' ')

    # 5) Lemmatisation ou Stemmatisation
    # docs = docs.apply(lambda x: lemmatize(x))
    docs = stemmatize(docs)

    # 6) Complément préprocessing : suppression des valeurs numériques + accents + blancs multiples
    docs = preprocess_complete(docs)

    # 7) Suppression des mots qui n'apparaissent qu'une seule fois dans tout le corpus
    # count_words = listing_count_words(docs)
    # liste = list(list_one_appearance_word(count_words))
    # docs = docs.apply(lambda x: " ".join(remove_words(x.split(), list(liste))))

    return docs


##  Suppression des adresses email
#
# Remplace les adresses email d'un corpus par un espace
# Cherche dans le texte des adresses email via la regex ^[a-zA-Z0-9_.-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$
# Explication de la regex :
# \s                : Un mot commençant par :
# [a-zA-Z0-9_.-]+   : au moins un caractère alphanumérique ou caractère spécial autorisé
# @                 : suivi du symbole @
# [a-zA-Z0-9-]+     : puis au moins un caractère alphanumérique ou un tiret,
# \.                : un point
# [a-zA-Z0-9-.]+    : et encore au moins un caractère autorisé
# \s                : fin du mot
#
# @param docs               Le corpus à nettoyer
# @return Le corpus une fois ses textes nettoyés
def remove_emails(docs):
    try:
        return docs.str.replace(r"(\s[a-zA-Z0-9_.-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\s)", ' ')
    except AttributeError:
        raise TypeError("Le paramètre docs doit être une série pandas")


## Nettoyage du texte - Base
#
# Effectue quelques actions de nettoyage de texte d'un corpus
# - Passage en minuscules
# - Suppression de la ponctuation
# - Suppression des blancs multiples
#
# @param docs               Le corpus à nettoyer
# @param del_parenthesis    [bool] Indique si les parenthèses doivent être enlevées ou non (par défaut : False)
# @return Le corpus une fois ses textes nettoyés
def preprocess_fundation(docs, del_parenthesis=False):
    # Passage en minuscules
    docs = docs.apply(lambda x: " ".join(x.lower() for x in x.split()))
    docs = docs.str.replace('permis B', 'permisB')
    # Suppression des ponctuations
    docs = docs.str.replace('[\',;:.!\*-]', ' ')  # On ne retire NI le / NI les parenthèses dans un premier temps
    if del_parenthesis:
        docs = docs.str.replace('[()\/]', ' ')
    # Suppression des blancs multiples
    docs = docs.str.replace('\s{2,}', ' ')
    return docs


##  Nettoyage du texte - Complétion
#
# Finalise le nettoyage d'un corpus de textes
# - Suppression des valeurs numériques
# - Suppression des Stop Words
# - Suppression des accents
# - Suppression des blancs multiples
#
# @param docs               Le corpus à nettoyer
# @return Le corpus une fois ses textes nettoyés
def preprocess_complete(docs):
    # Suppression des valeurs numériques
    docs = docs.str.replace('([0-9]+)', ' ')
    # Suppression des Stop Words
    docs = docs.apply(lambda x: " ".join(stopwords.remove_stopwords(x.split(), opt='all')))
    # Suppression des accents
    docs = docs.apply(lambda x: stopwords.remove_accents(str(x)))
    # Suppression des blancs multiples
    docs = docs.str.replace('\s{2,}', ' ')

    return docs


def remove_gender_synonyms(docs):
    d = syn.malefemale_listing()
    liste_syn = d.listing_synonym(docs)  # On commence par définir la liste des synonymes
    d = syn.replace_synonym()
    docs = d.doc_replace_synonym(docs, liste_syn)
    return docs


def lemmatize(docs):
    docs = pd.Series(docs).str.replace('[()\/]', ' ')
    docs = docs.apply(lambda x: lemmatizer.lemmatizer(x))
    return docs


##  Stemmatisation des mots
#
# Stemmatise les mots d'un corpus de texte soit :
# Supprime les suffixes des mots en fonction de règles préfédinies
# Afin de retrouver la racine des mots
#
# @param docs               Le corpus à nettoyer
# @return Le corpus une fois ses textes nettoyés
def stemmatize(docs):
    stemmer = FrenchStemmer()
    try:
        return docs.apply(lambda x: " ".join(stemmer.stem(x) for x in x.split()))
    except AttributeError:
        raise TypeError("Le paramètre docs doit être une série pandas")


# ---------------------------------------------------------------------------------------------------------------------------
# Fonctions listiing_count_words, fonctions list_one_appearance_word, remove_words
# ---------------------------------------------------------------------------------------------------------------------------


## Liste de mots uniques
#
# Retourne un data frame donnant l'ensemble des mots apparaissant dans un corpus de texte ainsi que leurs fréquences
# Comptabilisation des occurences des mots + tri par fréquence décroissante (data frame en sortie)
#
# @param docs
# @return count_words Data Frame
def listing_count_words(docs):
    words = list([word for sentence in docs.str.split() for word in sentence])
    count_words = collections.Counter(words)
    count_words = pd.DataFrame(sorted(count_words.items(), key=lambda t: t[1], reverse=True),
                               columns=["word", "count"])

    return count_words


## Liste des mots d'occurence unique
#
# Retourne un listing des mots n'apparaissant qu'une seule fois dans un corpus de textes
#
# @param count_words    Dataframe de comptage des mots fourni par la fonction listing_count_words
# @return Liste des mots d'occurence unique
def list_one_appearance_word(count_words):
    stop_less_commun_words = count_words["word"][count_words["count"] <= 1]
    return stop_less_commun_words


# Filtre des mots
#
# Retourne les mots du texte non présents dans la liste de mots à enlever
#
# @param  text              [str] texte à filtrer
# @param words_to_remove    Liste de mots à enlever
# @return Liste des mots du texte filtrés
def remove_words(text, words_to_remove):
    return [w for w in text if w not in words_to_remove]

# Dictionnaire des fonctions de pré-processing disponibles
usage = {
    'remove_punct': basic.remove_punct,
    'to_lower': basic.to_lower,
    'remove_numeric': basic.remove_numeric
}

# Chaîne de traitement de pré-processing
#
# A partir d'une pandas Series de documents, retourne une nouvelle
# pandas Series contenant les documents modifiés par les fonctions
# contenues dans pipeline
#
# @param docs           [pandas.Series] corpus en entrée
# @param pipeline       [str list] fonctions à appliquer (ayant une correspondance dans usage)
# @return corpus pré-processé
@utils.data_agnostic_function(modify_data=True, chunk_size=50000)
@utils.process_docs_keep_everything()
def preprocess_pipeline(docs, pipeline=['notnull', 'to_lower', 'pe_matching', 'remove_punct',
                             'trim_string', 'remove_gender_synonyms', 'remove_numeric',
                                        'stemmatize', 'remove_accents', 'remove_stopwords']):
    for item in pipeline:
        if item in usage.keys():
            docs = usage[item](docs)

    return docs
