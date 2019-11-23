# -*- coding: utf-8 -*-

## Utils - fonctions-outils
# Auteurs : Task Force Sémantique
# Date : 25/06/2018
#
# Fonctions :
# - file_length
# - df_from_lake
# - df_from_csv_generator
# - sample_from_gen


import os
import pandas as pd
import errno
import pyodbc
import shutil
from io import StringIO
import itertools
import tempfile

HIVE_ODBC_URL = r'Driver={Hortonworks Hive ODBC Driver};ServiceDiscoveryMode=1;Host=hp1edge02.pole-emploi' \
                r'.intra:2181,hp1namenode01.pole-emploi.intra:2181,' \
                r'hp1namenode02.pole-emploi.intra:2181;ZKNamespace=hiveserver2;' \
                r'UID=huero;PWD=huero'


## file_length - Renvoie la longueur d'un fichier
#
# Prend en paramètre le fichier accessible via le chemin "Filename" passé en paramètre,
#  et renvoie sa longueur (nombre de lignes).
#
# @param filename [str] Chemin du fichier pour lequel on souhaite récupérer la longueur.
# @return [int] Nombre de lignes du fichier
def file_length(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
    n = sum(1 for line in open(filename, encoding='utf8'))
    return n


## df_from_lake - Récupère des données du lac
#
# Fonction permettant de récupérer des données du lac via une requête donnée.
# Les données sont renvoyées sous la forme d'un dataframe.
# Requires :
# https://s3.amazonaws.com/public-repo-1.hortonworks.com/HDP/hive-odbc/2.1.10.1014/windows/HortonworksHiveODBC64.msi
#
# @param query [str] Requête SQL/HQL permettant de récupérer les données souhaitées
# @return DataFrame (pandas) contenant les données souhaitées.
def df_from_lake(query):
    # Requires :
    # https://s3.amazonaws.com/public-repo-1.hortonworks.com/HDP/hive-odbc/2.1.10.1014/windows/HortonworksHiveODBC64.msi
    cnx = pyodbc.connect(HIVE_ODBC_URL,
                         autocommit=True)
    df = pd.read_sql(query, cnx)
    cnx.close()
    return df


## df_from_csv_generator - Générateur de DataFrames
#
# Fonction permettant de générer un DataFrame (pandas) par ligne du csv d'origine
# Le fichier doit être un CSV sur 2 colonnes minimum :
#  La première colonne sera considérée comme un document
#  La seconde colonne sera considérée comme un tag
#
# @param filename       [str] Chemin du fichier (.csv)
# @param verbose_text   [str] Chaine à afficher pour suivre l'avancement du générateur (par défaut : chaine vide)
# @return Générateur de DataFrames
def df_from_csv_generator(filename, verbose_text=""):
    if not os.path.isfile(filename):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
    with(open(filename, encoding='utf8')) as f:
        for i, l in enumerate(f):
            if i % 1000 == 0:
                print(verbose_text+' %i ' %i)
            df = pd.read_csv(StringIO(l), names=['docs', 'tags'])
            yield df


## sample_from_gen - Extraction a partir d'un générateur
#
# Extrait un échantillon de la taille souhaitée à partir d'un générateur donné.
#
# @param gen    Générateur à partir duquel extraire un échantillon
# @param n      [int] Taille de l'échantillon souhaité
# @return Liste de taille n contenant des n premiers éléments de gen
def sample_from_gen(gen, n=1000):
    return list(itertools.islice(gen, n))




FILE_ALIASES = ['file', 'filename', 'fichier', 'data']

# Chercher un nom de fichier dans les arguments
#
# Si un [str] correspondant à un fichier existant est présent en première position dans args
# ou si un [str] correspondant à un fichier existant est présent dans kwargs avec un des noms
# de FILE_ALIASES; alors l'argument est retiré de la liste et est retourné.
#
# @param *args          liste non nommée d'arguments
# @param **kwargs       dictionnaire d'arguments
# @return found         [bool] True si un fichier a été trouvé
# @return filename      [str]  le nom du fichier trouvé ('' par défaut)
# @return args          la liste non nommée des arguments restants
# @return kwargs        la liste nommée des arguments restants
def find_file_in_args(*args, **kwargs):
    found = False
    filename = ''
    if len(args) > 0:
        if isinstance(args[0], str) and os.path.isfile(args[0]):
            found = True
            filename = args[0]
            args = [x for x in args[1:]]
    if not found:
        for item in FILE_ALIASES:
            if item in kwargs.keys() and isinstance(kwargs[item], str) and os.path.isfile(kwargs[item]):
                found = True
                filename = kwargs[item]
                del kwargs[item]
                break
    return found, filename, args, kwargs


DATA_FRAME_ALIASES = ['docs', 'data', 'df']


# Chercher une pandas.DataFrame dans les arguments
#
# Si un objet correspondant à une pandas.DataFrame est présent en première position dans args
# ou si un objet correspondant à une pandas.DataFrame est présent dans kwargs avec un des noms
# de DATA_FRAME_ALIASES; alors l'argument est retiré de la liste et est retourné.
#
# @param *args          liste non nommée d'arguments
# @param **kwargs       dictionnaire d'arguments
# @return found         [bool] True si une pandas.DataFrame a été trouvé
# @return filename      [pandas.DataFrame]  la pandas.DataFrame retournée
# @return args          la liste non nommée des arguments restants
# @return kwargs        la liste nommée des arguments restants
def find_df_in_args(*args, **kwargs):
    found = False
    df = None
    if len(args) > 0:
        if isinstance(args[0], pd.DataFrame):
            found = True
            df = args[0]
            args = [x for x in args[1:]]
    if not found:
        for item in DATA_FRAME_ALIASES:
            if item in kwargs.keys() and isinstance(kwargs[item], pd.DataFrame):
                found = True
                df = kwargs[item]
                del kwargs[item]
                break
    return found, df, args, kwargs


# Fonction à utiliser comme decorator
#
# Permet de passer un nom de fichier contenant des données csv à une
# fonction attendant une pandas.DataFrame. En pratique, la fonction décorée
# est appelée x fois avec chunk_size éléments des données lues dans le csv
#
# @param modify_data      [bool] si la fonction décorée va modifier les données en input
# @param chunk_size       [int]  taille des pandas.DataFrame lues dans le fichier à chaque itération
# @param output_file      [str]  nom du fichier de sortie (si non spécifié : fichier temporaire)
# @return Si la fonction décorée est appelée avec une pandas.DataFrame ou si modify_data == False : comportement normal,
# si modify_data==True alors ce decorator retourne un nom de fichier contenant les données modifiées
def data_agnostic_function(modify_data=False, chunk_size=1, output_file=None):
    def decorate(func):
        def func_wrapper(*args, **kwargs):
            has_file, filename, args, kwargs = find_file_in_args(*args, **kwargs)
            if has_file:
                print('Found file, proceeding with streaming on %s' % filename)
                gen = pd.read_csv(filename,  chunksize=chunk_size)
                if modify_data:
                    with (tempfile.NamedTemporaryFile(delete=False)) as tmp:
                        for chunk in gen:
                            res = func(chunk, *args, **kwargs)
                            res.to_csv(tmp.name, mode='a', index=False)
                    if output_file is not None:
                        shutil.copy(tmp, output_file)
                        os.remove(tmp.name)
                        return output_file
                    else:
                        print('Results stored in %s' % tmp.name)
                        return tmp.name
                else:
                    return [func(chunk, *args, **kwargs) for chunk in gen]

            else:
                print('File not found in args, using Pandas DataFrame')
                return func(*args, **kwargs)
        return func_wrapper
    return decorate

# Fonction à utiliser comme decorator
#
# Permet de passer une pandas.DataFrame ayant plusieurs colonnes
# à une fonction qui n'attend qu'une pandas.Series. La première colonne
# OU une colonne s'appelant col_names est passée à la fonction décorée;
# les autres colonnes sont sauvegardées et rajoutées au résultat de la fonction
#
# @param col_names      [str] nom de la colonne contenant les données à passer
# @return le même objet que celui retourné par la fonction décorée auquel sont rajoutés
# les autres colonnes
def process_docs_keep_everything(col_names='docs'):
    def decorate(func):
        def func_wrapper(*args, **kwargs):
            has_df, df, args, kwargs = find_df_in_args(*args, **kwargs)
            if has_df:
                if df.shape[1] > 1:
                    print('DataFrame found in args, passing docs, keeping tags')
                    if col_names in df.columns.values:
                        docs = df[col_names]
                        docs = func(docs, *args, **kwargs)
                        df[col_names] = docs
                        return df
                    else:
                        docs = df[df.columns[0]]
                        docs = func(docs, *args, **kwargs)
                        df[df.columns[0]] = docs
                        return df
                else:
                    print('DataFrame found in args, but no tags found, passing df')
                    return pd.DataFrame(func(df[df.columns[0]], *args, **kwargs))
            else:
                return func(*args, **kwargs)
        return func_wrapper
    return decorate
