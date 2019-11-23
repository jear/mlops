Preprocessing ( version light ) :

- Lancer preprocess.py


Séparation du jeu d'entrainement et du jeu de test :

- lancer SplitMailsTrainTest

Entraînementdu réseau de neurones :

- Lancer train.py

Prédiction sur des nouveaux mails

- Lancer predict.py

Evaluation des résultats :

- Lancer eval.py

--------------------------------------------------------------

La pipeline classique d'utilisation :

Preprocess, puis split, puis train , puis predict, puis eval .


NB : les chemins sont entrés en durs, en relative path .

Par défaut le train est lançé sur le fichier en sortie du split.
Par défaut la phase de prétraitement n'est pas incluse dans la pipeline. Pour l'inclure, il faut changer le fichier en entrée du split ( variable pandaToSplit devient pandaToSplit = pd.read_csv('../data/sample_preprocessed.csv', sep=';', header=0, encoding='latin-1') ), puis appliquer l'entrainement sur le texte prépocéssé ( mettre : messages = pandaFinal['Texte_preprocessed'][0:].values )



NB2 : les packages nécessaires pour lancer les différents scripts se trouvent dans requirements.txt