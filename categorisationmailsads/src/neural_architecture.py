from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Conv1D, MaxPooling1D
from keras.initializers import RandomUniform
from keras.layers import Dense, Dropout, Flatten, ZeroPadding1D, GlobalAveragePooling1D
from keras.layers.embeddings import Embedding





def ltsm_model(embeddings,
				vocabulary_size = 50000,
				embed_size = 100,
				message_length = 250,
				nbtargets = 22,
				chosenLoss = 'categorical_crossentropy',
				chosenOptimizer = 'adam') :
    """
    Architecture used for v2 : classify PE mails among the 21 Cogito categories

    embeddings : np.array,
        Pretrained embedding matrix.

    seq_max : int, optional
        Maximum input length.
        Default value, 250.

    embed_size : size of the vector used for a word. This parameter is set during the embedding process.
        Default value : 100

     loss : str, optional
        Loss function for training.
        Default value, 'categorical_crossentropy'.

    ntargets : int, optional
        Dimension of model output.
        Default value, 22.

    Returns
    -------
    Model instance

    """
    model = Sequential()
    model.add(Embedding(vocabulary_size, embed_size, weights=[embeddings], input_length=message_length, trainable=False))
    model.add(LSTM(150, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(75, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nbtargets, activation='softmax'))
    model.compile(loss = chosenLoss, optimizer = chosenOptimizer, metrics=['accuracy'])
    return model



def cnn_model_test(embeddings,
				vocabulary_size = 50000,
				embed_size = 100,
				message_length = 250,
				nbtargets = 22,
				chosenLoss = 'categorical_crossentropy',
				chosenOptimizer = 'adam') :

    model = Sequential()
    model.add(Embedding(vocabulary_size, embed_size, weights=[embeddings], input_length=message_length, trainable=False))

    model.add(ZeroPadding1D(1))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))

    model.add(ZeroPadding1D(1))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Conv1D(64, 3, activation='relu'))

    model.add(ZeroPadding1D(1))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(Conv1D(32, 3, activation='relu'))

    model.add(GlobalAveragePooling1D('channels_last'))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nbtargets, activation='softmax'))
    model.compile(optimizer=chosenOptimizer, loss=chosenLoss, metrics=['accuracy'])
    model.summary()
    return model