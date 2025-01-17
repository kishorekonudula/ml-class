import wandb
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU
from tensorflow.python.client import device_lib
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.datasets import imdb
import util

# set parameters:
wandb.init()
config = wandb.config
config.vocab_size = 1000 # Increasing this will improve accuracy (10000) Most commons words in the corpus
config.maxlen = 300 # Restricting the input to 300 words per document
config.batch_size = 32
config.embedding_dims = 50 # This is something we should know about the word embedding
config.filters = 250
config.kernel_size = 3
config.hidden_dims = 100 # Size of the vector thats passing between LSTM layers. Higher the number, more complicated the LSTM. This can overfit
config.epochs = 10

# Load and tokenize input
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=config.vocab_size)

# Example of manual data loading
"""
import util
(X_train, y_train), (X_test, y_test) = util.load_imdb()
tokenizer = text.Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
"""

# Ensure all input is the same size
X_train = sequence.pad_sequences(
    X_train, maxlen=config.maxlen)
X_test = sequence.pad_sequences(
    X_test, maxlen=config.maxlen)

# overide LSTM & GRU
if 'GPU' in str(device_lib.list_local_devices()):
    print("Using CUDA for RNN layers")
    LSTM = CuDNNLSTM
    GRU = CuDNNGRU

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(config.vocab_size,
                                    config.embedding_dims,
                                    input_length=config.maxlen))
model.add(tf.keras.layers.Bidirectional(LSTM(config.hidden_dims))) # Run the LSTM in both directions
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=config.batch_size,
          epochs=config.epochs,
          validation_data=(X_test, y_test), callbacks=[util.TextLogger(X_test[:20], y_test[:20]),
                                                       wandb.keras.WandbCallback(save_model=False)])

model.save("seniment.h5")
