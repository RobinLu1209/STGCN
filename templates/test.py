
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

pwd = '/home/blu/workspace/graduate_project/STGCN/beijing_traffic/mini_data/'

#train
x_train = np.load(pwd + "x_train.npy")
y_train = np.load(pwd + "y_train.npy")
#val
x_val = np.load(pwd + "x_val.npy")
y_val = np.load(pwd + "y_val.npy")

x_train_mean = x_train.mean()
x_train_std = x_train.std()
y_train_mean = y_train.mean()
y_train_std = y_train.std()
x_train_uni = (x_train - x_train.mean())/x_train_std
y_train_uni = (y_train - y_train.mean())/y_train_std

x_val_mean = x_val.mean()
x_val_std = x_val.std()
y_val_mean = y_val.mean()
y_val_std = y_val.std()
x_val_uni = (x_val - x_val.mean())/x_val_std
y_val_uni = (y_val - y_val.mean())/y_val_std

print("x_train_uni:", x_train_uni.shape, "y_train_uni:", y_train_uni.shape)
print("x_val_uni:", x_val_uni.shape, "y_val_uni:", y_val_uni.shape)


en_num_units = 128
de_num_units = 128

num_outputs = 216
encoder_emb_inp = tf.keras.Input(shape=(12, 216), name='Input')

encoder = tf.keras.layers.LSTM(en_num_units, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_emb_inp)
encoder_state = [state_h, state_c]

print(encoder_outputs.shape)

decoder_emb_inp = encoder_outputs

# Sampler
sampler = tfa.seq2seq.sampler.TrainingSampler()

# Decoder
decoder_cell = tf.keras.layers.LSTMCell(de_num_units)
projection_layer = tf.keras.layers.Dense(num_outputs)
decoder = tfa.seq2seq.BasicDecoder(
            decoder_cell, sampler, output_layer=projection_layer)

outputs, _, _ = decoder(decoder_emb_inp, initial_state=encoder_state, sequence_length=(12,))

logits = outputs.rnn_output


model = tf.keras.Model(inputs=encoder_emb_inp, outputs=logits, name='seq2seq')

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())

history = model.fit(x_train_uni, y_train_uni, batch_size=64, epochs=1, validation_split=0.1)





"""
# Sampler
sampler = tfa.seq2seq.sampler.TrainingSampler()

# Decoder
decoder_cell = tf.keras.layers.LSTMCell(de_num_units)
projection_layer = tf.keras.layers.Dense(num_outputs)
decoder = tfa.seq2seq.BasicDecoder(
            decoder_cell, sampler, output_layer=projection_layer)

outputs, _, _ = decoder(
            decoder_emb_inp,
                initial_state=encoder_state,
                    sequence_length=decoder_lengths)
logits = outputs.rnn_output
"""
