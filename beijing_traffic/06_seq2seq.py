from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_addons as tfa

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import pandas as pd


pwd = '/home/blu/workspace/graduate_project/STGCN/beijing_traffic/mini_data/'

#train
x_train = np.load(pwd + "x_train.npy")
y_train = np.load(pwd + "y_train.npy")
x_train_mean = x_train.mean()
x_train_std = x_train.std()
y_train_mean = y_train.mean()
y_train_std = y_train.std()
x_train_uni = (x_train - x_train.mean())/x_train_std
y_train_uni = (y_train - y_train.mean())/y_train_std

#val
x_val = np.load(pwd + "x_val.npy")
y_val = np.load(pwd + "y_val.npy")
x_val_mean = x_val.mean()
x_val_std = x_val.std()
y_val_mean = y_val.mean()
y_val_std = y_val.std()
x_val_uni = (x_val - x_val.mean())/x_val_std
y_val_uni = (y_val - y_val.mean())/y_val_std

#test
x_test = np.load(pwd + "x_test.npy")
y_test = np.load(pwd + "y_test.npy")
x_test_mean = x_test.mean()
x_test_std = x_test.std()
y_test_mean = y_test.mean()
y_test_std = y_test.std()
x_test_uni = (x_test - x_test.mean())/x_test_std
y_test_uni = (y_test - y_test.mean())/y_test_std

x_train = x_train_uni
x_val = x_val_uni
y_train = np.squeeze(y_train_uni)
y_val = np.squeeze(y_val_uni)
x_test = x_test_uni
y_test = np.squeeze(y_test_uni)

print("INFO: x_train:", x_train.shape, "y_train:", y_train.shape)
print("INFO: x_val:", x_val.shape, "y_val:", y_val.shape)
print("INFO: x_test:", x_test.shape, "y_test:", y_test.shape)

# #Basic LSTM
# encoder_emb_inp = tf.keras.Input(shape=(12, 216), name='Input')
# # print("Input.shape = ", encoder_emb_inp.shape)
# lstm1,state_h,state_c = tf.keras.layers.LSTM(256, return_sequences=False, return_state=True)(encoder_emb_inp)
# lstm2 = tf.keras.layers.LSTM(256, return_sequences=False)(lstm1[:,np.newaxis,:])
# lstm2_out = tf.keras.layers.Dense(units=216, activation=tf.nn.relu)(lstm2)
# model = tf.keras.Model(inputs = encoder_emb_inp, outputs = lstm2_out, name = "seq2seq")
# # y = model.predict(x_train[0:3])
# # y.shape
# model.summary()
# model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError(), 
#               metrics=[tf.keras.metrics.MeanSquaredError(), 
#                        tf.keras.metrics.MeanAbsolutePercentageError(), 
#                        tf.keras.metrics.MeanAbsoluteError(), 
#                        tf.keras.metrics.RootMeanSquaredError()])

#Basic LSTM
# encoder_emb_inp = tf.keras.Input(shape=(12, 216), name='Input')
# gru1 = tf.keras.layers.GRU(216, return_sequences=False, return_state=False, activation='tanh')(encoder_emb_inp)
# gru2 = tf.keras.layers.GRU(216, return_sequences=False, return_state=False, activation='relu')(gru1[:,np.newaxis,:])
# model = tf.keras.Model(inputs = encoder_emb_inp, outputs = gru2, name = "gru")
# model.summary()

#BiRNN
encoder_emb_inp = tf.keras.Input(shape=(12, 216), name='Input')
biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(216, return_sequences=False, return_state=False))(encoder_emb_inp)
lstm2 = tf.keras.layers.LSTM(216, return_sequences=False)(biLSTM[:,np.newaxis,:])
model = tf.keras.Model(inputs = encoder_emb_inp, outputs = lstm2, name = "BiRNN")
model.summary()

#Basic
# encoder_emb_inp = tf.keras.Input(shape=(12, 216), name='Input')
# # print("Input.shape = ", encoder_emb_inp.shape)
# lstm1,state_h,state_c = tf.keras.layers.LSTM(216, return_sequences=False, return_state=True, activation = tf.nn.relu, use_bias = True)(encoder_emb_inp)
# print("lstm1.shape = ", lstm1.shape)
# print("state_h.shape = ", state_h.shape)
# print("state_c.shape = ", state_c.shape)
# print('lstm1.shape = ', lstm1[:,np.newaxis,:].shape)
# lstm2 = tf.keras.layers.LSTM(216, return_sequences=False)(lstm1[:,np.newaxis,:])
# model = tf.keras.Model(inputs = encoder_emb_inp, outputs = [lstm2], name = "seq2seq")
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError(), 
              metrics=[tf.keras.metrics.MeanSquaredError(), 
                       tf.keras.metrics.MeanAbsolutePercentageError(), 
                       tf.keras.metrics.MeanAbsoluteError(), 
                       tf.keras.metrics.RootMeanSquaredError()])

print("INFO: Start training...")

history = model.fit(x_train, 
                    y_train,
                    batch_size=256,
                    epochs=200,
                    validation_data=(x_val,y_val))

print("INFO: Start testing...")
dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
dataset = dataset.batch(256)
model.evaluate(dataset)

model.save('/home/blu/workspace/graduate_project/STGCN/beijing_traffic/seq2seq_gru.h5')
