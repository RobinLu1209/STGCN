from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

x_train = np.load("/home/blu/workspace/graduate_project/STGCN/METR-LA/dataset/x_train.npy")
y_train = np.load("/home/blu/workspace/graduate_project/STGCN/METR-LA/dataset/y_train.npy")
x_val = np.load("/home/blu/workspace/graduate_project/STGCN/METR-LA/dataset/x_val.npy")
y_val = np.load("/home/blu/workspace/graduate_project/STGCN/METR-LA/dataset/y_val.npy")
x_test = np.load("/home/blu/workspace/graduate_project/STGCN/METR-LA/dataset/x_test.npy")
y_test = np.load("/home/blu/workspace/graduate_project/STGCN/METR-LA/dataset/y_test.npy")
A = np.load("/home/blu/workspace/GCN/STGCN-PyTorch/data/adj_mat.npy")

class GCN(tf.keras.layers.Layer):
  def __init__(self, units=207, input_dim=207, matrix = None):
    super(GCN, self).__init__()
    w_init = tf.random_normal_initializer()
    self.w1 = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                              dtype='float32'), trainable=True)
    self.w2 = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                              dtype='float32'), trainable=True)
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(initial_value=b_init(shape=(1,),
                                              dtype='float32'), trainable=True)
    self.matrix = tf.convert_to_tensor(matrix)
    self.dense = tf.keras.layers.Dense(units=207, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(1e-3))

  def call(self, inputs):
    weight_1 = tf.multiply(self.matrix,self.w1)
    weight_2 = tf.multiply(self.matrix,self.w2)
    inputs = tf.reshape(inputs, [-1, 207])
    y1 = tf.nn.relu(tf.matmul(inputs, weight_1))
    y2 = tf.nn.relu(tf.matmul(y1, weight_2) + self.b)
    output = tf.reshape(y2, [-1, 12, 207])
    return output

encoder_emb_inp = tf.keras.Input(shape=(12, 207), name='Input')

gcn_layer = GCN(207, 207, A)
gcn_output = gcn_layer(encoder_emb_inp)

x = tf.concat([gcn_output, encoder_emb_inp], axis = 1)
gru1 = tf.keras.layers.GRU(256, return_sequences=True, return_state=False, activation='tanh')(x)
gru2 = tf.keras.layers.GRU(256, return_sequences=False, return_state=False, activation='relu')(gru1)
gru2_out = tf.keras.layers.Dense(units=207, activation=tf.nn.relu)(gru2)

model = tf.keras.Model(inputs = encoder_emb_inp, outputs = gru2_out, name = "system_model")

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError(), 
              metrics=[tf.keras.metrics.MeanSquaredError(), 
                       tf.keras.metrics.MeanAbsolutePercentageError(), 
                       tf.keras.metrics.MeanAbsoluteError(), 
                       tf.keras.metrics.RootMeanSquaredError()])
print("INFO: Start training...")

history = model.fit(x_train, 
                    y_train,
                    batch_size=1024,
                    epochs=800,
                    validation_data=(x_val,y_val))

print("INFO: Start testing...")
dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
dataset = dataset.batch(512)
model.evaluate(dataset)
