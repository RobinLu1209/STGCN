import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import scipy.sparse as sp
import tensorflow as tf

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return normalized_laplacian

def calculate_random_walk_matrix(adj_mx):
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx)
    return random_walk_mx

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)

x_train = np.load("/home/blu/workspace/graduate_project/STGCN/METR-LA/new_dataset/x_train.npy")
y_train = np.load("/home/blu/workspace/graduate_project/STGCN/METR-LA/new_dataset/y_train.npy")
time_train = np.load("/home/blu/workspace/graduate_project/STGCN/METR-LA/new_dataset/time_train.npy")
x_val = np.load("/home/blu/workspace/graduate_project/STGCN/METR-LA/new_dataset/x_val.npy")
y_val = np.load("/home/blu/workspace/graduate_project/STGCN/METR-LA/new_dataset/y_val.npy")
time_val = np.load("/home/blu/workspace/graduate_project/STGCN/METR-LA/new_dataset/time_val.npy")
x_test = np.load("/home/blu/workspace/graduate_project/STGCN/METR-LA/new_dataset/x_test.npy")
y_test = np.load("/home/blu/workspace/graduate_project/STGCN/METR-LA/new_dataset/y_test.npy")
time_test = np.load("/home/blu/workspace/graduate_project/STGCN/METR-LA/new_dataset/time_test.npy")

y_train = y_train[:,0,:]
y_val = y_val[:,0,:]
y_test = y_test[:,0,:]

time_train_df = pd.DataFrame(time_train[:,0,:], columns = ['week', 'hour', 'minute'])
week_oh = tf.keras.utils.to_categorical(time_train_df['week'])
hour_oh = tf.keras.utils.to_categorical(time_train_df['hour'])
minute_oh = tf.keras.utils.to_categorical(time_train_df['minute'])
print("INFO: Train | week_oh:", week_oh.shape, "hour_oh:", hour_oh.shape, "minute_oh:", minute_oh.shape)
time_train_2 = tf.concat([week_oh, hour_oh, minute_oh], axis = 1)

time_val_df = pd.DataFrame(time_val[:,0,:], columns = ['week', 'hour', 'minute'])
week_oh = tf.keras.utils.to_categorical(time_val_df['week'])
hour_oh = tf.keras.utils.to_categorical(time_val_df['hour'])
minute_oh = tf.keras.utils.to_categorical(time_val_df['minute'])
print("INFO: Val | week_oh:", week_oh.shape, "hour_oh:", hour_oh.shape, "minute_oh:", minute_oh.shape)
time_val_2 = tf.concat([week_oh, hour_oh, minute_oh], axis = 1)

time_test_df = pd.DataFrame(time_test[:,0,:], columns = ['week', 'hour', 'minute'])
week_oh = tf.keras.utils.to_categorical(time_test_df['week'])
hour_oh = tf.keras.utils.to_categorical(time_test_df['hour'])
minute_oh = tf.keras.utils.to_categorical(time_test_df['minute'])
print("INFO: Test | week_oh:", week_oh.shape, "hour_oh:", hour_oh.shape, "minute_oh:", minute_oh.shape)
time_test_2 = tf.concat([week_oh, hour_oh, minute_oh], axis = 1)

print("INFO: x_train:", x_train.shape, "y_train:", y_train.shape, "time_train", time_train_2.shape)
print("INFO: x_val:", x_val.shape, "y_val:", y_val.shape, "time_val", time_val_2.shape)
print("INFO: x_test:", x_test.shape, "y_test:", y_test.shape, "time_test", time_test_2.shape)

A = np.load("/home/blu/workspace/GCN/STGCN-PyTorch/data/adj_mat.npy")
random_walk = calculate_random_walk_matrix(A)
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
time_inp = tf.keras.Input(shape=(43,), name='time_Input')

gcn_layer = GCN(207, 207, random_walk)
gcn_output = gcn_layer(encoder_emb_inp)

x = tf.concat([gcn_output, encoder_emb_inp], axis = 1)
gru1 = tf.keras.layers.GRU(256, return_sequences=True, return_state=False, activation='tanh')(x)
gru2 = tf.keras.layers.GRU(256, return_sequences=False, return_state=False, activation='relu')(gru1)

concat_out = tf.concat([gru2, time_inp], axis = 1)
output = tf.keras.layers.Dense(units=207, activation=tf.nn.relu)(concat_out)

model = tf.keras.Model(inputs = [encoder_emb_inp, time_inp], outputs = output, name = "system_model")

# predict = model.predict([x_train[0:10,:,:],time_train_2[0:10,:]])
# print(predict.shape)
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanSquaredError(),
                       tf.keras.metrics.MeanAbsolutePercentageError(),
                       tf.keras.metrics.MeanAbsoluteError(),
                       tf.keras.metrics.RootMeanSquaredError()])
print("INFO: Start training...")

history = model.fit([x_train, time_train_2],
                    y_train,
                    batch_size=1024,
                    epochs=520,
                    validation_data=([x_val,time_val_2],y_val))

print("INFO: Start testing...")
dataset = tf.data.Dataset.from_tensor_slices(([x_test,time_test_2], y_test))
dataset = dataset.batch(512)
model.evaluate(dataset)
