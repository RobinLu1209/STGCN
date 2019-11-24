import numpy as np
import torch
import os

def load_metr_la_data():
    A = np.load("/home/blu/workspace/GCN/STGCN-PyTorch/data/adj_mat.npy")
    X = np.load("/home/blu/workspace/GCN/STGCN-PyTorch/data/node_values.npy").transpose((1, 2, 0))
    X = X.astype(np.float32)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    return A, X, means, stds

def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))

A, X, means, stds = load_metr_la_data()
split_line1 = int(X.shape[2] * 0.6)
split_line2 = int(X.shape[2] * 0.8)

train_original_data = X[:, :, :split_line1]
val_original_data = X[:, :, split_line1:split_line2]
test_original_data = X[:, :, split_line2:]

print("INFO: train:", train_original_data.shape)
print("INFO: val:", val_original_data.shape)
print("INFO: test:", test_original_data.shape)

num_timesteps_input = 12
num_timesteps_output = 1

training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)

print("INFO: x_train:", training_input.shape, "y_train:", training_target.shape)
print("INFO: x_val:", val_input.shape, "y_val:", val_target.shape)
print("INFO: x_test:", test_input.shape, "y_test:", test_target.shape)

np.save("/home/blu/workspace/graduate_project/STGCN/METR-LA/x_train.npy", training_input)
np.save("/home/blu/workspace/graduate_project/STGCN/METR-LA/y_train.npy", training_target)
np.save("/home/blu/workspace/graduate_project/STGCN/METR-LA/x_val.npy", val_input)
np.save("/home/blu/workspace/graduate_project/STGCN/METR-LA/y_val.npy", val_target)
np.save("/home/blu/workspace/graduate_project/STGCN/METR-LA/x_test.npy", test_input)
np.save("/home/blu/workspace/graduate_project/STGCN/METR-LA/y_test.npy", test_target)
