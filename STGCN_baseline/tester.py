import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from stgcn import STGCN
from utils import generate_dataset, load_metr_la_data, get_normalized_adj

use_gpu = True
num_timesteps_input = 12
num_timesteps_output = 3

epochs = 50
batch_size = 50

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true', default = True, 
                    help='Enable CUDA')
args = parser.parse_args()
args.device = None
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    print("Use GPU.")
else:
    args.device = torch.device('cpu')
    print("Use CPU.")


if __name__ == '__main__':
    torch.manual_seed(7)
    A, X, means, stds = load_metr_la_data()
    split_line2 = int(X.shape[2] * 0.8)
    test_original_data = X[:, :, split_line2:]
    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)
    print("INFO: Test data load finish!")
    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)

    A_wave = A_wave.to(device=args.device)

    net = STGCN(A_wave.shape[0],
                test_input.shape[3],
                num_timesteps_input,
                num_timesteps_output).to(device=args.device)
    
    net.load_state_dict(torch.load('parameter.pkl'))
    print("INFO: Load model finish!")

    loss_criterion = nn.MSELoss()
    
    permutation = torch.randperm(test_input.shape[0])
    epoch_test_losses = []
    epoch_test_mae = []
    for i in range(0, test_input.shape[0], batch_size):
        net.eval()
        indices = permutation[i:i + batch_size]
        X_batch, y_batch = test_input[indices], test_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)
        out = net(A_wave, X_batch)
        out_unnormalized = out.detach().cpu().numpy()*stds[0]+means[0]
        target_unnormalized = y_batch.detach().cpu().numpy()*stds[0]+means[0]
        mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
        loss = loss_criterion(out, y_batch)
        epoch_test_losses.append(loss.detach().cpu().numpy())
        epoch_test_mae.append(mae)
    test_losses = sum(epoch_test_losses)/len(epoch_test_losses)
    test_mae = sum(epoch_test_mae)/len(epoch_test_mae)
    print("Test loss: {}".format(test_losses))
    print("Test MAE: {}".format(test_mae))
