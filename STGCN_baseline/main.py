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


def train_epoch(training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(A_wave, X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)


if __name__ == '__main__':
    torch.manual_seed(7)

    A, X, means, stds = load_metr_la_data()

    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)

    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    # test_original_data = X[:, :, split_line2:]

    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    # test_input, test_target = generate_dataset(test_original_data,
    #                                            num_timesteps_input=num_timesteps_input,
    #                                            num_timesteps_output=num_timesteps_output)
    print("INFO: Load data finish!")
    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)

    A_wave = A_wave.to(device=args.device)

    net = STGCN(A_wave.shape[0],
                training_input.shape[3],
                num_timesteps_input,
                num_timesteps_output).to(device=args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []
    validation_maes = []
    for epoch in tqdm(range(epochs)):
        loss = train_epoch(training_input, training_target,
                           batch_size=batch_size)
        training_losses.append(loss)
        torch.cuda.empty_cache()
        # Run validation
        permutation = torch.randperm(val_input.shape[0])
        with torch.no_grad():
            epoch_val_losses = []
            epoch_val_mae = []
            for i in range(0, val_input.shape[0], batch_size):
                net.eval()
                indices = permutation[i:i + batch_size]
                X_batch, y_batch = val_input[indices], val_target[indices]
                X_batch = X_batch.to(device=args.device)
                y_batch = y_batch.to(device=args.device)
                out = net(A_wave, X_batch)
                out_unnormalized = out.detach().cpu().numpy()*stds[0]+means[0]
                target_unnormalized = y_batch.detach().cpu().numpy()*stds[0]+means[0]
                mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
                loss = loss_criterion(out, y_batch)
                epoch_val_losses.append(loss.detach().cpu().numpy())
                epoch_val_mae.append(mae)
            val_loss = sum(epoch_val_losses)/len(epoch_val_losses)
            val_mae = sum(epoch_val_mae)/len(epoch_val_mae)
            validation_losses.append(val_loss)
            validation_maes.append(val_mae)

        print("Training loss: {}".format(training_losses[-1]))
        print("Validation loss: {}".format(validation_losses[-1]))
        print("Validation MAE: {}".format(validation_maes[-1]))
        # plt.plot(training_losses, label="training loss")
        # plt.plot(validation_losses, label="validation loss")
        # plt.legend()
        # plt.show()

        checkpoint_path = "checkpoints/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        with open("checkpoints/losses.pk", "wb") as fd:
            pk.dump((training_losses, validation_losses, validation_maes), fd)
        
    print("INFO: Training process is finished.")
    torch.save(net.state_dict(), 'parameter.pkl')
    
