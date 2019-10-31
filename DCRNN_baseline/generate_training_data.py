from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
import gc
import h5py
import time
from tqdm import tqdm
from tqdm._tqdm import trange


def generate_graph_seq2seq_io_data(
        x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    file=h5py.File('speed_list.h5','r')
    speed_data = file['speed_data'][:]
    time_data = file['time_id'][:]
    file.close()

    num_samples, num_nodes = speed_data.shape
    print("num_samples = ", num_samples)
    print("num_nodes = ", num_nodes)
    data = np.expand_dims(speed_data.values, axis=-1)
    data_list = [data]

    ####################### Q_Traffic ##########################
    
    q_index = time_data
    time_ind = np.true_divide(q_index - np.floor_divide(q_index,96) * 96.0, 96.0)
    time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
    data_list.append(time_in_day)

    ######################## Origin code ########################

    # if add_time_in_day:
    #     time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    #     time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
    #     data_list.append(time_in_day)
    # if add_day_in_week:
    #     day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
    #     day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
    #     data_list.append(day_in_week)

    ############################################################
    print("Start concatenating...")
    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in tqmd(range(min_t, max_t)):
        x_t = data[t + x_offsets, ...]
        x.append(x_t)
        y_t = data[t + y_offsets, ...]
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

def generate_train_val_test(args):
    print("Read h5py success.")
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="/home/blu/workspace/graduate_project/STGCN/data_processing/", help="Output directory."
    )
    parser.add_argument(
        "--traffic_filename",
        type=str,
        default="/home/blu/workspace/graduate_project/STGCN/DCRNN_baseline/speed_list.h5",
        help="Raw traffic readings.",
    )
    args = parser.parse_args()
    main(args)
