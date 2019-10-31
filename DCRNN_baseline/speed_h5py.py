import pandas as pd
import numpy as np
import h5py
import time
from tqdm import tqdm
from tqdm._tqdm import trange

df = pd.read_table('/home/blu/jupyter_files/Data_test/traffic_speed_sub-dataset.v2', sep = ',', names = ["road_id","time","speed"])
time_col = df['time'].unique()
road_col = df['road_id'].unique()
time_num = len(time_col)
road_num = len(road_col)
print("INFO: time_num = ", time_num)
print("INFO: road_num = ", road_num)

speed_list = (df.loc[df['time'] == 0]['speed'][:]).values

for i in tqdm(range(1,time_num)):
    add_array = (df.loc[df['time'] == i]['speed'][:]).values
    speed_list = np.row_stack((speed_list, add_array))

print("INFO: Start saving...")

file = h5py.File('speed_list.h5','w')
file.create_dataset('speed_data', data = speed_list)
file.create_dataset('time_id', data = time_col)
file.close()

print("INFO: h5py file finish!")

with open("success","w") as f:
        f.write("success！")


