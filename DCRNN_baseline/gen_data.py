import pandas as pd
import numpy as np
import h5py

h5f = h5py.File('speed_list.h5', 'r')
speed_data = h5f['data']
h5f.close()

print(type(speed_data))


