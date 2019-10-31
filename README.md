# Spatial-Temporal Data Analysis on Graph Convolutional Network


## Data

The data provider gives 15073 central road and its neighbour information, so there are totally 45148 roads data(speed/road netwok/gps) provided. The total time slot number is 5856(61days * 24hours * 4quarter).

Filename | Dimension | Instance | Tips
:-:|:-:|:-:|:-: 
traffic_speed_sub-dataset | 3 * (5856*45148) | road_id = 1562548955, timeslot_id = 0, speed = 41.3480687196 | No headings, sep = ' '
road_network_sub-dataset | 8 * 45148(-Heading) | road_id = 1562548955, width = 30, direction = 3, snodeid = 1520445066, enodeid = 1549742690, length = 0.038, speedclass = 6, lanenum = 1 | Headings, sep = '\t'
link_gps | 3 * 45418 | road_id = 1562548955, longtitude = 116.367557, latitude = 39.899537 | No headings, sep = ' ' 
query_sub-dataset | 61 * 6 * N | search_time = 2017-04-01 19:42:23, start_pos = (116.325461 40.036083), end_pos = (116.350811 40.090999), travel_time = 33 | No headings, sep = ' ' or ','
neighbours_1km.txt | 15073 * 11 | road_id = xx, pre1, pre2, ..., pre5, next1, next2, ..., next5 | 


## Baseline Analysis

#### DCRNN
1. Data pre-processing
	- [speed_h5py.py](https://github.com/RobinLu1209/STGCN/blob/master/DCRNN_baseline/speed_h5py.py) is used to generate speed dataset in h5 format. 
	- Then, use this speed_dataset to generate train/validate/test data by code [generate_training_data.py](https://github.com/RobinLu1209/STGCN/blob/master/DCRNN_baseline/generate_training_data.py) .
	- [gen_adj_mx.py](https://github.com/RobinLu1209/STGCN/blob/master/DCRNN_baseline/gen_adj_mx.py) is used to generate road_map.



## Basic Models
1. ChebNet: [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://github.com/mdeff/cnn_graph)
2. STGCN: [Spatio-Temporal Graph Convolutional Networks](https://github.com/PKUAI26/STGCN-IJCAI-18)
3. DCRNN: [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://github.com/liyaguang/DCRNN)

## Environmental Data

1. [Targeted source detection for environment data](https://arxiv.org/pdf/1908.11056.pdf)

## Basic Methods
1. [K-SVD in Dictionary learning](https://www.cnblogs.com/endlesscoding/p/10090866.html) There are codes and some illustration. 
