# Data Analysis on PeMSD7 Dataset

## Dataset Description

Dataset | Node num | Time | Duration | Time slot | Scene
:-:|:-:|:-:|:-:|:-:|:-:
PeMSD7 | 288 | Weekdays of May and June of 2012 |44 days| 5mins | Sensor data to detect the car speed

## Data processing

1. Use PeMSD7_roadmap.ipynb to generate adj_mx_pemsd7.pkl
2. Use PeMSD7_traindata.ipynb to generate train/validation/test data
3. Create tensorflow-gpu environment for training.
```
source activate tensorflow-gpu
```
4. Run the training, but meet with some bug about data type.
```
python dcrnn_train.py --config_filename=data/model/dcrnn_pemsd7.yaml
```
