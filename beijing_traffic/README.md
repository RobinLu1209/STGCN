# Beijing Traffic Data Processing

### Data info
boundary: [116.46812, 39.929101], [116.457053, 39.919362]

index | value
:-:|:-:
number of nodes | 334
number of edges | 707
average node degree | 4.233
intersection count | 307


### Road feature Extraction(2019.11.13)

Road feature matrix: [01_road_feature_matrix_full.ipynb](https://github.com/RobinLu1209/STGCN/blob/master/beijing_traffic/01_road_feature_matrix_full.ipynb)

&nbsp;|width | direction | length | speedclass | lanenum
:-:|:-:|:-:|:-:|:-:|:-:
class number | 3 | 7 | 4 | 7 | 3

Select node from Big graph: [02_select_node_from_LargeMap.ipynb](https://github.com/RobinLu1209/STGCN/blob/master/beijing_traffic/02_select_node_from_LargeMap.ipynb)

According to selected node to extract road feature information in road_feature_matrix: [03_select_road_map_feature.ipynb](https://github.com/RobinLu1209/STGCN/blob/master/beijing_traffic/03_select_road_map_feature.ipynb)


