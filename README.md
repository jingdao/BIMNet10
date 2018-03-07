BIMNet10
========

![Dataset](dataset.png?raw=true)

The H5 files consists of building element point clouds in 10 categories.
The "data" MxNx3 matrix represents M objects, each with N points, each with x,y,z coordinates.
The "label" M - length array represents an integer class label from 0 to 9.

	bimnet10_train.h5:
	<HDF5 dataset "data": shape (153, 2048, 3), type "<f4">
	<HDF5 dataset "label": shape (153,), type "|u1">
	bimnet10_test.h5 
	<HDF5 dataset "data": shape (47, 2048, 3), type "<f4">
	<HDF5 dataset "label": shape (47,), type "|u1">
	bimnet10_test_deform.h5 
	<HDF5 dataset "data": shape (564, 2048, 3), type "<f4">
	<HDF5 dataset "label": shape (564,), type "|u1">
	

| Object  | Count |
| ------  | ----- |
| balcony | 9     |
| beam    | 5     |
| column  | 30    |
| door    | 31    |
| fence   | 37    |
| floor   | 11    |
| roof    | 21    |
| stairs  | 15    |
| wall    | 24    |
| window  | 17    |
| *total* | *200* |

Code
----

Convert point clouds from H5 file to folder of PCD files:

	python h52pcd.py bimnet10_train.h5 pcd/

Convert folder of point clouds in PCD format to H5 file:

	python pcd2h5.py pcd/ bimnet10_train.h5

Visualize point cloud data (requires [Point Cloud Library](http://pointclouds.org/))

	pcl_viewer pcd/0-cloud.pcd

Train / Test Network:

	python bimnet.py

Citation
--------

	@inproceedings{ChenCho2018,
   		author = "Jingdao Chen and Yong K Cho and Jun Ueda",
   		booktitle = {Proceedings of the 2018 IEEE Conference on Robotics and Automation (ICRA)},
   		title = "Sampled-Point Network for Classification of Deformed Building Element Point Clouds",
   		year = {2018}
	}


