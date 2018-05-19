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
| balcony | 15    |
| beam    | 15    |
| column  | 30    |
| door    | 31    |
| fence   | 37    |
| floor   | 11    |
| roof    | 21    |
| stairs  | 15    |
| wall    | 24    |
| window  | 17    |
| *total* | *216* |

Point Cloud Deformation Code
------

	pcl_viewer door.pcd
	
![Door1](deform1.png?raw=true)

	python deform_pcd.py door.pcd out.pcd --rotate; pcl_viewer out.pcd	

![Door2](deform2.png?raw=true)

	python deform_pcd.py door.pcd out.pcd --noise; pcl_viewer out.pcd	

![Door3](deform3.png?raw=true)

	python deform_pcd.py door.pcd out.pcd --bend; pcl_viewer out.pcd	

![Door4](deform4.png?raw=true)

	python deform_pcd.py door.pcd out.pcd --truncate; pcl_viewer out.pcd	

![Door5](deform5.png?raw=true)

	python deform_pcd.py door.pcd out.pcd --rotate --noise --bend --truncate; pcl_viewer out.pcd	

![Door6](deform6.png?raw=true)

Network Code
------

Convert CAD model (PLY file) to point cloud (PCD file):

	python mesh2pcd.py door.ply door.pcd

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


