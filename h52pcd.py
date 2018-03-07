#!/usr/bin/python

import sys
if len(sys.argv)<3:
	print("%s in.h5 pcd_dir/"%sys.argv[0])
	sys.exit(1)

import numpy
import h5py

def writePCD(filename,subset):
	f=open(filename,'w')
	f.write("# .PCD v0.7 - Point Cloud Data file format\n")
	f.write("VERSION 0.7\n")
	f.write("FIELDS x y z\n")
	f.write("SIZE 4 4 4\n")
	f.write("TYPE F F F\n")
	f.write("COUNT 1 1 1\n")
	f.write("WIDTH "+str(len(subset))+"\n")
	f.write("HEIGHT 1\n")
	f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
	f.write("POINTS "+str(len(subset))+"\n")
	f.write("DATA ascii\n")
	for p in subset:
		f.write("%f %f %f\n" % (p[0],p[1],p[2]))
	f.close()
	print 'Wrote '+str(len(subset))+' points to '+filename

if sys.argv[1].endswith('.h5'):
	f = h5py.File(sys.argv[1])
	data = f['data'][:]
	label = numpy.squeeze(f['label'][:])
	f.close()
elif sys.argv[1].endswith('.npz'):
	f = numpy.load(sys.argv[1])
	data = f['features']
	label = f['targets']
else:
	sys.exit(1)

numOutputs = len(data)
#classes=["airplane ","bathtub ","bed ","bench ","bookshelf ","bottle ","bowl ","car ","chair ","cone ","cup ","curtain ","desk ","door ","dresser ","flower_pot ","glass_box ","guitar ","keyboard ","lamp ","laptop ","mantel ","monitor ","night_stand ","person ","piano ","plant ","radio ","range_hood ","sink ","sofa ","stairs ","stool ","table ","tent ","toilet ","tv_stand ","vase ","wardrobe ","xbox"]
classes=["balcony ","beam ","column ","door ","fence ","floor ","roof ","stairs ","wall ","window"]
labelfile = open('%s/label.txt'%(sys.argv[2]),'w')
for i in range(numOutputs):
	if len(data.shape) > 3:
		voxels = data[i][0]
		pcd = zip(*numpy.nonzero(voxels))
	else:
		pcd = data[i]
	writePCD('%s/%d-cloud.pcd'%(sys.argv[2],i),pcd)
	labelfile.write('%d %s\n'%(label[i]+1,classes[label[i]]))
labelfile.close()
