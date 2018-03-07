#!/usr/bin/python

import sys
if len(sys.argv)<3:
	print("%s pcd_dir/ target"%sys.argv[0])
	sys.exit(1)

import numpy
import h5py

def readPCD(filename):
	pcd = open(filename,'r')
	for l in pcd:
		if l.startswith('DATA'):
			break
	points = []
	for l in pcd:
		ll = l.split()
		x = float(ll[0])
		y = float(ll[1])
		z = float(ll[2])
		points.append([x,y,z])
	pcd.close()
	return numpy.array(points)

def normalize(pcd):
	#normalize mean/stddev
	centroid = pcd.mean(axis=0)
	pcd -= centroid
	R = numpy.sum(pcd**2,axis=1)
	pcd /= numpy.sqrt(numpy.max(R))
	return pcd

labels = []
f = open(sys.argv[1]+'/label.txt')
for l in f:
	labels.append(int(l.split()[0])-1)
f.close()

idx = numpy.arange(len(labels))
numpy.random.shuffle(idx)

data = []
label = []
for j in range(len(labels)):
	pcd = readPCD('%s/%d-cloud.pcd'%(sys.argv[1],idx[j]))
	pcd = normalize(pcd)
	data.append(pcd)
	label.append(labels[idx[j]])

h5filename = sys.argv[2]
h5_fout = h5py.File(h5filename,'w')
h5_fout.create_dataset(
	'data', data=data,
	compression='gzip', compression_opts=4,
	dtype=numpy.float32)
h5_fout.create_dataset(
	'label', data=label,
	compression='gzip', compression_opts=1,
	dtype='uint8')
h5_fout.close()

print('Wrote %d models to %s'%(len(labels),sys.argv[2]))

