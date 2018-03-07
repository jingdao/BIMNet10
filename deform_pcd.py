#!/usr/bin/python

import sys
import math
import numpy

if len(sys.argv) < 3:
	print('deform_pcd.py in.pcd out.pcd [--rotate --bend --truncate --noise]')
	sys.exit(1)

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
		if len(ll)>3:
			rgb = int(ll[3])
			points.append([x,y,z,rgb])
		else:
			points.append([x,y,z])
	pcd.close()
	return points

def writePCD(filename,subset):
	f=open(filename,'w')
	useColor = len(subset[0]) > 3
	if useColor:
		f.write("# .PCD v0.7 - Point Cloud Data file format\n")
		f.write("VERSION 0.7\n")
		f.write("FIELDS x y z rgb\n")
		f.write("SIZE 4 4 4 4\n")
		f.write("TYPE F F F I\n")
		f.write("COUNT 1 1 1 1\n")
		f.write("WIDTH "+str(len(subset))+"\n")
		f.write("HEIGHT 1\n")
		f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
		f.write("POINTS "+str(len(subset))+"\n")
		f.write("DATA ascii\n")
		for p in subset:
			f.write("%f %f %f %d\n" % (p[0],p[1],p[2],p[3]))
	else:
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

def bend(vertices):
	minX = min([v[0] for v in vertices])
	maxX = max([v[0] for v in vertices])
	minY = min([v[1] for v in vertices])
	maxY = max([v[1] for v in vertices])
	minZ = min([v[2] for v in vertices])
	maxZ = max([v[2] for v in vertices])
	centroid = [(minX+maxX)/2,(minY+maxY)/2,(minZ+maxZ)/2]
	dimensions = [maxX-minX,maxY-minY,maxZ-minZ]
	id_min = numpy.argmin(dimensions)
	id_mid = numpy.argsort(dimensions)[1]
	id_max = numpy.argmax(dimensions)
	F = [0,0,0]
	G = [0,0,0]
	F[id_min] = dimensions[id_mid]
	G[id_max] = 1
	new_vertices = []
#	shift = 0.125 * numpy.random.randint(-2,2)
	shift = 0
	scale = numpy.random.randint(1,3)
#	scale = numpy.random.random()*2
	displacements = []
	for v in vertices:
		vd = (numpy.array(v[:3]) - centroid) / (numpy.array(dimensions) + 1e-5) + shift
		diff = numpy.cosh(vd*scale).dot(G) * numpy.array(F)
		displacements.append(diff)
	mn = numpy.min(displacements,axis=0)
	mx = numpy.max(displacements,axis=0)
	displacements = numpy.array(displacements) - 0.5 * (mn + mx)
	new_vertices = numpy.array(vertices)
	new_vertices[:,:3] += displacements
	return new_vertices

def truncate(vertices):
	minX = min([v[0] for v in vertices])
	maxX = max([v[0] for v in vertices])
	minY = min([v[1] for v in vertices])
	maxY = max([v[1] for v in vertices])
	minZ = min([v[2] for v in vertices])
	maxZ = max([v[2] for v in vertices])
	xlim = maxX
	ylim = maxY
	zlim = maxZ
	if numpy.random.random() > 0.5:
		xlim = minX + (maxX-minX) * (0.75 + numpy.random.random()/4)
	if numpy.random.random() > 0.5:
		ylim = minY + (maxY-minY) * (0.75 + numpy.random.random()/4)
	if numpy.random.random() > 0.5:
		zlim = minZ + (maxZ-minZ) * (0.75 + numpy.random.random()/4)
	return [v for v in vertices if v[0]<=xlim and v[1]<=ylim and v[2]<=zlim]

def noise(vertices,num_spikes):
	minX = min([v[0] for v in vertices])
	maxX = max([v[0] for v in vertices])
	minY = min([v[1] for v in vertices])
	maxY = max([v[1] for v in vertices])
	minZ = min([v[2] for v in vertices])
	maxZ = max([v[2] for v in vertices])
	midX = 0.5 * (minX + maxX)
	midY = 0.5 * (minY + maxY)
	midZ = 0.5 * (minZ + maxZ)
	R = max(maxX-minX,maxY-minY,maxZ-minZ)
	new_vertices = list(vertices)
	for i in range(num_spikes):
		x = midX + (numpy.random.random()-0.5) * R
		y = midY + (numpy.random.random()-0.5) * R
		z = midZ + (numpy.random.random()-0.5) * R
		w = [x,y,z]
		w.extend(vertices[0][3:])
		new_vertices.append(w)
	return new_vertices

def rotate(vertices):
	pcd = numpy.array(vertices)
	centroid = pcd.mean(axis=0)
	pcd -= centroid
	theta = numpy.random.random() * 2 * numpy.pi
	ct = numpy.cos(theta)
	st = numpy.sin(theta)
	R = numpy.array([[ct,-st],[st,ct]])
	pcd[:,:2] = pcd[:,:2].dot(R)
	pcd += centroid
	return pcd

def fillDuplicate(vertices,num_points):
	new_vertices = numpy.array(vertices)
	samples = numpy.random.randint(len(vertices),size=num_points)
	return new_vertices[samples]

vertices = readPCD(sys.argv[1])
num_points = len(vertices)
if '--rotate' in sys.argv:
	vertices = rotate(vertices)
if '--bend' in sys.argv:
	vertices = bend(vertices)
if '--truncate' in sys.argv:
	vertices = truncate(vertices)
if '--noise' in sys.argv:
	vertices = noise(vertices,int(0.1 * num_points))
vertices = fillDuplicate(vertices,num_points)
writePCD(sys.argv[2],vertices)
