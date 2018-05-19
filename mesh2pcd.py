#!/usr/bin/python
import sys
import numpy

if len(sys.argv) < 3:
	print 'Usage: mesh2pcd.py in.ply out.pcd'
	sys.exit(1)

def triangleArea(p1,p2,p3):
	v1=p2-p1
	v2=p3-p1
	area=0.5*numpy.linalg.norm(numpy.cross(v1,v2))
	return area

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

density=0.01
numVertex=0
numFace=0
vertices=[]
faces=[]
points=[]
f=open(sys.argv[1],'r')
while True:
	l=f.readline()
	if l.startswith('element vertex'):
		numVertex=int(l.split()[2])
	elif l.startswith('element face'):
		numFace=int(l.split()[2])
	elif l.startswith('end_header'):
		break

for i in range(numVertex):
	l=f.readline()
	v=[float(j) for j in l.split()]
	vertices.append(v)

for i in range(numFace):
	ll=f.readline().split()
	p1=numpy.array(vertices[int(ll[1])])
	p2=numpy.array(vertices[int(ll[2])])
	p3=numpy.array(vertices[int(ll[3])])
	v1=p2-p1
	v2=p3-p1
	v3=v1+v2
	area=triangleArea(p1,p2,p3)
	numSamples = area/density
	r = numSamples - int(numSamples)
	numSamples = int(numSamples)
	if numpy.random.random() < r:
		numSamples += 1
	for n in range(numSamples):
		a=numpy.random.random()
		b=numpy.random.random()
		x = p1 + a*v1 + b*v2
		A1 = triangleArea(p1,p2,x)
		A2 = triangleArea(p1,p3,x)
		A3 = triangleArea(p2,p3,x)
		if abs(A1 + A2 + A3 - area) > 1e-6:
			x = p1 + v3 - a*v1 - b*v2
		points.append(x)

num_resample = 2048
resample_idx = numpy.random.choice(len(points), num_resample, replace=len(points)<num_resample)
writePCD(sys.argv[2], numpy.array(points)[resample_idx])
