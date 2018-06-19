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

def readPLY(filename):
	vertices = []
	faces = []
	numV = 0
	numF = 0
	f = open(filename,'r')
	while True:
		l = f.readline()
		if l.startswith('element vertex'):
			numV = int(l.split()[2])
		elif l.startswith('element face'):
			numF = int(l.split()[2])
		elif l.startswith('end_header'):
			break
	for i in range(numV):
		l = f.readline()
		vertices.append([float(j) for j in l.split()])
	for i in range(numF):
		l = f.readline()
		faces.append([int(j) for j in l.split()[1:4]])
	f.close()
	return vertices,faces

def readOFF(filename):
	vertices = []
	faces = []
	numV = 0
	numF = 0
	f = open(filename,'r')
	l = f.readline()
	if len(l) > 4: #merged header line
		l = l[3:]
	else:
		l = f.readline()
	numV = int(l.split()[0])
	numF = int(l.split()[1])
	for i in range(numV):
		l = f.readline()
		vertices.append([float(j) for j in l.split()])
	for i in range(numF):
		l = f.readline()
		faces.append([int(j) for j in l.split()[1:4]])
	f.close()
	return vertices,faces

numVertex=0
numFace=0
vertices=[]
faces=[]

for i in range(numVertex):
	l=f.readline()
	v=[float(j) for j in l.split()]
	vertices.append(v)

if 'ply' in sys.argv[1]:
	vertices, faces = readPLY(sys.argv[1])
elif 'off' in sys.argv[1]:
	vertices, faces = readOFF(sys.argv[1])
else:
	sys.exit(1)

xmin=min(v[0] for v in vertices)
xmax=max(v[0] for v in vertices)
ymin=min(v[1] for v in vertices)
ymax=max(v[1] for v in vertices)
zmin=min(v[2] for v in vertices)
zmax=max(v[2] for v in vertices)
#density=0.5
density = 0.01 * max(xmax-xmin,ymax-ymin,zmax-zmin)
maxPoints = 2048

while True:
	points=[]
	for f in faces:
		p1=numpy.array(vertices[f[0]])
		p2=numpy.array(vertices[f[1]])
		p3=numpy.array(vertices[f[2]])
		v1=p2-p1
		v2=p3-p1
		v3=v1+v2
		area=triangleArea(p1,p2,p3)
		numSamples = numpy.sqrt(area)/density
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

	points = numpy.array(points)
	if len(points) >= maxPoints:
		idx = numpy.arange(len(points))
		numpy.random.shuffle(idx)
		idx = idx[:maxPoints]
		points = points[idx,:]
		writePCD(sys.argv[2],points)
		break
	else:
		density /= 2

