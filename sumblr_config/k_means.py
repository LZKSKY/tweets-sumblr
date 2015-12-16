from tweet import *
from canopy import *

import numpy as np
import matplotlib
matplotlib.use('Agg')
from scipy.cluster import vq
import pylab
pylab.close()


def convert_to_array(obj):
	x = []
	y = []
	for i in range(len(obj)):
		x.append(obj[i].tf)
		y.append(obj[i].idf)
	x = np.asarray(x)
	y = np.asarray(y)
	x = x.reshape(len(x), 1)
	y = y.reshape(len(y), 1)
	z = np.hstack((x, y))
	return z

def k_means_init(data, centroids):
	# print data
	# print centroids
	center, dist = vq.kmeans(data, centroids)
	print center
	code, distance = vq.vq(data, center)
	print distance
	print code

def init(M):
	Centroids = canopy(M,1,10)
	arr_M = convert_to_array(M)
	arr_cent = convert_to_array(Centroids)
	code = k_means_init(arr_M, arr_cent)
	# print code

l = []
t1 = tweet("",1,0)
l.append(t1)
t2 = tweet("",2,0)
l.append(t2)
t3 = tweet("",5,0)
l.append(t3)
t4 = tweet("",6,0)
l.append(t4)
t5 = tweet("",15,0)
l.append(t5)
t6 = tweet("",17,0)
l.append(t6)

init(l)


