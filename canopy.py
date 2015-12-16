from tweet import *
import random
import math


def dist(t1, t2):
	# return math.sqrt(((t1.tf - t2.tf)*(t1.tf - t2.tf) + (t1.idf - t2.idf)*(t1.idf - t2.idf))) 
	sum = 0
	for i in range(len(t1.tv)):
		sum = sum + (t1.tv[i] - t2.tv[i]) * (t1.tv[i] - t2.tv[i])
	return math.sqrt(sum) 

def canopy(M,T1,T2):
	size = len(M)
	M_copy = M
	Centroids = []
	while len(M_copy) > 0:
		idx = random.randint(0, len(M_copy)-1)
		nbr = []
		ext = []
		for i in range(len(M_copy)):
			if i != idx:
				d = dist(M_copy[i], M_copy[idx])
				if d < T1:
					nbr.append(i)
				if d > T2:
					ext.append(i)
		temp = []
		for i in range(len(M_copy)):
			if i not in nbr and i not in ext and i != idx:
				temp.append(M_copy[i])
		Centroids.append(M_copy[idx])
		M_copy = temp
	return Centroids

# l = []
# t1 = tweet("",1,0)
# l.append(t1)
# t2 = tweet("",2,0)
# l.append(t2)
# t3 = tweet("",5,0)
# l.append(t3)
# t4 = tweet("",6,0)
# l.append(t4)
# canopy(l,2,10)

