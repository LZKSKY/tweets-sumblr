#TODO
#timestamp
from __future__ import print_function
from xlrd import open_workbook
from tweet import *
from nltk.corpus import stopwords
from canopy import *
from cluster import *
from config import *
from tf_idf import *
import xlwt
import math
import operator
from ClusterSorter import *
import numpy as np
import itertools
import matplotlib
import time
matplotlib.use('Agg')
from scipy.cluster import vq
import pylab
from sets import Set
pylab.close()

#cachedStopWords = stopwords.words("english")

def norm(v1):
	norm = 0
	for i in range(len(v1)):
		norm = norm + v1[i] * v1[i]
	norm = math.sqrt(norm)
	return norm

def dotproduct(v1, v2):
	return sum(map(lambda x: x[0] * x[1], itertools.izip(v1, v2)))

def calc_sim(v1, v2):
	prod = dotproduct(v1, v2)
	len1 = math.sqrt(dotproduct(v1, v1))
	len2 = math.sqrt(dotproduct(v2, v2))
	try:
		return prod / (len1 * len2)
	except:
		return 0

def convert_to_array(obj):
	z = obj[0].tv
	z = np.asarray(z)
	for i in range(1,len(obj)):
		x = obj[i].tv
		x = np.asarray(x)
		z = np.vstack((z, x))
	return z

def power_method(m, epsilon):
    n = len( m )
    p = [1.0 / n] * n
    while True:
        new_p = [0] * n
        for i in xrange( n ):
            for j in xrange( n ):
                new_p[i] += m[j][i] * p[j]
        total = 0
        for x in xrange( n ):
            total += ( new_p[i] - p[i] ) ** 2
        p = new_p
        if total < epsilon:
            break
    return p

def rank_documents(doc_list, threshold=0.1, tolerance=0.00001):
    n = len(doc_list)
    #Initialises the adjacency matrix
    adjacency_matrix = np.zeros([n, n])

    degree = np.zeros([n])
    scores = np.zeros([n])

    for i, documenti in enumerate(doc_list):
        for j, documentj in enumerate(doc_list):
            adjacency_matrix[i][j] = calc_sim(documenti.tv, documentj.tv)

            if adjacency_matrix[i][j] > threshold:
                adjacency_matrix[i][j] = 1.0
                degree[i] += 1
            else:
                adjacency_matrix[i][j] = 0

    for i in xrange(n):
        for j in xrange(n):
            if degree[i] == 0: degree[i] = 1.0 #at least similar to itself
            adjacency_matrix[i][j] = adjacency_matrix[i][j] / degree[i]

    scores = power_method(adjacency_matrix, tolerance)

    for i in xrange( 0, n ):
        doc_list[i].dist = scores[i]

    sorted_documents = sorted(doc_list, key=lambda document: document.cluster)
    return sorted_documents

def getnMax():
	m = len(clusters[0].tweets)
	for i in range(1, len(clusters)):
		if len(clusters[i].tweets) > m:
			m = len(clusters[i].tweets)
	return m

def getVI(tweet):
	nMax = getnMax()
	nti = len(clusters[tweet.cluster].tweets)
	lrti = tweet.dist
	sim = 0
	for i in range(len(summarySet)):
		sim = sim + calc_sim(tweet.tv,summarySet[i].tv)
	if len(summarySet) == 0:
		vi = lamda * nti * lrti / nMax
	else:
		vi = lamda * nti * lrti / nMax - (1 - lamda) * sim / len(summarySet)
	return vi


def GetSummary(sorted_documents):
	del summarySet[:]
	i = 0
	pos = 0
	flag = 0
	finalList = []
	while True:
		if flag == 1:
			break
		if i == len(sorted_documents):
			finalList.append(maxT)
			sorted_documents[pos].flag = i
			break
		maxT = sorted_documents[i]
		pos = i
		while sorted_documents[i].cluster == maxT.cluster:
			if sorted_documents[i].dist > maxT.dist:
				maxT = sorted_documents[i]
				pos = i
			i = i + 1
			if i == len(sorted_documents):
				finalList.append(maxT)
				sorted_documents[pos].flag = 1
				flag = 1
				break

		if flag == 0:
			finalList.append(maxT)
			sorted_documents[pos].flag = 1
	# print finalList
	while len(summarySet) < L and len(finalList) > 0:
		m = getVI(finalList[0])
		pos = 0
		for i in range(1,len(finalList)):
			cur = getVI(finalList[i])
			if cur > m:
				m = cur
				pos = i

		summarySet.append(finalList[pos])
		del finalList[pos]

	while len(summarySet) < L and len(sorted_documents) > len(clusters):
		i = 0
		while sorted_documents[i].flag != 0:
			i = i + 1
		pos = i
		m = getVI(sorted_documents[i])
		i = i + 1
		while i < len(sorted_documents):
			if sorted_documents[i].flag == 0:
				cur = getVI(sorted_documents[i])
				if cur > m:
					m = cur
					pos = i
			i = i + 1
		summarySet.append(sorted_documents[pos])
		del sorted_documents[pos]
	# print summarySet

def k_means_init(data, centroids):
	# print data
	# print centroids
	center, dist = vq.kmeans(data, centroids)

	code, distance = vq.vq(data, center)
	return code, center

def modifyMap(extra):
	n = len(wordDict)
	words = extra.keys()
	# print words
	for i in range(len(words)):
		if words[i] not in wordDict:
			wordDict[words[i]] = n
			n = n + 1
			idf_score.append(1)

def updateCluster(i):
	clusters[i].tcv.sum_v = [0] * Tweet.dim
	clusters[i].tcv.wsum_v = [0] * Tweet.dim
	for j in range(len(clusters[i].tweets)):
		norm_fac = norm(clusters[i].tweets[j].tv)
		clusters[i].tweets[j].normtv = norm_fac
		if norm_fac == 0:
                	clusters[i].tweets[j].normtv = 1
                	norm_fac = 1
		newList1 = [x / norm_fac for x in clusters[i].tweets[j].tv]
		newList2 = [x * clusters[i].tweets[j].w for x in clusters[i].tweets[j].tv]
		clusters[i].tcv.sum_v = [x + y for x, y in zip(clusters[i].tcv.sum_v, newList1)]
		clusters[i].tcv.wsum_v = [x + y for x, y in zip(clusters[i].tcv.wsum_v, newList2)]
	clusters[i].centroid = [x / clusters[i].tcv.n for x in clusters[i].tcv.wsum_v]

def updateTweets(lenExtra):
	for i in range(len(clusters)):
		for j in range(len(clusters[i].tweets)):
			for k in range(len(clusters[i].tweets[j].tv)):
				f = return_idf(idf_score[k] - flag[k], len(M))
				if f == 0:
					f = 1
				clusters[i].tweets[j].tv[k] = clusters[i].tweets[j].tv[k] * return_idf(idf_score[k], len(M) + 1) / f
			for k in range(lenExtra):
				clusters[i].tweets[j].tv.append(0)
		for j in range(lenExtra):
			clusters[i].tcv.sum_v.append(0)
			clusters[i].tcv.wsum_v.append(0)
			clusters[i].centroid.append(0)
		updateCluster(i)

def getScore(text, word, index, flag):
	tf = getTF(word, text)
	if flag == 1:
		#word has occurred before
		idf = return_idf(idf_score[index] + 1, len(M))
	else:
		#word occurring for the first time
		idf = return_idf(1, len(M) + 1)
	return tf * idf

def wordFromExistingCorpus(t):
	words = t.split()
	v = [0] * Tweet.dim
	global flag
	flag = [0] * Tweet.dim
	extra = {}
	for i in range(len(words)):
		if words[i] in wordDict:
			index = wordDict[words[i]]
			v[index] = getScore(t, words[i], index, 1)
			if flag[index] == 0:
				flag[index] = 1
				idf_score[index] = idf_score[index] + 1
		else:
			if words[i] not in extra:
				extra[words[i]] = getScore(t, words[i], 0, 0)
				if extra[words[i]] == 0:
					extra[words[i]] = 1
	return v, extra

def formatTweet(t, v, extra):
	nt = Tweet(t, [], 1, 0)
	nt.tv = v
	for key, val in extra.items():
		nt.tv.append(val)
	return nt


def get_ftset(index, tw):
	MinSim = 2
	pos = -1
	for i in range(len(clusters[index].tcv.ft_set)):
		sim = calc_sim(clusters[index].tcv.ft_set[i].tv,clusters[index].centroid)
		if sim < MinSim:
			MinSim = sim
			pos = i
	newSim = calc_sim(tw.tv, clusters[index].centroid)
	if newSim > MinSim:
		clusters[index].tcv.ft_set[pos] = tw

def summarize():
	twL = []
	print ("generating lexrank graph...")
	for i in range(len(clusters)):
		# print i
		for j in range(len(clusters[i].tweets)):
			twL.append(clusters[i].tweets[j])
	sorted_docs = rank_documents(twL)
	print ("computing highest priority tweets...")
	GetSummary(sorted_docs)
	# print summarySet

def addTweet(t):
	v, extra = wordFromExistingCorpus(t)
	MaxSim = calc_sim(v, clusters[0].centroid)
	allotment = 0
        i=len(clusters)
	for i in range(1,len(clusters)):
		sim = calc_sim(v, clusters[i].centroid)
		if(sim > MaxSim):
			allotment = i
			MaxSim = sim
	vec_wsum = clusters[i].tcv.wsum_v
	vec_sum = clusters[i].tcv.sum_v
	# print vec_wsum
	# print vec_sum
        clst_tcv_n = clusters[i].tcv.n
        nrm_cluster =  norm(clusters[i].tcv.wsum_v)
        if clst_tcv_n ==0:
            clst_tcv_n = 1
        if nrm_cluster ==0:
            nrm_cluster = 1

	MBS = beta * (dotproduct(vec_sum,vec_wsum) / (clst_tcv_n * nrm_cluster))
	#modify wordmap and textual vectors
	modifyMap(extra)
	Tweet.dim = Tweet.dim + len(extra)
	updateTweets(len(extra))
	nt = formatTweet(t, v, extra)
	M.append(nt)
	if MaxSim < MBS:
		#create new cluster
		norm_fac = norm(nt.tv)
		nt.normtv = norm_fac
		nt.cluster = len(clusters)
                if norm_fac == 0:
                    norm_fac = 1
		newList1 = [x / norm_fac for x in nt.tv]
		newList2 = [x * nt.w for x in nt.tv]
		ntcv = TCV(newList1,newList2,0,0,1,[nt])
		nList = []
		nList.append(nt)
		ncluster = Cluster(ntcv,[nt],newList2)
		clusters.append(ncluster)

	else:
		#add it to existing
		norm_fac = norm(nt.tv)
		nt.cluster = allotment
		nt.normtv = norm_fac
		clusters[allotment].tweets.append(nt)
                if norm_fac ==0:
                    norm_fac = 1
		newList1 = [x / norm_fac for x in nt.tv]
		newList2 = [x * nt.w for x in nt.tv]
		clusters[allotment].tcv.sum_v = [x + y for x, y in zip(clusters[i].tcv.sum_v, newList1)]
		clusters[allotment].tcv.wsum_v = [x + y for x, y in zip(clusters[i].tcv.wsum_v, newList2)]
		clusters[allotment].tcv.n = clusters[allotment].tcv.n + 1
		clusters[allotment].centroid = [x / clusters[allotment].tcv.n for x in clusters[allotment].tcv.wsum_v]
		if len(clusters[allotment].tcv.ft_set) >= size_ftset:
			get_ftset(allotment,nt)
		else:
			clusters[allotment].tcv.ft_set.append(nt)
		#update timestamps

def update(lenExtra):
	for i in range(len(M)):
		for k in range(len(M[i].tv)):
			f = return_idf(idf_score[k] - flag[k], len(M))
			if f == 0:
				f = f + 1
			M[i].tv[k] = M[i].tv[k] * return_idf(idf_score[k], len(M) + 1) / f
		for k in range(lenExtra):
			M[i].tv.append(0)
		# print idf_score


def setupTweets():
	for i in range(len(tweetList)):
		v, extra = wordFromExistingCorpus(tweetList[i])
		modifyMap(extra)
		# print idf_score
		Tweet.dim = Tweet.dim + len(extra)
		update(len(extra))
		nt = formatTweet(tweetList[i], v, extra)
		M.append(nt)


def init():
	global clusters
	Tweet.dim = 0
	setupTweets()
	print ("computing initial centroids...")
	Centroids = canopy(M,T1,T2)
	# print len(Centroids)
	arr_M = convert_to_array(M)
	# print arr_M.shape
	arr_cent = convert_to_array(Centroids)
	# print arr_cent.shape
	global code
	print ("performing k-means...")
	code, center = k_means_init(arr_M, arr_cent)
	clusters = [[None]] * len(center)
	for i in range(len(clusters)):
		clusters[i] = Cluster(TCV(0,0,0,0,0,[]),[],center[i].tolist())
	print ("computing ft-set...")
	for i in range(len(M)):
		M[i].cluster = code[i]
		clusters[code[i]].tweets.append(M[i])
		clusters[code[i]].tcv.sum_v = [0] * Tweet.dim
		clusters[code[i]].tcv.wsum_v = [0] * Tweet.dim
		clusters[code[i]].tcv.n = len(clusters[code[i]].tweets)
		if len(clusters[code[i]].tcv.ft_set) >= size_ftset:
			get_ftset(code[i],M[i])
		else:
			clusters[code[i]].tcv.ft_set.append(M[i])
	print ("computing tweet cluster vectors...")
	for i in range(len(clusters)):
		for j in range(len(clusters[i].tweets)):
			#print(clusters[i])
			norm_fac = norm(clusters[i].tweets[j].tv)
			clusters[i].tweets[j].normtv = norm_fac
			if norm_fac == 0:
                        	clusters[i].tweets[j].normtv = 1
                        	norm_fac = 1
			newList1 = [x / norm_fac for x in clusters[i].tweets[j].tv]
			newList2 = [x * clusters[i].tweets[j].w for x in clusters[i].tweets[j].tv]
			clusters[i].tcv.sum_v = [x + y for x, y in zip(clusters[i].tcv.sum_v, newList1)]
			clusters[i].tcv.wsum_v = [x + y for x, y in zip(clusters[i].tcv.wsum_v, newList2)]


def merge():
	arr = []
	for i in range(len(clusters)):
		for j in range(len(clusters)):
			if i != j:
				sim = calc_sim(clusters[i].centroid,clusters[j].centroid)
				arr.append(ClusterSorter(i,j,sim))
	arr = sorted(arr, key=operator.attrgetter('dist'), reverse = True)

global M
global wordDict
global idf_score
global summarySet
global tweetList
summarySet = []
idf_score = []
tweetList = []
wordDict = {}
M = []
# t1 = Tweet("",[1, 0, 1, 2, 1], 1, 0)
# M.append(t1)
# t2 = Tweet("",[2, 0, 1,2, 1], 1, 0)
# M.append(t2)
# t3 = Tweet("",[5, 0, 1,2, 1], 1, 0)
# M.append(t3)
# t4 = Tweet("",[6, 0, 1,2, 1], 1, 0)
# M.append(t4)
# t5 = Tweet("",[12, 0, 1,2, 1], 1, 0)
# M.append(t5)
# t6 = Tweet("",[13, 0, 1,2, 1], 1, 0)
# M.append(t6)
# t7 = Tweet("",[20, 0, 1,2, 1], 1, 0)
# M.append(t7)
# t8 = Tweet("",[30, 0], 1, 0)

# t0 = "Football is popular in all parts of the world"
# tweetList.append(t0)
# t1 = "Football is a very popular sport"
# tweetList.append(t1)
# t2 = "Israel Gaza air strikes"
# tweetList.append(t2)
# t3 = "Israel fire another air strike on Gaza"
# # tweetList.append(t3)
# init()
# addTweet(t3)
# print clusters[3].tcv

def printSummaryAlt():
	myFileOut = open('hagupit_output.txt','w')
	for m in range(len(summarySet)):
		try:
			myFileOut.write(summarySet[m].text + "\n")
		except:
			pass
	myFileOut.close()


def printSummary():
	sh = book.add_sheet("Summary")
	for m in range(len(summarySet)):
		sh.write(m, 0, summarySet[m].text)

def printWords():
	print ("saving one-grams to file...")
	sh = book.add_sheet("One-Grams")
	m = 0
	for key, value in wordDict.items():
		sh.write(m, 0, key)
		m = m + 1

def extractDataAlt():
	myFile = open('harda_fact.txt','r')
	r = []
	r1 = []
	for row in range(0,initialBound):
		s = myFile.readline()
		s = ' '.join([word for word in s.split()])
		r.append(s)
		tweetList.append(r[len(r)-1])
	for row in range(initialBound,finalBound):
		s = myFile.readline()
		try:
			s = ' '.join([word for word in s.split()])
		except:
			pass
		r1.append(s)
	myFile.close()
	t0 = time.clock()
	print ("forming initial clusters...")
	init()
	print ("performing incremental clustering...")
	print ("adding tweets")
	for i in range(len(r1)):
		if len(clusters) > NMax:
			merge()
		# if i % 10 == 0:
			# print (i)
			# print (len(clusters))
		addTweet(r1[i])
	#t1 = time.clock()
	# print ("time elapsed = " + str(t1 - t0))
	print ("")
	print ("summarizing...")
	summarize()
	t1 = time.clock()
	print ("saving to file...")
	printSummaryAlt()
	print ("time elapsed = " + str(t1 - t0))


def extractData():
	wb = open_workbook("processed_uttarakhand_flood_tweets.xlsx")
	r = []
	r1 = []
	s = wb.sheets()[0]
	for row in range(0,initialBound):
		r.append(s.cell(row,0).value)
		tweetList.append(r[len(r)-1])
	for row in range(initialBound,finalBound):
		r1.append(s.cell(row,0).value)
	print ("forming initial clusters...")
	init()
	print ("performing incremental clustering...")
	print ("adding tweets", end="")
	for i in range(len(r1)):
		if i % 10 == 0:
			print('.', end="")
		addTweet(r1[i])
	print ("")
	print ("summarizing...")
	summarize()
	print ("saving to file...")
	printSummary()

def compareData():
	print ("Computing stats...")
	autoGen = set()
	manualGen = set()
	wb = open_workbook("processed_uttarakhand_flood_tweets.xlsx")
	s = wb.sheets()[1]
	for row in range(0, L):
		txt = s.cell(row,0).value
		words = txt.split()
		manualGen |= set(words)


	wb = open_workbook("result_uttarakhand_flood_tweets.xlsx")
	s = wb.sheets()[0]
	for row in range(0, L):
		txt = s.cell(row,0).value
		words = txt.split()
		autoGen |= set(words)

	pr1 = 0
	tot1 = 0
	for w in autoGen:
		if w in manualGen:
			pr1 = pr1 + 1
		tot1 = tot1 + 1


	pr2 = 0
	tot2 = 0
	for w in manualGen:
		if w in autoGen:
			pr2 = pr2 + 1
		tot2 = tot2 + 1

	res1 = float(pr1) / tot1
	res2 = float(pr2) / tot2

	print (res1)
	print (res2)

# book = xlwt.Workbook()
extractDataAlt()
# printWords()

# book.save("result_uttarakhand_flood_tweets.xlsx")
# compareData()


