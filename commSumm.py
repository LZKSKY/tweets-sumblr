from myGraphNode import *
import networkx as nx
import re
from nltk.corpus import wordnet as wn
from config import *
import time
import operator
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

global G, summarySet
G = nx.Graph()
summarySet = []
cumSummarySet = []

cachedStopWords = stopwords.words("english")

freqSet = {}

lmtzr = WordNetLemmatizer()

wordsCovered = set()

def numToWord(number):
    word = []
    if number < 0 or number > 999999:
        return number
        # raise ValueError("You must type a number between 0 and 999999")
    ones = ["","one","two","three","four","five",
            "six","seven","eight","nine","ten","eleven","twelve",
            "thirteen","fourteen","fifteen","sixteen","seventeen",
            "eighteen","nineteen"]
    if number == 0: return "zero"
    if number > 9 and number < 20:
        return ones[number]
    tens = ["","ten","twenty","thirty","forty","fifty",
            "sixty","seventy","eighty","ninety"]
    word.append(ones[int(str(number)[-1])])
    if number >= 10:
        word.append(tens[int(str(number)[-2])])
    if number >= 100:
        word.append("hundred")
        word.append(ones[int(str(number)[-3])])
    if number >= 1000 and number < 1000000:
        word.append("thousand")
        word.append(numToWord(int(str(number)[:-3])))
    for i,value in enumerate(word):
        if value == '':
            word.pop(i)
    return ' '.join(word[::-1])



def parseFreqSet(ipF):
	f6 = open(ipF,'r')
	s = f6.readline()
	while s != "":
		tokens = s.split(' ')
		freq = int(tokens[1][:-2])
		freqSet[tokens[0]] = tokens[1]
		s = f6.readline()
	print "Frequency set parsed"

def addToDict(word):
	wordsCovered.add(word)

def getData(word):
	synsetlist = []
	try:
		syns = wn.synsets(word)
		for ss in syns:
			lemmaN = ss.lemma_names()
			for lss in lemmaN:
				synsetlist.append(lss)
	except:
		print "error"
	synsetlist.append(word)
	return synsetlist
parseFreqSet('en.txt')
f1 = open('hydb_english_tag.txt','r')
f2 = open('english_tweet.txt','r')
f3 = open('english_label.txt','r')
f4 = open('hydb_predict_class.txt','r')
f5 = open('graph_english_tweet_output.txt','w')
f6 = open('eval_file.txt','w')
t0 = time.clock()
cnt = 0
cnt1 = 0
counter = 0
timeMap = {}
tweetMap = {}
k = 0

def findSim(t1, t2):
	words1 = set(t1.split(" "))
	words2 = set(t2.split(" "))
	common = len(words1 & words2)
	wt =  float(common) / min(len(words2), len(words1))
	return wt
f7 = open('CDetect/targetwords'+str(0)+'.txt','w')
f7.close()

for i in range(1,983):
	#generate summary if new count has exceeded
	# tempflag = 0
	counter = counter + 1
	# if cnt1 == 1:
		# print counter
	if cnt >= maxYnodes:
		#run infomap
		counter = 1
		CG = nx.Graph()
		X_nodes = set(n for n,d in G.nodes(data=True) if d['bipartite']==0)
		Y_nodes = set(n for n,d in G.nodes(data=True) if d['bipartite']==1)
		for y in Y_nodes:
			wordsCovered.add(y)
		for x in X_nodes:
			CG.add_node(x)
			x_nbr = G.neighbors(x)
			for y in x_nbr:
				y_nbr = G.neighbors(y)
				for x1 in y_nbr:
					if timeMap[x] < timeMap[x1]:
						CG.add_node(x1)
						CG.add_edge(x, x1, weight = findSim(x, x1))
		f7 = open('CDetect/commdetect' + str(cnt1) + '.txt','w')
		for edge in CG.edges():
			v1 = edge[0]
			v2 = edge[1]
			wt = CG[v1][v2]['weight']
			v1 = timeMap[v1]
			v2 = timeMap[v2]
			f7.write(str(v1) + " " + str(v2) + " " + str(wt))
			f7.write("\n")
		f7.close()
		cnt1 = cnt1 + 1
		CG.clear()
		G.clear()
		timeMap = {}
		tweetMap = {}
		cnt = 0
		f7 = open('CDetect/targetwords'+str(cnt1)+'.txt','w')
		f7.close()

		
	nt = f2.readline()
	nt = ' '.join([word for word in nt.split() if word not in cachedStopWords])
	if i < 501:
		flag = f3.readline()
		flag = flag[:-1]
		# print flag
	else:
		flag = f4.readline()
		flag = (flag.split('\t'))[2]
	if flag == '1': #if fact
		G.add_node(nt, bipartite = 0)
		timeMap[nt] = counter
		tweetMap[counter] = nt
		s = f1.readline()
		tokens = s.split('\t')
		while len(tokens) > 1:
			if tokens[1] == '^' or tokens[1] == '$' or tokens[1] == 'N':
				if tokens[1] == '$':
					tokens[0] = numToWord(tokens[0])
					# print tokens[0]
				# if tokens[1] == 'U':
				# 	try:
				# 		resp = urllib2.urlopen(tokens[0])
				# 		if resp.getcode() == 200:
				# 			tokens[0] = resp.url
				# 	except:
				# 		pass
				if tokens[0] not in wordsCovered:
					try:
						tokens[0] = lmtzr.lemmatize(tokens[0])
					except:
						print 'error in lemmatize'
					if tokens[1] != 'N' and tokens[0] not in wordsCovered:
						f7 = open('CDetect/targetwords'+str(cnt1)+'.txt','a')
						f7.write(tokens[0] + '\n')
						f7.close()
					# if cnt1 == 1:
					# 	print wordsCovered
					# synsetlist = getData(tokens[0])
					# for word in synsetlist:
					G.add_node(tokens[0], bipartite = 1)
					G.add_edge(nt, tokens[0], weight = 1)
					cnt = cnt + 1
			s = f1.readline()
			tokens = s.split('\t')

	else: #if opinion
		s = f1.readline()
		tokens = s.split('\t')
		while len(tokens) > 1:
			s = f1.readline()
			tokens = s.split('\t')

t1 = time.clock()
print "time taken = " + str(t1 - t0)
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()