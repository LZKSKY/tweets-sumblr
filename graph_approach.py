from myGraphNode import *
import networkx as nx
import re
from nltk.corpus import wordnet as wn
from config import *
import time
import operator
from nltk.corpus import stopwords

global G, summarySet
G = nx.Graph()
summarySet = []
cumSummarySet = []

cachedStopWords = stopwords.words("english")

freqSet = {}

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

def addToDict(newWords):
	for word in newWords:
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
		pass
	synsetlist.append(word)
	return synsetlist
parseFreqSet('en.txt')
f1 = open('hydb_english_tag.txt','r')
f22 = open('english_tweet.txt','r')
f2 = open('graph_english_tweet_output.txt','r')
f3 = open('english_label.txt','r')
f4 = open('hydb_predict_class.txt','r')
f55 = open('graph_english_tweet_output.txt','w')
f5 = open('final_graph_english_tweet_output.txt','w')
f6 = open('eval_file.txt','w')
t0 = time.clock()
cnt = 0
timeMap = {}
k = 0

for i in range(0,55):
	#generate summary if new count has exceeded
	# tempflag = 0
	# if cnt >= maxYnodes:
		#get nodes in first bipartition
		X_nodes = set(n for n,d in G.nodes(data=True) if d['bipartite']==0)
		X_list = []
		for node in X_nodes:
			deg = G.degree(node)
			m = MyNode(node,deg,timeMap[node])
			X_list.append(m)
		#sort according to degree
		X_list = sorted(X_list, key=operator.attrgetter('degree'), reverse=True)
		for node in X_list:
			nbrs = G.neighbors(node.nt)
			if len(nbrs) > 0:
				summarySet.append(node)
			for nbr in nbrs:
				addToDict(nbr)
				G.remove_node(nbr)
			Y_nodes = set(n for n,d in G.nodes(data=True) if d['bipartite']==1)
			if len(list(Y_nodes)) == 0:
				summarySet = sorted(summarySet, key=operator.attrgetter('timestamp'), reverse=False)
				cnt = 0
				G.clear()
				f7 = open('part_graph_summary_' + str(k) + '.txt', 'w')
				cumSummarySet = cumSummarySet + summarySet
				# f6.write(tId)
				# f6.write(len(summarySet))
				for tweet in cumSummarySet:
					f7.write(tweet.nt)
				for tweet in summarySet:
					f5.write(summarySet)
				summarySet = []
				f7.close()
				k = k + 1
				timeMap = {}
				# tempflag = 1
				break
		# if tempflag == 1:
			# break
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
		timeMap[nt] = i
		s = f1.readline()
		tokens = s.split('\t')
		while len(tokens) > 1:
			if tokens[1] == '^' or tokens[1] == '$' or tokens[1] == 'U':
				if tokens[1] == '$':
					tokens[1] = numToWord(tokens[1])
				# if tokens[1] == 'U':
				# 	try:
				# 		resp = urllib2.urlopen(tokens[0])
				# 		if resp.getcode() == 200:
				# 			tokens[0] = resp.url
				# 	except:
				# 		pass
				if tokens[0] not in wordsCovered:
					synsetlist = getData(tokens[0])
					t = tuple(synsetlist)
					G.add_node(t, bipartite = 1)
					G.add_edge(nt, t, weight = 1)
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