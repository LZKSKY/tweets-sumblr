from myGraphNode import *
import networkx as nx
import re
# from nltk.corpus import wordnet as wn
from config import *
import time
import operator
# from nltk.corpus import stopwords

global G, summarySet
summarySet = []
cumSummarySet = []

# cachedStopWords = stopwords.words("english")

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

f1 = open('hydb_english_tag.txt','r')
f2 = open('english_tweet.txt','r')
f3 = open('english_label.txt','r')
f4 = open('hydb_predict_class.txt','r')
f5 = open('graph_english_tweet_output.txt','w')
f6 = open('eval_file.txt','w')
f10 = open('CDetect/output/graph_summary.txt', 'w')
f10.close()
f8 = open('lncs_ouput.txt','w')
f8.close()
t0 = time.clock()
cnt = 0
counter = 0
timeMap = {}
k = 0
summarySet = []
i = 1
while i < 983:
	#generate summary if new count has exceeded
	# tempflag = 0
	print i
	print counter
	if counter == 4:
		print i
		break
	f7 = open('CDetect/infomapOut/commdetect' + str(counter) + '.clu','r')
	f7.readline()
	nV = int(f7.readline().split(" ")[1])
	G = []
	G1 = nx.Graph()
	tfMap = {}
	j = 0
	while j < nV:
		nt = f2.readline()
		# nt = ' '.join([word for word in nt.split() if word not in cachedStopWords])
		flag = 1
		if i < 501:
			flag = f3.readline()
			flag = flag[:-1]
			# print flag
		else:
			flag = f4.readline()
			flag = (flag.split('\t'))[2]
		if flag == '1':
			j = j + 1
			cID = int(f7.readline())
			if cID > len(G):
				while len(G) != cID:
					G.append(nx.DiGraph())
			s = f1.readline()
			tokens = s.split('\t')
			prevWord = ""
			prevWord1 = ""
			trail = ""
			k1 = 0
			k2 = 0
			window = []
			while len(tokens) > 1:
				prevWord1 = tokens[0]
				if tfMap.has_key(tokens[0]):
					tfMap[tokens[0]] = tfMap[tokens[0]] + 1
				else:
					tfMap[tokens[0]] = 1
				k1 = k1 + 1
				if k1 == 1:
					G1.add_node(tokens[0],start=1,end=0)
				else:
					G1.add_node(tokens[0],start=0,end=0)
				for r in range(len(window)):
					offset = float(1)/(len(window)-r)
					if G1.has_edge(window[r],tokens[0]):
						G1.edge[window[r]][tokens[0]]['weight'] = G1.edge[window[r]][tokens[0]]['weight'] + offset
					else:
						G1.add_edge(window[r],tokens[0],weight=offset)
				window.append(tokens[0])
				if tokens[1] == '^' or tokens[1] == '$' or tokens[1] == 'U':
					k2 = k2 + 1
					if tokens[1] == '$':
						tokens[0] = numToWord(tokens[0])
					if G[cID-1].has_node(tokens[0]):
						if k2 == 1:
							G[cID-1].node[tokens[0]]['start'] = 1
							G[cID-1].node[tokens[0]]['prefix'].append(trail)
							trail = ""
						else:
							G[cID-1].node[tokens[0]]['prefix'].append(trail)
							if prevWord != "":
								G[cID-1].add_edge(prevWord,tokens[0])
								G[cID-1].node[prevWord]['suffix'].append(trail)
							trail = ""
						prevWord = tokens[0]
					else:
						if k2 != 1:
							G[cID-1].add_node(tokens[0],start=0,end=0,suffix=[],prefix=[trail])
							if prevWord != "":
								G[cID-1].add_edge(prevWord,tokens[0])
								G[cID-1].node[prevWord]['suffix'].append(trail)
							trail = ""
						if k2 == 1:
							G[cID-1].add_node(tokens[0],start=1,end=0,suffix=[],prefix=[trail])
							trail = ""	
						prevWord = tokens[0]
					if tokens[0] == "timesnow":
						print "**********"
						print G[cID-1].node[tokens[0]]['prefix']
						print G[cID-1].node[tokens[0]]['suffix']
						print "**********"
				else:
					trail = trail + " " + tokens[0]
				s = f1.readline()
				tokens = s.split('\t')
			G1.node[prevWord1]['end'] = 1
			# if trail == " .":
				# print G[cID-1].node[prevWord] 
			G[cID-1].node[prevWord]['end'] = 1
			# else:
			G[cID-1].node[prevWord]['suffix'] = [trail]
		else: #if opinion
			s = f1.readline()
			tokens = s.split('\t')
			while len(tokens) > 1:
				s = f1.readline()
				tokens = s.split('\t')
		i = i + 1
	f7.close()
	# for edge in G1.edges():
	# 	v1 = edge[0]
	# 	v2 = edge[1]
	# 	d = G1.edge[v1][v2]['weight']
	# 	wt = (float(tfMap[v1] + tfMap[v2]))/(tfMap[v1] * tfMap[v2] * d)
	# 	G1.edge[v1][v2]['weight'] = wt
	# s_nodes = [n for n,attrdict in G1.node.items() if attrdict['start'] == 1]
	# e_nodes = [n for n,attrdict in G1.node.items() if attrdict['end'] == 1]
	# f8 = open('lncs_ouput.txt','a')
	# for s in s_nodes:
	# 	for e in e_nodes:
	# 		sPath = nx.shortest_path(G1,s,e,'weight')
	# 		for word in sPath:
	# 			f8.write(word + " ")
	# 	f8.write("\n")
	# f8.close()
	for j in range(len(G)):
		s_nodes = [n for n,attrdict in G[j].node.items() if attrdict['start'] == 1]
		e_nodes = [n for n,attrdict in G[j].node.items() if attrdict['end'] == 1]
		# print s_nodes
		# print e_nodes
		# print "-----"
		# print s_nodes
		# print e_nodes
		for s in s_nodes:
			for e in e_nodes:
				for path in nx.all_simple_paths(G[j], source=s, target=e):
					print path
					# line = ""
					line = G[j].node[path[0]]['prefix'][0]
					q = 0
					valid = 1
					while q < len(path) - 1:
						comm = list(set(G[j].node[path[q]]['suffix']) & set(G[j].node[path[q+1]]['prefix']))
						if len(comm) > 0:
							line = line + " " + path[q] + " " + comm[0]
						else:
							valid = 0
							break
						q = q + 1
					if valid == 1:
						line = line + " " + e + G[j].node[e]['suffix'][0]
						summarySet.append(line)
	print "summary:"
	print summarySet
	f9 = open('CDetect/targetwords' + str(counter) + '.txt','r')
	s = f9.readline()
	s = s[:-1]
	FG = nx.Graph()
	words = []
	while s != "":
		words.append(s)
		FG.add_node(s,bipartite=1)
		s = f9.readline()
		s = s[:-1]
	f9.close()
	for line in summarySet:
		FG.add_node(line,bipartite=0)
	print "words:"
	print words
	for word in words:
		for line in summarySet:
			presWords = line.split()
			if word in presWords:
				FG.add_edge(line,word)
	print "edges:"
	print FG.edges()
	# print FG.nodes()
	X_nodes = set(n for n,d in FG.nodes(data=True) if d['bipartite']==0)
	# print X_nodes
	X_list = []
	summarySet = []
	for node in X_nodes:
		deg = FG.degree(node)
		# print deg
		m = MyNode(node,deg,0)
		X_list.append(m)
	#sort according to degree
	X_list = sorted(X_list, key=operator.attrgetter('degree'), reverse=True)
	flag = 0
	for node in X_list:
		nbrs = FG.neighbors(node.nt)
		print nbrs
		print "**"
		if len(nbrs) > 0:
			summarySet.append(node)
		for nbr in nbrs:
			FG.remove_node(nbr)
		Y_nodes = set(n for n,d in FG.nodes(data=True) if d['bipartite']==1)
		print Y_nodes
		if len(list(Y_nodes)) == 0:
			# summarySet = sorted(summarySet, key=operator.attrgetter('timestamp'), reverse=False)
			flag = 1
			FG.clear()
			f10 = open('CDetect/output/graph_summary.txt', 'a')
			# f6.write(tId)
			# f6.write(len(summarySet))
			for tweet in summarySet:
				f10.write(summarySet)
			summarySet = []
			f10.close()
			# timeMap = {}
			# tempflag = 1
			break
	if flag == 0:
		f10 = open('CDetect/output/graph_summary.txt', 'a')
		# f6.write(tId)
		# f6.write(len(summarySet))
		for tweet in summarySet:
			f10.write(tweet.nt + "\n")
		summarySet = []
		f10.close()
	counter = counter + 1
t1 = time.clock()
print "time taken = " + str(t1 - t0)
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()