from mytweet import *
from config import *
from nltk.corpus import wordnet as wn
import re
import time
import urllib2

summarySet = []
arch = []

def getData(twM):
	for i in range(len(twM)):
		try:
			s = wn.synsets(twM[i])
			for ss in s:
				lemmaN = ss.lemma_names()
				for lss in lemmaN:
					twM.append(lss)
		except:
			pass
	return twM

def findSimilarityArch(lwordNT):
	global arch
	lwordNT = getData(lwordNT)
	wordNT = set(lwordNT)
	l1 = len(list(wordNT))
	flag = 0
	for i in range(len(arch)):
		sim = len(list(wordNT & arch[i].imp))
		l2 = len(list(arch[i].imp))
		if l1 > l2:
			coefficient = float(sim) / l1
		else:
			coefficient = float(sim) / l2
		# print coefficient
		if coefficient > thrFF:
			flag = 1
			print arch[i].text
			break
	if flag == 0:
		return False
	else:
		return True


def findSimilarity(nt,lwordNT,index,highImp):
	global summarySet
	global f5
	lwordNT = getData(lwordNT)
	wordNT = set(lwordNT)
	l1 = len(list(wordNT))
	flag = 0
	flag2 = 0
	for i in range(len(summarySet)):
		if len(list(highImp)) == 0:
			flag2 = 1
		if flag2 == 0: 
			sim = len(list(summarySet[i].highImp & highImp))
			if sim > 0:
				flag2 = 1
		sim = len(list(wordNT & summarySet[i].imp))
		l2 = len(list(summarySet[i].imp))
		if l1 > l2:
			coefficient = float(sim) / l1
		else:
			coefficient = float(sim) / l2
		# print coefficient
		if coefficient > thrFF:
			flag = 1
			break
	if flag == 0 or flag2 == 0:
		newT = MyTweet(nt, wordNT, highImp)
		f5.write(nt)
		summarySet.append(newT)
		return True
	else:
		return False
		# print newT.text

f1 = open('hydb_english_tag.txt','r')
f2 = open('english_tweet.txt','r')
f3 = open('english_label.txt','r')
f4 = open('hydb_predict_class.txt','r')
f5 = open('english_tweet_output.txt','w')
print "summarizing..."
t0 = time.clock()
cnt = 0
for i in range(0,983):
	if len(summarySet) == TMax + exc:
		arch = arch + summarySet[:exc]
		summarySet = summarySet[exc:TMax+exc]
	nt = f2.readline()
	if i < 501:
		flag = f3.readline()
		flag = flag[:-1]
		# print flag
	else:
		flag = f4.readline()
		flag = (flag.split('\t'))[2]
	if flag == '1': #if fact
		s = f1.readline()
		tokens = s.split('\t')
		words = []
		highImp = []
		while len(tokens) > 1:
			if tokens[1] == '^' or tokens[1] == 'N' or tokens[1] == '$' or tokens[1] == 'U':
				# if tokens[1] == 'U':
				# 	try:
				# 		resp = urllib2.urlopen(tokens[0])
				# 		if resp.getcode() == 200:
				# 			tokens[0] = resp.url
				# 	except:
				# 		pass
				words.append(tokens[0])
				if tokens[1] == '$':
					highImp.append(tokens[0])
			s = f1.readline()
			tokens = s.split('\t')
		if len(words) > 0:
			highImp = set(highImp)
			if findSimilarity(nt,words,i,highImp) == False:
				if findSimilarityArch(words) == True:
					print nt
					print i
					cnt = cnt + 1

	else: #if opinion
		s = f1.readline()
		tokens = s.split('\t')
		while len(tokens) > 1:
			s = f1.readline()
			tokens = s.split('\t')

t1 = time.clock()
print "time elapsed = " + str(t1 - t0)
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f1 = open('arch_english_tweet_output.txt','w')
for i in range(len(arch)):
	f1.write(arch[i].text)
f1.close()
# print len(summarySet)
print cnt
print len(arch)
print len(summarySet)