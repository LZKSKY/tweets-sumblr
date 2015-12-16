import re

def findSim(ipf1,ipf2):
	f1 = open(ipf1,'r')
	f2 = open(ipf2,'r')
	words1 = set()
	words2 = set()
	s = f1.readline()
	while s != "":
		words1.add(s)
		s = f1.readline()
	f1.close()
	s = f2.readline()
	while s != "":
		words2.add(s)
		s = f2.readline()
	f2.close()
	# print len(list(words2 & words1))
	common = len(list(words1 & words2))
	print float(common) / len(list(words2))

for i in range(0,10)
findSim('graph_english_tweet_output_one_grams_no_common.txt','manual_english_tweet_output_one_grams.txt')