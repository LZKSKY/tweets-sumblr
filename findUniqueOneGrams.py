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
	print len(words1)
	print len(words2)
	s1 = words2.difference(words1)
	print len(s1)
	f3 = open('uniqueWords.txt','w')
	for word in s1:
		f3.write(word)
	f3.close()

findSim('graph_english_tweet_output_one_grams_no_common.txt','graph_english_tweet_output_one_grams.txt')