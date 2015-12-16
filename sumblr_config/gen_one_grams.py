import re

def findOneGrams(ipF,opF):
	fIn = open(ipF,'r')
	fOut = open(opF,'w')
	dic = set()
	s = fIn.readline()
	while s != "":
		tokens = s.split(' ')
		for token in tokens:
			dic.add(token)
		s = fIn.readline()
	fIn.close()
	for word in dic:
		if word[:4] != "http":
			fOut.write(word)
			fOut.write("\n")

findOneGrams('graph_english_tweet_output.txt','graph_english_tweet_output_one_grams_no_common.txt')