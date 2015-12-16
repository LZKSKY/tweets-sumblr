from nltk.corpus import wordnet as wn

apple = wn.synset('apple.n.01')
pear = wn.synset('pear.n.01')

print apple[0]