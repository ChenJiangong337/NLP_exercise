# Filename: NLP_chapter2.py

#2.1 获取文本语料库

import nltk

nltk.corpus.gutenberg.fileids()
emma = nltk.corpus.gutenberg.words('austen-emma.txt')
emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))

from nltk.corpus import gutenberg
gutenberg.fileids()
emma = gutenberg.words('austen-emma.txt')

for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set([w.lower() for w in gutenberg.words(fileid)]))
    print(int(num_chars/num_words),int(num_words/num_sents),int(num_words/num_vocab),fileid)

from nltk.corpus import webtext
from nltk.corpus import nps_chat

from nltk.corpus import brown

cfd = nltk.ConditionalFreqDist(
    (genre,word)
    for genre in brown.categories()
    for word in brown.words(categories=genre))
genres = ['news','religion','hobbies','science_fiction','romance','humor']
modals = ['can','could','may','might','must','will']
cfd.tabulate(conditions=genres,samples=modals)

from nltk.corpus import inaugural
print([fileid[:4] for fileid in inaugural.fileids()])

from nltk.corpus import PlaintextCorpusReader
corpus_root = '/Users/chenjiangong/Desktop/research in Tsinghua/NLP_code'
wordlists = PlaintextCorpusReader(corpus_root, '.*')

from nltk.corpus import BracketParseCorpusReader

#2.2 条件频率分布

cfd = nltk.ConditionalFreqDist(
    (genre,word)
    for genre in brown.categories()
    for word in brown.words(categories=genre))

genre_word = [(genre, word)
              for genre in ['news', 'romance']
              for word in brown.words(categories=genre)]
cfd = nltk.ConditionalFreqDist(genre_word)

cfd
cfd.conditions()

cfd = nltk.ConditionalFreqDist(
    (target,fileid[:4])
    for fileid in inaugural.fileids()
    for target in ['america','citizen']
    for w in inaugural.words(fileid)
    if w.lower().startswith(target))

def generate_model(cfdist, word, num=15):
    for i in range(num):
        print (word,end=' ')
        word = cfdist[word].max()

text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)

cfd['living']
generate_model(cfd,'living')

#2.4 词典资源

from nltk.corpus import words

from nltk.corpus import stopwords

names = nltk.corpus.names

entries = nltk.corpus.cmudict.entries()

for word,pron in entries:
    if len(pron) == 3:
        ph1,ph2,ph3 = pron
        if ph1 == 'P' and ph3 == 'T':
            print(word,ph2,end=' ')

print([w for w, pron in entries if pron[-1] == 'M' and w[-1] == 'n'])

print(sorted(set(w[:2] for w, pron in entries if pron[0] == 'N' and w[0] != 'n')))

def stress(pron):
    return [char for phone in pron for char in phone if char.isdigit()]

p3 = [(pron[0]+'-'+pron[2],word)
      for (word,pron) in entries
      if pron[0]=='P' and len(pron)==3]
cfd = nltk.ConditionalFreqDist(p3)
for template in sorted(cfd.conditions()):
    if len(cfd[template])>10:
        words = sorted(cfd[template])
        wordstring=' '.join(words)
        print(template,wordstring[:70]+'...')

prondict = nltk.corpus.cmudict.dict()
prondict['fire']

text = ['natural', 'language', 'processing']
print([ph for w in text for ph in prondict[w][0]])

from nltk.corpus import swadesh

fr2en = swadesh.entries(['fr', 'en'])
translate = dict(fr2en) 
print(translate['chien'])

de2en = swadesh.entries(['de', 'en']) # German-English
es2en = swadesh.entries(['es', 'en']) # Spanish-English
translate.update(dict(de2en))
translate.update(dict(es2en))

#2.5 WordNet

from nltk.corpus import wordnet as wn
wn.synsets('motorcar')

wn.synset('car.n.01').lemma_names()

wn.synset('car.n.01').definition()
wn.synset('car.n.01').examples()

wn.synset('car.n.01').lemmas()
wn.lemma('car.n.01.automobile')
wn.lemma('car.n.01.automobile').synset()
wn.lemma('car.n.01.automobile').name()

wn.lemmas('car')
for dish in wn.lemmas('dish'):
    print(dish.synset().definition())

motorcar = wn.synset('car.n.01')
types_of_motorcar = motorcar.hyponyms()

motorcar.hypernyms()
paths = motorcar.hypernym_paths()	#上位词路径，即层次结构
print([synset.name() for synset in paths[0]])
print([synset.name() for synset in paths[1]])

motorcar.root_hypernyms()

wn.synset('tree.n.01').part_meronyms()
wn.synset('tree.n.01').substance_meronyms()
wn.synset('tree.n.01').member_holonyms()
wn.synsets('mint', wn.NOUN)
wn.synset('walk.v.01').entailments()
wn.lemma('supply.n.02.supply').antonyms()

right = wn.synset('right_whale.n.01')
orca = wn.synset('orca.n.01')
minke = wn.synset('minke_whale.n.01')
tortoise = wn.synset('tortoise.n.01')
novel = wn.synset('novel.n.01')
right.lowest_common_hypernyms(minke)
right.lowest_common_hypernyms(orca)
right.lowest_common_hypernyms(tortoise)
right.lowest_common_hypernyms(novel)

wn.synset('baleen_whale.n.01').min_depth()
wn.synset('whale.n.02').min_depth()
wn.synset('vertebrate.n.01').min_depth()
wn.synset('entity.n.01').min_depth()

right.path_similarity(minke)
right.path_similarity(orca)
right.path_similarity(tortoise)
right.path_similarity(novel)
