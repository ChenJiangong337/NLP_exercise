#   Filename:   NLP_chapter5.py

import nltk
from nltk.corpus import brown

#   5.1 使用词性标注器

text = nltk.word_tokenize("And now for something completely different")
print(nltk.pos_tag(text))

text = nltk.word_tokenize("They refuse to permit us to obtain the refuse permit")
print(nltk.pos_tag(text))
                          
text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
text.similar('woman')
text.similar('bought')
text.similar('over')
text.similar('the')

#   5.2 标注语料库

tagged_token = nltk.tag.str2tuple('fly/NN')
print(tagged_token)
print(tagged_token[0])
print(tagged_token[1])

sent = '''
The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/IN
other/AP topics/NNS ,/, AMONG/IN them/PPO the/AT Atlanta/NP and/CC
Fulton/NP-tl County/NN-tl purchasing/VBG departments/NNS which/WDT it/PP
said/VBD ``/`` ARE/BER well/QL operated/VBN and/CC follow/VB generally/R
accepted/VBN practices/NNS which/WDT inure/VB to/IN the/AT best/JJT
interest/NN of/IN both/ABX governments/NNS ''/'' ./.
'''
print([nltk.tag.str2tuple(t) for t in sent.split()])

print(nltk.corpus.brown.tagged_words()[:50])
print(nltk.corpus.brown.tagged_words(tagset='universal')[:50])
print(nltk.corpus.nps_chat.tagged_words()[:50])
print(nltk.corpus.conll2000.tagged_words()[:50])
print(nltk.corpus.treebank.tagged_words()[:50])
print(nltk.corpus.treebank.tagged_words(tagset='universal')[:50])

print(nltk.corpus.sinica_treebank.tagged_words()[:20])
print(nltk.corpus.indian.tagged_words()[:20])
print(nltk.corpus.mac_morpho.tagged_words()[:20])
print(nltk.corpus.conll2002.tagged_words()[:20])
print(nltk.corpus.cess_cat.tagged_words()[:20])

brown_news_tagged = brown.tagged_words(categories='news',tagset='universal')
tag_fd = nltk.FreqDist(tag for (word,tag) in brown_news_tagged)
print(tag_fd.most_common(10))
tag_fd.plot(cumulative=True)

word_tag_pairs = nltk.bigrams(brown_news_tagged)
fdist=nltk.FreqDist([a[1] for (a,b) in word_tag_pairs if b[1] =='NOUN'])
print([tag for (tag,_) in fdist.most_common()])

wsj = nltk.corpus.treebank.tagged_words(tagset='universal')
word_tag_fd = nltk.FreqDist(wsj)
print([wt[0] for (wt,_) in word_tag_fd.most_common(50) if wt[1] == 'VERB'])

cfd1 = nltk.ConditionalFreqDist(wsj)
print(cfd1['yield'].most_common())
print(cfd1['cut'].most_common())

wsj = nltk.corpus.treebank.tagged_words()
cfd2 = nltk.ConditionalFreqDist((tag,word) for (word,tag) in wsj)
print(cfd2['VBN'].most_common(20))

cfd1 = nltk.ConditionalFreqDist(wsj)
print([w for w in cfd1.conditions() if 'VBD' in cfd1[w] and 'VBN' in cfd1[w]][:20])
idx1 = wsj.index(('kicked','VBD'))
print(wsj[idx1-4:idx1+1])
idx2 = wsj.index(('kicked','VBN'))
print(wsj[idx2-4:idx2+1])

vn = list(cfd2['VBN'])
for w in vn[:20]:
    idx3 = wsj.index((w,'VBN'))
    print(wsj[idx3-1])

def findtags(tag_prefix,tagged_text):
    cfd = nltk.ConditionalFreqDist((tag,word) for (word,tag) in tagged_text
                                   if tag.startswith(tag_prefix))
    return dict((tag,cfd[tag].most_common(5)) for tag in cfd.conditions())

tagdict = findtags('NN',brown.tagged_words(categories='news'))
for tag in sorted(tagdict):
    print(tag,tagdict[tag])

brown_learned_text = brown.words(categories='learned')
print(sorted(set(b for (a,b) in nltk.bigrams(brown_learned_text) if a=='often'))[:20])

brown_lrnd_tagged = brown.tagged_words(categories='learned',tagset='universal')
tags = [b[1] for (a,b) in nltk.bigrams(brown_lrnd_tagged) if a[0] == 'often']
fd = nltk.FreqDist(tags)
fd.tabulate()

def process(sentence):
    for (w1,t1),(w2,t2),(w3,t3) in nltk.trigrams(sentence):
        if (t1.startswith('V') and t2 == 'TO' and t3.startswith('V')):
            print(w1,w2,w3)

for tagged_sent in brown.tagged_sents()[:50]:
    process(tagged_sent)

brown_news_tagged = brown.tagged_words(categories='news',tagset='universal')
data = nltk.ConditionalFreqDist((word.lower(),tag)
                                for (word,tag) in brown_news_tagged)
for word in data.conditions():
    if len(data[word])>3:
        tags = [tag for (tag,_) in data[word].most_common()]
        print(word,' '.join(tags))

#   5.3 使用 Python 字典映射词及其属性

pos = {}
print(pos)
pos['colorless'] = 'ADJ'
print(pos)
pos['ideas'] = 'N'
pos['sleep'] = 'V'
pos['furiously'] = 'ADV'
print(pos)

print(pos['ideas'])
print(pos['colorless'])

print(list(pos))
print(sorted(pos))
print([w for w in pos if w.endswith('s')])

for word in sorted(pos):
    print(word+':',pos[word])

print(list(pos.keys()))
print(list(pos.values()))
print(list(pos.items()))
for key,val in sorted(pos.items()):
    print(key+':',val)

pos['sleep'] = 'V'
print(pos['sleep'])
pos['sleep'] = 'N'
print(pos['sleep'])
pos['sleep'] = ['N','V']
print(pos['sleep'])

pos = {'colerless':'ADJ','ideas':'N','sleep':'V','furiously': 'ADV'}
pos = dict(colerless='ADJ',ideas='N',sleep='V', furiously='ADV')
print(pos)

frequency = nltk.defaultdict(int)
frequency['colorless'] = 4
print(frequency['ideas'])
pos = nltk.defaultdict(list)
pos['sleep'] = ['V','N']
print(pos['ideas'])

pos = nltk.defaultdict(lambda:'N')
pos['colorless'] = 'ADJ'
print(pos['blog'])
print(list(pos.items()))

alice = nltk.corpus.gutenberg.words('carroll-alice.txt')
vocab = nltk.FreqDist(alice)
v1000 = [word for (word,_) in vocab.most_common(1000)]
mapping = nltk.defaultdict(lambda:'UNK')
for v in v1000:
    mapping[v] = v
alice2 = [mapping[v] for v in alice]
print(alice2[:50])
print(len(set(alice2)))

counts = nltk.defaultdict(int)
for (word,tag) in brown.tagged_words(categories='news',tagset='universal'):
    counts[tag] += 1
print(counts['NOUN'])
print(sorted(counts)[:20])

from operator import itemgetter
print(sorted(counts.items(),key=itemgetter(1),reverse=True))
print([t for t,c in sorted(counts.items(),key=itemgetter(1),reverse=True)])

pair = ('NP',8336)
print(pair[1])
print(itemgetter(1)(pair))

last_letters = nltk.defaultdict(list)
words = nltk.corpus.words.words('en')
for word in words:
    key = word[-2:]
    last_letters[key].append(word)

print(last_letters['ly'][:20])
print(last_letters['zy'][:20])

anagrams = nltk.defaultdict(list)
for word in words:
    key = ''.join(sorted(word))
    anagrams[key].append(word)

print(anagrams['aeilnrt'])

anagrams = nltk.Index((''.join(sorted(w)),w) for w in words)
print(anagrams['aeilnrt'])

pos = nltk.defaultdict(lambda:nltk.defaultdict(int))
brown_news_tagged = brown.tagged_words(categories='news',tagset='universal')
for ((w1,t1),(w2,t2)) in nltk.bigrams(brown_news_tagged):
    pos[(t1,w2)][t2]+=1
print(pos[('DET','right')])

counts = nltk.defaultdict(int)
for word in nltk.corpus.gutenberg.words('milton-paradise.txt'):
    counts[word]+=1
print([key for (key,value) in counts.items() if value == 32])

pos = {'colorless': 'ADJ', 'ideas': 'N', 'sleep': 'V', 'furiously': 'ADV'}
pos2 = dict((value,key) for (key,value) in pos.items())
print(pos2['N'])

pos.update({'cats': 'N','scratch': 'V', 'peacefully': 'ADV', 'old': 'ADJ'})
pos2 = nltk.defaultdict(list)
for key,value in pos.items():
    pos2[value].append(key)

print(pos2['ADV'])

pos2 = nltk.Index((value,key) for (key,value) in pos.items())
print(pos2['ADV'])

#   5.4 自动标注

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')

tags = [tag for (word,tag) in brown.tagged_words(categories='news')]
print(nltk.FreqDist(tags).max())
        
raw = 'I do not like green eggs and ham, I do not like them Sam I am!'
tokens = nltk.word_tokenize(raw)
default_tagger = nltk.DefaultTagger('NN')
print(default_tagger.tag(tokens))

print(default_tagger.evaluate(brown_tagged_sents))

patterns = [
    (r'.*ing$','VBG'),
    (r'.*ed$','VBD'),
    (r'.*es$','VBZ'),
    (r'.*ould$','MD'),
    (r'.*\'s$','NN$'),
    (r'.*s$','NNS'),
    (r'^-?[0-9]+(.[0-9]+)?$','CD'),
    (r'.*','NN')
    ]

regexp_tagger = nltk.RegexpTagger(patterns)
print(regexp_tagger.tag(brown_sents[3])[:20])
print(regexp_tagger.evaluate(brown_tagged_sents))

fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = fd.most_common(100)
likely_tags = dict((word,cfd[word].max()) for (word,_) in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
print(baseline_tagger.evaluate(brown_tagged_sents))

sent = brown.sents(categories='news')[3]
print(baseline_tagger.tag(sent))

baseline_tagger = nltk.UnigramTagger(model=likely_tags,
                                     backoff=nltk.DefaultTagger('NN'))

def performance(cfd,wordlist):
    lt = dict((word,cfd[word].max()) for word in wordlist)
    baseline_tagger = nltk.UnigramTagger(model=lt,backoff=nltk.DefaultTagger('NN'))
    return baseline_tagger.evaluate(brown.tagged_sents(categories='news'))

def display():
    import pylab
    word_freqs = nltk.FreqDist(brown.words(categories='news')).most_common()
    words_by_freq = [w for (w,_) in word_freqs]
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
    sizes = 2 ** pylab.arange(15)
    perfs = [performance(cfd, words_by_freq[:size]) for size in sizes]
    pylab.plot(sizes,perfs,'-bo')
    pylab.title('Lookup Tagger Performance with Varying Model Size')
    pylab.xlabel('Model Size')
    pylab.ylabel('Performance')
    pylab.show()

display()

#   5.5 N-gram 标注

brown_tagged_sents = brown.tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')

unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
print(unigram_tagger.tag(brown_sents[2007]))
print(unigram_tagger.evaluate(brown_tagged_sents))

size = int(len(brown_tagged_sents)*0.9)
print(size)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
print(unigram_tagger.evaluate(test_sents))

bigram_tagger = nltk.BigramTagger(train_sents)
print(bigram_tagger.tag(brown_sents[2007]))
unseen_sent = brown_sents[4203]
print(bigram_tagger.tag(unseen_sent))

print(bigram_tagger.evaluate(test_sents))

t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents,backoff=t0)
print(t1.evaluate(test_sents))
t2 = nltk.BigramTagger(train_sents,backoff=t1)
print(t2.evaluate(test_sents))
t3 = nltk.TrigramTagger(train_sents,backoff=t2)
print(t3.evaluate(test_sents))

#>>> from pickle import dump
#>>> output = open('t2.pkl','wb')
#>>> dump(t2,output,-1)
#>>> output.close()

from pickle import load
input = open('t2.pkl','rb')
tagger = load(input)
input.close()

text = """The board's action shows what free enterprise
is up against in our complex maze of regulatory laws ."""
tokens = text.split()
print(tagger.tag(tokens))

cfd = nltk.ConditionalFreqDist(
    ((x[1],y[1],z[0]),z[1])
     for sent in brown_tagged_sents
     for x,y,z in nltk.trigrams(sent))
ambiguous_contexts = [c for c in cfd.conditions() if len(cfd[c])>1]
print(sum(cfd[c].N() for c in ambiguous_contexts)/cfd.N())

test_tags = [tag for sent in brown.sents(categories='editorial')
             for (word,tag) in t2.tag(sent)]
gold_tags = [tag for (word,tag) in brown.tagged_words(categories='editorial')]
print(nltk.ConfusionMatrix(gold_tags,test_tags))

