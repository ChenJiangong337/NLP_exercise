#   Filename: NLP_chapter4.py

import nltk,pprint,re
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
from urllib import request
from bs4 import BeautifulSoup
from timeit import Timer

#   4.1 回到基础

foo = 'Monty'
bar = foo
foo = 'Python'
print(bar)

foo = ['Monty','Python']
bar = foo
foo[1] = 'Bodkin'
print(bar)

empty = []
nested = [empty,empty,empty]
print(nested)
nested[1].append('Python')
print(nested)

nested = [[]]*3
print(nested)
nested[1].append('0')
print(nested)
print(id(nested[0]))
print(id(nested[1]))
print(id(nested[2]))

nested = [[]]*3
nested[1].append('Python')
nested[1] = ['Monty']
print(nested)

size = 5
python = ['Python']
snake_nest = [python]*size
print(snake_nest[0] == snake_nest[1] ==snake_nest[2] == snake_nest[3] == snake_nest[4])
print(snake_nest[0] is snake_nest[1] is snake_nest[2] is snake_nest[3] is snake_nest[4])

import random
position = random.choice(range(size))
snake_nest[position] = ['Python']
print(snake_nest)
print(snake_nest[0] == snake_nest[1] ==snake_nest[2] == snake_nest[3] == snake_nest[4])
print(snake_nest[0] is snake_nest[1] is snake_nest[2] is snake_nest[3] is snake_nest[4])

print([id(snake) for snake in snake_nest])

mixed = ['cat','',['dog'],[]]
for element in mixed:
    if element:
        print(element)

animals = ['cat','dog']
if 'cat' in animals:
    print(1)
elif 'dog' in animals:
    print(2)

sent = ['No','good', 'fish', 'goes', 'anywhere', 'without', 'a', 'porpoise', '.']
print(all(len(w) > 4 for w in sent))
print(any(len(w) > 4 for w in sent))

#   4.2 序列

t = 'walk','fem',3
print(t)
print(t[0])
print(t[1:])
print(len(t))

raw = 'I turned off the spectroroute'
text = ['I','turned','off','the','spectroroute']
pair = (6,'turned')
print(raw[2],text[3],pair[1])
print(raw[-3:],text[-3:],pair[-3:])
print(len(raw),len(text),len(pair))

raw = 'Red lorry, yellow lorry, red lorry, yellow lorry.'
text = nltk.word_tokenize(raw)
fdist = nltk.FreqDist(text)
print(list(fdist))
for key in fdist:
    print(fdist[key],end=' ')

words = ['I','tumed','off','the','spectrotoute']
words[2],words[3],words[4]=words[3],words[4],words[2]
print(words)

words = ['I','tumed','off','the','spectrotoute']
tags = ['noun','verb','prep','det','noun']
print(zip(words,tags))
print(list(zip(words,tags)))
print(list(enumerate(words)))

text = nltk.corpus.nps_chat.words()
cut = int(0.9*len(text))
training_data,test_data = text[:cut],text[cut:]
print(text == training_data + test_data)
print(len(training_data)/len(test_data))

words = 'I turned off the spectroroute'.split()
wordlens = [(len(word),word) for word in words]
wordlens.sort()
print(' '.join(w for (_,w) in wordlens))

lexicon = [
    ('the','det',['Di:','D@']),
    ('off','prep',['Qf','O:f'])
    ]

lexicon.sort()
lexicon[1] = ('turned','VBD',['t3:nd','t3`nd'])
del lexicon[0]
print(lexicon)

text = '''"When I use a word," Humpty Dumpty said in rather a scornful tone,
"it means just what I choose it to mean - neither more nor less."'''
print([w.lower() for w in nltk.word_tokenize(text)])

print(max([w.lower() for w in nltk.word_tokenize(text)]))
print(max(w.lower() for w in nltk.word_tokenize(text)))

#   4.3 风格的问题

fd = nltk.FreqDist(nltk.corpus.brown.words())
cumulative = 0.0
most_common_words = [word for (word,count) in fd.most_common()]
for rank,word in enumerate(most_common_words):
    cumulative += fd.freq(word)
    print("{:3} {:6.2%} {}".format(rank+1,cumulative,word))
    if cumulative>0.25:
        break

sent = ['The','dog','gave','John','the','newspaper']
n = 3
print([sent[i:i+n] for i in range(len(sent)-n+1)])

m,n = 3,7
array = [[set() for i in range(n)] for j in range(m)]
array[2][5].add('Alice')
pprint.pprint(array)

print(['very' for i in range(3)])

array = [[set()]*n]*m
array[2][5].add(7)
pprint.pprint(array)

#   4.4 函数:结构化编程的基础

def get_text(file):
    '''Read text from a file, normalizing whitespace and stripping HTML makeup.'''
    text = open(file).read()
    text = re.sub('\s+',' ',text)
    text = re.sub(r'<.*?>',' ',text)
    return text

help(get_text)

def repeat(msg,num):
    return ' '.join([msg]*num)
monty = 'Monty Python'
print(repeat(monty,3))

def monty():
    return 'Monty Python'
print(monty())

print(repeat(monty(),3))
print(repeat('Monty Python',3))

def my_sort1(mylist):
    # good:modifies its argument, no return value
    mylist.sort()
def my_sort2(mylist):
    # good: doesn't touch its argument, returns value
    return sorted(mylist)
def my_sort3(mylist):
    # bad: modifies its argument and also returns it
    mylist.sort()
    return mylist

def set_up(word,properties):
    word = 'lolcat'
    properties.append('noun')
    properties[0] = 5
w = ''
p = []
set_up(w,p)
print(w)
print(p)

def tag(word):
    if word in ['a','the','all']:
        return 'det'
    else:
        return 'noun'

print(tag('the'))
print(tag('knight'))
print(tag(["'Tis','but','a','scratch'"]))

def tag(word):
    assert isinstance(word,str),"argument to tag()  must be a string"
    if word in ['a','the','all']:
        return 'det'
    else:
        return 'noun'

print(tag('the'))
print(tag('knight'))
#print(tag(["'Tis','but','a','scratch'"]))

def freq_words(url,freqdist,n):
    html = request.urlopen(url).read().decode('utf8')
    raw = BeautifulSoup(html).get_text()
    for word in word_tokenize(raw):
        freqdist[word.lower()]+=1
    result = []
    for word,count in freqdist.most_common(n):
        result = result + [word]
    print(result)

constitution = "http://www.archives.gov/exhibits/charters/constitution_transcript.html"
fd = nltk.FreqDist()
freq_words(constitution,fd,30)

def freq_words(url,n):
    html = request.urlopen(url).read().decode('utf8')
    text = BeautifulSoup(html).get_text()
    freqdist = nltk.FreqDist(word.lower() for word in word_tokenize(text))
    return [word for (word,_) in fd.most_common(n)]

print(freq_words(constitution,30))

def accuracy(reference, test):
    """
    Calculate the fraction of test items that equal the corresponding reference items.
    Given a list of reference values and a corresponding list of test values,
    return the fraction of corresponding values that are equal.
    In particular, return the fraction of indexes
    {0<i<=len(test)} such that C{test[i] == reference[i]}.
    >>> accuracy(['ADJ', 'N', 'V', 'N'], ['N', 'N', 'V', 'ADJ'])
    0.5
    @param reference: An ordered list of reference values.
    @type reference: C{list}
    @param test: A list of values to compare against the corresponding reference values.
    @type test: C{list}
    @rtype: C{float}
    @raise ValueError: If C{reference} and C{length} do not have thesame length.
    """
    if len(reference) != len(test):
        raise ValueError("Lists must have the same length.")
    num_correct = 0
    for x, y in zip(reference, test):
        if x == y:
            num_correct += 1
    return float(num_correct) / len(reference)

#   4.5 更多关于函数

sent = ['Take', 'care', 'of', 'the', 'sense', ',', 'and', 'the',
        'sounds', 'will', 'take', 'care', 'of', 'themselves', '.']
def extract_property(prop):
    return [prop(word) for word in sent]

print(extract_property(len))

def last_letter(word):
    return word[-1]
print(extract_property(last_letter))
print(extract_property(lambda w:w[-1]))

def search1(substring,words):
    result = []
    for word in words:
        if substring in word:
            result.append(word)
    return result
def search2(substring,words):
    for word in words:
        if substring in word:
            yield word
print('search1:')
for item in search1('zz',nltk.corpus.brown.words())[:20]:
    print(item)
print('search2:')
i = 0
for item in search2('zz',nltk.corpus.brown.words()):
    print(item)
    i +=1
    if (i > 20):
        break

def permutations(seq):
    if len(seq)<=1:
        yield seq
    else:
        for perm in permutations(seq[1:]):
            for i in range(len(perm)+1):
                yield perm[:i] + seq[0:1] +perm[i:]
print(list(permutations(['police','fish','buffalo'])))

def is_content_word(word):
    return word.lower() not in ['a','of','the','and','will',',','.']
sent = ['Take', 'care', 'of', 'the', 'sense', ',', 'and', 'the',
        'sounds', 'will', 'take', 'care', 'of', 'themselves', '.']
print(list(filter(is_content_word,sent)))
print([w for w in sent if is_content_word(w)])
                                
lengths = list(map(len,nltk.corpus.brown.sents(categories='news')))
print(sum(lengths)/len(lengths))
lengths = [len(w) for w in nltk.corpus.brown.sents(categories='news')]
print(sum(lengths)/len(lengths))

sent = ['Take', 'care', 'of', 'the', 'sense', ',', 'and', 'the',
        'sounds', 'will', 'take', 'care', 'of', 'themselves', '.']
print(list(map(lambda w:len(list(filter(lambda c: c.lower() in 'aeiou',w))),sent)))
print([len([c for c in w if c.lower() in 'aeiou'])for w in sent])

def generic(*args,**kwargs):
    print(args)
    print(kwargs)
generic(1,'African swallow',monty='Python')

song = [['four', 'calling', 'birds'],
        ['three', 'French', 'hens'],
        ['two', 'turtle', 'doves']]
print(list(zip(song[0],song[1],song[2])))
print(list(zip(*song)))

#   4.6 程序开发

def find_words(text,wordlength,result=[]):
    for word in text:
        if len(word) == wordlength:
            result.append(word)
    return result
print(find_words(['omg','teh','lolcat','sitted','on','teh','mat'],3))
print(find_words(['omg','teh','lolcat','sitted','on','teh','mat'],2,['ur']))
print(find_words(['omg','teh','lolcat','sitted','on','teh','mat'],3))

#>>> import pdb
#>>> import mymodule
#>>> pdb.run('mymodule.myfunction()')

#>>> find_words(['cat'],3)
#['omg', 'teh', 'teh', 'mat', 'omg', 'teh', 'teh', 'mat', 'cat']
#>>> pdb.run("find_words(['dogs'],3)")
#> <string>(1)<module>()
#(Pdb) s
#--Call--
#> /Users/chenjiangong/Desktop/research in Tsinghua/NLP_code/test.py(6)find_words()
#-> def find_words(text,wordlength,result=[]):
#(Pdb) args
#text = ['dogs']
#wordlength = 3
#result = ['omg', 'teh', 'teh', 'mat', 'omg', 'teh', 'teh', 'mat', 'cat']
#(Pdb) quit

#   4.7 算法设计

def factoriall(n):
    result = 1
    for i in range(n):
        result *= (i+1)
    return result

def factoriall2(n):
    if n == 1:
        return 1
    else:
        return n*factoriall2(n-1)

def size1(s):
    return 1+sum(size1(child) for child in s.hyponyms())

def size2(s):
    layer = [s]
    total = 0
    while layer:
        total += len(layer)
        layer = [h for c in layer for h in c.hyponyms()]
    return total

dog = wn.synset('dog.n.01')
print(size1(dog))
print(size2(dog))

def insert(trie,key,value):
    if key:
        first,rest = key[0],key[1:]
        if first not in trie:
            trie[first] = {}
        insert(trie[first],rest,value)
    else:
        trie['value'] = value
trie = nltk.defaultdict(dict)
insert(trie,'chat','cat')
insert(trie,'chien','dog')
insert(trie,'chair','flesh')
insert(trie,'chic','stylish')
trie = dict(trie)   #for nicer printing
print(trie['c']['h']['a']['t']['value'])
pprint.pprint(trie,width=40)

def raw(file):
    contents = open(file).read()
    contents = re.sub(r'<.*?>',' ',contents)
    contents = re.sub('\s+',' ',contents)
    return contents
def snippet(doc,term):  #buggy
    text = ' '*30 + raw(doc) +' '*30
    pos = text.index(term)
    return text[pos-30:pos+30]
print("Building Index...")
files = nltk.corpus.movie_reviews.abspaths()
idx = nltk.Index((w,f) for f in files for w in raw(f).split())
query = ''
while query != "quit":
    query = input("query> ")
    if query in idx:
        for doc in idx[query]:
            print(snippet(doc,query))
    else:
        print("Not found.")
                  
def preprocess(tagged_corpus):
    words = set()
    tags = set()
    for sent in tagged_corpus:
        for word,tag in sent:
            words.add(word)
            tags.add(tag)
    wm = dict((w,i) for (i,w) in enumerate(words))
    tm = dict((t,i) for (i,t) in enumerate(tags))
    return [[(wm[w],tm[t]) for (w,t) in sent] for sent in tagged_corpus]

vocab_size = 100000
setup_list = "import random; vocab = range(%d)" %vocab_size
setup_set = "import random; vocab = set(range(%d))"%vocab_size
statement = "random.randint(0,%d) in vocab"%(vocab_size*2)
print(Timer(statement,setup_list).timeit(1000))
print(Timer(statement,setup_set).timeit(1000))

def virahanka1(n):
    if n == 0:
        return[""]
    elif n == 1:
        return["S"]
    else:
        s = ["S" + prosody for prosody in virahanka1(n-1)]
        l = ["L" + prosody for prosody in virahanka1(n-2)]
        return s+l
def virahanka2(n):
    lookup = [[""],["S"]]
    for i in range(n-1):
        s = ["S" + prosody for prosody in lookup[i+1]]
        l = ["L" + prosody for prosody in lookup[i]]
        print(s)
        print(l)
        lookup.append(s+l)
        print(lookup)       
    return lookup[n]
def virahanka3(n,lookup={0:[""],1:["S"]}):
    if n not in lookup:
        s = ["S" + prosody for prosody in virahanka3(n-1)]
        l = ["L" + prosody for prosody in virahanka3(n-2)]
        lookup[n] = s+l
    return lookup[n]
from nltk import memoize
@memoize
def virahanka4(n):
    if n == 0:
        return[""]
    elif n == 1:
        return["S"]
    else:
        s = ["S" + prosody for prosody in virahanka4(n-1)]
        l = ["L" + prosody for prosody in virahanka4(n-2)]
        return s+l
print(virahanka1(4))
print(virahanka2(4))
print(virahanka3(4))
print(virahanka4(4))
    
