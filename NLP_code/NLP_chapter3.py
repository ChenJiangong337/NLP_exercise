#   Filename: NLP_chapter3.py

import nltk,re,pprint
from nltk import word_tokenize

#   3.1 从网络和硬盘访问文本

from urllib import request
url = "http://www.gutenberg.org/files/2554/2554-0.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')
type(raw)

tokens = word_tokenize(raw)
text = nltk.Text(tokens)

print(raw.find("PARTI"))
print(raw.rfind("End of Project Gutenberg's Crime"))
raw = raw[5303:1157681]

url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
html = request.urlopen(url).read().decode('utf8')
#print(html)

from bs4 import BeautifulSoup
raw = BeautifulSoup(html).get_text()
tokens = word_tokenize(raw)

import feedparser
llog = feedparser.parse("http://languagelog.ldc.upenn.edu/nll/?feed=atom")
llog['feed']['title']
len(llog.entries)
post = llog.entries[2]
post.title		#注意这里的方法是不加()的
content = post.content[0].value
content[:70]
word_tokenize(BeautifulSoup(content).get_text())

f = open('document.txt')
print(f.read())

import os
os.listdir('.')

f = open('document.txt','rU')	#一定要以通用方式打开
for line in f:
    print(line.strip())

path = nltk.data.find('corpora/gutenberg/melville-moby_dick.txt') 
raw = open(path, 'rU').read()

s = input('Enter some text:')
print('You typed',len(word_tokenize(s)),'words')

#   3.3 使用 Unicode 进行文字处理

path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')
f = open(path,encoding='latin2')
for line in f:
    line = line.strip()
    print(line)

f = open(path,encoding='latin2')
for line in f:
    line = line.strip()
    print(line.encode('unicode_escape'))

print(ord('ń'))

nacute = '\u0144'
print(nacute)

print(nacute.encode('utf8'))

import unicodedata
lines = open(path,encoding='latin2').readlines()
line = lines[2]
print(line.encode('unicode_escape'))
for c in line:
    if ord(c) > 127:
        print('{} U+{:04x} {}'.format(c.encode('utf8'),ord(c),unicodedata.name(c)))

print(line)
print(line.find('zosta\u0142y'))
line = line.lower()
print(line)
print(line.encode('unicode_escape'))
import re	      
m = re.search('\u015b\w*',line)
print(m.group())
print(word_tokenize(line))

#   3.4 使用正则表达式检测词组搭配

wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]

print([w for w in wordlist if re.search('ed$', w)][:50])

print([w for w in wordlist if re.search('^..j..t..$',w)][:50])

print([w for w in wordlist if re.search('^[ghi][mno][jkl][def]$',w)])

chat_words = sorted(set(w for w in nltk.corpus.nps_chat.words()))
print([w for w in chat_words if re.search('^m+i+n+e+$', w)])
print([w for w in chat_words if re.search('^[ha]+$', w)][:50])

wsj = sorted(set(nltk.corpus.treebank.words()))
print([w for w in wsj if re.search('^[0-9]+\.[0-9]+$', w)][:50])
print([w for w in wsj if re.search('^[A-Z]+\$$', w)])
print([w for w in wsj if re.search('^[0-9]{4}$', w)][:50])
print([w for w in wsj if re.search('^[0-9]+-[a-z]{3,5}$', w)][:50])
print([w for w in wsj if re.search('^[a-z]{5,}-[a-z]{2,3}-[a-z]{,6}$', w)])
print([w for w in wsj if re.search('(ed|ing)$', w)][:50])

#   3.5 正则表达式的有益应用

word = 'supercalifragilisticexpialidocious'
print(re.findall(r'[aeiou]',word))

wsj = sorted(set(nltk.corpus.treebank.words()))
fd = nltk.FreqDist(vs for word in wsj
		       for vs in re.findall(r'[aeiou]{2,}',word))
print(fd.most_common(12))

print([int(n) for n in re.findall(r'[0-9]{2,4}','2009-12-31')])

regexp = r'^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]'
def compress(word):
    pieces = re.findall(regexp,word)
    return ''.join(pieces)
english_udhr = nltk.corpus.udhr.words('English-Latin1')
print(nltk.tokenwrap(compress(w) for w in english_udhr[:75]))

rotokas_words = nltk.corpus.toolbox.words('rotokas.dic')
cvs = [cv for w in rotokas_words for cv in re.findall(r'[ptksvr][aeiou]',w)]
cfd = nltk.ConditionalFreqDist(cvs)
cfd.tabulate()

cv_word_pairs = [(cv,w) for w in rotokas_words
		     for cv in re.findall(r'[ptksvr][aeiou]',w)]
cv_index = nltk.Index(cv_word_pairs)
cv_index['su']

def stem0(word):
    for suffix in ['ing','ly','ed','ious','ies','ive','es','s','ment']:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

print(re.findall(r'^.*(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing'))
print(re.findall(r'^.*(?:ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing'))
print(re.findall(r'^(.*)(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing'))
print(re.findall(r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processes'))
print(re.findall(r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$', 'language'))

def stem1(word):
    regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
    stem,suffix = re.findall(regexp,word)[0]
    return stem

from nltk.corpus import gutenberg,nps_chat
moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
print(moby.findall(r'<a>(<.*>)<man>'))

chat = nltk.Text(nps_chat.words())
print(chat.findall(r'<.*><.*><bro>'))

print(chat.findall(r'<l.*>{3,}'))

from nltk.corpus import brown
hobbies_learned = nltk.Text(brown.words(categories=['hobbies', 'learned']))
print(hobbies_learned.findall(r"<\w*> <and> <other> <\w*s>"))

#   3.6 规范化文本

raw = """DENNIS: Listen, strange women lying in ponds distributing swords
...is no basis for a system of government. Supreme executive power derives from
... a mandate from the masses, not from some farcical aquatic ceremony."""
tokens = nltk.word_tokenize(raw)

porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
print([porter.stem(t) for t in tokens])
print([lancaster.stem(t) for t in tokens])

wnl = nltk.WordNetLemmatizer()
print([wnl.lemmatize(t) for t in tokens])

#   3.7 用正则表达式为文本分词

raw = '''When I'M a Duchess,' she said to herself, (not in a very hopeful
tone though), 'I won't have any pepper in my kitchen AT ALL. Soup does very
well without--Maybe it's always pepper that makes people hot-tempered,'...'''

print(re.split(r' ',raw))
print(re.split(r'[ \t\n]+',raw))

print(re.split(r'\W+',raw))

print(re.findall(r'\w+|\S\w*',raw))

print(re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*",raw))

#   3.8 分割

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
sents = sent_tokenizer.tokenize(text)
pprint.pprint(sents[171:181])

#   3.9 格式化:从链表到字符串

print('{:6}'.format(41))
print('{:<6}'.format(41))
print('{:6}'.format('dog'))
print('{:>6}'.format('dog'))

import math
print('{:.4f}'.format(math.pi))

count,total=3205,9375
print('accuracy for {} words: {:.4%}'.format(total,count/total))

print( '{:{width}}'.format('Monty Python',width=15))

output_file = open('output.txt','w')
words = set(nltk.corpus.genesis.words('english-kjv.txt'))
for word in sorted(words):
    print(word,file=output_file)

print(str(len(words)),file=output_file)

saying = ['After', 'all', 'is', 'said', 'and', 'done', ',','more', 'is', 'said', 'than', 'done', '.']
from textwrap import fill
format = '%s (%d),'
pieces = [format % (word,len(word)) for word in saying]
output = ' '.join(pieces)
wrapped = fill(output)
print(wrapped)
