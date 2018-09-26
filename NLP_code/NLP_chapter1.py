# Filename: NLP_chapter1.py

#1   Computing with Language: Texts and Words

from nltk.book import *

text1
text2

text1.concordance('monstrous')

text1.similar('monstrous')
text2.similar('monstrous')

text2.common_contexts(['monstrous','very'])

text4.dispersion_plot(['citizens','democracy','freedom','duties','America'])

print(len(text3))

print(sorted(set(text3))[:50])
print(len(set(text3)))

print(len(set(text3))/len(text3))

text3.count('smote')
print(100*text4.count('a')/len(text4))

def lexical_diversity(text):
    return len(set(text))/len(text)

def percentage(count,total):
    return 100*count/total

lexical_diversity(text3)
lexical_diversity(text5)
percentage(4,5)
percentage(text4.count('a'),len(text4))

sent1 = ['Call','me','Ishmael','.']

print(sent1)
print(len(sent1))
print(lexical_diversity(sent1))

print(sent2)
print(sent3)

print(['Monty','Python']+['and','the','Holy','Grail'])

print(sent4+sent1)

sent1.append('Some')
print(sent1)

print(text4[173])

print(text4.index('awaken'))

text5[16715:16735]
text6[1600:1625]

sent = ['word1', 'word2', 'word3', 'word4', 'word5',
        'word6', 'word7', 'word8', 'word9', 'word10']
print(sent[0])
print(sent[9])

print(sent[5:8])
print(sent[5])
print(sent[6])
print(sent[7])

print(sent[:3])
print(text2[141525:])

sent[0] = 'First'
sent[9] = 'Last'
print(len(sent))
sent[1:9] = ['Second','Third']
print(sent)

#2   A Closer Look at Python: Texts as Lists of Words

sent1 = ['Call', 'me', 'Ishmael', '.']

my_sent = ['Bravely', 'bold', 'Sir', 'Robin', ',', 'rode',
           'forth', 'from', 'Camelot', '.']
noun_phrase = my_sent[1:4]
print(noun_phrase)
wOrDs = sorted(noun_phrase)
print(wOrDs)

vocab = set(text1)
vocab_size = len(vocab)
print(vocab_size)

name = 'Monty'
print(name[0])
print(name[:4])

print(name*2)
print(name+'!')

' '.join(['Monty','Python'])
'Monty Python'.split()

#3   Computing with Language: Simple Statistics

saying = ['After', 'all', 'is', 'said', 'and', 'done',
          'more', 'is', 'said', 'than', 'done']
tokens = set(saying)
tokens = sorted(tokens)
tokens[-2:]

fdist1 = FreqDist(text1)
print(fdist1)
fdist1.most_common(50)
fdist1['whale']
      
V = set(text1)
long_words = [w for w in V if len(w) > 15]
sorted(long_words)

fdist5 = FreqDist(text5)
sorted(w for w in set(text5) if len(w) > 7 and fdist5[w] > 7)

list(bigrams(['more', 'is', 'said', 'than', 'done']))

text4.collocations()
text8.collocations()

print([len(w) for w in text1][:50])
fdist = FreqDist(len(w) for w in text1)
print(fdist)
fdist

fdist.most_common()
fdist.max()
fdist[3]
fdist.freq(3)

#4   Back to Python: Making Decisions and Taking Control

print(sent7)

print([w for w in sent7 if len(w) < 4])
print([w for w in sent7 if len(w) <= 4])
print([w for w in sent7 if len(w) == 4])
print([w for w in sent7 if len(w) != 4])

sorted(w for w in set(text1) if w.endswith('ableness'))
sorted(term for term in set(text4) if 'gnt' in term)
sorted(item for item in set(text6) if item.istitle())
sorted(item for item in set(sent7) if item.isdigit())

sorted(w for w in set(text7) if '-' in w and 'index' in w)
sorted(wd for wd in set(text3) if wd.istitle() and len(wd) > 10)
sorted(w for w in set(sent7) if not w.islower())
sorted(t for t in set(text2) if 'cie' in t or 'cei' in t)

print([len(w) for w in text1][:50])
[1, 4, 4, 2, 6, 8, 4, 1, 9, 1, 1, 8, 2, 1, 4, 11, 5, 2, 1, 7, 6, 1, 3, 4, 5, 2, ...]
print([w.upper() for w in text1][:50])

len(text1)
len(set(text1))
len(set(word.lower() for word in text1))

len(set(word.lower() for word in text1 if word.isalpha()))

word = 'cat'
if len(word)<5:
    print('word length is less than 5')

if len(word)>=5:
    print('word length is greater than or equal to 5')

for word in ['Call','me','Ishmael','.']:
    print(word)

sent1 = ['Call', 'me', 'Ishmael', '.']
for xyzzy in sent1:
    if xyzzy.endswith('l'):
        print(xyzzy)

for token in sent1:
    if token.islower():
        print(token, 'is a lowercase word')
    elif token.istitle():
        print(token, 'is a titlecase word')
    else:
        print(token, 'is punctuation')

tricky = sorted(w for w in set(text2) if 'cie' in w or 'cei' in w)
for word in tricky[:50]:
    print(word, end=' ')
