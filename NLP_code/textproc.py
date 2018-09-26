# Filename:textproc.py

import nltk

def plural(word):
    if word.endswith('y'):
        return word[:-1]+'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh','ch']:
        return word+'es'
    elif word.endswith('an'):
        return word[:-2]+'en'
    else:
        return word+'s'

def unusual_words(text):
    text_vocab=set(w.lower() for w in text if w.isalpha())
    english_vocab=set(w.lower() for w in nltk.corpus.words.words())
    unusual=text_vocab.difference(english_vocab)
    return sorted(unusual)
    
def content_fraction(text):
    stopwords=nltk.corpus.stopwords.words('english')
    content=[w for w in text if w.lower() not in stopwords]
    return len(content)/len(text)

def segment(text,segs):
    words=[]
    last = 0
    for i in range(len(segs)):
        if segs[i] == '1':
            words.append(text[last:i+1])
            last = i+1
    words.append(text[last:])
    return words

def evaluate(text,segs):
    words = segment(text,segs)
    text_size = len(words)
    lexicon_size = len(' '.join(list(set(words))))
    return text_size + lexicon_size

from random import randint
def flip(segs,pos):
    return segs[:pos] + str(1-int(segs[pos])) + segs[pos+1:]
def flip_n(segs,n):
    for i in range(n):
        segs = flip(segs,randint(0,len(segs)-1))
    return segs
def anneal(text,segs,iterations,cooling_rate):
    temperature = float(len(segs))
    while temperature > 0.5:
        best_segs,best = segs,evaluate(text,segs)
        for i in range(iterations):
            guess = flip_n(segs,round(temperature))
            score = evaluate(text,guess)
            if score < best:
                best,best_segs = score,guess
        score,segs = best,best_segs
        temperature = temperature/cooling_rate
        print(evaluate(text,segs),segment(text,segs))
    print()
    return segs

def tabulate(cfdist,words,categories):
    print('{:16}'.format('Category'),end=' ')           #column headings
    for word in words:
        print('{:>6}'.format(word),end=' ')
    print()
    for category in categories:
        print('{:16}'.format(category),end=' ')         #row heading
        for word in words:                              #for each word
            print('{:6}'.format(cfdist[category][word]),end=' ')    #print table cell
        print()                                         #end the row
