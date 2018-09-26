#   Filename:   NLP_chapter7.py

import nltk,re,pprint
from nltk.corpus import brown
from nltk.corpus import conll2000

#   7.1 信息提取

def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]

#   7.2 分块

sentence = [("the","DT"),("little","JJ"),("yellow","JJ"),
            ("dog","NN"),("barked","VBD"),("at","IN"),("the","DT"),("cat","NN")]
grammar = "NP:{<DT>?<JJ>*<NN>}"
cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
print(result)
result.draw()

grammar = r'''
NP: {<DT|PP\$>?<JJ>*<NN>}
    {<NNP>+}
    '''
cp = nltk.RegexpParser(grammar)
sentence = [("Rapunzel", "NNP"), ("let", "VBD"), ("down", "RP"),
            ("her", "PP$"), ("long", "JJ"), ("golden", "JJ"), ("hair", "NN")]
print(cp.parse(sentence))

nouns = [("money","NN"),("market","NN"),("fund","NN")]
grammar = "NP:{<NN>+}"
cp = nltk.RegexpParser(grammar)
print(cp.parse(nouns))

def find_chunks(chunk):
    cp = nltk.RegexpParser(chunk)
    for sent in brown.tagged_sents()[:20]:
        tree = cp.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == 'CHUNK':
                print(subtree)
            elif subtree.label() == 'NOUNS':
                print(subtree)

find_chunks('CHUNK: {<V.*><TO><V.*>}')
find_chunks('NOUNS: {<N.*>{4,}}')

grammar = r'''
NP:
{<.*>+}
}<VBD|IN>+{
'''
sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),
            ("dog", "NN"), ("barked", "VBD"), ("at", "IN"), ("the", "DT"), ("cat", "NN")]
cp = nltk.RegexpParser(grammar)
print(cp.parse(sentence))

#   7.3 开发和评估分块器

text = '''
he PRP B-NP
accepted VBD B-VP
the DT B-NP
position NN I-NP
of IN B-PP
vice NN B-NP
chairman NN I-NP
of IN B-PP
Carlyle NNP B-NP
Group NNP I-NP
, , O
a DT B-NP
merchant NN I-NP
banking NN I-NP
concern NN I-NP
. . O
'''
nltk.chunk.conllstr2tree(text,chunk_types=['NP']).draw()

print(conll2000.chunked_sents('train.txt')[99])

print(conll2000.chunked_sents('train.txt',chunk_types=['NP'])[99])

cp = nltk.RegexpParser('')
test_sents = conll2000.chunked_sents('test.txt',chunk_types=['NP'])
print(cp.evaluate(test_sents))

grammar = r"NP:{<[CDJNP].*>+}"
cp = nltk.RegexpParser(grammar)
print(cp.evaluate(test_sents))

class UnigramChunker(nltk.ChunkParserI):
    def __init__(self,train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data)
    def parse(self,sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos,chunktag) in tagged_pos_tags]
        conlltags = [(word,pos,chunktag) for ((word,pos),chunktag)
                     in zip(sentence,chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)

test_sents = conll2000.chunked_sents('test.txt',chunk_types=['NP'])
train_sents = conll2000.chunked_sents('train.txt',chunk_types=['NP'])
unigram_chunker = UnigramChunker(train_sents)
print(unigram_chunker.evaluate(test_sents))

postags = sorted(set(pos for sent in train_sents
                     for (word,pos) in sent.leaves()))
print(unigram_chunker.tagger.tag(postags))

class BigramChunker(nltk.ChunkParserI):
    def __init__(self,train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)
    def parse(self,sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos,chunktag) in tagged_pos_tags]
        conlltags = [(word,pos,chunktag) for ((word,pos),chunktag)
                     in zip(sentence,chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)

bigram_chunker = BigramChunker(train_sents)
print(bigram_chunker.evaluate(test_sents))

class ConsecutiveNPChunkTagger(nltk.TaggerI):
    def __init__(self,train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i,(word,tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent,i,history)
                train_set.append((featureset,tag))
                history.append(tag)
        self.classifier = nltk.NaiveBayesClassifier.train(
            train_set)
    def tag(self,sentence):
        history = []
        for i,word in enumerate(sentence):
            featureset = npchunk_features(sentence,i,history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence,history)

class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self,train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)
    def parse(self,sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)

def npchunk_features(sentence,i,history):
    word,pos = sentence[i]
    return {"pos": pos}
chunker = ConsecutiveNPChunker(train_sents)
print(chunker.evaluate(test_sents))

def npchunk_features(sentence,i,history):
    word,pos = sentence[i]
    if i == 0:
        prevword,prevpos = "<START>","<START>"
    else:
        prevword,prevpos = sentence[i-1]
    return {"pos":pos,"prevpos":prevpos}
chunker = ConsecutiveNPChunker(train_sents)
print(chunker.evaluate(test_sents))

def npchunk_features(sentence,i,history):
    word,pos = sentence[i]
    if i == 0:
        prevword,prevpos = "<START>","<START>"
    else:
        prevword,prevpos = sentence[i-1]
    return {"pos":pos,"word":word,"prevpos":prevpos}
chunker = ConsecutiveNPChunker(train_sents)
print(chunker.evaluate(test_sents))

def npchunk_features(sentence,i,history):
    word,pos = sentence[i]
    if i == 0:
        prevword,prevpos = "<START>","<START>"
    else:
        prevword,prevpos = sentence[i-1]
    if i == len(sentence)-1:
        nextword,nextpos = "<END>","<END>"
    else:
        nextword,nextpos = sentence[i+1]
    return {"pos":pos,
            "word":word,
            "prevpos":prevpos,
            "nextpos":nextpos,
            "prevpos+pos":"{}+{}".format(prevpos,pos),
            "pos+nextpos":"{}+{}".format(pos,nextpos),
            "tags-since-dt":tags_since_dt(sentence,i)}
def tags_since_dt(sentence,i):
    tags = set()
    for word,pos in sentence[:i]:
        if pos == 'DT':
            tags = set()
        else:
            tags.add(pos)
    return '+'.join(sorted(tags))
chunker = ConsecutiveNPChunker(train_sents)
print(chunker.evaluate(test_sents))

#   7.4 语言结构中的递归

grammar = r'''
NP:{<DT|JJ|NN.*>+}
PP:{<IN><NP>}
VP:{<VB.*><NP|PP|CLAUSE>+$}
CLAUSE:{<NP><VP>}
'''
cp = nltk.RegexpParser(grammar)
sentence = [("Mary", "NN"), ("saw", "VBD"), ("the", "DT"), ("cat", "NN"),
            ("sit", "VB"), ("on", "IN"), ("the", "DT"), ("mat", "NN")]
print(cp.parse(sentence))

sentence = [("John", "NNP"), ("thinks", "VBZ"), ("Mary", "NN"),
            ("saw", "VBD"), ("the", "DT"), ("cat", "NN"), ("sit", "VB"),
            ("on", "IN"), ("the", "DT"), ("mat", "NN")]
print(cp.parse(sentence))

cp = nltk.RegexpParser(grammar,loop=2)
print(cp.parse(sentence))

tree1 = nltk.Tree('NP',['Alice'])
print(tree1)
tree2 = nltk.Tree('NP',['the','rabbit'])
print(tree2)

tree3 = nltk.Tree('VP',['chased',tree2])
tree4 = nltk.Tree('S',[tree1,tree3])
print(tree4)

print(tree4[1])
print(tree4[1].label())
print(tree4.leaves())
print(tree4[1][1][1])

tree4.draw()

def traverse(t):
    try:
        t.label()
    except AttributeError:
        print(t,end=' ')
    else:
        print('(',t.label(),end=' ')
        for child in t:
            traverse(child)
        print(')',end=' ')
t = '(S (NP Alice) (VP chased (NP the rabbit)))'
traverse(t)

#   7.5 命名实体识别

sent = nltk.corpus.treebank.tagged_sents()[22]
print(nltk.ne_chunk(sent,binary=True))
print(nltk.ne_chunk(sent))

#   7.6 关系抽取

IN = re.compile(r'.*\bin\b(?!\b.+ing)')
for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
    for rel in nltk.sem.extract_rels('ORG','LOC',doc,
                                     corpus='ieer',pattern=IN):
        print(nltk.sem.rtuple(rel))

from nltk.corpus import conll2002
vnv = '''
(
is/V|
was/V|
werd/V|
wordt/V
)
.*
van/Prep
'''
VAN = re.compile(vnv,re.VERBOSE)
for doc in conll2002.chunked_sents('ned.train'):
    for r in nltk.sem.extract_rels('PER','ORG',doc,
                                   corpus='conll2002',pattern=VAN):
        print(nltk.sem.clause(r,relsym='VAN'))
        print(nltk.sem.rtuple(r,lcon=True,rcon=True))
