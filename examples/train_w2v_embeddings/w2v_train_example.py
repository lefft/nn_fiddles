import re
import string
import requests
import gensim



### 1. acquire mini-corpus -------------------------------------------------- 
url = 'http://lefft.xyz/stuff/posts/btc/input/unabomber-manifesto.txt'
txt = requests.get(url).text

print(txt[:200])



### 2. preprocess text ------------------------------------------------------ 
def prep_text(s):
  # TODO: remove stopwords(?) 
  exclude = set(string.punctuation)
  exclude.remove('-')
  s = ''.join(ch for ch in s if ch not in exclude)
  s = re.sub('[LR]QUOTE', '', s)
  s = re.sub('\n', ' ', s)
  s = re.sub(' +', ' ', s)
  s = s.strip()
  s = s.lower() 
  return s


sentence_tokens = []

for sentence in txt.split('.'):
  sentence = prep_text(sentence)
  sentence_tokens.append(sentence.split(' '))

sentence_tokens = [l for l in sentence_tokens if len(l) > 2]

print(sentence_tokens[:3]) 
print(len(sentence_tokens))
print(round(sum([len(l) for l in sentence_tokens]) / len(sentence_tokens)))



### 3. 'train' embeddings + play around ------------------------------------- 
# NOTE: EITHER SOMETHING OFF OR ELSE JUST GARBAGE W THIS SMALL AMOUNT OF TXT
#       (PROBABLY JUST THE LATTER BUT ALSO SEEMS WAY FASTER THAN EXPECTED...)
model = gensim.models.Word2Vec(sentence_tokens, min_count=2)

# weird, **everything** coming out as unreasonably similar
model.similarity('industrial', 'society')
model.similarity('man', 'woman')
model.similarity('man', 'men')

model.most_similar('man')

model.doesnt_match('man woman child dog'.split(' ')) # woman :(  

model.most_similar(positive=['woman', 'man'], negative=['child'], topn=1)
# [('engineering', 0.994199275970459)]


