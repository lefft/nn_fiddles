
'''
movie review sentiment datasets: 
  1. imdb data that ships w keras/tensorflow (diff ones??)
      - from tensorflow.keras.datasets import imdb
  2. kaggle rotten tomatoes dataset
      - https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data
  3. learning word vecs for sentiment paper (same?): 
      - http://ai.stanford.edu/~amaas/data/sentiment/ (84mb zip, 221mb unz)
      - http://www.cs.cornell.edu/people/pabo/movie-review-data/ (smaller...)
      - paper: http://www.aclweb.org/anthology/P/P11/P11-1015.pdf
  4. data source from sentiment trees paper:
      - https://nlp.stanford.edu/sentiment/code.html
  5. umass sentiment corpora
      - https://semanticsarchive.net/Archive/jQ0ZGZiM/readme.html

need to decide which one to use for experiment 1. 

to do this, for each data source, will identify: 
    i. outvar type (binary, ordered cat, etc.) 
   ii. text format (e.g. raw, preprocessed, vectorized) 
  iii. num data points 
   iv. length distro 
    v. pre-split into test/train/val? 
   vi. used in other research? 




list of papers using dataset from 3.:
  - http://www.cs.cornell.edu/people/pabo/movie-review-data/otherexperiments.html

other relevant resources:
  - http://sentiment.christopherpotts.net/
  - http://sentiment.christopherpotts.net/lingstruc.html
  - http://sentiment.christopherpotts.net/code-data/happyfuntokenizing.py
  - http://help.sentiment140.com/for-students
  - https://web.stanford.edu/~cgpotts/talks/potts-starsem2018-slides.pdf



'''

l = ['elem1', 'elem2', 'elem3']
for x, y in enumerate(l):
  print(f'x = `{x}` (type `{type(x)}`)', 
        f'y = `{y}` (type `{type(y)}`)', '', sep='\n')


### 1. imdb data that ships w keras ------------------------------------------
import tensorflow as tf

# load imdb sentiment data 
imdb = tf.keras.datasets.imdb

# ys: each a binary label, xs: each a review coded as int array
(x_train, y_train), (x_test, y_test) = imdb.load_data()

print(f'train n: {len(y_train)}, test n: {len(y_test)}')
print(y_train[:10], np.unique(y_train, return_counts=True), sep='\n')


# dict associating int indices w words 
wdict = imdb.get_word_index()
# see some word/index pairs: list(wdict.items())[:10]

## funcs for translating between words and indices 
# get a word's numeric index 
def w_to_idx(w):
  return wdict[w]

# get an index's corresponding word 
def idx_to_w(widx):
  return [w for w, idx in wdict.items() if idx == widx][0]


# example word indices and words 
widxs = [2289, 70691, 10092]
words = ['woody', 'boosh', 'yikes']

# get a word from idx, or vice versa 
print('idx', w_to_idx('woody'), 'is word', idx_to_w(2289))

# get a list of words from list of idxs, or vice versa 
print('words for `widxs`: ', [idx_to_w(idx) for idx in widxs])
print('idxs  for `words`: ', [w_to_idx(w) for w in words])

# translate init segment of a review from idxs to words 
# [note that x inputs already just bags of lowercase words]
print('first five word indices:', x[:5])
print('word translations:', [idx_to_w(idx) for idx in x[:5]])







### 2. kaggle rotten tomatoes dataset ----------------------------------------
# data page: https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data


### 3. learning word vecs for sentiment paper --------------------------------
# paper: http://www.aclweb.org/anthology/P/P11/P11-1015.pdf
# data page: http://ai.stanford.edu/~amaas/data/sentiment/ 
# data page: http://www.cs.cornell.edu/people/pabo/movie-review-data/
# 
# NOTE: in stanford version, urls shd have sfx 'reviews', not 'usercomments'
# SEE README IN '~/downloads/aclImdb/README.md'


### 4. data source from sentiment trees paper -------------------------------
# data page: https://nlp.stanford.edu/sentiment/code.html 

### 5. umass sentiment corpora ----------------------------------------------
# data page: https://semanticsarchive.net/Archive/jQ0ZGZiM/readme.html
