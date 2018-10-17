'''
this script decodes the imdb movie reviews data in `keras.datasets.imdb`

they ship coded as lists of integers, but here we convert them back 
to their original (preprocessed) texts. we adapt the strategy from 
DLwP, page 73 for decoding reviews. 

the dict returned by `imdb.get_word_index()` is a mapping from words 
to their indices. so we extract the keys from the index lists, and 
then join them together with ' ' to recover the reviews. after that, 
we create a pandas.DataFrame holding the review, the associated 
label (positive/negative), and a column indicating whether the 
review came as part of the test set or the train set. 

the decoded reviews are saved to 'data/imdb_decoded.csv'
'''

import re
import warnings

from typing import List

from numpy import quantile
from pandas import DataFrame

from keras.datasets import imdb



outfile = '../data/imdb_decoded.csv'

# load data 
(train_data, train_labels), (test_data, test_labels) = imdb.load_data()

# dict mapping words to their indices 
word_index = imdb.get_word_index()

# dict mapping indices to their words 
index_to_word = dict([(value, key) for (key, value) in word_index.items()])

# func to decode a single review with `index_to_word`
# (a review is a list of integer indices)
# (offset of i-3 is for [TODO: special symbols in idx 0:2?!])
def decode_review(idx_list: List[int]):
  out = ' '.join([index_to_word.get(i-3, '<>') for i in idx_list])
  return re.sub(r'^<> ', '', out)



# decode the train texts and the test texts 
train_texts = [decode_review(review) for review in train_data]
test_texts = [decode_review(review) for review in test_data]

# encode the labels as lists (not np.array's)
train_labels = list(train_labels)
test_labels = list(test_labels)

# get length of each text in number of words (could use idxs instead)
train_lens = [len(review.split(' ')) for review in train_texts]
test_lens = [len(review.split(' ')) for review in test_texts]

# check that corresponding slices are of same length 
assert len(train_texts) == len(train_labels) == len(train_lens)
assert len(test_texts) == len(test_labels) == len(test_lens)

# create output table (will add length bin in next step)
dat = DataFrame({
  'text': train_texts + test_texts, 
  'label': train_labels + test_labels,
  'length': train_lens + test_lens, 
  'subset': ['train' for _ in range(len(train_texts))] + 
            ['test' for _ in range(len(test_texts))]})



# func to assign a quartile to a length given boundaries 
def assign_quantile(value: int, bin_boundaries: List[int]) -> int:
  for idx, boundary in enumerate(bin_boundaries):
    if value <= boundary:
      return idx
  raise ValueError('value above highest bin boundary!')

# find quartile boundaries for text length 
bin_boundaries = [int(quantile(dat.length, q)) for q in [.25, .5, .75, 1]]

# assign length quartiles to each row 
dat['length_bin'] = [assign_quantile(n, bin_boundaries) for n in dat.length]



# rearrange columns 
dat = dat[['subset','length','length_bin','label','text']]

# print summary and write data to `outfile` 
print(f'writing imdb data w dims {dat.shape} to file {outfile}...')

dat.to_csv(outfile, index=False)

