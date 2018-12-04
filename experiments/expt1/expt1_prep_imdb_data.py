'''Prepare IMDB sentiment dataset for Experiment 1

This script decodes and reformats the IMDB movie reviews sentiment 
dataset and writes the reformatted version to a .csv file. The 
resulting rectangular format makes the data easier to experiment with, 
and makes it possible for a user to actually read/inspect the reviews 
while experimenting with them. Metadata fields about the length (in words)
of each review are also calculated and included in the output. 

The output file `imdb_decoded.csv` has 50k rows and the following fields: 

  - `subset`: indicates train/test status (as in `keras.datasets.imdb`)
  - `length`: number of words in the review 
  - `length_bin`: length-quartile of the review (with this dataset)
  - `label`: binarized sentiment of the review (1=positive, 0=negative) 
  - `text`: the text of the review, preprocessed in the following way:
      - lowercased
      - most punctuation removed (apostrophes, quotes preserved)
      - TODO -- truncated to ??? characters


NOTE: The IMDB dataset ships with Keras's `keras.datasets` module, and 
can be accessed by calling `keras.datasets.imdb.load_data()`. However, 
in this version of the data, reviews have already been integer-encoded, 
which makes it impossible for the user to peruse the actual review texts. 
This script converts the IMDB reviews back to strings of (preprocessed) 
English text, and adds a couple of useful meta-data fields.


TODO: 
  - write argparse CLI w just a couple options 
  - modify refs to this script as appropriate (bc cli)
  - shuffle data w frozen rng seed before write 
  - consider getting the *really* original version of this dataset 
  - consider including identifiers for each review (for introspection etc.)
  - ...
'''

import re
import warnings

from typing import List

from numpy import quantile
from pandas import DataFrame

from keras.datasets import imdb



outfile = '../data/imdb_decoded.csv'

# load data (each review is encoded as a sequence of integer indices) 
(train_data, train_labels), (test_data, test_labels) = imdb.load_data()

# dict mapping words to their indices 
word_index = imdb.get_word_index()

# dict mapping indices to their words 
index_to_word = dict([(value, key) for (key, value) in word_index.items()])

# func to decode a single review with `index_to_word` (adapted from DLwP)
# a review is a list of integer indices
# offset of i-3 is for reserved chars/idxs -- see DLwP, p73 
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

