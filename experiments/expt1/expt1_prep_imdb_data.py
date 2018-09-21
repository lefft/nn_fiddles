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

from typing import List
from pandas import DataFrame
from keras.datasets import imdb


outfile = 'data/imdb_decoded.csv'

# load data 
(train_data, train_labels), (test_data, test_labels) = imdb.load_data()

# mapping from words to indices 
word_index = imdb.get_word_index()

# mapping from indices to words 
index_to_word = dict([(value, key) for (key, value) in word_index.items()])

# define a function to decode a single review with `index_to_word`
# (a review is a list of integer indices)
# (offset of i-3 is for [TODO: special symbols in idx 0:2?!])
def decode_review(idx_list: List[int]):
  return ' '.join([index_to_word.get(i-3, '<>') for i in idx_list])


# decode the train texts and the test texts 
train_texts = [decode_review(review) for review in train_data]
test_texts = [decode_review(review) for review in test_data]

# encode the labels as lists 
train_labels = list(train_labels)
test_labels = list(test_labels)

assert len(train_texts) == len(train_labels)
assert len(test_texts) == len(test_labels)

data = {'text': train_texts + test_texts, 
        'label': train_labels + test_labels,
        'subset': ['train' for _ in range(len(train_texts))] + 
                  ['test' for _ in range(len(test_texts))]}

df = DataFrame(data)

print(f'writing imdb data w dims {df.shape} to file {outfile}...')

df.to_csv(outfile, index=False)

