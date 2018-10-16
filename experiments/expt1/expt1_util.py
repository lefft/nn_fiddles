'''
as of oct09, just contains a couple utils for transforming 
a set of texts into a binary bag-of-words matrix 

this can be used to prep text data for neural net models. 

next step: use this in `expt1_keras_models.py`


should still investigate: 
  - is this totally necessary?? 
  - can a reggie dtm be used?? 
  - do we lose info from using binary instead of counts?? 
  - ... 

TODO, oct11: 
  - [ ] rewrite func for corpus ~~> dtm transform
  - [ ] want params for `mode`, vocab size, preprocessing, etc. 
  - [ ] maybe params for int encode or not?! idk... 
  - [ ] write useful docstrings throughout 
  - [ ] func to preprocess text (lowercase, remove punct, etc.)
  - [ ] want to have a set of nice viz functions 
  - [ ] ... 

'''


from keras.preprocessing.text import Tokenizer


### TODO: START HERE!!! THEN WRITE DOCSTRING + TEST 
def docs_to_dtm(docs, mode='binary', num_words=None):
  '''
  transform a set of documents (corpus) into a document-term matrix. 
  
  TODO: 
    - this needs to have vocab in internal state, so that train/test 
      splits can be done w/o having to use different vocabs, etc. ...
    - so this shd prob be a wrapper class just like the clf's in expt1.ipynb
    - implement a callable `preprocessor` param for e.g. lowercasing 
      , preprocessor=None
    - also want to be able to query the word index to see term-int mappings 
    - ... 
  '''
  # instantiate Tokenizer class (`num_words` to restrict vocab size)
  tokenizer = Tokenizer(num_words=num_words)
  
  # extract vocab and count words (makes several attrs available) 
  tokenizer.fit_on_texts(docs)
  
  # integer encode the documents 
  docs_int_encoded = tokenizer.texts_to_sequences(docs)
  
  # transform encoded docs into a DTM (default is binary)
  # `mode` can be one of "binary", "count", "tfidf", "freq"
  dtm = tokenizer.sequences_to_matrix(docs_int_encoded, mode=mode)
  
  return dtm












###### BELOW HERE IS PRE-OCT11 -- WHEN ABOVE IS READY, ELIMINATE THIS 
import numpy as np
from typing import List
from keras.preprocessing.text import Tokenizer


def vectorize_sequences(sequences, dim):
  '''
  taken from chap3 dlwp, see for motivation 
  TODO: 
    - already a built-in implementation in keras?! 
    - write func for vectorizing labels too 
  '''
  results = np.zeros((len(sequences), dim))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1.
  return results

def docs_to_binary_bow(docs: List[str], dim: int):
  '''
  take a corpus and encode it as a binary bag of words matrix 
  see chap3 dlwp for motivation
  see also `scratch.py` for example usage (until we put exx in docstring)
  '''
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(docs)
  idx_docs = tokenizer.texts_to_sequences(docs)
  idx_docs_vectors = vectorize_sequences(idx_docs, dim)
  return idx_docs_vectors

