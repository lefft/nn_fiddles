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

'''

import numpy as np
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

