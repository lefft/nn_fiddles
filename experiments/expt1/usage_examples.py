'''
Illustrate usage of wrappers in `expt1_util.py` to train 
classifiers from a template. The utilities are designed for
training and tweaking many different classifiers at once, but 
usage for individual fits is show here. See expt1.py for 
example of training many classifiers using "templates". 

Contents:
  1. Multinomial Naive Bayes (sklearn.naive_bayes.MultinomialNB)
      - requires doc-term feature matrix 
      - use quick_vectorize() for feature extraction 
      - read in hyper params from json file 
  2. Logistic Regression (sklearn.linear_model.LogisticRegression)
      - requires doc-term feature matrix 
      - use quick_vectorize() for feature extraction 
      - read in hyper params from json file 
  3. Feed-forward neural net (keras.models.Sequential, keras.layers.Dense)
      - requires vectorized docs (equivalent to but diff format from DTM)
      - use quick_dtmize() for feature extraction 
      - specify params in format illustrated below 
  4. LSTM neural net (keras.models.Sequential, keras.layers.{Embedding, LSTM})
      - requires padded token sequence inputs (int-encoded)
      - use quick_docpad() for feature extraction 
      - specify params in format illustrated below 


If you run this as a script, you should see something like the 
following at the end of the console output:

  naive bayes results:
    >> {'f1': 0.86, 'accuracy': 0.85, 'precision': 0.87, 'recall': 0.84}
  logistic regression results:
    >> {'f1': 0.88, 'accuracy': 0.87, 'precision': 0.87, 'recall': 0.88}
  feedfwd neural net results:
    >> {'f1': 0.734, 'accuracy': 0.758, 'precision': 0.865, 'recall': 0.637}
  LSTM neural net results:
    >> {'f1': 0.82, 'accuracy': 0.805, 'precision': 0.793, 'recall': 0.848}


If you execute this interactively in REPL, you should be able to 
produce something like the following after running everything: 

  >>> print(mnb_results)
  ## {'f1': 0.86, 'accuracy': 0.85, 'precision': 0.87, 'recall': 0.84}
  >>> print(lgr_results)
  ## {'f1': 0.88, 'accuracy': 0.87, 'precision': 0.87, 'recall': 0.88}
  >>> print(nn1_results)
  ## {'f1': 0.835, 'accuracy': 0.807, 'precision': 0.757, 'recall': 0.932}
  >>> print(nn4_results)
  ## {'f1': 0.813, 'accuracy': 0.81, 'precision': 0.841, 'recall': 0.787}

'''

import json

from functools import partial

import pandas as pd

import keras.layers
import keras.models

import sklearn.naive_bayes
import sklearn.linear_model


from expt1_util import get_imdb_subset
from expt1_util import get_params_subset

from expt1_util import quick_vectorize
from expt1_util import quick_dtmize
from expt1_util import quick_docpad

from expt1_util import train_sklearn
from expt1_util import train_keras

from expt1_util import quick_clfreport




### in/out files -------------------------------------------------------------
imdb_data_fname = 'data/imdb_decoded.csv'
hypers_file = 'sklearn_tuned_hypers.json'




### load/prep data + get hypers ----------------------------------------------
imdb_data = pd.read_csv(imdb_data_fname)

# split into train and test (any split is fine -- using short reviews here)
train = get_imdb_subset(imdb_data, subset='train', lbin=0)
test = get_imdb_subset(imdb_data, subset='test', lbin=0)

# # or to use the whole train and test subsets, you can use this:
# train = imdb_data[imdb_data.subset=='train']
# test = imdb_data[imdb_data.subset=='test']


with open(hypers_file, 'r') as f:
  # load optimized sklearn hypers 
  hypers = json.load(f)





### 1. MultinomialNB #####################################################
mnb_class = sklearn.naive_bayes.MultinomialNB
mnb_hypers = get_params_subset(hypers, clf_key='MultinomialNB')

# construct features for sklearn API 
train_dtm, test_dtm = quick_vectorize(train.text, test.text)

# bind train data to the train_sklearn function (defined above)
train_sklearn_partial = partial(train_sklearn, Xs=train_dtm, ys=train.label)

# call train func for each model to get predict func, apply that to test_dtm
mnb_preds = train_sklearn_partial(mnb_class, mnb_hypers)(test_dtm)

# generate lil classification report 
mnb_results = quick_clfreport(test.label, mnb_preds, 2)

print(mnb_results)
## {'f1': 0.86, 'accuracy': 0.85, 'precision': 0.87, 'recall': 0.84}





### 2. LogisticRegression ################################################
lgr_class = sklearn.linear_model.LogisticRegression
lgr_hypers = get_params_subset(hypers, clf_key='LogisticRegression')

# construct features for sklearn API 
train_dtm, test_dtm = quick_vectorize(train.text, test.text)

# bind train data to the train_sklearn function (defined above)
train_sklearn_partial = partial(train_sklearn, Xs=train_dtm, ys=train.label)

# call train func for each model to get predict func, apply that to test_dtm
lgr_preds = train_sklearn_partial(lgr_class, lgr_hypers)(test_dtm)

# generate lil classification report 
lgr_results = quick_clfreport(test.label, lgr_preds, 2)

print(lgr_results)
## {'f1': 0.88, 'accuracy': 0.87, 'precision': 0.87, 'recall': 0.88}





### 3. Feed-forward neural net ###########################################
nn1_params = {
  # specify layer classes and their params as 2-tuples (to loop over)
  'layers': [
    (keras.layers.Dense,   # input layer 
     {'units': 16, 'activation': 'relu', 'input_shape': (1000, )}), 
    (keras.layers.Dense,   # hidden layer 
     {'units': 16, 'activation': 'relu'}),
    (keras.layers.Dense,   # output layer 
     {'units': 1, 'activation': 'sigmoid'})
  ],
  # specify config/compile params as a dict (to supply as .compile(**kwargs))
  'config': {'optimizer': 'rmsprop', 'loss': 'binary_crossentropy', 
             'metrics': ['accuracy']},
  # specify train/eval params as a dict (to supply as .fit(**kwargs)) 
  'train': {'validation_split': .25, 'epochs': 3, 'batch_size': 256}
}

# construct features for keras API (equivalent to a count DTM)
train_vecs, test_vecs, word_idx = quick_dtmize(
  train.text, test.text, vocab_limit=1000, mode='count')

# bind train data to the train_sklearn function (defined above)
train_keras_partial = partial(train_keras, 
                              clf_class=keras.models.Sequential, 
                              Xs=train_vecs, ys=train.label)

# call train func for each model to get predict func, apply that to test_dtm
nn1_preds = train_keras_partial(hyper_dict=nn1_params)(test_vecs)

# generate lil classification report 
nn1_results = quick_clfreport(test.label, nn1_preds)

print(nn1_results)
## {'f1': 0.835, 'accuracy': 0.807, 'precision': 0.757, 'recall': 0.932}





### 4. LSTM neural net ###################################################
nn4_params = {
  'layers': [
    (keras.layers.Embedding,
     {'input_dim': 1000, 'output_dim': 100}), 
    (keras.layers.LSTM,
     {'units': 100, 'dropout': .2, 'recurrent_dropout': .2}),
    (keras.layers.Dense,
     {'units': 1, 'activation': 'sigmoid'})
  ],
  'config': {'optimizer': 'adam', 'loss': 'binary_crossentropy', 
             'metrics': ['accuracy']},
  'train': {'validation_split': .25, 'epochs': 3, 'batch_size': 256}
}

# construct int sequences for keras API (for recurrent nets)
train_seqs, test_seqs, word_idx = quick_docpad(
  train.text, test.text, vocab_limit=1000, out_length=50)

# bind train data to the train_sklearn function (defined above)
train_keras_partial = partial(train_keras, 
                              clf_class=keras.models.Sequential, 
                              Xs=train_seqs, ys=train.label)

# call train func for each model to get predict func, apply to test_seqs
nn4_preds = train_keras_partial(hyper_dict=nn4_params)(test_seqs)

# generate lil classification report 
nn4_results = quick_clfreport(test.label, nn4_preds)

print(nn4_results)
## {'f1': 0.813, 'accuracy': 0.81, 'precision': 0.841, 'recall': 0.787}




### show all results at once -------------------------------------------------
print(f'\n\n~~~ done with all fits. here\'s the results ~~~\n')
print(f'naive bayes results:\n  >> {mnb_results}')
print(f'logistic regression results:\n  >> {lgr_results}')
print(f'feedfwd neural net results:\n  >> {nn1_results}')
print(f'LSTM neural net results:\n  >> {nn4_results}')







### SCRATCH AREA ##################################################
# # this works now actually 
# nn5_params = {
#   'layers': [
#     (keras.layers.Embedding,
#      {'input_dim': 1000, 'output_dim': 100}), 
#     (keras.layers.LSTM,
#      {'units': 100, 'return_sequences': True}),
#     (keras.layers.LSTM,
#      {'units': 100, 'dropout': .2, 'recurrent_dropout': .2}),
#     (keras.layers.Dense,
#      {'units': 1, 'activation': 'sigmoid'})
#   ],
#   'config': {'optimizer': 'adam', 'loss': 'binary_crossentropy', 
#              'metrics': ['accuracy']},
#   'train': {'validation_split': .25, 'epochs': 3, 'batch_size': 256}
# }
# 
# # construct int sequences for keras API (for recurrent nets)
# train_seqs, test_seqs, word_idx = quick_docpad(
#   train.text, test.text, vocab_limit=1000, out_length=50)
# 
# # bind train data to the train_sklearn function (defined above)
# train_keras_partial = partial(train_keras, 
#                               clf_class=keras.models.Sequential, 
#                               Xs=train_seqs, ys=train.label)
# 
# # call train func for each model to get predict func, apply to test_seqs
# nn5_preds = train_keras_partial(hyper_dict=nn5_params)(test_seqs)
# 
# # generate lil classification report 
# nn5_results = quick_clfreport(test.label, nn5_preds)
# 
# print(nn5_results)
# ## {'f1': 0.813, 'accuracy': 0.81, 'precision': 0.841, 'recall': 0.787}


