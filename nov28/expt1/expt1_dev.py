'''Experiment 1 (simplified draft)

Question: how does average text length affect binary classification accuracy 
          for different classes of models? 

Classes of models considered: 
  - Naive Bayes, Logistic Regression 
  - Feed-forward neural networks w various structures
  - TODO: Type C: Recurrent neural networks 

Notes: 
  - all classifiers should use tuned hyper-parameters (grid search elsewhere)
  - different models require different transformation of input texts 

Outline:
  1. load + prep data 
  2. load tuned hypers for each model class 
  3. train models over length subsets + generate preds
  4. postprocess the results and write to file

TODO: 
  - parse results into a csv for plotting! 
  - expand range of nn models
  - expand range of other algos 
  - adjust keras training msgs verbosity 
'''

import json

import keras
import sklearn.naive_bayes
import sklearn.linear_model

import numpy as np
import pandas as pd

from functools import partial
from collections import OrderedDict

from keras.models import Sequential
from keras.layers import Dense


# TODO: dont use * imports like this! 
from expt1_util import *


np.random.seed(6933)


### i/o filepaths ---------------------------------------------------------
imdb_data_fname = '../../experiments/expt1/data/imdb_decoded.csv'
hypers_file = '../../experiments/expt1/sklearn_tuned_hypers.json'

prelim_results_json_outfile = 'results-nov28.json'




### load data --------------------------------------------------------------
imdb_data = pd.read_csv(imdb_data_fname)




### specify clf hypers and other params -------------------------------------
with open(hypers_file, 'r') as f:
  # load optimized sklearn hypers 
  hypers = json.load(f)

# get approprate params for each sklearn classifier 
mnb_key = 'MultinomialNB'
mnb_class = sklearn.naive_bayes.MultinomialNB
mnb_hypers = get_params_subset(hypers, clf_key=mnb_key)

lgr_key = 'LogisticRegression'
lgr_class = sklearn.linear_model.LogisticRegression
lgr_hypers = get_params_subset(hypers, clf_key=lgr_key)


# set keras params for each network 
nn1_key = 'feedfwd_1hidden'
nn2_key = 'feedfwd_2hidden'
nn3_key = 'feedfwd_3hidden'
# nn4_key = 'lstm_1hidden' # TODO: add this! 

# only consider `vocab_n`-most frequent words when constructing features 
vocab_n = 1000

# TODO: write a func so we can feed in vocab_n on import from module! 
nn1_params = {
  # specify layer classes and their params as 2-tuples (to loop over)
  'layers': [
    (keras.layers.Dense,   # input layer 
     {'units': 16, 'activation': 'relu', 'input_shape': (vocab_n, )}), 
    (keras.layers.Dense,   # hidden layer 
     {'units': 16, 'activation': 'relu'}),
    (keras.layers.Dense,   # output layer 
     {'units': 1, 'activation': 'sigmoid'})
  ],
  # specify config/compile params as a dict (to supply as .compile(**kwargs))
  'config': {'optimizer': 'rmsprop', 
             'loss': 'binary_crossentropy', 
             'metrics': ['accuracy']},
  # specify train/eval params as a dict (to supply as .fit(**kwargs)) 
  'train': {'validation_split': .25, 
            'epochs': 10, 
            'batch_size': 256}
}
nn2_params = {
  'layers': [
    (keras.layers.Dense,
     {'units': 64, 'activation': 'relu', 'input_shape': (vocab_n, )}), 
    (keras.layers.Dense,
     {'units': 32, 'activation': 'relu'}),
    (keras.layers.Dense,
     {'units': 16, 'activation': 'relu'}),
    (keras.layers.Dense,
     {'units': 1, 'activation': 'sigmoid'})
  ],
  'config': {'optimizer': 'rmsprop', 'loss': 'binary_crossentropy', 
             'metrics': ['accuracy']},
  'train': {'validation_split': .25, 'epochs': 10, 'batch_size': 256}
}
nn3_params = {
  'layers': [
    (keras.layers.Dense,
     {'units': 64, 'activation': 'relu', 'input_shape': (vocab_n, )}), 
    (keras.layers.Dense,
     {'units': 32, 'activation': 'relu'}),
    (keras.layers.Dense,
     {'units': 16, 'activation': 'relu'}),
    (keras.layers.Dense,
     {'units': 16, 'activation': 'relu'}),
    (keras.layers.Dense,
     {'units': 1, 'activation': 'sigmoid'})
  ],
  'config': {'optimizer': 'rmsprop', 'loss': 'binary_crossentropy', 
             'metrics': ['accuracy']},
  'train': {'validation_split': .25, 'epochs': 10, 'batch_size': 256}
}




### run the experiment woop woop! -------------------------------------------
length_bins = [0, 1, 2, 3]

results = {}

# train and evaluate models for each sklearn classifier at each subset 
for lbin in length_bins:

  print(f'\n\n*** working on {lbin}th length quartile (sklearn)... ***\n')
  
  train = get_imdb_subset(imdb_data, subset='train', lbin=lbin)
  test = get_imdb_subset(imdb_data, subset='test', lbin=lbin)
  
  # construct features for sklearn API 
  train_dtm, test_dtm = quick_vectorize(train.text, test.text)
  
  # bind train data to the train_sklearn function (defined above)
  train_sklearn_partial = partial(train_sklearn, Xs=train_dtm, ys=train.label)
  
  # call train func for each model to get predict func, apply that to test_dtm
  mnb_preds = train_sklearn_partial(mnb_class, mnb_hypers)(test_dtm)
  lgr_preds = train_sklearn_partial(lgr_class, lgr_hypers)(test_dtm)
  
  # generate lil classification reports (holds metrics dict)
  results[f'q{lbin}-{mnb_key}'] = quick_clfreport(test.label, mnb_preds, 2)
  results[f'q{lbin}-{lgr_key}'] = quick_clfreport(test.label, lgr_preds, 2)



# train and evaluate models for each keras network at each subset 
for lbin in length_bins:

  print(f'\n\n*** working on {lbin}th length quartile (keras)... ***\n')
  
  train = get_imdb_subset(imdb_data, subset='train', lbin=lbin)
  test = get_imdb_subset(imdb_data, subset='test', lbin=lbin)
  
  # construct features for keras API 
  train_vecs, test_vecs, word_idx = quick_dtmize(train.text, test.text, 
                                                 vocab_limit=vocab_n, 
                                                 mode='count')
  
  # bind train data to the train_sklearn function (defined above)
  train_keras_partial = partial(train_keras, clf_class=Sequential, 
                                Xs=train_vecs, ys=train.label)
  
  # call train func for each model to get predict func, apply that to test_dtm
  nn1_preds = train_keras_partial(hyper_dict=nn1_params)(test_vecs)
  nn2_preds = train_keras_partial(hyper_dict=nn2_params)(test_vecs)
  nn3_preds = train_keras_partial(hyper_dict=nn3_params)(test_vecs)
  
  results[f'q{lbin}-{nn1_key}'] = quick_clfreport(test.label, nn1_preds)
  results[f'q{lbin}-{nn2_key}'] = quick_clfreport(test.label, nn2_preds)
  results[f'q{lbin}-{nn3_key}'] = quick_clfreport(test.label, nn3_preds)


print('\n\n*** ~~~ done generating performance curve data ~~~ ***')
print('\n\n*** ~~~ done generating performance curve data ~~~ ***')
print('\n\n*** ~~~ done generating performance curve data ~~~ ***')



### postprocess results -----------------------------------------------------
with open(prelim_results_json_outfile, 'w') as f:
  # sort the results from each run by bin + algo, then write to json
  results_ordered = OrderedDict(sorted(results.items()))
  json.dump(results_ordered, f, indent=2)


# TODO: flatten the dict into a df + write to csv for plotting
# TODO: flatten the dict into a df + write to csv for plotting
# TODO: flatten the dict into a df + write to csv for plotting

# # to view the results for small dev runs 
# for key, val in results.items():
#   print(f'{key}:\n  >> {val}')


