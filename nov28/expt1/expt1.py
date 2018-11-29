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
  - plot results! 
  - expand nn models: pre-train embeddings (transfer learning ex)
  - expand other algos: SGDClassifier or SVC or smthg else 
  - adjust keras training msgs verbosity!!! 
  - also include clf metrics over the whole dataset for each clf
  - maybe also snag train metrics?!?! (useful for interpretation)
  - integrate plotting of network graphs! [SEE DEV AREA FOR PROGRESS]
  - specify all in/out shapes, so they're shown in network graphs! 
'''

import os
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


out_dir = 'results_nov29'
if not os.path.isdir(out_dir): os.mkdir(out_dir)
outfiles = {
  'results_json': os.path.join(out_dir, 'results-nov29.json'),
  'results_csv':  os.path.join(out_dir, 'results-nov29.csv'),
  'model_params': os.path.join(out_dir, 'results-nov29-params.json')
}




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
nn4_key = 'lstm_1hidden'
nn5_key = 'lstm_2hidden'


# only consider `vocab_n`-most frequent words when constructing features 
vocab_n = 5000         # used by all nn models 
doc_maxlen = 200       # used only by sequence models 
lstm_hidden_dim = 100  # used only by lstm models 
nn_epochs = 10         # used by all nns, might want to vary it later tho


# TODO: write func so we can feed in vocab_n et al. on import from module! 
nn1_params = {
  # specify layer classes + params as 2-tuples (to loop over in train_keras())
  'layers': [
    (keras.layers.Dense,   # input layer 
     {'units': 16, 'activation': 'relu', 'input_shape': (vocab_n, )}), 
    (keras.layers.Dense,   # hidden layer 
     {'units': 16, 'activation': 'relu'}),
    (keras.layers.Dense,   # output layer 
     {'units': 1, 'activation': 'sigmoid'})
  ],
  # specify config/compile params as a dict (to supply as .compile(**kwargs))
  'config': {'optimizer': 'rmsprop', 'loss': 'binary_crossentropy', 
             'metrics': ['accuracy']},
  # specify train/eval params as a dict (to supply as .fit(**kwargs)) 
  'train': {'validation_split': .25, 'epochs': nn_epochs, 'batch_size': 256}
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
  'train': {'validation_split': .25, 'epochs': nn_epochs, 'batch_size': 256}
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
  'train': {'validation_split': .25, 'epochs': nn_epochs, 'batch_size': 256}
}
nn4_params = {
  'layers': [
    (keras.layers.Embedding,
     {'input_dim': vocab_n, 'output_dim': lstm_hidden_dim}), 
    (keras.layers.LSTM,
     {'units': lstm_hidden_dim, 'dropout': .2, 'recurrent_dropout': .2}),
    (keras.layers.Dense,
     {'units': 1, 'activation': 'sigmoid'})
  ],
  'config': {'optimizer': 'adam', 'loss': 'binary_crossentropy', 
             'metrics': ['accuracy']},
  'train': {'validation_split': .25, 'epochs': nn_epochs, 'batch_size': 256}
}
nn5_params = {
  'layers': [
    (keras.layers.Embedding,
     {'input_dim': vocab_n, 'output_dim': lstm_hidden_dim}), 
    (keras.layers.LSTM,
     {'units': lstm_hidden_dim, 'return_sequences': True}),
    (keras.layers.LSTM,
     {'units': lstm_hidden_dim, 'dropout': .2, 'recurrent_dropout': .2}),
    (keras.layers.Dense,
     {'units': 1, 'activation': 'sigmoid'})
  ],
  'config': {'optimizer': 'adam', 'loss': 'binary_crossentropy', 
             'metrics': ['accuracy']},
  'train': {'validation_split': .25, 'epochs': nn_epochs, 'batch_size': 256}
}






### run the experiment woop woop! -------------------------------------------
length_bins = [0, 1, 2, 3]

results = {}

### *** train + eval models for each sklearn classifier at each subset *** ###
for lbin in length_bins:

  print(f'\n*** working on {lbin}th length quartile (sklearn)... ***\n')
  
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



### *** train + eval models for each keras ff network at each subset *** ###
for lbin in length_bins:

  print(f'\n\n*** working on {lbin}th length quartile (keras FFNs)... ***\n')
  
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



### *** train + eval models for each keras recurrent nn at each subset *** ###
for lbin in length_bins:

  print(f'\n\n*** working on {lbin}th length quartile (keras RNNs)... ***\n')
  
  train = get_imdb_subset(imdb_data, subset='train', lbin=lbin)
  test = get_imdb_subset(imdb_data, subset='test', lbin=lbin)
  
  # construct int sequences for keras API (for recurrent nets)
  train_seqs, test_seqs, word_idx = quick_docpad(
    train.text, test.text, vocab_limit=vocab_n, out_length=doc_maxlen)
  
  # bind train data to the train_sklearn function (defined above)
  train_keras_partial = partial(train_keras, clf_class=Sequential, 
                                Xs=train_seqs, ys=train.label)
  
  # call train func for each model to get predict func, apply to test_seqs
  nn4_preds = train_keras_partial(hyper_dict=nn4_params)(test_seqs)
  nn5_preds = train_keras_partial(hyper_dict=nn4_params)(test_seqs)

  results[f'q{lbin}-{nn4_key}'] = quick_clfreport(test.label, nn4_preds)
  results[f'q{lbin}-{nn5_key}'] = quick_clfreport(test.label, nn5_preds)


print('\n\n*** ~~~ done generating performance curve data ~~~ ***')




### postprocess + write params to file ---------------------------------------
all_params = {
  # sklearn classifiers 
  mnb_key: mnb_hypers, lgr_key: lgr_hypers, 
  # keras feed-forward nets
  nn1_key: nn1_params, nn2_key: nn2_params, nn3_key: nn3_params, 
  # keras recurrent nets 
  nn4_key: nn4_params, nn5_key: nn5_params
}
# TODO: want a better sol'n than this quick hack! 
all_params[nn1_key]['layers'] = str(all_params[nn1_key]['layers'])
all_params[nn2_key]['layers'] = str(all_params[nn2_key]['layers'])
all_params[nn3_key]['layers'] = str(all_params[nn3_key]['layers'])
all_params[nn4_key]['layers'] = str(all_params[nn4_key]['layers'])
all_params[nn5_key]['layers'] = str(all_params[nn5_key]['layers'])


with open(outfiles['model_params'], 'w') as f:
  json.dump(all_params, f, indent=2)




### postprocess + write results to file --------------------------------------
with open(outfiles['results_json'], 'w') as f:
  # sort the results from each run by bin + algo, then write to json
  results_ordered = OrderedDict(sorted(results.items()))
  json.dump(results_ordered, f, indent=2)

metric_names = ['f1', 'precision', 'recall', 'accuracy']
results_df = postprocess_results(results, metric_names)
results_df.to_csv(outfiles['results_csv'], index=False)













quit()
### DEV AREA!!! ##############################################################


### save network graph of each nn's architecture -----------------------------
# TODO: INTEGRATE THIS INTO ABOVE!!! CURRENTLY DOESNT RUN ON SCRIPT EXECUTE
from keras.utils import plot_model

def plot_keras_model(clf_key, hyper_dict, out_dir):
  '''visualize the structure of a keras network
  
  # wrapper that compiles model + then calls:
  plot_model(model, to_file=outfile, dpi=300,
             show_shapes=False, show_layer_names=True, expand_nested=False)
  '''
  outfile = os.path.join(out_dir, clf_key+'_graph.png')

  nn = keras.models.Sequential()
  for layer, layer_params in hyper_dict['layers']:
    nn.add(layer(**layer_params))
  nn.compile(**hyper_dict['config'])
  
  plot_model(nn, to_file=outfile,
             show_shapes=True, show_layer_names=False)
  print(f'wrote model network graph to file: `{outfile}`')


graphs_dir = os.path.join(out_dir, 'nn_graphs')
os.mkdir(graphs_dir)
plot_keras_model(clf_key=nn1_key, hyper_dict=nn1_params, out_dir=graphs_dir)
plot_keras_model(clf_key=nn2_key, hyper_dict=nn2_params, out_dir=graphs_dir)
plot_keras_model(clf_key=nn3_key, hyper_dict=nn3_params, out_dir=graphs_dir)
plot_keras_model(clf_key=nn4_key, hyper_dict=nn4_params, out_dir=graphs_dir)
plot_keras_model(clf_key=nn5_key, hyper_dict=nn5_params, out_dir=graphs_dir)

