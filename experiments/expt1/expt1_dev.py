'''Experiment 1 (simplified draft)

Question: how does average text length affect binary classification accuracy 
          for different classes of models? 

Classes of models considered: 
  - Type A: Naive Bayes, SVMs, Logistic Regression 
  - Type B: Feed-forward neural networks 
  - Type C: Recurrent neural networks 

Notes: 
  - all classifiers should use tuned hyper-parameters (grid search elsewhere)
  - different models require different transformation of input texts 
  - for each length-subset, text preprocessing steps are held constant 
  - ... 

TODO: 
  - fill out typeC classifier class 
  - need to get hypers integrated into output df 
  - want to get optimized hypers before running this 
  - take stock of text prep hypers, abstract over (include in grid searches?)
  - write argparse CLI that takes json of hypers as well as data and clf names
  - integrate this into postprocessing func(s):
    clf_names = {'clfA': clfclassA.__name__, 'clfB': clfclassB.__name__}
  - ... 
'''

import re

import keras
import pandas as pd

import sklearn.svm
import sklearn.naive_bayes

from functools import reduce

# TODO: decide if this is even useful or no?!?! (if not, eliminate altogether)
# classifier classes (A = sklearn interface, B = keras sequential interface)
from modules.expt1_classes import TypeA, TypeB

# TODO: decide if this is even useful or no?!?! (if not, eliminate altogether)
# (cd just use keras Tokenizer meths + pad_sequences() instead...) 
from modules.expt1_util import DTMize_factory, docpadder_factory
from modules.expt1_util import calculate_binary_clf_metrics
from modules.expt1_util import prob_to_binary


outfile = 'results/prelim_results-MNBvsFFNN-2018-11-20.csv'



### 1. load + prep data ------------------------------------------------------
imdb_data_fname = 'data/imdb_decoded.csv'

imdb_data = pd.read_csv(imdb_data_fname)

# split data into five subsets, defined by length quartiles 
imdb_subsets = {lbin: imdb_data[imdb_data.length_bin==lbin] 
                for lbin in sorted(imdb_data.length_bin.unique())}


# check proportions of total for each length bin + train/test 
# (they should all be close to the same size)
{**imdb_data.length_bin.value_counts(normalize=True)}
{**imdb_data.subset.value_counts(normalize=True)}





### 2. set global (held constant across fits) and clf-specific params --------

# global text preprocessing hypers (not all are relevant for all fits) 
maxlen = 200
vocab_limit = 10000


# specify the class of each classifier 
clfclassA = sklearn.naive_bayes.MultinomialNB
clfclassB = keras.models.Sequential


# set params for sklearn classifier ("typeA")
# MNB defaults are (1.0, True) 
hypersA = {'alpha': .9, 'fit_prior': True}


# set params for neural net taking DTM features ("typeB")
hypersB = {'config': {'optimizer': 'rmsprop', 
                      'loss': 'binary_crossentropy', 'metrics': ['accuracy']},
           'train': {'valset_prop': .25, 'epochs': 5, 'batch_size': 256}}

# for neural nets, also need to supply layer description + params
layersB = [keras.layers.Dense, keras.layers.Dense, keras.layers.Dense]
layersB_kwargs = [
  # params for the input layer ("layer 0")
  {'units': 16, 'activation': 'relu', 'input_shape': (vocab_limit, )}, 
  # params for the hidden layer ("layer 1")
  {'units': 16, 'activation': 'relu'}, 
  # params for the output layer ("layer 2")
  {'units': 1, 'activation': 'sigmoid'}]



# set params for feature-learning neural net ("typeC")
# TODO!!! # TODO!!! # TODO!!! # TODO!!! # TODO!!! # TODO!!!
# TODO!!! # TODO!!! # TODO!!! # TODO!!! # TODO!!! # TODO!!!
# TODO!!! # TODO!!! # TODO!!! # TODO!!! # TODO!!! # TODO!!!
## clfC = models.Sequential
## hypersC = {'config': {}, 'train': {}}
## layersC = []
## layersC_kwargs = []




### 3. train models over length subsets + generate preds ---------------------
length_bins = [0, 1, 2, 3]

results = {}

# for each subset defined by length...  
for lbin in length_bins:
  
  print(f'\n\n*** working on {lbin}th length quartile... ***\n')
  
  # carve out the subset of the data defined by `lbin` 
  data = imdb_subsets[lbin]
  train, test = data[data.subset=='train'], data[data.subset=='test']
  
  txt_train, txt_test = [*train.text], [*test.text]
  y_train, y_test = [*train.label], [*test.label]
  
  print(f'train docs: {len(train)}, test docs: {len(test)}')
  
  # create a docs-to-DTM transformer (grab the word-idx mapping too)
  DTMizer, word_idx = DTMize_factory(
    txt_train, vocab_limit=vocab_limit, return_word_idx=True)
  
  # create a docs-to-padded-int-seqs transformer (word-idx same as above)
  docpadder = docpadder_factory(txt_train, vocab_limit=vocab_limit)
    
  # TODO: add 'emb_token_matrix' as a text format! (need a nice util) 
  # TODO: add loop over keys of x_train/x_test 
  # TODO: integrate binary and tfidf mode DTMs too! 
  #          'binary_dtm': DTMizer(txt_train, mode='binary')
  #          'tfidf_dtm': DTMizer(txt_train, mode='tfidf')
  x_train = {
    'count_dtm': DTMizer(txt_train, mode='count'),
    'padded_tokens': docpadder(txt_train, out_length=maxlen)
  }
  x_test = {
    'count_dtm': DTMizer(txt_test, mode='count'),
    'padded_tokens': docpadder(txt_test, out_length=maxlen)
  }
  
  
  # instantiate and train typeA classifier 
  clfA = TypeA(clfclassA, **hypersA)
  clfA.train(x_train['count_dtm'], y_train)
  
  # instantiate, compile, and train typeB classifier 
  clfB = TypeB(clfclassB, layersB, layersB_kwargs)
  clfB.compile_model(**hypersB['config'])
  clfB.train(x_train['count_dtm'], y_train, **hypersB['train'])
  
  
  # generate model predictions on test data 
  predsA = clfA.predict(x_test['count_dtm'])
  predsB = clfB.predict(x_test['count_dtm'], pred_flattener=prob_to_binary)
  
  # calculate, print, and store performance metrics 
  metricsA = calculate_binary_clf_metrics(y_test, predsA)
  metricsB = calculate_binary_clf_metrics(y_test, predsB)
  
  print(f'results for type-A classifier:\n{metricsA}')
  print(f'results for type-B classifier:\n{metricsB}')
  
  results[f'clfA_bin{lbin}'] = {'metrics': metricsA, 'meta': clfA.clf_info}
  results[f'clfB_bin{lbin}'] = {'metrics': metricsB, 'meta': clfB.layers_info}
  
  print(f'\n\n*** finished {lbin+1}th length quartile ***\n')





### 4. postprocess the results and write to file -----------------------------
def results_dict_to_df(results, metric_name):
  # TODO: want option to name columns (for long-format later) 
  res = {key: val['metrics'][metric_name] for key, val in results.items()}
  res_df = pd.DataFrame(res, index=[metric_name]).T
  res_df.index.name = 'id'
  res_df.reset_index(inplace=True)
  return res_df

def postprocess_results(results, metric_names):
  # TODO: generalize/improve this(?)
  res_df_list = [results_dict_to_df(results, m) for m in metric_names]
  metrics_df = reduce(lambda l, r: pd.merge(l, r, on='id'), res_df_list)
  metrics_df['clf'] = [re.sub('_bin\\d', '', val) for val in metrics_df.id]
  metrics_df['lbin'] = [re.sub('clf[AB]_', '', val) for val in metrics_df.id]
  metrics_df = metrics_df[['clf', 'lbin'] + metric_names]
  return metrics_df


metric_names = ['f1', 'precision', 'recall', 'accuracy']
res = postprocess_results(results, metric_names)

print(res)
res.to_csv(outfile, index=False)

