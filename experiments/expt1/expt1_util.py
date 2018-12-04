import os, re

import pandas as pd

from functools import reduce

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix

from keras.utils import plot_model
from keras.models import Sequential



### train + predict wrappers ---------------------------------------------
def train_sklearn(clf_class, hyper_dict, Xs, ys):
  '''
  train a classifier of clf_class with hyper_dict params on 
  inputs Xs and targets ys. return the predict method of the fit.
  '''
  clf = clf_class(**hyper_dict)
  clf.fit(Xs, ys)
  return clf.predict


def train_keras(clf_class, hyper_dict, Xs, ys):
  '''
  train a keras sequential mode (clf_class) with params in 
  hyper_dict (must have top-level keys 'layers', 'config', 'train').
  return the predict_classes method of the fit. 
  hyper_dict['layers'] is a list of 2-tuples, w layer class + params.

  NOTE: VERBOSE PARAM NOT WORKING! 
  '''
  clf = clf_class()
  for layer, layer_params in hyper_dict['layers']:
    clf.add(layer(**layer_params))
  clf.compile(**hyper_dict['config'])
  clf.fit(Xs, ys, **hyper_dict['train'])
  return clf.predict_classes



#### load + prep data utils --------------------------------------
def get_imdb_subset(dat, subset, lbin):
  '''quickly access relevant subsets of the imdb data'''
  out = dat[(dat.subset==subset) & (dat.length_bin==lbin)]
  # print(f'retrieved {out.shape}-dim {lbin}th quartile of IMDB {subset}')
  return out



def get_params_subset(hypers_dict, clf_key, prefix='clf__'):
  '''assumes `hypers_dict` has structure of data in sklearn_tuned_hypers.json'''
  params = hypers_dict[clf_key]['best_params']
  out = {key.replace(prefix, ''): val
         for key, val in params.items() if key.startswith(prefix)}
  return out




def quick_vectorize(train_text, test_text, hypers={}):
  '''vectorize train and test text properly with one function call'''
  Vectorizer = CountVectorizer(**hypers)
  train_dtm = Vectorizer.fit_transform(train_text)
  test_dtm = Vectorizer.transform(test_text)
  return train_dtm, test_dtm



def quick_dtmize(train_text, test_text, vocab_limit, mode='count'):
  '''vectorize docs w keras Tokenizer API properly with one function call'''
  assert mode in ['binary','count','freq','tfidf'], 'supplied `mode` invalid!'
  tokenizer = Tokenizer(num_words=vocab_limit)
  tokenizer.fit_on_texts(train_text)
  
  train_intseqs = tokenizer.texts_to_sequences(train_text)
  test_intseqs = tokenizer.texts_to_sequences(test_text)
  
  train_x = tokenizer.sequences_to_matrix(train_intseqs, mode=mode)
  test_x = tokenizer.sequences_to_matrix(test_intseqs, mode=mode)
  
  return train_x, test_x, tokenizer.word_index



def quick_docpad(train_text, test_text, vocab_limit, out_length):
  '''pad docs w keras Tokenizer API properly with one function call'''
  tokenizer = Tokenizer(num_words=vocab_limit)
  tokenizer.fit_on_texts(train_text)
  
  train_intseqs = tokenizer.texts_to_sequences(train_text)
  test_intseqs = tokenizer.texts_to_sequences(test_text)
  
  train_x = pad_sequences(train_intseqs, maxlen=out_length)
  test_x = pad_sequences(test_intseqs, maxlen=out_length)
  
  return train_x, test_x, tokenizer.word_index






### performance evaluation utilities ----------------------------------------
def quick_clfreport(y_true, y_pred, digits=3):
  metrics = [f1_score, accuracy_score, precision_score, recall_score]
  fmt_metric = lambda f: f.__name__.replace('_score', '')
  report = {fmt_metric(f): round(f(y_true, y_pred), digits) for f in metrics}
  return report


def make_confmat_dict(y_obs, y_pred):
  conf_mat = confusion_matrix(y_obs, y_pred)
  tn, fp, fn, tp = conf_mat.ravel()
  return {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}





### postprocess results dict utils ---------------------------------------
def results_dict_to_df(results, metric_name):
  res = {key: val[metric_name] for key, val in results.items()}
  res_df = pd.DataFrame(res, index=[metric_name]).T
  res_df.index.name = 'id'
  res_df.reset_index(inplace=True)
  return res_df


def postprocess_results(results, metric_names):
  '''
  data munging! take the results dict + flatten it to a df, 
  by calling results_dict_to_df on each metric name
  '''
  res_df_list = [results_dict_to_df(results, m) for m in metric_names]
  metrics_df = reduce(lambda l, r: pd.merge(l, r, on='id'), res_df_list)
  metrics_df['clf'] = [re.sub('q\\d-', '', val) for val in metrics_df.id]
  metrics_df['lbin'] = [re.sub('-[0-9a-zA-Z_]+', '', val) for val in metrics_df.id]
  metrics_df = metrics_df[['clf', 'lbin'] + metric_names]
  return metrics_df







### func to make visualization of keras network graph --------------------
def plot_keras_model(clf_key, hyper_dict, out_dir):
  '''visualize the structure of a keras network, write to .png
  
  # wrapper that compiles model + then calls:
  plot_model(model, to_file=outfile, dpi=300,
             show_shapes=False, show_layer_names=True, expand_nested=False)
  '''
  outfile = os.path.join(out_dir, clf_key+'_graph.png')

  nn = Sequential()
  for layer, layer_params in hyper_dict['layers']:
    nn.add(layer(**layer_params))
  nn.compile(**hyper_dict['config'])
  
  # print(f'writing model network graph to file: `{outfile}`')
  plot_model(nn, to_file=outfile,
             show_shapes=True, show_layer_names=False)







### dev + unused stuff area -------------------------------------------------
### dev + unused stuff area -------------------------------------------------
### dev + unused stuff area -------------------------------------------------

# def train_clf(clf_identifier, hyper_dict, Xs, ys):
#   # NOTE: assumes hyper_dict is compatible with the relevant API 
#   # NOTE: assumes Xs and ys are prepped correctly for the clf! 
#   clf_API = clf_APIs[clf_identifier]
#   clf_class = clf_classes[clf_identifier]
#   train_function = {'sklearn': train_sklearn, 'keras': train_keras}[clf_API]
#   predict_function = train_function(clf_class, hyper_dict, Xs, ys)
#   return predict_function

### example of quick_docpad() usage:
# docs = ['this is me toy corp', 'a corp is just docs', 'this is a doc doc']
# moredocs = ['this is me last corp corp corp', 'waow disjoint vocab yikes']
# pd1, pd2, widx = quick_docpad(docs, moredocs, vocab_limit=5, out_length=10)
