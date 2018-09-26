'''
fit a bunch of out-of-box models to prepped imdb data

compare oob f1 performance of:
  - sklearn.naive_bayes.MultinomialNB
  - sklearn.linear_model.LogisticRegression
  - sklearn.linear_model.SGDClassifier
  - TODO: a few simple keras sequential models 

next step is to perform large grid search for each clf algo


observations: 
  - perceptron + sgd involve randomness, others dont 


TODO: 
  - modularize this, write argparse cli to vary metric/vect/etc.
  - integrate keras example from `expt1_sketch.py` next!!! 
  - eliminate annoying SGD future warning (but no probs)
  - note which clf's involve rng (e.g. perceptron + sgd, not logreg)
  - automatically freeze rng seed when needed  
  - ...  

'''

import warnings # TODO: find workaround so this isn't needed
warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import pandas as pd

from functools import partial

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier



# TODO: wrap everything up + write cli to vary metric/vect/etc.
# metrics: f1_score, precision_score, recall_score, accuracy_score
metric = f1_score
# vectorizers: CountVectorizer, TfidfVectorizer
Vect = CountVectorizer
sklearn_oob_outfile = 'results/sklearn_oob_results-f1-cvect.csv'
print(f'using metric `{metric}`, vectorizer `{Vect}`...')




### load data ------------------------------------------------------
data_file = 'data/imdb_decoded.csv'
dat = pd.read_csv(data_file)

print(f'read in {dat.shape[0]}x{dat.shape[1]} data...') 




### partition data -------------------------------------------------
def get_subset(length_bin: int, train_test: str) -> pd.DataFrame:
  assert 'length_bin' in dat.columns
  assert 'subset' in dat.columns
  return dat[(dat.length_bin == length_bin) & (dat.subset == train_test)]


length_bins = np.unique(dat.length_bin)
subsets = np.unique(dat.subset)

dat_subsets = {}
for lbin in length_bins:
  dat_subsets[str(lbin)] = {}
  for subset in subsets:
    dat_subsets[str(lbin)][subset] = get_subset(lbin, subset)

# TODO: reproduce below print version but w this nested structure
# print({key: list(tt.keys()) for key, tt in dat_subsets.items()})

# [not using bc nested dict easier to work with]
# dat_subsets = {}
# for length_bin in length_bins:
#   for subset in subsets:
#     subset_id = f'bin{length_bin}_{subset}'
#     dat_subsets[subset_id] = get_subset(length_bin, subset)
# print({key: x.shape[0] for key, x in dat_subsets.items()})
#   {'bin0_test': 6564, 'bin0_train': 6169, 
#    'bin1_test': 6166, 'bin1_train': 6184, 
#    'bin2_test': 6223, 'bin2_train': 6245, 
#    'bin3_test': 6047, 'bin3_train': 6402}




### define fit-eval func for sklearn models --------------------------
def fit_eval_skl(x_train, y_train, x_test, y_test, Vect, Clf, metric):
  vectorizer, classifier = Vect(), Clf()
  train_vecs = vectorizer.fit_transform(x_train)
  test_vecs = vectorizer.transform(x_test)
  classifier.fit(train_vecs, y_train)
  test_preds = classifier.predict(test_vecs)
  return metric(y_test, test_preds)




### fit-eval for keras models --------------------------------------

# TODO 




### fit-eval across subsets for sklearn models -----------------------

# create container to feed clfs in and catch results 
clfs_and_scores = dict(
  MNB={'clf': MultinomialNB,          'scores': []}, 
  LGR={'clf': LogisticRegression,     'scores': []}, 
  SGD={'clf': SGDClassifier,          'scores': []}, 
  DTR={'clf': DecisionTreeClassifier, 'scores': []}, 
  PTN={'clf': Perceptron,             'scores': []})


# fit and eval each model at each length_bin subset 
for length_bin in length_bins:
  
  # get train and test subsets 
  train = dat_subsets[str(length_bin)]['train']
  test = dat_subsets[str(length_bin)]['test']
  
  # bind data, vectorizer, and metric to the fit-eval func 
  fit_eval_bound = partial(fit_eval_skl, 
                           x_train=train.text, y_train=train.label, 
                           x_test=test.text, y_test=test.label, 
                           # Vect=CountVectorizer, metric=f1_score)
                           Vect=Vect, metric=metric)
                           
  
  # for each classifier: 
  for clf_id in clfs_and_scores.keys():

    print(f'working on {clf_id}, bin {length_bin}')
    
    # extract the clf class 
    Clf = clfs_and_scores[clf_id]['clf']

    # get the f1 score for current subset and, append to relevant list
    score = fit_eval_bound(Clf=Clf)
    clfs_and_scores[clf_id]['scores'].append(score)




### organize results and write to file -----------------------------------

# collect results into a dict, convert to a df 
results = {}
for clf_id in clfs_and_scores.keys():
  results[clf_id] = clfs_and_scores[clf_id]['scores']

sklearn_oob_results = pd.DataFrame(results)

# print results + write to file 
print(sklearn_oob_results)
print(f'\nwriting sklearn oob results to:\n  >> `{sklearn_oob_outfile}`')

sklearn_oob_results.to_csv(sklearn_oob_outfile, index=False)




quit()
# TODO: INTEGRATE KERAS EXAMPLE FROM `expt1_sketch.py` NEXT!!! 
# TODO: INTEGRATE KERAS EXAMPLE FROM `expt1_sketch.py` NEXT!!! 
# TODO: INTEGRATE KERAS EXAMPLE FROM `expt1_sketch.py` NEXT!!! 
# TODO: INTEGRATE KERAS EXAMPLE FROM `expt1_sketch.py` NEXT!!! 
# TODO: INTEGRATE KERAS EXAMPLE FROM `expt1_sketch.py` NEXT!!! 
# TODO: INTEGRATE KERAS EXAMPLE FROM `expt1_sketch.py` NEXT!!! 

