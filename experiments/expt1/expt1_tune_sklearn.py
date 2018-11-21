'''Optimize hypers for Expt1 sklearn classifiers 

Note: started job at 12:03pm...


Algorithms:
  a. LogisticRegression
  b. MultinomialNB
  c. SVC

Dataset: 
  - imdb sentiment (train subset) 


Gameplan:
  1. load + prep data 
  2. load param space grids 
  3. write wrapper to search param grids 
  4. search param grid for each clf 
  5. extract and write best hypers for each clf 

TODO: 
  - integrate searching vectorizer params too! (keep it simple for now)
  - 

'''

# for a bunch of sklearn future warnings
import warnings
warnings.filterwarnings('ignore')

import json
import pandas as pd

from functools import partial

# from sklearn.svm import SVC #  <-- not converging, replace!
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


optimized_params_outfile = 'sklearn_tuned_hypers.json'




### 1. load + prep data ------------------------------------------------------
imdb_data_fname = 'data/imdb_decoded.csv'

imdb_data = pd.read_csv(imdb_data_fname)

# only using the train data in this phase 
dat = imdb_data[imdb_data.subset=='train']




### 2. define param space grids ----------------------------------------------
hyper_grids = {
  # 36 param combos for logreg (*2 from vectorizer = 72)
  'LogisticRegression': {
    'vect__max_features': (None, 500), 
    'clf__penalty': ('l1', 'l2'), 
    'clf__tol': (1e-3, 1e-4, 1e-5), 
    'clf__C': (0.01, 0.1, 1.0), 
    'clf__fit_intercept': (True, False)
  }, 
  # 12 param combos for naive bayes (*2 from vectorizer = 24)
  'MultinomialNB': {
    'vect__max_features': (None, 500), 
    'clf__alpha': (0.1, .25, .5, .75, .9, 1.0), 
    'clf__fit_prior': (True, False)
  }
}





### 3. write func to search a param grid -------------------------------------
def search_param_grid(Vectorizer, Classifier, hyper_grid, X, y):
  '''Search a pipeline hyper-parameter grid, return optimized hypers 

  Params:
    Vectorizer: an sklearn vectorizer for constructing features from text
    Classifier: an sklearn classifier with .fit() and .predict() methods
    hyper_grid: a dict with hyper ranges, in sklearn.pipeline-style fields
    X: input texts, from which features are constructed
    y: labels/targets associated with the texts 

  Example:
    ```
    search_param_grid(
      Vectorizer=CountVectorizer, Classifier=MultinomialNB, 
      hyper_grid={'vect__max_features': (None, 5), 
                  'clf__alpha': (0, 1.0), 'clf__fit_prior': (True, False)}, 
      X=dat.text, y=dat.label)
    ```

  TODO: 
    - allow the vectorizer params to vary!!!

  '''
  pipeline = Pipeline([('vect', Vectorizer(max_features=1000)),
                       ('clf', Classifier())])
  
  grid_search = GridSearchCV(estimator=pipeline, param_grid=hyper_grid,
                             cv=StratifiedKFold(n_splits=3, random_state=69),
                             scoring='f1', n_jobs=-1, verbose=2)
  grid_search.fit(X, y)

  best_params = grid_search.best_estimator_.get_params()
  
  out = {'score': grid_search.best_score_, 'best_params': {}}
  
  for param_name in hyper_grid.keys():
    out['best_params'][param_name] = best_params[param_name]
  
  return out


# and bind the data and vectorizer params, which won't vary (yet!) 
gridsearch_partial = partial(
  search_param_grid, Vectorizer=CountVectorizer, X=dat.text, y=dat.label)




### 4. search param grid for each clf ----------------------------------------
classifiers = [LogisticRegression, MultinomialNB]

optimized_params = {}
for clf in classifiers:
  clf_name = clf.__name__
  clf_grid = hyper_grids[clf_name]
  print(f'\n\n\n*** now searching hypers grid for {clf_name}... ***')
  optimized_params[clf_name] = gridsearch_partial(Classifier=clf, 
                                                  hyper_grid=clf_grid)




### 5. extract and write best hypers for each clf ----------------------------
print(f'done optimizing hypers. results:\n{optimized_params}')
print(f'writing optimized hypers dict to:\n  >> {optimized_params_outfile}')

with open(optimized_params_outfile, 'w') as f:
  json.dump(optimized_params, f, indent=2)





### notes --------------------------------------------------------------------

# NOTE: SVC FITS ARE NOT CONVERGING! SO MUST REMOVE FOR NOW </3
# 8 param combos for support vector machine (*2 from vectorizer = 48) 
# note: better to consider more kernels -- do that if time later 
# 'clf__kernel': ('rbf', 'linear', 'poly', 'sigmoid') 
# 'SVC': {
#   # 'vect__max_features': (None, 500), # TODO: reintegrate!
#   # 'clf__C': (0.1, 1.0), # TODO: reintegrate!
#   'clf__kernel': ('rbf', 'sigmoid'), 
#   'clf__shrinking': (True, False), 
#   'clf__tol': (1e-2, 1e-3) # , 1e-4) # TODO: reintegrate!
# }
