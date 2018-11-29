from sklearn.feature_extraction.text import CountVectorizer


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_score, recall_score



### train + predict wrappers ---------------------------------------------
def train_clf(clf_identifier, hyper_dict, Xs, ys):
  # NOTE: assumes hyper_dict is compatible with the relevant API 
  # NOTE: assumes Xs and ys are prepped correctly for the clf! 
  clf_API = clf_APIs[clf_identifier]
  clf_class = clf_classes[clf_identifier]
  train_function = {'sklearn': train_sklearn, 'keras': train_keras}[clf_API]
  predict_function = train_function(clf_class, hyper_dict, Xs, ys)
  return predict_function


def train_sklearn(clf_class, hyper_dict, Xs, ys):
  clf = clf_class(**hyper_dict)
  clf.fit(Xs, ys)
  return clf.predict


def train_keras(clf_class, hyper_dict, Xs, ys):
  clf = clf_class()
  for layer, layer_params in hyper_dict['layers']:
    clf.add(layer(**layer_params))
  clf.compile(**hyper_dict['config'])
  clf.fit(Xs, ys, **hyper_dict['train'])
  return clf.predict_classes



#### load + prep data utils --------------------------------------
def get_imdb_subset(dat, subset, lbin):
  out = dat[(dat.subset==subset) & (dat.length_bin==lbin)]
  # print(f'retrieved {out.shape}-dim {lbin}th quartile of IMDB {subset}')
  return out



def quick_vectorize(train_text, test_text, hypers={}):
  Vectorizer = CountVectorizer(**hypers)
  train_dtm = Vectorizer.fit_transform(train_text)
  test_dtm = Vectorizer.transform(test_text)
  return train_dtm, test_dtm




# docs = ['this is me toy corp', 'a corp is just docs', 'this is a doc doc']
# moredocs = ['this is me last corp corp corp', 'waow disjoint vocab yikes']
# pd1, pd2, widx = quick_docpad(docs, moredocs, vocab_limit=5, out_length=10)
def quick_docpad(train_text, test_text, vocab_limit, out_length):
  tokenizer = Tokenizer(num_words=vocab_limit)
  tokenizer.fit_on_texts(train_text)
  
  train_intseqs = tokenizer.texts_to_sequences(train_text)
  test_intseqs = tokenizer.texts_to_sequences(test_text)
  
  train_x = pad_sequences(train_intseqs, maxlen=out_length)
  test_x = pad_sequences(test_intseqs, maxlen=out_length)
  
  return train_x, test_x, tokenizer.word_index



def quick_dtmize(train_text, test_text, vocab_limit, mode='count'):
  assert mode in ['binary','count','freq','tfidf'], 'supplied invalid mode arg'
  tokenizer = Tokenizer(num_words=vocab_limit)
  tokenizer.fit_on_texts(train_text)
  
  train_intseqs = tokenizer.texts_to_sequences(train_text)
  test_intseqs = tokenizer.texts_to_sequences(test_text)
  
  train_x = tokenizer.sequences_to_matrix(train_intseqs, mode='count')
  test_x = tokenizer.sequences_to_matrix(test_intseqs, mode='count')
  
  return train_x, test_x, tokenizer.word_index






def get_params_subset(hypers_dict, clf_key, prefix='clf__'):
  '''assumes `hypers_dict` has structure of data in sklearn_tuned_hypers.json'''
  params = hypers_dict[clf_key]['best_params']
  out = {key.replace(prefix, ''): val
         for key, val in params.items() if key.startswith(prefix)}
  return out




def quick_clfreport(y_true, y_pred, digits=3):
  metrics = [f1_score, accuracy_score, precision_score, recall_score]
  fmt_metric = lambda f: f.__name__.replace('_score', '')
  report = {fmt_metric(f): round(f(y_true, y_pred), digits) for f in metrics}
  return report




