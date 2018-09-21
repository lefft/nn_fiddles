# TODO: 
#   - reorganize all code in this file, split into >1 script, etc. 
#   - reassess what needs to happen on basis of practice today. 
#   - figure out whether it is *necessary* to prep data differently for nns
#   - compare clf accuracy depending on how you prep data 


import numpy as np
import pandas as pd

from functools import partial

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier


### prep data ------------------------------------------------------
practice_data = 'data/rottentom_phrases_prepped.csv'
dat = pd.read_csv(practice_data)

X_train, X_test, y_train, y_test = train_test_split(
  dat.text, dat.is_positive, test_size=1/3, random_state=6933)

# TODO
# train_texts = [decode_review(review) for review in train_data]
# test_texts = [decode_review(review) for review in test_data]


### define fit-eval routine -----------------------------------------
def fit_eval(X_train, y_train, X_test, y_test, vectorizer, clf, metric):
  vectorizer = vectorizer()
  classifier = clf()
  train_vecs = vectorizer.fit_transform(X_train)
  test_vecs = vectorizer.transform(X_test)
  classifier.fit(train_vecs, y_train)
  test_preds = classifier.predict(test_vecs)
  print(confusion_matrix(y_test, test_preds))
  return metric(y_test, test_preds)


fit_eval_with_data = partial(fit_eval, 
                             X_train=X_train, y_train=y_train, 
                             X_test=X_test, y_test=y_test)

# multinomial naive bayes f1 = .66ish
fit_eval_with_data(vectorizer=CountVectorizer, clf=MultinomialNB, metric=f1_score)

# logistic regression f1 = .69ish
fit_eval_with_data(vectorizer=CountVectorizer, clf=LogisticRegression, metric=f1_score)

# SGD f1 = .62ish
fit_eval_with_data(vectorizer=CountVectorizer, clf=SGDClassifier, metric=f1_score)

# support vector machine f1 = [not finishing + i am feeling impatient...]
fit_eval_with_data(vectorizer=CountVectorizer, clf=SVC, metric=f1_score)



### EVALUATE SKLEARN MODELS ON KERAS IMDB DATASET (out of order, FIX!)
# train_texts = [decode_review(review) for review in train_data]
# test_texts = [decode_review(review) for review in test_data]

# f1 score: .81
fit_eval(train_texts, train_labels, 
         test_texts, test_labels, 
         CountVectorizer, MultinomialNB, f1_score)
# f1 score: .85
fit_eval(train_texts, train_labels, 
         test_texts, test_labels, 
         CountVectorizer, LogisticRegression, f1_score)
# f1 score: .85
fit_eval(train_texts, train_labels, 
         test_texts, test_labels, 
         CountVectorizer, SGDClassifier, f1_score)





### [getting code starting on page 68 of dlwp]
from keras.datasets import imdb

# TODO: DONT USE NUM WORDS, SO WE CAN RECOVER THE ENTIRE ACTUAL REVIEW 
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
  num_words=10000)

word_index = imdb.get_word_index()

reverse_word_index = dict(
  [(value, key) for (key, value) in word_index.items()])

decoded_review = ' '.join(
  [reverse_word_index.get(i-3, '<missing>') for i in train_data[0]])

from typing import List

def decode_review(idx_list: List[int]):
  return ' '.join([reverse_word_index.get(i-3, '<>') for i in idx_list])

# TODO: REARRANGE ALL THIS SHIT!!!
# TODO: make a separate script for prepping this data, and then run everything on that 
train_texts = [decode_review(review) for review in train_data]
test_texts = [decode_review(review) for review in test_data]


# train_data[0]
# decoded_review

import numpy as np

def vectorize_sequences(sequences, dimension=10000):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1.
  return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# x_train[0]

# vectorize the labels too
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
# "Now the data is ready to be fed into a neural network." (p70)


### define the network 
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

def prob_to_bool(prob: float, min_value: int, max_value: int) -> int:
  midpoint = (min_value + max_value) / 2
  return int(prob > midpoint)

# see listing 3.5-3.6, p73 for tweaking the optimizer, loss, etc. 
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))


history_dict = history.history
history_dict.keys()


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# ugh matplotlib sucks </3 
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


# QUESTION: does a keras.model instance have a .predict method?! 
preds = model.predict(x_test)
preds_binary = [prob_to_bool(lab, 0, 1) for lab in preds]


from sklearn.metrics import accuracy, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

confusion_matrix(y_test, preds_binary)
f1_score(y_test, preds_binary) # .848!!! compare w above 


