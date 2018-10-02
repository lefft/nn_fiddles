'''
train-eval a few keras models:
  - first on all available data
  - then on the length-defined subsets 


TODO: 
  - re-encode text as idx list (or just use keras.imdb.load_imdb()...)
      - should prob re-encode tho, since this is mostly educational
      - to re-encode, use keras.preprocessing.text.{one_hot(), hashing_trick()} 
  - write uniform prediction plotting interface 
  - establish useful delimited format for model specs and performance metrics
  - uniformize this and sklearn oob script 


'''

import numpy as np
import pandas as pd

from keras import models
from keras import layers

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score





### load data ------------------------------------------------------
data_file = 'data/imdb_decoded.csv'
dat = pd.read_csv(data_file)

print(f'loaded {dat.shape[0]}x{dat.shape[1]} data') 



# TODO: pull `get_subset()` into a module + import it
#       (for now just training on all available data) 

train_texts = list(dat.text[dat.subset=='train'])
test_texts = list(dat.text[dat.subset=='test'])

train_labels = list(dat.label[dat.subset=='train'])
test_labels = list(dat.label[dat.subset=='test'])





### preprocess data ------------------------------------------------

# TODO: EITHER JUST USE keras.imdb.load_imdb() OR RE-ENCODE AS INTS...
# 
# if re-encode, use 
#   - keras.preprocessing.text.hashing_trick()  or 
#   - keras.preprocessing.text.one_hot()        (wraps hash trick)
# 
# BUT need to do that on *all* text first so that idx's are shared?! 

input_dim = 10000

# adapted from chollet book, ch3
def vectorize_sequences(sequences, dim):
  out = np.zeros((len(sequences), dim))
  for idx, sequence in enumerate(sequences):
    out[idx, sequence] = 1.
  return out

x_train = vectorize_sequences(train_texts, input_dim)
x_test = vectorize_sequences(test_texts, input_dim)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')





### define the network structure ------------------------------------------
model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(input_dim, )))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))





### specify optimization params and compile the network --------------------
optimizer = 'rmsprop'
loss = 'binary_crossentropy'
train_metrics = ['accuracy']

model.compile(optimizer=optimizer, loss=loss, metrics=train_metrics)





### specify train params and train the classifier ----------------------------
valset_size = 1e4

epochs = 20
batch_size = 512

# further partition the data into train and train validation subsets
x_valset = x_train[ :val_size]
y_valset = y_train[ :val_size]

partial_x_train = x_train[val_size: ]
partial_y_train = y_train[val_size: ]

training_history = model.fit(partial_x_train, partial_y_train,
                             epochs=epochs, batch_size=batch_size,
                             validation_data=(x_valset, y_valset))





### generate model predictions -----------------------------------------
def prob_to_binary(prob, threshold=.5, ret_type=bool):
  assert 0 <= prob <= 1
  assert 0 <= threshold <= 1
  return ret_type(prob > threshold)


# TODO: does a keras.model instance have a .predict method?! 
test_probs = model.predict(x_test)
test_preds_bool = [prob_to_binary(prob) for prob in test_probs]
# test_preds_int = [prob_to_binary(prob, ret_type=int) for prob in test_probs]





### evaluate model predictions ------------------------------------------------
confusion_matrix(y_test, preds_binary)

metrics = [f1_score, precision_score, recall_score, accuracy_score]

for func in metrics:
  print(f'{func.__name__}: {func(y_test, preds_binary)}')





### model training introspection ----------------------------------------------
# TODO: FIGURE OUT A USEFUL WAY TO DO THIS 
#       (or maybe wait till next phase since no corresponding thing for skl)
history_dict = training_history.history
print(history_dict)
# history_dict.keys()




# TODO: FIGURE OUT USEFUL WAY TO DISPLAY PLOTS!!!
#       (not including this until later) 
# 
# 
# import matplotlib.pyplot as plt
# 
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# 
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
# 
# epochs = range(1, len(acc) + 1)
# 
# # ugh matplotlib sucks </3 
# plt.plot(epochs, loss_values, 'bo', label='Training loss')
# plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
# # plt.title('Training and validation loss')
# # plt.xlabel('Epochs')
# # plt.ylabel('Loss')
# # plt.legend()
# # plt.show()

