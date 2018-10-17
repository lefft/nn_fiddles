'''Wrapper classes for sklearn and keras binary classifiers

Defines classes to quickly experiment with two types of classifiers: 

### TypeA(Classifier, **kwargs)
Convenience wrapper to sklearn binary classifiers. 
Methods:
  - .train(train_dtm, train_labels)
  - .predict(test_dtm)

### TypeB(KerasModel, layers_list=[], kwargs_list=[])
Convenience wrapper to keras.models.Sequential models. 
Methods: 
  - .train(train_X, train_y, valset_prop, epochs, batch_size)
  - .predict(test_X, pred_postprocessor=lambda x: x)
  - .add_layers(layers_list, kwargs_list)
  - .compile_model(optimizer, loss, metrics)


TODO: 
  - uniformize param names e.g. train_X versus train_dtm 
  - allow passing raw docs instead of dtm?? 
  - [see TODOs in each class def'n]

'''


import sklearn
import keras 


class TypeA():
  '''Wrapper class for `sklearn` binary text classifiers 
  
  on init:
    - instantiate `Classifier` class, with params given by `**kwargs`
    - store `Classifier.__name__` and param key-vals in .clf_info attr 
  methods:
    - .train(train_dtm, train_labels): call .clf.fit() on dtm and labels
    - .predict(test_dtm): generate predictions over unseen input data 
  
  attributes:
    - .clf: `sklearn.*.Classifier` instance
    - .clf_info: dict, stores classifier name and param key-value pairs 
    - .train_dtm, .train_labels: input data and labels fed to .train()
  
  usage example: 
    ```
    ### TODO -- integrate notebook ex (till then, see sec 3.0)
    ```
  
  TODO:
    - want to save weight matrix as an attr??
    - check handling of **kwargs 
    - ... 
  '''
  def __init__(self, Classifier, **kwargs):
    self.clf = Classifier(**kwargs)
    self.clf_info = dict({'clf': Classifier.__name__}, **kwargs)
  
  def train(self, train_dtm, train_labels):
    self.train_dtm, self.train_labels = train_dtm, train_labels
    self.clf.fit(self.train_dtm, self.train_labels)
  
  def predict(self, test_dtm):
    test_preds = self.clf.predict(test_dtm)
    return test_preds





class TypeB():
  '''Wrapper class for managing `keras.models.Sequential()` models
  
  on init: 
    - instantiates keras model 
    - adds supplied layers (if any)
    - sets a few attrs used during train/compile/predict 
  
  methods:
    - add_layers(layers_list, kwargs_list)
    - compile_model(optimizer, loss, metrics)
    - train(train_X, train_y, valset_prop, epochs, batch_size)
    - predict()
  
  attributes:
    - .model: instance of KerasModel class passed on init 
    - .layers_info: list of dicts with params and type of each layer 
    - .is_compiled, .layers_added: boolean, for tracking model state 
    - .train_X, .train_y: train data and labels, appropriately preprocessed
    - .history: a keras History object with info about training history  
  
  usage example:
    ```
    ### TODO -- integrate notebook ex (till then, see sec 3.0)
    ```
  
  TODO: 
    - can have multiple histories?? (if so, maybe append to lsit instead)
    - need to pass anything to KerasModel?!
    - track layer indices in .layers_info?! 
    - add print and/or repr and/or display method?!?! 
    - abstract over **kwargs for .compile_model()
    - abstract over **kwargs for .train()  
    - ... 
  '''
  def __init__(self, KerasModel, layers_list=[], kwargs_list=[]):
    self.model = KerasModel()
    self.layers_info = []
    self.is_compiled = False
    self.layers_added = False
    # validate KerasModel param 
    modtypes = (keras.engine.sequential.Sequential, keras.engine.training.Model)
    assert isinstance(self.model, modtypes), 'must supply a keras model!'
    # TODO: validate layers_list and kwargs_list params    
    assert len(layers_list) == len(kwargs_list)
    if len(layers_list) > 0: self.add_layers(layers_list, kwargs_list)
  
  def add_layers(self, layers_list, kwargs_list):
    for layer, kwargs in zip(layers_list, kwargs_list):
      self.model.add(layer(**kwargs))
      self.layers_info.append(dict({'layer': layer.__name__}, **kwargs))
    self.layers_added = True
  
  def compile_model(self, optimizer, loss, metrics):
    assert self.layers_added, 'must add layers to model before compiling!'
    self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    self.is_compiled = True
  
  def train(self, train_X, train_y, valset_prop, epochs, batch_size):
    self.train_X, self.train_y = train_X, train_y
    assert len(self.train_X) == len(self.train_y)
    assert self.is_compiled, 'must compile model before training!'
    valset_size = round(valset_prop * len(self.train_y))
    trn_X, val_X = self.train_X[valset_size:], self.train_X[:valset_size]
    trn_y, val_y = self.train_y[valset_size:], self.train_y[:valset_size]
    self.history = self.model.fit(trn_X, trn_y, 
                                  epochs=epochs, batch_size=batch_size, 
                                  validation_data=(val_X, val_y), verbose=2)
  
  def predict(self, test_X, pred_postprocessor=lambda x: x):
    test_probs = self.model.predict(test_X)
    test_preds = [pred_postprocessor(prob) for prob in test_probs]
    return test_preds





# `TypeC` interface same as `TypeB` since both are keras?! 
# TODO: figure out if we even need TypeC interface... 
# class TypeC(): pass
