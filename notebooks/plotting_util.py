'''Utilities for making simple plots with `matplotlib`

Defines:
  - plot_keras_loss_accuracy_curves(history)
  - human_readable_confusion_table(y_obs, y_pred)
  - ... 



TODO:
  - put this in a place where other stuff can import it!!
  - better/more general interface for plot_keras_...() (no keras restrict)
  - expand to 2-5 of other useful visuals:
      - confusion table or good/bad pred counts 
      - ... 
'''


import matplotlib.pyplot as plt

from keras.callbacks import History
from sklearn.metrics import confusion_matrix




def human_readable_confusion_table(y_obs, y_pred):
  conf_mat = confusion_matrix(y_obs, y_pred)
  tn, fp, fn, tp = conf_mat.ravel()
  out = '\n'.join([
    'count (% of total) for each label-pred combo:', 
    f'  >> true neg:  {tn} ({round(tn/conf_mat.sum()*100, 1)}%)',
    f'  >> false pos: {fp} ({round(fp/conf_mat.sum()*100, 1)}%)',
    f'  >> false neg: {fn} ({round(fn/conf_mat.sum()*100, 1)}%)',
    f'  >> true pos:  {tp} ({round(tp/conf_mat.sum()*100, 1)}%)'])
  # return out
  print('count (% of total) for each label-pred combo:')
  print(f'  >> true neg:  {tn} ({round(tn/conf_mat.sum()*100, 1)}%)')
  print(f'  >> false pos: {fp} ({round(fp/conf_mat.sum()*100, 1)}%)')
  print(f'  >> false neg: {fn} ({round(fn/conf_mat.sum()*100, 1)}%)')
  print(f'  >> true pos:  {tp} ({round(tp/conf_mat.sum()*100, 1)}%)')



def plot_keras_loss_accuracy_curves(history, figure_dpi=80):
  # only defined for `keras.callbacks.History` objects! 
  assert isinstance(history, History), 'must supply a keras History object!'
  # want to inspect loss and accuracy curves during train 
  losses, val_losses = history.history['loss'], history.history['val_loss']
  accs, val_accs = history.history['acc'], history.history['val_acc']  
  epoch_idxs = range(1, len(accs) + 1) # or [idx+1 for idx in history.epoch]
  # plot loss curves for train and validation sets during training 
  plt.subplot(2, 1, 1)
  plt.plot(epoch_idxs, losses, 'bo', label='train')
  plt.plot(epoch_idxs, val_losses, 'b', label='validation')
  plt.title('Loss during model training')
  plt.ylabel('loss')
  plt.legend(loc=3)
  # plot accuracy curves for train and validation sets during training 
  plt.subplot(2, 1, 2)
  plt.plot(epoch_idxs, accs, 'bo', label='train')
  plt.plot(epoch_idxs, val_accs, 'b', label='validation')
  plt.title('Accuracy during model training')
  plt.xlabel('epoch number')
  plt.ylabel('accuracy\n(prop correct)')
  plt.legend(loc=4)
  # check `plt.rcParams.keys()` to tune graphical params 
  plt.rcParams['figure.dpi'] = figure_dpi
  plt.tight_layout()
  plt.show()

