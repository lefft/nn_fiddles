'''Prepare Sentiment140 dataset for Experiment 1

This script adds fields to the Sentiment140 dataset and writees the 
reformatted version to a .csv file. The resulting format makes the data 
easier to experiment with. Metadata fields about the length (in words)
of each tweet and test/train status are also calculated and 
included in the output. 

The output file `sent140_prepped.csv` has TODO rows and the following fields: 

['id', 'subset', 'length', 'length_bin', 'label', 'text']

  - `id`: unique identifier for the tweet 
  - `subset`: indicates train/test status 
  - `length`: number of words in the tweet 
  - `length_bin`: length-bin of the tweet (upper/lower half w/in this dataset)
  - `label`: binarized sentiment of the tweet (1=positive, 0=negative) 
  - `text`: the text of the tweet, preprocessed in the following way:
      - lowercased
      - leading/trailing spaces stripped 
      - reduce multiple spaces to one 
      - TODO (remove urls +c)
'''

import re
import pandas as pd
import numpy as np


seed = 6933
np.random.seed(seed)




### i/o filenames and column specs -------------------------------------------
infile = 'data/sentiment140/training.1600000.processed.noemoticon.csv'
outfile = 'data/sent140_prepped.csv'

in_colnames = ['sentiment', 'id', 'datetime', 'query', 'handle', 'text']
out_colnames = ['id', 'subset', 'length', 'length_bin', 'label', 'text']




#### read + shuffle data -----------------------------------------------------
print(f'reading sent140 data from:\n  >> {infile}')
dat = pd.read_csv(infile, names=in_colnames, encoding='latin1')

# shuffle rows before doing anything 
dat = dat.sample(frac=1).reset_index(drop=True)




### preprocess text ----------------------------------------------------------
dat.text = [tweet.strip().lower() for tweet in dat.text]
dat.text = [re.sub(r' +', ' ', tweet) for tweet in dat.text]
## TODO: remove urls and unpad edges
## [re.sub(r'https?:\/\/.*[\r\n]*', '', t) for t in dat.text[:20]]




### mark each tweet with word count ------------------------------------------
dat['length'] = [len(tweet.split(' ')) for tweet in dat.text]




### mark each tweet with length bin (only two bins here) ---------------------
def assign_quantile(value, bin_boundaries):
  for idx, boundary in enumerate(bin_boundaries):
    if value <= boundary:
      return idx
  raise ValueError('value above highest bin boundary!')

# find quartile boundaries for text length 
bin_boundaries = [int(np.quantile(dat.length, q)) for q in [.5, 1]]

# assign length bin (2-ile) to each row 
dat['length_bin'] = [assign_quantile(n, bin_boundaries) for n in dat.length]

## dat['length_bin'].value_counts()
## 0 ==> 863789
## 1 ==> 736211




### assign test/train subset to each tweet -----------------------------------
prop_test = .3
n_test = int(prop_test * dat.shape[0])


# need to make sure the test set is even w.r.t. length bin 
bin0_idxs = dat.index[dat.length_bin == 0].tolist()
bin1_idxs = dat.index[dat.length_bin == 1].tolist()

bin0_test_idxs = np.random.choice(bin0_idxs, size=n_test//2, replace=False)
bin1_test_idxs = np.random.choice(bin1_idxs, size=n_test//2, replace=False)


# set everything as train, then set just the sampled row idxs as test 
dat['subset'] = 'train'

dat.loc[bin0_test_idxs, 'subset'] = 'test'
dat.loc[bin1_test_idxs, 'subset'] = 'test'

## pd.crosstab(dat.subset, dat.length_bin, margins=True)
## [note: sentiment crosstab w subset is reasonable too]
##         0       1       All
## test    24e4    24e4    48e4
## train   623789  496211  112e4
## All     863789  736211  16e5





### re-encode labels as binary (int, 0=negative, 1=positive) -----------------
dat['label'] = dat.sentiment.replace([0, 4], [0, 1])

## dat.sentiment.value_counts()
## 4 ==> 8e5, 0 ==> 8e5
## original coding is 0=negative, 4=positive
## 
## can spot-check that conversion was done correctly:
## dat[['label','sentiment']].head(25)




### rearrange columns + write to outfile -------------------------------------
dat = dat[out_colnames]

# print summary and write data to `outfile` 
print(f'writing sent140 data w dims {dat.shape} to file {outfile}...')

dat.to_csv(outfile, index=False)


