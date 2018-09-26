'''
this script preps the kaggle rotten tomatoes train subset for experimentation.

source: https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data


the output of this script is a .csv written to: 
  'data/rottentom_phrases_prepped.csv'
  
the output table has the following columns:
  - id (int):           unique phrase identifier 
  - sent_id (int):      identifier of sentence text comes from
  - length (int):       token count on `text` field  
  - text (str):         free-form text (**use as inputs to predict `label`**)
  - is_positive (bool): binary label (**to predict from `text`**) 


NOTE:   the resulting dataset has some weird properties, 
        so should **only** be used to prep experiment code so that 
        when we have a good, solid dataset, we can just plug it into 
        the scaffolding we build while using this dataset. 

UPDATE: going to use decoded version of `keras.datasets.imdb` 
        in file 'data/imdb_decoded.csv' 
        (generate the csv with script `expt1_prep_imdb_data.py`)

'''

import numpy as np
import pandas as pd



# note that kaggle test sets don't have labels (eval in kaggle)
# test_file = '../../../data/kaggle_rotten_tomatoes/test.tsv'
train_file = 'data/rottentom_kaggle-train.tsv'
prepped_outfile = 'data/rottentom_phrases_prepped.csv'


dat = pd.read_csv(train_file, sep='\t')
dat.head(10)

# check that phrases don't repeat (cd make problem impossible)
# dat.apply(lambda col: len(np.unique(col)) == dat.shape[0])
assert len(dat.Phrase) == len(np.unique(dat.Phrase))


# create boolean labels (select most convenient at this stage)
dat['is_positive'] = [val  > 2 for val in dat.Sentiment]
dat['is_neutral']  = [val == 2 for val in dat.Sentiment]
dat['is_negative'] = [val  < 2 for val in dat.Sentiment]

# check distribution of each label, and of original scale 
print('distribution of sentiment scale:\n', dat.Sentiment.value_counts())
dat[['is_positive','is_neutral','is_negative']].apply(pd.value_counts)


# create length field (number of words) 
dat['length'] = [len(s.split(' ')) for s in dat.Phrase]


# rearrange, drop unneeded fields, 
col_renames = {'PhraseId': 'id','SentenceId':'sent_id', 'Phrase':'text'}
dat = dat.rename(index=str, columns=col_renames)
dat = dat[['id','sent_id','length','text','is_positive']]


# write the data to csv for subsequent experimentation 
dat.to_csv(prepped_outfile, index=False)


