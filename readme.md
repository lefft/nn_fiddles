## neural nets fiddle repo `¯\_(ツ)_/¯`
###### currently working on expt1 [updated: sept26/2018]

final steps for expt1: 

- [ ] tune nn models 
- [ ] generate performance curves 


<br><br><br>

- expt1: movie reviews sentiment classification (varying text length) 
- expt2: topic detection in social media posts (varying train n) 
- and maybe an expt3(?) varying class distribution 



<br><br><br>

### expt 1 notes 

#### `TODO`


- [ ] solifify tuning plan for keras models 
- [ ] write grid search function(s) for keras sequential models
- [ ] write nn grid search script 
- [ ] document all code thoroughly 
- [ ] sketch model evaluation in step 4. below 
- [ ] add `sklearn.linear_model.SGDClassifier` to skl algos space 
- [ ] expand sklearn grid search + rerun (bigger param space, add sgd) 



#### objectives 

- want to know if there is an effect of text length on sentiment classification difficulty
- if so, want to know how this interacts with classifier type (neural net-based versus others)
- operationalized: create accuracy/F1 curves across text length subsets, w separate lines for (tuned) `sklearn` models and for a few tuned `keras` nn models





#### associated files 

###### source 

- `expt1_explore_imdb.rmd`: fiddling area + scratchpad
- `expt1_make_practice_data.py`: reformat rotten tomatoes data for fiddling 
- `expt1_oob_models.py`: fit-eval some out-of-box classifiers on decoded imdb data
- `expt1_prep_imdb_data.py`: decode `keras.datasets.imdb`, add fields for word count and length bin (quartile) 
- `expt1_sketch.py`: sketchpad for fiddling with keras models 


###### data 

- `data/imdb_decoded.csv`: prepped data to use for expt1 (should confirm)







#### expt format sketch 

here is a bare-bones sketch of what the first experiment will be like. 
note that important details are omitted in this overview. 
details will be filled in as we begin setting the experiment up. 

relevant details that are omitted: 

- text preprocessing 
- choice of performance metric 
- cross-validated performance metrics 
- hyperparameter tuning/model selection 
- network structure for neural net models 
- more rigorous test/train/eval splitting strategy 
- stratification of tt-split by label 



###### 1. stratified train-test split 

here we split the available data into train and test sets, 
stratifying by the review length. since length (in words) is 
a quantity, we need to bin it before stratification is possible. 

call the complete set of available data `docs`. 

associate a boolean label `y_doc` with each `doc in docs`. 
call the complete set of labels `ys`. 

assume that we can always co-index `docs` with `ys` so that 
`ys[i]` is the label for `docs[i]` for `i in 1:len(docs)`. 

assume that when `doc` is in some relevant subset, 
there is a corresponding subset of labels containing `y_doc`. 

define the *length* of a review `doc in docs` as its word-count. 


let `C` be the following set of length categories (first-pass). 

- `C[0]` = 0-49 words
- `C[1]` = 50-99 words
- `C[2]` = 100-149 words
- `C[3]` = 150-199 words
- `C[4]` = 200+ words


place each review into its appropriate length category `c in C`. 

call the set of length-`c` reviews `docs_c` for `c in C`. 

for `c in C`: 

- randomly put 30% of `docs_c` into `docs_c[test]` 
- put the remaining 70% into `docs_c[train]` 


for convenience, assume `len(docs_c) == 100` for `c in C`. 

let `ys_c` be the set of labels for `docs_c`. 

let `docs[test]` be the union of `docs_c[test]` over `C`. 
let `docs[train]` be the union of `docs_c[train]` over `C`. 

then `len(docs[test]) == 30*5 == 150`. 
then `len(docs[train]) == 70*5 == 350`. 




###### 2. define range of classifiers to consider 

let `f_*: str --> bool` be a binary classifier over strings. 

we'll assess the performance of six different values of `*`:

- `f_logreg` is a logistic regression classifier; 
- `f_svm` is a support-vector classifier; 
- `f_mnb` is a multinomial naive bayes classifier; 
- `f_nn1` is a neural net-based classifier; 
- `f_nn2` is another nn-based classifier (w diff structure); and
- `f_nn3` is yet another nn-based classifier w diff structure. 


define `F = {f_logreg,f_svm,f_mnb,f_nn1,f_nn2,f_nn3}`. 



###### 3. define experiment routine 

want to generate a curve across `C` for each `f in F`. 

- x-axis: indices of `C` -- `[0,1,2,3,4]` (or length upper-bound). 
- y-axis: some classification performance metric (prob F1 score). 


to do this, fit each `f` to each subset `docs_c`.  


```
### pseudocode walkthru of basic idea 
### (no need to stick to specific implementation)
### 

# dict holding performance curves for each clf 
performance_curves = {}

for f in F:
    # initialize list to hold scores across `C` 
    f_clf_scores = []

    for c in C: 
        # split `docs_c` and `ys_c` into test and train subsets  
        train_docs, train_ys = docs_c[train], ys_c[train]
        test_docs, test_ys = docs_c[test], ys_c[test]

        # train `f` on the train docs and train labels 
        f_trained = f.train(train_docs, train_ys)

        # generate predictions over the test docs 
        f_preds = f_trained.predict(test_docs)

        # compare test predictions to test labels 
        f_performance_on_c = performance(f_preds, test_ys)

        # add the current performance metric to current curve 
        f_clf_scores.append(f_performance_on_c)

    # create dict entry for `f`, holding its curve across `C` 
    performance_curves[f] = f_clf_scores

```


###### 4. create performance curves 

`TODO`: fill in steps/outline sketch!  









<br><br><br>

### expt 2 notes 

#### `TODO`

- [ ] get started! 
