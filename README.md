# Genetic Algorithm - Hamming Weight (GA_HW)

# Description

Python implementation of the GA_HW algorithm.

The GA_HW is a Genetic Algorithm-based method which can be used for
variable selection, or more specifically for the abbreviation of a
categorical interview. In the context of psychiatric interviews, it can
be applied to obtain optimal screeners for the disorder of interest. The
method considers a set of m variables, and obtains the subset of n\<m
variables which maximices a classification metric (fitness) by means of
a `DecisionTreeClassifier` model predicting a categorical dependent
variable. During the evaluation of the fitness of each subset/screener,
a 10-Fold Cross-Validation is performed.

This is an early version of the algorithm which allows the selection of
three different fitness functions corresponding to different
optimization strategies:

- $AUCROC$: the algorithm optimizes over-all discrimination based on
  predicted probabilities
- The $F_2$ score: the algorithm optimizes the classification of
  positive cases (priority to `recall`)
- The $Matthews$ $Correlation$ $Coefficient$: the algorithm
  optimizes over-all classification considering dataset imbalance

# General Usage

In order to run the algorithm, prepare a dataset in `pandas` `dataframe`
format with respondents as rows and variables as columns. Keep this file
in the same directory as the `GA_HW` Python script you are running. When
using categorical variables, dummy encoding is highly advised. For
instance:

``` bash
db = pd.get_dummies(db0, drop_first=True)
```

``` bash
results = GA_HW.opt(db, variable_names, dep_variable, n, N=250, max_gen=10, cr_prob=0.6, fit_fun='auc', thr_search=False)
```

The four main arguments of the algorithm are:

- db: `pandas` dataframe with respondents as rows and variables as
  columns
- variable_names: `list` of `string` names of all independent variables
  (predictors, questions). Names must coincide with the column names of
  variables in the dataframe. Example, original interview with 10
  questions:

``` bash
variable_names = ['Q1', 'Q2, 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']
```

- dep_variable: `string` name of the dependent variable. Must coincide
  with a column name among variables in the dataframe.
- n: `integer` representing the size of the obtained subset. Corresponds
  to the size of the screener/number of questions/variables selected by
  the model.

# Additional Parameters/Arguments

The GA_HW also requires the following parameters:

- N: `integer`, size of the population on the GA. Default value is 250.
- max_gen: `integer`, number of generations created by the GA
  (iterations of the GA). Default value is 10.
- cr_prob: `float`, probability of cross-over inside the GA. Must be in
  range (0,1). Default is 0.6.
- fit_fun: `string`, classification metric (`sklearn.metric`) selected as fitness function. Default is 'auc'. Three choices are available:
 	- 'auc': $AUCROC$ (`auc`)
	- 'f2': $F_2$ metric (`fbeta_score` with `beta=2`)
	- 'mcc': $MCC$ metric (`matthews_corrcoef`)

- thr_search: `boolean`, whether to perform a grid search on the
  evaluation of the fitness to find the classification threshold of the
  `DecisionTreeClassifier` model which optimizes the cross-validated
  estimation of the fitness metric. Default is `False`, meaning the
  classification threshold of the model is selected as the default value
  (0.5) during the classifications.

# Model Outputs

The algorithm returns a `results` object containing the following atributes:

- results.fitness_list: a `list` of `max_gen` `floats` with the fitness values
  of the fittest individual on each generation.
- results.mean_fitness_list: a `list` of `max_gen` `floats` with the mean
  fitness values of all individuals on each generation.
- results.names: a `list` of variable names (`string`) included in the
  optimal solution, corresponding to the selected variables/optimal
  screener.
- results.threshold: a `float` corresponding to the classification threshold used
  by the `DecisionTreeClassifier` of the optimal solution. If the
  argument thr_search is `False`, this value is always 0.5.
- results.fitness: a `float` corresponding to the fitness value of the optimal solution. Calculate as the men of the 10 estimates obtained through the 10-fold Cross-validation.
- results.fitness_sem: a `float` corresponding to the standard error of the fitness value of the optimal solution. Calculated as the SE of the mean (`scipy.stats.sem`) of the 10 estimates obtained throuh the 10-fold Cross-validation.

# FAQ
```

**Q: How do I select the right values for the hyper-parameters of the model? Are the default values reliable?**

A: The used should be cautious when selecting the hyperparameters of the model. The default values are only sensible suggestions but are usually too conservative. The choice of the population size (`N`) and the number of generations (`max_gen`) depend on the characteristics of the dataset and precisely on the size of the search space. In applications considering psychiatric interviews where the number of independent variables is 30-60, population size (`N`) should be increased to 500-1000, depending on the size of the screener `n` (greater `n` means bigger search space). If the number of independent variables in the dataset is >100, the population size should not be lower than 1000. 

As for the cross-over probability, `cr_prob`=0.6 is a sensible value which has shown good results in test data. However, we encourage the user to experiment with different values to explore convergence. 

------------------------------------------------------------------------

**Q: What does the `thr_search` parameter do? When should I turn it `True`?**

A: In the GA_HW, the screeners in the population are evaluated by training a prediction model (`DecisionTreeClassifier`) and calculating the fitness metric on the resulting classification using 10-fold Cross-validation. A prediction model assigns a probability of positivity to each instance of the dataset. By default, this probability is transformed into a binary classification using a threshold of 0.5. However, the threshold 0.5 for a given prediction model might not be the one which maximizes the fitness metric. When the parameter `thr_search`= `True`, the GA_HW performs a grid search on each evaluation of the fitness to find the optimal value of the classification threshold which maximizes the fitness metric for that specific prediction model. 

This parameter is irrelevant in the case of selecting the 'auc' as fitness, since no specific classification threshold is used in its evaluation. In other cases, activating the grid search might result in a more optimal solution found by the algorithm, since a optimal screener with a classification threshold different than 0.5 might outperform the more simple solution. In any case, the grid search feature will never provide a worst solution, since the value 0.5 is included on the searched interval.

The user should be aware that turning this parameter `True` will considerably increase computation time, since the number of evaluations of the fitness function is multiplied by 10 for each screener.