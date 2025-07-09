Genetic Algorithm - Hamming Weight (GA_HW)
======

Description
-----------

Python implementation of the GA_HW algorithm.

The GA_HW is a Genetic Algorithm-based method which can be used for variable selection, or more specifically for the abbreviation of a categorical interview. In the 
context of psychiatric interviews, it can be used to obtain optimal screeners for the disorder of interest.
The method considers a set of m variables, and obtains the subset of n<m variables which maximices a classification metric (fitness) by means of a :code:`DecisionTreeClassifier` model
predicting a categorical dependent variable. During the evaluation of the fitness of each subset/screener, a 10-Fold Cross-Validation is performed.

This is an early version of the algorithm which allows the selection of three different fitness functions corresponding to different optimization strategies:

- AUCROC: the algorithm optimizes over-all discrimination based on predicted probabilities
- The $F_2$ score: the algorithm optimizes the classification of positive cases (priority to recall)
- The Matthew's Correlation Coefficient: the algorithm optimizes over-all classification considering dataset imbalance


General Usage
-----

In order to run the algorithm, prepare a dataset in :code:`pandas` :code:`dataframe` format with respondents as rows and variables as columns.
Keep this file in the same directory as the :code:`GA_HW` Python script you are running. When using categorical variables, dummy encoding is highly advised. For instance:
.. code:: python
    db_dummies = pd.get_dummies(db, drop_first=True)

Call the following function to run the GA:

.. code:: python
    import GA_HW
    results = GA_HW.opt(db, variable_names, dep_variable, n, N=250, max_gen=10, cr_prob=0.6, fit_fun='auc', thr_search=False)

The four main arguments of the algorith are:

- db: :code:`pandas` dataframe with respondents as rows and variables as columns
- variable_names: list of string names of all independent variables (predictors, questions). Names must coincide with the column names of variables in the dataframe. Example, original interview with 10 questions:
.. code:: python
    variable_names = ['Q1', 'Q2, 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']

- dep_variable: string name of the dependent variable. Must coincide with a column name among variables in the dataframe.
- n: integer representing the size of the obtained subset. Corresponds to the size of the screener/number of questions/variables selected by the model.

Additional Parameters/Arguments
-----

The GA_HW also requires the following parameters:

- N: integer, size of the population on the GA. Default value is 250.
- max_gen: integer, number of generations created by the GA (iterations of the GA). Default value is 10.
- cr_prob: float, probability of cross-over inside the GA. Must be in range (0,1). Default is 0.6.
- fit_fun: string, classification metric (:code:`sklearn.metric`) selected as fitness function. Default is 'auc'. Three choices are available:
	- 'auc': AUCROC (:code:`auc`)
	- 'f2': $F_2$ metric (:code:`fbeta_score` with :code:`beta=2`)
	- 'mcc': MCC metric (:code:`matthews_corrcoef`)
- thr_search: boolean, whether to perform a grid search on the evaluation of the fitness to find the classification threshold of the :code:`DecisionTreeClassifier` model which optimizes the cross-validated estimation of the fitness metric. Default is 'False', meaning the classification threshold of the model is selected as the default value (0.5) during the classifications.


Model Outputs
------------------

The algorithm returns a list containing the following elements, in that order:

- fitness_list: a list of :code:`max_gen` floats with the fitness values of the fittest individual on each generation.
- mean_fitness_list: a list of :code:`max_gen` floats with the mean fitness values of all individuals on each generation.
- best_vars: a list of variable names included in the optimal solution, corresponding to the selected variables/optimal screener. 
- best_thr: a float corresponding to the classification threshold used by the :code:`DecisionTreeClassifier` of the optimal solution. If the argument thr_search is False, this value is always 0.5.
- best: an instance of the individual class corresponding to the optimal solution. The fitness value of the optimal solution can be accessed as :code:`best.fitness`.

FAQ
---
