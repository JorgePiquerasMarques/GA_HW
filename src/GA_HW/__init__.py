'''
                    -GENETIC ALGORITHM FOR SCREENING OPTIMIZATION IN CATEGORICAL DIAGNOSTIC SURVEYS-

In order to run the algorithm, prepare a dataset in pandas dataframe format with respondents as rows and variables as columns.
Keep this file in the same directory as the Python script you are running.

Call the following function to run the GA:

results = GA_HW.opt(db, variable_names, dep_variable, n, N, max_gen, cr_prob, fit_fun, thr_search)

    inputs:
        db: pandas dataframe. When using categorical variables, dummy encoding must be performed. For instance:
            db_dummies = pd.get_dummies(db, drop_first=True)
        variable_names: list of string names of all independent variables (predictors, questions). Names must coincide with the 
            column names of variables in the dataframe
        dep_variable: string name of the dependent/response variable. Must be dichotomous variable and be one of the columns on the db.
        n: int. size of the screener to search
        N: int. population size
        max_gen: int. number of maximum generations (iterations of the GA)
        cr_prob: float. probability of cross-over (0-1)
        fit_fun: string. fitness function ot optimize ('auc', 'f2' or 'mcc')
        thr_search: Boolean. Whether to perform a grid search for the optimal classification threshold (True) or not (False)

    outputs:
        results: object with the following attributes
            fitness_list: list of the fitness values of the fittests individuals on each generation
            mean_fitness_list: list of the mean fitness of each generation.
            names: names of variables included in the solution
            threshold: optimal classification threshold of the DT model
            fitness: fitness of the optimal solution
            fitness_sem: sem of the fitness of the optimal solution
'''

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import roc_curve, auc, fbeta_score, matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import random

class individual:
    ''' 
        This is a class to store the individuals' genes and fitness/thr
    '''
    def __init__(self, chrom, fitness, fitness_sem, threshold):
        self.chrom = chrom
        self.fitness = fitness
        self.fitness_sem = fitness_sem
        self.threshold = threshold
        
class result:
    '''
        This is a class to store the final results of the model
    '''
    def __init__(self, fitness_list, mean_fitness_list, names, fitness, fitness_sem, threshold):
        self.fitness_list = fitness_list
        self.mean_fitness_list = mean_fitness_list
        self.names = names
        self.fitness = fitness
        self.fitness_sem = fitness_sem
        self.threshold = threshold
                
def Kfoldsets(db):
    '''
        This function takes the original database does the split into 10 subsets of equal size. 
        Returns a list of 10 datasets.
        
        input: dataframe
        output: list
    '''
 
    r = 42
    X_sets = []
    train_i = db
    
    for i in range(0,9):
        if (i == 8):
            set_i, set_j = train_test_split(train_i, test_size=1/(10-i), random_state=r)
            X_sets.append(set_i)
            X_sets.append(set_j)
        else:
            train_i, set_i = train_test_split(train_i, test_size=1/(10-i), random_state=r)
            X_sets.append(set_i)
        
    return X_sets
    
def Kfold_DT(dep_variable, X_sets, variables, f):
    '''
        This function takes an individual's variable list and performs a 10-Fold cross validation on a DecisionTreeClassifier.
        Returns the mean of the metric selected as fitness.
        input: dataframe, list, list, int
        output: float
    '''
    
    x_sets = X_sets
    
    if (f==1):
        # If AUC is the fitness metric, no tuning of the threshold is required, so the 10-Fold is performed only once
        metrics = []
        for k in range(0,10):
            x = [x_sets[i][variables] for i in range(0,10)]
            y = [x_sets[i][dep_variable] for i in range(0,10)]
       
            x.pop(k)
            y.pop(k)
        
            x_train = pd.concat(x, ignore_index=True)
            x_test = X_sets[k][variables]
        
            y_train = pd.concat(y, ignore_index=True)
            y_test = X_sets[k][dep_variable]
            
            # Fit the prediction model on the k fold
            clf = DecisionTreeClassifier(max_depth = len(variables), class_weight = "balanced", random_state=42)
        
            clf.fit(x_train,y_train)
            # Obtain predicted probabilities on the k fold
            Y_pred_proba = clf.predict_proba(x_test)[:,1]
            fpr,tpr,thresholds = roc_curve(y_test, Y_pred_proba)
            metrics.append(auc(fpr, tpr))
            
        # Calculate the fitness value as the mean of the k-estimates
        result = np.mean(metrics)
        result_sem = scipy.stats.sem(metrics)
        best_thr = 0 # thr is set to an arbitrary number
    else:
        # If F2 or MCC are the fitness functions, a search for the optimal threshold is performed
        best_thr_metric = 0
        best_thr = 0
        for thr in np.linspace(0.1,0.9,9):
            metrics = []
            # Perform 10-Fold crossvalidation to estimate the fitness of each threshold
            for k in range(0,10):
        
                x = [x_sets[i][variables] for i in range(0,10)]
                y = [x_sets[i][dep_variable] for i in range(0,10)]
       
                x.pop(k)
                y.pop(k)
        
                x_train = pd.concat(x, ignore_index=True)
                x_test = X_sets[k][variables]
        
                y_train = pd.concat(y, ignore_index=True)
                y_test = X_sets[k][dep_variable]
        
                clf = DecisionTreeClassifier(max_depth = len(variables), class_weight = "balanced", random_state=42)
        
                clf.fit(x_train,y_train)

                Y_pred_proba = clf.predict_proba(x_test)[:,1]
                
                # Construct the binary classification for the threshold 'thr'
                Y_pred = []
                for i in range(0, len(Y_pred_proba)):
                    if (Y_pred_proba[i]>=thr):
                        Y_pred.append(1)
                    else:
                        Y_pred.append(0)
                
                # Use the selected fitness function to estimate the metric on the k-fold of the threshold thr
                if (f == 2): 
                    metrics.append(fbeta_score(y_test, Y_pred, average='binary', beta=2))
                elif (f == 3):
                    metrics.append(matthews_corrcoef(y_test, Y_pred))
            
            # The fitness value of the model using thr is calculated as the mean of the k-estimates
            thr_metric = np.mean(metrics)
            
            # Check if the fitness obtained for thr is the best yet
            if (thr_metric>=best_thr_metric):
                best_thr_metric = thr_metric
                best_thr_sem = scipy.stats.sem(metrics)
                best_thr = thr
                
        # The fitness value corresponds to that of the best thr model
        result = best_thr_metric
        result_sem = best_thr_sem
        

    return result, result_sem, best_thr

def Kfold_DT_nothr(dep_variable, X_sets, variables, f):
    '''
        This function takes an individual's variable list and performs a 10-Fold cross validation on a DecisionTreeClassifier.
        Returns the mean of the metric selected as fitness.
        input: dataframe, list, list, int
        output: float
    '''
    
    x_sets = X_sets
    
    # no tuning of the threshold is required, so the 10-Fold is performed only once
    metrics = []
    for k in range(0,10):
        x = [x_sets[i][variables] for i in range(0,10)]
        y = [x_sets[i][dep_variable] for i in range(0,10)]
       
        x.pop(k)
        y.pop(k)
        
        x_train = pd.concat(x, ignore_index=True)
        x_test = X_sets[k][variables]
        
        y_train = pd.concat(y, ignore_index=True)
        y_test = X_sets[k][dep_variable]
            
        # Fit the prediction model on the k fold
        clf = DecisionTreeClassifier(max_depth = len(variables), class_weight = "balanced", random_state=42)
        
        clf.fit(x_train,y_train)
        # Obtain predicted probabilities on the k fold
        if (f==1):
            Y_pred_proba = clf.predict_proba(x_test)[:,1]
            fpr,tpr,thresholds = roc_curve(y_test, Y_pred_proba)
            metrics.append(auc(fpr, tpr))
        elif(f==2):
            Y_pred = clf.predict(x_test)
            metrics.append(fbeta_score(y_test, Y_pred, average='binary', beta=2))
        elif(f==3):
            Y_pred = clf.predict(x_test)
            metrics.append(matthews_corrcoef(y_test, Y_pred))
                
    # Calculate the fitness value as the mean of the k-estimates
    result = np.mean(metrics)
    result_sem = scipy.stats.sem(metrics)
    best_thr = 0.5 # thr is set to an arbitrary number
        
    return result, result_sem, best_thr


def get_fitness(dep_variable, variable_names, X_sets, chrom, f, thr_search):
    ''' 
        This function transforms the bitstring into a variable list and computes the fitness value using the accuracy metrics from the               classification algorithm.
        
        input: string, list, list, list, int
        output: float, float
    '''
    
    # First, we transform the bitstring into a list of variables
    variables = []
    inds = [ind for ind, bit in enumerate(chrom) if bit== 1]
    for ind in inds:
        variables.append(variable_names[ind])
    
    # Then, the function which evaluates the fitness is called
    
    if (thr_search):
        fitness, fitness_sem, thr = Kfold_DT(dep_variable, X_sets, variables, f)
    else:
        fitness, fitness_sem, thr = Kfold_DT_nothr(dep_variable, X_sets, variables, f)
    
    return fitness, fitness_sem, thr
    
    
def random_ind(dep_variable, variable_names, n, f, X_sets, thr_search):
    ''' 
        This function creates a random individual of a fixed hamming weight n.
        
        input: string, list, int, int, list
        output: intividual     
    '''
    # First, the random question list is generated with HW=n
    chrom = ([1]*n) + ([0]*(len(variable_names)-n))
    random.shuffle(chrom)
    
    # Then, the individual is created including its fitness and the thr of the optimal classification
    fitness, fitness_sem, threshold = get_fitness(dep_variable, variable_names, X_sets, chrom, f, thr_search)
    ind = individual(chrom, fitness, fitness_sem, threshold)
    
    return ind
    
def init_population(dep_variable, variable_names, n, N, f, X_sets, thr_search):
    '''
        This function initialices the population, by creating a list of N individuals, of Hamming weight n
        
        input: string, list, int, int, int, list
        output: list
    '''
        
    population = []
    for i in range(0, N):
        population.append(random_ind(dep_variable, variable_names, n, f, X_sets, thr_search))
        
    return population

def Map1Crossover(variable_names, p1, p2, n):
    '''
       This function performs Map of 1s Crossover. Receives the two parents, and returns the resulting children chromosome
        
       input: list, individual, individual, int
       output: list
    '''
    
    # Transform chromosomes into lists of indexes of 1s
    inds_par1 = [ind for ind, bit in enumerate(p1.chrom) if bit == 1]
    inds_par2 = [ind for ind, bit in enumerate(p2.chrom) if bit == 1]
    
    # Map of 1s Crossover algorithm:
    
    # Map of ones of the new child
    inds_child = []
    # Intersection list of the parents
    comm_list = list(set(inds_par1).intersection(inds_par2))
    
    for i in range(0, n):
        p = random.uniform(0,1)
        
        #Randomly select a parent
        if (p<0.5):
            cpar = inds_par1
        else:
            cpar = inds_par2
        
        # Randomly select an element from the selected parent map of ones
        cind = random.choice(cpar)
        # Include the element on the child's map of ones
        inds_child.append(cind)
        
        # Remove the element from one or both of the parent's map of ones
        if (p<0.5):
            inds_par1.remove(cind)
            if (cind in comm_list):
                inds_par2.remove(cind)
        else:
            inds_par2.remove(cind)
            if (cind in comm_list):
                inds_par1.remove(cind)
        
    # Transform back into chromosome
    child_chrom = []
    for i in range(0, len(variable_names)):
        if (i in inds_child):
            child_chrom.append(1)
        else:
            child_chrom.append(0)
    
    return child_chrom

def Map1Crossover_Mutation(variable_names, parent1, parent2, n, prob, g, g_max):
    '''
        This function performs Map of 1s Crossover of two individuals, with a probability 'prob', then calls the 
        Mutation function after calculating the probability with a dynamic increasing model. Receives two individuals and returns the               variable list of the children.
        
        input: list, individual, individual, int, float, int, int
        output: list, int
    '''
    # checks if cross-over and mutation happens (0:happened, 1,2: did not happen and parent_check was copied)
    check_cr = 0
    # Random float to compare with the crossover probability
    p = random.uniform(0,1)
    
    if(p < prob):
        child_chrom = Map1Crossover(variable_names, parent1, parent2, n)     # Call the crossover function
    else:                                                    # Crossover doesn't happen, so the best of the two parents goes on
        if(parent1.fitness >= parent2.fitness):   
            check_cr = 1
            child_chrom = parent1.chrom
        else:
            check_cr = 2
            child_chrom = parent2.chrom
            
    # We define the mutation probability as an increasing function in terms of the generation number 
    a = 0.2
    b = 0.12
    m = a + (b*((g-g_max)/g_max))
        
    mchild_chrom, check = Mutation(variable_names, child_chrom, m, check_cr)
    
    return mchild_chrom, check

def Mutation(variable_names, chrom, prob, check_cr):
    ''' 
        This functions performs Mutation on an chromosome, with a probability 'prob'. Returns the mutated chromosome.
        
        input: list, list, float
        output: list
    '''
    
    # Random float to compare with the mutation probability
    p = random.uniform(0,1)
    # Get list of indexes of 1s, and 0s
    inds_par = [ind for ind, bit in enumerate(chrom) if bit== 1]
    inds_par_0 = [ind for ind, bit in enumerate(chrom) if bit== 0]
    
    if(p<prob):
        check_cr = 0  # Mutation did happen, so set the check to 0
        pos_ind = random.randint(0,len(inds_par)-1)
        # Eliminate a 1 at random
        inds_par.pop(pos_ind)
       
        neg_ind = random.randint(0,len(inds_par_0)-1)
        # Transform a random 0 into a 1
        inds_par.append(inds_par_0[neg_ind])
        
        # Transform back to bitstring
        new_chrom = []
        for i in range(0, len(variable_names)):
            if (i in inds_par):
                new_chrom.append(1)
            else:
                new_chrom.append(0)
    else:  # Mutation does not happen, so the original chromosome is returned and check is mantained
        new_chrom = chrom
    
    return new_chrom, check_cr
    
def tournament_selection(pop):
    '''
        This function performs a tournament selection from a population of individuals. It takes two random individuals from the
        pool and returns the fittest one.
        
        input: list
        output: individual
    '''
    
    # Select two individuals from the population at random
    first_fighter = random.choice(pop)
    second_fighter = random.choice(pop)
    
    if(first_fighter.fitness >= second_fighter.fitness):    # Check for the fittest (winner)
        winner = first_fighter
    else:
        winner = second_fighter
    
    return winner
    
def get_results_pop(variable_names, pop):
    '''
        This function takes a population (list of individuals) and finds the mean fitness, and also the fitness and variable
        list of the fittest individual (best)
        
        input: list, list
        output: individual, list, float
    '''
    
    fitness_array = np.array([ind.fitness for ind in pop])
    mean_fitness = np.mean(fitness_array)
    sort_index = np.argsort(fitness_array)
    best = pop[sort_index[-1]]
            
    best_vars = []
    inds = [ind for ind, bit in enumerate(best.chrom) if bit== 1]
    for ind in inds:
        best_vars.append(variable_names[ind])
    
    best_thr = best.threshold
    
    return best, best_vars, mean_fitness, best_thr
    

def GeneticAlgorithm(dep_variable, variable_names, population, db, n, N, max_gen, f, X_sets, cr_prob, thr_search):
    
    generation = 0
    
    fitness_list = []
    mean_fitness_list = []
    
    while(generation <= max_gen):        
        next_gen = []
        
        print(f"\n\n------------- Generation number: {generation} -------------\n")
 
        fitness_array = np.array([ind.fitness for ind in population])
        sort_index = np.argsort(fitness_array)
        best = population[sort_index[-1]]
        second_best = population[sort_index[-2]]
        
        aux = population
        
        # Starts a loop, each iteration adds a new individual to the next generation
        for i in range(0, N-1):
            if(i==0):                       # First 2 individuals correspond to the fittest (Elitism)
                next_gen = [*next_gen, best, second_best]
            else:
                # First, we choose the two parent chromosomes using tournament selection
                parent1 = tournament_selection(aux)
                parent2 = tournament_selection(aux)
                
                # Second, the child is created using crossover and mutation
                child_chrom, check = Map1Crossover_Mutation(variable_names, parent1, parent2, n, cr_prob, generation, max_gen)
               
                # Third, the next generation is updated, including the previous list plus the new individual
                if (check==0):
                    # The new individual suffered a cross-over or mutation
                    fitness, fitness_sem, threshold = get_fitness(dep_variable, variable_names, X_sets, child_chrom, f, thr_search)
                    next_gen = [*next_gen, individual(child_chrom, fitness, fitness_sem, threshold)]
                elif (check==1):
                    # The new individual is a copy of parent1 
                    next_gen = [*next_gen, parent1]
                else:
                    # The new individual is a copy of parent2
                    next_gen = [*next_gen, parent2]
            
        
        # Gets results for initial population
        if(generation==0):
            best, best_vars, mean_fitness, best_thr = get_results_pop(variable_names, population)
             
            print(f"\nFittest individual phenotype of initial population: {best_vars}")
            if (f==1):            
                # If the AUC is the fitness function, threshold value is irrelevant
                print(f"\nFittest individual fitness value of initial population = {best.fitness}")
            else:
                print(f"\nFittest individual fitness value of initial population = {best.fitness}, with threshold = {best_thr}")
            print(f"\nMean fitness of initial population = {mean_fitness}")
        
        population = next_gen
        
        # Get results for best generation
        
        best, best_vars, mean_fitness, best_thr = get_results_pop(variable_names, population)
            
        print(f"\nFittest individual phenotype: {best_vars}")
        if (f==1):    
            # If the AUC is the fitness function, threshold value is irrelevant
            print(f"\nFittest individual fitness value = {best.fitness}")
        else:
            print(f"\nFittest individual fitness value = {best.fitness}, with threshold = {best_thr}")
        print(f"\nMean fitness = {mean_fitness}")
        
        # Each generation, we keep track of the fitness of the optimal solution and of the mean fitness of the population
        fitness_list.append(best.fitness)
        mean_fitness_list.append(mean_fitness)
        
       
        
        generation = generation + 1            
    
    results = result(fitness_list, mean_fitness_list, best_vars, best.fitness, best.fitness_sem, best.threshold)
    
    return results
    


def opt(db, variable_names, dep_variable, n, N=250, max_gen=10, cr_prob=0.6, fit_fun='auc', thr_search=False):
    if (fit_fun=='auc'): f=1
    elif (fit_fun=='f2'): f=2
    elif (fit_fun=='mcc'): f=3
    else: 
        print(f"Unspecified fitness function. Stopping execution.")
        return 0
    
    print(f"Preparing the dataset for 10-Fold cross-validation...")
    X_sets = Kfoldsets(db)    # Creation of the 10 sets for the 10-fold CV
    print(f"Dataset ready!")
    print(f"Generating the first population...")
    population = init_population(dep_variable, variable_names, n, N, f, X_sets, thr_search)    
    print(f"Initialization of the population completed!")
        
    results = GeneticAlgorithm(dep_variable, variable_names, population, db, n, N, max_gen, f, X_sets, cr_prob, thr_search)
    
    return results
    