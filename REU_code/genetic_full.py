# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 09:07:11 2017

@author: gandhik
"""

import cmath
import random
import time
imag = cmath.sqrt(-1)
from conv_utils import con_psl, gen_population, second_max, fast_autocorrelations
from crossover_algs import crossover_random, crossover_halfhalf, crossover_randpoint
from memetic_tabu_search import k_select_parent
from anneal_full import gen_neighbor

def select_parents_random(population, retain, random_select, k, N=1, phase_unity = 2, num_elem = None, min_fn = second_max):
    """
    Function to select the parents from the population, via a random weighted selection algorithm.

    :param population: list of arrays comprising our population
    :type population: array
    :param retain: fraction of population that we retain as is, unchanged
    :type retain: float
    :param random_select: unused variable
    :param k: unused variable
    :param N: doppler width
    :type N: int
    :param phase_unity: order of the roots of unity in our codes
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: the population after the selection process, in the array parents
    """
    # sort the population by min_fn value
    psl_values = [(fast_autocorrelations(x, N, num_elem, min_fn), x) for x in population]
    pop_length = len(psl_values)
    psl_values.sort(key=lambda x: x[0])
    seqs_sorted_by_psl = [x[1] for x in psl_values]
    num_retain = int(pop_length * retain) # number of parents we will retain
    min_fitness = psl_values[0][0]
    parents = []
    count = 0
    while count <num_retain: # until we reach num_retain members, select a random parent, add it with probability weighted by its min_fn value.
        select_point = random.randint(0, pop_length - 1)
        poss_parent = seqs_sorted_by_psl[select_point]
        prob = min_fitness/fast_autocorrelations(poss_parent, N, num_elem, min_fn)
        if prob>random.random():
            count+=1
            parents.append(poss_parent)
        
    return parents

def select_parents_part_random(population, retain, random_select, k, N=1, phase_unity = 2, num_elem = None, min_fn = second_max):
    """
    Function to select the parents from the population, via a partially random, partially fitness-based selection process.

    :param population: list of arrays comprising our population
    :type population: array
    :param retain: fraction of population that we retain as is, unchanged
    :type retain: float
    :param random_select: probability for selecting additional parents, in addition to the most fit
    :type random_select: float
    :param k: unused variable
    :param N: doppler width
    :type N: int
    :param phase_unity: order of the roots of unity in our codes
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: the population after the selection process, in the array parents
    """
    # sort the population by min_fn value
    psl_values = [(fast_autocorrelations(x, N, num_elem, min_fn), x) for x in population]
    psl_values.sort(key=lambda x: x[0])
    seqs_sorted_by_psl = [x[1] for x in psl_values]
    num_retain = int(len(seqs_sorted_by_psl) * retain)
    parents = seqs_sorted_by_psl[:num_retain] # add the num_retain most fit parents

    for individual in seqs_sorted_by_psl[num_retain:]: # add a randomly selected additional set of parents from the population
        if random_select > random.random():
            parents.append(individual)

    return parents

def tournament_selection(population, retain, random_select, k, N=1, phase_unity = 2, num_elem = None, min_fn = second_max):
    """
    Function to select the parents from the population, via a partially random, partially fitness-based selection process.

    :param population: list of arrays comprising our population
    :type population: array
    :param retain: fraction of population that we retain as is, unchanged
    :type retain: float
    :param random_select: unused variable
    :param k: number of members for tournament selection
    :type k: int
    :param N: doppler width
    :type N: int
    :param phase_unity: order of the roots of unity in our codes
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: the population after the selection process, in the array parents
    """
    num_retain = int(len(population) * retain)
    parents = []
    for i in range(num_retain): # add num_retain tournament-selected parents
        parents.append(population[k_select_parent(population, k, N, phase_unity, num_elem, min_fn)])

    return parents

def mutate_all(parents, mutate, phase_unity = 2):
    """
    Function to mutate a parent population.  For each parent, there is a _mutate_
    probability the parent will be mutated to a direct neighbor.
    
    :param parents: population of parents to mutate
    :type parents: array
    :param mutate: probability of mutating a given parent
    :type mutate: float
    :param phase_unity: order of the roots of unity in our codes
    :type phase_unity: int
    """
    for i in parents: # for all codes in parents, mutate it with some probability by moving it to a direct neighbor.
        if mutate > random.random():
            i = gen_neighbor(i, phase_unity, 1)
    
    return parents    

crossover_funcs = [crossover_random, crossover_halfhalf, crossover_randpoint]
selection_funcs = [select_parents_part_random, select_parents_random]
mutate_funcs = [mutate_all]

def evolve(population, parent_func, mut_func, crossover_func, retain, random_select, mutate, k, N=1, phase_unity = 2, num_elem = None, min_fn = second_max):
    """
    The evolution step of our algorithm, which selects parents, mutates them, and then crosses them over to create the resultant population.
    
    :param population: array of codes at a given generational step
    :type population: array
    :param parent_func: function to select the parents of the next generation
    :type parent_func: function
    :param mut_func: function to mutate the members of a population
    :type mut_func: function
    :param crossover_func: function to crossover two parents to create a child
    :type crossover_func: function
    :param retain: fraction of parents to retain
    :type retain: float
    :param random_select: fraction of parents to randomly select, if necessary
    :type random_select: float
    :param mutate: fraction of population to mutate
    :type mutate: float
    :param k: number of members for tournament selection
    :type k: int
    :param N: doppler width
    :type N: int
    :param phase_unity: order of the roots of unity in our codes
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: the updated set of parents
    """
    parents = parent_func(population, retain, random_select, k, N, phase_unity, num_elem, min_fn)
    parents = mut_func(parents, mutate, phase_unity)
    parents = crossover_func(len(population), parents)
    return parents

def run_evolution(length, num_members = 100, num_gens = 25, parent_func = select_parents_part_random, mut_func = mutate_all, crossover_func = crossover_random, retain = 0.2, random_select = 0.05, mutate = 0.02, k = 2, N=1, phase_unity = 2, num_elem = None, min_fn = second_max):
    """
    Handler for running evolution; for a fixed number of number of generations, generates a population, evolves it, and returns the best code found.

    :param length: length of the codes in question
    :type length: int
    :param num_members: number of members to put in the population
    :type num_members: int
    :param parent_func: function to select the parents of the next generation
    :type parent_func: function
    :param mut_func: function to mutate the members of a population
    :type mut_func: function
    :param crossover_func: function to crossover two parents to create a child
    :type crossover_func: function
    :param retain: fraction of parents to retain
    :type retain: float
    :param random_select: fraction of parents to randomly select, if necessary
    :type random_select: float
    :param mutate: fraction of population to mutate
    :type mutate: float
    :param k: number of members for tournament selection
    :type k: int
    :param N: doppler width
    :type N: int
    :param phase_unity: order of the roots of unity in our codes
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: (best_code, best_val): a tuple of a complex value and numpy array representing the minimum min_fn code found
    """

    f = open('best_known_sidelobes.txt') # best known sidelobes in the literature
    lines = f.readlines()
    best_code = None
    best_val = float('inf')
    cutoff_val = int(lines[length - 1]) # best possible value our algorithm can find
    time_limit = 0
    if length <= 30:
        time_limit = 0.5*300
    else:
        time_limit = 0.5 * (300 + 60 * (length - 30))
    start_time = time.time()
    while time.time() - start_time < time_limit and (best_val != cutoff_val or N != 1 or phase_unity != 2):
        pop = gen_population(num_members, length, phase_unity) # generate a population of codes
        for i in range(num_gens):
            pop = evolve(pop, parent_func, mut_func, crossover_func, retain, random_select, mutate, k) # evolve that population
            # sort the population by min_fn, if best fitness improves on the previous best, update best_val.
            psl_values = [(fast_autocorrelations(x, N, num_elem, min_fn), x) for x in pop]
            psl_values.sort(key=lambda x: x[0])
            seqs_sorted_by_psl = [x[1] for x in psl_values]
            new_val = fast_autocorrelations(seqs_sorted_by_psl[0], N, num_elem, min_fn)
            if new_val < best_val:  # keep track of the best value found so far, in case we lose it.
                best_code = seqs_sorted_by_psl[0]
                best_val = new_val

            if best_val == cutoff_val:
                break

    return (best_code, best_val)
