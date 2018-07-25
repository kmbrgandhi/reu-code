# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:16:29 2017

@author: gandhik
"""
import random
import time
import cmath
import numpy as np
from conv_utils import con_psl, gen_population, fast_autocorrelations, gen_rand_seq, second_max
from anneal_full import gen_neighbor
from hill_climbing import sdls_local_search
imag = cmath.sqrt(-1)

# Note that this code is based on http://www.lcc.uma.es/~afdez/Papers/labsASC.pdf, and follows it in many ways.  The
# code is adapted to this problem and extended to doppler, polyphase, other minimization functions, parameters are chosen,
# and some of the algorithm is changed.

def ts_local_search(child, N=1, phase_unity = 2, num_elem = None, min_fn = second_max):
    """
    Tabu local search algorithm; works by climbing to better neighbors, excluding those with coordinates in a tabu,
    for a number of iterations.

    :param child: initial sequence
    :type child: array
    :param N: doppler width of our ambiguity function
    :type N: int
    :param phase_unity: order of the roots of unity filling our codes
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: locally near-optimal value near child, found by tabu local search, which allows for more potential movement
    away from a very close local minima
    """
    length = len(child)
    tabu_table = [0 for x in range(length)] # table to store most recently changed coordinates
    max_iters = 20
    min_iters = 2
    extra_iters = 1
    best_code = np.copy(child)
    best_val = fast_autocorrelations(child, N, num_elem, min_fn)
    for k in range(max_iters):
        code_this_gen = np.array([])
        val_this_gen = float('inf')
        which_i = None
        for i in range(length): # loop over all coordinates
            for j in range(1, phase_unity): # loop over all possible phase additions
                seq_new = np.copy(child)
                if phase_unity == 2:
                    seq_new[i] = -seq_new[i]
                else:
                    seq_new[i] = seq_new[i] * cmath.exp(j * 2 * cmath.pi * imag / phase_unity)
                val = fast_autocorrelations(seq_new, N, num_elem, min_fn)
                if k >= tabu_table[i] or val < best_val: # if it improves on the best value or it hasn't beeen changed in a while, move to that code.
                    if val < val_this_gen:
                        val_this_gen = val
                        code_this_gen = seq_new
                        which_i = i
        if code_this_gen.any():
            child = np.copy(code_this_gen)
        if which_i != None: # increase value of tabu_table for the coordinate that changed
            tabu_table[which_i] = k + min_iters + random.randint(0, extra_iters)
        if val_this_gen < best_val:
            best_code = np.copy(code_this_gen)
            best_val = val_this_gen
    return best_code


def select_parent(population, N=1, phase_unity = 2, num_elem = None, min_fn = second_max):
    """
    A tournament selection algorithm to select a parent, with k = 2.

    :param population: array of the codes at our current generation
    :type population: array
    :param N: doppler width of our ambiguity function
    :type N: int
    :param phase_unity: order of the roots of unity filling our codes
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: a parent for the next generation with the maximal fitness among 2 randomly selected

    """
    poss_parents = random.sample(range(0, len(population)), 2)
    if fast_autocorrelations(population[poss_parents[0]], N, num_elem, min_fn) < fast_autocorrelations(population[poss_parents[1]], N, num_elem, min_fn):
        return poss_parents[0]
    else:
        return poss_parents[1]

def k_select_parent(population, k, N=1, phase_unity = 2, num_elem = None, min_fn = second_max):
    """
        A tournament selection algorithm to select a parent, for arbitrary k.

        :param population: array of the codes at our current generation
        :type population: array
        :param k: number of members from which we are choosing
        :type k: int
        :param N: doppler width of our ambiguity function
        :type N: int
        :param phase_unity: order of the roots of unity filling our codes
        :type phase_unity: int
        :param num_elem: number of elements that min_fn is being evaluated on
        :type num_elem: int
        :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
        :type min_fn: function
        :return: a parent for the next generation with the maximal fitness among k randomly selected

    """
    poss_parents = random.sample(range(0, len(population)), k)
    poss_parents_psl = np.array([fast_autocorrelations(population[x], N, num_elem, min_fn) for x in poss_parents])
    max_index = np.argmax(poss_parents_psl) # utility of using np; argmax exists
    return poss_parents[max_index]
    
def run_evolution_memet(length, local_search_type, amount_retain = 0.1, N=1, phase_unity = 2, num_elem = None, min_fn = second_max):
    """
    Handler for memetic algorithm.

    :param length: length of the codes in question
    :type length: int
    :param local_search_type: the local search function we are using to minimize min_fn
    :type local_search_type: function
    :param amount_retain: the fraction of parents we will retain for the next generation (give or take mutations)
    :type amount_retain: float
    :param N: doppler width of our ambiguity function
    :type N: int
    :param phase_unity: order of the roots of unity filling our codes
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: (best_code, best_val): a tuple of a complex value and numpy array representing the minimum min_fn code found
    """
    f=open('best_known_sidelobes.txt') # best known sidelobes in the literature
    lines=f.readlines()
    best_code = None
    best_val = float('inf')
    cutoff_val = int(lines[length - 1]) # best min_fun value our algorithm could find
    time_limit = 0
    if length<=30:
        time_limit = 1.5*300
    else:
        time_limit = 1.5*(300  + 60 * (length - 30))
    start_time = time.time()
    while time.time() -start_time < time_limit:
        partial_restart = 8
        num_members = 100 # number of members of our population
        num_generations = 30
        mut_prob = 1./length
        offspring= 500 # number of offspring created at each step
        pop = gen_population(num_members, length, phase_unity) # initial population
        amount_restart = int(1.5 * length)
        for i in range(num_generations):
            if i % partial_restart == 0:
                pop_add = gen_population(amount_restart, length, phase_unity)
                pop = np.concatenate((pop, pop_add), 0)
            pop = evolve_mem(pop, mut_prob, offspring, local_search_type, amount_retain, N, phase_unity, num_elem, min_fn) # main step in the code: evolving the population, giving an offspring size, initial pop, and mutation probability
            psl_values = [(fast_autocorrelations(x, N, num_elem, min_fn), x) for x in pop]
            psl_values.sort(key=lambda x: x[0])
            seqs_sorted_by_psl = [x[1] for x in psl_values]
            new_val = fast_autocorrelations(seqs_sorted_by_psl[0], N, num_elem, min_fn)
            if new_val < best_val:  # keep track of the best value found so far, in case we lose it.
                best_code = seqs_sorted_by_psl[0]
                best_val = new_val
            #print((seqs_sorted_by_psl[0], con_psl(seqs_sorted_by_psl[0])))
            if best_val == cutoff_val and phase_unity==2:
                break
        if best_val == cutoff_val and phase_unity == 2:
            break
    print(time.time()-start_time)
    return (best_code, best_val)



        
def crossover_random_single(population, father, mother):
    """
    Function to calculate random crossover of two arrays
    :param population: population holding all of the codes
    :type population: np array
    :param father: the father code
    :type father: array
    :param mother: the mother code
    :type mother: array
    :return: array with each of its elements randomly chosen from mother and father
    """
    father = population[father]
    mother = population[mother]
    child = []
    for i in range(len(father)):
        m_or_f = random.sample([-1, 1],  1)
        if m_or_f == 1:
            child.append(mother[i])
        else:
            child.append(father[i])
    
    return child
    
def evolve_mem(population, mut_prob, offspring, local_search_type, amount_retain = 0.1, N=1, phase_unity = 2, num_elem = None, min_fn = second_max):
    """
    Fundamental memetic algorithm.  Works by taking the initial population, crossing over or retaining parents,
    mutating some of them, and locally maximizing each of them.

    :param population: array containing the codes at the current step of consideration
    :type population: array
    :param mut_prob: probability of mutating a given offspring
    :type mut_prob: float
    :param offspring: the number of offspring for this generation
    :type offspring: int
    :param local_search_type: the local search function we are using to minimize min_fn
    :type local_search_type: function
    :param amount_retain: the fraction of parents we will retain for the next generation (give or take mutations)
    :type amount_retain: float
    :param N: doppler width of our ambiguity function
    :type N: int
    :param phase_unity: order of the roots of unity filling our codes
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: the updated population
    """
    children = []
    while len(children) < offspring:
        if random.random() > amount_retain: # we select the child from the current poulation
            child = population[select_parent(population)]
        else: # we crossover two parents
            parent_1 = select_parent(population)
            parent_2 = select_parent(population)
            child = crossover_random_single(population, parent_1, parent_2)

        if random.random() > mut_prob: # mutate this child with mut_prob probability
            child = gen_neighbor(child, phase_unity, 2)

        child = local_search_type(child, N, phase_unity, num_elem, min_fn) # we locally search to find the best local solution for this child

        children.append(child)

    children = children[:len(population)]
    return children
