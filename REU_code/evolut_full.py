# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:25:02 2017

@author: gandhik
"""

import cmath
import numpy as np
import time
import random
from anneal_full import climb_ineff, gen_neighbor

imag = cmath.sqrt(-1)
from conv_utils import gen_population, fast_autocorrelations, second_max

def climb(seq, num_elem = None, min_fn = second_max):
    """
    Efficient algorithm to climb, in nondoppler, binary space, to the highest neighbor.  IMPORTANT NOTE: is outdated,
    does not work with updated fast_autocorrelation because the list there was cut in half.  To fix, need to change inner
    if statements below in some way.

    :param seq: initial code in question
    :type seq: array
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: the best neighbor, and the corresponding min_fn value
    """
    lst = fast_autocorrelations(seq, 1)
    current_seq = seq
    min_val = min_fn(lst, num_elem)
    for i in range(len(seq)): # looping over all possible coordinates
        lst_new = list(lst)
        length = len(lst_new)
        for j in range(length): # looping over length of autocorrelation
            if j < ((length - 1) / 2):
                if j >= i:
                    lst_new[j] = lst_new[j] - 2 * seq[i] * seq[len(seq) - 1 - j + i]  # exploiting singled changed element on left side
                if i + j + 1 >= len(seq):
                    lst_new[j] -= 2 * seq[i] * seq[i + j + 1 - len(seq)] # exploiting single changed element on right side
            elif j > ((length - 1) / 2) and j >= i:
                lst_new[j] = lst_new[length - 1 - j]
        min_pos_new = min_fn(lst_new, num_elem)
        if min_pos_new < min_val:
            min_val = min_pos_new
            seq_2 = np.copy(seq)
            seq_2[i] = -seq_2[i]
            current_seq = seq_2

    return (current_seq, min_val)

# equivalent to the above but for doppler.  would speed up most of our algorithms by a small factor, if written.
def climb_doppler(seq):
    return

# code here based on a number of evolutionary algorithms in the literature, incorporating components of them

def evolut_algo(length, N=1, phase_unity=2, num_elem = None, min_fn = second_max):
    """
    Our most in-depth evolutionary algorithm, incorporating a neighborhood selection step, a climb step, but no crossover.

    :param length: length of the codes we are considering
    :type length: int
    :param N: doppler width
    :type N: int
    :param phase_unity: order of the roots of unity in our codes
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: the best code found, along with its minimum min_fn value.
    """
    f = open('best_known_sidelobes.txt') # file with the best known sidelobes in the literature
    lines = f.readlines()
    best_code = None
    best_val = float('inf')
    cutoff_val = int(lines[length - 1]) # best possible value our algorithm can find
    time_limit = 0
    if length <= 30:
        time_limit = 2 * 300
    else:
        time_limit = 2 * (300 + 60 * (length - 30))
    start_time = time.time()
    while time.time() - start_time < time_limit and (best_val != cutoff_val or N!=1 or phase_unity!=2):
        time_to_restart = 8
        num_generations = 25
        num_members = 100
        num_offspring = 500
        pop = gen_population(num_members, length, phase_unity)
        restart_members= 15
        for i in range(len(pop)):
            pop[i] = climb_ineff(pop[i], N, phase_unity, num_elem, min_fn) # first move the population to its best state
        for i in range(1, num_generations + 1):
            # partial restart step; for the sake of introducing variance, we add some new members to the population every partial_restart steps
            if i % time_to_restart == 0:
                pop_add = gen_population(restart_members, length, phase_unity)
                for i in range(len(pop_add)):
                    pop_add[i] = climb_ineff(pop_add[i], N, phase_unity, num_elem, min_fn)
                pop = np.concatenate((pop, pop_add), 0)
            children = []
            while len(children) < num_offspring: # mutate and climb until we have sufficient offspring
                father = random.randint(0, len(pop) - 1)
                father_mut = gen_neighbor(pop[father], phase_unity, min(length-1, 3))
                father_mut_best = climb_ineff(father_mut, N, phase_unity, num_elem, min_fn)
                children.append(father_mut_best)
            pop = np.concatenate((pop, children), axis=0)
            psl_values = [(fast_autocorrelations(x, N, num_elem, min_fn), x) for x in pop]
            psl_values.sort(key=lambda x: x[0])  # can make more efficient, only need top num_members
            seqs_sorted_by_psl = [x[1] for x in psl_values]
            if psl_values[0][0] < best_val:
                best_val = psl_values[0][0]
                best_code = psl_values[0][1]
            pop = seqs_sorted_by_psl[:num_members]

            if best_val == cutoff_val:
                break

    return (best_code, best_val)

def evolut_algo_naive(length, N=1, phase_unity=2, num_elem = None, min_fn = second_max):
    """
    Our basic evolutionary algorithm, incorporating only crossover and fitness-based selection.

    :param length: length of the codes we are considering
    :type length: int
    :param N: doppler width
    :type N: int
    :param phase_unity: order of the roots of unity in our codes
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: the best code found, along with its minimum min_fn value.
    """
    f = open('best_known_sidelobes.txt') # file with the best known sidelobes in the literature
    lines = f.readlines()
    best_code = None
    best_val = float('inf')
    cutoff_val = int(lines[length - 1]) # best possible value our algorithm can find
    time_limit = 0
    if length <= 30:
        time_limit = 2 * 300
    else:
        time_limit = 2 * (300 + 60 * (length - 30))
    start_time = time.time()
    while time.time() - start_time < time_limit and (best_val != cutoff_val or N != 1 or phase_unity != 2):
        num_members = 100
        num_generations = 25
        num_offspring = 500
        pop = gen_population(num_members, length, phase_unity)
        for i in range(num_generations):
            children = []
            while len(children) < num_offspring:
                father = random.randint(0, len(pop) - 1)
                mother = random.randint(0, len(pop) - 1)
                if father != mother:
                    father = pop[father]
                    mother = pop[mother]
                    splice_point = random.randint(0, len(father) - 1) # analogue of crossover_randpoint.
                    child = np.append(father[:splice_point], mother[splice_point:])
                    children.append(child)
            pop = np.concatenate((pop, children), axis=0)
            psl_values = [(fast_autocorrelations(x, N, num_elem, min_fn), x) for x in pop]
            psl_values.sort(key=lambda x: x[0])
            seqs_sorted_by_psl = [x[1] for x in psl_values]
            if psl_values[0][0] < best_val:
                best_val = psl_values[0][0]
                best_code = psl_values[0][1]
            pop = seqs_sorted_by_psl[:num_members]

            if best_val == cutoff_val:
                break
    return (best_code, best_val)


