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
from alternating_codes import autocorrelation_addition, gen_rand_binary_seq
import pandas as pd

imag = cmath.sqrt(-1)


# Note that this code is based on http://www.lcc.uma.es/~afdez/Papers/labsASC.pdf, and follows it in many ways.  The
# code is adapted to this problem and extended to doppler, polyphase, other minimization functions, parameters are chosen,
# and some of the algorithm is changed.

def ts_local_search(child, num):
    length = len(child)
    tabu_table = [0 for x in range(length)]  # table to store most recently changed coordinates
    max_iters = 20
    min_iters = 2
    extra_iters = 1
    best_code = np.copy(child)
    best_val = autocorrelation_addition(child, num)
    for k in range(max_iters):
        code_this_gen = np.array([])
        val_this_gen = float('inf')
        which_i = None
        for i in range(length):  # loop over all coordinates
            seq_new = np.copy(child)
            seq_new[i] = -seq_new[i]
            val = autocorrelation_addition(seq_new, num)
            if k >= tabu_table[i] or val < best_val:  # if it improves on the best value or it hasn't beeen changed in a while, move to that code.
                if val < val_this_gen:
                    val_this_gen = val
                    code_this_gen = seq_new
                    which_i = i
        if code_this_gen.any():
            child = np.copy(code_this_gen)
        if which_i != None:  # increase value of tabu_table for the coordinate that changed
            tabu_table[which_i] = k + min_iters + random.randint(0, extra_iters)
        if val_this_gen < best_val:
            best_code = np.copy(code_this_gen)
            best_val = val_this_gen
    return best_code


def select_parent(population, num):
    poss_parents = random.sample(range(0, len(population)), 2)
    if autocorrelation_addition(population[poss_parents[0]], num) < autocorrelation_addition(population[poss_parents[1]], num):
        return poss_parents[0]
    else:
        return poss_parents[1]


def gen_alt_population(num_members, length):
    return np.array([gen_rand_binary_seq(length) for x in xrange(num_members)])

def run_evolution_memet_alternating(length, local_search_type, num, amount_retain=0.1):
    best_code = None
    best_val = float('inf')
    time_limit = 0
    if length <= 30:
        time_limit = 1500
    else:
        time_limit = 1 * (300 + 60 * (length - 30))
    start_time = time.time()
    while time.time() - start_time < time_limit:
        partial_restart = 7
        num_members = 23  # number of members of our population
        num_generations = 20
        mut_prob = 1. / length
        offspring = 88  # number of offspring created at each step
        pop = gen_alt_population(num_members, num*length)  # initial population
        amount_restart = int(0.4 * length)
        for i in range(num_generations):
            if i % partial_restart == 0:
                pop_add = gen_alt_population(amount_restart, num*length)
                pop = np.concatenate((pop, pop_add), 0)
            pop = evolve_mem(pop, num, mut_prob, offspring, local_search_type, amount_retain)  # main step in the code: evolving the population, giving an offspring size, initial pop, and mutation probability
            psl_values = [(autocorrelation_addition(x, num), x) for x in pop]
            psl_values.sort(key=lambda x: x[0])
            seqs_sorted_by_psl = [x[1] for x in psl_values]
            new_val = autocorrelation_addition(seqs_sorted_by_psl[0], num)
            if new_val < best_val:  # keep track of the best value found so far, in case we lose it.
                best_code = seqs_sorted_by_psl[0]
                best_val = new_val
            if best_val == 0:
                break
        if best_val == 0:
            break

    print(time.time() - start_time)
    return (best_code, best_val)


def crossover_random_single(population, father, mother):
    father = population[father]
    mother = population[mother]
    child = []
    for i in range(len(father)):
        m_or_f = random.sample([-1, 1], 1)
        if m_or_f == 1:
            child.append(mother[i])
        else:
            child.append(father[i])

    return child

def gen_binary_neighbor(code, movement_length = 1):
    seq_2 = np.copy(code) # copy of the sequence in question
    mut_points = random.sample(range(0, len(seq_2)), movement_length) # coordinates to mutate
    for i in mut_points:
        seq_2[i] = -seq_2[i]
    return seq_2

def evolve_mem(population, num, mut_prob, offspring, local_search_type, amount_retain=0.1):
    children = []
    while len(children) < offspring:
        if random.random() > amount_retain:  # we select the child from the current poulation
            child = population[select_parent(population, num)]
        else:  # we crossover two parents
            parent_1 = select_parent(population, num)
            parent_2 = select_parent(population, num)
            child = crossover_random_single(population, parent_1, parent_2)

        if random.random() > mut_prob:  # mutate this child with mut_prob probability
            child = gen_binary_neighbor(child, 2)

        child = local_search_type(child, num)  # we locally search to find the best local solution for this child

        children.append(child)

    children = children[:len(population)]
    return children

def memetic_alternating_test_0():
    print('memetic_alternating')
    length = []
    number_of_codes = []
    memetic_codes = []
    memetic_values = []
    for i in range(8, 14):
        print(i)
        for j in range(4, 10, 2):
            print(j)
            length.append(i)
            number_of_codes.append(j)
            x = run_evolution_memet_alternating(i, ts_local_search, j, 0.1)
            memetic_codes.append(x[0])
            memetic_values.append(x[1])
            print(x[1])

    df = pd.DataFrame({'Length': length, 'Number of codes': number_of_codes,
                       'Code': memetic_codes, 'PSL Value': memetic_values})
    writer = pd.ExcelWriter('memetic_alternating_0.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def memetic_alternating_test_1():
    print('memetic_alternating')
    length = []
    number_of_codes = []
    memetic_codes = []
    memetic_values = []
    for i in range(14, 18):
        print(i)
        for j in range(4, 10, 2):
            print(j)
            length.append(i)
            number_of_codes.append(j)
            x = run_evolution_memet_alternating(i, ts_local_search, j, 0.1)
            memetic_codes.append(x[0])
            memetic_values.append(x[1])
            print(x[1])

    df = pd.DataFrame({'Length': length, 'Number of codes': number_of_codes,
                       'Code': memetic_codes, 'PSL Value': memetic_values})
    writer = pd.ExcelWriter('memetic_alternating_1.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def memetic_alternating_test_2():
    print('memetic_alternating')
    length = []
    number_of_codes = []
    memetic_codes = []
    memetic_values = []
    for i in range(18, 22):
        print(i)
        for j in range(4, 10, 2):
            print(j)
            length.append(i)
            number_of_codes.append(j)
            x = run_evolution_memet_alternating(i, ts_local_search, j, 0.1)
            memetic_codes.append(x[0])
            memetic_values.append(x[1])
            print(x[1])

    df = pd.DataFrame({'Length': length, 'Number of codes': number_of_codes,
                       'Code': memetic_codes, 'PSL Value': memetic_values})
    writer = pd.ExcelWriter('memetic_alternating_2.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def memetic_alternating_test_4():
    print('memetic_alternating')
    length = []
    number_of_codes = []
    memetic_codes = []
    memetic_values = []
    for i in range(16, 21):
        print(i)
        for j in range(4, 16, 2):
            print(j)
            length.append(i)
            number_of_codes.append(j)
            x = run_evolution_memet_alternating(i, ts_local_search, j, 0.1)
            memetic_codes.append(x[0])
            memetic_values.append(x[1])
            print(x[1])

    df = pd.DataFrame({'Length': length, 'Number of codes': number_of_codes,
                       'Code': memetic_codes, 'PSL Value': memetic_values})
    writer = pd.ExcelWriter('memetic_alternating_3.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()


memetic_alternating_test_4()

