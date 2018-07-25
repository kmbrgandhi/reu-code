# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:35:06 2017

@author: gandhik
"""

import time
import cmath
import math
import numpy as np
import random
imag = cmath.sqrt(-1)
from collections import deque
from conv_utils import gen_rand_seq, fast_autocorrelations, second_max
from hill_climbing import simple_local_search


# note: annealing code based on that from http://katrinaeg.com/simulated-annealing.html, but mostly changed in the process
# of applying it to this problem.  That code was not originally designed for this problem.

    
def anneal_mult(length, N, cooling_schedule, anneal_fn, phase_unity=2, num_elem = None, min_fn = second_max):
    """
    The handler for our annealing algorithm.  This function runs our annealing function a number of times, proportional to the
    length of the code, and returns the best code found over all such iterations.

    :param length: length of the codes under consideration
    :type length: int
    :param N: doppler width of our ambiguity function
    :type N: int
    :param cooling_schedule: function that describes how the temperature decreases from generation to generation
    :type cooling_schedule: function
    :param anneal_fn: specific annealing function that we will use
    :type anneal_fn: function
    :param phase_unity: order of the roots of unity that fill the codes we will be considering
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: a tuple of the best code and its corresponding min_fn value that we find
    """
    kappa = 0.5
    num = int(5 * length**(kappa))
    best_psl_total = float('inf')
    best_code = None
    f=open('best_known_sidelobes.txt') # best known sidelobes in the literature
    lines=f.readlines()
    cutoff_val = int(lines[length - 1]) # best possible value our algorithm can find
    time_limit = 0
    if length<=30:
        time_limit = 1.1*300
    else:
        time_limit = 1.1*(300  + 60 * (length - 30))
    start_time = time.time()
    while (time.time() - start_time) < time_limit and (best_psl_total != cutoff_val or N!=1 or phase_unity!=2):
        for i in range(num):
            x = gen_rand_seq(length, phase_unity) # generate a sequence
            best = anneal_fn(x, N, cooling_schedule, phase_unity, num_elem, min_fn) # anneal the sequence and find a minima, improve best_psl_total if possible
            if best[1] < best_psl_total:
                best_psl_total = best[1]
                best_code = best[0]
            if best_psl_total == cutoff_val:
                break
    print(time.time()-start_time)
    return best_psl_total
        
def anneal(code, N, cooling_schedule, phase_unity, num_elem = None, min_fn = second_max):
    """
    Our fundamental annealing algorithm, which starts with a random polyphase code (phase described by phase_unity)
    and moves around, returning the best code and min_fn value found.  The premise of the algorithm is simple:
    start with a high temperature, which allows more movement around the current code, even if it increases the min_fn
    and over time lower the temperature.  This is the bare-bones algorithm, with no extra features besides that.

    :param code: the initial code
    :type code: array
    :param N: doppler width of our ambiguity function
    :type N: int
    :param cooling_schedule: function that describes how the temperature decreases from generation to generation
    :type cooling_schedule: function
    :param phase_unity: order of the roots of unity that fill the codes we will be considering
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: a tuple of the best code and its corresponding min_fn value found by the algorithm

    See also: anneal_with_local_search, anneal_with_tabu
    """
    init_temperature = 2*len(code)
    temperature = init_temperature
    psl = fast_autocorrelations(code, N, num_elem, min_fn) # initial peak sidelobe value
    count = 0
    kappa = 0.2
    num_gen = 25 * len(code) # number of generations at a fixed temperature
    best_found = code
    best_psl = psl
    stuck_count = 0 # number of iterations we have been stuck at a given code
    num_temp_gens = int(2.9*init_temperature)
    upp_bound = 2*len(code)**(kappa)
    for j in range(1, num_temp_gens): # loop from high temperature to low temperature
        for i in range(num_gen): # loop at a given temperature of movements from neighbor to neighbor
            new_code = gen_neighbor(code, phase_unity, min(len(code)-1, 3)) # new_code found by gen_neighbor
            new_psl = fast_autocorrelations(new_code, N, num_elem, min_fn)
            if new_psl < psl: # move to the new code if the peak sidelobe value improves
                code = new_code
                stuck_count = 0 
                if new_psl < best_psl: # update best_psl as well if it is improved on
                    best_found = new_code
                    best_psl = new_psl
            else:
                if acceptance_probability(psl, new_psl, temperature) > random.random(): # otherwise, only move to the code if a randomly generated number is greater than acceptance_probability
                    code = new_code
                    stuck_count = 0
                else:
                    stuck_count += 1
            
            if stuck_count>upp_bound: # if we have been stuck for a while, break out of the loop
                count += 1
                break
        if stuck_count>upp_bound:
            break
    

        temperature = cooling_schedule(temperature, init_temperature, 0.25, num_temp_gens, j)

    return (best_found, best_psl)

def anneal_with_local_search(code, N, cooling_schedule, phase_unity, num_elem = None, min_fn = second_max):
    """
    Our fundamental annealing algorithm, which starts with a random polyphase code (phase described by phase_unity)
    and moves around, returning the best code and min_fn value found.  The premise of the algorithm is simple:
    start with a high temperature, which allows more movement around the current code, even if it increases the min_fn value
    and over time lower the temperature.  This is one altered version of the algorithm, where we perform a local search
    after generating each new code.

    Note: I do not provide additional comments here, outside of comments on the new sections of the code.

    :param code: the initial code
    :type code: array
    :param N: doppler width of our ambiguity function
    :type N: int
    :param cooling_schedule: function that describes how the temperature decreases from generation to generation
    :type cooling_schedule: function
    :param phase_unity: order of the roots of unity that fill the codes we will be considering
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: a tuple of the best code and its corresponding min_fn value found by the algorithm

    See also: anneal, anneal_with_tabu
    """
    init_temperature = 2*len(code)
    temperature = init_temperature
    psl = fast_autocorrelations(code, N, num_elem)
    count = 0
    kappa = 0.2
    num_gen = 3 * len(code)
    best_found = code
    best_psl = psl
    stuck_count = 0
    num_temp_gens = int(1.2*init_temperature)
    upp_bound = 2*len(code)**(kappa)
    for j in range(1, num_temp_gens):
        for i in range(num_gen):
            new_code = gen_neighbor(code, phase_unity, min(len(code)-1, 3))
            new_code = simple_local_search(new_code, N, phase_unity, num_elem, min_fn) # locally search and find the best point around this new neighbor
            
            new_psl = fast_autocorrelations(new_code, N, num_elem, min_fn)

            if new_psl < psl:
                code = new_code
                stuck_count = 0 
                if new_psl < best_psl:
                    best_found = new_code
                    best_psl = new_psl
            else:
                if acceptance_probability(psl, new_psl, temperature) > random.random():
                    code = new_code
                    stuck_count = 0
                else:
                    stuck_count+=1
            if stuck_count>upp_bound:
                count+=1
                break
        if stuck_count>upp_bound:
            break
        temperature = cooling_schedule(temperature, init_temperature, 0.25, num_temp_gens, j)

    return (best_found, best_psl)

def anneal_with_tabu(code, N, cooling_schedule, phase_unity, num_elem = None, min_fn = second_max):
    """
    Our fundamental annealing algorithm, which starts with a random polyphase code (phase described by phase_unity)
    and moves around, returning the best code and min_fn value found.  The premise of the algorithm is simple:
    start with a high temperature, which allows more movement around the current code, even if it increases the min_fn value
    and over time lower the temperature.  This is one altered version of the algorithm, where we store recent codes
    in a tabu and do not proceed with any normal steps when a considered code is in this tabu.

    Note: I do not provide additional comments here, outside of comments on the new sections of the code.

    :param code: the initial code
    :type code: array
    :param N: doppler width of our ambiguity function
    :type N: int
    :param cooling_schedule: function that describes how the temperature decreases from generation to generation
    :type cooling_schedule: function
    :param phase_unity: order of the roots of unity that fill the codes we will be considering
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: a tuple of the best code and its corresponding min_fn value found by the algorithm

    See also: anneal, anneal_with_local_search
    """
    tabu = deque() # initialize an empty queue
    init_temperature = 2*len(code)
    temperature = init_temperature
    psl = fast_autocorrelations(code, N, num_elem, min_fn)
    count = 0
    kappa = 0.2
    num_gen = 25 * len(code)
    best_found = code
    best_psl = psl
    stuck_count = 0
    num_temp_gens = int(2.9*init_temperature)
    upp_bound = 2*len(code)**(kappa)
    for j in range(1, num_temp_gens):
        for i in range(num_gen):
            new_code = gen_neighbor(code, phase_unity, min(len(code)-1, 3))
            new_psl = fast_autocorrelations(new_code, N, num_elem, min_fn)
            list_v = new_code.tolist()
            if list_v not in tabu: # if the new_code is not in the tabu, we can proceed in the normal steps
                if new_psl < psl:
                    code = new_code
                    stuck_count = 0 
                    if new_psl < best_psl:
                        best_found = new_code
                        best_psl = new_psl
                else:
                    if acceptance_probability(psl, new_psl, temperature) > random.random():
                        code = new_code
                        stuck_count = 0
                    else:
                        stuck_count+=1
                tabu.append(list_v) # we add this code to the tabu so we do not consider it again
                if len(tabu)==101: # if the tabu is full, we pop off the front of the queue
                    tabu.popleft()
            if stuck_count>upp_bound:
                count+=1
                break
        if stuck_count>upp_bound:
            break
    

        temperature = cooling_schedule(temperature, init_temperature, 0.25, num_temp_gens, j)
    return (best_found, best_psl)

def semi_exponential(temperature, init_temperature, final_temperature, num_gens, current_gen):
    """
        Function to generate the temperature for the next generation of the annealing algorithm. This function
        in particular uses an power model, or semi_exponential, as I called it.

        :param temperature: current temperature
        :type temperature: float
        :param init_temperature: initial temperature
        :type init_temperature: float
        :param final_temperature: final, stopping temperature
        :type final_temperature: float
        :param num_gens: number of generations we will run our annealing algorithm
        :type num_gens: int
        :param current_gen current generation
        :type current_gen: int
        :return: new temperature for the next generation
    """
    beta = 1.3
    return init_temperature/(current_gen**beta)
    
def mult_exponential(temperature, init_temperature, final_temperature, num_gens, current_gen):
    """
        Function to generate the temperature for the next generation of the annealing algorithm. This function
        in particular uses an multiplicative exponential model.

        :param temperature: current temperature
        :type temperature: float
        :param init_temperature: initial temperature
        :type init_temperature: float
        :param final_temperature: final, stopping temperature
        :type final_temperature: float
        :param num_gens: number of generations we will run our annealing algorithm
        :type num_gens: int
        :param current_gen current generation
        :type current_gen: int
        :return: new temperature for the next generation
    """
    beta = 1.3
    return init_temperature/(beta**current_gen)
    
def log_mult(temperature, init_temperature, final_temperature, num_gens, current_gen):
    """
        Function to generate the temperature for the next generation of the annealing algorithm. This function
        in particular uses an multiplicative logarithmic model.

        :param temperature: current temperature
        :type temperature: float
        :param init_temperature: initial temperature
        :type init_temperature: float
        :param final_temperature: final, stopping temperature
        :type final_temperature: float
        :param num_gens: number of generations we will run our annealing algorithm
        :type num_gens: int
        :param current_gen current generation
        :type current_gen: int
        :return: new temperature for the next generation
    """
    beta = 1.3
    return init_temperature/(1+beta*math.log(1+current_gen))

def lin_mult(temperature, init_temperature, final_temperature, num_gens, current_gen):
    """
        Function to generate the temperature for the next generation of the annealing algorithm. This function
        in particular uses an multiplicative linear model.

        :param temperature: current temperature
        :type temperature: float
        :param init_temperature: initial temperature
        :type init_temperature: float
        :param final_temperature: final, stopping temperature
        :type final_temperature: float
        :param num_gens: number of generations we will run our annealing algorithm
        :type num_gens: int
        :param current_gen current generation
        :type current_gen: int
        :return: new temperature for the next generation
    """
    beta = 0.8
    return init_temperature/(1+beta*current_gen)
    
def quad_mult(temperature, init_temperature, final_temperature, num_gens, current_gen):
    """
        Function to generate the temperature for the next generation of the annealing algorithm. This function
        in particular uses an multiplicative quadratic model.

        :param temperature: current temperature
        :type temperature: float
        :param init_temperature: initial temperature
        :type init_temperature: float
        :param final_temperature: final, stopping temperature
        :type final_temperature: float
        :param num_gens: number of generations we will run our annealing algorithm
        :type num_gens: int
        :param current_gen current generation
        :type current_gen: int
        :return: new temperature for the next generation
    """
    beta = 0.8
    return init_temperature/(1+beta*current_gen**2)

def lin_add(temperature, init_temperature, final_temperature, num_gens, current_gen):
    """
    Function to generate the temperature for the next generation of the annealing algorithm. This function
    in particular uses an additive linear model.

    :param temperature: current temperature
    :type temperature: float
    :param init_temperature: initial temperature
    :type init_temperature: float
    :param final_temperature: final, stopping temperature
    :type final_temperature: float
    :param num_gens: number of generations we will run our annealing algorithm
    :type num_gens: int
    :param current_gen current generation
    :type current_gen: int
    :return: new temperature for the next generation
    """
    return final_temperature + (init_temperature - final_temperature) * ((num_gens - current_gen)/num_gens)

def quad_add(temperature, init_temperature, final_temperature, num_gens, current_gen):
    """
        Function to generate the temperature for the next generation of the annealing algorithm. This function
        in particular uses an additive quadratic model.

        :param temperature: current temperature
        :type temperature: float
        :param init_temperature: initial temperature
        :type init_temperature: float
        :param final_temperature: final, stopping temperature
        :type final_temperature: float
        :param num_gens: number of generations we will run our annealing algorithm
        :type num_gens: int
        :param current_gen current generation
        :type current_gen: int
        :return: new temperature for the next generation
    """
    return final_temperature + (init_temperature - final_temperature) * ((num_gens - current_gen)/num_gens)**2

def expon_add(temperature, init_temperature, final_temperature, num_gens, current_gen):
    """
        Function to generate the temperature for the next generation of the annealing algorithm. This function
        in particular uses an additive exponential model.

        :param temperature: current temperature
        :type temperature: float
        :param init_temperature: initial temperature
        :type init_temperature: float
        :param final_temperature: final, stopping temperature
        :type final_temperature: float
        :param num_gens: number of generations we will run our annealing algorithm
        :type num_gens: int
        :param current_gen current generation
        :type current_gen: int
        :return: new temperature for the next generation
        """
    return final_temperature + (init_temperature - final_temperature) * 1/(1+ math.exp(2*math.log(init_temperature - final_temperature)*(current_gen - 0.5*num_gens)/num_gens))

def trigon_add(temperature, init_temperature, final_temperature, num_gens, current_gen):
    """
        Function to generate the temperature for the next generation of the annealing algorithm. This function
        in particular uses an additive trigonometric model.

        :param temperature: current temperature
        :type temperature: float
        :param init_temperature: initial temperature
        :type init_temperature: float
        :param final_temperature: final, stopping temperature
        :type final_temperature: float
        :param num_gens: number of generations we will run our annealing algorithm
        :type num_gens: int
        :param current_gen current generation
        :type current_gen: int
        :return: new temperature for the next generation
    """
    return final_temperature + (init_temperature - final_temperature) * 0.5 * (1 + math.cos(current_gen*math.pi/num_gens))

def quenching(temperature, init_temperature, final_temperature, num_gens, current_gen):
    """
            Function to generate the temperature for the next generation of the annealing algorithm. This function
            in particular uses a quenching model.

            :param temperature: current temperature
            :type temperature: float
            :param init_temperature: initial temperature
            :type init_temperature: float
            :param final_temperature: final, stopping temperature
            :type final_temperature: float
            :param num_gens: number of generations we will run our annealing algorithm
            :type num_gens: int
            :param current_gen current generation
            :type current_gen: int
            :return: new temperature for the next generation
        """
    const = 0.7
    return init_temperature/(math.exp((1 - const) * current_gen))

# potential adaptive simulated annealing temperature step.  Note that I never wrote this, but it would have been inspired
# by the work of Lester Ingber, who has written extensively on the topic.
def adaptive_temp_step():
    return

def climb_ineff(seq, N, phase_unity = 2, num_elem = None, min_fn = second_max):
    """
        Function to find the optimal neighbor in the one-neighborhood of a polyphase code with values drawn from
        the roots of unity of order phase_unity.

        :param seq: the code we are considering
        :type seq: array
        :param N: the doppler width
        :type N: int
        :param phase_unity: the order of the roots of unity
        :type phase_unity: int
        :param num_elem: number of elements that min_fn is being evaluated on
        :type num_elem: int
        :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
        :type min_fn: function
        :return: an updated sequence, the optimal neighbor of seq
    """
    val = fast_autocorrelations(seq, N, num_elem, min_fn) # initial peak sidelobe value
    current_seq = np.copy(seq)
    min_val = val
    for i in range(len(seq)): # loop over all possible coordinates to change
        for j in range(1, phase_unity): # loop over all possible phase additions
            seq_new = np.copy(seq)
            phase_add = random.randint(1, phase_unity - 1)
            seq_new[i] = seq_new[i]*cmath.exp(phase_add * 2 * cmath.pi * imag / phase_unity)
            new_val = fast_autocorrelations(seq_new, N, num_elem, min_fn)
            if new_val < min_val:
                min_val = new_val
                current_seq = seq_new
        
    return current_seq


def gen_neighbor(code, phase_unity, movement_length = 1):
    """
    Function to generate a neighbor of a given code by changing movement_length of the coordinates.

    :param code: the code we are considering
    :type code: array
    :param phase_unity: the roots of unity we are considering
    :type phase_unity: int
    :param movement_length: the number of coordinates to change
    :type movement_length: int
    :return: an updated sequence with movement_length coordinates changed to different roots of unity
    """
    seq_2 = np.copy(code) # copy of the sequence in question
    mut_points = random.sample(range(0, len(seq_2)), movement_length) # coordinates to mutate
    for i in mut_points:
        phase = random.randint(0, phase_unity-1) # the new phase of the coordinate
        new_value = cmath.exp(phase * 2 * cmath.pi * imag/phase_unity) # the corresponding value
        if new_value == seq_2[i]: # if we picked the same value as before, add a random phase to the chosen value that is nonzero
            phase_add = random.randint(1, phase_unity-1)
            new_value = cmath.exp((phase + phase_add) * 2 * cmath.pi * imag/phase_unity)
        seq_2[i] = new_value
    return seq_2
    
def acceptance_probability(old_psl, new_psl, T):
    """
    Function to generate an acceptance probability of a new code, given its min_fn value, the old minimal value, and the temperature.
    This function is only called when new_psl ≥ old_psl; otherwise, the new code is always accepted.

    :param old_psl: old minimal min_fn value
    :type old_psl: float
    :param new_psl: min_fn value for the code currently under consideration
    :type new_psl: float
    :param T: temperature at the current step in te annealing algorithm
    :type T: float
    :return: a probability of the new code being accepted
    """

    alpha = 1.5
    exp_constant = abs(new_psl) -abs(old_psl)
    exp_constant = (exp_constant/T)**alpha
    return math.exp(-exp_constant)


local_search_options = [climb_ineff] 
