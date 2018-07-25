# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:51:51 2017

@author: gandhik
"""

import time
import cmath
import math
import random
imag = cmath.sqrt(-1)
from conv_utils import con_psl, fast_autocorrelations, second_max, gen_population, gen_rand_seq
from anneal_full import acceptance_probability, semi_exponential

# great deluge algo, modeled off of the pseudocode http://www.cellsoft.de/study/barker.pdf
#cmath.exp(num*imag)

def gen_rand_seq_poly(length):
    """
    Function to generate a random sequence of phases for our polyphase sequence.

    :param length: length of the sequence
    :type length: int
    :return: a sequence of phases with a given length

    See also: gen_rand_seq_phases, gen_rand_seq_phases_with_norm
    """
    lst = []
    for i in range(length):
        num = random.uniform(0, 2*math.pi)
        lst.append(num)
    
    return lst
    
def gen_rand_seq_phases(length):
    """
    Function to generate a random sequence of phases to add/subtract from our polyphase sequences.
    Only difference from gen_rand_seq_poly: elements range from 0 to π, rather than 0 to 2π.

    :param length: length of the sequence
    :type length: int
    :return: a sequence of phases with a given length

    See also: gen_rand_seq_poly, gen_rand_seq_phases_with_norm
    """
    lst = []
    for i in range(length):
        num = random.uniform(0, math.pi)
        lst.append(num)
    
    return lst

def gen_rand_seq_phases_with_norm(length, desired_norm):
    """
    Function to generate a random sequence of phases of a given norm.

    :param length: length of the sequence in question
    :type length: int
    :param desired_norm: norm that we want for the sequence
    :type desired_norm: float
    :return: a phase sequence of a given norm

    See also: gen_rand_seq_phases
    """
    lst = gen_rand_seq_phases(length)

    current_norm = norm(lst)
    scale = desired_norm/current_norm

    for i in range(len(lst)): # multiply each element of the lst by the scale to get it to the desired norm.
        lst[i]*=scale

    return lst
    
def norm(seq):
    """
    Function that returns the norm of a sequence.

    :param seq: sequence in question
    :type seq: array
    :return: norm of seq
    """
    sum_seq = 0
    for elem in seq:
        sum_seq = sum_seq + abs(elem)**2
    return sum_seq**(0.5)

def fast_autocorrelations_phases(code, N=1, num_elem = None, min_fn = second_max):
    """
    Calculate the min_fn value of the corresponding polyphase code to a given phase sequence.
    :param code: The phase sequence in question
    :type code: array
    :param N: doppler width of our ambiguity function
    :type N: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: the min_fn value of the polyphase code corresponding to the phase sequence.
    """
    act_code = []
    for i in range(len(code)): # construct the actual polyphase code corresponding to a sequence of phases
        act_code.append(cmath.exp(code[i]*imag))
    return fast_autocorrelations(act_code, N, num_elem, min_fn)



def great_deluge(length, seq, num = None, N=1, num_elem = None, min_fn = second_max):
    """
    Given a starting polyphase code, alter it to find a nearby min_fn minima.

    :param length: length of the code in question
    :type length: int
    :param seq: the starting sequence
    :type seq: array
    :param N: doppler width of our ambiguity function
    :type N: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: the minimal min_fn value found by the algorithm
    """
    phases = gen_rand_seq_phases(length)
    min_phase_stepsize = 0.00001 # the minimal norm of the phase change past which the algorithm ends
    phase_divisor = 1.3 # how fast the change in phase values moves down in morm
    rain_speed = 0.03 # how fast the water level moves down
    water_level = fast_autocorrelations_phases(seq, N, num_elem, min_fn)#int(length * math.log(length))
    best_code = seq
    best_val = water_level
    unsucc_alter = 0
    while norm(phases) > min_phase_stepsize:
        print(water_level)
        dry_steps = 0
        for i in range(len(seq)):
            # move in each direction of phases, check if it improves the min_fn value.  If so, update the sequence, and best_val.
            seq[i] += phases[i]
            val = fast_autocorrelations_phases(seq, N, num_elem, min_fn)
            if val < water_level:
                if val < best_val:
                    best_val = val
                    best_code = seq
                dry_steps +=1
                water_level = water_level - rain_speed
            else:
                phases[i] = -phases[i]
                seq[i] += phases[i]
        # if we didn't move anywhere in the last two iterations, make the phase change smaller, and try again.
        if dry_steps == 0:
            unsucc_alter +=1
            if unsucc_alter == 2:
                for i in range(len(phases)):
                    phases[i] = phases[i]/phase_divisor
                unsucc_alter = 0
        else:
            unsucc_alter = 0
    
    return (best_code, best_val)

def random_arbitrary_polyphase(length):
    """
    Randomized algorithm to find a minimal min_fn value for polyphase codes of a given length.

    :param length: length of the codes in question
    :type length: int
    :return: the best code and min_fn value found by our algorithm
    """
    best_code = None
    best_val = float('inf')
    time_limit = 0
    if length <= 30:
        time_limit = 300
    else:
        time_limit = (300 + 60 * (length - 30))
    start_time = time.time()
    while time.time() - start_time < time_limit and best_val > 1:
        seq = gen_rand_seq_poly(length)
        val = fast_autocorrelations_phases(seq)
        if val < best_val:
            best_code = seq
            best_val = val

    return (best_code, best_val)

def great_deluge_handler(length, great_deluge_subroutine = great_deluge, num = 1, N=1, num_elem = None, min_fn = second_max):
    """
    Handler for the great deluge algorithm, which aims to minimize the min_fn value of polyphase codes of a given length.

    :param length: length of the codes for which we want to minimize min_fn
    :type length: int
    :param N: doppler width of our ambiguity function
    :type N: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: the best code and min_fn value found by our algorithm
    """
    best_code = None
    best_val = float('inf')
    time_limit = 0
    if length <= 30:
        time_limit = 2.25 * 300
    else:
        time_limit = 2.25 * (300 + 60 * (length - 30))
    num_phases = int(length**0.5 * 10)
    start_time = time.time()
    while time.time() - start_time < time_limit: #and ((best_val > 1 and num == 1) or (best_val != 0 and num>1)):
        seq = gen_rand_seq_poly(num*length) # generate a sequence of phases of a given length (longer if num > 1, in which case we are looking for alternating codes)
        for j in range(num_phases): # loop over a number of possible phase additions.
            x = great_deluge_subroutine(length, seq, num, N, num_elem, min_fn) # run the great_deluge subroutine, and update the best_val if it is improved.
            if x[1] < best_val:
                best_code = x[0]
                best_val = x[1]

    return (best_code, best_val)

def anneal_phases_handler(length, cooling_schedule, N=1, num_elem = None, min_fn = second_max):
    """
    The handler for our annealing algorithm.  This function runs our annealing function a number of times, proportional to the
    length of the code, and returns the best code found over all such iterations.

    :param length: length of the codes under consideration
    :type length: int
    :param cooling_schedule: function that describes how the temperature decreases from generation to generation
    :type cooling_schedule: function
    :param N: doppler width of our ambiguity function
    :type N: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: a tuple of the best polyphase code and its corresponding min_fn value that we find
    """
    time_limit = 0
    if length <= 30:
        time_limit = 300
    else:
        time_limit = 1 * (300 + 60 * (length - 30))
    start_time = time.time()
    best_psl_total = float('inf')
    best_code = None
    while time.time() - start_time < time_limit and best_psl_total > 1:
        x = gen_rand_seq_poly(length) # generate a sequence of phases of a given length
        best = anneal_phases(x, cooling_schedule, great_deluge_helper, N, num_elem, min_fn)  #run the anneal subroutine, and update the best_val if it is improved.
        if best[1] < best_psl_total:
            best_psl_total = best[1]
            best_code = best[0]
        if best_psl_total <=1.0:
            break
    return (best_code, best_psl_total)

def gen_neighbor_with_phase(code, norm):
    """
    Generate a neighbor of a given phase sequence at a distance _norm_ away, using gen_rand_seq_phases_with_norm.

    :param code: original sequence
    :type code: array
    :param norm: norm of phase sequence to add
    :type norm: float
    :return: a new phase sequence, norm away from code.
    """
    new_code = code[:]
    phase_add = gen_rand_seq_phases_with_norm(len(code), norm) # add a specific phase of a given norm to the code
    for i in range(len(new_code)):
        new_code[i] = new_code[i] + phase_add[i]
    return new_code



def great_deluge_helper(length, seq, N=1, num_elem = None, min_fn = second_max):
    """
    Given a starting polyphase code, alter it to find a nearby min_fn minima.  Used as a local search algorithm
    in anneal_phases here, with a small min_phase_stepsize and larger rain_speed so the algorithm goes faster.

    :param length: length of the code in question
    :type length: int
    :param seq: the starting sequence
    :type seq: array
    :param N: doppler width of our ambiguity function
    :type N: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: the minimal min_fn value for a polyphase code found by the algorithm

    Note: no comments are provided here (see the great_deluge function for the necessary comments).
    """
    phases = gen_rand_seq_phases(length)
    min_phase_stepsize = 0.01
    phase_divisor = 1.6
    rain_speed = 0.4
    water_level = fast_autocorrelations_phases(seq, N, num_elem, min_fn)  # int(length * math.log(length))
    best_code = seq
    best_val = water_level
    unsucc_alter = 0
    while norm(phases) > min_phase_stepsize:
        dry_steps = 0
        for i in range(len(seq)):
            seq[i] += phases[i]
            val = fast_autocorrelations_phases(seq, N, num_elem, min_fn)
            if val < water_level:
                if val < best_val:
                    best_val = val
                    best_code = seq
                dry_steps += 1
                water_level = water_level - rain_speed
            else:
                phases[i] = -phases[i]
                seq[i] += phases[i]
        if dry_steps == 0:
            unsucc_alter += 1
            if unsucc_alter == 2:
                for i in range(len(phases)):
                    phases[i] = phases[i] / phase_divisor
                unsucc_alter = 0
        else:
            unsucc_alter = 0

    return best_code

def anneal_phases(code, cooling_schedule, local_search = great_deluge_helper, N=1, num_elem = None, min_fn = second_max):
    """
    Simulated annealing algorithm for a phase sequence.  Works similarly to a simulated annealing algorithm, uses
    a great deluge local search algorithm by default.

    :param code: starting code in question
    :type code: array
    :param cooling_schedule: function that describes how the temperature decreases from generation to generation
    :type cooling_schedule: function
    :param local_search: local search algorithm used to locally minimize after updating our current sequennce
    :type local_search: function
    :param N: doppler width of our ambiguity function
    :type N: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: the best polyphase code and corresponding min_fan value near a given start sequence.
    """
    init_temperature = len(code)
    temperature = init_temperature
    psl = fast_autocorrelations_phases(code, N, num_elem, min_fn)  # initial peak sidelobe value
    count = 0
    kappa = 0.2
    num_gen = int(1.5 * len(code))  # number of generations at a fixed temperature
    best_found = code
    best_psl = psl
    stuck_count = 0  # number of iterations we have been stuck at a given code
    num_temp_gens = int(2 * init_temperature)
    upp_bound = 2 * len(code) ** (kappa)
    for j in range(1, num_temp_gens):  # loop from high temperature to low temperature
        for i in range(num_gen):  # loop at a given temperature of movements from neighbor to neighbor
            new_code = gen_neighbor_with_phase(code, 2)  # new_code found by gen_neighbor
            new_code = local_search(len(code), new_code, N, num_elem, min_fn) # locally search for the best phase sequence nearby
            new_psl = fast_autocorrelations_phases(new_code, N, num_elem, min_fn)
            if new_psl < psl:  # move to the new code if the peak sidelobe value improves
                code = new_code
                stuck_count = 0
                if new_psl < best_psl:  # update best_psl as well if it is improved on
                    best_found = new_code
                    best_psl = new_psl
            else:
                if acceptance_probability(psl, new_psl,
                                          temperature) > random.random():  # otherwise, only move to the code if a randomly generated number is greater than acceptance_probability
                    code = new_code
                    stuck_count = 0
                else:
                    stuck_count += 1
            if best_found <=1.0:
                break
            if stuck_count > upp_bound:  # if we have been stuck for a while, break out of the loop
                count += 1
                break
        if best_found<=1.0:
            break
        if stuck_count > upp_bound:
            break

        temperature = cooling_schedule(temperature, init_temperature, 0.25, num_temp_gens, j)

    return (best_found, best_psl)

            


        
        
    
    
    
    
    
    