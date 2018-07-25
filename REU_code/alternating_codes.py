
from conv_utils import con_psl_matrix, fast_autocorrelations, gen_population, second_max, fast_autocorrelations_matrix, gen_rand_seq
from exhaustive import gen_binary_codes
from memetic_tabu_search import evolve_mem, crossover_random_single
import time
import numpy as np
import random
from polyphase import gen_rand_seq_phases_with_norm, gen_rand_seq_phases, gen_rand_seq_poly, norm, great_deluge_handler
import cmath
imag = cmath.sqrt(-1)
import pandas as pd


def autocorrelation_addition(code, num):
    """
    Given a code composed of num subcodes (of the same length), computes the autocorrelation of each and adds them, outputting
    the psl of that addition.
    :param code: initial sequence
    :type code: array
    :param num: number of sub_codes
    :type num: int
    :return: psl of the sum of the codes (how near-alternating the codes are).
    """
    if len(code) % num !=0: # throw an error if we cannot properly divide the code
        raise ValueError
    else:
        length = len(code)/num # length of each code
        current_count = 1
        subcodes = []
        for i in range(num): # compute the subcodes
            new_code = code[length*i:length*(i+1)]
            subcodes.append(new_code)

        autocorrelation_lsts  = []
        for subcode in subcodes: # compute their autocorrelations
            autocorrelation_lsts.append(con_psl_matrix(subcode))
        final_autocorrelation = []
        for i in range(len(autocorrelation_lsts[0])): # add entry by entry
            sum = 0
            for j in range(len(autocorrelation_lsts)):
                sum+=autocorrelation_lsts[j][i]
            final_autocorrelation.append(sum)
        return second_max(final_autocorrelation)

def autocorrelations_phases(code, num):
    """
    Given a code composed of the phase representation of num subcodes, computes the autocorrelation_addition of them.
    :param code: initial phase sequence
    :type code: array
    :param num: number of subcodes
    :type num: int
    :return: psl of the sum of the codes (how near-alternating they are).
    """
    act_code = []
    for i in range(len(code)):
        act_code.append(cmath.exp(code[i]*imag))
    return autocorrelation_addition(act_code, num)


def gen_rand_binary_seq(length):
    """
    Function to generate a random binary sequence, not with complex numbers.
    :param length: length of sequence
    :type length: int
    :return: a binary sequence of length _length_
    """
    poss = [-1, 1]
    probs = [0.5, 0.5]
    return np.random.choice(poss, size=(length,), p=probs)

def random_alternating_binary(length, num):
    """
    Random algorithm for generating alternating binary sequences.  Generates codes of length num*length continuously, outputting
    the one with the smallest autocorrelation_addition.

    :param length: length of codes in question
    :type length: int
    :param num: number of codes in our alternating code
    :type num_length
    :return: sequence with the minimal autocorrelation_addition
    """
    best_code = None
    best_val = float('inf')
    time_limit = 0
    if length <= 30:
        time_limit = 300
    else:
        time_limit = 1 * (300 + 60 * (length - 30))
    start_time = time.time()
    while time.time() - start_time < time_limit and best_val != 0:
        x = gen_rand_binary_seq(length * num) # generate a random binary sequence of the desired length
        val = autocorrelation_addition(x, num) # calculate the autocorrelation_addition and change best_val if it improves
        if val < best_val:
            best_code = x
            best_val = val

    return (best_code, best_val)

def random_alternating_polyphase(length, num):
    """
        Random algorithm for generating alternating arbitrary polyphase sequences.  Generates codes of length num*length continuously, outputting
        the one with the smallest autocorrelation_addition.

        :param length: length of codes in question
        :type length: int
        :param num: number of codes in our alternating code
        :type num_length
        :return: sequence with the minimal autocorrelation_addition
        """
    best_code = None
    best_val = float('inf')
    time_limit = 0
    if length <= 30:
        time_limit = 300
    else:
        time_limit = 1 * (300 + 60 * (length - 30))
    start_time = time.time()
    while time.time() - start_time < time_limit and best_val != 0:
        x = gen_rand_seq_poly(length * num) # generate a random phase sequence of the desired length
        val = autocorrelations_phases(x, num)
        if val < best_val: # calculate the autocorrelation_addition and change best_val if it improves
            best_code = x
            best_val = val

    return (best_code, best_val)

def hill_climbing_iterative_alternating(length, num, climb_type):
    """
    Hill climbing handler for generating alternating binary sequences.  Works similarly to hill_climbing,
    except the function that we are minimizing is autocorrelation_addition instead of fast_autocorrelations.

    :param length: length of the codes in question
    :type length: int
    :param num: number of codes in our alternating code
    :type num: int
    :param climb_type: climb subroutine we are using
    :type climb_type: fn
    :return: binary sequence with the minimal autocorrelation_addition, as well as that value
    """
    best_code = None
    best_val = float('inf')
    time_limit = 0
    if length <= 30:
        time_limit = 900
    else:
        time_limit = 1 * (300 + 60 * (length - 30))
    start_time = time.time()
    while time.time() - start_time < time_limit and best_val!=0:
        x = gen_rand_binary_seq(length*num) # generate a random binary sequence of the desired length
        x = climb_type(x, num) # climb to the best neighbor
        val = autocorrelation_addition(x, num) # calculate the autocorrelation_addition and change best_val if it improves
        if val < best_val:
            best_code = x
            best_val = val

    return (best_code, best_val)

def climb_ineff_check_change(seq, num):
    """
    Function to execute, inefficiently, a climb to the best neighbor for an alternating code sequence.
    :param seq: initial sequence
    :type seq: array
    :param num: number of codes in our alternating code
    :type num: int
    :return: the best neighbor of seq, as well as whether it changed.
    """
    val = autocorrelation_addition(seq, num)  # initial autocorrelation_addition psl value
    current_seq = np.copy(seq)
    min_val = val
    changed = False
    for i in range(len(seq)):  # loop over all possible coordinates to change
        seq_new = np.copy(seq)
        seq_new[i] = -seq_new[i]
        new_val = autocorrelation_addition(seq_new, num)
        if new_val < min_val:
            min_val = new_val
            current_seq = seq_new
            changed = True
    return current_seq, changed

def sdls_local_search(child, num):
    """
    Function to execute a steepest descent local search on binary sequences with autocorrelation_addition as our minimization
    function.

    :param child: initial sequence
    :type child: array
    :param num: number of codes in our alternating code
    :type num: int
    :return: A local minima near seq
    """
    not_done = True
    to_return = child
    while not_done:
        to_return, not_done = climb_ineff_check_change(to_return, num)
    return to_return


def great_deluge_alternating(length, seq, num, N=1, num_elem = None, min_fn = second_max):
    """
    Great deluge subroutine for alternating polyphase sequences.  Notice that we can use the same handler,
    given the num updates that we gave to it in polyphase.py.  There are no major changes to the main function,
    so no comments, are added, except for the fact that autocorrelations_phases instead of fast_autocorrelations_phases is used.

    :param length: length of the codes in question
    :type length: int
    :param seq: initial sequence of phases
    :type seq: array
    :param num: number of codes in our alternating code
    :type num: int
    :param N: doppler width of our ambiguity function
    :type N: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: arbitrary polyphase sequence with the minimal autocorrelation_addition found by our algorithm, as well as that value
    """
    phases = gen_rand_seq_phases(num*length)
    min_phase_stepsize = 0.00001
    phase_divisor = 1.3
    rain_speed = 0.03
    water_level = autocorrelations_phases(seq, num)  # int(length * math.log(length))
    best_code = seq
    best_val = water_level
    unsucc_alter = 0
    while norm(phases) > min_phase_stepsize:
        dry_steps = 0
        for i in range(len(seq)):
            seq[i] += phases[i]
            val = autocorrelations_phases(seq, num)
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

    return (best_code, best_val)


# An unsuccessful naive attempt at generating alternating codes

def gen_alternating_codes(length):
    """
    Function to generate pairs of alternating codes; uses the naive approach of looping over all codes.

    :param length: length of the codes in question
    :type length: int
    :return: none, just prints the codes.
    """
    all_codes = gen_binary_codes(length)  # generate all binary codes
    for code_1 in all_codes:  # double for loop over all codes, sums their con_psl_matrixes and checks if they are alternating
        for code_2 in all_codes:
            x = sum_codes(con_psl_matrix(code_1), con_psl_matrix(code_2))
            if is_alternating(x):
                print(x)
                print(code_1)
                print(code_2)

def sum_codes(code_1, code_2):
    """
    Sums two codes entry by entry.  Outdated, could just use np.add.
    :param code_1: first code
    :type code_1: array
    :param code_2: second code
    :type code_2: array
    :return: array with entry i equal to code_1[i] + code_2[i]
    """
    new_code = []
    for i in range(len(code_1)):
        new_code.append(code_1[i] + code_2[i])

    return new_code


def is_alternating(sum_of_codes):
    """
    Checks if two codes are alternating by checking if their sum has only one nonzero entry.
    :param sum_of_codes: the sum of the two codes that we are checking
    :type sum_of_codes: array
    :return: True if sum_codes has one nonzero entry, False otherwise.
    """
    nonzero = 0
    for i in range(len(sum_of_codes)):
        if sum_of_codes[i] != 0: # if nonzero, increment our nonzero count
            nonzero += 1

    if nonzero == 1:
        return True
    else:
        return False

def hill_climbing_alternating_test():
    print('hill_climbing_alternating')
    length = []
    number_of_codes = []
    hill_climbing_codes = []
    hill_climbing_values = []
    for i in range(21, 35):
        print(i)
        for j in range(4, 10, 2):
            print(j)
            length.append(i)
            number_of_codes.append(j)
            x = hill_climbing_iterative_alternating(i, j, sdls_local_search)
            hill_climbing_codes.append(x[0])
            hill_climbing_values.append(x[1])
            print(x[1])

    df = pd.DataFrame({'Length': length, 'Number of codes': number_of_codes,
                       'Code': hill_climbing_codes, 'PSL Value': hill_climbing_values})
    writer = pd.ExcelWriter('hill_climb_alternating_second.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()


def hill_climbing_alternating_test():
    print('hill_climbing_alternating')
    length = []
    number_of_codes = []
    hill_climbing_codes = []
    hill_climbing_values = []
    for i in range(14, 25):
        print(i)
        for j in range(4, 10, 2):
            print(j)
            length.append(i)
            number_of_codes.append(j)
            x = hill_climbing_iterative_alternating(i, j, sdls_local_search)
            hill_climbing_codes.append(x[0])
            hill_climbing_values.append(x[1])
            print(x[1])

    df = pd.DataFrame({'Length': length, 'Number of codes': number_of_codes,
                       'Code': hill_climbing_codes, 'PSL Value': hill_climbing_values})
    writer = pd.ExcelWriter('hill_climb_alternating_second.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
