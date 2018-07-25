

import time
import cmath
from anneal_full import gen_neighbor
imag = cmath.sqrt(-1)
from conv_utils import gen_rand_seq, fast_autocorrelations, second_max




def threshold(code, N, threshold_schedule, phase_unity, num_elem = None, min_fn = second_max):
    """
    The core threshold accepting algorithm, which starts with a random polyphase code (with generating phase described
    by phase_unity) and moves around with gen_neighbor, excluding movements for which the new code has a min_fn value
    that exceeds our threshold.

    :param code: the code in question
    :type code: array
    :param N: doppler width of our ambiguity function
    :type N: int
    :param threshold_schedule: function that describes how the threshold decreases from generation to generation
    :type threshold_schedule: function
    :param phase_unity: order of the roots of unity that fill the codes we will be considering
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: a tuple of the best code and its corresponding min_fn value found by the algorithm, given the particular starting point
    """
    psl = fast_autocorrelations(code, N, num_elem, min_fn)  # initial peak sidelobe value
    curr_threshold = int(psl * 2)
    count = 0
    kappa = 0.2
    num_gen = 30 * len(code)  # number of generations at a fixed threshold
    best_found = code
    best_psl = psl
    stuck_count = 0  # number of iterations we have been stuck at a given code
    upp_bound = 2 * len(code) ** (kappa)
    num_threshold_gens = int(len(code)/2)
    for j in range(1, num_threshold_gens):
        for i in range(num_gen):  # loop at a given threshold of movements from neighbor to neighbor
            new_code = gen_neighbor(code, phase_unity, 3)  # new_code found by gen_neighbor

            new_psl = fast_autocorrelations(new_code, N, num_elem, min_fn)
            if new_psl < curr_threshold:  # move to the new code if it is better than the threshold
                code = new_code
                stuck_count = 0
                if new_psl < best_psl:  # update best_psl as well if it is improved on
                    best_found = new_code
                    best_psl = new_psl
            else:
                stuck_count+=1

            if stuck_count > upp_bound:  # if we have been stuck for a while, break out of the loop
                count += 1
                break
        if stuck_count > upp_bound:
            break


        curr_threshold = semiexponential(curr_threshold, j) # update the threshold

    return (best_found, best_psl)

def semiexponential(curr_thres, current_gen):
    """
    Function to generate the threshold for the next generation of the threshold acceptance algorithm. This function
        in particular uses an power model, or semi_exponential, as I called it.

    :param curr_thres: current threshold
    :type curr_thres: float
    :param current_gen: current generation
    :type current_gen: int
    :return: new threshold for the next generation
    """
    beta = 1.3
    return curr_thres/(current_gen ** beta)


def threshold_handler(length, N = 1, threshold_schedule = semiexponential, threshold_fn = threshold, phase_unity=2, num_elem = None, min_fn = second_max):
    """
    Handler for my threshold acceptance algorithm.  Works similarly to a annealing algorithm, except instead of
    a probabilistic move, we have a threshold that decreases over generations that governs whether
    we can move to a newly found code.

    :param length: length of the codes under consideration
    :type length: int
    :param N: doppler width of our ambiguity function
    :type N: int
    :param threshold_schedule: function that describes how the threshold decreases from generation to generation
    :type threshold_schedule: function
    :param threshold_fn_fn: specific threshold accepting function that we will use
    :type threshold_fn: function
    :param phase_unity: order of the roots of unity that fill the codes we will be considering
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: a tuple of the best code and its corresponding min_fn value that we find
    """
    kappa = 0.5
    num = int(5 * length ** (kappa))
    best_psl_total = float('inf')
    best_code = None
    f = open('best_known_sidelobes.txt') # already discovered best sidelobes, used to speed up the process for binary
    lines = f.readlines()
    cutoff_val = int(lines[length - 1]) # lowest possible value we can find
    time_limit = 0
    if length <= 30:
        time_limit = 300
    else:
        time_limit = (300 + 60 * (length - 30))
    start_time = time.time()
    while (time.time() - start_time) < time_limit and (best_psl_total != cutoff_val or N!=1 or phase_unity!=2):
        for i in range(num):
            x = gen_rand_seq(length, phase_unity)
            best = threshold_fn(x, N, threshold_schedule, phase_unity, num_elem, min_fn)
            if best[1] < best_psl_total:
                best_psl_total = best[1]
                best_code = best[0]
            if best_psl_total == cutoff_val:
                break
        if best_psl_total == cutoff_val:
            break

    return best_psl_total