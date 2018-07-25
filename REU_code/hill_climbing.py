

from conv_utils import gen_rand_seq, fast_autocorrelations, sum_square_values, second_max
import time
import numpy as np
import math
import random
import cmath
imag = cmath.sqrt(-1)

def hill_climbing_iterative(length, climb_type, N=1, phase_unity=2, min_fn = second_max, num_elem = None):
    """
    An iterative, partial-restart based hill climbing algorithm with flexibility for doppler width, polyphase codes,
    and different types of local search.  This algorithm is used to find codes with the minimal min_fn value.

    :param length: length of the codes in question
    :type length: int
    :param climb_type: local search function
    :type climb_type: function
    :param N: doppler width of the ambiguity function
    :type N: int
    :param phase_unity: order of the roots of unity in our codes
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: the minimal min_fn value found by our algorithm, as well as the corresponding polyphase code
    """
    f = open('best_known_sidelobes.txt') # best known sidelobes from the literature
    lines = f.readlines()
    best_code = None
    best_val = float('inf')
    cutoff_val = int(lines[length - 1]) # lowest possible value we can find
    time_limit = 0
    if length <= 30:
        time_limit = 300
    else:
        time_limit = (300 + 60 * (length - 30))
    start_time = time.time()
    while time.time() - start_time < time_limit and (best_val != cutoff_val or phase_unity!=2):
        x = gen_rand_seq(length, phase_unity) # generate a random sequence of the given specifications
        x = climb_type(x, N, phase_unity, num_elem, min_fn) # climb using the given climb_type to a nearby minima
        val = fast_autocorrelations(x, N, num_elem, min_fn) # compute the psl/min_fn value, update if it improves on the previous best
        if val < best_val:
            best_code = x
            best_val = val

    return (best_code, best_val)

def simple_climb(seq, N, phase_unity, num_elem = None, min_fn = second_max):
    """
    An algorithm to climb to a local minima from a given sequence, where each step taken is the first that improves
    the current sequence.

    :param seq: initial sequence
    :type seq: array
    :param N: doppler width of the ambiguity function over which we are minimizing
    :type N: int
    :param phase_unity: order of the roots of unity in our codes
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: a code neighboring seq with an improved min_fn value, as well as whether climbing there required at least one change to seq.
    """
    val = fast_autocorrelations(seq, N, num_elem, min_fn)
    current_seq = np.copy(seq)
    min_val = val
    changed = False
    for i in range(len(seq)):  # loop over all possible coordinates to change
        for j in range(1, phase_unity):  # loop over all possible phase additions
            seq_new = np.copy(seq)
            seq_new[i] = seq_new[i] * cmath.exp(j* 2 * cmath.pi * imag / phase_unity) # multiply by the given root of unity
            new_val = fast_autocorrelations(seq_new, N, num_elem, min_fn)
            if new_val < min_val:
                min_val = new_val
                current_seq = seq_new
                changed = True
                return current_seq, changed

    return current_seq, changed

def simple_local_search(seq, N, phase_unity, num_elem = None, min_fn = second_max):
    """
    Handler for our simple climb algorithm, where we keep climbing until we reach a local minima.

    :param seq: initial sequence
    :type seq: array
    :param N: doppler width of the ambiguity function over which we are minimizing
    :type N: int
    :param phase_unity: order of the roots of unity in our codes
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: a locally minimal code near seq
    """
    not_done = True
    to_return = seq
    while not_done:
        to_return, not_done = simple_climb(to_return, N, phase_unity, num_elem, min_fn)
    return to_return

def stochastic_climb(seq, N, phase_unity, num_elem = None, min_fn = second_max):
    """
    An algorithm to climb to a neighbor of a given sequence, where each step taken is always taken if it improves the
    min_fn value, and sometimes taken if it does not, probabilistically according to the difference between the new and old min_fn values.

    :param seq: initial sequence
    :type seq: array
    :param N: doppler width of the ambiguity function over which we are minimizing
    :type N: int
    :param phase_unity: order of the roots of unity in our codes
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: a neighbor of seq
    """
    val = fast_autocorrelations(seq, N, num_elem, min_fn)
    current_seq = np.copy(seq)
    min_val = val
    for i in range(len(seq)):  # loop over all possible coordinates to change
        for j in range(1, phase_unity):  # loop over all possible phase additions
            seq_new = np.copy(seq)
            phase_add = random.randint(1, phase_unity - 1)
            seq_new[i] = seq_new[i] * cmath.exp(phase_add * 2 * cmath.pi * imag / phase_unity) # multiply by the given root of unity
            new_val = fast_autocorrelations(seq_new, N, num_elem, min_fn)
            if new_val < min_val:
                min_val = new_val
                current_seq = seq_new
                return current_seq
            else:
                if prob_of_acceptance(min_val, new_val) > random.random(): # if the min_fn value is worse, move with some probability.
                    current_seq = seq_new
                    return current_seq
    return current_seq

def prob_of_acceptance(old_val, new_val):
    """
    Helper function to get the probability we should move to a new code, given the original min_fn value and the new value.
    Assumes that old_val < new_val.

    :param old_val: min_fn evaluated on the ambiguity function of the code we are currently at
    :type old_val: float
    :param new_val: min_fn evaluated on the ambiguity function of the code we are considering
    :type new_val: float
    :return: a probability of moving to the new code
    """
    return math.exp((abs(old_val) - abs(new_val))/2)


def stochastic_local_search(seq, N, phase_unity, num_elem = None, min_fn = second_max): # note: for future, would be good to collapse this with sdls, simple into one algor
    """
    Handler for our simple climb algorithm, where we keep climbing for a fixed number of steps using stochastic_climb.

    :param seq: initial sequence
    :type seq: array
    :param N: doppler width over which we are minimizing
    :type N: int
    :param phase_unity: order of the roots of unity in our codes
    :type phase_unity: int
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: probabilistically, a locally minimal code near seq
    """
    max_iters = 40 # perhaps fewer iterations, if want to optimize
    to_return = seq
    best_found = seq
    best_val = fast_autocorrelations(seq, N, num_elem, min_fn)
    for i in range(max_iters):
        to_return = stochastic_climb(to_return, N, phase_unity, num_elem, min_fn) # climb stochastically to a code
        new_val = fast_autocorrelations(to_return, N, num_elem, min_fn) # compute min_fn value of the ambiguity function, update best_val if it improves.
        if new_val < best_val:
            best_val = new_val
            best_found = np.copy(to_return)
    return best_found

# not yet completed function to perturb minima in some of the climb/local search algorithms, if we were to use them as subroutines
# for other optimization algorithms.
def perturb_minima(seq):
    return


def climb_ineff_check_change(seq, N, phase_unity=2, num_elem=None, min_fn=second_max):
    """
        Function to find the optimal neighbor in the one-neighborhood of a polyphase code with values drawn from
        the roots of unity of order phase_unity.

        :param seq: the code we are considering
        :type seq: array
        :param N: doppler width of our ambiguity function
        :type N: int
        :param phase_unity: order of the roots of unity filling our codes
        :type phase_unity: int
        :param num_elem: number of elements that min_fn is being evaluated on
        :type num_elem: int
        :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
        :type min_fn: function
        :return: an updated sequence, the optimal neighbor of seq
    """
    val = fast_autocorrelations(seq, N, num_elem, min_fn)  # initial peak sidelobe value
    current_seq = np.copy(seq)
    min_val = val
    changed = False
    for i in range(len(seq)):  # loop over all possible coordinates to change
        for j in range(1, phase_unity):  # loop over all possible phase additions
            seq_new = np.copy(seq)
            if phase_unity == 2:
                seq_new[i] = -seq_new[i]
            else:
                seq_new[i] = seq_new[i] * cmath.exp(j * 2 * cmath.pi * imag / phase_unity)
            new_val = fast_autocorrelations(seq_new, N, num_elem, min_fn)
            if new_val < min_val:
                min_val = new_val
                current_seq = seq_new
                changed = True
    return current_seq, changed


def sdls_local_search(child, N=1, phase_unity=2, num_elem=None, min_fn=second_max):
    """
    Steepest descent local search algorithm; works by continually climbing to the best neighbor until reaching a local minima.

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
    :return: local minima near child, found by steepest descent local search.
    """
    not_done = True
    to_return = child
    while not_done:
        to_return, not_done = climb_ineff_check_change(to_return, N, phase_unity, num_elem, min_fn)
    return to_return

