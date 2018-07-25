# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 11:04:01 2017

@author: gandhik
"""
import cmath
import numpy as np
from scipy import signal
import time
imag = cmath.sqrt(-1)

binary_dict = {}
binary_dict[0] = [[]]
binary_dict[1] = np.array([[1], [-1]])

def gen_rand_seq(length, m=2):
    """
    Function to generate a random sequence of a given length, with values taken from the mth roots of unity.
    
    Arguments:
        length (int): length of the sequence to be generated
        m (int): the roots of unity to fill the sequence.  Defaults to 2 (binary sequence)

    Returns:
        a numpy array of size length, filled with random roots of unity.        
        
    """
    poss = []
    probs = []
    for i in range(m): # append the mth roots of unities as the possibilities, each with probability 1/m
        poss.append(cmath.exp(2*cmath.pi*i*imag/m))  
        probs.append(1./m)
    return np.random.choice(poss, size = (length,), p = probs) # generate the desired sequence

    
def gen_population(num_members, length, m=2):
    """
    Function to generate a population of random sequences of a given length, with values taken from the mth roots of unity.
    
    Arguments:
        num_members (int): number of sequences to be generated
        length (int): length of the sequences to be generated
        m (int): the roots of unity to fill the sequences.  Defaults to 2 (binary sequence)

    Returns:
        an numpy array of numpy arrays of size length filled with random roots of unity.      
        
    """
    return np.array([gen_rand_seq(length, m) for x in xrange(num_members)])
    
def gen_binary_codes(length, optional_limit = None):
    """
    Function to generate all of the binary codes of a given length.
    
    Arguments:
        length (int): length of the codes to be generated
        optional_limit (int): not relevant to this code, used as a placeholder.

    Returns:
        an numpy array of arrays, each of which is a binary code of size length.          
        
    """
    if length in binary_dict:
        return binary_dict[length] # for efficiency's sake, if we want to store these for faster use
    else:
        all_codes = []
        trunc_codes = gen_binary_codes(length-1, optional_limit) # the codes of length one shorter
        for code in trunc_codes: # a for loop to add a 1 and -1 to the end of each of these possible codes
            code_1 = np.append(code, [1])
            code_2 = np.append(code, [-1])
            all_codes.append(code_1)
            all_codes.append(code_2)
        return np.array(all_codes)
    
def gen_polyphase_codes(length, phase_unity):
    """
        Function to generate all of the polyphase codes of a given length and phase_unity.

        Arguments:
            length (int): length of the codes to be generated
            phase_unity (int): order of the roots of unity in our codes

        Returns:
            an numpy array of arrays, each of which is a polyphase code of the given length.

        """
    polyphase_dict = {}
    polyphase_dict[1] = np.array([])

    for i in range(phase_unity):
        polyphase_dict[1] = np.append(polyphase_dict[1], [cmath.exp(2*cmath.pi*imag*i/phase_unity)])

    return gen_polyphase_codes_helper(length, phase_unity, polyphase_dict)

def gen_polyphase_codes_helper(length, phase_unity, polyphase_dict):
    """
        Helper function that recursively generates the polyphase codes for gen_polyphase_codes

        Arguments:
            length (int): length of the codes to be generated
            phase_unity (int): order of the roots of unity in our codes
            polyphase_dict (dictionary): dictionary storing the polyphase codes

        Returns:
            an numpy array of arrays, each of which is a polyphase code of the given length.

        See also: gen_polyphase_codes

        """
    if length in polyphase_dict:
        return polyphase_dict[length] # return the codes of a given length if they have already been generated

    else:
        all_codes = []
        trunc_codes = gen_polyphase_codes_helper(length-1, phase_unity, polyphase_dict) # the codes of length one shorter
        for code in trunc_codes:  # a for loop to add all possible additions to these codes
            for elem in range(phase_unity):
                new_code = np.append(code, [cmath.exp(2*cmath.pi*imag*elem/phase_unity)])
                all_codes.append(new_code)
        return np.array(all_codes)

def gen_arb_amplitude_codes(length, limit):
    """
    Function to generate all of the codes of a given length, with amplitudes ranging integrally to a provided limit.
    
    Arguments:
        length (int): length of the codes to be generated
        limit (int): limit on the absolute value of the amplitude of the code (we only use integral amplitudes)

    Returns:
        an numpy array of arrays, each of which is a arbitrary amplitude code of size length.          
        
    """
    arb_dict = {}
    arb_dict[1] = np.array([])

    for i in range(1, limit): # generating the base case of all possible one element codes
        arb_dict[1] = np.append(arb_dict[1], [i])
        arb_dict[1] = np.append(arb_dict[1], [-i])

    return gen_arb_ampl_helper(length, limit, arb_dict)

def gen_arb_ampl_helper(length, limit, arb_dict):
    """
    Helper function that recursively generates the arbitrary amplitude codes for gen_arb_amplitude_codes
    
    Arguments:
        length (int): length of the codes to be generated
        limit (int): limit on the absolute value of the amplitude of the code (we only use integral amplitudes)
        arb_dict (dictionary): dictionary storing the arbitrary amplitude codes

    Returns:
        an numpy array of arrays, each of which is a arbitrary amplitude code of size length.          
    
    See also: gen_arb_amplitude_codes
        
    """
    if length in arb_dict:
        return arb_dict[length] # return the codes of a given length if they have already been generated

    else:
        all_codes = []
        trunc_codes = gen_arb_ampl_helper(length-1, limit, arb_dict) # the codes of length one shorter
        for code in trunc_codes:  # a for loop to add all possible additions to these codes
            for elem in range(1, limit):
                code_1 = np.append(code, [elem])
                code_2 = np.append(code, [-elem])
                all_codes.append(code_1)
                all_codes.append(code_2)
        return np.array(all_codes)
        

def second_max(conv, optarg = None):
    """
    Function to calculate the second-to-max value in a list.**
    
    Arguments:
        conv (array): autocorrelation of the code in question
        optarg (arbitrary type): filler variable.

    Returns:
        the second-to-max value in conv         
    
    See also: second_max_nondoppler
    
    **: code taken from stackoverflow, in large part, although rewritten from my purposes.  
    """
    count = 0
    first_max = second_max = float('-inf') # variables to keep track of the maxima in the list
    for x in conv:
        count += 1
        if abs(x) > second_max: 
            if abs(x) >= first_max:
                first_max, second_max = abs(x), first_max  # altering the values when encountering an x that is greater than first_max            
            else:
                second_max = abs(x) # altering the values when encountering an x that is greater than second_max, but not first_max
    return second_max if count >= 2 else None
    

def second_max_nondoppler(conv, optarg):
    """
    Alternative function to calculate the second-to-max value in a list, given that the peak is in the center.
    
    Arguments:
        conv (array): autocorrelation of the code in question
        optarg (arbitrary type): filler variable.

    Returns:
        the second-to-max value in conv
    
    See also: second_max            
        
    """
    max_val = 0
    for i in range(len(conv)):
        if i != (len(conv)-1)/2:
            if abs(conv[i]) > max_val:
                max_val = abs(conv[i])
    
    return max_val


def rand_select_algorithm_doppler(length, phase_unity, freq_band, num_elem=None, min_fn=second_max):
    """
    Function to execute a random selection algorithm for the minimizing peak sidelobe problem.  In particular,
    this algorithm generates codes of length _length_, with values taken from the _phase_unity_th roots of unity for a given time.
    The algorithm then outputs the minimum sidelobe in the range-doppler space, with the doppler width given by freq_band.

    Arguments:
        length (int): length of the codes to generate
        phase_unity (int): number from which to draw the roots of unity
        freq_band (int): doppler width in which we are minimizing
        num_elem (int): the number of elements min_fn takes in as inputs
        min_fn (fn): the function we are minimizing

    Returns:
        a tuple consisting of a code and a value, corresponding to the minimal code and value found by the algorithm

    """
    f = open('best_known_sidelobes.txt')
    lines = f.readlines()
    best_code = None
    best_val = float('inf')
    cutoff_val = int(lines[length - 1])  # cutoff value after which the algorithm can stop
    time_limit = 0
    if length <= 30:
        time_limit = 1*15
    else:
        time_limit = 1 * (300 + 60 * (length - 30))
    start_time = time.time()
    while time.time() - start_time < time_limit and (best_val != cutoff_val or freq_band!=1 or phase_unity!=2):
        x = gen_rand_seq(length, phase_unity) # generate a random sequence given the specs
        psl_value = fast_autocorrelations(x, freq_band, num_elem, min_fn) # find the psl_value, update the best value if this improves on it.
        if psl_value < best_val:
            best_val = psl_value
            best_code = x

    return (best_code, best_val)


def fft_psl(code):
    """
    Function to calculate the peak sidelobe of a code.  In particular, it calculates the autocorrelation by convolving the code
    with its reverse.  To calculate the convolution, a fft is used here.
    
    Arguments:
        code (array): the code for which we wish to find the peak sidelobe

    Returns:
        the peak sidelobe of _code_       
    
    See also: con_psl
    """
    lst = signal.fftconvolve(code, code[::-1])
    return second_max_nondoppler(lst, None)
    
def con_psl(code):
    """
    Function to calculate the peak sidelobe of a code.  In particular, it calculates the autocorrelation by convolving the code
    with its reverse.  The convolution is calculated directly here.
    
    Arguments:
        code (array): the code for which we wish to find the peak sidelobe

    Returns:
        the peak sidelobe of _code_
    
    See also: fft_psl
        
    """
    lst = signal.convolve(code, code[::-1])
    return second_max(lst, None)

def con_psl_matrix(code):
    """
    Function to calculate the autocorrelation of a code.

    Arguments:
        code (array): the code in question

    Returns:
        the autocorrelation of the given code.
    """
    return signal.convolve(code, code[::-1])
    
def fast_autocorrelations(code, N, num_elem = None, op_on_list = second_max):
    """
    Function to calculate op_on_list on the ambiguity function of the code, with frequency band N.
    
    Arguments:
        code (array): the code on which we are evaluating op_on_list
        N (int): doppler width in which we are computing
        num_elem (int): the number of elements of the list that concern us
        op_on_list (fn): the function we are computing

    Returns:
        a value corresponding to the evaluation of op_on_list on the ambiguity function of code.
        
    """
    """
    if N==1 and op_on_list == second_max:
        return con_psl(code)
    """
    overall_lst = []
    for freq_delay in range(N):  
        freq_shifted_code = []
        for index in range(len(code)):
            constant_multiplier = cmath.exp(-2*cmath.pi*imag*index*freq_delay/N) # constant shifting multiplier to multiply the given part of the code we are considering
            freq_shifted_code.append(constant_multiplier*code[index].conjugate())
        lst = signal.convolve(code, freq_shifted_code[::-1]) # convolving the original code with the frequency shifted code
        lst = lst[:(len(lst) / 2 + 1)]
        for i in lst:
            overall_lst.append(abs(i)) # appending the absolute value to our list, on which we will evaluate op_on_list
    return op_on_list(overall_lst, num_elem)


def fast_autocorrelations_matrix(code, N, num_elem=None):
    """
    Function to calculate the ambiguity function of the code, with frequency band N.

    Arguments:
        code (array): the code in question
        N (int): doppler width in which we are computing
        num_elem (int): the number of elements of the list that concern us

    Returns:
        a list of values corresponding to those in the ambiguity function of the given code

    """
    overall_lst = []
    for freq_delay in range(N):
        freq_shifted_code = []
        for index in range(len(code)):
            constant_multiplier = cmath.exp(-2 * cmath.pi * imag * index * freq_delay / N)  # constant shifting multiplier to multiply the given part of the code we are considering
            freq_shifted_code.append(constant_multiplier * code[index].conjugate())
        lst = signal.convolve(code,
                              freq_shifted_code[::-1])  # convolving the original code with the frequency shifted code
        lst = lst[:(len(lst)/2 + 1)]
        for i in lst:
            overall_lst.append(abs(i))  # appending the absolute value to our list, on which we will evaluate op_on_list
    return overall_lst

def gen_fn_top_j_values(lst, fn, j):
    """
    Abstracted function to calculate a function of the top j values (excluding the peak) of a array.
    
    Arguments:
        lst (array): array we are considering
        fn (fn): function we are evaluating on lst
        j (int): the number of elements we are considering

    Returns:
        fn evaluated on the top j elements (excluding the peak) of lst       
        
    """
    def fn_top_j_values(lst, j):
        lst = np.array(lst)
        lst = np.sort(lst)
        sl = 0
        for m in range(1,min(j+1, len(lst))):
            sl = sl + fn(lst[len(lst)-m-1]) # we importantly exclude the peak by ranging from m=1, not m=0
        return sl
    
    return fn_top_j_values

def sum_values(lst, j):
    """
    Function to provide to fast_autocorrelations that sums the top j values (excluding the peak) of a list.
    
    Arguments:
        lst (array): array we are considering
        j (int): the number of elements we are considering

    Returns:
        the sum of the top j values, excluding the peak, of lst     
        
    """
    fn = gen_fn_top_j_values(lst, lambda x: x, j)
    return fn(lst, j)

def sum_square_values(lst, j):
    """
    Function to provide to fast_autocorrelations that sums the squares of the top j values (excluding the peak) of a list.
    
    Arguments:
        lst (array): array we are considering
        j (int): the number of elements we are considering

    Returns:
        the sum of the squares of the top j values, excluding the peak, of lst     
        
    """
    fn = gen_fn_top_j_values(lst, lambda x: x**2, j)
    return fn(lst, j)

def isl(lst):
    """
    Function to provide to fast_autocorrelations that calculates the integrated sidelobe level of a code.

    Arguments:
        lst (array): array we are considering

    Returns:
        the integrated sidelobe level of lst
    """
    return sum_square_values(lst, len(lst)-1)

    
def mult_codes(code_1, code_2):
    """
    Function to multiply two codes together, calculating the absolute value of the dot product of the two vectors.
    
    Arguments:
        code_1 (array): the first code to be multiplied
        code_2 (array): the second code to be multiplied

    Returns:
        the absolute value of sum_i code_1[i] * code_2[i].         
        
    """
    total = 0
    for i in range(len(code_1)):
        total += code_1[i] * code_2[i]
    return abs(total)