

import cmath
imag = cmath.sqrt(-1)

from conv_utils import gen_binary_codes, fast_autocorrelations, sum_values, mult_codes, second_max, gen_polyphase_codes

def peak_sidelobes_single(length):
    """
    A function to return the binary code of a given length with the minimum peak sidelobe

    :param length: length of the binary codes in question
    :type length: int
    :return: the binary code with minimal peak sidelobe, as well as that peak sidelobe value.

    See also: peak_sidelobes_general, peak_sidelobes_efficient
    """
    codes = gen_binary_codes(length, None)
    
    min_sl = float("inf")
    min_code = None
    for code in codes:
        val = fast_autocorrelations(code, 1) # calculate the peak sidelobe
        if val < min_sl: # update if it improves on the previous value
            min_sl = val
            min_code = code
        if min_sl == 1:
            break
    
    return (min_code, min_sl)

def peak_sidelobes_general(length, num_elem, N, gen_codes, op_on_list = second_max, optional_limit = 0):
    """

    :param length: length of the code in question
    :type length: int
    :param num_elem: number of elements that concern our minimization function
    :type num_elem: int
    :param N: doppler window for which we are calculating the ambiguity function
    :type N: int
    :param op_on_list: the operation we are minimizing of the ambiguity function
    :type op_on_list: fn
    :param gen_codes: function to generate the codes we are considering
    :type gen_codes: fn
    :param optional_limit: optimal limit for gen_codes, if we are dealing with arbitrary amplitude codes
    :type optional_limit: int
    :return: the code with the minimum value of op_on_list on its ambiguity function, as well as that value.

    See also: peak_sidelobes_single, peak_sidelobes_efficient
    """
    all_codes = gen_codes(length, optional_limit) # generate the codes over which we are minimizing
    min_sl = float('inf')
    min_code = None
    for code in all_codes:
        sidelobe_size = fast_autocorrelations(code, N, num_elem, op_on_list) # calculate the value of op_on_list on this code
        if sidelobe_size < min_sl: # update the value if it improves
            min_sl = sidelobe_size
            min_code = code
        if min_sl == 1: # break if we reach 1, since we can never do better.
            break
    return (min_code, min_sl)


# Efficient Exhaustive; note that this is modeled off of pseudocode and ideas of Coxson, Russ.

def conv_ineff(left, right, N, length):
    """
    Inefficient convolution of the finished code from our peak sidelobe exhaustive algorithm, which outputs the peak sidelobe
    of the finished code.  If the length of the code in question is odd, it tries both possible values for the middle entry.

    :param left: left-hand side of the code built up
    :type left: array
    :param right: right-hand side of the code
    :type right: array
    :param N: doppler width
    :type N: int
    :param length: length of the codes we are considering
    :type length: int
    :return: peak sidelobe of possible codes from combining left and right
    """
    if length %2 == 0: # if it is even, the only possible code is exactly their concatenation
        l_copy = left[:]
        r_copy = right[:]
        r_copy.reverse()
        signal = l_copy + r_copy
        convert(signal) # convert to -1, 1 form
        return fast_autocorrelations(signal, N)
    else: # otherwise, the middle entry can be 1 or -1; we check both
        l_copy = left[:]
        l_copy.append(1)
        r_copy = right[:]
        r_copy.reverse()
        signal = l_copy + r_copy
        convert(signal)
        corr_1 = fast_autocorrelations(signal, N)
        l_copy[-1] = 0
        signal = l_copy + r_copy
        convert(signal)
        corr_2 = fast_autocorrelations(signal, N)
        return min(corr_1, corr_2)

def convert(signal):
    """
    Helper function that converts a 0,1 signal to -1,1 signal

    :param signal: code in question
    :type signal: array
    :return: a -1, 1 version of the signal
    """
    for i in range(len(signal)):
        if signal[i] == 0:
            signal[i] = -1

    return signal


def inc(left, right, vals):
    """
    General function for incrementing left, right to the next possibility.

    :param left: left-hand side of the code built up
    :type left: array
    :param right: right-hand side of the code
    :type right: array
    :param vals: array storing the value of the left (and reversed right) in binary
    :return: the updated left, right arrays
    """
    inc_helper(left, right, vals)
    while vals[1] > vals[0]: # if the right-hand side has a larger binary form, we can discard this option.
        inc_helper(left, right, vals)

    return left, right


def inc_helper(left, right, vals):
    """
    Helper function for incrementing the left and right arrays.
    :param left: left-hand side of the code built up
    :type left: array
    :param right: right-hand side of the code
    :type right: array
    :param vals: array storing the value of the left (and reversed right) in binary
    :return: the updated left, right arrays
    """
    if right[-1] == 0: # increment 0 to 1 in the right array if possible
        right[-1] = 1
        vals[1]+=1
    else: # otherwise, increment the left 0 to 1, right to 0, or, if this is not possible, pop both and reapply the function to the remainder.
        if left[-1] == 0:
            left[-1] = 1
            right[-1] = 0
            vals[0]+=1
            vals[1]-=1
        else:
            left.pop()
            right.pop()
            vals[0], vals[1] = (vals[0]-1)/2, (vals[1]-1)/2
            inc_helper(left, right, vals)
    if vals[1] > vals[0]: # perhaps redundant
        inc_helper(left, right, vals)

    return left, right


def s_nondoppler(lst1, lst2, N):
    """
    Handler function for calculating the next sidelobe, given the current state of left, right.
    :param lst1: The left array.
    :type lst1: array
    :param lst2: The right array.
    :type lst2: array
    :return: the sidelobe corresponding to the multiplication of these two arrays.  An analogue of mult_codes for 0, 1 arrays.
    """
    return bal(xnor(lst1, lst2))


def s_doppler(lst1, lst2, N):
    """
    Function for updating the maximal sidelobe given the current state of left, right, in the doppler regime.

    :param lst1: the left array
    :type lst1: array
    :param lst2: the right array
    :type lst2: array
    :param N: doppler width of our ambiguity function
    :type N: int
    :return: the maximal sidelobe, over all freq_delays, corresponding to the multiplication of these left/right arrays
    """
    max_val = -float('inf')
    lst1_copy = lst1[:]
    lst2_copy = lst2[:]
    lst1_copy = convert(lst1_copy)
    lst2_copy = convert(lst2_copy)

    # using the copied, converted lists, perform a similar operation to fast_autocorrelation except only for one specific
    # range delay and all of the frequency delays, returning the max sidelobe over all of them.
    for freq_delay in range(N):
        new_lst1 = lst1_copy[:]
        for index in range(len(lst1_copy)):
            lst1_copy[index]*=cmath.exp(-2 * cmath.pi * imag * index * freq_delay / N)
        val = mult_codes(lst1_copy, lst2_copy)
        max_val = max(val, max_val)
    return max_val


def xnor(lst1, lst2):
    """
    Function computing the xnor, element-by-element, of two arrays.

    :param lst1: the first array
    :type lst1: array
    :param lst2: the second array
    :type lst2: array
    :return: the xnor of the two arrays
    """
    new_lst = []
    length = len(lst1)
    for i in range(length):
        if lst1[i] == lst2[length-1-i]: # multiplying 1 and 1 or -1 and -1.
            new_lst.append(1)
        else:
            new_lst.append(0)

    return new_lst

def bal(lst):
    """
    The balance of a list; the difference between the number of 1's and -1's.

    :param lst: the list in question
    :type lst: array
    :return: the difference between the # of 1's and -1's in lst.
    """
    num_1 = 0
    num_0 = 0
    for i in range(len(lst)):
        if lst[i] == 1:
            num_1 +=1
        else:
            num_0 +=1
    return num_1 - num_0

def peak_sidelobes_efficient(length, s_computation = s_nondoppler, init_length = 0, thresh = float('inf'), N=1):
    """
    Efficient peak sidelobe generator for doppler space, binary codes.

    :param length: length of the code in consideration
    :type length: int
    :param s_computation: function to compute sidelobes
    :type s_computation: function
    :param init_length: initial length of left, right for parallelization purposes
    :type init_length: int
    :param thresh: threshold
    :type thresh: int
    :param N: doppler width
    :type N: int
    :return: minimum peak sidelobe value for the given length

    See also: peak_sidelobes_general, peak_sidelobes_single
    """
    left = []
    right = []
    best_code = None
    vals = [0, 0] # storing binary forms of right, left
    n2 = length/2
    left.append(0)
    right.append(0)
    """
    cutoff_val = float('inf')
    if length < 14:
        cutoff_val = 1
    elif length < 29:
        cutoff_val = 2
    elif length< 52:
        cutoff_val = 3
    else:
        cutoff_val = 4
    """
    while left[0]==0: # by negation, alternation, this is only option with length is even
        s = s_computation(left, right, N)
        while(abs(s) < thresh and len(left) < n2): # building up by adding 0's.  inc changes 0's to 1's.
            left.append(0)
            right.append(0)
            vals[0]*=2
            vals[1]*=2
            s = s_computation(left, right, N)
        if len(left) ==n2: # if fully built, check if left, right form a code that has a good convolution.  if so, update thresh.
            if abs(s) < thresh:
                x = conv_ineff(left, right, N, length)
                if x < thresh:
                    best_code = left + right
                    thresh = x
        inc(left, right, vals)
        #if thresh == cutoff_val:
            #break

    if length %2 == 1: #and thresh!=cutoff_val: # if length is odd, alternation argument does not provide redundancy, we need this extra check.
        left = [1]
        right = [0]
        vals = [1, 0]
        while right[0] == 0:
            s = s_computation(left, right, N)
            while (abs(s) < thresh and len(left) < n2):
                left.append(0)
                right.append(0)
                vals[0] *= 2
                vals[1] *= 2
                s = s_computation(left, right, N)
            if len(left) == n2:
                if abs(s) < thresh:
                    x = conv_ineff(left, right, N, length)
                    if x < thresh:
                        thresh = x
            inc(left, right, vals)
            #if thresh == cutoff_val:
                #break


    return (best_code, thresh)

# this was not completed, but would follow a similar model to peak_sidelobe_efficient.
def peak_sidelobe_efficient_polyphase(length, s_computation = s_nondoppler, N=1, thresh = float('inf')):
    return



