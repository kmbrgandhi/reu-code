# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 10:30:57 2017

@author: gandhik
"""
import time
import cmath
import math
import numpy as np
import random
from scipy import signal
imag = cmath.sqrt(-1)
from scipy import integrate
import scipy
import matplotlib.pyplot as plt
from conv_utils import second_max

def autocorrelation(waveform):
    """
    Function to calculate the autocorrelation of a waveform.
    :param waveform: the waveform in question
    :type waveform: Waveform object
    :return: the autocorrelation in question, with a unit separation
    """
    lst = []

    for delay in range(waveform.length):
        fn = delay_fn(waveform.function, delay)
        # note that we have to integrate the real and imaginary parts separately, due to scipy's constraints
        real_val = integrate.quad(lambda x: scipy.real(fn(x)), 0, waveform.length + delay)
        imag_val = integrate.quad(lambda x: scipy.imag(fn(x)), 0, waveform.length + delay)
        lst.append(abs(real_val[0] + 1j*imag_val[0]))

    return lst

def delay_fn(fn, delay, freq = 0):
    """
    Function to return, given a waveform function, the delay function integrated for the ambiguity function.

    :param fn: the waveform function in question
    :type fn: fn
    :param delay: the delay amount
    :type delay: int or float
    :param freq: the frequency delay
    :type freq: int or float
    :return: the delay function to be integrated by the autocorrelation or ambiguity function
    """
    def fn_to_return(x):
        return fn(x) * np.conj(fn(x - delay)) * cmath.exp(imag*2*cmath.pi*freq*x)
    return fn_to_return


def ambiguity_function(waveform, N):
    """
    Function to calculate the ambiguity function of a waveform with a specified doppler width N, as well as units
    scaled down by const_1 and const_2.

    :param waveform: the waveform in question
    :type waveform: Waveform object
    :param N: the doppler width of our ambiguity function
    :type N: int
    :return: the ambiguity function of waveform.
    """
    lst = []
    const_1 = 3
    const_2 = 3
    bound_1 = const_1*waveform.length # range of delays
    for delay in range(-bound_1, bound_1 + 1):
        row = []
        bound_2 = N * const_2 # range of frequencies
        for freq in range(-bound_2, bound_2 + 1):
            fn = delay_fn(waveform.function, delay/const_1, freq/const_2)
            # note that we have to integrate the real and imaginary parts separately, due to scipy's constraints
            real_val = integrate.quad(lambda x: scipy.real(fn(x)), -float('inf'), float('inf'))
            imag_val = integrate.quad(lambda x: scipy.imag(fn(x)), -float('inf'), float('inf'))
            row.append(abs(real_val[0] + 1j*imag_val[0]))
        lst.append(row)
    return lst

# a bare-bones waveform class.
class Waveform:
    def __init__(self, length, function):
        self.length = length
        self.function = bound_fn(function, length)

def bound_fn(function, length):
    """
    Function to return a bound version of a function from 0 to length.
    :param function: the function in question
    :type function: fn
    :param length: the length limit
    :type length: float
    :return: a bounded version f' of a function f, where f'(x) = f(x) when 0<=x<=length, and f'(x) = 0 elsewhere.
    """
    def fn_to_return(x):
        if 0<=x<=length:
            return function(x)
        else:
            return 0
    
    return fn_to_return

def square_pulse():
    """
    Function to return a square pulse function, fixed amplitude function.

    :return: a square pulse function
    """
    def fn_to_return(x):
        return 1
    return fn_to_return

def lin_chirp_fn(init_freq, freq_change, init_phase = 0):
    """
    Function to return a linear frequency modulation chirp function with a given initial frequency, phase, and freq_change rate.
    :param init_freq: initial frequency of the chirp
    :type init_freq: float
    :param freq_change: rate of change of the frequency
    :type freq_change: float
    :param init_phase: initial phase of the wave
    :type init_phase: float
    :return: a linear chirp function
    """
    def fn_to_return(x):
        return cmath.sin(init_phase + 2*cmath.pi*(init_freq * x + freq_change*x*x/2))
    return fn_to_return

def poly_chirp_fn(degree, init_freq, freq_change, init_phase = 0):
    """
    Function to return a polynomial frequency modulation chirp function with a given degree, initial frequency, phase, and freq_change rate.
    :param degree: degree of the polynomial
    :type degree: int
    :param init_freq: initial frequency of the chirp
    :type init_freq: float
    :param freq_change: rate of change of the frequency
    :type freq_change: float
    :param init_phase: initial phase of the wave
    :type init_phase: float
    :return: a polynomial chirp function of the given degree
    """
    def fn_to_return(x):
        return cmath.sin(init_phase + 2*cmath.pi*(init_freq * x + freq_change*(x**(degree+1))/(degree+1)))
    return fn_to_return

def exp_chirp_fn(init_freq, freq_change, init_phase = 0):
    """
    Function to return a exponential frequency modulation chirp function with a given initial frequency, phase, and freq_change rate.
    :param init_freq: initial frequency of the chirp
    :type init_freq: float
    :param freq_change: rate of change of the frequency
    :type freq_change: float
    :param init_phase: initial phase of the wave
    :type init_phase: float
    :return: a exponential chirp function
    """
    def fn_to_return(x):
        return cmath.sin(init_phase + 2*cmath.pi*init_freq * (freq_change**(x) - 1)/(cmath.log(freq_change)))
    return fn_to_return

"""
waveform = Waveform(10, square_pulse())
lst = ambiguity_function(waveform, 5)
fig = plt.figure(figsize = (6, 3.2))
ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(lst)
ax.set_aspect('equal')

plt.show()

for i in range(1, 50):
	print(i)
	waveform = Waveform(10, lin_chirp_fn(i/100, 0.5))
	lst = autocorrelation(waveform)

"""



