

from conv_utils import con_psl, fast_autocorrelations, second_max, gen_population, gen_rand_seq
from polyphase import gen_rand_seq_poly, fast_autocorrelations_phases, norm
import numpy as np
import random
import time
import math, cmath


class Particle:
    """
    Particle class.  Each particle is characterized by a given code, velocity, and best found min_fn state.  At a given time,
    the particle moves around according to its velocity, which is updated using omega, const_1, and const_2.  P_best
    is updated using N, num_elem, min_fn, as well as the global best from the entire series of particles.
    """
    def __init__(self, code, omega, const_1, const_2, N=1, num_elem = None, min_fn = second_max):
        """
        Function to initialize a particle.
        :param code: the starting code
        :type code: array
        :param omega: weight on the initial velocity in the update equation
        :type omega: float
        :param const_1: weight on the p_best in the update equation
        :type const_1: float
        :param const_2: weight on the g_best in the update equation
        :type const_2: float
        :param N: doppler width of our ambiguity function
        :type N: int
        :param num_elem: number of elements that min_fn is being evaluated on
        :type num_elem: int
        :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
        :type min_fn: function
        """
        self.code = code
        self.p_best = code
        self.val_best = fast_autocorrelations_phases(code, N, num_elem, min_fn)
        self.velocity = np.random.rand(len(code))
        self.c1 = const_1
        self.c2 = const_2
        self.omega = omega
        self.N = N
        self.num_elem = num_elem
        self.min_fn = second_max

    def update_velocity(self, g_best):
        """
        Updates the particle's velocity given its p_best, velocity, constants, and g_best.  Adds omega*itself to
        const_1 times its distance from p_best to const_2 times its distance to g_best.

        :param g_best: global best of all particles in the swarm
        :type g_best: array
        :return: none
        """
        self.velocity = np.multiply(self.velocity, self.omega)
        first_scale = self.c1 * random.random()
        first_array = np.subtract(self.p_best, self.code) # distance from particle's best; tends us to move towards there
        first_to_add = np.multiply(first_array, first_scale)
        self.velocity = np.add(self.velocity, first_to_add)
        second_scale = self.c2 * random.random()
        second_array = np.subtract(g_best, self.code) # distance from global best; tends us to move towards there
        second_to_add = np.multiply(second_array, second_scale)
        self.velocity = np.add(self.velocity, second_to_add)
        self.velocity = normalize(self.velocity, cmath.pi) # normalize so the velocity doesn't grow unbounded

    def update_code(self, g_best):
        """
        Updates the particle's position given its p_best, velocity.  First updates velocity using update_velocity, then
        updates position, then updates p_best if necessary.

        :param g_best: global best of all particles in the swarm
        :type g_best: array
        :return: none
        """
        self.update_velocity(g_best)
        self.code = np.add(self.code, self.velocity)
        self.code = make_normal(self.code) # normalize code so that it is mod 2*pi.
        val = fast_autocorrelations_phases(self.code, self.N, self.num_elem, self.min_fn) # calculate min_fn value; if it improves on val_best, update val_best and p_best
        if val < self.val_best:
            self.val_best = val
            self.p_best = self.code

def make_normal(seq):
    """
    Function to normalize a phase sequence so that all phases are between 0 and 2*pi.

    :param seq: phase sequence in question
    :type seq: array
    :return: normalized sequence mod 2*pi
    """
    for i in range(len(seq)):
        seq[i] = math.fmod(seq[i], 2 * math.pi)

    return seq

def normalize(seq, desired):
    """
    Function to normalize a sequence to a given norm.

    :param seq: phase sequence in question, normally the velocity of a particle
    :param desired: desired norm
    :return: scaled version of seq with a norm _desired_
    """
    norm_seq = norm(seq)
    seq = np.multiply(seq, desired/norm_seq)
    return seq


def gen_best_particle(pop, best_code, best_val):
    """
    Function to generate the particle with the best min_fn particle among a population of particles.

    :param pop: population of particles
    :type pop: array
    :param best_code: current best code
    :type best_code: array
    :param best_val: current best value
    :type best_val: complex number or float.
    :return: two outputs, the new best_code and new best_val found by our algorithm
    """
    min_i = 0
    min_val = float('inf')
    for i in range(len(pop)):
        if pop[i].val_best < min_val:
            min_i = i
            min_val = pop[i].val_best

    if min_val < best_val:
        best_val = min_val
        best_code = pop[min_i].code

    return best_code, best_val

def gen_particle_population(num_members, length, phase_unity, omega = 1, const_1 = 2, const_2 = 2, N = 1, num_elem = None, min_fn = second_max):
    """
    Function to generate a population of particles with codes of a given length, as well as rules for the minimization and
    constants for the velocity updates.

    :param num_members: number of particles in our population
    :type num_members: int
    :param length: length of our codes
    :type length: int
    :param phase_unity: order of the roots of unity filling our codes
    :type phase_unity: int
    :param omega: weighting factor on the initial velocity in the update step
    :param const_1: weighting factor on the particle best distance in the update step
    :param const_2: weighting factor on the global best distance in the update step
    :param N: doppler width of our ambiguity function
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: population of particles with initialized code of length _length_
    """
    pop = []

    # initialize particles
    for i in range(num_members):
        new_particle = Particle(gen_rand_seq_poly(length), omega, const_1, const_2, N, num_elem, min_fn) # new, random particle with a given length code
        pop.append(new_particle)
    return pop


def particle_swarm_polyphase(length, phase_unity = 2, omega = 1, const_1 = 2, const_2 = 2, N = 1, num_elem = None, min_fn = second_max):
    """
    Particle swarm algorithm to find a arbitrary polyphase code with a minimal min_fn value.  Simulates a swarm of particles moving
    around, with their velocities guided by their own best and the global best found so far.

    :param length: length of the codes in question
    :param phase_unity: order of the roots of unity filling our codes
    :type phase_unity: int
    :param omega: weighting factor on the initial velocity in the update step
    :param const_1: weighting factor on the particle best distance in the update step
    :param const_2: weighting factor on the global best distance in the update step
    :param N: doppler width of our ambiguity function
    :param num_elem: number of elements that min_fn is being evaluated on
    :type num_elem: int
    :param min_fn: function of the ambiguity function/autocorrelation we are minimizing
    :type min_fn: function
    :return: the best polyphase code and the associated min_fn value found
    """
    num_members = 20*length
    pop = gen_particle_population(num_members, length, phase_unity, omega, const_1, const_2, N, num_elem, min_fn) # generate the population of particles
    best_code = None
    best_val = float('inf')
    time_limit = 0
    if length <= 30:
        time_limit = 15
    else:
        time_limit = 1 * (300 + 60 * (length - 30))
    start_time = time.time()

    best_code, best_val = gen_best_particle(pop, best_code, best_val) # generate the first best particle, best_val pair
    count = 1
    while time.time() - start_time < time_limit and best_val > 1:
        for i in range(len(pop)): # update all of the positions
            pop[i].update_code(best_code)

        best_code, best_val = gen_best_particle(pop, best_code, best_val) #update the best particle, best_val pair
    return (best_code, best_val)



