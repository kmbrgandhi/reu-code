# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:23:53 2017

@author: gandhik
"""

import cmath
import random
import numpy as np
imag = cmath.sqrt(-1)

def crossover_random(population_length, parents):
    """
    Function to crossover parents and produce enough to fill a population.  As a note, this could be a general population
    already defined, or could simply be a placeholder list to generate a set amount of offspring.  Here, we use a
    random crossover model.

    :param population_length: the desired population length
    :type population_length: int
    :param parents: array of parents, each of which are arrays
    :type parents: array
    :return: the updated parents array, to which the children have been added.
    """
    parents_length = len(parents)
    desired_length = population_length - parents_length
    children = []
    while len(children) < desired_length:  # add children until we have enough to fill the population
        father = random.randint(0, parents_length - 1)
        mother = random.randint(0, parents_length - 1) # generate a father and mother, and breed them
        if father != mother:
            father = parents[father]
            mother = parents[mother]
            child = []
            for i in range(len(father)):  # to generate the child, for each index, randomly sample either the father or mother
                m_or_f = random.sample([-1, 1],  1)
                if m_or_f == 1:
                    child.append(mother[i])
                else:
                    child.append(father[i])
            
            children.append(child)
    
    parents.extend(children)
    
    return parents

def crossover_halfhalf(population_length, parents):
    """
        Function to crossover parents and produce enough to fill a population.  As a note, this could be a general population
        already defined, or could simply be a placeholder list to generate a set amount of offspring.  Here, we use a
        half-half crossover model.

        :param population_length: the desired population length
        :type population_length: int
        :param parents: array of parents, each of which are arrays
        :type parents: array
        :return: the updated parents array, to which the children have been added.
    """
    parents_length = len(parents)
    desired_length = population_length - parents_length
    children = []
    while len(children) < desired_length: # add children until we have enough to fill the population
        father = random.randint(0, parents_length - 1)
        mother = random.randint(0, parents_length-1) # generate a father and mother, and breed them
        if father != mother:
            father = parents[father]
            mother = parents[mother]
            point = len(father)/2
            child = []
            for i in range(len(father)): # to generate the child, take the first half of the father and second half of the mother
                if i<point:
                    child.append(father[i])
                else:
                    child.append(mother[i])

            
            children.append(child)
    
    parents.extend(children)
    
    return parents

def crossover_randpoint(population_length, parents):
    """
        Function to crossover parents and produce enough to fill a population.  As a note, this could be a general population
        already defined, or could simply be a placeholder list to generate a set amount of offspring.  Here, we use a
        rand-point crossover model.

        :param population_length: the desired population length
        :type population_length: int
        :param parents: array of parents, each of which are arrays
        :type parents: array
        :return: the updated parents array, to which the children have been added.
    """
    parents_length = len(parents)
    desired_length = population_length - parents_length
    children = []
    while len(children) < desired_length: # add children until we have enough to fill the population
        father = random.randint(0, parents_length - 1)
        mother = random.randint(0, parents_length-1) # generate a father and mother, and breed them
        if father != mother:
            father = parents[father]
            mother = parents[mother]
            point = random.randint(0, len(father) - 1)
            child = []
            for i in range(len(father)):
                if i<point:
                    child.append(father[i])
                else:
                    child.append(mother[i])

            children.append(child)
    
    parents.extend(children)
    
    return parents