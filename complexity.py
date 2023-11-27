import numpy as np
from math import comb


"""
This document contains three classes to compute the complexity of Additive Vector Quantization (AVQ),
Residual Vector Quantization (RVQ), and Product Vector Quantization (PVQ) methods as weighted million operations
per second (WMOPS) based on ITU-T software tool library (user's manual 2009). For all methods, we break down the
vector quantization into four basic operations of addition, multiplication, move (assignment or copy), and branching
(conditional statement). We use WMOPS from ITU-T standard to calculate complexity values in a more realistic way, 
since WMOPS shows the approximation of typical complexity of operations when performed on cpu.
Here is the link for ITU-T software tool library: https://www.itu.int/rec/T-REC-G.191-200911-S/en
"""


class AVQ_Complexity():
    def __init__(self, input_dim, num_codebooks, num_stages, beam_width, single_precision=True):

        self.dim = input_dim # data samples or codebook entries dimension
        self.num_codebooks = num_codebooks # number of codebook entries (embeddings) per stage
        self.num_stages = num_stages # number of stages used for vector quantization
        self.beam_width = beam_width # beam width for beam searching in additive VQ

        if num_codebooks < beam_width:
            raise ValueError("beam width cannot be greater than the number of codebook entries!")

        self.num_additions = 0 # counter for addition operation
        self.num_multips = 0 # counter for multiplication operation
        self.num_moves = 0 # counter for move (assignment or copy) operation
        self.num_branches = 0 # counter for branch (condition) operation

        # defining the complexity weights for each operation based on ITU-T software tool library
        if single_precision:
            self.addition_weight = 1
            self.multiplication_weight = 1
            self.move_weight = 1
            self.branching_weight = 4
        else: # complexity weights for double precision
            self.addition_weight = 2
            self.multiplication_weight = 2
            self.move_weight = 2
            self.branching_weight = 4

    # call this function to calculate the complexity
    def complexity_calculator(self):
        self.distance_calculation(self.num_stages * self.num_codebooks, self.dim, 1)
        self.sort_calculation(self.num_stages * self.num_codebooks, self.beam_width, 1)
        self.num_moves += self.beam_width

        for i in np.arange(1,self.num_stages):
            self.num_moves += i * (self.beam_width * self.dim) # quantization
            self.num_additions += i * self.beam_width * self.dim # remaidner calculation
            self.num_branches += comb(self.num_stages, i) - 1
            self.distance_calculation((self.num_stages-i) * self.num_codebooks, self.dim, self.beam_width)
            self.sort_calculation((self.num_stages-i) * self.num_codebooks, self.beam_width, self.beam_width)
            self.num_moves += self.beam_width ** 2 # storing bw**2 tuples
            self.num_moves += (self.beam_width ** 2) * self.dim # quantize in 11th
            self.num_additions += (self.beam_width**2) * (i * self.dim) # final quantized input
            self.mse_calculation(self.beam_width, self.dim)
            if i == self.num_stages - 1:
                self.argmin_calculation(self.beam_width ** 2)
            else:
                self.sort_calculation(self.beam_width ** 2, self.beam_width, 1)
                self.num_moves += self.beam_width  # storing final bw tuples

        # divided by 1e6 to return the complexity in millions
        total_complexity = ((self.num_additions * self.addition_weight) + (self.num_multips * self.multiplication_weight)
                            + (self.num_moves * self.move_weight) + (self.num_branches * self.branching_weight)) / 1e6

        return total_complexity


    # calculates the operations needed for distance calculation
    def distance_calculation(self, num_cbs, dim, num_times):
        self.num_additions += num_times * (num_cbs * ((2 * dim) - 1))
        self.num_multips += num_times * (num_cbs * dim)


    # calculates the operations needed for mean squared error (MSE) calculation
    def mse_calculation(self, beam_width, dim):
        self.num_additions += (beam_width**2) * ((2*dim) - 1)
        self.num_multips += (beam_width**2) * (dim + 1)


    # calculates the operations needed for argmin function
    def argmin_calculation(self, num_indices):
        self.num_additions += num_indices-1
        self.num_moves += num_indices-1


    # calculates the operations needed for sort function
    def sort_calculation(self, num_cbs, num_selections, num_times):
        for i in range(num_times):
            for j in range(num_selections):
                self.argmin_calculation(num_cbs-j)



class RVQ_Complexity():
    def __init__(self, input_dim, num_codebooks, num_stages, single_precision=True):

        self.dim = input_dim # data samples or codebook entries dimension
        self.num_codebooks = num_codebooks # number of codebook entries (embeddings) per stage
        self.num_stages = num_stages # number of stages used for vector quantization

        self.num_additions = 0 # counter for addition operation
        self.num_multips = 0 # counter for multiplication operation
        self.num_moves = 0 # counter for move (assignment or copy) operation
        self.num_branches = 0 # counter for branch (condition) operation

        # defining the complexity weights for each operation based on ITU-T software tool library
        if single_precision:
            self.addition_weight = 1
            self.multiplication_weight = 1
            self.move_weight = 1
            self.branching_weight = 4
        else: # complexity weights for double precision
            self.addition_weight = 2
            self.multiplication_weight = 2
            self.move_weight = 2
            self.branching_weight = 4

    # call this function to calculate the complexity
    def complexity_calculator(self):

        for i in np.arange(1,self.num_stages):
            self.distance_calculation(self.num_codebooks, self.dim, 1)
            self.argmin_calculation(self.num_codebooks)
            self.num_moves += self.dim
            if i == self.num_stages - 1:
                pass
            else:
                self.num_additions += self.dim # remainder calculation

        self.num_additions += (self.num_stages - 1) * self.dim # final quantized input

        # divided by 1e6 to return the complexity in millions
        total_complexity = ((self.num_additions * self.addition_weight) + (self.num_multips * self.multiplication_weight)
                            + (self.num_moves * self.move_weight) + (self.num_branches * self.branching_weight)) / 1e6

        return total_complexity


    # calculates the operations needed for distance calculation
    def distance_calculation(self, num_cbs, dim, num_times):
        self.num_additions += num_times * (num_cbs * ((2 * dim) - 1))
        self.num_multips += num_times * (num_cbs * dim)


    # calculates the operations needed for argmin function
    def argmin_calculation(self, num_indices):
        self.num_additions += num_indices-1
        self.num_moves += num_indices-1



class PVQ_Complexity():
    def __init__(self, input_dim, num_codebooks, num_stages, single_precision=True):

        self.dim = input_dim # data samples or codebook entries dimension
        self.num_codebooks = num_codebooks # number of codebook entries (embeddings) per stage
        self.num_stages = num_stages # number of stages used for vector quantization

        self.num_additions = 0 # counter for addition operation
        self.num_multips = 0 # counter for multiplication operation
        self.num_moves = 0 # counter for move (assignment or copy) operation
        self.num_branches = 0 # counter for branch (condition) operation

        # defining the complexity weights for each operation based on ITU-T software tool library
        if single_precision:
            self.addition_weight = 1
            self.multiplication_weight = 1
            self.move_weight = 1
            self.branching_weight = 4
        else: # complexity weights for double precision
            self.addition_weight = 2
            self.multiplication_weight = 2
            self.move_weight = 2
            self.branching_weight = 4

    # call this function to calculate the complexity
    def complexity_calculator(self):

        self.num_moves += self.dim

        for i in np.arange(1,self.num_stages):
            self.distance_calculation(self.num_codebooks, self.dim / self.num_stages, 1)
            self.argmin_calculation(self.num_codebooks)
            self.num_moves += self.dim / self.num_stages

        self.num_moves += self.dim

        # divided by 1e6 to return the complexity in millions
        total_complexity = ((self.num_additions * self.addition_weight) + (self.num_multips * self.multiplication_weight)
                            + (self.num_moves * self.move_weight) + (self.num_branches * self.branching_weight)) / 1e6

        return total_complexity


    # calculates the operations needed for distance calculation
    def distance_calculation(self, num_cbs, dim, num_times):
        self.num_additions += num_times * (num_cbs * ((2 * dim) - 1))
        self.num_multips += num_times * (num_cbs * dim)


    # calculates the operations needed for argmin function
    def argmin_calculation(self, num_indices):
        self.num_additions += num_indices-1
        self.num_moves += num_indices-1