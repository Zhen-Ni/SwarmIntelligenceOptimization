#!/usr/bin/env python3

"""Genetic algorithm with binary encoding.
"""


from __future__ import annotations
import typing
from random import random, randrange

from .ga import GeneticAlgorithmSolver, Chromosome


__all__ = 'GeneticAlgorithmSolver2', 'BinaryIntervalRepresentation'

P_CROSSOVER = 0.9
P_MUTATION = 0.01


class TargetFunction(typing.Protocol):
    def __call__(self, x: int, *args: typing.Any
                 ) -> float: ...


class BinaryChromosome(Chromosome):
    """Chromosome with gene_size 2.

    BinaryChromosome stores genes and performs crossover and mutation.
    The gene are binary-encoded and can be either 0 or 1.

    Parameters
    ----------
    fun : callable
        The objective funciton to be minimized. The function takes
        a non-negative python int (genes) as input and outputs a
        floating point number.
    genes : int, non-negative
        The initial genes. The chromosome is a bit array represented
        by a positive python int, and its size is given by `length`.
    length : int
        The length of the chromosome.

    See Also
    --------
    Chromosome
        Chromosome with arbitaray gene sizes. BinaryChromosome works
        exactly the same as Chromosome with `gene_size` sets to 2, but
        much more efficient.
        """

    def __init__(self,
                 fun: TargetFunction,
                 genes: int,
                 length: int
                 ):
        self._fun = fun
        self._length = length
        # Use attribute setter for initializing self._genes.
        self._genes: int
        self.genes = genes
        # Cache for storing objective function result.
        self._y: float | None = None

    @property
    def genes(self) -> int:
        return self._genes

    @genes.setter
    def genes(self, genes: int):
        genes = int(genes)
        if genes < 0:
            raise ValueError('gene must be non-negative')
        if not genes < (1 << self._length):
            raise OverflowError('gene length too large')
        self._y = None
        self._genes = genes

    def cross(self, other: BinaryChromosome, p: float) -> None:
        """Crossover, with probability p."""
        if p < random():
            return
        self._y = None
        other._y = None
        n = randrange(self._length)
        mask = (1 << n) - 1
        lhs_tail = self._genes & mask
        rhs_tail = other._genes & mask
        self._genes = self._genes & ~mask | rhs_tail
        other._genes = other._genes & ~mask | lhs_tail

    def mutate(self, p: float) -> None:
        """Mutation, with probability p."""
        p /= self._length
        for i in range(self._length):
            if p < random():
                continue
            self._y = None
            self._genes ^= (1 << i)


class GeneticAlgorithmSolver2(GeneticAlgorithmSolver):
    """Genetic Algorithm with binary encoding.

    The output of given objective function is minimized by finding the
    input (genes) using the genetic algorithm. The input to the
    objective function is the genes, which should be an int value
    carrying `length` bits.

    Parameters
    ----------
    fun : callable
        The objective funciton to be minimized. The function takes an
        int value (which acts like a bit array carrying genes) as
        input and outputs a floating point number.
    genes : list of int, size p
        The initial genes. List of ints, where the length of the
        list (p) is the number of polulation. The genes of each
        individual are represented by a positive python int, and its
        size is given by `length`.
    length : int
        The number of bits stored in the genes.
    p_crossover : float, optional
        Crossover probability. Defaults to 0.9.
    p_mutation : float, optimal
        Mutation probability. Defaults to 0.01.
        """

    def __init__(self,
                 fun: TargetFunction,
                 genes: typing.Sequence[int],
                 length: int,
                 *,
                 p_crossover: float = P_CROSSOVER,
                 p_mutation: float = P_MUTATION,
                 ):
        self._chromosomes = []
        for i in range(len(genes)):
            chromosome = BinaryChromosome(fun, genes[i], length)
            self._chromosomes.append(chromosome)

        self._p_crossover = p_crossover
        self._p_mutation = p_mutation

        self._polulation_best = self._get_population_best()


class BinaryIntervalRepresentation:
    """Encode or decode a floating point number defined in an interval.

    The number is represented as:
    number = lb + k / (2 ** size) * (ub - lb)
    where lb and ub are the lower and upper bounds, size is the number
    of bits used for the storage and k is the number represented by
    the genes using Gray code.

    Parameters
    ----------
    lb : float
        The lower bound for the floating point number (inclusive).
    ub : float
        The upper bound for the floating point number (exclusive).
    size : int
        The number of bits used.
    """

    def __init__(self, lb: float, ub: float, size: int):
        self._lb = lb
        self._interval = ub - lb
        self._resolution = 1 << size

    def decode(self, genes: int) -> float:
        k = gray_to_binary(genes)
        if genes < 0:
            raise ValueError('genes should be non-negative int')
        if not k < self._resolution:
            raise OverflowError('genes too long for decoding')
        return self._lb + k / self._resolution * self._interval

    def encode(self, number: float) -> int:
        percentile = (number - self._lb) / self._interval
        if percentile < 0:
            raise ValueError('number for encoding should be non-negative')
        if not percentile < 1:
            raise OverflowError('number too large for encoding')
        binary = round(percentile * self._resolution)
        return binary_to_gray(binary)


def binary_to_gray(num: int) -> int:
    return num ^ (num >> 1)


def gray_to_binary(num: int) -> int:
    mask = num
    while mask:
        mask >>= 1
        num ^= mask
    return num
