#!/usr/bin/env python3

"""Genetic algorithm.
"""


from __future__ import annotations
import typing

from random import random, randrange, choices
import copy
import numpy.typing as npt


__all__ = 'GeneticAlgorithmSolver', 'DecimalRepresentation'

P_CROSSOVER = 0.9
P_MUTATION = 0.01


class TargetFunction(typing.Protocol):
    def __call__(self, x: typing.Sequence[int], *args: typing.Any
                 ) -> float: ...


class Chromosome:
    """Chromosome stores genes and performs crossover and mutation.

    Parameters
    ----------
    fun : callable
        The objective funciton to be minimized. The function takes
        a sequence of ints (genes) as input and outputs a float
        number.
    genes : sequence of ints
        The initial genes. It contains a sequence of ints from
        0 (inclusice) to `gene_size` (exclusize).
    gene_size : int
        The number of choices of each gene. should be larger than 1.

        """

    def __init__(self,
                 fun: TargetFunction,
                 genes: typing.Sequence[int],
                 gene_size: int):
        self._fun = fun
        self.genes = list(genes)
        self._gene_size = gene_size
        if gene_size < 2:
            raise ValueError('gene_size should be larger than 1')
        for v in self.genes:
            if not v < self._gene_size:
                raise OverflowError('gene value larger than gene size')
        self._y: float | None = None

    def cross(self, other: Chromosome, p: float):
        """Crossover, with probability p."""
        if p < random():
            return
        self._y = None
        n = randrange(len(self.genes))
        self.genes[n:], other.genes[n:] = \
            other.genes[n:], self.genes[n:]

    def mutate(self, p: float):
        """Mutation, with probability p."""
        p /= len(self.genes)
        # Correct the probability if a gene mutates to itself.
        p *= self._gene_size / (self._gene_size - 1)
        for i in range(len(self.genes)):
            if p < random():
                continue
            self._y = None
            self.genes[i] = randrange(self._gene_size)

    def evaluate(self) -> float:
        """Get the result of the objective function.

        The result is cached, and lazy-evaluated after crossover and
        mutation operations.
        """
        if self._y is None:
            self._y = self._fun(self.genes)
        return self._y


class GeneticAlgorithmSolver:
    """Genetic Algorithm for general purpose optimization.

    The output of given objective function is minimized by finding
    the input (genes) using the genetic algorithm. The input to the
    objective should be encoded by a list of integers.

    Parameters
    ----------
    fun : callable
        The objective funciton to be minimized. The function takes
        a sequence of ints (genes) as input and outputs a float
        number.
    genes : ndarray, shape (n, p)
        The initial genes. Array of ints of size (n, p), where n is
        the length of each gene and p is the size of polulation. The
        ints should be in [0, gene_size).
    gene_size : int
        The number of choices of each gene.
    p_crossover : float, optional
        Crossover probability. Defaults to 0.9.
    p_mutation : float, optimal
        Mutation probability. Defaults to 0.01.
    """

    def __init__(self,
                 fun: TargetFunction,
                 genes: npt.NDArray,
                 gene_size: int,
                 *,
                 p_crossover: float = P_CROSSOVER,
                 p_mutation: float = P_MUTATION,
                 ):
        self._chromosomes = []
        for i in range(genes.shape[1]):
            chromosome = Chromosome(fun, genes[:, i], gene_size)
            self._chromosomes.append(chromosome)

        self._p_crossover = p_crossover
        self._p_mutation = p_mutation

        self._polulation_best = self._get_population_best()

    def _select(self, size) -> list[Chromosome]:
        """Randomly select `size` chromosomes by their fitness."""
        ys = [y.evaluate() for y in self._chromosomes]
        ymax = max(ys)
        fitness = [ymax - i for i in ys]
        if sum(fitness) == 0.0:
            fitness = [1.] * len(fitness)
        return choices(self._chromosomes, fitness, k=size)

    def _cross(self):
        """Crossover and update self._chromosomes."""
        size = len(self._chromosomes)
        npairs = size // 2 + 1
        parents = [copy.deepcopy(i) for i in self._select(npairs * 2)]
        for i in range(npairs):
            parents[i].cross(parents[npairs+i], self._p_crossover)
            # The second call performs a possible 2-point crossover.
            parents[i].cross(parents[npairs+i], self._p_crossover)
        self._chromosomes = parents[:size]

    def _mutate(self):
        """Mutate and update self._chromosomes."""
        for c in self._chromosomes:
            c.mutate(self._p_mutation)

    def _get_population_best(self):
        return min(self._chromosomes,
                   key=lambda c: c.evaluate())

    def step(self) -> float:
        """Evolution to the next generation.

        The best result of the objective funciton is returned after
        this evolution.
        """
        self._cross()
        self._mutate()
        self._polulation_best = self._get_population_best()
        return self.y

    @property
    def x(self):
        """The optimal gene."""
        return self._polulation_best.genes

    @property
    def y(self):
        """The optimal output of objective function."""
        return self._polulation_best.evaluate()


class DecimalRepresentation:
    """Encode or decode a number into decimal representation.

    The number is represented as:
    number = (+-)a[0].a[1]a[2]...a[size-1] * 10 ** exponent
    where a is the output genes.

    Parameters
    ----------
    size : int
        The number of digits used.
    exponent : int
        The exponent.
    sign : bool
        Whether to add an additional slot for representing the
        sign of the number. This is put at the beginning of the
        genes, odd digit stands for negative and even digit stands
        for positive number.
    """

    def __init__(self, size: int, exponent: int, sign: bool = True):
        self._size = size
        self._exponent = exponent
        self._sign = bool(sign)

    def decode(self, genes: typing.Sequence[int]) -> float:
        start_idx = 0
        coeff = 1
        if self._sign:
            start_idx = 1
            coeff = -1 if genes[0] % 2 else 1
        number = 0.
        for i, v in enumerate(genes[start_idx:]):
            number += v
            number *= 10
        if i + 1 != self._size:
            raise DecimalRepresentationError(
                'length of genes should be {}, got {} instead'.format(
                    self._size + self._sign, i + self._sign + 1))
        return coeff * number * 10 ** (self._exponent - self._size)

    def encode(self, number: float) -> typing.Sequence[int]:
        res = [int(number < 0.)] if self._sign else []
        number = number / 10 ** (self._exponent - self._size)
        number = abs(int(round(number, -1)))
        divider = 10 ** self._size
        if not number < divider * 10:
            raise DecimalRepresentationError(
                "number too large to ""encode")
        for i in range(self._size):
            res.append(number // divider)
            number = number % divider
            divider //= 10
        return res


class DecimalRepresentationError(Exception):
    pass
