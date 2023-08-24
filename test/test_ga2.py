#!/usr/bin/env python3

import unittest
import numpy as np
from scipy.optimize import rosen

from swarm import GeneticAlgorithmSolver2, BinaryIntervalRepresentation


class TestGeneticAlgorithm(unittest.TestCase):

    def test_binary_interval_representation(self):
        bir = BinaryIntervalRepresentation(0., 100., 8)
        number = 12.3456
        genes = bir.encode(number)
        number = bir.decode(genes)
        self.assertAlmostEqual(number, 12.346, delta=100/2**8)

    def test_quadratic(self):
        def fun(x):
            return (x - 1) ** 2

        bir = BinaryIntervalRepresentation(0, 2, 4)

        def coded_fun(genes):
            x = bir.decode(genes)
            return fun(x)
        np.random.seed(0)
        genes = np.random.randint(0, 2**4, [50])
        ga = GeneticAlgorithmSolver2(coded_fun, genes, 4)
        for i in range(200):
            tol = ga.step()
            if tol < 1e-3:
                break
        self.assertTrue(np.allclose([bir.decode(ga.x)], [1.0],
                                    atol=0.1))
        self.assertAlmostEqual(ga.y, 0.0, delta=0.05)

    def test_quadratic_2d(self):
        # Two-dimension quadratic
        def fun(x):
            return (x[0] - 2.2) ** 2 + (x[1] - 1.5) ** 2 + 2

        bir = BinaryIntervalRepresentation(0, 5, 6)

        def coded_fun(genes):
            x0 = bir.decode(genes & 0b000000111111)
            x1 = bir.decode(genes >> 6)
            return fun([x0, x1])

        np.random.seed(0)
        genes = np.random.randint(0, 1 << 12, [400])
        ga = GeneticAlgorithmSolver2(coded_fun, genes, 12)
        for i in range(100):
            ga.step()
        self.assertAlmostEqual(bir.decode(ga.x & 0b000000111111),
                               2.2, delta=0.1)
        self.assertAlmostEqual(bir.decode(ga.x >> 6),
                               1.5, delta=0.1)
        self.assertAlmostEqual(ga.y, 2.0, delta=0.01)

    def test_rosen(self):
        # Three-dimensional rosen function.
        dim = 3
        nbits = 4
        np.random.seed(0)
        bir = BinaryIntervalRepresentation(0., 2., nbits)
        genes = np.random.randint(0, 1 << (dim*nbits), [400])

        def decode(x):
            xs = []
            for i in range(dim):
                xi = bir.decode((x >> (i * nbits)) & ((1 << nbits) - 1))
                xs.append(xi)
            return xs

        def coded_rosen(x):
            xs = decode(x)
            return rosen(xs)
        ga = GeneticAlgorithmSolver2(
            coded_rosen, genes, nbits*dim, p_mutation=0.1)
        for i in range(100):
            ga.step()

        xs = decode(ga.x)
        self.assertTrue(np.allclose(xs, [1, 1, 1], atol=0.1))
        self.assertAlmostEqual(ga.y, 0.0, delta=0.1)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
    from scipy.optimize import fmin
