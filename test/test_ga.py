#!/usr/bin/env python3

import unittest
import numpy as np

from swarm import GeneticAlgorithmSolver, DecimalRepresentation


class TestGeneticAlgorithm(unittest.TestCase):

    def test_decimal_representation(self):
        dr = DecimalRepresentation(5, 2)
        number = 123.456
        genes = dr.encode(number)
        number = dr.decode(genes)
        self.assertAlmostEqual(number, 123.46)
        self.assertEqual(genes, [0, 1, 2, 3, 4, 6])

    def test_quadratic(self):
        def fun(x):
            return (x - 1) ** 2

        dr = DecimalRepresentation(5, 1)

        def coded_fun(genes):
            x = dr.decode(genes)
            return fun(x)
        np.random.seed(0)
        genes = np.random.randint(0, 10, [6, 100])
        ga = GeneticAlgorithmSolver(coded_fun, genes, 10)
        last_y = ga.y
        count = 0
        while y := ga.step():
            if y == last_y:
                count += 1
                if count == 50:
                    break
            else:
                count = 0
            last_y = y
        self.assertTrue(np.allclose([dr.decode(ga.x)], [1.0],
                                    atol=0.05))
        self.assertAlmostEqual(ga.y, 0.0, delta=0.01)

    def test_quadratic_2d(self):
        # Two-dimension quadratic
        def fun(x):
            return (x[0] - 2.2) ** 2 + (x[1] - 1.5) ** 2 + 2

        dr = DecimalRepresentation(3, 0, sign=False)

        def coded_fun(genes):
            x0 = dr.decode(genes[:3])
            x1 = dr.decode(genes[3:])
            return fun([x0, x1])

        np.random.seed(0)

        genes = np.random.randint(0, 10, [6, 1000])
        ga = GeneticAlgorithmSolver(coded_fun, genes, 10)
        for i in range(200):
            ga.step()
        self.assertAlmostEqual(dr.decode(ga.x[:3]), 2.2, delta=0.1)
        self.assertAlmostEqual(dr.decode(ga.x[3:]), 1.5, delta=0.1)
        self.assertAlmostEqual(ga.y, 2.0, delta=0.1)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
