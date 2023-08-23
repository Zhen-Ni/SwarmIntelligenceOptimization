#!/usr/bin/env python3

import unittest

import numpy as np
from scipy.optimize import rosen

from swarm import ParticleSwarmSolver, pso_coefficients


class TestPSO(unittest.TestCase):

    def test_quadratic(self):
        def fun(x): return (x[0] - 2.2) ** 2 + (x[1] - 1.5) ** 2 + 2
        np.random.seed(0)
        x0s = np.random.random([2, 10]) * 10 - 5
        pso = ParticleSwarmSolver(fun, x0s)
        while True:
            tol = pso.step()
            if tol < 1e-3:
                break
        self.assertTrue(np.allclose(pso.x, [2.2, 1.5], atol=0.1))
        self.assertAlmostEqual(pso.y, 2.0, delta=0.01)

    def test_rosen(self):
        # Eight-dimension rosen function.
        np.random.seed(0)
        x0s = np.random.random([8, 160]) * 2
        pso = ParticleSwarmSolver(rosen, x0s, **pso_coefficients(4.1))
        while True:
            tol = pso.step()
            if tol < 1e-6:
                break
        self.assertTrue(np.allclose(pso.x, np.ones(8)))
        self.assertAlmostEqual(pso.y, 0.0)

    def test_constraints(self):
        # Eight-dimension rosen function.
        np.random.seed(0)
        x0s = np.random.random([2, 10]) * 2 + 1.
        def func(x): return x[0] + x[1] ** 2
        constraints = (lambda x: - x[0] + 1,
                       lambda x: - x[1] + 1.)
        pso = ParticleSwarmSolver(func, x0s, constraints=constraints)
        while True:
            tol = pso.step()
            if tol < 1e-6:
                break
        self.assertTrue(np.allclose(pso.x, np.ones(2)))
        self.assertAlmostEqual(pso.y, 2, delta=0.01)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
