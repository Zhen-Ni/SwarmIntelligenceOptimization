#!/usr/bin/env python3

import unittest

import numpy as np
from scipy.optimize import rosen

from pso import ParticleSwarm, pso_coefficients

class TestPSO(unittest.TestCase):

    def test_quadratic(self):
        fun = lambda x: (x[0] - 2.2) ** 2 + (x[1] - 1.5) ** 2 + 2
        x0s = np.random.random([2, 10]) * 10 - 5
        pso = ParticleSwarm(fun, x0s)
        while True:
            tol = pso.step()
            if tol < 1e-3:
                break
        self.assertTrue(np.allclose(pso.x, [2.2, 1.5]))
        self.assertAlmostEqual(pso.y, 2.0)


    def test_rosen(self):
        # Eight-dimension rosen function.
        x0s = np.random.random([8, 160]) * 10 - 5
        pso = ParticleSwarm(rosen, x0s, **pso_coefficients(4.1))
        while True:
            tol = pso.step()
            if tol < 1e-6:
                break
        self.assertTrue(np.allclose(pso.x, np.ones(8)))
        self.assertAlmostEqual(pso.y, 0.0)


        
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
    

