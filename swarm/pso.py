#!/usr/bin/env python3

"""Particle swarm optimization.
"""


from __future__ import annotations
import typing

from random import random
import numpy as np
import numpy.typing as npt

__all__ = ('ParticleSwarmSolver', 'pso_coefficients')

DEFAULT_W = 0.5
DEFAULT_C1 = 2.
DEFAULT_C2 = 2.


class PSOInitError(ValueError):
    pass


class TargetFunction(typing.Protocol):
    def __call__(self, x: npt.NDArray, *args: typing.Any) -> float: ...


class Particle:
    """A paricle, or sometimes called candidate solution.

    The paricle searches in the domain of the input value. Assuming
    its current position is x and its speed is v, after each
    iteration, it becomes:

    v_new = w * v + c1 * (x_opt - x) + c2 * (p_opt - x)
    x_new = x + v_new

    where w denotes inertia weight, c1 and c2 are also coefficients
    denoting cognitive coefficient and social coefficient, x_opt is
    the particle's best known position, p_opt is the swarm's best
    known position in the current step.

    For optimization problems with constraints, the positions of the
    particles are evaluated by the constraint functions. The
    constraint functions are inequality functions with general form:

    constraint_func(x) <= 0

    To avoid any effect on the performance of the algorithm, the
    velocity and infeasible position of the particles outside the
    constrainted domain are left unaltered. The evaluation of the
    objective function is skipped, thereby preventing the infeasible
    position from being set as a personal and/or global best.  Using
    this method, particles outside the feasible search space will
    eventually be drawn back within the space by the influence of
    their personal and neighborhood bests. [1]

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is a 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    x0 : ndarray, shape (n,)
        Initial position of the particle. Array of real elements
        of size (n,), where ``n`` is the number of independent
        variables (dimension of the input variable of fun).
    v0 : ndarray, shape (n,), optional
        Initial speed of the particle. Defaults to all zeros.
    constraints : list of callable, optional
        Constraints definition. The constraint have general
        inequality form:
            ``fun(x) <= 0``
        Here the vector of independent variables x is passed as
        ndarray of shape (n,) and fun returns a float.
    w : float, optional
        Inertia weight. Defaults to 0.5.
    c1 : float, optional
        Cognitive coefficient. Defaults to 2.0.
    c2 : float, optional
        Social coefficient. Defaults to 2.0.
    v_limit : float, optional
        The max value of particle. If a non-None value is set,
        Particle speed will be normalized to make sure its L-∞
        norm does not exceed v_limit. Defaults to None.

    Reference
    ---------
    [1] BRATTON D, KENNEDY J. Defining a Standard for Particle Swarm
        Optimization[C/OL]//2007 IEEE Swarm Intelligence
        Symposium. Honolulu, HI, USA: IEEE, 2007: 120-127[2023-08-16].
        """

    def __init__(self,
                 fun: TargetFunction,
                 x0: npt.NDArray,
                 v0: npt.NDArray | None = None,
                 *,
                 constraints: tuple[TargetFunction, ...] = (),
                 w: float | None = None,
                 c1: float | None = None,
                 c2: float | None = None,
                 v_limit: float | None = None):
        # Objective function
        self._fun = fun
        self._constraints = constraints

        # Position and speed of the particle
        self._x = x0
        self._v = np.zeros_like(x0) if v0 is None else v0

        # Value of this particle
        self._y = self._objective_function(self._x)

        # Weights
        self._w = DEFAULT_W if w is None else w
        self._c1 = DEFAULT_C1 if c1 is None else c1
        self._c2 = DEFAULT_C2 if c2 is None else c2

        # Max speed of the paricle
        self._v_limit = v_limit
        self._v_norm = self._get_norm(self._v)

        # Best known position and value of this particle.
        self._x_opt = self._x
        self._y_opt = self._y

    @property
    def x(self) -> npt.NDArray:
        return self._x

    @property
    def y(self) -> float | None:
        return self._y

    @property
    def w(self) -> float:
        return self._w

    @w.setter
    def w(self, w: float):
        self._w = w

    @property
    def c1(self) -> float:
        return self._c1

    @c1.setter
    def c1(self, c1: float):
        self._c1 = c1

    @property
    def c2(self) -> float:
        return self._c2

    @c2.setter
    def c2(self, c2: float):
        self._c2 = c2

    @property
    def v_limit(self) -> float | None:
        return self._v_limit

    @v_limit.setter
    def v_limit(self, v_limit: float | None):
        self._v_limit = v_limit

    @property
    def x_opt(self) -> npt.NDArray:
        return self._x_opt

    @property
    def y_opt(self) -> float | None:
        return self._y_opt

    @property
    def v_norm(self) -> float:
        return self._v_norm

    def _get_norm(self, v) -> float:
        return np.abs(v).max()

    def update(self, p_opt: npt.NDArray):
        """Update particle's position and velocity.

        Parameters
        ----------
        p_opt: ndarray, shape (n,)
            The swarm's best known position.
        """
        # Update velocity of the particle.
        v = self._w * self._v
        v += self._c1 * random() * (self._x_opt - self._x)
        v += self._c2 * random() * (p_opt - self._x)

        # Normalize
        self._v_norm = self._get_norm(v)
        if self._v_limit is not None:
            if self._v_norm > self._v_limit:
                v *= (self._v_limit / self._v_norm)
                self._v_norm = self._v_limit

        # Updata particle
        self._x = self._x + v
        self._v = v
        self._y = self._objective_function(self._x)
        if self._y is not None:
            if (self._y_opt is None) or (self._y < self._y_opt):
                self._x_opt = self._x
                self._y_opt = self._y

    def _objective_function(self, x: npt.NDArray) -> float | None:
        """Get the result of the objective function.

        The constraints are also checked here. If the constraints are
        not satisfied, this funciton returns None.
        """
        for c in self._constraints:
            if c(x) > 0:
                return None
        return self._fun(x)


class ParticleSwarmSolver:
    """Particle swarm optimization algorithm.

    It works by having a population (called a swarm) of candidate
    solutions (called particles). These particles are moved around the
    search space guided by their own best known position and the
    swarm's best known position. When improved positions are being
    discovered these will then come to guide the movements of the
    swarm. The process is repeated and by doing so it is hoped, but
    not guaranteed, that a satisfactory solution will eventually be
    discovered. [1]

    For each particle, assume its current position is x and its speed
    is v. It updates by the following equations:

    v_new = w * v + c1 * (x_opt - x) + c2 * (p_opt - x)
    x_new = x + v_new

    where w denotes inertia weight, c1 and c2 are also coefficients
    denoting cognitive coefficient and social coefficient, x_opt is
    the particle's best known position, p_opt is the swarm's best
    known position in the current step.

    The coefficients w, c1 and c2 cam be determined by a single
    parameters phi by calling `pso_coefficients(phi)`, which returns a
    dict of w, c1, c2 with their corresponding values. Thus, it is
    convenient to instantiation ParticleSwarmSolver by invoking:
    `ParticleSwarmSolver(fun, x0s, v0s, v_limit=v_limit,
    **pso_coefficients(phi))`. To ensure convergence, phi should be
    larger than 4.

    For optimization problems with constraints, the positions of the
    particles are evaluated by the constraint functions. The
    constraint functions are inequality functions with general form:
    constraint_func(x) <= 0.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is a 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.

    x0s : ndarray, shape (n, s)
        The initial positions of the particles. Array of real elements
        of size (n, s), where ``n`` is the number of independent
        variables (dimension of the input variable of fun), and ``s``
        is the number of particles.
    v0s : ndarray, shape (n, s), optional
        Initial speed of the particle. Defaults to all zeros.    
    constraints : list of callable, optional
        Constraints definition. The constraint have general
        inequality form:
            ``fun(x) <= 0``
        Here the vector of independent variables x is passed as
        ndarray of shape (n,) and fun returns a float.
    w : float, optional
        Inertia weight. Defaults to 0.5.
    c1 : float, optional
        Cognitive coefficient. Defaults to 2.0.
    c2 : float, optional
        Social coefficient. Defaults to 2.0.
    v_limit : float, optional
        The max value of particle. If a non-None value is set,
        Particle speed will be normalized to make sure its L-∞
        norm does not exceed v_limit. Defaults to None.

    See alse
    --------
    pso_coefficients
        Determing w, c1 and c2 by one single argument.

    Reference
    ---------
    [1] https://en.wikipedia.org/wiki/Particle_swarm_optimization

    """

    def __init__(self,
                 fun: TargetFunction,
                 x0s: npt.NDArray,
                 v0s: npt.NDArray | None = None,
                 *,
                 constraints: tuple[TargetFunction, ...] = (),
                 w: float | None = None,
                 c1: float | None = None,
                 c2: float | None = None,
                 v_limit: float | None = None):
        self._particles = []
        has_valid_particle = False
        for i in range(x0s.shape[1]):
            x0 = x0s[:, i]
            v0 = v0s[:, i] if v0s is not None else None
            p = Particle(fun, x0, v0, constraints=constraints, w=w,
                         c1=c1, c2=c2, v_limit=v_limit)
            self._particles.append(p)
            if p.y_opt is not None:
                has_valid_particle = True
        # Raise PSOInitError if None of the initial particles meet the
        # constraints.
        if not has_valid_particle:
            raise PSOInitError('no valid initial particle found')

        self._swarm_best = self._get_swarm_best()
        self._global_best: Particle | None = None

    def set_w(self, w: float):
        for p in self._particles:
            p.w = w

    def set_c1(self, c1: float):
        for p in self._particles:
            p.c1 = c1

    def set_c2(self, c2: float):
        for p in self._particles:
            p.c2 = c2

    def set_v_limit(self, v_limit: float):
        for p in self._particles:
            p.v_limit = v_limit

    def __getitem__(self,
                    index: int | slice
                    ) -> Particle | list[Particle]:
        return self._particles[index]

    def _get_swarm_best(self) -> Particle:
        """Get the best particle in the current iteration."""
        return min(self._particles,
                   key=lambda p: p.y if p.y is not None else np.inf)

    def _get_global_best(self) -> Particle:
        """Get the particle with best value whthin the whole searching
        procedure."""
        return min(self._particles,
                   key=lambda p: p.y_opt if p.y_opt is not None else np.inf)

    def step(self):
        """Update all particles.

        The maximum speed of the particles is returned. This can
        approximately evaluating the error and used as torlerance
        for termination. Note that this is not a good criterion
        for some functions such as Rosenbrock function.
        """
        self._global_best = None
        p_opt = self._swarm_best.x
        for p in self._particles:
            p.update(p_opt)
        self._swarm_best = self._get_swarm_best()
        return max(p.v_norm for p in self._particles)

    @property
    def x(self) -> npt.NDArray:
        """The estimated optimal solution."""
        self._global_best = self._global_best or self._get_global_best()
        return self._global_best.x_opt

    @property
    def y(self) -> float:
        """Value of the objective function corresponding to the
        estimated optimal solution."""
        self._global_best = self._global_best or self._get_global_best()
        # Global best must not be None after a success initialization.
        assert self._global_best.y_opt is not None
        return self._global_best.y_opt


def _get_Χ_from_φ(φ: float) -> float:
    """Get w, c1 and c2 from another set of parameters.

    To ensure convergence, φ should be larger than 4.

    Reference
    ---------
    [1] BRATTON D, KENNEDY J. Defining a Standard for Particle Swarm
        Optimization[C/OL]//2007 IEEE Swarm Intelligence
        Symposium. Honolulu, HI, USA: IEEE, 2007: 120-127[2023-08-16].
    """
    return 2 / abs(2 - φ - ((φ - 4) * φ) ** .5)


def pso_coefficients(phi) -> dict[str, float]:
    """Get w, c1 and c2 for ParticleSwarmSolver from a single argument.

    To ensure convergence, phi should be larger than 4. Larger phi
    makes it better for local search while smaller phi makes the
    algorithm better for global search.

    Reference
    ---------
    [1] BRATTON D, KENNEDY J. Defining a Standard for Particle Swarm
        Optimization[C/OL]//2007 IEEE Swarm Intelligence
        Symposium. Honolulu, HI, USA: IEEE, 2007: 120-127[2023-08-16].
    """
    x = _get_Χ_from_φ(phi)
    w = x
    c = x * phi / 2
    return dict(w=w, c1=c, c2=c)
