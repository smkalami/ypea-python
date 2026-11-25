"""Benchmark functions. Each takes x with shape (..., d) and returns values with shape (...)."""

from .classic import ackley, rastrigin, rosenbrock, sphere

__all__ = ["ackley", "rastrigin", "rosenbrock", "sphere"]
