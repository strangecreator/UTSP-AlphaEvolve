"""
OpenEvolve: An open-source implementation of AlphaEvolve
"""

from openevolve._version import __version__
from openevolve.controller import OpenEvolve
from openevolve.api import (
    run_evolution,
    evolve_function,
    evolve_algorithm,
    evolve_code,
    EvolutionResult
)

__all__ = [
    "OpenEvolve", 
    "__version__",
    "run_evolution",
    "evolve_function", 
    "evolve_algorithm",
    "evolve_code",
    "EvolutionResult"
]
