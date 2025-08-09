"""
Turbulent Propagation Library

A GPU-accelerated library for simulating wave propagation through the turbulent atmosphere.
Built with JAX for high-performance numerical computing.
"""

# Import main functionality to make it available at package level
from .phase_screens import (
    statistical_structure_function,
    phase_screen,
)
from .spectra import (
    modified_von_karman_spectrum,
    von_karman_spectrum,
    kolmogorov_spectrum,
    hill_andrews_spectrum,
)
from .expected_correlation_functions import (
    expected_correlation_function,
)

from .free_propagation import angular_spectrum_propagation

from .turbulent_propagation import turbulent_propagation

from .utils import atmospheric_coherence_length, rytov_variance

# Define what gets imported with "from turbulent_propagation import *"
__all__ = [
    "statistical_structure_function",
    "phase_screen",
    "expected_correlation_function",
    "modified_von_karman_spectrum",
    "von_karman_spectrum",
    "kolmogorov_spectrum",
    "hill_andrews_spectrum",
    "angular_spectrum_propagation",
    "turbulent_propagation",
    "atmospheric_coherence_length",
    "rytov_variance",
]
