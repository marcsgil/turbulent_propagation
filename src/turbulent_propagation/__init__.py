"""
Turbulent Propagation Library

A GPU-accelerated library for simulating wave propagation through the turbulent atmosphere.
Built with JAX for high-performance numerical computing.
"""

# Import main functionality to make it available at package level
from .phase_screens import (
    statistical_structure_function,
    fourier_phase_screen,
    expected_fourier_correlation_function,
    phase_screen,
    expected_correlation_function,
)
from .spectra import von_karman_spectrum

# Define what gets imported with "from turbulent_propagation import *"
__all__ = [
    "statistical_structure_function",
    "fourier_phase_screen",
    "expected_fourier_correlation_function",
    "phase_screen",
    "expected_correlation_function",
    "von_karman_spectrum",
]
