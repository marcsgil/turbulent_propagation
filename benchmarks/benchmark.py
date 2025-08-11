import pyperf

setup = """
from turbulent_propagation import (
    angular_spectrum_propagation,
    phase_screen,
    hill_andrews_spectrum,
)
import jax.numpy as jnp
from jax import random

N = 256
nsamples = 100
u = jnp.zeros((nsamples, N, N), dtype=complex)
key = random.key(42)
"""

stmt1 = """
angular_spectrum_propagation(u, 1, 1, 1, 1, 1).block_until_ready()
"""

stmt2 = """
phase_screen(
    hill_andrews_spectrum, N, N, 1, 1, nsamples=nsamples, key=key, r0=1, L0=1, l0=1
).block_until_ready()
"""

runner = pyperf.Runner()
runner.timeit("angular_spectrum_propagation", stmt1, setup)
runner.timeit("phase_screen", stmt2, setup)
