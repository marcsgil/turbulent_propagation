from jax import Array
from typing import Callable
from .free_propagation import angular_spectrum_propagation
from .phase_screens import phase_screen
import jax.numpy as jnp
from jax import jit
from functools import partial
from jax import random


@partial(jit, static_argnames=("spectrum", "nsamples", "nsteps"))
def turbulent_propagation(
    u: Array,
    dx: float,
    dy: float,
    dz: float,
    wavelength: float,
    magnification: float,
    spectrum: Callable,
    key: Array,
    nsamples: int = 1,
    nsteps: int = 1,
    *args,
    **kwargs,
) -> Array:
    Nx = u.shape[-1]
    Ny = u.shape[-2]

    m = magnification ** (1 / 2 / nsteps)
    L = dz / nsteps / 2

    keys = random.split(key, nsteps)

    for key in keys:
        u = angular_spectrum_propagation(u, dx, dy, L, wavelength, m)
        phase_screens = phase_screen(
            spectrum,
            Nx,
            Ny,
            dx * m,
            dy * m,
            nsamples=nsamples,
            Np=3,
            key=key,
            *args,
            **kwargs,
        )
        u = u * jnp.exp(1j * phase_screens)
        u = angular_spectrum_propagation(u, dx, dy, L, wavelength, m)

    return u
