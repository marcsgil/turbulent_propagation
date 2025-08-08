from jax import Array
from typing import Callable
from .free_propagation import angular_spectrum_propagation
from .phase_screens import phase_screen
from math import sqrt
import jax.numpy as jnp


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
    *args,
    **kwargs,
) -> Array:
    Nx = u.shape[-1]
    Ny = u.shape[-2]

    m = sqrt(magnification)

    u = angular_spectrum_propagation(u, dx, dy, dz / 2, wavelength, m)

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

    u = angular_spectrum_propagation(u, dx, dy, dz / 2, wavelength, m)

    return u
