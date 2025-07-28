import jax.numpy as jnp
from jax import Array, jit
from typing import Callable
from functools import partial


def expected_fourier_correlation_function(
    spectrum: Callable,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    *args,
    **kwargs,
) -> Array:
    qxs = jnp.fft.fftfreq(Nx, d=dx / 2 / jnp.pi)
    qys = jnp.fft.fftfreq(Ny, d=dy / 2 / jnp.pi)
    Qxs, Qys = jnp.meshgrid(qxs, qys, indexing="ij")

    dqx = qxs[1] - qxs[0]
    dqy = qys[1] - qys[0]

    spectrum_value = spectrum(Qxs, Qys, *args, **kwargs) * dqx * dqy

    spectrum_value = spectrum_value.at[0, 0].multiply(0)

    spectrum_value = spectrum_value.at[1, 0].multiply(0.5)
    spectrum_value = spectrum_value.at[-1, 0].multiply(0.5)
    spectrum_value = spectrum_value.at[0, 1].multiply(0.5)
    spectrum_value = spectrum_value.at[0, -1].multiply(0.5)

    """ In the paper, it is said that the corners are multiplied by 3/4,
    but I've found that, to match the expected correlation function,
    they should be multiplied by 1/4.
    This is probably a mistake in the paper, because the method of
    Herman and Strugala uses the same 1/4 factor. """
    spectrum_value = spectrum_value.at[1, 1].multiply(0.25)
    spectrum_value = spectrum_value.at[-1, 1].multiply(0.25)
    spectrum_value = spectrum_value.at[1, -1].multiply(0.25)
    spectrum_value = spectrum_value.at[-1, -1].multiply(0.25)

    return jnp.fft.fft2(spectrum_value)


@partial(jit, static_argnames=("spectrum", "Nx", "Ny", "Np"))
def expected_correlation_function(
    spectrum: Callable,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    Np: int = 3,
    *args,
    **kwargs,
) -> Array:
    """
    Calculate the expected correlation function for a given spectrum following the method
    by Johansson & Gavel (1994) on "Simulation of stellar speckle imaging".
    """
    output = expected_fourier_correlation_function(
        spectrum, Nx, Ny, dx, dy, *args, **kwargs
    )

    xs = jnp.arange(Nx) * dx
    ys = jnp.arange(Ny) * dy
    Xs, Ys = jnp.meshgrid(xs, ys, sparse=True)

    for i in range(-3, 3):
        for j in range(-3, 3):
            if i in (-1, 0) and j in (-1, 0):
                continue
            for p in range(1, Np + 1):
                dqx = 2 * jnp.pi / (Nx * dx * 3**p)
                dqy = 2 * jnp.pi / (Ny * dy * 3**p)
                qx = (i + 0.5) * dqx
                qy = (j + 0.5) * dqy
                spectrum_val = spectrum(qx, qy, *args, **kwargs) * dqx * dqy
                output += spectrum_val * jnp.exp(1j * (qx * Xs + qy * Ys))

    return output.real
