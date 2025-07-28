import jax.numpy as jnp
from jax import Array, random, jit
from typing import Callable
from functools import partial


def statistical_structure_function1d(data: Array) -> Array:
    """
    Computes the statistical structure function for a 1D array.

    Parameters:
    data (Array): Input 1D array.

    Returns:
    Array: The computed statistical structure function.
    """
    N = len(data)
    return (data[: N // 2] - data[0]) ** 2


def statistical_structure_function(data: Array) -> Array:
    """
    Computes the statistical structure function of the input data.

    For 1D arrays: returns (data[0:N//2] - data[0])^2
    For N-D arrays: computes the structure function along the first axis
    and averages over all other dimensions.

    Parameters:
    data (Array): Input data array.

    Returns:
    Array: The computed statistical structure function.
    """
    if data.ndim == 1:
        return statistical_structure_function1d(data)
    else:
        # N-D case: apply structure function along first axis (axis=0)
        result = jnp.apply_along_axis(
            statistical_structure_function1d, axis=-1, arr=data
        )
        # Average over all axes except the structure function axis
        return result.mean(axis=range(0, data.ndim - 1))


def fourier_phase_screen(
    spectrum: Callable,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    nsamples: int = 1,
    key: Array = random.key(42),
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

    spectrum_value = spectrum_value.at[1, 1].multiply(0.25)
    spectrum_value = spectrum_value.at[-1, 1].multiply(0.25)
    spectrum_value = spectrum_value.at[1, -1].multiply(0.25)
    spectrum_value = spectrum_value.at[-1, -1].multiply(0.25)

    random_numbers = random.normal(key, (nsamples, Ny, Nx), dtype=jnp.complex64)

    return jnp.fft.fft2(random_numbers * jnp.sqrt(2 * spectrum_value))


@partial(jit, static_argnames=("spectrum", "Nx", "Ny", "Np", "nsamples"))
def phase_screen(
    spectrum: Callable,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    nsamples: int = 1,
    Np: int = 3,
    key: Array = random.key(42),
    *args,
    **kwargs,
) -> Array:
    key, subkey = random.split(key)
    output = fourier_phase_screen(
        spectrum, Nx, Ny, dx, dy, nsamples, key, *args, **kwargs
    )

    xs = jnp.arange(Nx) * dx
    ys = jnp.arange(Ny) * dy
    Xs, Ys = jnp.meshgrid(xs, ys, sparse=True)

    random_numbers = random.normal(subkey, (nsamples, 32, Np), dtype=output.dtype)
    counter = 0

    for i in range(-3, 3):
        for j in range(-3, 3):
            if i in (-1, 0) and j in (-1, 0):
                continue
            for p in range(Np):
                dqx = 2 * jnp.pi / (Nx * dx * 3 ** (p + 1))
                dqy = 2 * jnp.pi / (Ny * dy * 3 ** (p + 1))
                qx = (i + 0.5) * dqx
                qy = (j + 0.5) * dqy
                spectrum_val = spectrum(qx, qy, *args, **kwargs) * dqx * dqy
                output += (
                    random_numbers[:, counter, p].reshape((nsamples, 1, 1))
                    * jnp.sqrt(spectrum_val)
                    * jnp.exp(1j * (qx * Xs + qy * Ys))
                )
                counter += 1

    return output.real
