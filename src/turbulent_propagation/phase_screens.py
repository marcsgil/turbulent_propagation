import jax.numpy as jnp
from jax import Array, random, jit
from typing import Callable
from functools import partial


def hermitian_normals(*args, axes=(-2, -1), **kwargs):
    real_noise = random.normal(*args, **kwargs, dtype=jnp.float32)
    return jnp.fft.fftn(real_noise, norm="ortho", axes=axes)


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

    random_numbers = hermitian_normals(key, shape=(nsamples, Ny, Nx))

    return jnp.fft.ifft2(random_numbers * jnp.sqrt(spectrum_value), norm="forward")


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

    random_numbers = random.normal(
        subkey, shape=(nsamples, Np, 6, 3, 1, 1), dtype=jnp.complex64
    )
    random_numbers = jnp.concat(
        (random_numbers, jnp.flip(random_numbers, axis=(2, 3)).conj()), axis=3
    )

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
                    random_numbers[:, p, i, j, :, :]
                    * jnp.sqrt(spectrum_val)
                    * jnp.exp(1j * (qx * Xs + qy * Ys))
                )

    return output.real
