from jax import Array
from jax import numpy as jnp
from jax import jit


@jit
def quadratic_phase_kernel(u: Array, qx: Array, qy: Array, alpha: float) -> Array:
    return u * jnp.exp(1j * alpha * (qx**2 + qy**2))


def angular_spectrum_propagation(
    u: Array, dx: float, dy: float, dz: float, wavelength: float, magnifictation: float
) -> Array:
    Nx = u.shape[-1]
    Ny = u.shape[-2]
    xs = jnp.arange(-Nx // 2, Nx // 2) * dx
    ys = jnp.arange(-Ny // 2, Ny // 2) * dy
    xs, ys = jnp.meshgrid(xs, ys, sparse=True)

    qx = jnp.fft.fftfreq(Nx, d=dx)
    qy = jnp.fft.fftfreq(Ny, d=dy)
    qx, qy = jnp.meshgrid(qx, qy, sparse=True)

    u /= magnifictation
    u = quadratic_phase_kernel(
        u, xs, ys, jnp.pi * (1 - magnifictation) / wavelength / dz
    )
    u = jnp.fft.fft2(u)
    u = quadratic_phase_kernel(
        u, qx, qy, -dz * wavelength / 4 / jnp.pi / magnifictation
    )
    u = jnp.fft.ifft2(u)
    u = quadratic_phase_kernel(
        u, xs, ys, jnp.pi * (magnifictation - 1) * magnifictation / wavelength / dz
    )

    return u
