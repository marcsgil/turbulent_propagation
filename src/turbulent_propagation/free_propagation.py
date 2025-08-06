from jax import Array
from jax import numpy as jnp
from jax import jit


@jit
def quadratic_phase_kernel(u: Array, qx: Array, qy: Array, alpha: float) -> Array:
    return u * jnp.exp(1j * alpha * (qx**2 + qy**2))


def angular_spectrum_propagation(
    u: Array, dx: float, dy: float, dz: float, wavelength: float, magnification: float
) -> Array:
    Nx = u.shape[-1]
    Ny = u.shape[-2]

    # Match Julia's StepRangeLen behavior more closely
    xs = jnp.linspace(-dx * (Nx // 2), dx * (Nx // 2 - 1), Nx)
    ys = jnp.linspace(-dy * (Ny // 2), dy * (Ny // 2 - 1), Ny)
    xs, ys = jnp.meshgrid(xs, ys, sparse=True)

    # Match Julia's fftfreq scaling: fftfreq(N, 2π/dx) = fftfreq(N, 1/dx) * 2π
    qx = jnp.fft.fftfreq(Nx, d=dx) * 2 * jnp.pi
    qy = jnp.fft.fftfreq(Ny, d=dy) * 2 * jnp.pi
    qx, qy = jnp.meshgrid(qx, qy, sparse=True)

    u = u / magnification
    u = quadratic_phase_kernel(
        u, xs, ys, jnp.pi * (1 - magnification) / wavelength / dz
    )
    u = jnp.fft.fft2(u)
    u = quadratic_phase_kernel(u, qx, qy, -dz * wavelength / 4 / jnp.pi / magnification)
    u = jnp.fft.ifft2(u)
    u = quadratic_phase_kernel(
        u, xs, ys, jnp.pi * (magnification - 1) * magnification / wavelength / dz
    )

    return u
