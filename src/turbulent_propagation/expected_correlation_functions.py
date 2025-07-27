import jax.numpy as jnp
from jax import Array
from typing import Callable


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

    # Set origin to zero to avoid pole issues (as mentioned in the paper)
    spectrum_value = spectrum_value.at[0, 0].set(0.0)

    # Apply overlap compensation for hybrid method
    # Scale points that will overlap with subharmonic contributions
    # Points at indices (±1, 0) and (0, ±1) are scaled by 1/2
    # Points at indices (±1, ±1) are scaled by 1/4
    spectrum_value = spectrum_value.at[1, 0].multiply(0.5)
    spectrum_value = spectrum_value.at[-1, 0].multiply(0.5)
    spectrum_value = spectrum_value.at[0, 1].multiply(0.5)
    spectrum_value = spectrum_value.at[0, -1].multiply(0.5)
    spectrum_value = spectrum_value.at[1, 1].multiply(0.25)
    spectrum_value = spectrum_value.at[1, -1].multiply(0.25)
    spectrum_value = spectrum_value.at[-1, 1].multiply(0.25)
    spectrum_value = spectrum_value.at[-1, -1].multiply(0.25)

    return jnp.fft.fft2(spectrum_value).real


def expected_single_subharmonic_correlation_function(
    spectrum: Callable,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    p: int,
    *args,
    **kwargs,
) -> Array:
    # Frequency spacing for this subharmonic level
    dqx_base = 2 * jnp.pi / (Nx * dx)
    dqy_base = 2 * jnp.pi / (Ny * dy)

    dqx = dqx_base / (3**p)
    dqy = dqy_base / (3**p)

    # Create the 32 sample points per level as described in the hybrid method
    # Each of the 9 sub-patches (from -3 to +3) is subdivided into 4 smaller patches
    # But we use the simplified approach from the paper: 6x6 grid with 0.5 offset
    qx_indices = jnp.arange(-3, 3) + 0.5  # 6 points: [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    qy_indices = jnp.arange(-3, 3) + 0.5  # 6 points: [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

    qxs = qx_indices * dqx
    qys = qy_indices * dqy

    # Spatial coordinates
    xs = jnp.arange(Nx) * dx
    ys = jnp.arange(Ny) * dy

    # Initialize correlation function
    correlation = jnp.zeros((Ny, Nx))

    # Sum over all frequency points in this subharmonic level
    for qx in qxs:
        for qy in qys:
            # Calculate spectrum value at this frequency point
            spectrum_val = spectrum(qx, qy, *args, **kwargs)

            # Calculate phase for all spatial points
            # Use broadcasting to create the phase matrix efficiently
            phase_matrix = jnp.outer(qy * ys, jnp.ones(Nx)) + jnp.outer(
                jnp.ones(Ny), qx * xs
            )

            # Add contribution to correlation (real part of inverse Fourier transform)
            correlation += spectrum_val * jnp.cos(phase_matrix) * dqx * dqy

    return correlation


def expected_correlation_function(
    spectrum: Callable,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    Np: int,
    *args,
    **kwargs,
) -> Array:
    result = expected_fourier_correlation_function(
        spectrum, Nx, Ny, dx, dy, *args, **kwargs
    )
    for p in range(1, Np + 1):
        result += expected_single_subharmonic_correlation_function(
            spectrum, Nx, Ny, dx, dy, p, *args, **kwargs
        )
    return result
