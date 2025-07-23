import jax.numpy as jnp
from jax import Array


def von_karman_spectrum(q: Array, r0: float, L0: float) -> Array:
    """
    Calculate the von Karman spectrum given by the formula:
    S(q) = 0.49 / r0^(5/3) / (|q|^2 + (2π/L0)^2)^(11/6)

    Parameters:
        q (Array): Wavenumber vector.
        r0 (float): Atmospheric coherence length.
        L0 (float): Outer scale of turbulence.

    Returns:
        Array: The calculated von Karman spectrum.
    """
    k_squared = q[0] ** 2 + q[1] ** 2
    k0 = 2 * jnp.pi / L0
    return 0.49 / r0 ** (5 / 3) / (k_squared + k0**2) ** (11 / 6)
