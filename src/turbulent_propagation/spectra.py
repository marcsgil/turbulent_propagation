"""
Turbulent atmosphere power spectral density functions.

This module implements various power spectral density functions used in atmospheric turbulence modeling,
including the modified von Karman, von Karman, Kolmogorov, and Hill-Andrews spectra.
"""

import jax.numpy as jnp
from jax.typing import ArrayLike


def modified_von_karman_spectrum(
    qx: ArrayLike, qy: ArrayLike, r0: float, L0: float, l0: float
) -> ArrayLike:
    """
    Calculate the modified von Karman spectrum for given wavevector components.

    The modified von Karman spectrum is:
    Φ(q) = 0.033 Cn² * exp(-|q|² / (5.92/l₀)²) / (|q|² + (2π/L₀)²)^(11/6)

    Parameters:
        qx (ArrayLike): Wavenumber vector (x-direction) from meshgrid.
        qy (ArrayLike): Wavenumber vector (y-direction) from meshgrid.
        r0 (float): Atmospheric coherence length.
        L0 (float): Outer scale of turbulence.
        l0 (float): Inner scale of turbulence.

    Returns:
        ArrayLike: The calculated modified von Karman spectrum.
    """
    q_squared = qx**2 + qy**2
    q_m_squared = (5.92 / l0) ** 2
    q_0_squared = (2 * jnp.pi / L0) ** 2

    return (
        0.49
        / r0 ** (5 / 3)
        * jnp.exp(-q_squared / q_m_squared)
        / (q_squared + q_0_squared) ** (11 / 6)
    )


def von_karman_spectrum(
    qx: ArrayLike, qy: ArrayLike, Cn2: float, L0: float
) -> ArrayLike:
    """
    Calculate the von Karman spectrum for given wavevector components.

    The von Karman spectrum is a special case of the modified von Karman spectrum with l₀ = 0.

    Parameters:
        qx (ArrayLike): Wavenumber vector (x-direction) from meshgrid.
        qy (ArrayLike): Wavenumber vector (y-direction) from meshgrid.
        Cn2 (float): Refractive index structure function
        L0 (float): Outer scale of turbulence.

    Returns:
        ArrayLike: The calculated von Karman spectrum.
    """
    q_squared = qx**2 + qy**2
    q_0_squared = (2 * jnp.pi / L0) ** 2

    return 0.033 * Cn2 / (q_squared + q_0_squared) ** (11 / 6)


def kolmogorov_spectrum(qx: ArrayLike, qy: ArrayLike, r0: float) -> ArrayLike:
    """
    Calculate the Kolmogorov spectrum for given wavevector components.

    The Kolmogorov spectrum is a special case of the von Karman spectrum with L₀ = ∞.

    Parameters:
        qx (ArrayLike): Wavenumber vector (x-direction) from meshgrid.
        qy (ArrayLike): Wavenumber vector (y-direction) from meshgrid.
        r0 (float): Atmospheric coherence length.

    Returns:
        ArrayLike: The calculated Kolmogorov spectrum.
    """
    return 0.49 / r0 ** (5 / 3) / (qx**2 + qy**2) ** (11 / 6)


def hill_andrews_spectrum(
    qx: ArrayLike, qy: ArrayLike, r0: float, L0: float, l0: float
) -> ArrayLike:
    """
    Calculate the Hill-Andrews spectrum for given wavevector components.

    The Hill-Andrews spectrum is given by:
    Φ(q) = Φ_MK(q) * (1 + 1.802 * sqrt(q²/ql²) - 0.254 * (q²/ql²)^(7/12))
    where ql = 3.3 / l₀ and Φ_MK is the modified von Karman spectrum.

    Parameters:
        qx (ArrayLike): Wavenumber vector (x-direction) from meshgrid.
        qy (ArrayLike): Wavenumber vector (y-direction) from meshgrid.
        r0 (float): Atmospheric coherence length.
        L0 (float): Outer scale of turbulence.
        l0 (float): Inner scale of turbulence.

    Returns:
        ArrayLike: The calculated Hill-Andrews spectrum.
    """
    q_squared = qx**2 + qy**2
    ql_squared = (3.3 / l0) ** 2
    q0_squared = (2 * jnp.pi / L0) ** 2

    return (
        0.49
        / r0 ** (5 / 3)
        * jnp.exp(-q_squared / ql_squared)
        / (q_squared + q0_squared) ** (11 / 6)
        * (
            1
            + 1.802 * jnp.sqrt(q_squared / ql_squared)
            - 0.254 * (q_squared / ql_squared) ** (7 / 12)
        )
    )
