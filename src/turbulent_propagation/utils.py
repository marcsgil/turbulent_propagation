import jax.numpy as jnp


def atmospheric_coherence_length(z: float, Cn2: float, wavelength: float) -> float:
    """
    Calculate the atmospheric coherence length (r0), also known as the Fried parameter,
    based on the propagation distance (z) and the refractive index structure
    constant (Cn2).

    Parameters:
    - z: Propagation distance (m)
    - Cn2: Refractive index structure constant (m^-2/3)

    Returns:
    - Coherence length (r0) (m)
    """
    return 0.185 * (wavelength**2 / Cn2 / z) ** (3 / 5)


def rytov_variance(z: float, Cn2: float, wavelength: float) -> float:
    """
    Calculate the Rytov variance based on the propagation distance (z),
    the refractive index structure constant (Cn2), and the wavelength.

    Parameters:
    - z: Propagation distance (m)
    - Cn2: Refractive index structure constant (m^-2/3)
    - wavelength: Wavelength of the wave (m)

    Returns:
    - Rytov variance (dimensionless)
    """
    return 1.23 * Cn2 * (2 * jnp.pi / wavelength) ** (7 / 6) * z ** (11 / 6)
