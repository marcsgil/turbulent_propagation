def atmospheric_coherence_length(dz: float, Cn2: float, wavelength: float) -> float:
    """
    Calculate the atmospheric coherence length (r0) based on the
    propagation distance (dz) and the refractive index structure
    constant (Cn2).

    Parameters:
    - dz: Propagation distance (m)
    - Cn2: Refractive index structure constant (m^-2/3)

    Returns:
    - Coherence length (r0) (m)
    """
    return 0.185 * (wavelength**2 / Cn2 / dz) ** (3 / 5)
