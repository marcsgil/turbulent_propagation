import matplotlib.pyplot as plt
from turbulent_propagation.spectra import von_karman_spectrum
from turbulent_propagation.expected_correlation_functions import (
    expected_correlation_function,
)

import jax.numpy as jnp
from scipy.special import gamma, kv


def von_karman_structure_function(r, r0, L0):
    """
    Calculate von Karman structure function according to the given formula:

    D_φ(r) = 6.16 * r0^(-5/3) * [3/5 * (L0/(2π))^(5/3) - (rL0/(4π))^(5/6) / Γ(11/6) * K_(5/6)(2πr/L0)]

    Parameters:
    r (float): position
    r0 (float): atmospheric coherence length
    L0 (float): outer scale of turbulence

    Returns:
    float: The calculated value of D_φ(r)
    """

    # Calculate r0^(-5/3)
    r0_power = r0 ** (-5 / 3)

    # First term: 3/5 * (L0/(2π))^(5/3)
    first_term = (3 / 5) * (L0 / (2 * jnp.pi)) ** (5 / 3)

    # Second term components
    # (rL0/(4π))^(5/6)
    second_numerator = (r * L0 / (4 * jnp.pi)) ** (5 / 6)

    # Γ(11/6)
    gamma_11_6 = gamma(11 / 6)

    # K_(5/6)(2πr/L0) - Modified Bessel function of the second kind
    bessel_arg = 2 * jnp.pi * r / L0
    bessel_k = kv(5 / 6, bessel_arg)

    # Second term: (rL0/(4π))^(5/6) / Γ(11/6) * K_(5/6)(2πr/L0)
    second_term = (second_numerator / gamma_11_6) * bessel_k

    # Complete calculation
    D_phi_result = 6.16 * r0_power * (first_term - second_term)

    return D_phi_result


N = 256
L = 2
d = L / N
r0 = 0.1
L0 = 10
Np = 3


expected_correlation_function = expected_correlation_function(
    spectrum=von_karman_spectrum,
    Nx=N,
    Ny=N,
    dx=d,
    dy=d,
    Np=Np,
    r0=r0,
    L0=L0,
)

expected_structure_function = 2 * (
    expected_correlation_function[0, 0] - expected_correlation_function[0, : N // 2]
)

rs = jnp.arange(N // 2) * d

plt.plot(rs, expected_structure_function, label="Expected Structure Function")
plt.plot(
    rs, von_karman_structure_function(rs, r0, L0), label="Von Karman Structure Function"
)

plt.legend()
plt.show()
