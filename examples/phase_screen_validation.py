import matplotlib.pyplot as plt
from turbulent_propagation import (
    von_karman_spectrum,
    phase_screen,
    atmospheric_coherence_length,
)
import jax.numpy as jnp
from scipy.special import gamma, kv
import jax
from jax import Array


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


def von_karman_structure_function(r, z, Cn2, L0, wavelength):
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

    r0 = atmospheric_coherence_length(z, Cn2, wavelength)

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

Cn2 = 1e-14
wavelength = 633e-9
z = 100
print(atmospheric_coherence_length(z, Cn2, wavelength))

k = 2 * jnp.pi / wavelength

screen = (
    2
    * jnp.pi
    * k**2
    * z
    * phase_screen(
        spectrum=von_karman_spectrum,
        Nx=N,
        Ny=N,
        dx=d,
        dy=d,
        Np=Np,
        nsamples=1000,
        key=jax.random.key(42),
        Cn2=Cn2,
        L0=L0,
    )
)

plt.imshow(screen[2])
plt.show()
plt.close()

statistical_structure_function_val = statistical_structure_function(screen)

rs = jnp.arange(N // 2) * d

plt.plot(
    rs,
    von_karman_structure_function(rs, z, Cn2, L0, wavelength),
    label="Von Karman Structure Function",
    linestyle="--",
)
#plt.plot(rs, statistical_structure_function_val, label="Statistical Structure Function")

plt.legend()
plt.show()
