import jax.numpy as jnp
from jax import Array, random
from typing import Callable


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


def calculate_spectrum(
    spectrum: Callable,
    qxs: Array,
    qys: Array,
    *args,
    **kwargs,
) -> Array:
    qs = jnp.meshgrid(qxs, qys)
    dqxs = qxs[1] - qxs[0]
    dqys = qys[1] - qys[0]
    return spectrum(qs, *args, **kwargs) * dqxs * dqys


def calculate_fft_spectrum(
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

    return calculate_spectrum(spectrum, qxs, qys, *args, **kwargs)


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
    random_numbers = random.normal(key, (nsamples, Ny, Nx), dtype=jnp.complex64)
    spectrum_value = calculate_fft_spectrum(spectrum, Nx, Ny, dx, dy, *args, **kwargs)
    return jnp.fft.fft2(random_numbers * jnp.sqrt(2 * spectrum_value)).real


def expected_fourier_correlation_function(
    spectrum: Callable,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    *args,
    **kwargs,
) -> Array:
    spectrum_value = calculate_fft_spectrum(spectrum, Nx, Ny, dx, dy, *args, **kwargs)
    return jnp.fft.fft2(spectrum_value).real


def calculate_subharmonic_spectrum(
    spectrum: Callable,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    p: int,
    *args,
    **kwargs,
) -> Array:
    dqx = 2 * jnp.pi / (3**p * Nx * dx)
    dqy = 2 * jnp.pi / (3**p * Ny * dy)

    qxs = jnp.arange(-3, 3) * dqx
    qys = jnp.arange(-3, 3) * dqy

    return calculate_spectrum(spectrum, qxs, qys, *args, **kwargs)


def single_subharmonic_phase_screen(
    spectrum: Callable,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    p: int,
    nsamples: int = 1,
    key: Array = random.key(42),
    *args,
    **kwargs,
) -> Array:
    random_numbers = random.normal(key, (nsamples, Ny, Nx), dtype=jnp.complex64)
    spectrum_value = calculate_subharmonic_spectrum(
        spectrum, Nx, Ny, dx, dy, p, *args, **kwargs
    )

    

    return jnp.fft.fft2(random_numbers * jnp.sqrt(2 * spectrum_value)).real


def subharmonic_phase_screen(
    spectrum: Callable,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    Np: int,
    nsamples: int = 1,
    key: Array = random.key(42),
    *args,
    **kwargs,
) -> Array:
    result = single_subharmonic_phase_screen(
        spectrum,
        Nx,
        Ny,
        dx,
        dy,
        p=1,
        nsamples=nsamples,
        key=key,
        *args,
        **kwargs,
    )
    for p in range(2, Np):
        key, subkey = random.split(key)
        result += single_subharmonic_phase_screen(
            spectrum,
            Nx,
            Ny,
            dx,
            dy,
            p,
            nsamples,
            key,
            *args,
            **kwargs,
        )
    return result


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
    spectrum_value = calculate_subharmonic_spectrum(
        spectrum, Nx, Ny, dx, dy, p, *args, **kwargs
    )
    return jnp.fft.fft2(spectrum_value).real


def expected_subharmonic_correlation_function(
    spectrum: Callable,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    Np: int,
    *args,
    **kwargs,
) -> Array:
    result = expected_single_subharmonic_correlation_function(
        spectrum, Nx, Ny, dx, dy, 1, *args, **kwargs
    )
    for p in range(2, Np):
        result += expected_single_subharmonic_correlation_function(
            spectrum, Nx, Ny, dx, dy, p, *args, **kwargs
        )
    return result


def phase_screen(
    spectrum: Callable,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    Np: int,
    nsamples: int = 1,
    key: Array = random.key(42),
    *args,
    **kwargs,
) -> Array:
    key, subkey = random.split(key)
    return fourier_phase_screen(
        spectrum, Nx, Ny, dx, dy, nsamples, key, *args, **kwargs
    ) + subharmonic_phase_screen(
        spectrum, Nx, Ny, dx, dy, Np, nsamples, key=subkey, *args, **kwargs
    )


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
    return expected_fourier_correlation_function(
        spectrum, Nx, Ny, dx, dy, *args, **kwargs
    ) + expected_subharmonic_correlation_function(
        spectrum, Nx, Ny, dx, dy, Np, *args, **kwargs
    )
