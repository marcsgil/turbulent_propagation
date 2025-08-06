import unittest
import jax.numpy as jnp
from turbulent_propagation import angular_spectrum_propagation


def gaussian_beam(x, y, z, w0, wavelength):
    zr = jnp.pi * w0**2 / wavelength
    w = w0 * jnp.sqrt(1 + (z / zr) ** 2)
    if z == 0:
        R = jnp.inf
    else:
        R = z * (1 + (zr / z) ** 2)
    psi = jnp.arctan(z / zr)
    r2 = x**2 + y**2
    k = 2 * jnp.pi / wavelength
    return (w0 / w) * jnp.exp(-r2 / w**2 - 1j * (k * r2 / 2 / R - psi))


class TestAngularSpectrumPropagation(unittest.TestCase):
    def setUp(self):
        # Set up a simple test case
        self.L = 8.0  # Size of the domain
        self.N = 256  # Number of points in each dimension
        self.dx = self.L / self.N  # Spatial step size
        self.dy = self.dx  # Assuming square grid
        self.wavelength = 2 * jnp.pi  # Wavelength of the wave

        xs = jnp.arange(-self.L / 2, self.L / 2, self.dx)
        ys = jnp.arange(-self.L / 2, self.L / 2, self.dy)
        xs, ys = jnp.meshgrid(xs, ys)
        self.xs = xs
        self.ys = ys
        self.u0 = gaussian_beam(xs, ys, 0, 1.0, self.wavelength)

    def test_analytical_solution(self):
        zs = jnp.linspace(0.01, 0.6, 32)
        for z in zs:
            # Perform the propagation
            u = angular_spectrum_propagation(
                self.u0, self.dx, self.dy, z, self.wavelength, 1
            )
            u_analytic = gaussian_beam(self.xs, self.ys, z, 1.0, self.wavelength)

            self.assertTrue(jnp.allclose(jnp.abs(u), jnp.abs(u_analytic), atol=1e-3))

    def test_angular_spectrum_propagation(self):
        zs = jnp.linspace(0.01, 1, 32)
        magnifications = jnp.sqrt(1 + 4 * zs**2)  # Magnification factor
        for m, z in zip(magnifications, zs):
            # Perform the propagation
            u = angular_spectrum_propagation(
                self.u0, self.dx, self.dy, z, self.wavelength, m
            )
            self.assertTrue(jnp.allclose(jnp.abs(u * m), jnp.abs(self.u0), atol=1e-6))
