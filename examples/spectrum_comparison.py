import matplotlib.pyplot as plt
import jax.numpy as jnp
from turbulent_propagation import kolmogorov_spectrum, von_karman_spectrum, modified_von_karman_spectrum, hill_andrews_spectrum

r0 = 0.2
l0 = 1e-4
L0 = 50

qs = jnp.logspace(-2, 6, 256)

kolmogorov = kolmogorov_spectrum(qs, 0, r0)
von_karman = von_karman_spectrum(qs, 0, r0, L0)
modified_von_karman = modified_von_karman_spectrum(qs, 0, r0, L0, l0)
hill_andrews = hill_andrews_spectrum(qs, 0, r0, L0, l0)

plt.plot(qs, von_karman / kolmogorov, label='Von Karman')
plt.plot(qs, modified_von_karman / kolmogorov, label='Modified Von Karman', linestyle='--')
plt.plot(qs, hill_andrews / kolmogorov, label='Hill-Andrews', linestyle=':')
plt.xscale('log')
plt.xlabel('Spatial Frequency (1/m)')
plt.ylabel('Normalized spectrum')
plt.title('Turbulence Spectra Comparison')
plt.legend()
plt.grid()
plt.show()