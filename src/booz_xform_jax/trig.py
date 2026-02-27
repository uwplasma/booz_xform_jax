"""Shared trigonometric table helpers for booz_xform_jax."""

from __future__ import annotations

from functools import partial

try:
    import jax
    import jax.numpy as jnp
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The booz_xform_jax package requires JAX. Please install jax and "
        "jaxlib before using this module."
    ) from e


@partial(jax.jit, static_argnums=(2, 3, 4))
def _init_trig(
    theta_grid: jnp.ndarray,
    zeta_grid: jnp.ndarray,
    mmax: int,
    nmax: int,
    nfp: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build trigonometric tables on a flattened (theta, zeta) grid."""
    theta = theta_grid[:, None]
    zeta = zeta_grid[:, None]

    m_vals = jnp.arange(0, mmax + 1, dtype=jnp.float64)[None, :]
    n_vals = jnp.arange(0, nmax + 1, dtype=jnp.float64)[None, :]

    cosm = jnp.cos(theta * m_vals)
    sinm = jnp.sin(theta * m_vals)
    cosn = jnp.cos(zeta * (n_vals * nfp))
    sinn = jnp.sin(zeta * (n_vals * nfp))

    return cosm, sinm, cosn, sinn
