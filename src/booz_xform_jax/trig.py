"""Shared trigonometric table helpers for booz_xform_jax."""

from __future__ import annotations

from functools import partial

import numpy as _np

try:
    import jax
    import jax.numpy as jnp
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The booz_xform_jax package requires JAX. Please install jax and "
        "jaxlib before using this module."
    ) from e


def _init_trig_np(
    theta_grid: _np.ndarray,
    zeta_grid: _np.ndarray,
    mmax: int,
    nmax: int,
    nfp: int,
) -> tuple[_np.ndarray, _np.ndarray, _np.ndarray, _np.ndarray]:
    """Pure-NumPy version of :func:`_init_trig` — no JAX dispatch overhead.

    Returns four arrays:
      cosm[i, m] = cos(m * theta[i])
      sinm[i, m] = sin(m * theta[i])
      cosn[i, n] = cos(n * nfp * zeta[i])
      sinn[i, n] = sin(n * nfp * zeta[i])
    with m in [0, mmax] and n in [0, nmax].
    """
    m_vals = _np.arange(0, mmax + 1, dtype=float)          # (mmax+1,)
    n_vals = _np.arange(0, nmax + 1, dtype=float) * nfp    # (nmax+1,)
    theta_m = _np.outer(theta_grid, m_vals)                 # (N, mmax+1)
    zeta_n  = _np.outer(zeta_grid,  n_vals)                 # (N, nmax+1)
    return _np.cos(theta_m), _np.sin(theta_m), _np.cos(zeta_n), _np.sin(zeta_n)


def _init_trig_np_T(
    theta_grid: _np.ndarray,
    zeta_grid: _np.ndarray,
    mmax: int,
    nmax: int,
    nfp: int,
) -> tuple[_np.ndarray, _np.ndarray, _np.ndarray, _np.ndarray]:
    """Like :func:`_init_trig_np` but returns **transposed** (mode-major) tables.

    Returns shape (mmax+1, N) for cos/sin_m and (nmax+1, N) for cos/sin_n.
    Row-major storage means each row (one m or n value) is contiguous in memory,
    so ``table[mode_indices, :]`` is a fast sequential-copy gather instead of a
    stride-heavy column gather.  Used for the per-surface Boozer trig tables.
    """
    m_vals = _np.arange(0, mmax + 1, dtype=float)
    n_vals = _np.arange(0, nmax + 1, dtype=float) * nfp
    theta_m = _np.outer(theta_grid, m_vals)   # (N, mmax+1)
    zeta_n  = _np.outer(zeta_grid,  n_vals)   # (N, nmax+1)
    # Transpose so that axis-0 is the mode index → contiguous row-gather
    return (
        _np.cos(theta_m).T.copy(),   # (mmax+1, N) C-contiguous
        _np.sin(theta_m).T.copy(),
        _np.cos(zeta_n).T.copy(),    # (nmax+1, N) C-contiguous
        _np.sin(zeta_n).T.copy(),
    )


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
