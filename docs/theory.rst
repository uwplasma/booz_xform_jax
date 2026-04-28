Theory And Derivations
======================

Overview
--------

BOOZ_XFORM converts the Fourier representation produced by VMEC into Boozer
coordinates, a straight-field-line coordinate system widely used in stellarator
and tokamak analysis. In the notation used by the STELLOPT BOOZ_XFORM
documentation, the magnetic field can be written as

.. math::

   \mathbf{B} = \nabla \psi(s) \times \nabla \zeta_B
   + \nabla \theta_B \times \nabla \chi(s),

where :math:`\theta_B` and :math:`\zeta_B` are Boozer poloidal and toroidal
angles. ``booz_xform_jax`` follows the same physical transform, but expresses
the numerically expensive steps in vectorized JAX primitives.

The starting point is a VMEC equilibrium represented on a radial half grid.
VMEC provides Fourier coefficients for:

- geometry: :math:`R(\theta, \zeta, s)` and :math:`Z(\theta, \zeta, s)`,
- the field-line straightening angle shift :math:`\lambda(\theta, \zeta, s)`,
- Nyquist-resolution magnetic-field quantities :math:`|B|`,
  :math:`B_\theta`, and :math:`B_\zeta`.

From these coefficients the code constructs Boozer-space spectra for:

- :math:`|B|`,
- :math:`R` and :math:`Z`,
- the toroidal angle shift :math:`\nu`,
- Jacobian-related harmonics,
- the Boozer current profiles :math:`I(s)` and :math:`G(s)`.

Coordinate Conventions
----------------------

The implementation matches the classical BOOZ_XFORM conventions:

- the Boozer mode list uses :math:`m = 0, 1, \ldots, m_{boz} - 1`,
- for :math:`m = 0`, only non-negative :math:`n` are kept,
- for :math:`m > 0`, toroidal modes run from :math:`-n_{boz}` to
  :math:`n_{boz}`,
- toroidal mode numbers are stored internally as ``xn = n * nfp``, following
  VMEC conventions,
- the output ``pmns`` and ``pmnc`` variables store
  :math:`p = \zeta_{vmec} - \zeta_{Boozer} = -\nu`.

That last sign convention is inherited from legacy BOOZ_XFORM and matters when
post-processing ``boozmn`` files or comparing against older utilities.

Derivation As Implemented Here
------------------------------

For each selected radial surface, the code proceeds in the following sequence.

1. Reconstruct VMEC-space fields on a tensor grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the non-Nyquist VMEC coefficients, the code synthesizes
:math:`R(\theta,\zeta)`, :math:`Z(\theta,\zeta)`, and
:math:`\lambda(\theta,\zeta)` on a tensor-product grid in VMEC angles. Fourier
derivatives are formed directly by multiplying the coefficient vectors by
:math:`m` and :math:`n`.

2. Construct the auxiliary field :math:`w`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Nyquist covariant magnetic-field components determine an auxiliary spectrum
for :math:`w`, using the same logic as the legacy BOOZ_XFORM implementation:

.. math::

   w_{mn} =
   \begin{cases}
   B_{\theta,mn}/m, & m \neq 0, \\
   -B_{\zeta,mn}/n, & m = 0,\ n \neq 0, \\
   0, & m = n = 0.
   \end{cases}

The code then reconstructs :math:`w(\theta,\zeta)` and its derivatives
:math:`\partial_\theta w` and :math:`\partial_\zeta w` on the VMEC grid.

3. Recover the Boozer angle shift
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The m=n=0 Nyquist coefficients give the Boozer current profiles
:math:`I(s)` and :math:`G(s)`. With the rotational transform :math:`\iota(s)`,
the toroidal shift :math:`\nu` is formed as

.. math::

   \nu(\theta,\zeta) = \frac{w(\theta,\zeta) - I \lambda(\theta,\zeta)}
   {G + \iota I}.

The Boozer angles follow immediately:

.. math::

   \theta_B = \theta + \lambda + \iota \nu,
   \qquad
   \zeta_B = \zeta + \nu.

The implementation also forms the derivatives
:math:`\partial_\theta \nu` and :math:`\partial_\zeta \nu`, needed in the
Fourier-integral weight.

4. Assemble the Jacobian factor used in the integrals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core code uses

.. math::

   dB/d(vmec) =
   (1 + \partial_\theta \lambda)(1 + \partial_\zeta \nu)
   + (\iota - \partial_\zeta \lambda)\partial_\theta \nu,

which is the coordinate-transformation factor entering the Boozer-space Fourier
integrals in this implementation.

5. Integrate against Boozer trigonometric tables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once :math:`\theta_B` and :math:`\zeta_B` are known on the grid, the code
builds cosine and sine tables in Boozer coordinates and computes the spectral
coefficients for :math:`|B|`, :math:`R`, :math:`Z`, :math:`\nu`, and the
Jacobian harmonics.

The Boozer Jacobian represented by ``gmnc_b`` and the file-compatible
``gmn_b`` alias is based on

.. math::

   \sqrt{g}_B = \frac{G + \iota I}{|B|^2}.

In practice, ``booz_xform_jax`` evaluates these integrals through
``jax.numpy.einsum`` contractions rather than explicit nested loops over modes
and grid points.

Magnetic-Field Component Conventions
------------------------------------

The STELLOPT documentation records the following covariant components in the
``boozmn`` file:

.. math::

   B_s = -\eta,
   \qquad
   B_\theta = \frac{I_{toroidal}}{2\pi},
   \qquad
   B_\phi = \frac{I_{poloidal}}{2\pi}.

It further identifies

.. math::

   I_{poloidal} = \frac{2\pi \langle B_v \rangle}{\mu_0},
   \qquad
   I_{toroidal} = \frac{2\pi \langle B_u \rangle}{\mu_0},

with ``buco`` and ``bvco`` corresponding to the flux functions stored in the
output. The contravariant components are then

.. math::

   B^\theta = 2\pi \frac{d\Psi_{poloidal}}{dV}\frac{B^2}{\langle B^2 \rangle},
   \qquad
   B^\phi = 2\pi \frac{d\Psi_{toroidal}}{dV}\frac{B^2}{\langle B^2 \rangle}.

These conventions matter when comparing against STELLOPT or when interpreting
``boozmn`` files in downstream tools.

Resolution Guidance
-------------------

The original STELLOPT BOOZ_XFORM documentation advises choosing a Boozer
resolution significantly higher than the underlying VMEC resolution, typically
five to six times more poloidal harmonics and two to three times more toroidal
harmonics. ``booz_xform_jax`` follows the same practical guidance:

- insufficient ``mboz`` or ``nboz`` truncates the Boozer spectra,
- high-resolution transforms recover sharper structure in :math:`|B|`,
- the required resolution depends strongly on the configuration and symmetry.

Related References
------------------

- A. H. Boozer, `Plasma equilibrium with rational magnetic surfaces
  <https://doi.org/10.1063/1.863813>`_.
- R. Sanchez, S. P. Hirshman, A. S. Ware, L. A. Berry, and D. A. Spong,
  `Ballooning stability optimization of low-aspect-ratio stellarators
  <https://doi.org/10.1088/0741-3335/42/7/307>`_.
- `STELLOPT BOOZ_XFORM documentation
  <https://princetonuniversity.github.io/STELLOPT/BOOZ_XFORM.html>`_.
- `HiddenSymmetries BOOZ_XFORM theory notes
  <https://hiddensymmetries.github.io/booz_xform/>`_.
