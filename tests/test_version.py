"""The package version is declared twice; keep the two declarations equal."""

from __future__ import annotations

from pathlib import Path
import re

import booz_xform_jax


def test_dunder_version_matches_pyproject() -> None:
    # __version__ also feeds the version shown in the built documentation, and
    # it read 0.1.0 through the 0.1.1, 0.2.0 and 0.3.0 releases.
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.MULTILINE)
    assert match is not None
    assert booz_xform_jax.__version__ == match.group(1)
