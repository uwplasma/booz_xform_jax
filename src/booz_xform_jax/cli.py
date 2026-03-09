"""Legacy-compatible command line interface for ``booz_xform_jax``.

This module mirrors the STELLOPT ``xbooz_xform`` driver:

    xbooz_xform <infile> [T|F]

where ``<infile>`` contains:

1. ``mboz nboz`` on the first line,
2. a VMEC extension or wout filename on the second line,
3. an optional whitespace-separated list of full-grid surface indices.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import time

from .core import Booz_xform


HELP_TEXT = """ ENTER INPUT FILE NAME ON COMMAND LINE
 For example: xbooz_xform in_booz.ext

 WHERE in_booz.ext is the input file

 Optional command line argument
 xbooz_xform <infile> (T or F)

 where F suppresses output to the screen
"""


def _iter_existing(paths: list[Path]) -> Path | None:
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path.resolve()
    return None


def _resolve_wout_path(file_or_extension: str, *, input_dir: Path) -> Path:
    """Resolve a VMEC wout file from a legacy extension string."""
    raw = file_or_extension.strip()
    roots = [input_dir, Path.cwd()]

    def _expand(candidate: str) -> list[Path]:
        path = Path(candidate)
        if path.is_absolute():
            return [path]
        return [root / path for root in roots]

    looks_like_filename = (
        "/" in raw
        or raw.endswith(".nc")
        or raw.endswith(".txt")
        or raw.startswith("wout_")
        or raw.startswith("wout.")
    )

    candidates: list[Path] = []
    if looks_like_filename:
        candidates.extend(_expand(raw))
        if not raw.endswith((".nc", ".txt")):
            candidates.extend(_expand(raw + ".nc"))
            candidates.extend(_expand(raw + ".txt"))
    else:
        candidates.extend(_expand(f"wout_{raw}.nc"))
        candidates.extend(_expand(f"wout.{raw}.nc"))
        candidates.extend(_expand(f"wout_{raw}.txt"))
        candidates.extend(_expand(f"wout.{raw}.txt"))
        candidates.extend(_expand(f"wout_{raw}"))
        candidates.extend(_expand(f"wout.{raw}"))
        candidates.extend(_expand(raw))
        candidates.extend(_expand(raw + ".nc"))
        candidates.extend(_expand(raw + ".txt"))

    resolved = _iter_existing(candidates)
    if resolved is None:
        raise FileNotFoundError(f"Could not find VMEC output for extension '{raw}'")
    return resolved


def _normalize_output_extension(file_or_extension: str) -> str:
    """Convert a legacy extension or wout filename into the boozmn suffix."""
    name = Path(file_or_extension.strip()).name
    if name.endswith(".nc"):
        name = name[:-3]
    elif name.endswith(".txt"):
        name = name[:-4]
    if name.startswith("wout_"):
        name = name[5:]
    elif name.startswith("wout."):
        name = name[5:]
    return name


def _parse_input_file(path: Path) -> tuple[int, int, str, list[int] | None, bool]:
    """Parse a legacy ``in_booz`` file."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise OSError("Error opening input file in booz_xform") from exc

    nonempty = [line.strip() for line in lines if line.strip()]
    if len(nonempty) < 2:
        raise ValueError("Error reading input file in booz_xform")

    first = nonempty[0].split()
    if len(first) < 2:
        raise ValueError("Error reading input file in booz_xform")

    try:
        mboz = int(first[0])
        nboz = int(first[1])
    except ValueError as exc:
        raise ValueError("Error reading input file in booz_xform") from exc

    extension = nonempty[1]
    remainder = "\n".join(nonempty[2:])
    surface_tokens = [int(token) for token in re.findall(r"-?\d+", remainder)]
    if not surface_tokens:
        return mboz, nboz, extension, None, True
    return mboz, nboz, extension, surface_tokens, False


def _select_compute_surfs(surface_indices: list[int] | None, *, ns_in: int) -> list[int] | None:
    """Map full-grid Boozer surface indices to internal half-grid indices."""
    if surface_indices is None:
        return None
    ns_full = ns_in + 1
    valid = sorted({idx for idx in surface_indices if 1 < idx <= ns_full})
    return [idx - 2 for idx in valid]


def run_from_legacy_input(input_file: str | Path, *, screen_output: bool = True) -> Path:
    """Run ``booz_xform_jax`` from an STELLOPT-style ``in_booz`` file."""
    input_path = Path(input_file)
    mboz_in, nboz_in, extension, surface_indices, missing_surfaces = _parse_input_file(input_path)

    if missing_surfaces:
        print(" No jlist data was found in Boozer input file. Will assume that all surfaces are needed.")
        print(" Iostat:   -1")

    wout_path = _resolve_wout_path(extension, input_dir=input_path.parent.resolve())
    output_extension = _normalize_output_extension(extension)
    output_path = input_path.parent / f"boozmn_{output_extension}.nc"

    booz = Booz_xform(verbose=1 if screen_output else 0)
    booz.read_wout(str(wout_path), flux=True)
    booz.mboz = max(int(mboz_in), 2, 6 * int(booz.mpol))
    booz.nboz = max(int(nboz_in), 0, 2 * int(booz.ntor) - 1)
    booz.compute_surfs = _select_compute_surfs(surface_indices, ns_in=int(booz.ns_in))

    t1 = time.perf_counter()
    booz.run()
    booz.write_boozmn(str(output_path))
    t2 = time.perf_counter()

    if screen_output:
        print()
        print(f" TIME IN BOOZER TRANSFORM CODE: {t2 - t1:12.2E} SEC")

    return output_path


def main(argv: list[str] | None = None) -> int:
    """Command line entrypoint."""
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Invalid command line in calling xbooz_xform")
        print("Type xbooz_xform -h to get more information")
        return 1

    if argv[0] in {"-h", "/h", "--help"}:
        print(HELP_TEXT, end="")
        return 0

    if len(argv) > 2:
        print("Invalid command line in calling xbooz_xform")
        print("Type xbooz_xform -h to get more information")
        return 1

    screen_output = True
    if len(argv) == 2:
        flag = argv[1].strip()
        if flag and flag[0] in {"f", "F"}:
            screen_output = False

    try:
        run_from_legacy_input(argv[0], screen_output=screen_output)
    except Exception as exc:
        print(str(exc))
        return 1

    return 0

