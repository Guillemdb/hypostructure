#!/usr/bin/env python3
"""Audit the Burgers ground-truth theorem axiom boundary.

This script asks Lean for two things in the same environment:

1. the declared boundary `burgersGroundTruthAxiomBoundary`, and
2. the kernel-level output of
   `#print axioms burgers_groundTruth_dataset_theorem_from_axioms`.

It succeeds only when the custom kernel axioms, after dropping the known Lean
foundations, exactly match the declared boundary by constant basename.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


LEAN_INPUT = r"""
import Hypostructure.Backends.Burgers1D.GroundTruthAudit

namespace Hypostructure.Backends.Burgers1D

#eval do
  IO.println "BEGIN_DECLARED_BOUNDARY"
  for name in burgersGroundTruthAxiomBoundary do
    IO.println name
  IO.println "END_DECLARED_BOUNDARY"

#print axioms burgers_groundTruth_dataset_theorem_from_axioms

end Hypostructure.Backends.Burgers1D
"""


KNOWN_LEAN_FOUNDATIONS = {
    "propext",
    "Classical.choice",
    "Quot.sound",
}


def fail(message: str) -> int:
    print(f"Burgers axiom audit failed: {message}", file=sys.stderr)
    return 1


def parse_declared_boundary(output: str) -> list[str]:
    lines = output.splitlines()
    try:
        start = lines.index("BEGIN_DECLARED_BOUNDARY") + 1
        end = lines.index("END_DECLARED_BOUNDARY")
    except ValueError as exc:
        raise ValueError("could not find declared-boundary markers in Lean output") from exc
    return [line.strip() for line in lines[start:end] if line.strip()]


def parse_kernel_axioms(output: str) -> list[str]:
    match = re.search(r"depends on axioms:\s*\[(.*?)\]\s*$", output, re.DOTALL)
    if match is None:
        raise ValueError("could not find `#print axioms` list in Lean output")
    return [token.strip() for token in match.group(1).split(",") if token.strip()]


def basename(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def main() -> int:
    lean_dir = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        ["lake", "env", "lean", "--stdin"],
        cwd=lean_dir,
        input=LEAN_INPUT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        return fail(f"Lean exited with status {proc.returncode}")

    try:
        declared = parse_declared_boundary(proc.stdout)
        kernel_axioms = parse_kernel_axioms(proc.stdout)
    except ValueError as exc:
        print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        return fail(str(exc))

    unexpected_foundations = sorted(
        axiom
        for axiom in kernel_axioms
        if not axiom.startswith("Hypostructure.") and axiom not in KNOWN_LEAN_FOUNDATIONS
    )
    if unexpected_foundations:
        print("Unexpected non-Hypostructure kernel axioms:", file=sys.stderr)
        for axiom in unexpected_foundations:
            print(f"  {axiom}", file=sys.stderr)
        return fail("unexpected foundation/mathlib axiom dependency")

    kernel_custom_full = [axiom for axiom in kernel_axioms if axiom.startswith("Hypostructure.")]
    kernel_custom = [basename(axiom) for axiom in kernel_custom_full]

    declared_set = set(declared)
    kernel_set = set(kernel_custom)

    declared_duplicates = sorted(name for name in declared_set if declared.count(name) > 1)
    kernel_duplicates = sorted(name for name in kernel_set if kernel_custom.count(name) > 1)
    if declared_duplicates or kernel_duplicates:
        if declared_duplicates:
            print("Duplicate declared boundary names:", file=sys.stderr)
            for name in declared_duplicates:
                print(f"  {name}", file=sys.stderr)
        if kernel_duplicates:
            print("Duplicate kernel custom axiom basenames:", file=sys.stderr)
            for name in kernel_duplicates:
                print(f"  {name}", file=sys.stderr)
        return fail("duplicate boundary names prevent exact comparison")

    missing = sorted(declared_set - kernel_set)
    extra = sorted(kernel_set - declared_set)
    if missing or extra:
        if missing:
            print("Declared boundary names missing from kernel axioms:", file=sys.stderr)
            for name in missing:
                print(f"  {name}", file=sys.stderr)
        if extra:
            print("Kernel custom axioms missing from declared boundary:", file=sys.stderr)
            for name in extra:
                full = [axiom for axiom in kernel_custom_full if basename(axiom) == name]
                print(f"  {name} ({', '.join(full)})", file=sys.stderr)
        return fail("declared boundary and kernel custom axioms differ")

    foundations = sorted(axiom for axiom in kernel_axioms if axiom in KNOWN_LEAN_FOUNDATIONS)
    print("Burgers axiom audit passed.")
    print(f"Declared custom boundary entries: {len(declared)}")
    print(f"Kernel custom axioms: {len(kernel_custom)}")
    print("Ignored Lean foundations: " + ", ".join(foundations))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
