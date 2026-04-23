# hypostructure

**soft math for solving hard problems**

[![DOI](https://zenodo.org/badge/1104911391.svg)](https://doi.org/10.5281/zenodo.18041040)

## Overview

Hypostructure is a categorical framework for analyzing mathematical problems through structural constraints. It provides a systematic method—the **Structural Sieve**—to determine whether problems admit solutions by examining their intrinsic structural properties rather than through direct computation.

The framework operates within a cohesive (∞, 1)-topos and encodes problems as **hypostructure objects**: tuples (𝒳, ∇, Φ, τ, ∂) representing state space, dynamics, energy/complexity, truncation structure, and boundary morphisms. Six core axioms (Compatibility, Dissipation, Symmetry Compatibility, Local Stiffness, Capacity, and Topological Background) govern when solutions exist.

## Key Components

- **Structural Sieve**: A 17-node diagnostic automaton that emits typed certificates (YES, NO-witness, or NO-inconclusive) for each problem analyzed
- **Metatheorems**: Proof factories enabling type-based instantiation from minimal primitives
- **Dataset**: 40 test problems spanning Millennium Prize problems, classical theorems, PDEs, number theory, and computational complexity
- **Machine Learning Implementations**: Neural network architectures (`hypoatlas.py`, `combined.py`, `hypodiscovery.py`) applying hypostructure principles to learn manifold structure from data

## Documentation

- `docs/source/hypopermits_jb.md` — Complete categorical formalism and proofs
- `docs/source/metalearning.md` — Meta-learning axioms and learning theory
- `docs/source/reference.md` — Quick reference for executing the sieve
- `docs/source/dataset/` — Annotated problem dataset with verdicts
- `docs/source/proofs/` — Individual metatheorem proofs

## Citation

If you use hypostructure in your research, please cite:

```bibtex
@software{guillemdb_hypostructure_2025,
  author       = {Duran Ballester, Guillem},
  title        = {hypostructure: soft math for solving hard problems},
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18041040},
  url          = {https://doi.org/10.5281/zenodo.18041040}
}
```

Or in text form:

> Duran Ballester, G. (2025). *hypostructure: soft math for solving hard problems*. Zenodo. https://doi.org/10.5281/zenodo.18041040

## License

See repository for license details.


plese you job now is to refview and implement every single item in @file:paperIV_revision_plan_readable.md into the paper located in @file:paperIV_residual_branch.tex  item by item. first you are going to review taht if the item is present. in case it is make sure it is correctly implemented, and in case it is missing your job will be implementing it. before making any edits you are going to evaluate if the requested change is correct, and if it requires anything going against your goal then dont implement it and move to the next item. your goal are: - Make the paper airtight and reazdy to submission to a top journal. - make sure that the paper uses standard pde notation, language and professinal tone. - do not alter the proof strategy, you are only allowed to add new amth taht expands it and makes it more explicit but never change the overall diurection. the goal is a valid state space stratification strategy taht avoids hard global estimates. - you will iterate one item after the other and implement it correctly. -you will not stop until all items are implemented. - be extremely exhaustive formatl and rigorous. - make sure to implement every single improvement