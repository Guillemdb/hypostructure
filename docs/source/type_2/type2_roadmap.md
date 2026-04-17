# Local CKN Type II exclusion roadmap

This roadmap tracks the Type II program after replacing whole-space
critical-norm entry assumptions by the local CKN concentration-entry theorem.  The bridge
is [local_ckn_typeII_bridge.md](local_ckn_typeII_bridge.md).

## Entry Theorem

For a suitable weak solution, the local concentration-entry theorem proves
\[
(x_0,T)\text{ singular}
\Longrightarrow
\limsup_{r\downarrow0}\bigl(C((x_0,T),r)+D((x_0,T),r)\bigr)>0.
\]
Thus no-concentration is not a Type II branch.  With the Type I criterion assumed,
every remaining singularity enters the local Type II regime.

## Local Type II Chain

1. **Positive local CKN concentration.** Choose \(r_n\downarrow0\) with
   \(C+D\ge\eta\).  Parabolic rescalings are suitable on expanding backward
   cylinders.
2. **Local compactness.** If the local suitable package is bounded on compact
   cylinders, extract ancient suitable weak limits.  If those bounds fail or
   compact-cylinder mass splits, enter multibubble/cascade analysis.
3. **Single-core gauge.** On a retained single core, use compactly supported
   scale and centering gauges.  Gauge degeneracy is the multibubble/gauge
   degeneracy alternative.
4. **Local pressure.** Use \(P=P_{\mathrm{loc}}+H\) on compact balls, with
   \(P_{\mathrm{loc}}\) controlled by Calderon-Zygmund estimates and \(H\) by
   harmonic estimates.
5. **Local compact criterion.** Positive local CKN mass plus local gauge
   nondegeneracy, local Caccioppoli, local windowed \(H^1\), and finite
   localized Type II cost contradict the low-dissipation limit.
6. **Residual analysis.** Multibubble/gauge-degenerate and cascade branches
   are excluded by the local multibubble/cascade assembly in
   [local_typeII_exclusion_assembly.md](local_typeII_exclusion_assembly.md).

## Status Ledger

| Item | Status |
|---|---|
| Local no-concentration regularity theorem | Proved in `../first_nodes`. |
| Bridge to positive local Type II concentration | Implemented in `local_ckn_typeII_bridge.md`. |
| Local pressure replacement | Implemented as compact-ball pressure decomposition. |
| Local repaired gauge | Local implicit-function step; degeneracy enters multibubble. |
| Local compact Type II criterion | Adapted in `local_typeII_exclusion_assembly.md`. |
| U5a Caccioppoli | Valid because it is a compact-cylinder estimate. |
| Type I branch | Assumed blocked by the Type I criterion. |
| Multibubble/gauge-degenerate branch | Excluded by the local multibubble/cascade assembly. |
| Scale-collapse branch | Handled by the local scale-rigidity theorem inside the assembly. |

## Deprecated Entry Hypotheses

The following are no longer entry hypotheses for Type II exclusion from a
suitable weak singularity:

- whole-space critical-norm normalization;
- whole-space tightness;
- bounded terminal critical data on all of space;
- terminal critical profile decomposition;
- whole-space weighted repaired gauges.

They may appear only as auxiliary facts in optional global-data subtheorems, not
as assumptions needed to enter or close the local Type II singularity theorem.
