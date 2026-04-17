# Local CKN Type II proof architecture

This folder organizes the local Caffarelli-Kohn-Nirenberg Type II exclusion
program for the three-dimensional incompressible Navier-Stokes equations.  The
starting point is [local_ckn_typeII_bridge.md](local_ckn_typeII_bridge.md), which
connects the local concentration theorem in
[../first_nodes](../first_nodes/README.md) to the Type II analysis.

The current initial reduction theorem is local.  A putative singular point is first tested
by the CKN scale-invariant density
\[
C(z_0,r)+D(z_0,r),
\]
where the pressure term is normalized by subtracting its spatial mean on the
ball.  If this density vanishes as \(r\downarrow0\), the CKN epsilon regularity
theorem gives regularity.  If the point is singular, the density is therefore
positive along a sequence of compact parabolic cylinders.  With the Type I
criterion assumed upstream, every remaining positive local concentration enters
the Type II branch.

No whole-space critical-norm control, whole-space tightness, or terminal profile
decomposition is used as an entry hypothesis for the Type II singularity
argument.  Such tools may still appear in optional global-data subtheorems, but
they are not part of the local singularity entry.

## Local Objects

The Type II proof is now formulated using only compact-cylinder objects:

1. positive CKN density on parabolic cylinders;
2. compact-cylinder bounds for the local suitable package under parabolic rescaling;
3. local pressure decomposition into a Calderon-Zygmund part and a harmonic
   part;
4. compactly supported repaired scale and centering gauges on a selected core;
5. local Caccioppoli estimates and local windowed \(H^1\) control;
6. a local compact Type II criterion;
7. a multibubble or gauge-degenerate alternative when no unique core can be
   selected.

The phrase "critical mass" in this folder means local CKN mass on compact
parabolic cylinders unless a document explicitly says otherwise.

## Main Bridge

The local implication theorem proves the following local implication:
\[
\text{finite-time singularity}
\Longrightarrow
\text{positive local CKN concentration}.
\]
After the Type I criterion removes the self-similar scale regime, a remaining
positive local concentration has two Type II outcomes.

1. **Single-core branch.** The local suitable package is bounded on compact
   cylinders and a localized repaired gauge is nondegenerate.  This branch is
   excluded by the single-core Type II criterion.
2. **Multibubble or gauge-degenerate branch.** Compact-cylinder bounds fail, or
   a unique local core cannot be selected.  This branch is sent to the
   multibubble and scale-rigidity analysis.

Thus the bridge removes the former starting assumptions.  Full Type II exclusion
is established in [local_typeII_exclusion_assembly.md](local_typeII_exclusion_assembly.md).
After the Type I criterion, the adapted single-core and multibubble/cascade
criteria exclude the Type II branch.

## Document Map

- [local_ckn_typeII_bridge.md](local_ckn_typeII_bridge.md): bridge from the
  local concentration theorem to local Type II concentration and Type II
  exclusion after the Type I criterion.
- [local_typeII_exclusion_assembly.md](local_typeII_exclusion_assembly.md):
  adaptation of the good-window single-core criterion and multibubble/cascade
  exclusion to the local CKN entry.
- [type2_roadmap.md](type2_roadmap.md): current local roadmap and local branches.
- [cluster_u5a_bare_data_caccioppoli.md](cluster_u5a_bare_data_caccioppoli.md):
  compact-cylinder Caccioppoli input.
- [cluster_s4_multibubble_residue_classification.md](cluster_s4_multibubble_residue_classification.md)
  through
  [cluster_s8_terminal_nonlinear_profile_decoupling.md](cluster_s8_terminal_nonlinear_profile_decoupling.md):
  multibubble and decoupling layer, read through the local CKN entry.
- [cluster_s3_scale_collapse_attractor_stratification.md](cluster_s3_scale_collapse_attractor_stratification.md):
  scale-collapse rigidity layer.

## Paper Series

The papers in `docs/source/type_2/papers` follow the same local reduction
convention.

1. Paper I: local compact Type II criterion.
2. Paper II: localized repaired gauges and local pressure representation.
3. Paper III: local compactness, radiation, and multibubble alternatives.
4. Paper IV: local Caccioppoli regularity and rough-core exclusion.

These papers should not state whole-space critical control as an entry
assumption for a singularity.  Their entry data are local CKN concentration,
local suitability, and compact-cylinder estimates.
