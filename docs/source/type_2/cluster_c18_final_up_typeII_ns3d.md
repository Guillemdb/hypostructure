# C18 final `UP-TypeII` theorem for the declared NS3D Type II backend

This note is the final assembly theorem for the Type II program in this
folder. It does not add a new PDE estimate. It composes the existing C-series
classification stack, the S-series scale-collapse and multibubble closures,
and the C10 NS-valid promotion bridge.

The output is the Type-II-specific statement

```{math}
\forall\omega\in\mathcal U_{\mathrm{II}}^{NS},
\qquad
K_{\mathrm{SC}_\lambda}^{\sim}(\omega),
```

inside the declared NS3D repaired-gauge Type II backend.

This is the terminal NS3D Type II certificate assembly. It says that once a
candidate is routed into the declared Type II backend and all listed backend
payloads are emitted, no unresolved Type II branch remains.

## Abstract final package

::::{prf:definition} Abstract final NS3D `UP-TypeII` package
:label: def-c18-abstract-final-ns3d-uptypeii-package

The certificate

```{math}
K_{\mathrm{UPTypeII},NS3D}^{\mathrm{final}}
```

means the conjunction

```{math}
K_{\mathrm{ClassComplete}}^+
\wedge
K_{\mathrm{RadBlk}}^+
\wedge
K_{\mathrm{RoughCoreBlk}}^+
\wedge
K_{\mathrm{NS\text{-}UPTypeII}}^+.
```

Here:

1. \(K_{\mathrm{ClassComplete}}^+\) is the C1--C5 declared-backend
   classification-completeness package.
2. \(K_{\mathrm{RadBlk}}^+\) is the universal C6 radiative/noncompact blocker,
   equivalently universal \(K_{L^3\mathrm{Tight}}^+\).
3. \(K_{\mathrm{RoughCoreBlk}}^+\) is the universal C7 rough-core blocker,
   equivalently universal \(K_{\mathrm{WinH1}}^+\) on represented branches.
4. \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) is the C10 NS-valid promotion
   certificate.

::::

::::{prf:theorem} Final abstract NS3D `UP-TypeII`
:label: thm-c18-final-abstract-ns3d-uptypeii

Assume

```{math}
K_{\mathrm{UPTypeII},NS3D}^{\mathrm{final}}.
```

Then every declared Type II candidate emits

```{math}
K_{\mathrm{SC}_\lambda}^{\sim}.
```

Equivalently, the declared NS3D Type II backend has no admissible unresolved
Type II branch.

::::

:::{prf:proof}
This is exactly C8. By Definition
{prf:ref}`def-c18-abstract-final-ns3d-uptypeii-package`, the hypotheses of
Theorem {prf:ref}`thm-c8-typeII-branch-exclusion` are present. Therefore every
declared candidate \(\omega\in\mathcal U_{\mathrm{II}}^{NS}\) emits
\(K_{\mathrm{SC}_\lambda}^{\sim}(\omega)\). \(\square\)
:::

## Expanded terminal-backend package

The abstract package is useful, but it hides the S-series discharges. The next
definition expands the pieces that remove scale-collapse, multibubble, and
radiative residues in the current terminal backend.

::::{prf:definition} Terminal-backend final payload
:label: def-c18-terminal-backend-final-payload

The certificate

```{math}
K_{\mathrm{UPTypeII},NS3D}^{\mathrm{term}}
```

means the conjunction

```{math}
K_{\mathrm{ClassComplete}}^+
\wedge
K_{\mathrm{TechTypeII}}^{\mathrm{blk},+}
\wedge
K_{\mathrm{C6Route}}^+
\wedge
K_{\mathrm{StateStratExh}}^+
\wedge
K_{\mathrm{StratCritPacket}}^+
\wedge
K_{\mathrm{SmallDataStab}_{L^3}}^+
\wedge
K_{\mathrm{CriticalNSProfDecomp}}^+
\wedge
K_{\mathrm{ScattBranch}}^-
\wedge
K_{\mathrm{ExtRegDiscard}}^+
\wedge
K_{\mathrm{RepBridge}}^+
\wedge
K_{\mathrm{CaccioppoliReg}}^+
\wedge
K_{\mathrm{S3NRSPayload}}^+
\wedge
K_{\mathrm{NS\text{-}UPTypeII}}^+.
```

The payload meanings are:

1. \(K_{\mathrm{ClassComplete}}^+\) sends every declared Type II candidate to
   the C5 ordered evaluator.
2. \(K_{\mathrm{TechTypeII}}^{\mathrm{blk},+}\) is the aggregate technical
   Type II package used by S4--S8 for blocked Type II conclusions; it excludes
   the promotion package, which C18 handles separately through C10.
3. \(K_{\mathrm{C6Route}}^+\) is the aggregate C6 no-radiation/tightness route
   emitting \(K_{\mathrm{RadBlk}}^+\) when its route payload is discharged.
4. \(K_{\mathrm{StateStratExh}}^+\) is the exhaustive terminal state-space
   partition, and \(K_{\mathrm{StratCritPacket}}^+\) is the local finite
   critical-mass packet certificate for retained active terminal strata. This
   supplies the terminal profile route's critical-mass input.
5. \(K_{\mathrm{RepBridge}}^+\) emits
   \(K_{\mathrm{TermSeqFromOrbit}}^+\) by S14, because terminal-camera
   construction is built into the declared S8 terminal backend.
6. \(K_{\mathrm{SmallDataStab}_{L^3}}^+\) is the small-data critical stability
   ledger used by S12.
7. \(K_{\mathrm{CriticalNSProfDecomp}}^+\) is the critical NS profile
   decomposition and nonlinear stability theorem accepted by S12.
8. These routed terminal-profile factors emit the bounded terminal stratum
   profile packet required by S12, hence
   \(K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+\), by U3b, S13, and S12.
9. \(K_{\mathrm{ScattBranch}}^-\) removes the scattering branch from the active
   Type II analysis.
10. \(K_{\mathrm{ExtRegDiscard}}^+\) discards physically exterior regular
   profiles.
11. \(K_{\mathrm{RepBridge}}^+\) supplies repaired-gauge representation.
12. \(K_{\mathrm{CaccioppoliReg}}^+\) supplies the Caccioppoli/windowed
   regularity input needed by the rough-core and S8 decoupling routes.
13. \(K_{\mathrm{S3NRSPayload}}^+\) blocks the scale-collapse generalized
   self-similar branch and supplies the S3 rigidity input used by S6--S8.
14. \(K_{\mathrm{NS\text{-}UPTypeII}}^+\) promotes a blocked Type II certificate
   to the suppressed scale-interface certificate.

::::

This package can be replaced by any stronger package that emits the same
abstract outputs \(K_{\mathrm{RadBlk}}^+\), \(K_{\mathrm{RoughCoreBlk}}^+\),
and \(K_{\mathrm{NS\text{-}UPTypeII}}^+\).

## Discharge of the abstract blockers from the terminal package

::::{prf:lemma} Terminal payload emits the radiative blocker
:label: lem-c18-terminal-payload-emits-radblk

Assume \(K_{\mathrm{UPTypeII},NS3D}^{\mathrm{term}}\). Then

```{math}
K_{\mathrm{RadBlk}}^+.
```

::::

:::{prf:proof}
By S14, the factor
\[
K_{\mathrm{RepBridge}}^+
\]
emit \(K_{\mathrm{TermSeqFromOrbit}}^+\). By U3b, S13, and S12, the factors
\[
K_{\mathrm{StateStratExh}}^+
\wedge
K_{\mathrm{StratCritPacket}}^+
\wedge
K_{\mathrm{SmallDataStab}_{L^3}}^+
\wedge
K_{\mathrm{CriticalNSProfDecomp}}^+
\]
emit
\(K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+\). Hence S8.11 applies to
the payload

```{math}
K_{\mathrm{TechTypeII}}^{\mathrm{blk},+}
\wedge
K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+
\wedge
K_{\mathrm{ScattBranch}}^-
\wedge
K_{\mathrm{ExtRegDiscard}}^+
\wedge
K_{\mathrm{RepBridge}}^+
\wedge
K_{\mathrm{CaccioppoliReg}}^+
\wedge
K_{\mathrm{S3NRSPayload}}^+
```

implies that there is no active multibubble Type II candidate. By S4.5, after
the same technical bridge payloads, exterior-regular discard, scale-collapse
removal by S3, and the S6--S8 multibubble discharges, multibubble failure is
an upstream defect rather than an additional singularity class. The remaining
non-bubble radiative/noncompact branch is exactly the C6 target. The terminal
package includes \(K_{\mathrm{C6Route}}^+\), and C6 proves
\(K_{\mathrm{C6Route}}^+\Rightarrow K_{\mathrm{RadBlk}}^+\). Therefore the
universal \(K_{\mathrm{RadBlk}}^+\), equivalently universal
\(K_{L^3\mathrm{Tight}}^+\), is emitted. \(\square\)
:::

::::{prf:lemma} Terminal payload emits the rough-core blocker
:label: lem-c18-terminal-payload-emits-roughcoreblk

Assume \(K_{\mathrm{UPTypeII},NS3D}^{\mathrm{term}}\). Then

```{math}
K_{\mathrm{RoughCoreBlk}}^+.
```

::::

:::{prf:proof}
The C7 rough-core composition emits \(K_{\mathrm{RoughCoreBlk}}^+\) once the
represented branch has the Caccioppoli/windowed regularity, pressure,
modulation, and bounded-critical-norm payloads. These are included in
(K_{mathrm{TechTypeII}}^{mathrm{blk},+}), (K_{mathrm{RepBridge}}^+), and
\(K_{\mathrm{CaccioppoliReg}}^+\) by Definition
{prf:ref}`def-c18-terminal-backend-final-payload`. Hence the universal
rough-core blocker is available. \(\square\)
:::

::::{prf:lemma} Terminal payload emits the abstract final package
:label: lem-c18-terminal-payload-emits-abstract-final-package

Assume \(K_{\mathrm{UPTypeII},NS3D}^{\mathrm{term}}\). Then

```{math}
K_{\mathrm{UPTypeII},NS3D}^{\mathrm{final}}.
```

::::

:::{prf:proof}
The terminal package includes \(K_{\mathrm{ClassComplete}}^+\) and
\(K_{\mathrm{NS\text{-}UPTypeII}}^+\) by definition. Lemma
{prf:ref}`lem-c18-terminal-payload-emits-radblk` gives
\(K_{\mathrm{RadBlk}}^+\), and Lemma
{prf:ref}`lem-c18-terminal-payload-emits-roughcoreblk` gives
\(K_{\mathrm{RoughCoreBlk}}^+\). These are exactly the four factors in
Definition {prf:ref}`def-c18-abstract-final-ns3d-uptypeii-package`. \(\square\)
:::

## Final expanded theorem

::::{prf:theorem} C18 terminal-backend `UP-TypeII` theorem for NS3D
:label: thm-c18-terminal-backend-uptypeii-ns3d

Assume

```{math}
K_{\mathrm{UPTypeII},NS3D}^{\mathrm{term}}.
```

Then, for every declared Type II candidate
\(\omega\in\mathcal U_{\mathrm{II}}^{NS}\),

```{math}
K_{\mathrm{SC}_\lambda}^{\sim}(\omega)
```

is emitted.

::::

:::{prf:proof}
By Lemma
{prf:ref}`lem-c18-terminal-payload-emits-abstract-final-package`, the abstract
final package \(K_{\mathrm{UPTypeII},NS3D}^{\mathrm{final}}\) holds. Apply
Theorem {prf:ref}`thm-c18-final-abstract-ns3d-uptypeii`. \(\square\)
:::

## Direct blocked-certificate variant

The theorem above uses C8. There is also a direct branchwise reading that is
often more intuitive:

1. compact single-core branches are blocked by the compact Type II barrier;
2. scale-collapse branches are blocked by S3, or by C17 if the scale-collapse
   cost bridge is registered;
3. multibubble/radiative branches are removed by S4--S8 under the terminal
   payload;
4. rough-core branches are blocked by C7;
5. C10 promotes every emitted \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\) to
   \(K_{\mathrm{SC}_\lambda}^{\sim}\) under
   \(K_{\mathrm{NS\text{-}UPTypeII}}^+\).

This direct reading is equivalent to the C8 assembly when the blockers are
emitted universally.

## Live terminal inputs

C18 is a final Type II theorem for the declared terminal backend. The live
inputs consumed at the C18 node are precisely the payloads in
Definition {prf:ref}`def-c18-terminal-backend-final-payload`.

In particular, the theorem still depends on:

1. declared-backend classification completeness,
2. terminal active-camera local compactness,
3. C6 no-radiation/tightness route,
4. scattering-branch removal,
5. exterior-regular discard,
6. repaired-gauge representation,
7. Caccioppoli/windowed regularity,
8. S3 scale-collapse rigidity,
9. NS-valid Type II promotion.

Once these are accepted as backend certificates, `UP-TypeII` is closed for all
declared NS3D Type II candidates in the terminal repaired-gauge backend.
