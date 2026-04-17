# S4: Type II reduction to multibubble residue

This note records the current NS3D-specific classification endpoint. It
separates technical bridge payloads from genuine structural Type II residues.
Once the bridge payloads and the closeable-strata discharge payloads are
supplied, every remaining Type II candidate is a multibubble candidate.

The result is conditional. It does not prove full Type II exclusion. It proves
that the unresolved residue is precisely secondary concentration: either a
same-point scale-separated bubble or a multi-point bubble.

---

## S4.1 (Technical payload package)

Define
\[
K_{\mathrm{TechTypeII}}^{\mathrm{blk},+}
\]
to be the conjunction of the technical NS3D payloads needed to run the compact
Type II machinery and emit blocked Type II certificates:

1. backend exhaustion:
   \[
   K_{\mathrm{TypeIIExhaust}}^+;
   \]
2. repaired-gauge representation:
   \[
   K_{\mathrm{RepBridge}}^+;
   \]
3. positive finite critical mass:
   \[
   K_{L^3\mathrm{Norm}}^+;
   \]
4. pressure reconstruction:
   \[
   K_{\mathrm{PressureRep}}^+;
   \]
5. repaired-gauge modulation control:
   \[
   K_{\mathrm{ModBd}}^+;
   \]
6. Caccioppoli regularity:
   \[
   K_{\mathrm{CaccioppoliReg}}^+;
   \]
7. cost compilation:
   \[
   K_{\mathrm{CostBridge}}^+;
   \]

If one of these payloads fails, the candidate is classified by the
corresponding bridge defect, not by the multibubble residue.

The promoted technical package is
\[
K_{\mathrm{TechTypeII}}^+
:=
K_{\mathrm{TechTypeII}}^{\mathrm{blk},+}
\wedge
K_{\mathrm{FormalUPTypeIIApp},NS3D}^+.
\]
Use \(K_{\mathrm{TechTypeII}}^{\mathrm{blk},+}\) when the conclusion is only
\(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\). Use
\(K_{\mathrm{TechTypeII}}^+\) only when this S4 route itself is asked to
produce post-UP suppression through the formal application package.

---

## S4.2 (Closeable non-multibubble strata)

Under \(K_{\mathrm{TechTypeII}}^{\mathrm{blk},+}\), the following apparent
survivor strata are closeable.

### Far radiation

The far-radiation stratum consists of radiative critical mass whose physical
radius stays bounded away from the blowup point:
\[
\lambda(\tau_n)R_n\ge r_*>0
\]
along a radiative annular sequence \(R_n\to\infty\). Under the single-point
blowup localization payload
\[
K_{\mathrm{SinglePointBlowup}}^+,
\]
this stratum is physically regular: the corresponding physical annuli remain
in a compact subset of spacetime away from the singular point. It is regular
exterior mass, not an unresolved Type II core pathology.

To remove this mass from the renormalized Type II survivor ledger one also
needs the exterior-regular discard payload
\[
K_{\mathrm{ExtRegDiscard}}^+.
\]
This payload asserts that critical mass carried by a physical region staying a
positive distance from the blowup point can be separated from the concentrating
Type II core and does not count as an unresolved core-radiation defect.

### Rough core

The rough-core stratum is closed by C6/T13:
\[
K_{\mathrm{RepBridge}}^+
\wedge
K_{L^3\mathrm{Bd}}^+
\wedge
K_{\mathrm{PressureRep}}^+
\wedge
K_{\mathrm{ModBd}}^+
\wedge
K_{\mathrm{CaccioppoliReg}}^+
\Longrightarrow
K_{\mathrm{WinH1}}^+.
\]
Hence, after the technical payload package supplies bounded critical mass,
pressure reconstruction, bounded modulation, and Caccioppoli regularity, the
rough-core defect \(K_{\mathrm{WinH1}}^-\) cannot survive.

### Scale-collapse drift

The scale-collapse drift stratum is closed once the S3 rigidity payload is
available. Denote this payload by
\[
K_{\mathrm{S3NRSPayload}}^+.
\]
It contains:

1. scale-collapse pigeonholing produces windows with negative average scale
   drift;
2. compactness and tightness extract a nonzero autonomous limit with constant
   drift parameter \(a_\infty<0\);
3. the stationary or invariant-measure extraction payload produces an
   \(L^3(\mathbb R^3)\) stationary profile of the corresponding self-similar
   reduced equation;
4. the parameter reduction putting the extracted stationary profile in the
   Nečas--Růžička--Šverák backward self-similar class;
5. the Nečas--Růžička--Šverák rigidity theorem ruling out every nonzero
   \(L^3\) profile in that class.

Thus the residual S3 input is the complete package that extracts a nonzero
stationary profile and verifies that its reduced equation is covered by the
NRŠ rigidity theorem. The nonzero conclusion uses the same positive
critical-mass plus tightness mechanism as the compact barrier.

---

## S4.3 (Multibubble residue)

After the technical payload package and the closeable strata in S4.2 are
discharged, the only remaining Type II residue is multibubble concentration.
S5 removes the regular strict same-point cascade subcase under its explicit
profile-decoupling, active-bubble mass-floor, inner-representation, and
perturbative S3 payloads. S5 also proves that the full nonlinear decoupling
payload \(K_{\mathrm{SamePointNLDec}}^+\) removes every same-point cascade.
S6 proves that the separated-point camera-decoupling payload
\(K_{\mathrm{MultiPointCamDec}}^+\) removes every multi-point cascade by
reducing each active physical point to its own single-camera S3 branch.
S7 proves that both decoupling payloads follow from the nonlinear
profile-evolution theorem
\[
K_{\mathrm{NLProfDec},NS3D}^+.
\]
S8 proves this theorem for terminal active cameras from profile completeness,
scattering removal, exterior-regular discard, repaired-gauge representation,
and Caccioppoli regularity.
At the S4/S6 reduction level, before importing S7 and S8, the multibubble
residue has two forms.

### Same-point secondary concentration

There are at least two concentration scales at the same physical blowup point:
\[
\lambda_1(t)\ll \lambda_2(t)\to0.
\]
In the primary renormalized coordinates, the secondary bubble appears as
radiation at radius
\[
R(t):=\frac{\lambda_2(t)}{\lambda_1(t)}\to\infty.
\]
Regular strict cascades of this form are excluded by S5, and all same-point
cascades are excluded under \(K_{\mathrm{SamePointNLDec}}^+\). Before S7/S8,
the same-point case that remains is \(K_{\mathrm{SamePointNLDec}}^-\), the
failure of nonlinear no-splitting/profile compatibility for scale-separated
profiles.

### Multi-point concentration

There are at least two physical concentration centers. In a renormalization
centered at one bubble, the other bubble appears as an escaping profile. This
multi-point case is excluded by S6 under \(K_{\mathrm{MultiPointCamDec}}^+\).
The remaining multi-point residue is exactly
\[
K_{\mathrm{MultiPointCamDec}}^-,
\]
the failure of separated-point camera decoupling, before S7/S8 are imported.

Both cases are multibubble. At the S4 level they are not removed by
pressure-tail control, repaired-gauge nondegeneracy, or the vorticity subtype
argument alone. They require the same-point or separated-point
no-splitting/profile compatibility payload, both of which are discharged by S7
from \(K_{\mathrm{NLProfDec},NS3D}^+\). S8 proves that payload in terminal
active cameras using profile completeness, exterior-regular discard,
repaired-gauge representation, and Caccioppoli regularity.

---

## S4.4 (Reduction theorem)

Assume a declared NS3D Type II candidate satisfies:

1. \(K_{\mathrm{TechTypeII}}^{\mathrm{blk},+}\);
2. \(K_{\mathrm{SinglePointBlowup}}^+\) and
   \(K_{\mathrm{ExtRegDiscard}}^+\) for the far-radiation discharge;
3. \(K_{\mathrm{S3NRSPayload}}^+\), so that scale-collapse drift is reduced
   to the NRŠ self-similar rigidity theorem;
4. the candidate is not a multibubble candidate in either same-point
   scale-separated or multi-point form.

Then the candidate emits the blocked compact Type II certificate
\[
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}.
\]
If the formal promotion payload in \(K_{\mathrm{TechTypeII}}^+\) is included,
then the candidate also emits the post-UP suppression certificate
\[
K_{\mathrm{SC}_\lambda}^{\sim}.
\]

### Proof

The technical payload package supplies representation, critical-mass control,
pressure reconstruction, bounded modulation, Caccioppoli regularity, and cost
compilation.

If the branch is globally \(L^3\)-tight, then C6/T13 gives
\[
K_{\mathrm{WinH1}}^+.
\]
Together with normalization and tightness, Theorem A'' gives infinite
localized Type II cost, and \(K_{\mathrm{CostBridge}}^+\) emits
\[
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}.
\]

If global \(L^3\)-tightness fails, the radiative branch is present. Far
radiation is discharged by
\[
K_{\mathrm{SinglePointBlowup}}^+
\wedge
K_{\mathrm{ExtRegDiscard}}^+,
\]
because it lies in the physically smooth exterior region and is separated from
the concentrating Type II core. S5 removes regular strict same-point cascades
under its explicit cascade payloads and removes all same-point cascades under
\(K_{\mathrm{SamePointNLDec}}^+\). S6 removes separated-point multibubbles
under \(K_{\mathrm{MultiPointCamDec}}^+\). The only remaining radiative
possibilities are \(K_{\mathrm{SamePointNLDec}}^-\) and
\(K_{\mathrm{MultiPointCamDec}}^-\), precisely the multibubble cases excluded
by assumption. Therefore the
noncompact/radiative branch is removed.

If scale-collapse drift occurs, \(K_{\mathrm{S3NRSPayload}}^+\) extracts a
nonzero \(L^3\) stationary profile in the NRŠ-covered self-similar class. NRŠ
forces every such profile to vanish, contradicting nonzero extraction. Hence
scale-collapse drift is removed.

The remaining rough-core branch is removed by the C6/T13 Caccioppoli bridge,
since \(K_{\mathrm{TechTypeII}}^{\mathrm{blk},+}\) includes the required pressure,
modulation, and regularity payloads.

All non-multibubble survivor branches are therefore exhausted. The compact
barrier emits \(K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}\). If
\(K_{\mathrm{FormalUPTypeIIApp},NS3D}^+\) is included, C10--C14 promote this
to \(K_{\mathrm{SC}_\lambda}^{\sim}\).

\(\square\)

---

## S4.5 (Current endpoint)

The current endpoint of the folder is:
\[
\text{NS3D Type II}
\quad\Longrightarrow\quad
\text{technical bridge defect}
\quad\text{or}\quad
\text{multibubble residue}
\quad\text{or}\quad
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}.
\]
With the formal promotion payload:
\[
K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}
\wedge
K_{\mathrm{FormalUPTypeIIApp},NS3D}^+
\Longrightarrow
K_{\mathrm{SC}_\lambda}^{\sim}.
\]

Thus, after the technical bridge payloads, the S5 regular cascade discharge
payloads, and S6 camera reduction are supplied, the two multibubble defects at
the S6 level are:

1. \(K_{\mathrm{SamePointNLDec}}^-\);
2. \(K_{\mathrm{MultiPointCamDec}}^-\).

By S7 these are both consequences of the nonlinear profile-evolution
obstruction
\[
K_{\mathrm{NLProfDec},NS3D}^-.
\]
By S8, this obstruction is discharged in terminal active cameras by
\(K_{\mathrm{ProfComplete},NS3D,\mathrm{term}}^+\), scattering removal,
\(K_{\mathrm{ExtRegDiscard}}^+\), \(K_{\mathrm{RepBridge}}^+\), and
\(K_{\mathrm{CaccioppoliReg}}^+\). Hence a
multibubble failure is an upstream profile-completeness, representation,
Caccioppoli, or S3-rigidity defect, not an additional singularity class.
