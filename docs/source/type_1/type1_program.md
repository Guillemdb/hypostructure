# Type I Singularity Exclusion Program for NS3D

## A PDE Paper Series

This document reorganizes the Type I singularity program as a sequence of
traditional PDE papers.  Papers I--VII are written in standard analysis style:
abstract, main theorems, auxiliary lemmas, proof architecture, and open points.
All formal bookkeeping is deferred to the final standalone paper in the series.

The mathematical correction from `drafts.md` is built into the series:
centered Type I ancient limits are not Type II same-point cascades.  The
renormalized Type I equation has a fixed self-similar scale and center, so the
program is organized around ancient dynamics, compactness, extremizers, and
rigidity, not camera-on-innermost cascade reduction.

The intended mathematical dependency chain is:
$$
\mathrm{I}
\Longrightarrow
\mathrm{II}
\Longrightarrow
\mathrm{III}
\Longrightarrow
\mathrm{IV}
\Longrightarrow
\mathrm{V}
\Longrightarrow
\mathrm{VI}
\Longrightarrow
\mathrm{VII}
\Longrightarrow
\mathrm{VIII}.
$$

The papers are:

1. **Paper I. Type I Blow-up Limits and the Centered Ancient Equation.**
2. **Paper II. Tight Ancient Classes and Stationary Self-similar Rigidity.**
3. **Paper III. Minimal Ancient Solutions and the Extremal Reduction.**
4. **Paper IV. Compact Critical Elements and Invariant Measures.**
5. **Paper V. Equilibrium Rigidity for Compact Ancient Dynamics.**
6. **Paper VI. Structured Ancient Solutions: Symmetry, Swirl, and Decay.**
7. **Paper VII. Seregin Limits and Full Type I Exclusion.**
8. **Paper VIII. Formal Translation of the Series.**

---

## Common Notation

The Navier-Stokes equations on $\mathbb{R}^3$ are
$$
\partial_t u+(u\cdot\nabla)u+\nabla p=\Delta u,\qquad
\nabla\cdot u=0.
$$

A suitable Leray-Hopf solution
$u:\mathbb{R}^3\times[0,T^*)\to\mathbb{R}^3$ has a Type I singularity at
$(x_*,T^*)$ if
$$
\limsup_{t\uparrow T^*}\sqrt{T^*-t}\,
\|u(t)\|_{L^\infty(\mathbb{R}^3)}<\infty
$$
and $(x_*,T^*)$ is nevertheless singular.

For an ancient solution $U$ on $\mathbb{R}^3\times(-\infty,0)$, the centered
self-similar variables are
$$
y=\frac{x}{\sqrt{-t}},\qquad
\tau=-\log(-t),\qquad
V(y,\tau)=\sqrt{-t}\,U(y\sqrt{-t},t).
$$
Then $V$ solves
$$
\partial_\tau V+(V\cdot\nabla)V+\nabla\Pi
=\Delta V-\frac12(V+y\cdot\nabla V),
\qquad \nabla\cdot V=0.
$$

The renormalized equation is autonomous, but the drift term fixes the Type I
scale and center.  In particular, the map
$$
V(y,\tau)\mapsto \alpha V(\alpha y,\alpha^2\tau)
$$
does not preserve the equation for any positive $\alpha\ne1$, and constant
translations in $y$ are not symmetries.  This is the structural reason the Type
II cascade formalism is not imported into this Type I program.

An admissible ancient class $\mathcal X$ is a class of renormalized ancient
solutions satisfying:

- local smooth compactness after pressure gauges are fixed;
- invariance under time translation and rotation;
- a fixed nontriviality normalization, such as
  $$
  \mathcal N_{R_0}(V)
  :=
  \sup_{\tau\in\mathbb{R}}
  \int_{B_{R_0}}|V(y,\tau)|^3\,dy
  \ge \eta_0;
  $$
- enough decay or tightness to pass stationary limits into the known
  stationary rigidity theory, unless the paper explicitly studies a
  non-decaying class.

The main examples are:

- $\mathcal X_{L^3\mathrm{-tight}}$: uniformly $L^3$-bounded and uniformly
  $L^3$-tight in $y$;
- $\mathcal X_{\mathrm{fast}}$: weighted or pointwise fast decay;
- $\mathcal X_{\mathrm{axi},0}$: axisymmetric without swirl;
- $\mathcal X_{\mathrm{axi},sw}$: axisymmetric with controlled swirl;
- $\mathcal X_{\mathrm{Ser}}$: ancient limits actually produced by the Type I
  blow-up reduction.

---

## Paper I. Type I Blow-up Limits and the Centered Ancient Equation

### Proposed title

**Type I Blow-up Limits for Navier-Stokes and the Centered Self-similar Ancient
Equation**

### Abstract

This paper establishes the reduction from a Type I singularity of a suitable
weak Navier-Stokes solution to a nontrivial bounded ancient solution of the
centered self-similar equation.  It also records the basic structural fact that
the centered Type I equation has no nontrivial scaling symmetry, so the
multi-scale cascade analysis used in Type II regimes is not available in this
setting.

### Main results

**Theorem 1.1 (Type I blow-up limit).**  Let $u$ be a suitable Leray-Hopf
solution on $\mathbb{R}^3\times[0,T^*)$ with a Type I singularity at
$(x_*,T^*)$.  Then there exist $\lambda_k\downarrow0$ such that the rescaled
solutions
$$
u_k(x,t)=\lambda_k
u(x_*+\lambda_k x,T^*+\lambda_k^2t)
$$
converge locally, after passing to a subsequence and choosing pressure gauges,
to an ancient suitable weak solution $U$ on
$\mathbb{R}^3\times(-\infty,0)$.  Moreover
$$
\sup_{t<0}\sqrt{-t}\,\|U(t)\|_{L^\infty(\mathbb{R}^3)}<\infty.
$$

**Theorem 1.2 (Nontriviality of the ancient limit).**  If $(x_*,T^*)$ is a
genuine singular point, the sequence in Theorem 1.1 can be chosen so that the
renormalized ancient solution $V$ satisfies a fixed local nontriviality bound
$$
\sup_{\tau\in\mathbb{R}}\int_{B_{R_0}}|V(y,\tau)|^3\,dy\ge\eta_0
$$
for some $R_0,\eta_0>0$.

**Proposition 1.3 (Renormalized equation).**  The field
$$
V(y,\tau)=\sqrt{-t}\,U(y\sqrt{-t},t),
\qquad \tau=-\log(-t),
$$
is a bounded ancient solution of
$$
\partial_\tau V+(V\cdot\nabla)V+\nabla\Pi
=\Delta V-\frac12(V+y\cdot\nabla V),
\qquad \nabla\cdot V=0,
$$
on $\mathbb{R}^3\times\mathbb{R}$.

**Proposition 1.4 (Absence of Type II cascade symmetry).**  The centered
self-similar equation is not invariant under nontrivial positive scalings or
constant translations in $y$.  Consequently, Type I ancient limits do not carry
the same same-point scale-cascade structure as Type II blow-up sequences.

### Proof architecture

Section 1 states the Type I assumptions and recalls suitable weak solutions and
local energy inequalities.  Section 2 proves uniform local energy and pressure
bounds for the rescaled sequence.  Section 3 passes to the ancient limit.
Section 4 proves nontriviality by contradiction using a local regularity
criterion.  Section 5 derives the centered renormalized equation and the
no-scaling observation.

### Role in the series

Paper I supplies the basic object for every later paper: a nonzero bounded full
orbit of the centered self-similar Navier-Stokes flow.

### Status

This paper is closeable with standard compactness, pressure, and
epsilon-regularity tools.  The only delicate point is the clean formulation of
the nontriviality normalization.

---

## Paper II. Tight Ancient Classes and Stationary Self-similar Rigidity

### Proposed title

**Tight Ancient Navier-Stokes Solutions and Stationary Self-similar Rigidity**

### Abstract

This paper isolates the ancient classes in which stationary self-similar
rigidity can be applied.  The natural first target is a uniformly tight
$L^3$ class, or a stronger fast-decay class.  In these classes, small bounded
ancient solutions vanish and stationary limits fall under the
Nečas-Růžička-Šverák theorem.

### Main results

**Theorem 2.1 (Regularity and pressure gauges).**  Every bounded Type I ancient
candidate produced in Paper I is smooth on compact subsets of
$\mathbb{R}^3\times(-\infty,0)$, and its renormalized representative is smooth
on $\mathbb{R}^3\times\mathbb{R}$.  Pressure gauges may be chosen so that
Calderón-Zygmund bounds are stable under local limits.

**Proposition 2.2 (Critical mass transfer under an explicit hypothesis).**  If
the original solution satisfies
$$
\sup_{t<T^*}\|u(t)\|_{L^3(\mathbb{R}^3)}<\infty,
$$
then the ancient limit satisfies
$$
\sup_{\tau\in\mathbb{R}}\|V(\tau)\|_{L^3(\mathbb{R}^3)}<\infty.
$$
This result is included for bookkeeping only; under the same global
$L^\infty_tL^3_x$ assumption, blow-up is already ruled out by
Escauriaza-Seregin-Šverák.

**Definition 2.3 (The tight ancient class).**  Define
$$
\mathcal X_{L^3\mathrm{-tight}}
=
\left\{
V:
\sup_\tau\|V(\tau)\|_{L^3}<\infty,\quad
\sup_\tau\int_{|y|>R}|V(y,\tau)|^3\,dy\to0
\right\}.
$$
The convergence as $R\to\infty$ is part of the class definition.

**Theorem 2.4 (Small ancient solutions vanish).**  There exists
$\varepsilon_0>0$ such that if
$V\in\mathcal X_{L^3\mathrm{-tight}}$ solves the centered self-similar equation
and
$$
\|V\|_{L^\infty_{\tau,y}}\le\varepsilon_0,
$$
then $V\equiv0$.

The formal energy identity behind the theorem is
$$
\frac{d}{d\tau}\int |V|^2G
=
-2\int|\nabla V|^2G
-\int|V|^2G
-\frac12\int(|V|^2+2\Pi)(V\cdot y)G,
\qquad
G=e^{-|y|^2/4}.
$$

**Theorem 2.5 (Stationary self-similar rigidity).**  If
$W\in L^3(\mathbb{R}^3)$ solves
$$
(W\cdot\nabla)W+\nabla\Pi
=\Delta W-\frac12(W+y\cdot\nabla W),
\qquad \nabla\cdot W=0,
$$
then $W\equiv0$.

This is the Nečas-Růžička-Šverák theorem, imported as a black-box stationary
Liouville theorem.

### Proof architecture

Section 1 fixes the tight and fast-decay classes.  Section 2 proves smoothness
and pressure stability.  Section 3 proves the Gaussian energy estimate and the
small-solution Liouville theorem.  Section 4 records the stationary
self-similar theorem and explains exactly which decay assumptions are needed
for later stationary limits.

### Role in the series

Paper II supplies the first class in which the extremal program can be run:
$\mathcal X_{L^3\mathrm{-tight}}$.  It also gives a positive lower bound for
the size of any nonzero ancient solution in that class.

### Status

Closeable in tight or fast-decay classes.  It does not claim that every
Seregin ancient limit is tight.

---

## Paper III. Minimal Ancient Solutions and the Extremal Reduction

### Proposed title

**Minimal Bounded Ancient Solutions for the Centered Navier-Stokes Flow**

### Abstract

This paper develops the reduction from arbitrary nonzero ancient solutions in a
chosen admissible class to minimal ancient solutions.  The main theorem is an
extremizer-or-defect alternative for minimizing sequences.  Once the defect
branches are discharged, the Liouville problem reduces to ruling out minimal
ancient solutions.

### Main definitions

Let $\mathcal X$ be an admissible ancient class and let $\mathcal M$ be a lower
semicontinuous size functional, for example
$$
\mathcal M(V)=\|V\|_{L^\infty_{\tau,y}}
$$
or a stronger tight critical norm.  Define
$$
m_{\mathcal X}
=
\inf\{
\mathcal M(V):
V\in\mathcal X,\ \mathcal N_{R_0}(V)\ge\eta_0
\}.
$$
A **minimal ancient solution** in $\mathcal X$ is a nonzero normalized solution
$V_*$ with $\mathcal M(V_*)=m_{\mathcal X}$.

### Main results

**Theorem 3.1 (Extremizer-or-defect alternative).**  Let
$V_n\in\mathcal X$ satisfy
$$
\mathcal N_{R_0}(V_n)\ge\eta_0,\qquad
\mathcal M(V_n)\downarrow m_{\mathcal X}.
$$
After passing to a subsequence and applying admissible time shifts and
rotations, exactly one of the following occurs:

- compactness: $V_n$ converges to a nonzero minimal ancient solution
  $V_*\in\mathcal X$;
- vanishing: local mass vanishes on every fixed ball;
- escape: nontrivial mass escapes to spatial infinity in the centered
  variables;
- splitting: the sequence decomposes into two or more nonzero nonlinear ancient
  profiles;
- Reynolds defect: a weak limit solves a Navier-Stokes-Reynolds system rather
  than the exact equation;
- loss of class: the limit exits the imposed tightness, decay, suitability, or
  pressure-gauge class.

**Theorem 3.2 (Defect discharge in the tight class).**  In
$\mathcal X_{L^3\mathrm{-tight}}$, vanishing and escape contradict the
normalization and tightness assumptions.  Splitting either produces a smaller
nonzero normalized ancient solution or contradicts the strict minimality after
the relevant decoupling inequality is proved.  Loss of class is impossible by
closedness of the tight class.

**Theorem 3.3 (Reduction to minimal solutions).**  Assume Theorems 3.1 and 3.2
hold in $\mathcal X$.  If no minimal ancient solution exists in $\mathcal X$,
then $\mathcal X$ contains no nonzero normalized ancient solution.

### Proof architecture

Section 1 sets up the minimizing problem.  Section 2 proves local compactness
of minimizing sequences.  Section 3 proves the defect alternative.  Section 4
uses tightness and lower semicontinuity to discharge vanishing, escape,
splitting, and loss-of-class defects.  Section 5 proves the reduction to the
minimal case.

### Role in the series

Paper III is the concentration-compactness part of the program.  It replaces
the obsolete cascade reduction with a minimal-counterexample reduction.

### Status

This is the first substantial research paper in the series.  The statement is
standard in form, but adapting it to bounded ancient Navier-Stokes solutions is
not a formal import from dispersive Kenig-Merle theory.

---

## Paper IV. Compact Critical Elements and Invariant Measures

### Proposed title

**Compact Critical Elements for the Centered Ancient Navier-Stokes Flow**

### Abstract

This paper studies minimal ancient solutions produced by Paper III.  The goal
is to prove that minimal ancient solutions have compact recurrent dynamics and
therefore generate invariant probability measures for the renormalized flow.
The main technical issue is to ensure that averaging does not produce a
Reynolds stress defect.

### Main results

**Theorem 4.1 (Precompactness of minimal orbits).**  Let $V_*$ be a minimal
ancient solution in $\mathcal X_{L^3\mathrm{-tight}}$.  Then the time-slice
orbit
$$
\mathcal O(V_*)=\{V_*(\cdot,\tau):\tau\in\mathbb{R}\}
$$
is precompact in the topology of $\mathcal X_{L^3\mathrm{-tight}}$, after
passing if necessary to a minimal compact invariant subset of its closure.

Equivalently, for every $\varepsilon>0$ there are finitely many profiles
$W_1,\dots,W_N$ such that
$$
\min_{1\le j\le N}
\operatorname{dist}_{\mathcal X}(V_*(\cdot,\tau),W_j)<\varepsilon
\qquad \forall \tau\in\mathbb{R}.
$$

**Theorem 4.2 (Invariant measures).**  The compact orbit closure
$\mathcal K$ of a minimal ancient solution supports an invariant probability
measure obtained as a weak limit of
$$
\mu_T
=
\frac1T\int_0^T\delta_{V_*(\cdot,s)}\,ds.
$$

**Theorem 4.3 (No Reynolds stress for extremal averages).**  Any invariant
statistical limit generated by a minimal ancient solution in the tight class is
compatible with the exact stationary Navier-Stokes equation.  Equivalently, the
averaging process does not leave an unresolved Reynolds stress term in the
stationary limit.

**Lemma 4.4 (Nonzero support).**  The invariant measure associated to a
minimal ancient solution is not supported at the zero solution.  If the zero
solution belongs to a minimal compact invariant set, then the entire set is
$\{0\}$, contradicting the nontriviality normalization.

### Proof architecture

Section 1 proves compactness of minimal orbits from the extremality obtained in
Paper III.  Section 2 constructs invariant measures by Krylov-Bogolyubov
averaging.  Section 3 analyzes nonlinear averages and rules out Reynolds
stress defects.  Section 4 proves the nonzero-support lemma.

### Role in the series

Paper IV turns minimal ancient solutions into compact dynamical objects.  This
sets up Paper V, where compact invariant dynamics are shown to be stationary.

### Status

The invariant measure construction is standard once compactness is known.  The
hard parts are compactness of minimal orbits and the no-Reynolds theorem.

---

## Paper V. Equilibrium Rigidity for Compact Ancient Dynamics

### Proposed title

**Equilibrium Rigidity for Compact Ancient Navier-Stokes Dynamics**

### Abstract

This paper proves the central rigidity statement of the Type I program:
compact minimal ancient dynamics in the centered self-similar variables are
supported on stationary solutions.  Combined with the stationary
Nečas-Růžička-Šverák theorem, this rules out minimal ancient solutions in the
tight class and gives conditional Type I exclusion.

### Main results

**Theorem 5.1 (Equilibrium support).**  Let $\mu$ be an invariant probability
measure generated by a compact minimal ancient solution in
$\mathcal X_{L^3\mathrm{-tight}}$.  Then
$$
\mu\bigl(\{W:\partial_\tau W=0\}\bigr)=1.
$$
Equivalently, compact minimal ancient dynamics in the tight class cannot be
genuinely periodic, quasi-periodic, or recurrent nonstationary orbits.

**Theorem 5.2 (No minimal tight ancient solutions).**  There is no nonzero
minimal ancient solution in $\mathcal X_{L^3\mathrm{-tight}}$.

Indeed, Theorem 5.1 and Paper IV produce nonzero stationary $L^3$ support,
while the stationary self-similar theorem from Paper II forces every such
stationary profile to vanish.

**Theorem 5.3 (Ancient Liouville theorem in the tight class).**  Every bounded
ancient solution in $\mathcal X_{L^3\mathrm{-tight}}$ satisfying the fixed
nontriviality normalization is identically zero.

**Theorem 5.4 (Conditional Type I exclusion).**  If every ancient limit
produced by the Type I reduction in Paper I belongs to
$\mathcal X_{L^3\mathrm{-tight}}$, then no Type I singularity occurs in that
class.

### Possible proof mechanisms for Theorem 5.1

- Construct a Lyapunov functional whose dissipation vanishes exactly at
  equilibria.
- Strengthen the Gaussian identity from Paper II using compactness of the
  minimal orbit.
- Show that any nonstationary recurrent orbit generates a smaller ancient
  solution, contradicting minimality.
- Upgrade invariant statistical solutions to exact stationary solutions using
  the no-Reynolds theorem from Paper IV.

### Proof architecture

Section 1 studies the invariant measure from Paper IV.  Section 2 proves an
equilibrium-support theorem.  Section 3 applies stationary self-similar
rigidity.  Section 4 combines Papers III--V to prove the tight-class ancient
Liouville theorem and the conditional Type I exclusion theorem.

### Role in the series

Paper V is the decisive rigidity paper for the tight-class program.

### Status

Theorem 5.1 is the central open problem in the series.  Once it is proved, the
rest of Paper V is a clean assembly.

---

## Paper VI. Structured Ancient Solutions: Symmetry, Swirl, and Decay

### Proposed title

**Structured Bounded Ancient Solutions of the Three-dimensional Navier-Stokes
Equations**

### Abstract

This paper applies the previous reductions, and in some cases bypasses them, in
classes with additional structure: axisymmetry without swirl, axisymmetry with
controlled swirl, and fast spatial decay.  These structured classes provide
intermediate Type I exclusion results and test cases for the general program.

### Part A: Axisymmetric no-swirl solutions

**Theorem 6.1 (No-swirl ancient Liouville).**  A bounded ancient suitable weak
solution that is axisymmetric without swirl is trivial under the hypotheses of
Koch-Nadirashvili-Seregin-Šverák.

**Corollary 6.2 (Type I exclusion in the no-swirl branch).**  A Type I
singularity whose ancient limit belongs to the axisymmetric no-swirl class is
excluded.

### Part B: Controlled swirl

**Theorem 6.3 (Controlled-swirl target).**  Let $U$ be an axisymmetric bounded
ancient solution with swirl satisfying a quantitative decay or smallness
condition, for instance
$$
|u_\theta(r,z,t)|\le C r^{-\gamma}
$$
in a range sufficient to close the cylindrical estimates.  Then $U$ is trivial.

The exact admissible range for $\gamma$ and the replacement hypotheses should
be stated as part of the paper, not hidden in the roadmap.

**Corollary 6.4 (Type I exclusion in the controlled-swirl branch).**  Type I
singularities whose ancient limits satisfy Theorem 6.3 are excluded.

### Part C: Fast decay

**Theorem 6.5 (Fast-decay ancient Liouville target).**  Bounded ancient
solutions with sufficiently strong pointwise or weighted decay are trivial.
The proof may proceed either through the extremal program of Papers III--V or
through a direct weighted energy/backward-uniqueness argument.

**Corollary 6.6 (Type I exclusion in the fast-decay branch).**  Type I
singularities whose ancient limits satisfy the fast-decay hypotheses are
excluded.

### Proof architecture

The no-swirl part is literature-driven.  The controlled-swirl part develops
cylindrical estimates and identifies the swirl hypotheses needed for closure.
The fast-decay part uses weighted energy identities, stationary self-similar
rigidity, and possibly the minimal-element machinery from Papers III--V.

### Role in the series

Paper VI broadens the program beyond the generic tight class.  It also gives
partial Type I exclusion theorems even if the generic equilibrium-rigidity
theorem in Paper V remains open.

### Status

The no-swirl branch is closeable by importing known Liouville theory.  The
controlled-swirl and fast-decay branches are genuine research projects.

---

## Paper VII. Seregin Limits and Full Type I Exclusion

### Proposed title

**Seregin Ancient Limits and Exclusion of Type I Navier-Stokes Singularities**

### Abstract

This paper attempts to close the gap between conditional ancient Liouville
theorems and full Type I exclusion.  The problem is to prove that every ancient
limit produced by a finite-energy Type I singularity lies in one of the
classes closed by Papers V and VI, or to introduce a replacement rigidity
mechanism for the residual non-tight class.

### Main results

**Theorem 7.1 (Routing of Seregin ancient limits).**  Every nonzero ancient
solution produced by the Type I reduction in Paper I satisfies at least one of:

- it belongs to $\mathcal X_{L^3\mathrm{-tight}}$;
- it belongs to $\mathcal X_{\mathrm{fast}}$;
- it belongs to a structured class treated in Paper VI;
- it belongs to an explicitly described residual non-tight/non-decaying class.

**Theorem 7.2 (No residual Seregin branch).**  The residual branch in
Theorem 7.1 is empty for finite-energy Type I blow-up limits.

This is the major new input needed for full Type I exclusion.  It requires
decay or tightness information for Seregin ancient limits without assuming a
global $L^\infty_tL^3_x$ bound, or else a rigidity theorem that applies without
tightness.

**Theorem 7.3 (Full Type I exclusion).**  If Theorems 7.1 and 7.2 hold, and the
relevant branches from Papers V and VI are closed, then no finite-energy
solution of the three-dimensional Navier-Stokes equations develops a Type I
singularity.

### Proof architecture

Section 1 reviews the ancient limits produced by Paper I.  Section 2 proves
all decay, localization, and tightness inherited from finite energy and
suitability.  Section 3 proves the routing theorem.  Section 4 eliminates the
residual class.  Section 5 assembles the full Type I exclusion theorem.

### Role in the series

Paper VII is the endpoint for the PDE program.  It converts conditional
ancient Liouville theorems into a full Type I regularity statement.

### Status

Aspirational.  The residual Seregin-class tightness problem is not known to be
closeable by the earlier papers alone.

---

## Practical Publication Order

1. Write Paper I first, because it fixes the correct ancient object and removes
   the cascade misframing.
2. Write Paper II next for the tight and fast-decay classes.
3. Develop Paper III in the tight class, where the minimal reduction has the
   cleanest chance of closing.
4. Write Paper IV once the minimal class is sufficiently compact.
5. Split Paper V if needed: one paper for conditional assembly and one for the
   equilibrium-support theorem.
6. Publish the no-swirl branch of Paper VI as a clean application if it is
   useful; treat controlled swirl and fast decay as separate research projects.
7. Attempt Paper VII only after the Seregin tightness or residual-class problem
   is understood.
8. Write Paper VIII last, after the PDE statements have stabilized.

---

## Honest Status

**Reliable or closeable with standard tools.**

- Paper I: rescaling, local compactness, centered equation, no-cascade
  correction.
- Paper II: smoothness, pressure gauges, stationary NRŠ input.
- Paper VI-A: axisymmetric no-swirl branch under the precise KNSŠ hypotheses.

**Closeable but technical.**

- Paper I nontriviality normalization.
- Paper II small-amplitude Liouville in tight or fast-decay classes.
- Paper III minimal reduction in a deliberately chosen critical/tight class.

**Hard research problems.**

- Paper III defect discharge in the bare $L^\infty_{\tau,y}$ setting.
- Paper IV no-Reynolds theorem for extremal averages.
- Paper V equilibrium-support rigidity for compact ancient dynamics.
- Paper VI-B controlled swirl beyond existing partial results.
- Paper VI-C fast-decay non-symmetric ancient Liouville in the desired range.
- Paper VII residual Seregin-class tightness or replacement rigidity.

The series is useful even before Paper VII is closed: Papers I--V yield a
precise conditional Type I theorem in tight classes, Paper VI gives structured
applications, and the final standalone paper records the formal dependency map.

---

## Paper VIII. Certificate Bookkeeping and Hypostructure Translation

### Proposed title

**Hypostructure Certificates for the Type I Navier-Stokes Program**

### Abstract

This standalone paper translates the PDE series into the hypostructure
framework.  It assigns certificates to the main theorems of Papers I--VII,
records their dependency graph, and identifies exactly which analytic results
are imported, proved, conditional, or open.  No analytic proof in Papers
I--VII depends on this formal translation.

### Certificate dictionary

Paper I exports:
$$
K_{\mathrm{P1\text{-}TypeIToAncient}}^+,\qquad
K_{\mathrm{P1\text{-}CenteredRenorm}}^+,\qquad
K_{\mathrm{P1\text{-}NoCascade}}^+.
$$

Paper II exports:
$$
K_{\mathrm{P2\text{-}TightClass}}^+,\qquad
K_{\mathrm{P2\text{-}SmallGap}}^+,\qquad
K_{\mathrm{P2\text{-}NRSInput}}^+.
$$

Paper III exports:
$$
K_{\mathrm{P3\text{-}ExtDefAlt}}^+,\qquad
K_{\mathrm{P3\text{-}DefectDischarge}}^+,\qquad
K_{\mathrm{P3\text{-}MinimalReduction}}^+.
$$

Paper IV exports:
$$
K_{\mathrm{P4\text{-}CompactMinimalOrbit}}^+,\qquad
K_{\mathrm{P4\text{-}InvariantMeasure}}^+,\qquad
K_{\mathrm{P4\text{-}NoReynolds}}^+,\qquad
K_{\mathrm{P4\text{-}NonzeroSupport}}^+.
$$

Paper V exports:
$$
K_{\mathrm{P5\text{-}EquilibriumSupport}}^+,\qquad
K_{\mathrm{P5\text{-}TightAncientLiouville}}^+,\qquad
K_{\mathrm{P5\text{-}ConditionalTypeIExclusion}}^+.
$$

Paper VI exports one certificate for each structured branch:
$$
K_{\mathrm{P6A\text{-}NoSwirl}}^+,\qquad
K_{\mathrm{P6B\text{-}ControlledSwirl}}^+,\qquad
K_{\mathrm{P6C\text{-}FastDecay}}^+.
$$

Paper VII exports:
$$
K_{\mathrm{P7\text{-}SereginRouting}}^+,\qquad
K_{\mathrm{P7\text{-}NoResidualBranch}}^+,\qquad
K_{\mathrm{P7\text{-}FullTypeIExclusion}}^+.
$$

### Hypostructure dependency graph

The tight-class conditional branch is:
$$
K_{\mathrm{P1\text{-}TypeIToAncient}}^+
\wedge
K_{\mathrm{P2\text{-}TightClass}}^+
\wedge
K_{\mathrm{P3\text{-}MinimalReduction}}^+
\wedge
K_{\mathrm{P4\text{-}CompactMinimalOrbit}}^+
\wedge
K_{\mathrm{P4\text{-}NoReynolds}}^+
\wedge
K_{\mathrm{P5\text{-}EquilibriumSupport}}^+
\wedge
K_{\mathrm{P2\text{-}NRSInput}}^+
\Longrightarrow
K_{\mathrm{P5\text{-}ConditionalTypeIExclusion}}^+.
$$

The full Type I branch is:
$$
K_{\mathrm{P5\text{-}ConditionalTypeIExclusion}}^+
\wedge
K_{\mathrm{P6A\text{-}NoSwirl}}^+
\wedge
K_{\mathrm{P6B\text{-}ControlledSwirl}}^+
\wedge
K_{\mathrm{P6C\text{-}FastDecay}}^+
\wedge
K_{\mathrm{P7\text{-}SereginRouting}}^+
\wedge
K_{\mathrm{P7\text{-}NoResidualBranch}}^+
\Longrightarrow
K_{\mathrm{P7\text{-}FullTypeIExclusion}}^+.
$$

### Open certificate slots

The currently open or research-level certificates are:

- $K_{\mathrm{P3\text{-}DefectDischarge}}^+$ in the bare
  $L^\infty_{\tau,y}$ class;
- $K_{\mathrm{P4\text{-}NoReynolds}}^+$ for extremal invariant averages;
- $K_{\mathrm{P5\text{-}EquilibriumSupport}}^+$ in the generic tight class;
- $K_{\mathrm{P6B\text{-}ControlledSwirl}}^+$ beyond known swirl regimes;
- $K_{\mathrm{P6C\text{-}FastDecay}}^+$ in the desired non-symmetric range;
- $K_{\mathrm{P7\text{-}NoResidualBranch}}^+$ for actual Seregin limits.

### Purpose of this final paper

Paper VIII is not a PDE proof paper.  Its role is to make the logical
interfaces machine-checkable inside the hypostructure framework, preserve the
distinction between proved and conditional inputs, and ensure that future
updates to one paper propagate cleanly through the dependency graph.
