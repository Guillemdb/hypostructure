# Three-Dimensional Navier--Stokes Local Blow-Up Analysis

This document records a step-by-step local blow-up analysis for the
three-dimensional incompressible Navier--Stokes equations in standard PDE
terminology. Each proof step is a local lemma, local estimate, finite-cover
argument, state-space ledger, or explicitly imported endpoint theorem. The
named steps

$$
H0,\ D_E,\ Rec_N,\ C_\mu,\ PS1,\ldots
$$

are only identifiers for the order of the argument; the mathematical content is
given by the stated estimates, compactness statements, reductions,
state-space ledgers, and explicitly imported endpoint theorems. In particular,
a node is not allowed to disguise a global regularity-scale hypothesis as a
single local estimate.

This file treats the pre-entry data, the profile analysis, the local
state-space residual closure, the compatibility checks, and the final local
exclusion record:

$$
H0,\ D_E,\ Rec_N,\ C_\mu,\ PS1,\ldots,\ PS30,\ ST0,\ldots,\ ST20,\ PS31,\ldots,\ PS35,\ Bound\_\partial,\ Bound_B,\ Bound_\Sigma,\ GC_T,\ FinalExcl.
$$

The main local routing diagram is:

$$
\begin{array}{ccccccccc}
H0&\to&D_E&\to&Rec_N&\to&C_\mu&\to&PS1\\
&&&&&&&&\downarrow\\
&&&&&&&&PS2\to PS3\to PS4\to PS5\\
&&&&&&&&\downarrow\\
&&&&&&&&PS6\to PS7\to PS8\\
&&&&&&&&\downarrow\\
&&&&&&&&
\begin{array}{c}
PS9\text{ Type I profile}\\
PS10\text{ raw Type II branch}
\end{array}
\end{array}
$$

The Type I terminal/tail row is localized as

$$
PS15
\quad\leadsto\quad
PS15a\to PS15b\to\cdots\to PS15n,
$$

where the subnodes convert non-tightness into covariant local observer states,
derive terminal indecomposability, assemble the backward sequence-$L^3$ input
from local alternatives, and assemble the finite-shift Duhamel/mildness input
from a residual ledger. Only after those inputs have been assembled may the
endpoint ancient theorem be imported and applied. Thus `PS15` is not a global
tightness assumption on the sieve; it is a branch point whose non-tight side
enters the local state-space closure.

The Type II row is localized as

$$
PS10,PS11
\quad\leadsto\quad
TII0\to TII1\to\cdots\to TII16.
$$

The nodes `TII0`--`TII16` replace global profile budgets, global far-field
tails, and whole-space pressure reconstructions by compact-window CKN budgets,
local pressure decompositions, repaired gauges, parabolic active-frame
classification, compact-window decoupling, and local state-space alternatives.
The older packet nodes `PS18`--`PS20` remain available only after their inputs
have been produced locally by the `TII` block, or in a separate non-Type-II
branch that explicitly supplies a theorem-matching global profile package.

Each named step is either a local verification problem, a finite-cover
argument, a defect split, an assembly ledger, an imported theorem
application, or an explicit obligation record. An exclusion is asserted only
where the selected endpoint theorem appears together with the hypotheses
proved or assembled before that step. A full NS3D local exclusion is used only
when its full theorem hypotheses appear in the same local branch data.

The external results invoked below are used only as named theorem inputs with
their hypotheses checked at the point of use: Leray's energy theory, the
Caffarelli--Kohn--Nirenberg $\varepsilon$-regularity theorem and its
velocity-only form, Serrin-type interior regularity, the
Necas--Ruzicka--Sverak stationary self-similar Liouville theorem, the
Albritton--Barker endpoint ancient $L^3$ Liouville theorem under its stated
whole-space hypotheses, Calderon--Zygmund pressure estimates, local harmonic
pressure estimates, and the Aubin--Lions--Simon compactness lemma.

## Locality Convention and Whole-Space Endpoint Rule

All estimates below are used only on explicitly named cylinders, selected
renormalized windows, or explicitly stated whole-space endpoint hypotheses.
The argument never infers a bound on $\mathbb R^3$ from local compactness
alone. Whenever an endpoint theorem requires a whole-space assumption, for
example

$$
W\in L^3(\mathbb R^3),
\qquad
\sup_{\tau\in\mathbb R}\int_{\mathbb R^3}|V(y,\tau)|^3\,dy<\infty,
$$

that assumption must be verified by a stated tail estimate, tightness
hypothesis, decay hypothesis, exact endpoint premise, or by the local
state-space mechanism `ST0`--`ST20` when that mechanism proves the required
endpoint sequence from terminal indecomposability. If the branch supplies only
compact-cylinder bounds, then the endpoint theorem is not applicable directly.
The missing global-looking input is either routed as a local terminal-state
branch, with covariant observer gauges and compact-window pressure
normalizations, or it remains an explicit obligation.

The sieve therefore enforces the following no-disguised-global-node rule.
A node may not ask for one super-hard whole-space property as though it were a
single computable local estimate. Whole-space $L^3$ sequences, tightness
moduli, global Duhamel/mildness formulations, and theorem-ready ancient
solution classes may appear only in two ways:

1. as hypotheses of a named external theorem after all hypotheses have already
   been assembled and matched; or
2. as outputs of many smaller local checks: dyadic shell decomposition,
   bounded-overlap local covers, active exterior core extraction, diffuse-tail
   compactification, pressure/source-defect routing, parasitic-mode removal,
   recurrence or finite-family discharge, and terminal indecomposability.

If the assembly is incomplete, the output is an explicit local obligation,
not a closed branch.

In particular, the following local bounds do not imply any whole-space bound
unless a tail condition is also present:

$$
\sup_{n}\|u_n\|_{L^3(Q_R)}<\infty,
\qquad
\sup_{n}\left(A(u_n;0,R)+E(u_n;0,R)+C(u_n;0,R)+D(p_n;0,R)\right)<\infty,
$$

for fixed $R<\infty$. They give compactness only after localization to smaller
cylinders.

Thus failure of global tightness, failure of a whole-space $L^3$ bound, and
failure of a global pressure-tail representative are not raw estimates to be
proved by the sieve. They are local residual states. The allowed replacement
principle is

$$
\text{global-looking failure}
\quad\Longrightarrow\quad
\text{local observer state}
\quad\Longrightarrow\quad
\text{local stratified exclusion}.
$$

The only places where a backward whole-space $L^3$ sequence is recovered from
the residual branch are `PS15l` and `ST17`, and there it is an output of local
terminal indecomposability rather than an assumed a priori estimate. The
Duhamel/mildness input is similarly handled by `PS15m` and `ST18` as a
residual ledger, not as a consequence of local smoothness alone.

## NS3D Objects Used

Every object used below must be produced by the three-dimensional
Navier--Stokes argument itself: an equation, gauge, scale, compactness
topology, lower bound, source term, branch datum, routing decision, or
theorem hypothesis.

---

# 1. Basic NS3D Setup

The equation is

$$
\partial_t u+(u\cdot\nabla)u+\nabla p=\Delta u,
\qquad
\nabla\cdot u=0
$$

on $\mathbb R^3$ or on a local spacetime cylinder. The pressure is defined
modulo functions of time.

For $z_0=(x_0,t_0)$ and $r>0$, write

$$
Q_r(z_0)=B_r(x_0)\times(t_0-r^2,t_0).
$$

A nonterminal cylinder is called admissible for a local argument when its
closure is compactly contained in the spacetime region on which suitability is
known. A terminal backward cylinder $Q_r(x,T)$ is admissible when the open
cylinder $B_r(x)\times(T-r^2,T)$ lies in the suitable region and every compact
subcylinder

$$
B_{r'}(x)\times[T-r'^2,T-\delta],
\qquad
0<r'<r,\quad 0<\delta<r'^2,
$$

is compactly contained there. Thus terminal arguments may approach the top
time $T$ through open backward cylinders, but every test function, compactness
window, and localized estimate still lives on a compactly contained truncated
subcylinder.

There are two levels of admissibility. A radius is geometrically admissible
when it satisfies the containment condition above. A radius is
entry-admissible when it is geometrically admissible and the local
energy-pressure entry package of `D_E` is finite on the cylinder being used:

$$
A(u;z_0,r)+C(u;z_0,r)+D(p;z_0,r)+E(u;z_0,r)<\infty.
$$

From `D_E` onward, the word admissible means entry-admissible unless the text
explicitly says geometrically admissible. This extra convention matters only
near terminal time faces: compact truncated windows give local estimates, but
the full terminal scale quantity used by `C_mu` and `PS1` must also be finite
before it can be passed to compactness. If finiteness is absent, the branch is
an entry or defect obstruction, not a completed concentration sequence.
This convention prevents a local estimate from silently crossing the spatial
boundary, the initial-time boundary, the terminal time face, or an infinite
scale-invariant entry quantity.

For a suitable weak solution, define the scale-invariant quantities

$$
A(u;z_0,r)=r^{-1}\operatorname*{ess\,sup}_{t_0-r^2<t<t_0}
\int_{B_r(x_0)}|u(x,t)|^2\,dx,
$$

$$
C(u;z_0,r)=r^{-2}\iint_{Q_r(z_0)}|u|^3\,dx\,dt,
$$

$$
D(p;z_0,r)=r^{-2}\int_{t_0-r^2}^{t_0}\int_{B_r(x_0)}
|p(x,t)-(p)_{B_r(x_0)}(t)|^{3/2}\,dx\,dt,
$$

and

$$
E(u;z_0,r)=r^{-1}\iint_{Q_r(z_0)}|\nabla u|^2\,dx\,dt.
$$

The parabolic scaling around $z_*=(x_*,T)$ is

$$
u^\lambda(x,t)=\lambda u(x_*+\lambda x,T+\lambda^2t),
\qquad
p^\lambda(x,t)=\lambda^2p(x_*+\lambda x,T+\lambda^2t).
$$

For $\lambda_k\downarrow0$, write

$$
u_k(x,t)=\lambda_k u(x_*+\lambda_k x,T+\lambda_k^2t),
\qquad
p_k(x,t)=\lambda_k^2p(x_*+\lambda_k x,T+\lambda_k^2t).
$$

For any time $\tau$ for which backward cylinders are part of the analysis,
define the time-slice singular set

$$
\Sigma(\tau)=
\{x\in\mathbb R^3:\ u\text{ is not locally bounded in any }
Q_r(x,\tau)\}.
$$

In a purely local-cylinder argument, this definition is restricted to spatial
points for which arbitrarily small backward cylinders are geometrically
admissible in the region of suitability. Thus $x\in\Sigma(\tau)$ always means
failure of local boundedness on every sufficiently small geometrically
admissible backward cylinder ending at $(x,\tau)$, and
$x\notin\Sigma(\tau)$ means boundedness on at least one such cylinder. The
terminal singularity analysis below specializes this notation to
$\tau=T$.

---

# 2. Named External Theorem Inputs

## 2.1 Suitable Weak Solutions

A pair $(u,p)$ is a suitable weak solution in an open spacetime set
$\mathcal O$ if

$$
u\in L^\infty_tL^2_{x,\mathrm{loc}}
\cap L^2_tH^1_{x,\mathrm{loc}},
\qquad
p\in L^{3/2}_{\mathrm{loc}},
\qquad
\nabla\cdot u=0,
$$

the Navier--Stokes equation holds in distributions, and the local energy
inequality holds locally. Concretely, for every nonnegative
$\phi\in C_c^\infty(\mathcal O)$ and a.e. $t_1<t_2$ such that the support of
$\phi$ between $t_1$ and $t_2$ is compactly contained in $\mathcal O$,

$$
\begin{aligned}
&\int_{\Omega_{t_2}}\frac{|u(x,t_2)|^2}{2}\phi(x,t_2)\,dx
+\int_{t_1}^{t_2}\int_{\Omega_t}|\nabla u|^2\phi\,dx\,dt \\
&\le
\int_{\Omega_{t_1}}\frac{|u(x,t_1)|^2}{2}\phi(x,t_1)\,dx
+\int_{t_1}^{t_2}\int_{\Omega_t}
\frac{|u|^2}{2}(\partial_t\phi+\Delta\phi)\,dx\,dt \\
&\quad
+\int_{t_1}^{t_2}\int_{\Omega_t}
\left(\frac{|u|^2}{2}+p\right)u\cdot\nabla\phi\,dx\,dt .
\end{aligned}
$$

Here $\Omega_t=\{x:(x,t)\in\mathcal O\}$. In the whole-space case
$\Omega_t=\mathbb R^3$. In a local cylinder, the displayed integrals may also
be read over any spatial ball containing the support of $\phi(\cdot,t)$; the
compact support makes the two formulations identical. No boundary term is
being assumed or hidden at this stage.

## 2.2 Caffarelli--Kohn--Nirenberg Regularity

There is $\varepsilon_0>0$ such that if $(u,p)$ is suitable in $Q_r(z_0)$ and

$$
C(u;z_0,r)+D(p;z_0,r)<\varepsilon_0,
$$

then $u$ is locally bounded in $Q_{r/2}(z_0)$.

There is also a velocity-only threshold $\varepsilon_v>0$ such that

$$
C(u;z_0,r)<\varepsilon_v
$$

implies local boundedness in $Q_{r/2}(z_0)$.

These Caffarelli--Kohn--Nirenberg/Lin inputs are used only through the
displayed hypotheses and conclusions.

## 2.3 Functional-Analytic Tools

The compactness steps use:

- the local Sobolev interpolation
  $L^\infty_tL^2_x\cap L^2_tH^1_x\subset L^3_{x,t}$ on bounded cylinders;
- Calderon--Zygmund boundedness of Riesz transforms on $L^{3/2}$;
- local harmonic estimates for the pressure remainder;
- Aubin--Lions--Simon compactness:
  if $X_0\Subset X\hookrightarrow X_1$, a sequence bounded in
  $L^p(I;X_0)$ with time derivative bounded in $L^q(I;X_1)$ is compact in
  $L^p(I;X)$ after localization.

## 2.4 First-Five Entry and Concentration Packet

The first local block proves only the following entry-and-concentration
packet. It does not exclude a singularity and it does not produce any
whole-space endpoint hypothesis.

**Proposition -- Local entry and concentration packet.**
Let $\mathcal O\subset\mathbb R^3\times\mathbb R$ be open, and let $(u,p)$ be
a suitable weak solution of 3D Navier--Stokes in $\mathcal O$. Fix

$$
z_*=(x_*,T).
$$

Assume:

1. $x_*\in\Sigma(T)$, meaning that $u$ is not locally bounded in any
   sufficiently small geometrically admissible backward cylinder ending at
   $(x_*,T)$.
2. There is at least one terminal radius $r_0>0$ such that $Q_{r_0}(z_*)$ is
   geometrically admissible and has finite entry:

   $$
   A(u;z_*,r_0)+C(u;z_*,r_0)+D(p;z_*,r_0)+E(u;z_*,r_0)<\infty.
   $$

Then:

1. Every smaller radius $0<r<r_0$ is entry-admissible.
2. For every $0<r<r_0$,

   $$
   C(u;z_*,r)+D(p;z_*,r)\ge\varepsilon_0,
   \qquad
   C(u;z_*,r)\ge\varepsilon_v.
   $$

3. For every sequence $r_n\downarrow0$ with $0<r_n<r_0$, the rescaled fields

   $$
   u_n(y,s)=r_nu(x_*+r_ny,T+r_n^2s),
   \qquad
   p_n(y,s)=r_n^2p(x_*+r_ny,T+r_n^2s)
   $$

   are suitable on every compact truncated rescaled cylinder

   $$
   B_R\times[-R^2,-\delta],
   \qquad R<\infty,\quad \delta>0,
   $$

   for all sufficiently large $n$, and satisfy

   $$
   C(u_n;0,1)+D(p_n;0,1)\ge\varepsilon_0,
   \qquad
   C(u_n;0,1)\ge\varepsilon_v.
   $$

**Proof.** Lemma D_E.3 shows that one finite terminal entry radius gives
finite entry at every smaller radius. Lemma Rec_N.2 gives suitability after
parabolic rescaling, and Lemma Rec_N.3 restricts terminal rescalings to
compact windows bounded away from the top face $s=0$. Lemma C_mu.2 applies the
CKN criterion and its velocity-only form in contrapositive form at the
selected point $x_*\in\Sigma(T)$. Lemma C_mu.3 selects the shrinking
original-scale concentration sequence, and Lemma PS1.2 moves the two lower
bounds from $Q_{r_n}(z_*)$ to $Q_1(0,0)$. No endpoint theorem and no
whole-space bound is used in this packet.

---

# 3. `H0` -- NS3D Initial Analytic Data

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknowns are the velocity and pressure of the three-dimensional
incompressible Navier--Stokes equations on $\mathcal O$.

### Standing Assumptions

The declared local proof starts with a suitable weak solution on the region
being analyzed. Whole-space finite-energy data are declared only for the
whole-space Type I case.

### Objects Inspected

Inspect the formulas for the PDE, divergence constraint, local energy
inequality, pressure mean subtraction, scaling, and singular set.

### Dependencies Used

No previous node contributes data to `H0`. Every later node imports some part
of this section.

### Local Obstruction Predicate

$P_{H0}$ holds exactly when the record contains an undefined or ambiguous
Navier--Stokes object required by a later node.

### Local Lemma to Prove

The entry lemma below proves local well-definedness of the quantities used by
`D_E`.

### Specific Estimate

The decisive estimate is the local interpolation

$$
L^\infty_tL^2_x\cap L^2_tH^1_x
\subset L^3_{x,t}
$$

on bounded cylinders, in the following localized form. If $B_r=B_r(x_0)$ and
$I$ is a time interval, then

$$
\int_I\|u(t)\|_{L^3(B_r)}^3\,dt
\le
C
\left(\operatorname*{ess\,sup}_{t\in I}\|u(t)\|_{L^2(B_r)}^2\right)^{3/4}
\left(
\int_I\|\nabla u(t)\|_{L^2(B_r)}^2\,dt
+r^{-2}\int_I\|u(t)\|_{L^2(B_r)}^2\,dt
\right)^{3/4}
|I|^{1/4}.
$$

The lower-order $r^{-1}\|u\|_2$ term is part of the local Sobolev estimate on
a ball; it cannot be dropped unless a zero-boundary or zero-mean condition has
also been imposed. The pressure-mean bound is

$$
\|p-(p)_B(t)\|_{L^{3/2}(B)}
\le C\|p\|_{L^{3/2}(B)}.
$$

### Practical Verification Steps

1. Declare the equation and divergence constraint.
2. Declare the suitable weak solution class.
3. Write the local energy inequality.
4. Define $Q_r(z_0)$ and the quantities $A,C,D,E$.
5. State the pressure convention modulo functions of time.
6. State the parabolic scaling and singular set.
7. State the target regularity criterion used to close a singular branch:
   local boundedness in some $Q_\rho(z_0)$.
8. Verify local well-definedness by the entry lemma.

## Estimate Step $B_{H0}$

The estimate step is the entry lemma: local energy and pressure integrability
make $A,C,D,E$ finite on compactly contained cylinders.

## Failure Case

Failure name: incomplete Navier--Stokes analytic data.

Analytic meaning: the local proof has an undefined solution class, pressure
convention, cylinder, critical quantity, scaling map, or singular set.

## Refinement Step

Allowed refinements:

1. specify the local solution domain;
2. specify suitability and the local energy inequality;
3. specify pressure modulo functions of time;
4. restrict to compactly contained cylinders;
5. declare the target time and singular set.

Progress measure: each refinement fills one named missing analytic datum.

## Data Passed Forward

The next proof step is `D_E`. The data passed forward are

$$
\Gamma_{H0}
=
\{u,p,\mathcal O,Q_r,A,C,D,E,\text{LEI},\text{pressure convention},
\text{scaling},\Sigma(\tau),\text{local regularity predicate}\}.
$$

## Data Required

The initial analytic data consist of:

1. a suitable weak solution $(u,p)$ on the local region under study;
2. the pressure convention modulo functions of time;
3. the local cylinders $Q_r(z_0)$;
4. the local energy inequality;
5. the scale-invariant quantities $A,C,D,E$;
6. the CKN regularity criterion;
7. the parabolic scaling law;
8. the time-slice singular sets $\Sigma(\tau)$, in particular the terminal
   set $\Sigma(T)$ when a terminal time is fixed;
9. the target local regularity predicate and the CKN sufficient condition for
   that predicate.

When the whole-space Type I finite-energy case is considered, the data also
include a suitable Leray--Hopf solution on $\mathbb R^3\times(0,T)$.

## Entry Lemma

**Lemma H0.1 -- Local interpolation on a ball.**
Let $B_r=B_r(x_0)$ and let $I$ be a bounded time interval. If

$$
u\in L^\infty(I;L^2(B_r))\cap L^2(I;H^1(B_r)),
$$

then

$$
\int_I\|u(t)\|_{L^3(B_r)}^3\,dt
\le
C
\left(\operatorname*{ess\,sup}_{t\in I}\|u(t)\|_{L^2(B_r)}^2\right)^{3/4}
\left(
\int_I\|\nabla u(t)\|_{L^2(B_r)}^2\,dt
+r^{-2}\int_I\|u(t)\|_{L^2(B_r)}^2\,dt
\right)^{3/4}
|I|^{1/4}.
$$

**Proof.** Fix a time $t$ for which $u(t)\in H^1(B_r)$. The scaled Sobolev
and Gagliardo--Nirenberg inequality on the ball gives

$$
\|f\|_{L^3(B_r)}
\le
C\|f\|_{L^2(B_r)}^{1/2}
\left(\|\nabla f\|_{L^2(B_r)}+r^{-1}\|f\|_{L^2(B_r)}\right)^{1/2}.
$$

The lower-order term $r^{-1}\|f\|_{L^2(B_r)}$ is necessary on a general ball:
no zero trace and no zero mean has been imposed. Applying the estimate to
$f=u(t)$ and raising to the third power gives

$$
\|u(t)\|_{L^3(B_r)}^3
\le
C\|u(t)\|_{L^2(B_r)}^{3/2}
\left(
\|\nabla u(t)\|_{L^2(B_r)}^2
+r^{-2}\|u(t)\|_{L^2(B_r)}^2
\right)^{3/4}.
$$

Set

$$
a(t)=\|u(t)\|_{L^2(B_r)}^2,
\qquad
b(t)=\|\nabla u(t)\|_{L^2(B_r)}^2
+r^{-2}\|u(t)\|_{L^2(B_r)}^2.
$$

Then

$$
\|u(t)\|_{L^3(B_r)}^3\le C a(t)^{3/4}b(t)^{3/4}.
$$

Integrating in time and using Hölder gives

$$
\begin{aligned}
\int_I a(t)^{3/4}b(t)^{3/4}\,dt
&\le
\left(\operatorname*{ess\,sup}_{t\in I}a(t)\right)^{3/4}
\int_I b(t)^{3/4}\,dt \\
&\le
\left(\operatorname*{ess\,sup}_{t\in I}a(t)\right)^{3/4}
\left(\int_I b(t)\,dt\right)^{3/4}
|I|^{1/4}.
\end{aligned}
$$

Substituting the definitions of $a$ and $b$ proves the displayed
interpolation estimate.

**Lemma H0.2 -- Pressure mean subtraction and gauge invariance.**
Let $q=3/2$ and let $B$ be a ball. For a.e. time $t$ with
$p(\cdot,t)\in L^q(B)$,

$$
\|p(\cdot,t)-(p)_B(t)\|_{L^q(B)}
\le C\|p(\cdot,t)\|_{L^q(B)}.
$$

Moreover, if $a(t)$ is any scalar function of time, then

$$
(p+a(t))-(p+a(t))_B=p-(p)_B.
$$

**Proof.** Jensen's inequality gives

$$
|(p)_B(t)|^q
=
\left|\frac1{|B|}\int_Bp(x,t)\,dx\right|^q
\le
\frac1{|B|}\int_B|p(x,t)|^q\,dx.
$$

Therefore

$$
\begin{aligned}
\int_B|p-(p)_B(t)|^q\,dx
&\le
2^{q-1}\int_B|p|^q\,dx
+2^{q-1}|B|\,|(p)_B(t)|^q \\
&\le
C\int_B|p|^q\,dx.
\end{aligned}
$$

Taking the $q$th root gives the norm estimate. For the gauge identity,
spatial averaging over $B$ gives $(p+a(t))_B=(p)_B+a(t)$ because $a(t)$ is
independent of $x$. Subtraction cancels the added function of time exactly.
Thus $D$ is invariant under replacing $p$ by $p+a(t)$.

**Lemma H0.3 -- Initial entry well-definedness.**
The data declared in `H0` make all quantities used in the first proof step
`D_E` mathematically defined. The local energy inequality is valid for
compactly supported nonnegative cutoffs, the pressure has a mean-subtracted
representative on balls, and $A,C,D,E$ are well-defined on every cylinder
compactly contained in the region of suitability.

**Proof.** The local energy inequality is part of the definition of a suitable
weak solution. Lemma H0.2 shows both that the pressure oscillation in $D$ is
independent of the time-dependent pressure gauge and that
$p-(p)_B(t)\in L^{3/2}$ on compact local cylinders whenever
$p\in L^{3/2}_{\mathrm{loc}}$.

The local energy class gives

$$
u\in L^\infty_tL^2_{x,\mathrm{loc}},
\qquad
\nabla u\in L^2_{x,t,\mathrm{loc}}.
$$

Applying Lemma H0.1 on a compact spatial ball and compact time interval gives
$u\in L^3_{\mathrm{loc}}$. Therefore $C$ and $D$ are finite on compactly
contained cylinders. The definitions of $A$ and $E$ are finite from the local
energy class itself. This proves the entry lemma.

---

# 4. `D_E` -- Local Energy Entry

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

Work on an open spacetime region $\mathcal O$ where $(u,p)$ is a suitable weak
solution of three-dimensional incompressible Navier--Stokes. The unknowns are
the velocity $u$ and pressure $p$. The topology is the local energy topology

$$
u\in L^\infty_tL^2_{x,\rm loc}\cap L^2_tH^1_{x,\rm loc},
\qquad
p\in L^{3/2}_{\rm loc}.
$$

### Standing Assumptions

The standing assumptions are exactly the data proved in `H0`,
$\Gamma_{H0}$:

1. $(u,p)$ solves NS distributionally;
2. $\nabla\cdot u=0$;
3. $(u,p)$ is suitable;
4. the local energy inequality holds;
5. pressure is understood modulo functions of time;
6. the quantities $A,C,D,E$ are defined by the formulas in Section 1.

### Objects Inspected

The proof inspects the local norms of $u$, $\nabla u$, and $p$, the spatial
mean $(p)_{B_r(x_0)}(t)$, and the local energy inequality on compactly
supported cutoffs.

### Dependencies Used

The equation, solution class, local energy inequality, pressure convention,
and definitions of $A,C,D,E$ come directly from `H0`. No rescaling, limiting
profile, concentration measure, or pressure gauge from later proof steps is
used.

### Local Obstruction Predicate

$P_{D_E}$ holds if at least one of the following is true:

- the record lacks suitability of $(u,p)$ on the current local cylinder;
- the record lacks $p\in L^{3/2}_{\rm loc}$;
- the local energy inequality is missing;
- the mean-subtracted pressure quantity $D$ is not defined;
- the local energy-pressure quantities $A,C,D,E$ are not finite;
- in the optional Type I setup, the no-escape estimate needed later is
  asserted without the hypotheses that imply it.

### Local Lemma to Prove

**Lemma D_E.1 -- Interior finite entry.**
Let $(u,p)$ be suitable in an open region $\mathcal O$. Assume
$Q_r(z_0)$ is nonterminal and geometrically admissible in the literal sense

$$
\overline{B_r(x_0)}\times[t_0-r^2,t_0]\Subset\mathcal O.
$$

Then $A(u;z_0,r)$, $C(u;z_0,r)$, $D(p;z_0,r)$, and $E(u;z_0,r)$ are finite,
and the local energy inequality holds on $Q_r(z_0)$.

**Proof.** The finiteness of $A$ and $E$ follows directly from
$u\in L^\infty_tL^2_{x,\rm loc}$ and
$\nabla u\in L^2_{x,t,\rm loc}$. For $C$, apply Lemma H0.1 on
$B_r(x_0)\times(t_0-r^2,t_0)$. Since no zero trace or zero spatial mean is
available for a local suitable solution, the estimate includes the lower-order
term from the ball:

$$
\int_{t_0-r^2}^{t_0}\|u(t)\|_{L^3(B_r)}^3\,dt
\le
C
\left(\operatorname*{ess\,sup}_{t_0-r^2<t<t_0}
\|u(t)\|_{L^2(B_r)}^2\right)^{3/4}
\left(
\int_{t_0-r^2}^{t_0}\|\nabla u(t)\|_{L^2(B_r)}^2\,dt
+r^{-2}\int_{t_0-r^2}^{t_0}\|u(t)\|_{L^2(B_r)}^2\,dt
\right)^{3/4}
r^{1/2}.
$$

Thus $C(u;z_0,r)<\infty$.

For the pressure mean, Lemma H0.2 gives
$p-(p)_{B_r(x_0)}(t)\in L^{3/2}(B_r(x_0))$ for a.e. time, with norm controlled
by $\|p(\cdot,t)\|_{L^{3/2}(B_r)}$. Integrating in time gives
$D(p;z_0,r)<\infty$. Suitability supplies the local energy inequality for
every nonnegative cutoff
$\phi\in C_c^\infty(Q_r(z_0))$. This proves the lemma.

**Lemma D_E.2 -- Terminal entry is not automatic.**
Suppose the later branch uses a terminal backward cylinder $Q_r(x,T)$. The
proof may use this radius in `C_mu` or `PS1` only after the finite entry
condition has been proved:

$$
A(u;(x,T),r)+C(u;(x,T),r)+D(p;(x,T),r)+E(u;(x,T),r)<\infty.
$$

If the finite entry condition is not known, the branch cannot pass a
scale-invariant terminal quantity to concentration or compactness. It must
either assume finite terminal entry as part of the local branch, prove it from
a stronger ambient hypothesis such as a suitable Leray--Hopf whole-space
solution with pressure control up to $T$, or record the entry obstruction as
the current branch status.

**Proof.** The terminal-cylinder convention supplies compactly contained
truncated windows

$$
B_{r'}(x)\times[T-r'^2,T-\delta].
$$

The local energy class, Lemma H0.1, and Lemma H0.2 apply on each such compact
truncation, or equivalently on a finite cover by interior backward cylinders
whose closures lie in $\mathcal O$. However, the scale-invariant quantities
used by CKN and by the normalization in `PS1` are integrals over the full open
backward cylinder up to the terminal time. Local integrability on every compact
truncation does not by itself imply integrability on their union. Therefore
terminal finiteness is an additional entry datum. When it is recorded, the CKN
quantities and scaling identities are legitimate finite numbers. When it is
absent, the correct mathematical conclusion is a missing entry estimate, not a
concentration sequence ready for compactness.

**Lemma D_E.3 -- One finite terminal entry radius gives all smaller radii.**
Let $z_*=(x_*,T)$. Assume $Q_{r_0}(z_*)$ is geometrically admissible and

$$
A(u;z_*,r_0)+C(u;z_*,r_0)+D(p;z_*,r_0)+E(u;z_*,r_0)<\infty.
$$

Then every $0<r<r_0$ is entry-admissible at $z_*$.

**Proof.** Geometric admissibility is inherited by smaller backward cylinders
because

$$
Q_r(z_*)\subset Q_{r_0}(z_*).
$$

For $A$, monotonicity of the ball and time interval gives

$$
\begin{aligned}
A(u;z_*,r)
&=
r^{-1}\operatorname*{ess\,sup}_{T-r^2<t<T}
\int_{B_r(x_*)}|u|^2\,dx \\
&\le
r^{-1}\operatorname*{ess\,sup}_{T-r_0^2<t<T}
\int_{B_{r_0}(x_*)}|u|^2\,dx
=\frac{r_0}{r}A(u;z_*,r_0)<\infty.
\end{aligned}
$$

For $C$,

$$
\begin{aligned}
C(u;z_*,r)
&=
r^{-2}\iint_{Q_r(z_*)}|u|^3\,dx\,dt \\
&\le
r^{-2}\iint_{Q_{r_0}(z_*)}|u|^3\,dx\,dt
=\left(\frac{r_0}{r}\right)^2C(u;z_*,r_0)<\infty.
\end{aligned}
$$

For $E$,

$$
E(u;z_*,r)
\le
\frac{r_0}{r}E(u;z_*,r_0)<\infty.
$$

The pressure term requires comparing different spatial means. Define

$$
a(t)=(p)_{B_{r_0}(x_*)}(t),
\qquad
q(x,t)=p(x,t)-a(t).
$$

The finite entry assumption at $r_0$ is exactly

$$
r_0^{-2}\int_{T-r_0^2}^{T}\int_{B_{r_0}(x_*)}|q(x,t)|^{3/2}\,dx\,dt<\infty.
$$

For $B_r(x_*)\subset B_{r_0}(x_*)$, spatial averaging gives

$$
p-(p)_{B_r(x_*)}=q-(q)_{B_r(x_*)}.
$$

Lemma H0.2 applied to $q$ on $B_r(x_*)$ yields, for a.e. $t$,

$$
\int_{B_r(x_*)}|p-(p)_{B_r(x_*)}|^{3/2}\,dx
=
\int_{B_r(x_*)}|q-(q)_{B_r(x_*)}|^{3/2}\,dx
\le
C\int_{B_r(x_*)}|q|^{3/2}\,dx.
$$

After integrating in time,

$$
\begin{aligned}
D(p;z_*,r)
&\le
Cr^{-2}\int_{T-r^2}^{T}\int_{B_r(x_*)}|q|^{3/2}\,dx\,dt \\
&\le
Cr^{-2}\int_{T-r_0^2}^{T}\int_{B_{r_0}(x_*)}|q|^{3/2}\,dx\,dt
=C\left(\frac{r_0}{r}\right)^2D(p;z_*,r_0)<\infty.
\end{aligned}
$$

Thus $A,C,D,E$ are finite on every smaller terminal radius, and each such
radius is entry-admissible.

### Specific Estimate

The decisive estimate is the local interpolation and pressure-mean estimate:

$$
L^\infty_tL^2_x\cap L^2_tH^1_x
\Longrightarrow
L^3_{x,t}
$$

on bounded cylinders, in the localized form

$$
\iint_{Q_r(z_0)}|u|^3\,dx\,dt
\le
C r^2
A(u;z_0,r)^{3/4}
\left(A(u;z_0,r)+E(u;z_0,r)\right)^{3/4}.
$$

The lower-order term in the local Sobolev estimate is what produces the
$A(u;z_0,r)$ contribution inside the last factor. The pressure estimate is

$$
\|p-(p)_{B_r}(t)\|_{L^{3/2}(Q_r)}
\le C\|p\|_{L^{3/2}(Q_r)}.
$$

Both estimates are local to $Q_r(z_0)$.

### Practical Verification Steps

1. Check that the closed nonterminal cylinder
   $\overline{B_r(x_0)}\times[t_0-r^2,t_0]$ is compactly contained in
   $\mathcal O$.
2. Check that $(u,p)$ is suitable on $\mathcal O$.
3. Use the local energy class to bound $A$ and $E$.
4. Apply Sobolev interpolation to bound $C$.
5. Apply Jensen's inequality to bound the mean-subtracted pressure term in
   $D$.
6. Record the local energy inequality for cutoffs supported in $Q_r(z_0)$.
7. If $Q_r$ is terminal, use Lemma D_E.2 to record that finite terminal entry
   is an additional datum, and use Lemma D_E.3 when one finite terminal entry
   radius is available to send all smaller radii to `C_mu` or `PS1`.

## Estimate Step $B_{D_E}$

### Estimate Proposition

The step applies the local entry lemma to the analytic data returned by `H0`.
The entry obstruction is controlled exactly by restricting or reformulating the
declared hypotheses until Lemma D_E.1 applies on a smaller local cylinder.

In the optional Type I setup, the following no-escape estimate is part of the
branch record only after its stated hypotheses have been verified.

**Lemma D_E.4 -- Type I no-escape estimate.**
Let $z_*=(x_*,T)$ satisfy the local Type I bound

$$
\|u(t)\|_{L^\infty(B_\rho(x_*))}
\le M(T-t)^{-1/2},
\qquad T-\rho^2<t<T,
$$

and let $\lambda_k\downarrow0$. If

$$
\sup_k A(u;z_*,\lambda_k)\le B<\infty,
$$

then the rescaled velocities

$$
u_k(x,t)=\lambda_k u(x_*+\lambda_kx,T+\lambda_k^2t)
$$

satisfy

$$
\lim_{\sigma\downarrow0}
\sup_k\int_{-\sigma}^{0}\int_{B_1}|u_k|^3\,dx\,dt=0.
$$

**Proof.** Choose $k_1$ so large that $\lambda_k<\rho$ for $k\ge k_1$.
Then for every $x\in B_1$ and $t\in(-1,0)$ the physical point
$x_*+\lambda_kx$ belongs to $B_\rho(x_*)$ and
$T+\lambda_k^2t\in(T-\rho^2,T)$. Therefore, for all $k\ge k_1$,

$$
\|u_k(t)\|_{L^\infty(B_1)}
\le M(-t)^{-1/2}.
$$

Scale invariance of $A$ gives

$$
\operatorname*{ess\,sup}_{-1<t<0}
\int_{B_1}|u_k(x,t)|^2\,dx\le B.
$$

Therefore

$$
\begin{aligned}
\int_{-\sigma}^{0}\int_{B_1}|u_k|^3\,dx\,dt
&\le
\int_{-\sigma}^{0}
\|u_k(t)\|_{L^\infty(B_1)}
\int_{B_1}|u_k(x,t)|^2\,dx\,dt \\
&\le
MB\int_{-\sigma}^{0}(-t)^{-1/2}\,dt
=2MB\sigma^{1/2}.
\end{aligned}
$$

For the finitely many indices $k<k_1$, each fixed function $u_k$ belongs to
$L^3(B_1\times(-1,0))$ by Lemma D_E.1 and scaling. Hence

$$
\lim_{\sigma\downarrow0}
\max_{k<k_1}
\int_{-\sigma}^{0}\int_{B_1}|u_k|^3\,dx\,dt=0
$$

by absolute continuity of the integral. Combining this finite-index statement
with the uniform bound for $k\ge k_1$ proves the asserted supremum limit.

## Failure Case

Failure name: local energy-entry failure.

Analytic meaning: the argument cannot begin the local concentration analysis
because it lacks the suitable-solution structure or the local pressure and
energy quantities needed by the CKN and compactness estimates.

The failure data are:

- the cylinder on which the entry fails;
- the missing term or hypothesis;
- the estimate from Lemma D_E.1, D_E.2, D_E.3, or D_E.4 that cannot be
  executed.

## Refinement Step

The only admissible refinements at `D_E` are the following local PDE
refinements:

1. shrink to a cylinder where suitability is recorded;
2. choose a pressure representative modulo a function of time;
3. replace an ambient Leray--Hopf formulation by its local suitable weak
   formulation;
4. in the Type I setup, restrict to scales for which the stated $A$ bound and
   Type I bound are recorded;
5. in a terminal setup, add the missing finite entry estimate up to $T$ or
   route the branch to a defect node instead of `C_mu`.

The refinement is valid only if after refinement Lemma D_E.1 applies, and, in
the Type I case, Lemma D_E.4 applies.

Progress measure: the refinement either strictly shrinks the analysis cylinder,
fixes one previously undefined representative, or records one concrete missing
hypothesis. It cannot loop without changing the local input data.

## Data Passed Forward

The next proof step is `Rec_N`. The data passed forward are

$$
\Gamma_{D_E}=
\{(u,p),Q_r(z_0),A,C,D,E,\text{local energy inequality},
\text{pressure mean convention},\text{finite terminal entry if terminal}\}.
$$

If the optional Type I estimate is used, $\Gamma_{D_E}$ also includes the
no-escape conclusion

$$
\lim_{\sigma\downarrow0}
\sup_k\int_{-\sigma}^{0}\int_{B_1}|u_k|^3\,dx\,dt=0.
$$

These data are admissible for `Rec_N`: they contain a suitable NS3D solution,
local cylinders, and the rescaling convention needed to select a terminal
point.

---

# 5. `Rec_N` -- Terminal Point and Rescaling Setup

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

Work with the suitable weak solution $(u,p)$ supplied by `D_E` near the time
$T$. The unknowns remain $u,p$. The geometry is a backward cylinder ending at
$(x_*,T)$.

### Standing Assumptions

The solution is suitable locally and the local quantities $A,C,D,E$ are
defined where entry-admissibility has been proved. The local singular branch
assumes

$$
\Sigma(T)\ne\emptyset
$$

and then selects one finite spatial point $x_*\in\Sigma(T)$.

### Objects Inspected

Inspect $\Sigma(T)$, candidate points $x_*$, and backward cylinders
$Q_r(x_*,T)$.

### Dependencies Used

The local energy structure comes from `D_E`. The singular set and rescaling
formulas come from `H0`. No concentration lower bound is used at this step;
this step fixes only the sequence format to which `C_mu` and `PS1` attach the
lower bound.

### Local Obstruction Predicate

$P_{\mathrm{Rec}_N}$ holds exactly when the proof has not produced
$x_*\in\Sigma(T)$, has not fixed the rescaling convention around $z_*$, or has
introduced centers/times/scales without proving that their cylinders are
admissible for the suitable-solution structure.

### Local Lemma to Prove

**Lemma Rec_N.1 -- Terminal point selection.**
If the local branch assumes a singularity at a finite spatial point $x_*$ at
time $T$, then the precise local assumption is $x_*\in\Sigma(T)$. Equivalently,
the branch assumes $\Sigma(T)\ne\emptyset$, chooses $x_*\in\Sigma(T)$, and
sets $z_*=(x_*,T)$. If $\Sigma(T)=\emptyset$, there is no local singular
profile to analyze at time $T$.

**Proof.** The singular set at time $T$ is

$$
\Sigma(T)=
\{x\in\mathbb R^3:\ u\notin L^\infty(Q_r(x,T))\text{ for every }r>0\}.
$$

In a local spacetime region, the quantifier over $r$ is restricted to
geometrically admissible radii, as fixed in the Basic NS3D Setup.

Thus

$$
\{x:\ x\notin\Sigma(T)\}
=
\{x:\ \exists\text{ a geometrically admissible }r>0,\ u\in L^\infty(Q_r(x,T))\}.
$$

Thus the local branch never uses an unspecified global phrase such as
"singular time" as a substitute for a pointwise hypothesis. It either records
$x_*\in\Sigma(T)$ directly, or it records $\Sigma(T)\ne\emptyset$ and then
selects such an $x_*$. This is exactly the datum needed by the CKN
contrapositive in `C_mu`.

**Lemma Rec_N.2 -- Rescaling preserves suitability.**
For every $\lambda>0$, define

$$
\mathcal O_\lambda
=
\{(y,s):(x_*+\lambda y,T+\lambda^2s)\in\mathcal O\},
$$

and

$$
u^\lambda(y,s)=\lambda u(x_*+\lambda y,T+\lambda^2s),
\qquad
p^\lambda(y,s)=\lambda^2p(x_*+\lambda y,T+\lambda^2s).
$$

If $(u,p)$ is suitable on the physical cylinder, then
$(u^\lambda,p^\lambda)$ is suitable on the corresponding rescaled cylinder
inside $\mathcal O_\lambda$.

**Proof.** The proof checks the four parts of suitability.

First, the local energy class is preserved. If $K\Subset\mathcal O_\lambda$,
then its physical image

$$
K^\lambda=\{(x_*+\lambda y,T+\lambda^2s):(y,s)\in K\}
$$

is compactly contained in $\mathcal O$. The change of variables
$X=x_*+\lambda y$, $S=T+\lambda^2s$ gives finite local
$L^\infty_sL^2_y$ and $L^2_sH^1_y$ norms for $u^\lambda$ on $K$ from the
corresponding finite norms of $u$ on $K^\lambda$. The same change of variables
gives $p^\lambda\in L^{3/2}_{\mathrm{loc}}(\mathcal O_\lambda)$.

Second, the divergence-free condition is preserved because

$$
\nabla_y\cdot u^\lambda(y,s)
=
\lambda^2(\nabla_X\cdot u)(X,S)=0.
$$

Third, the distributional Navier--Stokes equation is preserved. Pointwise for
smooth functions, and therefore in distributions by testing and changing
variables,

$$
\partial_su^\lambda=\lambda^3(\partial_Su)(X,S),
\qquad
(u^\lambda\cdot\nabla_y)u^\lambda
=\lambda^3((u\cdot\nabla_X)u)(X,S),
$$

$$
\nabla_y p^\lambda=\lambda^3(\nabla_Xp)(X,S),
\qquad
\Delta_yu^\lambda=\lambda^3(\Delta_Xu)(X,S).
$$

Every term receives the same factor $\lambda^3$, so the weak equation
transforms into the weak equation for $(u^\lambda,p^\lambda)$.

Fourth, the local energy inequality is preserved. Let
$\psi\in C_c^\infty(\mathcal O_\lambda)$ be nonnegative and set

$$
\phi(X,S)=
\psi\left(\frac{X-x_*}{\lambda},\frac{S-T}{\lambda^2}\right).
$$

Then $\phi\in C_c^\infty(\mathcal O)$, $\phi\ge0$, and

$$
\partial_S\phi=\lambda^{-2}\partial_s\psi,
\qquad
\nabla_X\phi=\lambda^{-1}\nabla_y\psi,
\qquad
\Delta_X\phi=\lambda^{-2}\Delta_y\psi.
$$

Substituting this $\phi$ into the physical local energy inequality and using
$dX\,dS=\lambda^5dy\,ds$, $u^\lambda=\lambda u$, and
$p^\lambda=\lambda^2p$, every term acquires the same positive factor. Dividing
by that factor gives the local energy inequality for
$(u^\lambda,p^\lambda)$ and $\psi$. Adding a function of time to $p$ becomes
adding a function of rescaled time to $p^\lambda$, so the pressure convention
is preserved.

**Lemma Rec_N.3 -- Geometrically admissible center-time-scale records.**
There are two geometrically admissible sequence formats at this stage.

1. Fixed-terminal local concentration format:

   $$
   x_n=x_*,
   \qquad
   t_n=T,
   \qquad
   \lambda_n=r_n\downarrow0.
   $$

   A radius $r_n$ is geometrically admissible exactly when every fixed compact
   truncated window

   $$
   B_R\times[-R^2,-\delta],
   \qquad R<\infty,\quad \delta>0,
   $$

   in rescaled variables maps into the suitable region for all sufficiently
   large $n$. The open top $s=0$ is the terminal face and is never required as
   part of a compact suitability window.

2. Moving Type I blow-up format:

   $$
   t_n\uparrow T,\qquad x_n\to x_*,
   \qquad \lambda_n\downarrow0,
   $$

   with

   $$
   Q_{R\lambda_n}(x_n,t_n)
   =
   B_{R\lambda_n}(x_n)\times(t_n-R^2\lambda_n^2,t_n)
   \Subset\mathcal O
   $$

   for every fixed $R<\infty$ and all sufficiently large $n$, with compact
   estimates taken on truncated windows bounded away from the rescaled top
   time.

In either format, the rescaled fields

$$
u_n(y,s)=\lambda_n u(x_n+\lambda_n y,t_n+\lambda_n^2s),
\qquad
p_n(y,s)=\lambda_n^2p(x_n+\lambda_n y,t_n+\lambda_n^2s)
$$

are suitable on each fixed compact truncated backward window in the rescaled
domain. This lemma does not assert that the full terminal scale quantities
$A,C,D,E$ are finite; that stronger entry-admissibility is supplied by `D_E`
when `C_mu` or `PS1` evaluate CKN quantities on a terminal radius.

**Proof.** In the fixed-terminal format, geometric admissibility follows from
$r_n\downarrow0$ and the terminal-cylinder convention in the Basic NS3D Setup.
For a fixed truncated rescaled window
$B_R\times[-R^2,-\delta]$, its physical image is

$$
B_{Rr_n}(x_*)\times[T-R^2r_n^2,T-\delta r_n^2],
$$

which is compactly contained in the open terminal cylinder
$B_{r_0}(x_*)\times(T-r_0^2,T)$ for all large $n$, after $r_0$ is chosen
admissible. The open top face $s=0$ is used only as the limiting terminal
face of the blow-up coordinates, not as part of the compact support of a test
function. In the moving format, the containment condition is exactly the
hypothesis, again with estimates taken on truncated compact windows when the
rescaled top time is approached. Lemma Rec_N.2 then applies with center
$(x_n,t_n)$ and scale $\lambda_n$ on each such compact window. Thus
suitability, the local energy inequality, divergence-free structure, and the
pressure convention are inherited by the rescaled sequence. The lower bound
for concentration is not asserted here; it is supplied by `C_mu` and `PS1`.

### Specific Estimate

The specific assertion is the invariance of the local suitable weak solution
class under parabolic Navier--Stokes scaling. For

$$
u^\lambda(x,t)=\lambda u(x_*+\lambda x,T+\lambda^2t),
\qquad
p^\lambda(x,t)=\lambda^2p(x_*+\lambda x,T+\lambda^2t),
$$

the scale-invariant quantities satisfy

$$
A(u^\lambda;0,r)=A(u;z_*,\lambda r),\quad
C(u^\lambda;0,r)=C(u;z_*,\lambda r),
$$

$$
D(p^\lambda;0,r)=D(p;z_*,\lambda r),\quad
E(u^\lambda;0,r)=E(u;z_*,\lambda r).
$$

For the pressure identity, the spatial mean transforms exactly as

$$
(p^\lambda)_{B_r}(s)
=
\lambda^2(p)_{B_{\lambda r}(x_*)}(T+\lambda^2s).
$$

Thus

$$
p^\lambda(y,s)-(p^\lambda)_{B_r}(s)
=
\lambda^2\left[
p(x_*+\lambda y,T+\lambda^2s)
-(p)_{B_{\lambda r}(x_*)}(T+\lambda^2s)
\right],
$$

and the change of variables gives $D(p^\lambda;0,r)=D(p;z_*,\lambda r)$.

The local energy inequality is preserved after the test-function substitution

$$
\phi(X,S)=\psi\!\left(\frac{X-x_*}{\lambda},
\frac{S-T}{\lambda^2}\right),
$$

with the common dimensional factor cancelled from both sides of the
inequality.

### Practical Verification Steps

1. Fix the time $T$.
2. If a singularity is assumed, choose $x_*\in\Sigma(T)$.
3. Define $z_*=(x_*,T)$.
4. Define the parabolic rescaling.
5. Record either the fixed-terminal sequence format or the optional moving
   Type I center-time-scale format.
6. Verify geometric admissibility of all cylinders produced by the selected
   format.
7. Verify the distributional equation and local energy inequality after
   change of variables.
8. Mark that finite entry-admissibility is still required before a terminal
   CKN quantity is used as finite data.

## Estimate Step $B_{\mathrm{Rec}_N}$

The estimate step is the rescaling-invariance lemma. It resolves the exact
obstruction consisting of a selected singular point without a written
parabolic rescaling formula.

## Failure Case

Failure name: terminal point unavailable.

Analytic meaning: either there is no singular point at time $T$, which is a
local regularity conclusion for this analysis, or the problem data do not
specify a singular time.

## Refinement Step

Allowed refinements:

1. choose a point $x_*\in\Sigma(T)$ if the set is nonempty;
2. shrink to a backward cylinder contained in the suitable region;
3. fix the parabolic scaling around $z_*$;
4. discard finitely many sequence indices until every fixed rescaled compact
   cylinder has admissible physical image.

Progress measure: the terminal time and point are fixed once. Repeated
refinement must either shrink the local cylinder or record non-applicability.

## Data Passed Forward

The next proof step is `C_mu`. The data passed forward are

$$
\Gamma_{\mathrm{Rec}_N}
=
\Gamma_{D_E}\cup
\{z_*=(x_*,T),\ x_*\in\Sigma(T),\ x_n,t_n,\lambda_n,
u^\lambda,\ p^\lambda,\ Q_r(z_*),\text{geometrically admissible cylinders}\}.
$$

This is admissible for `C_mu` because `C_mu` needs a singular point,
geometrically admissible shrinking cylinders, and scale-invariant quantities
near that point; the finite-entry part of admissibility is inherited from
`D_E` when those quantities are evaluated. It is also admissible for the
optional Type I entry route because the
center-time-scale record contains the data required to rescale a blow-up
sequence.

---

# 6. `C_mu` -- Local Concentration Alternative

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

Work near $z_*=(x_*,T)$ with $(u,p)$ suitable. The local quantities are
$C(u;z_*,r)$ and $D(p;z_*,r)$.

### Standing Assumptions

The incoming record contains $x_*\in\Sigma(T)$, and the `D_E` data supply
the suitable weak solution structure. In the terminal packet proved by the
first nodes, the incoming record also contains one finite entry radius
$r_0>0$ at $z_*$. Lemma D_E.3 then supplies every smaller
entry-admissible radius. If the record supplies only geometric admissibility
and no finite terminal entry radius, the proof cannot claim a positive
concentration sequence; it stops at the entry obstruction.

### Objects Inspected

Inspect $C(u;z_*,r)+D(p;z_*,r)$ and $C(u;z_*,r)$ as entry-admissible radii
$r\downarrow0$.

### Dependencies Used

The singular point comes from `Rec_N`. The CKN quantities and regularity
criterion come from `H0` and `D_E`.

### Local Obstruction Predicate

$P_{C_\mu}$ is the disjunction

$$
P_{C_\mu}=P_{\mathrm{entry}}\vee P_{\mathrm{regularity}}.
$$

Here $P_{\mathrm{entry}}$ means that the record lacks one finite terminal
entry radius and therefore lacks the arbitrarily small entry-admissible radii
needed to evaluate CKN quantities. The branch then returns to `D_E` as a
terminal finite-entry obstruction. The predicate
$P_{\mathrm{regularity}}$ means that some entry-admissible radius satisfies

$$
C(u;z_*,r)+D(p;z_*,r)<\varepsilon_0
\quad\text{or}\quad
C(u;z_*,r)<\varepsilon_v.
$$

Then the corresponding CKN theorem gives local boundedness in $Q_{r/2}(z_*)$
and contradicts the singular branch. If neither obstruction occurs, positive
concentration follows.

### Local Lemmas to Prove

**Lemma C_mu.1 -- One finite entry radius supplies all CKN test scales.**
Assume there is $r_0>0$ such that $Q_{r_0}(z_*)$ is geometrically admissible
and

$$
A(u;z_*,r_0)+C(u;z_*,r_0)+D(p;z_*,r_0)+E(u;z_*,r_0)<\infty.
$$

Then every $0<r<r_0$ is entry-admissible and the CKN quantities
$C(u;z_*,r)$, $D(p;z_*,r)$ are finite.

**Proof.** This is exactly Lemma D_E.3. The only extra point needed here is
logical: because $C$ and $D$ are finite at each smaller radius, the CKN
smallness hypotheses are meaningful numerical statements at those radii.

**Lemma C_mu.2 -- Singular points force CKN lower bounds at every smaller
entry radius.**
Assume $x_*\in\Sigma(T)$ and assume the finite terminal entry radius $r_0$
from Lemma C_mu.1. Then for every $0<r<r_0$,

$$
C(u;z_*,r)+D(p;z_*,r)\ge\varepsilon_0,
\qquad
C(u;z_*,r)\ge\varepsilon_v.
$$

**Proof.** Fix $0<r<r_0$. By Lemma C_mu.1 this radius is entry-admissible, so
$(u,p)$ is suitable on the open backward cylinder, the compactly supported
local energy inequality is available on its truncated subcylinders, and
$C(u;z_*,r)$ and $D(p;z_*,r)$ are finite.

Suppose first that

$$
C(u;z_*,r)+D(p;z_*,r)<\varepsilon_0.
$$

The Caffarelli--Kohn--Nirenberg criterion then gives local boundedness of
$u$ in $Q_{r/2}(z_*)$. This contradicts $x_*\in\Sigma(T)$, because
$Q_{r/2}(z_*)$ is a geometrically admissible backward cylinder ending at
$(x_*,T)$. Therefore

$$
C(u;z_*,r)+D(p;z_*,r)\ge\varepsilon_0.
$$

The velocity-only estimate is identical. If

$$
C(u;z_*,r)<\varepsilon_v,
$$

then the velocity-only CKN theorem gives local boundedness in
$Q_{r/2}(z_*)$, again contradicting $x_*\in\Sigma(T)$. Hence

$$
C(u;z_*,r)\ge\varepsilon_v.
$$

Since $r$ was arbitrary, both inequalities hold for every $0<r<r_0$.

**Lemma C_mu.3 -- Original-scale concentration sequence.**
Let $r_n\downarrow0$ with $0<r_n<r_0$, for instance
$r_n=2^{-n}r_0$. Then each $r_n$ is entry-admissible and

$$
C(u;z_*,r_n)+D(p;z_*,r_n)\ge\varepsilon_0,
\qquad
C(u;z_*,r_n)\ge\varepsilon_v.
$$

**Proof.** Lemma D_E.3 makes each $r_n$ entry-admissible because
$0<r_n<r_0$. Applying Lemma C_mu.2 at $r=r_n$ gives

$$
C(u;z_*,r_n)+D(p;z_*,r_n)
\ge\varepsilon_0,
$$

and

$$
C(u;z_*,r_n)\ge\varepsilon_v.
$$

These are the original-scale concentration bounds passed to `PS1`, where the
rescaled fixed-cylinder packet is formed.

### Specific Estimate

The decisive estimate is the contrapositive of CKN regularity:

$$
x_*\in\Sigma(T)
\quad\text{and one finite terminal entry radius exists}
\quad\Longrightarrow\quad
C(u;z_*,r)+D(p;z_*,r)\ge\varepsilon_0
\quad\text{and}\quad
C(u;z_*,r)\ge\varepsilon_v
$$

for every $0<r<r_0$.

### Practical Verification Steps

1. Verify $x_*\in\Sigma(T)$.
2. Verify one finite terminal entry radius $r_0>0$.
3. Use Lemma D_E.3 to mark every $0<r<r_0$ as entry-admissible.
4. If some such $r$ has $C+D<\varepsilon_0$ or $C<\varepsilon_v$, apply the
   corresponding CKN theorem and close the singular branch.
5. Otherwise record the lower bounds for every $0<r<r_0$.
6. Select any $r_n\downarrow0$ with $0<r_n<r_0$ and record the original-scale
   lower bounds for the selected sequence.

## Estimate Step $B_{C_\mu}$

The estimate proposition is CKN regularity plus the finite-entry inheritance
from Lemma D_E.3. Smallness of $C+D$ or $C$ at one entry-admissible scale
closes the singular branch by regularity. If the point is singular and one
terminal finite-entry radius is known, Lemma D_E.3 supplies every smaller test
scale and the same CKN criterion forces positive concentration at each of
them.

## Failure Case

Failure name: entry obstruction or persistent local CKN concentration.

Analytic meaning: either the branch lacks finite terminal entry and cannot
evaluate the concentration quantities, or the singular point cannot be removed
by the CKN smallness criterion and therefore carries a positive
scale-invariant concentration sequence.

## Refinement Step

Allowed refinements are:

1. return to `D_E` and prove one finite terminal entry radius;
2. close the branch if a CKN smallness witness is found;
3. otherwise select a concrete sequence $r_n\downarrow0$, such as
   $r_n=2^{-n}r_0$, and pass it to `PS1`.

The progress measure is either a new finite-entry datum, a regularity closure,
or fixed selection of the concentration sequence.

## Data Passed Forward

The next proof step is `PS1`; the separate audit step that would only repeat
the CKN contrapositive is removed, and that argument is merged here. The data
passed forward are

$$
\Gamma_{C_\mu}
=
\Gamma_{\mathrm{Rec}_N}\cup
\{r_0,r_n,\ C(u;z_*,r_n)+D(p;z_*,r_n)\ge\varepsilon_0,
C(u;z_*,r_n)\ge\varepsilon_v\}.
$$

---

# 7. `PS1` -- Positive Local Concentration Sequence

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

Work with $(u,p)$ near $z_*$. The node creates normalized unknowns
$(u_n,p_n)$ on $Q_1$ by parabolic scaling.

### Standing Assumptions

The incoming record from `C_mu` contains $x_*\in\Sigma(T)$, a finite terminal
entry radius $r_0>0$, a sequence $r_n\downarrow0$ with $0<r_n<r_0$, and the
original-scale lower bounds

$$
C(u;z_*,r_n)+D(p;z_*,r_n)\ge\varepsilon_0,
\qquad
C(u;z_*,r_n)\ge\varepsilon_v.
$$

All selected scales are entry-admissible by Lemma D_E.3. Thus the open
backward cylinders lie in the suitable region, the full scale-invariant
integrals are finite, and every compact truncated window used for tests is
compactly contained below the terminal face.

### Objects Inspected

Inspect the sequence $r_n$, the rescaled fields, and the fixed-cylinder
integrals of $|u_n|^3$ and $|p_n-(p_n)_{B_1}|^{3/2}$.

### Dependencies Used

The singular-entry lower bounds come from `C_mu`; scaling and pressure means
come from `H0` and `Rec_N`; local suitability and finite entry come from
`D_E`.

### Local Obstruction Predicate

$P_{\mathrm{PS1}}$ holds if the proof has not produced a shrinking scale
sequence with a nonzero normalized concentration lower bound.

### Local Lemmas to Prove

**Lemma PS1.1 -- Selection of a concentration sequence.**
For the sequence $r_n\downarrow0$ passed from `C_mu`,

$$
C(u;z_*,r_n)+D(p;z_*,r_n)\ge\varepsilon_0.
$$

The same sequence also satisfies

$$
C(u;z_*,r_n)\ge\varepsilon_v.
$$

**Proof.** Each selected radius $r_n$ is an entry-admissible radius at the
verified singular point $z_*$. Lemma C_mu.3 gives

$$
C(u;z_*,r_n)+D(p;z_*,r_n)\ge\varepsilon_0.
$$

For the velocity-only criterion, the same singular-point input gives
$C(u;z_*,r_n)\ge\varepsilon_v$ on the selected sequence. These are precisely
the two lower bounds stated in the lemma.

**Lemma PS1.2 -- Rescaled fixed-cylinder lower bound.**
For

$$
u_n(y,s)=r_nu(x_*+r_ny,T+r_n^2s),
\qquad
p_n(y,s)=r_n^2p(x_*+r_ny,T+r_n^2s),
$$

the pair $(u_n,p_n)$ is suitable on every compact truncated cylinder

$$
B_R\times[-R^2,-\delta],
\qquad R<\infty,\quad \delta>0,
$$

for all sufficiently large $n$, and one has

$$
\int_{Q_1}
\left(|u_n|^3+|p_n-(p_n)_{B_1}(s)|^{3/2}\right)\,dy\,ds
=
C(u;z_*,r_n)+D(p;z_*,r_n).
$$

In particular,

$$
C(u_n;0,1)+D(p_n;0,1)\ge\varepsilon_0,
\qquad
C(u_n;0,1)\ge\varepsilon_v.
$$

Equivalently, with

$$
\eta_0=\varepsilon_0,
\qquad
\eta_v=\varepsilon_v,
$$

the node records the two lower bounds

$$
\int_{Q_1}|u_n|^3\,dy\,ds\ge\eta_v,
$$

and

$$
\int_{Q_1}
\left(|u_n|^3+|p_n-(p_n)_{B_1}(s)|^{3/2}\right)\,dy\,ds
\ge\eta_0.
$$

These are lower bounds only. They do not imply any uniform upper bound on
$A(u_n;0,R)$, $E(u_n;0,R)$, $C(u_n;0,R)$, or $D(p_n;0,R)$ for any fixed
$R$. Every compactness argument after this node must separately prove the
upper bounds it uses.

**Proof.** Fix $R<\infty$ and $\delta>0$. The compact rescaled window maps to

$$
B_{Rr_n}(x_*)\times[T-R^2r_n^2,T-\delta r_n^2].
$$

Since $r_n\downarrow0$ and $0<r_n<r_0$, this physical window is compactly
contained in the terminal suitable region for all sufficiently large $n$.
Lemma Rec_N.2 gives suitability of $(u_n,p_n)$ on the fixed truncated window.
The top face $s=0$ is not part of the compact window.

For the fixed unit cylinder, use $x=x_*+r_ny$, $t=T+r_n^2s$. Then

$$
\int_{Q_1}|u_n|^3\,dy\,ds
=r_n^{-2}\iint_{Q_{r_n}(z_*)}|u|^3\,dx\,dt
=C(u;z_*,r_n).
$$

The pressure mean transforms as

$$
(p_n)_{B_1}(s)
=r_n^2(p)_{B_{r_n}(x_*)}(T+r_n^2s),
$$

so the pressure integral transforms exactly into $D(p;z_*,r_n)$.
Combining this identity with Lemma PS1.1 gives the two normalized lower
bounds.

### Specific Estimate

The decisive identities are the scaling equalities

$$
\int_{Q_1}|u_n|^3\,dy\,ds
=
C(u;z_*,r_n)
\ge\varepsilon_v,
$$

and

$$
\iint_{Q_1}\left(|u_n|^3+|p_n-(p_n)_{B_1}|^{3/2}\right)
=
C(u;z_*,r_n)+D(p;z_*,r_n),
$$

combined with the lower bounds from `C_mu`.

### Practical Verification Steps

1. Choose entry-admissible $r_n\downarrow0$.
2. Define $u_n,p_n$ by parabolic scaling.
3. Verify the pressure mean scaling.
4. Change variables in the $C$ and $D$ integrals.
5. Record the lower-bound constant on $Q_1$.
6. Record explicitly that no compactness upper bound has been proved at this
   node.

## Estimate Step $B_{\mathrm{PS1}}$

The estimate step is the scaling identity in Lemma PS1.2.

## Failure Case

Failure name: concentration-sequence normalization failure.

Analytic meaning: the proof has a singular point but has not produced a
normalized fixed-cylinder sequence carrying the local lower bound needed for
profile extraction.

## Refinement Step

Allowed refinements:

1. pass to an admissible subsequence of radii;
2. shrink radii so all cylinders lie in the suitable region;
3. fix the pressure mean convention before scaling.

Progress measure: subsequence selection or strict scale shrinkage.

## Data Passed Forward

The next proof step is `PS2`, with

$$
\Gamma_{\mathrm{PS1}}
=
\Gamma_{C_\mu}
\cup
\{u_n,p_n,Q_1,\eta_0=\varepsilon_0,\eta_v=\varepsilon_v,
C(u_n;0,1)+D(p_n;0,1)\ge\eta_0,\ C(u_n;0,1)\ge\eta_v,
\text{no compactness upper bounds yet}\}.
$$

---

## Continuous Proof of the First Five-Node Packet

This paragraph collects `H0`, `D_E`, `Rec_N`, `C_mu`, and `PS1` into one
continuous proof. It is only an entry-and-concentration proof near a selected
singular point. It does not exclude singularities, does not create a
whole-space endpoint hypothesis, and does not use any later compactness or
Liouville theorem.

Let $\mathcal O\subset\mathbb R^3\times\mathbb R$ be open, and let $(u,p)$ be
a suitable weak solution in $\mathcal O$. Thus

$$
u\in L^\infty_tL^2_{x,\mathrm{loc}}(\mathcal O)
\cap L^2_tH^1_{x,\mathrm{loc}}(\mathcal O),
\qquad
p\in L^{3/2}_{\mathrm{loc}}(\mathcal O),
$$

the Navier--Stokes equations and $\nabla\cdot u=0$ hold in distributions, and
the local energy inequality holds for every nonnegative compactly supported
test function in $\mathcal O$. The pressure is understood modulo functions of
time, and the quantity $D$ always uses the spatial mean subtraction on the ball
where it is evaluated.

Fix

$$
z_*=(x_*,T),
$$

and assume

$$
x_*\in\Sigma(T).
$$

This means that $u$ is not locally bounded in any sufficiently small
geometrically admissible backward cylinder ending at $(x_*,T)$. Also assume
there is one terminal finite-entry radius $r_0>0$ such that
$Q_{r_0}(z_*)$ is geometrically admissible and

$$
A(u;z_*,r_0)+C(u;z_*,r_0)+D(p;z_*,r_0)+E(u;z_*,r_0)<\infty.
$$

First, the analytic quantities are legitimate. On any compactly contained
nonterminal cylinder, the local energy class gives $A<\infty$ and
$E<\infty$. Lemma H0.1 gives $u\in L^3_{\mathrm{loc}}$, hence $C<\infty$.
Lemma H0.2 gives

$$
\|p-(p)_B(t)\|_{L^{3/2}(B)}
\le C\|p\|_{L^{3/2}(B)}
$$

for a.e. $t$, hence $D<\infty$ on compactly contained cylinders. The identity

$$
(p+a(t))-(p+a(t))_B=p-(p)_B
$$

shows that $D$ is independent of the time-dependent pressure gauge.

Second, the single finite terminal entry radius gives every smaller terminal
entry radius. Let $0<r<r_0$. Since $Q_r(z_*)\subset Q_{r_0}(z_*)$, geometric
admissibility is inherited. The velocity quantities satisfy

$$
A(u;z_*,r)
\le
\frac{r_0}{r}A(u;z_*,r_0),
$$

$$
C(u;z_*,r)
\le
\left(\frac{r_0}{r}\right)^2C(u;z_*,r_0),
$$

and

$$
E(u;z_*,r)
\le
\frac{r_0}{r}E(u;z_*,r_0).
$$

For the pressure term, set

$$
a(t)=(p)_{B_{r_0}(x_*)}(t),
\qquad
q(x,t)=p(x,t)-a(t).
$$

The finite $D$ assumption at radius $r_0$ says precisely that

$$
q\in L^{3/2}(Q_{r_0}(z_*)).
$$

For $B_r(x_*)\subset B_{r_0}(x_*)$,

$$
p-(p)_{B_r(x_*)}=q-(q)_{B_r(x_*)}.
$$

Using Lemma H0.2 on $q$ gives

$$
\int_{B_r(x_*)}|p-(p)_{B_r(x_*)}|^{3/2}\,dx
\le
C\int_{B_r(x_*)}|q|^{3/2}\,dx
$$

for a.e. time. Integrating over $(T-r^2,T)$ and enlarging the integration
region to $Q_{r_0}(z_*)$ gives

$$
D(p;z_*,r)
\le
C\left(\frac{r_0}{r}\right)^2D(p;z_*,r_0)<\infty.
$$

Thus every $0<r<r_0$ is entry-admissible.

Third, CKN smallness is impossible at any smaller radius. Fix $0<r<r_0$.
Since $r$ is entry-admissible, the quantities
$C(u;z_*,r)$ and $D(p;z_*,r)$ are finite and the CKN hypothesis is a meaningful
statement. If

$$
C(u;z_*,r)+D(p;z_*,r)<\varepsilon_0,
$$

then Caffarelli--Kohn--Nirenberg regularity gives local boundedness of $u$ in
$Q_{r/2}(z_*)$. This contradicts $x_*\in\Sigma(T)$. Therefore

$$
C(u;z_*,r)+D(p;z_*,r)\ge\varepsilon_0
$$

for every $0<r<r_0$. The velocity-only criterion gives the second lower bound:
if

$$
C(u;z_*,r)<\varepsilon_v,
$$

then the velocity-only CKN theorem gives local boundedness in
$Q_{r/2}(z_*)$, again contradicting $x_*\in\Sigma(T)$. Hence

$$
C(u;z_*,r)\ge\varepsilon_v
$$

for every $0<r<r_0$.

Fourth, choose any sequence

$$
r_n\downarrow0,
\qquad
0<r_n<r_0.
$$

For example, one may take $r_n=2^{-n}r_0$. By the previous paragraph,

$$
C(u;z_*,r_n)+D(p;z_*,r_n)\ge\varepsilon_0,
\qquad
C(u;z_*,r_n)\ge\varepsilon_v.
$$

Define the rescaled fields

$$
u_n(y,s)=r_nu(x_*+r_ny,T+r_n^2s),
\qquad
p_n(y,s)=r_n^2p(x_*+r_ny,T+r_n^2s).
$$

For fixed $R<\infty$ and $\delta>0$, the compact truncated rescaled cylinder

$$
B_R\times[-R^2,-\delta]
$$

maps to

$$
B_{Rr_n}(x_*)\times[T-R^2r_n^2,T-\delta r_n^2].
$$

For all sufficiently large $n$, this physical cylinder is compactly contained
in the suitable region below the terminal time. Thus the rescaling lemma
applies on this compact window: $(u_n,p_n)$ is suitable there, the
distributional equation and divergence condition are preserved, and the local
energy inequality is preserved by the test-function pullback

$$
\phi(X,S)=
\psi\left(\frac{X-x_*}{r_n},\frac{S-T}{r_n^2}\right).
$$

No compact window touching $s=0$ is asserted.

Finally, the scale-invariant quantities are exactly preserved. For the velocity
term,

$$
\int_{Q_1}|u_n|^3\,dy\,ds
=
r_n^{-2}\iint_{Q_{r_n}(z_*)}|u|^3\,dx\,dt
=
C(u;z_*,r_n).
$$

For pressure, the mean transforms as

$$
(p_n)_{B_1}(s)
=
r_n^2(p)_{B_{r_n}(x_*)}(T+r_n^2s).
$$

Therefore

$$
\int_{Q_1}|p_n-(p_n)_{B_1}(s)|^{3/2}\,dy\,ds
=
D(p;z_*,r_n).
$$

Combining these identities with the original-scale lower bounds gives

$$
C(u_n;0,1)+D(p_n;0,1)
=
C(u;z_*,r_n)+D(p;z_*,r_n)
\ge\varepsilon_0,
$$

and

$$
C(u_n;0,1)=C(u;z_*,r_n)\ge\varepsilon_v.
$$

This completes the continuous first-five-node proof. The output is exactly the
entry-and-concentration packet

$$
\{z_*,r_0,r_n,u_n,p_n,\text{compact truncated suitability},
C(u_n;0,1)+D(p_n;0,1)\ge\varepsilon_0,\ C(u_n;0,1)\ge\varepsilon_v\}.
$$

Nothing stronger is concluded at this stage.

---

# 8. `PS2` -- Center and Local Frame

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknowns remain the NS3D velocity and pressure, written either as
$(u_n,p_n)$ in Type I variables or as a locally represented pair in a Type II
frame. The node is purely geometric: it verifies that the change of variables
used later is admissible and does not erase the concentration lower bound.

### Standing Assumptions

The incoming record contains the two `PS1` lower bounds

$$
\int_{Q_1}|u_n|^3\,dy\,ds\ge\eta_v,
$$

and

$$
\int_{Q_1}
\left(|u_n|^3+|p_n-(p_n)_{B_1}(s)|^{3/2}\right)\,dy\,ds
\ge \eta_0.
$$

No compactness upper bound is part of the incoming data from `PS1`.

### Objects Inspected

Inspect the center $x_*$, the rescaled cylinders, and, in the Type II
single-core branch, the maps $X_{\rm pre}$, $\Lambda_{\rm pre}$, $t_c$, and
$\Phi_{\rm pre}$.

### Dependencies Used

The singular point comes from `Rec_N`; the concentration sequence and lower
bound come from `PS1`; the scaling map comes from `H0`. No rate classification
from `PS3` is used.

### Local Obstruction Predicate

$P_{\mathrm{PS2}}$ holds if the concentration lower bound has not been attached
to a stable local center/frame. In Type I this can only happen if the center
$x_*$ or the rescaling from `PS1` was not actually recorded. In a single-core
Type II branch it also holds if the frame loses invertibility or if
$\Lambda_{\rm pre}$ degenerates on the local window, or if the retained packet
has not been shown to lie in the image of the accepted compact frame cylinder.

### Local Lemmas to Prove

**Lemma PS2.1 -- Type I centering preserves the concentration lower bound.**
In the Type I terminal-point analysis, the concentration center is $x_*$. The
rescaled variables centered at $x_*$ preserve both fixed-cylinder lower bounds
from `PS1`; there is no additional frame.

**Proof.** The variables in `PS1` are

$$
u_n(y,s)=r_nu(x_*+r_ny,T+r_n^2s),
\qquad
p_n(y,s)=r_n^2p(x_*+r_ny,T+r_n^2s).
$$

The lower bounds in `PS1` are the change-of-variables identities from
$Q_{r_n}(x_*,T)$ to $Q_1(0,0)$. No additional recentering is applied, so the
Jacobian computation

$$
dy\,ds=r_n^{-5}\,dx\,dt,\qquad |u_n|^3=r_n^3|u|^3
$$

and the pressure mean identity

$$
(p_n)_{B_1}(s)=r_n^2(p)_{B_{r_n}(x_*)}(T+r_n^2s)
$$

preserve the normalized lower bounds with center $x_*$:

$$
\int_{Q_1}|u_n|^3\,dy\,ds\ge\eta_v,
$$

and

$$
\int_{Q_1}
\left(|u_n|^3+|p_n-(p_n)_{B_1}(s)|^{3/2}\right)\,dy\,ds
\ge\eta_0.
$$

**Lemma PS2.2 -- Local Type II frame under a single-core hypothesis.**
Let a local Type II single-core analysis select a physical core
$(x_c(t),\rho_c(t))$ and, after the time change
$dt_c/d\tau=\Lambda_{\rm pre}(\tau)^2$, has

$$
X_{\rm pre}(\tau)=x_c(t_c(\tau)),
\qquad
\Lambda_{\rm pre}(\tau)=\rho_c(t_c(\tau)),
$$

with

$$
0<\Lambda_{\min}\le \Lambda_{\rm pre}(\tau)
\le\Lambda_{\max}<\infty,
\qquad
X_{\rm pre},\Lambda_{\rm pre}\in W^{1,1}_{\rm loc}.
$$

Then $\Phi_{\rm pre}$ is a legitimate local change of variables on each
compact renormalized cylinder. Its spatial Jacobian is
$\Lambda_{\rm pre}(\tau)^3$, and its spatial inverse is uniformly Lipschitz on
compact $\tau$-intervals.

**Proof.** For fixed $\tau$,

$$
\nabla_Y\Phi_{\rm pre}(\tau,\cdot)
=\Lambda_{\rm pre}(\tau)I,
\qquad
\det\nabla_Y\Phi_{\rm pre}
=\Lambda_{\rm pre}(\tau)^3.
$$

The lower and upper bounds on $\Lambda_{\rm pre}$ imply

$$
\Lambda_{\min}|Y_1-Y_2|
\le
|\Phi_{\rm pre}(\tau,Y_1)-\Phi_{\rm pre}(\tau,Y_2)|
\le
\Lambda_{\max}|Y_1-Y_2|.
$$

Thus the spatial map is bi-Lipschitz on each compact $\tau$-window.
$W^{1,1}_{\rm loc}$ regularity of $X_{\rm pre}$, $\Lambda_{\rm pre}$, and
$t_c$ gives an admissible time-dependent frame for distributional changes of
variables.

**Lemma PS2.3 -- Frame transfer does not erase a retained packet.**
Assume a compact renormalized cylinder
$\mathcal Q=K\times(\tau_1,\tau_2)$ is mapped by $\Phi_{\rm pre}$ into the
physical suitable region, and assume the retained physical packet satisfies

$$
\int_{\Phi_{\rm pre}(\mathcal Q)}
|u(x,t)|^3\,dx\,dt\ge\eta_{\mathcal Q}>0.
$$

Define the represented velocity by

$$
V(Y,\tau)=\Lambda_{\rm pre}(\tau)
u(X_{\rm pre}(\tau)+\Lambda_{\rm pre}(\tau)Y,t_c(\tau)).
$$

Then

$$
\int_{\mathcal Q}|V(Y,\tau)|^3\,dY\,d\tau
\ge
\Lambda_{\max}^{-2}
\int_{\Phi_{\rm pre}(\mathcal Q)}
|u(x,t)|^3\,dx\,dt
\ge
\Lambda_{\max}^{-2}\eta_{\mathcal Q}.
$$

The same controlled transfer holds for the mean-subtracted pressure density
with the normalization $P=\Lambda_{\rm pre}^2p$: if spatial means are taken
over $K$ in represented variables and over
$X_{\rm pre}(\tau)+\Lambda_{\rm pre}(\tau)K$ in physical variables, then

$$
\int_{\mathcal Q}|P(Y,\tau)-(P)_K(\tau)|^{3/2}\,dY\,d\tau
\ge
\Lambda_{\max}^{-2}
\int_{\Phi_{\rm pre}(\mathcal Q)}
|p(x,t)-(p)_{X_{\rm pre}+\Lambda_{\rm pre}K}(t)|^{3/2}\,dx\,dt.
$$

**Proof.** The time change gives $dt=\Lambda_{\rm pre}(\tau)^2\,d\tau$ and the
spatial map gives $dx=\Lambda_{\rm pre}(\tau)^3\,dY$. Since
$u=\Lambda_{\rm pre}^{-1}V$,

$$
|V|^3\,dY\,d\tau
=
\Lambda_{\rm pre}(\tau)^{-2}|u|^3\,dx\,dt.
$$

The upper bound $\Lambda_{\rm pre}\le\Lambda_{\max}$ gives the displayed
lower bound. For pressure,

$$
(P)_K(\tau)
=
\Lambda_{\rm pre}(\tau)^2
(p)_{X_{\rm pre}(\tau)+\Lambda_{\rm pre}(\tau)K}(t_c(\tau)),
$$

because the spatial Jacobian cancels in the average. Hence
$P-(P)_K=\Lambda_{\rm pre}^2(p-(p)_{X_{\rm pre}+\Lambda_{\rm pre}K})$, and
the same factor $\Lambda_{\rm pre}^{-2}$ appears in the represented
$L^{3/2}$ density after changing variables. Thus a retained packet is not
lost under the frame, provided the packet is actually contained in the frame
image and the pressure means are taken over corresponding spatial sets. A
Type II frame is not accepted merely because it is written down. It is
accepted only after the retained concentration packet is shown to lie inside
the image of a compact frame cylinder on which the Jacobian bounds hold.

### Specific Estimate

The decisive estimate is the bi-Lipschitz and Jacobian bound

$$
0<\Lambda_{\min}^3
\le
\det\nabla_Y\Phi_{\rm pre}
\le
\Lambda_{\max}^3<\infty.
$$

Together with Lemma PS2.3, it guarantees that local $L^p$ norms,
distributional identities, and retained concentration packets transform with
controlled constants on compact windows. It does not supply any compactness
upper bound.

### Practical Verification Steps

1. In the Type I branch, verify that the center is exactly $x_*$.
2. Check that the lower bound from `PS1` is written in variables centered at
   $x_*$.
3. In the Type II single-core branch, verify
   $0<\Lambda_{\min}\le\Lambda_{\rm pre}\le\Lambda_{\max}$ locally.
4. Verify $X_{\rm pre},\Lambda_{\rm pre},t_c\in W^{1,1}_{\rm loc}$.
5. Compute the Jacobian and inverse Lipschitz constants.
6. Verify that any retained Type II packet lies in the image of the accepted
   compact frame cylinder.
7. Record whether the branch is Type I centered, Type II single-core framed,
   or not representable by a single-core frame.

## Estimate Step $B_{\mathrm{PS2}}$

The estimate step is the frame-admissibility estimate in Lemma PS2.2 and the
retained-packet transfer in Lemma PS2.3, together with the Type I centering
identity in Lemma PS2.1.

## Failure Case

Failure name: local frame failure.

Analytic meaning: the concentration packet cannot be represented in the
proposed center/frame without losing local invertibility or the positive
concentration lower bound.

## Refinement Step

Allowed refinements:

1. pass to a subsequence with a stable center;
2. shrink the local time window so the frame bounds hold;
3. in the Type II branch, replace the proposed frame by a single-core frame
   satisfying the bi-Lipschitz bounds;
4. if no single-core frame exists, record non-applicability of the single-core
   subcase and leave the branch for later multicore, cascade, or residual
   analysis.

Progress measure: either a center is fixed once, the time window is strictly
shrunk, or the single-core subcase is explicitly marked non-applicable.

## Data Passed Forward

The next proof step is `PS3`. The data passed forward are

$$
\Gamma_{\mathrm{PS2}}^{I}
=
\Gamma_{\mathrm{PS1}}
\cup
\{\text{center }x_*,\text{ identity Type I frame},
\int_{Q_1}|u_n|^3\ge\eta_v,\
\int_{Q_1}\left(|u_n|^3+
|p_n-(p_n)_{B_1}(s)|^{3/2}\right)\ge\eta_0\}
$$

in the Type I centered branch, or

$$
\Gamma_{\mathrm{PS2}}^{II}
=
\Gamma_{\mathrm{PS1}}
\cup
\{\Phi_{\rm pre},\Lambda_{\min},\Lambda_{\max},
X_{\rm pre},\Lambda_{\rm pre},t_c,
\text{retained packet in the frame image}\}
$$

in the Type II single-core branch. These data are admissible for `PS3` because
scale and rate classification require the center/frame in which the local
scale is measured.

---

# 9. `PS3` -- Scale and Local Rate Classification

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

Work near $z_*=(x_*,T)$ with the centered rescaled sequence. The unknowns are
the rescaled velocity and pressure. The rate classification is local in a
backward cylinder and is measured by the scale-invariant Type I quantity
$\sqrt{T-t}\|u(t)\|_{L^\infty}$. This essential supremum is understood as an
extended real number; no local boundedness is assumed before the Type I
alternative has been verified.

### Standing Assumptions

The incoming record contains the center/frame from `PS2` and both lower bounds
from `PS1`. In the Type I compact-window sublemma, the lemma hypotheses must
contain the velocity lower bound and the no-escape estimate from `D_E`. The
local Type I rate bound alone does not supply no-escape.

### Objects Inspected

Inspect the scale sequence, the local $L^\infty$ rate, and the distribution of
the $L^3$ concentration in the interval $(-1,0)$ after rescaling.

### Dependencies Used

The lower bound comes from `PS1`; the center/frame comes from `PS2`; the
scaling law comes from `H0`; the Type I no-escape estimate, when used, comes
from `D_E`.

### Local Obstruction Predicate

$P_{\mathrm{PS3}}$ holds exactly when the proof cannot say which local rate
alternative applies, or when the Type I branch has not isolated a compact
negative-time window carrying nonzero concentration.

### Local Lemmas to Prove

**Lemma PS3.1 -- Scaling invariance of $A,C,D,E$.**
Under the parabolic scaling

$$
u^\lambda(x,t)=\lambda u(x_*+\lambda x,T+\lambda^2t),
\qquad
p^\lambda(x,t)=\lambda^2p(x_*+\lambda x,T+\lambda^2t),
$$

one has

$$
A(u^\lambda;0,r)=A(u;z_*,\lambda r),
\quad
C(u^\lambda;0,r)=C(u;z_*,\lambda r),
$$

and likewise

$$
D(p^\lambda;0,r)=D(p;z_*,\lambda r),
\qquad
E(u^\lambda;0,r)=E(u;z_*,\lambda r).
$$

**Proof.** With $X=x_*+\lambda x$ and $S=T+\lambda^2t$,
$dx\,dt=\lambda^{-5}dX\,dS$. For $C$, the factor is
$\lambda^3\lambda^{-5}=\lambda^{-2}$, which is exactly the scale factor in
$(\lambda r)^{-2}$. For $D$, the pressure mean transforms as

$$
(p^\lambda)_{B_r}(t)
=\lambda^2(p)_{B_{\lambda r}(x_*)}(S),
$$

and the factor $\lambda^3\lambda^{-5}=\lambda^{-2}$ again matches the
definition of $D$. For $A$, the factor in
$\int_{B_r}|u^\lambda|^2dx$ is $\lambda^{-1}$, which is canceled by the
change from $r^{-1}$ to $(\lambda r)^{-1}$. For $E$,
$|\nabla u^\lambda|^2dx\,dt$ has factor $\lambda^{-1}$, giving the same
invariance.

**Lemma PS3.2 -- Local Type I/Type II dichotomy.**
For $\rho>0$, define the extended-real quantity

$$
M(\rho)
=
\operatorname*{ess\,sup}_{T-\rho^2<t<T}
\sqrt{T-t}\,\|u(t)\|_{L^\infty(B_\rho(x_*))}.
$$

At $z_*=(x_*,T)$, exactly one of the following alternatives holds:

1. there exist $\rho>0$ and $M<\infty$ such that

   $$
   M(\rho)<\infty;
   $$

2. for every $\rho>0$,

   $$
   M(\rho)=\infty.
   $$

**Proof.** The local Type I predicate is

$$
\exists \rho>0:\quad M(\rho)<\infty.
$$

Its logical negation is

$$
\forall \rho>0:\quad M(\rho)=\infty,
$$

because $M(\rho)$ is allowed to be $+\infty$. The two alternatives are
therefore complementary by the law of excluded middle applied to this
extended-real predicate. No additional regularity is needed merely to state
the alternative.

**Lemma PS3.3 -- Type I compact subcylinder selection.**
The lemma hypotheses contain the Type I case, a concentration sequence $\lambda_k\downarrow0$, the
velocity lower bound

$$
\int_{-1}^{0}\int_{B_1}|u_k|^3\,dx\,dt\ge\eta_v,
$$

which must come from the velocity-only CKN contrapositive in `C_mu` and its
normalized form in `PS1`,

and the no-escape estimate

$$
\lim_{\sigma\downarrow0}\sup_k
\int_{-\sigma}^{0}\int_{B_1}|u_k|^3\,dx\,dt=0.
$$

The Type I branch may pass from terminal concentration on
$B_1\times(-1,0)$ to compact negative-time concentration on
$B_1\times(-1,-\sigma)$ only after this no-escape estimate is present in the
branch record. The local Type I bound alone does not justify this step. A safe
sufficient condition is the `D_E` no-escape hypothesis: a local Type I bound
near $z_*$ together with

$$
\sup_k A(u;z_*,\lambda_k)\le B<\infty.
$$

Then there are $\sigma\in(0,1)$ and $\eta_1>0$ such that

$$
\int_{-1}^{-\sigma}\int_{B_1}|u_k|^3\,dx\,dt\ge\eta_1
\qquad\text{for every }k.
$$

**Proof.** Choose $\sigma>0$ so that

$$
\sup_k\int_{-\sigma}^{0}\int_{B_1}|u_k|^3\,dx\,dt
\le\frac{\eta_v}{2}.
$$

Then

$$
\int_{-1}^{-\sigma}\int_{B_1}|u_k|^3\,dx\,dt
\ge
\eta_v-\frac{\eta_v}{2}
=\frac{\eta_v}{2}.
$$

Set $\eta_1=\eta_v/2$. The choice of $\sigma$ is made before passing to the
next node and is uniform in $k$, so the compact cylinder
$B_1\times(-1,-\sigma)$ carries a fixed nonzero amount of velocity
concentration. This prevents later compactness from using only terminal-time
tails near $t=0$.

**Lemma PS3.4 -- Type II scale-state table.**
If the local Type I predicate in Lemma PS3.2 fails, then the branch record must
contain the Type II scale-state table

$$
\mathfrak S_{\mathrm{II}}
=
\{
\mathrm{cascade},
\mathrm{absolute\ drift},
\mathrm{finite\ cost},
\mathrm{residual\ scale}
\}.
$$

Each entry is assigned one of the statuses

$$
\mathrm{verified},\qquad
\mathrm{excluded},\qquad
\mathrm{deferred},\qquad
\mathrm{undecided}.
$$

At `PS3`, a status may be `verified` or `excluded` only if the exact estimate
proving that status is already present in the branch record. Otherwise the
entry is marked `deferred` with destination `PS10`, `PS11`, `PS28`, or `PS34`
as appropriate.

**Proof.** The failure of the Type I predicate identifies the branch as local
Type II, but it does not by itself distinguish same-point scale cascade,
absolute scale drift, finite-cost transition, or residual scale behavior.
Those distinctions require later Type II estimates. Therefore the airtight
result of this step is a finite proof table whose entries are either backed by
estimates already present or explicitly deferred to the later nodes that prove
those estimates. This prevents the Type II route from silently assuming a
scale-collapse or finite-cost alternative before it has been checked.

### Specific Estimate

The decisive Type I estimate is the no-escape bound near $t=0$ combined with
positive $L^3$ concentration:

$$
\int_{-1}^{0}\int_{B_1}|u_k|^3\ge\eta_v,
\qquad
\lim_{\sigma\downarrow0}\sup_k
\int_{-\sigma}^{0}\int_{B_1}|u_k|^3=0.
$$

Together they force a nonzero amount of concentration to remain on a compact
negative-time cylinder.

In the Type II branch, the decisive conclusion is not an exclusion estimate at
this node but the finite table $\mathfrak S_{\mathrm{II}}$ with every
undischarged scale predicate assigned to a later proof node.

### Practical Verification Steps

1. Verify scale invariance of the quantities being used.
2. Evaluate the local Type I bound near $z_*$.
3. If Type I holds, record $\rho,M$.
4. If Type I fails, record the Type II alternative.
5. In the Type II branch, populate the scale-state table and attach every
   deferred entry to its later node.
6. In the Type I branch, use the no-escape estimate to choose $\sigma$ and
   $\eta_1$.
7. If the no-escape estimate is absent, do not form $Q_{\mathrm{act}}$; return
   to `D_E` or record the missing no-escape obligation.
8. Pass the selected scale and rate alternative to `PS4`.

## Estimate Step $B_{\mathrm{PS3}}$

The estimate step consists of the scaling identities, the local rate
dichotomy, the Type I compact-window lemma, and the Type II scale-table
assignment.

## Failure Case

Failure name: scale or rate-classification failure.

Analytic meaning: the proof has a concentration sequence but has not attached
it to a stable scale alternative suitable for pressure normalization and
compactness.

## Refinement Step

Allowed refinements:

1. pass to a subsequence of scales;
2. replace arbitrary scales by dyadic scales;
3. shrink the local cylinder on which the Type I estimate is evaluated;
4. in the Type I branch, choose a smaller terminal tail parameter $\sigma$;
5. in the Type II branch, record that later Type II nodes must supply the
   scale-collapse, finite-cost, or residual scale analysis.

Progress measure: scale subsequence selection or strict terminal-tail
shrinkage.

## Data Passed Forward

The next proof step is `PS4`. The data passed forward are

$$
\Gamma_{\mathrm{PS3}}
=
\Gamma_{\mathrm{PS2}}
\cup
\{\text{selected scale},\text{ Type I/Type II status},
\mathfrak S_{\mathrm{II}}\text{ in the Type II branch}\}.
$$

In the Type I branch it also contains the verified no-escape hypotheses,
$Q_{\mathrm{act}}=B_1\times(-1,-\sigma)$, and

$$
\int_{Q_{\mathrm{act}}}|u_k|^3\,dx\,dt\ge\eta_1.
$$

In the Type II branch it contains only the routing table
$\mathfrak S_{\mathrm{II}}$ unless a later-node estimate has already verified
or excluded one of its entries.

---

# 10. `PS4` -- Pressure Gauge and Local Modulation

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

Work with the normalized suitable sequence $(u_n,p_n)$, or with the
represented pair $(V_n,P_n)$ after a Type II frame has been accepted by `PS2`.
The pressure is normalized by subtracting a spatial mean or another
time-dependent function. In the Type II single-core case, the represented
velocity is also constrained by finitely many orthogonality conditions
selecting the center and scale.

In this node write $(U_n,\mathfrak p_n)$ for the current normalized variables:
$(U_n,\mathfrak p_n)=(u_n,p_n)$ in the centered branch and
$(U_n,\mathfrak p_n)=(V_n,P_n)$ in a represented-frame branch.

### Standing Assumptions

The incoming hypotheses contain local suitability, divergence-free velocity,
and the local rate/scale data from `PS3`. Pressure compactness is not inferred
from local velocity bounds. On every compact cylinder where pressure control is
used, the branch must contain one of the pressure inputs in Lemma PS4.2. In
the Type I whole-space branch, one possible input is the whole-space pressure
representation

$$
p=\mathcal R_i\mathcal R_j(u_i u_j)
$$

up to a function of time, whenever the whole-space reconstruction is used.

### Objects Inspected

Inspect $\mathfrak p_n-a_{n,R}(t)$ on compact cylinders, the localized pressure
equation, the harmonic remainder, the modulation constraint map
$\mathcal F(\lambda,c;V)$, and the repaired gauge-source list
$\mathcal E_{\rm gauge}$.

### Dependencies Used

The pressure convention comes from `H0`; local suitability comes from `D_E`;
scale/rate information comes from `PS3`; the frame for modulation comes from
`PS2`.

### Local Obstruction Predicate

$P_{\mathrm{PS4}}$ holds if pressure cannot be placed in a stable
mean-subtracted $L^{3/2}_{\rm loc}$ class, if the Type II representation
requires modulation parameters that have not been determined, or if the
repaired gauge omits a source term created by localization or modulation.

### Local Lemmas to Prove

**Lemma PS4.1 -- Pressure gauge invariance.**
If $(v,q)$ is suitable in a cylinder and
$\alpha(t)\in L^{3/2}_{\rm loc}$ depends only on time, then
$(v,q-\alpha(t))$ satisfies the same distributional NS equation, the same
divergence condition, and the same local energy inequality.

**Proof.** Since $\nabla\alpha(t)=0$, the momentum equation is unchanged in
distributions. In the local energy inequality the pressure term changes by a
multiple of

$$
-\int \alpha(t)v(x,t)\cdot\nabla\phi(x,t)\,dx\,dt.
$$

For a.e. $t$, $\nabla\cdot v(\cdot,t)=0$ and
$\phi(\cdot,t)$ is compactly supported, so

$$
\int v(x,t)\cdot\nabla\phi(x,t)\,dx=0.
$$

The extra term vanishes. If the local energy inequality is written with
endpoint time cutoffs, approximate those cutoffs from below by compactly
supported time cutoffs and pass to the endpoint by monotone convergence in the
nonnegative energy terms and dominated convergence in the finite flux terms.
Thus subtracting a time function changes neither suitability nor the pressure
gradient.

**Lemma PS4.2 -- Conditional local pressure control after normalization.**
Let $(U_n,\mathfrak p_n)$ be the current normalized or represented sequence and
let

$$
Q=B_R\times I
$$

be a compact normalized cylinder. Choose a larger cylinder

$$
Q'=B_{\Theta R}\times I',
\qquad
\Theta>1,
\qquad
Q\Subset Q'.
$$

Pressure control on $Q$ may be used only if one of the following inputs is
present.

1. Larger-ball pressure oscillation:

$$
\sup_n
\int_{I'}\int_{B_{\Theta R}}
|\mathfrak p_n-(\mathfrak p_n)_{B_{\Theta R}}(t)|^{3/2}\,dx\,dt<\infty.
$$

2. Whole-space pressure reconstruction, in a genuinely whole-space branch:

$$
\mathfrak p_n=\mathcal R_i\mathcal R_j(U_{n,i}U_{n,j})
$$

up to a function of time, together with the whole-space bound needed for
Calderon--Zygmund, for example
$\sup_n\|U_n\|_{L^3(\mathbb R^3\times I')}<\infty$.

3. Local pressure decomposition with harmonic remainder control:

$$
\mathfrak p_n=q_n+h_n,
\qquad
q_n=\mathcal R_i\mathcal R_j(\chi U_{n,i}U_{n,j}),
$$

with $h_n$ harmonic on a smaller ball, plus a stated bound on
$h_n-c_n(t)$ in $L^{3/2}(Q')$ or an equivalent larger-ball pressure
oscillation estimate.

If one of these inputs is verified, then with the local mean

$$
a_{n,R}(t)=(\mathfrak p_n)_{B_R}(t)
$$

one has

$$
\sup_n\|\mathfrak p_n-a_{n,R}(t)\|_{L^{3/2}(Q)}<\infty.
$$

**Proof.** In option 1, set

$$
q_n=\mathfrak p_n-(\mathfrak p_n)_{B_{\Theta R}}(t).
$$

Then

$$
\mathfrak p_n-(\mathfrak p_n)_{B_R}(t)=q_n-(q_n)_{B_R}(t).
$$

Lemma H0.2 gives, for a.e. $t$,

$$
\int_{B_R}|q_n-(q_n)_{B_R}(t)|^{3/2}\,dx
\le
C\int_{B_R}|q_n|^{3/2}\,dx
\le
C\int_{B_{\Theta R}}|\mathfrak p_n-(\mathfrak p_n)_{B_{\Theta R}}(t)|^{3/2}\,dx.
$$

Integrating over $I\subset I'$ proves the claim.

In option 2, Calderon--Zygmund gives

$$
\|\mathfrak p_n(\cdot,t)\|_{L^{3/2}(\mathbb R^3)}
\le
C\|U_n(\cdot,t)\|_{L^3(\mathbb R^3)}^2
$$

after the time function has been fixed. Integrating in time and applying
Lemma H0.2 on $B_R$ gives the local mean-subtracted bound. This route is
whole-space only and cannot be inferred from compact-cylinder velocity bounds.

In option 3, Calderon--Zygmund controls only the localized part $q_n$ from the
local $L^3$ velocity bound. The harmonic part is controlled only by the
separate hypothesis on $h_n-c_n(t)$, or by the equivalent larger-ball
oscillation bound. Jensen's inequality then replaces the auxiliary time
constant by the local mean $a_{n,R}(t)$. Without one of these pressure inputs,
this lemma supplies no pressure bound.

**Lemma PS4.3 -- Local modulation gauge under a nondegeneracy assumption.**
Let $V(\tau)$ be a local represented velocity close, in the topology used for
the modulation argument, to a finite-dimensional orbit parameterized by
$(\lambda,c)\in(0,\infty)\times\mathbb R^3$. Suppose the constraint map

$$
\mathcal F(\lambda,c;V)
$$

is explicitly defined on a Banach space $X$, the curve
$\tau\mapsto V(\tau)$ belongs to $AC(I;X)$, and

$$
\partial_{(\lambda,c)}\mathcal F(\lambda,c;V)
$$

is invertible at the reference point with a uniform inverse bound after
shrinking the neighborhood. Then, on a short time interval, there are
absolutely continuous parameters $(\lambda(\tau),c(\tau))$ such that

$$
\mathcal F(\lambda(\tau),c(\tau);V(\tau))=0.
$$

**Proof.** The verified nondegeneracy condition is

$$
\left\|
\left(\partial_{(\lambda,c)}
\mathcal F(\lambda,c;V)\right)^{-1}
\right\|_{\mathcal L(Y,\mathbb R^4)}
\le M_{\mathcal F}<\infty
$$

on the neighborhood in the Banach space $X$ where the modulation is used; here
$Y$ is the finite-dimensional target space of the constraints. The
finite-dimensional implicit-function theorem applied to this uniformly
invertible matrix produces neighborhoods and a $C^1$ map
$V\mapsto(\lambda(V),c(V))$ satisfying
$\mathcal F(\lambda(V),c(V);V)=0$. The stated hypothesis
$V\in AC(I;X)$ gives
$(\lambda(\tau),c(\tau))\in AC(I;\mathbb R^4)$ by composition with the $C^1$
map.
Differentiating the constraint gives

$$
\partial_{(\lambda,c)}\mathcal F
\binom{\lambda'(\tau)}{c'(\tau)}
=
-\partial_V\mathcal F[V'(\tau)],
$$

and the coefficient matrix is invertible by the displayed inverse-bound
condition. If the constraint map, the topology $X$, or the uniform inverse
bound is absent, this lemma is not invoked; the branch is routed as a
modulation defect or symmetry-degeneracy branch rather than silently choosing
a gauge.

**Lemma PS4.4 -- Repaired gauge data for the Type II branch.**
In a Type II single-core branch, admissible repaired gauge data are the
tuple

$$
\mathcal G_{\rm rep}
=
(\Phi_{\rm pre},\pi_{n,R},\lambda(\tau),c(\tau),\chi_R,\mathcal E_{\rm gauge}),
$$

where $\Phi_{\rm pre}$ is the accepted frame from `PS2`,

$$
\pi_{n,R}=\mathfrak p_n-a_{n,R}(t),
\qquad
a_{n,R}(t)=(\mathfrak p_n)_{B_R}(t),
$$

is the local mean-subtracted pressure from Lemma PS4.2 on the cylinder being
used, $(\lambda,c)$ are the modulation parameters from Lemma PS4.3,
$\chi_R$ is the localization cutoff, and
$\mathcal E_{\rm gauge}$ is the finite list of drift, cutoff, pressure, and
modulation source terms created by these choices. These data are admissible
only if each item of $\mathcal E_{\rm gauge}$ has a stated norm, cylinder, and
destination: controlled in `PS5`, treated as a defect in `PS30`, or deferred to
the Type II scale nodes.

**Proof.** Lemma PS2.2 supplies the frame and its Jacobian bounds, Lemma
PS4.1 shows that subtracting time functions from pressure preserves
suitability, Lemma PS4.2 supplies compactness-ready local pressure bounds
under its stated hypotheses, and Lemma PS4.3 supplies modulation parameters
under its nondegeneracy hypothesis. These statements do not eliminate the
terms produced by localization or time-dependent coordinates. Therefore the
gauge is repaired only after those terms are included explicitly in
$\mathcal E_{\rm gauge}$ with their estimates or destinations. If any term is
missing from these data, the gauge is incomplete and cannot be used in the
normalized equation at `PS5`.

### Specific Estimate

The decisive pressure estimate is not a velocity-only estimate. On each
compact cylinder $Q=B_R\times I$ it is the conditional conclusion

$$
\sup_n\|\mathfrak p_n-a_{n,R}(t)\|_{L^{3/2}(Q)}<\infty,
\qquad
a_{n,R}(t)=(\mathfrak p_n)_{B_R}(t),
$$

obtained only after one of the inputs in Lemma PS4.2 has been verified on a
larger cylinder $Q'$. A larger-ball pressure oscillation bound gives this by
Jensen directly; a whole-space reconstruction gives it by
Calderon--Zygmund on $\mathbb R^3$; and a local decomposition gives it only
after the harmonic remainder has its own oscillation bound.

### Practical Verification Steps

1. Choose a compact cylinder $Q$ for the next compactness step.
2. Choose the local pressure mean
   $a_{n,R}(t)=(\mathfrak p_n)_{B_R}(t)$ on that
   cylinder, or a larger-ball mean used to prove the local bound.
3. Verify that the local energy inequality is unchanged.
4. Verify one pressure-control input from Lemma PS4.2: larger-ball pressure
   oscillation, whole-space reconstruction, or local decomposition with
   harmonic remainder control.
5. Prove a uniform $L^{3/2}(Q)$ bound for
   $\pi_{n,R}=\mathfrak p_n-a_{n,R}(t)$.
6. In a Type II single-core branch, verify the modulation nondegeneracy
   condition, including the constraint topology and a uniform inverse bound,
   and solve the finite-dimensional constraints.
7. Build $\mathcal G_{\rm rep}$ and list every gauge-source term with its
   target estimate or later defect node.

## Estimate Step $B_{\mathrm{PS4}}$

The estimate step is Lemma PS4.1 plus Lemma PS4.2, and, in the modulated
branch, Lemmas PS4.3 and PS4.4.

## Failure Case

Failure name: pressure or modulation-gauge failure.

Analytic meaning: the compactness step cannot be run because the pressure is
not controlled in the mean-subtracted $L^{3/2}$ topology, or because the
finite-dimensional gauge parameters are not determined.

## Refinement Step

Allowed refinements:

1. shrink the compact cylinder;
2. enlarge the pressure-decomposition cylinder;
3. replace the pressure by another representative modulo a function of time;
4. pass to a subsequence on which the modulation nondegeneracy condition is
   stable;
5. if the nondegeneracy condition fails, identify a later symmetry or
   degenerate-direction branch.

Progress measure: cylinder shrinkage, pressure representative fixation, or
finite-dimensional gauge selection.

## Data Passed Forward

The next proof step is `PS5`. The data passed forward are

$$
\Gamma_{\mathrm{PS4}}
=
\Gamma_{\mathrm{PS3}}
\cup
\{\pi_{n,R}=\mathfrak p_n-a_{n,R}(t),\
  a_{n,R}(t)=(\mathfrak p_n)_{B_R}(t),\
  \sup_n\|\pi_{n,R}\|_{L^{3/2}(Q)}<\infty,
  \text{verified pressure-input route from Lemma PS4.2}\}
$$

and, in the modulated branch,

$$
\Gamma_{\mathrm{PS4}}
\ni
(\lambda(\tau),c(\tau),\mathcal G_{\rm rep}).
$$

---

# 11. `PS5` -- Renormalized Navier--Stokes Equation

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknowns are the normalized velocity and pressure. Depending on the branch,
these are $(u^\lambda,p^\lambda)$, $(V,\Pi)$ in Type I self-similar variables,
or $(V,P)$ in a time-dependent represented frame.

### Standing Assumptions

The incoming hypotheses state that the original pair solves NS
distributionally, the velocity is divergence-free, the pressure representative
is fixed, and the coordinate changes are locally absolutely continuous in time
and smooth affine maps in space. In a time-dependent frame, the active compact
time window has

$$
X,\Lambda\in W^{1,1}_{\rm loc},
\qquad
0<\Lambda_{\min}\le\Lambda(\tau)\le\Lambda_{\max}<\infty.
$$

Thus the drift coefficients produced by the frame are locally integrable in
time, not necessarily bounded.

### Objects Inspected

Inspect every term in the transformed equation:

$$
\partial_t u,\quad (u\cdot\nabla)u,\quad \nabla p,\quad \Delta u,\quad
\nabla\cdot u.
$$

For represented variables also inspect the drift terms produced by
$X'(\tau)$ and $\Lambda'(\tau)$.

### Dependencies Used

The equation comes from `H0`; the pressure representative comes from `PS4`;
the frame comes from `PS2`; the scale/rate choice comes from `PS3`.

### Local Obstruction Predicate

$P_{\mathrm{PS5}}$ holds if there is any untracked term in the transformed or
localized equation, if the transformed pressure cannot be identified as the
gradient term in the normalized system, or if localization creates a
divergence defect that has not been included in the equation.

### Local Lemmas to Prove

**Lemma PS5.1 -- Parabolic rescaling of NS.**
If $(u,p)$ solves NS distributionally, then
$(u^\lambda,p^\lambda)$ defined by

$$
u^\lambda(x,t)=\lambda u(x_*+\lambda x,T+\lambda^2t),
\qquad
p^\lambda(x,t)=\lambda^2p(x_*+\lambda x,T+\lambda^2t)
$$

solves NS distributionally on the rescaled domain.

**Proof.** With $X=x_*+\lambda x$ and $S=T+\lambda^2t$,

$$
\partial_tu^\lambda=\lambda^3(\partial_Su)(X,S),
\quad
(u^\lambda\cdot\nabla)u^\lambda
=\lambda^3((u\cdot\nabla)u)(X,S),
$$

$$
\nabla p^\lambda=\lambda^3(\nabla p)(X,S),
\qquad
\Delta u^\lambda=\lambda^3(\Delta u)(X,S).
$$

Every term is multiplied by the same factor $\lambda^3$. Also

$$
\nabla\cdot u^\lambda
=\lambda^2(\nabla\cdot u)(X,S)=0.
$$

For the distributional statement, pull compactly supported tests on the
rescaled cylinder back by $(x,t)\mapsto(X,S)$. The Jacobian and the derivatives
of the pulled-back tests contribute the same nonzero global factor to every
term in the weak formulation. Since the original weak formulation vanishes for
all pulled-back tests, the rescaled weak formulation vanishes as well.

**Lemma PS5.2 -- Centered Type I equation.**
If $(U,Q)$ is an ancient solution on
$\mathbb R^3\times(-\infty,0)$ and

$$
V(y,\tau)=\sqrt{-t}\,U(y\sqrt{-t},t),
\qquad
\Pi(y,\tau)=(-t)Q(y\sqrt{-t},t),
\qquad
\tau=-\log(-t),
$$

then $(V,\Pi)$ satisfies

$$
\partial_\tau V+(V\cdot\nabla)V+\nabla\Pi-\Delta V
+\frac12 V+\frac12 y\cdot\nabla V=0,
\qquad
\nabla\cdot V=0.
$$

**Proof.** Set $s=-t=e^{-\tau}$ and $x=s^{1/2}y$. Since
$U(x,t)=s^{-1/2}V(y,\tau)$ and $ds/dt=-1$,

$$
\partial_tU
=s^{-3/2}
\left(\partial_\tau V+\frac12V+\frac12y\cdot\nabla V\right),
$$

while

$$
(U\cdot\nabla_x)U=s^{-3/2}(V\cdot\nabla)V,
\quad
\Delta_xU=s^{-3/2}\Delta V,
\quad
\nabla_xQ=s^{-3/2}\nabla\Pi.
$$

Substitution into
$\partial_tU+(U\cdot\nabla)U+\nabla Q-\Delta U=0$ and multiplication by
$s^{3/2}$ gives the stated equation.

**Lemma PS5.3 -- Local represented-frame equation.**
Let

$$
Y=\frac{x-X(\tau)}{\Lambda(\tau)},\qquad
u(x,t)=\Lambda(\tau)^{-1}V(Y,\tau),\qquad
p(x,t)=\Lambda(\tau)^{-2}P(Y,\tau),
$$

with $dt/d\tau=\Lambda(\tau)^2$ and $\Lambda(\tau)>0$. Define

$$
a(\tau)=-\frac{\Lambda'(\tau)}{\Lambda(\tau)},
\qquad
b(\tau)=-\frac{X'(\tau)}{\Lambda(\tau)}.
$$

Assume on the compact time interval under consideration that

$$
X,\Lambda\in W^{1,1},
\qquad
0<\Lambda_{\min}\le\Lambda(\tau)\le\Lambda_{\max}<\infty.
$$

Then $a,b\in L^1_{\rm loc}$ and $(V,P)$ satisfies, in distributions,

$$
\partial_\tau V+(V\cdot\nabla)V+\nabla P-\Delta V
+a(\tau)(V+Y\cdot\nabla V)+b(\tau)\cdot\nabla V=0,
\qquad
\nabla\cdot V=0.
$$

Moreover,

$$
-\Delta P=\partial_i\partial_j(V_iV_j)
$$

in distributions, modulo the usual addition of a function of time to $P$.

All derivatives of $X$ and $\Lambda$ below are interpreted a.e. in $\tau$.
For merely absolutely continuous frames, the identity is obtained by testing
against compact smooth functions, applying the one-dimensional chain rule for
AC maps, and then using density from smooth frame approximations. The
positive lower bound for $\Lambda$ turns $X',\Lambda'\in L^1$ into
$a,b\in L^1$.

**Proof.** Since $d\tau/dt=\Lambda^{-2}$ and, at fixed $x$,

$$
\frac{dY}{d\tau}
=-\frac{X'(\tau)}{\Lambda(\tau)}
-\frac{\Lambda'(\tau)}{\Lambda(\tau)}Y,
$$

one has

$$
\partial_tu
=\Lambda^{-3}
\left[
\partial_\tau V
-\frac{\Lambda'}{\Lambda}(V+Y\cdot\nabla_YV)
-\frac{X'}{\Lambda}\cdot\nabla_YV
\right].
$$

The other terms satisfy

$$
(u\cdot\nabla_x)u=\Lambda^{-3}(V\cdot\nabla_Y)V,
\quad
\nabla_xp=\Lambda^{-3}\nabla_YP,
\quad
\Delta_xu=\Lambda^{-3}\Delta_YV.
$$

Multiplying the transformed NS equation by $\Lambda^3$ and inserting the
definitions of $a$ and $b$ gives the represented-frame equation.
The divergence condition transforms as
$\nabla_x\cdot u=\Lambda^{-2}\nabla_Y\cdot V=0$.

Taking divergence of the represented equation and using
$\nabla\cdot V=0$ gives

$$
\Delta P=-\partial_i\partial_j(V_iV_j),
$$

because
$\nabla\cdot(V+Y\cdot\nabla V)=0$ and
$\nabla\cdot(b\cdot\nabla V)=b\cdot\nabla(\nabla\cdot V)=0$ when $a,b$
depend only on time. In index notation,
$\nabla\cdot((V\cdot\nabla)V)=\partial_i\partial_j(V_iV_j)$ for
divergence-free $V$. This is equivalent to
$-\Delta P=\partial_i\partial_j(V_iV_j)$, with $P$ still determined only
modulo functions of $\tau$.

**Lemma PS5.4 -- Localized equation and explicit cutoff sources.**
Let $(V,P)$ satisfy the represented equation of Lemma PS5.3 on a cylinder, and
let $\chi\in C_c^\infty$ be a spatial cutoff. Set $W=\chi V$. Then

$$
\begin{aligned}
&\partial_\tau W+(V\cdot\nabla)W+\nabla(\chi P)-\Delta W
+a(\tau)(W+Y\cdot\nabla W)+b(\tau)\cdot\nabla W \\
&=
\mathcal E_\chi,
\end{aligned}
$$

where

$$
\mathcal E_\chi
=
(V\cdot\nabla\chi)V
+P\nabla\chi
-2(\nabla\chi\cdot\nabla)V
-V\Delta\chi
+a(\tau)(Y\cdot\nabla\chi)V
+(b(\tau)\cdot\nabla\chi)V.
$$

The localized divergence is

$$
\nabla\cdot W=V\cdot\nabla\chi.
$$

Consequently, a localized compactness argument must either work with compact
tests in the unlocalized equation, or else carry the divergence defect and
the full source vector $\mathcal E_\chi$ with norms sufficient for the target
topology.

**Proof.** Multiply the represented equation by $\chi$ and compare the result
with the equation obtained by applying the displayed operators to $W=\chi V$.
The product rules give

$$
(V\cdot\nabla)W
=
\chi(V\cdot\nabla)V+(V\cdot\nabla\chi)V,
$$

$$
\nabla(\chi P)=\chi\nabla P+P\nabla\chi,
$$

and

$$
-\Delta(\chi V)
=
-\chi\Delta V-2(\nabla\chi\cdot\nabla)V-V\Delta\chi.
$$

The drift terms satisfy

$$
a(Y\cdot\nabla W)
=
a\chi\,Y\cdot\nabla V+a(Y\cdot\nabla\chi)V,
\qquad
b\cdot\nabla W
=
\chi b\cdot\nabla V+(b\cdot\nabla\chi)V.
$$

Collecting the non-$\chi$ terms gives $\mathcal E_\chi$. Finally
$\nabla\cdot(\chi V)=V\cdot\nabla\chi$ because $\nabla\cdot V=0$.

**Lemma PS5.5 -- Activity-compatible cutoff.**
Suppose the active compact packet from `PS3` is

$$
Q_{\mathrm{act}}=B_1\times(-1,-\sigma),
\qquad
\int_{Q_{\mathrm{act}}}|V_n|^3\,dY\,d\tau\ge\eta_1.
$$

If the variable passed to compactness is

$$
W_n=\chi V_n,
$$

then either $\chi\equiv1$ on $B_1$ or the branch must reprove a positive
lower bound for $W_n$ on the compactness cylinder. In the first case,

$$
\int_{Q_{\mathrm{act}}}|W_n|^3\,dY\,d\tau
=
\int_{Q_{\mathrm{act}}}|V_n|^3\,dY\,d\tau
\ge\eta_1.
$$

**Proof.** If $\chi\equiv1$ on $B_1$, then $W_n=V_n$ on
$Q_{\mathrm{act}}$, so the equality and the lower bound are immediate. If
$\chi$ is not identically one on the active spatial region, multiplication by
$\chi$ can remove part or all of the recorded $L^3$ activity. Therefore no
lower bound for $W_n$ follows from the lower bound for $V_n$ without an
additional packet-retention proof.

**Lemma PS5.6 -- Cutoff-source bounds in the compactness topology.**
Let $\chi\in C_c^\infty(B_{2R})$ with all derivatives fixed, and suppose on
the time interval $I$ that

$$
V\in L^\infty_tL^2_x(B_{2R})
\cap L^2_tH^1_x(B_{2R})
\cap L^3(B_{2R}\times I),
\qquad
P\in L^{3/2}(B_{2R}\times I),
$$

with $a,b\in L^1(I)$. Then the source vector
$\mathcal E_\chi$ from Lemma PS5.4 is bounded in
$L^1_tH^{-m}_x(I\times B_{2R})$ for any $m$ with
$H^m_0(B_{2R})\hookrightarrow W^{1,\infty}(B_{2R})$. More precisely, for
every test field $\varphi\in C_c^\infty(B_{2R})$,

$$
\begin{aligned}
|\langle\mathcal E_\chi,\varphi\rangle|
\le C_{\chi,R}\Big[
&\|V\|_{L^3(B_{2R})}^2\|\varphi\|_{L^3(B_{2R})}
+\|P\|_{L^{3/2}(B_{2R})}\|\varphi\|_{L^3(B_{2R})}\\
&+\|\nabla V\|_{L^2(B_{2R})}\|\varphi\|_{L^2(B_{2R})}
+(1+|a|+|b|)\|V\|_{L^2(B_{2R})}
\|\varphi\|_{L^2(B_{2R})}
\Big].
\end{aligned}
$$

Consequently, localized cutoff terms are compactness-admissible only after
this bound, or a stronger branch-specific bound, has been proved. If such a
bound is absent, the cutoff cannot be treated as harmless; the branch has an
equation defect rather than a closed localized equation.

**Proof.** Each term in $\mathcal E_\chi$ is supported where $\chi$ and its
derivatives are fixed smooth functions. The quadratic term satisfies

$$
\left|\int (V\cdot\nabla\chi)V\cdot\varphi\right|
\le C_\chi\|V\|_{L^3}^2\|\varphi\|_{L^3}.
$$

The pressure cutoff term is bounded by Holder's inequality as

$$
\left|\int P\nabla\chi\cdot\varphi\right|
\le C_\chi\|P\|_{L^{3/2}}\|\varphi\|_{L^3}.
$$

The commutator terms involving $\nabla V$ and $\Delta\chi$ obey

$$
\left|\int 2(\nabla\chi\cdot\nabla)V\cdot\varphi\right|
\le C_\chi\|\nabla V\|_{L^2}\|\varphi\|_{L^2},
\qquad
\left|\int V\Delta\chi\cdot\varphi\right|
\le C_\chi\|V\|_{L^2}\|\varphi\|_{L^2}.
$$

On the compact support of $\nabla\chi$, $|Y|\le C_R$, so the modulation-source
terms satisfy

$$
\left|\int a(Y\cdot\nabla\chi)V\cdot\varphi\right|
+\left|\int (b\cdot\nabla\chi)V\cdot\varphi\right|
\le C_{\chi,R}(|a|+|b|)\|V\|_{L^2}\|\varphi\|_{L^2}.
$$

Combining these estimates gives the displayed bound. The assumed energy and
pressure bounds, together with $a,b\in L^1(I)$ and
$V\in L^\infty_I L^2(B_{2R})$, make the right-hand side integrable in time.
The Sobolev embedding of $H^m_0$ into $W^{1,\infty}$, and hence into
$L^2\cap L^3$, identifies the source as a bounded element of
$L^1_tH^{-m}_x$.

**Lemma PS5.7 -- Time-derivative bound supplied by the transformed
equation.**
Let $Q=B_R\times I$ be a compact normalized cylinder. Assume

$$
V\in L^\infty_I L^2(B_{2R})
\cap L^2_IH^1(B_{2R})
\cap L^3(B_{2R}\times I),
\qquad
P\in L^{3/2}(B_{2R}\times I),
\qquad
a,b\in L^1(I).
$$

If no cutoff is used, then the represented equation of Lemma PS5.3 gives

$$
\partial_\tau V
\in
L^1_IH^{-m}(B_R)
$$

for every integer $m\ge3$, hence for every $m$ with
$H^m_0(B_R)\hookrightarrow W^{1,\infty}(B_R)$. If $W=\chi V$ is used instead,
then

$$
\partial_\tau W
\in
L^1_IH^{-m}(B_{2R}),
$$

provided the localized equation of Lemma PS5.4, the activity-compatible
cutoff condition of Lemma PS5.5 when activity is needed, and the cutoff-source
bound of Lemma PS5.6 are part of the branch data.

**Proof.** Test the unlocalized represented equation against
$\varphi\in C_c^\infty(B_R)$. The nonlinear term is bounded by

$$
\left|\int (V\cdot\nabla V)\cdot\varphi\right|
=
\left|\int (V\otimes V):\nabla\varphi\right|
\le
\|V\|_{L^3(B_R)}^2\|\nabla\varphi\|_{L^3(B_R)}.
$$

The pressure term is bounded by

$$
\left|\int P\,\nabla\cdot\varphi\right|
\le
\|P\|_{L^{3/2}(B_R)}\|\nabla\varphi\|_{L^3(B_R)}.
$$

The viscous term satisfies

$$
\left|\int \nabla V:\nabla\varphi\right|
\le
\|\nabla V\|_{L^2(B_R)}\|\nabla\varphi\|_{L^2(B_R)}.
$$

For the drift terms, integrate by parts in space. Since $a$ and $b$ depend
only on time,

$$
\langle Y\cdot\nabla V,\varphi\rangle
=
-\int_{B_R}V\cdot(3\varphi+Y\cdot\nabla\varphi),
$$

and

$$
\langle b\cdot\nabla V,\varphi\rangle
=
-\int_{B_R}V\cdot(b\cdot\nabla\varphi).
$$

Therefore

$$
\left|
\left\langle a(V+Y\cdot\nabla V),\varphi\right\rangle
\right|
\le
C_R|a|\|V\|_{L^2(B_R)}
\|\varphi\|_{W^{1,\infty}(B_R)}
$$

and

$$
\left|
\left\langle b\cdot\nabla V,\varphi\right\rangle
\right|
\le
C_R|b|\|V\|_{L^2(B_R)}
\|\varphi\|_{W^{1,\infty}(B_R)}.
$$

Thus the drift terms require only $a,b\in L^1(I)$ together with
$V\in L^\infty_I L^2(B_R)$. The assumed spacetime bounds make all right-hand
sides integrable in time, after Holder's inequality in time for the $L^3$ and
$L^{3/2}$ terms. The embedding $H^m_0\hookrightarrow W^{1,\infty}$ for
$m\ge3$ converts the test bounds into an $H^{-m}$ bound.

For $W=\chi V$, use the localized equation from Lemma PS5.4. The convection,
pressure, diffusion, and drift terms are bounded exactly as above on
$B_{2R}$, with $V$ appearing in the transport coefficient. Lemma PS5.6
supplies the remaining $\mathcal E_\chi$ term, and Lemma PS5.4 keeps the
divergence defect $V\cdot\nabla\chi$ explicit. Thus the localized time
derivative is also bounded in $L^1_IH^{-m}(B_{2R})$. The localized statement
is a compactness-ready controlled-source equation; it is not an assertion
that $W$ is divergence-free or solves unforced Navier--Stokes.

### Specific Estimate

The decisive identity is that every NS term transforms with the same
parabolic factor, while the time-dependent affine frame produces exactly the
two drift terms

$$
a(\tau)(V+Y\cdot\nabla V),
\qquad
b(\tau)\cdot\nabla V.
$$

There are no missing error terms in the unlocalized affine equation. If a
cutoff is introduced, the exact source vector is $\mathcal E_\chi$ from Lemma
PS5.4, the exact divergence defect is $V\cdot\nabla\chi$, and the source must
satisfy the compactness-topology bound of Lemma PS5.6. If compactness is
applied to $W_n=\chi V_n$ and activity is needed later, Lemma PS5.5 requires
$\chi\equiv1$ on the active packet or a new lower-bound proof for $W_n$. The
proof also needs the time-derivative estimate of Lemma PS5.7 before `PS6`
uses Aubin--Lions
compactness.

### Practical Verification Steps

1. Write the coordinate transformation explicitly.
2. Differentiate $u$ and $p$ term by term.
3. Check that the nonlinear, pressure, and viscous terms have the same
   scaling factor.
4. Verify the divergence constraint.
5. Identify all drift/modulation terms.
6. If a cutoff is used, compute $\mathcal E_\chi$ and the divergence defect,
   choose it to preserve the active packet or reprove activity for the
   localized variable, then prove the source bound needed by the compactness
   topology.
7. Prove the negative-Sobolev time-derivative bound for the exact variable
   passed to compactness.
8. Verify the pressure Poisson relation in the normalized variables.

## Estimate Step $B_{\mathrm{PS5}}$

The estimate step is the explicit chain-rule derivation in Lemmas PS5.1--PS5.3,
the localized product-rule computation in Lemma PS5.4, the activity-compatible
cutoff check in Lemma PS5.5, the cutoff-source bound in Lemma PS5.6, and the
compactness-topology time-derivative bound in Lemma PS5.7.

## Failure Case

Failure name: transformed-equation defect.

Analytic meaning: the hypotheses lack a closed local PDE for the normalized
variables, so compactness and passage to a limit are not admissible.

## Refinement Step

Allowed refinements:

1. restrict to a smaller time interval where the frame is absolutely
   continuous;
2. redefine $a$ and $b$ using normalized derivatives
   $-\Lambda'/\Lambda$ and $-X'/\Lambda$;
3. include any genuinely present forcing, cutoff, commutator, or gauge term
   explicitly in the equation;
4. return to `PS4` if the pressure representative is the source of the defect.

Progress measure: all missing transformed terms are added explicitly, or the
coordinate window is strictly shrunk.

## Data Passed Forward

The next proof step is `PS6`. The data passed forward are

$$
\Gamma_{\mathrm{PS5}}
=
\Gamma_{\mathrm{PS4}}
\cup
\{\text{closed normalized NS equation},\ \nabla\cdot V=0,\quad
a,b\in L^1_{\rm loc},\quad
-\Delta P=\partial_i\partial_j(V_iV_j)\}.
$$

If localization is used, the data also include

$$
\mathcal E_\chi,\qquad
\nabla\cdot(\chi V)=V\cdot\nabla\chi,
$$

the compactness-topology bounds for $\mathcal E_\chi$, and the statement that
the cutoff is activity-compatible on $Q_{\mathrm{act}}$ or that a replacement
activity lower bound has been proved for $\chi V$. Finally, the branch records
the time-derivative estimate for the actual variable passed to `PS6`:

$$
\partial_\tau V\in L^1_IH^{-m}(B_R)
\quad\text{or}\quad
\partial_\tau(\chi V)\in L^1_IH^{-m}(B_{2R}).
$$

---

# 12. `PS6` -- Compactness and Limiting Profile

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

Work on compact normalized cylinders in the variables produced by `PS5`. In
the unlocalized branch the sequence is $(V_n,P_n)$ and solves the closed
normalized Navier--Stokes equation proved in `PS5`. In the localized branch
the compactness variable is

$$
W_n=\chi V_n,
$$

and the equation carries the explicit source $\mathcal E_{\chi,n}$ and
divergence defect

$$
g_n=\nabla\cdot W_n=V_n\cdot\nabla\chi .
$$

The proof does not erase these defects. It either produces an ordinary
limiting profile for the unlocalized or cutoff-inactive case, or a controlled
localized limit together with the limiting source and divergence defects.

### Standing Assumptions

Lower concentration from `PS1`--`PS5` is not a compactness hypothesis. On each
compact cylinder

$$
Q=B_R\times I\Subset\Omega_{\rm norm},
$$

choose a slightly larger cylinder

$$
Q^+=B_{2R}\times I^+,
\qquad
Q\Subset Q^+.
$$

The branch may run `PS6` on $Q$ only after the following upper-bound package
has been proved:

$$
\sup_n\left(
\|V_n\|_{L^\infty(I^+;L^2(B_{2R}))}
+\|\nabla V_n\|_{L^2(Q^+)}
+\|P_n-(P_n)_{B_{2R}}(t)\|_{L^{3/2}(Q^+)}
\right)<\infty
$$

and a time-derivative bound

$$
\partial_\tau V_n
\quad\hbox{bounded in}\quad
L^1_IH^{-m}(B_R)
$$

for $m$ large enough that $H^m_0(B_R)\hookrightarrow W^{1,\infty}(B_R)$.
If the represented equation contains drift coefficients $a_n(\tau)$ and
$b_n(\tau)$, then either $a_n,b_n$ are fixed in $n$, or

$$
a_n\to a,\qquad b_n\to b
\quad\hbox{strongly in }L^2_{\rm loc}
$$

on the compact time window being used. If only boundedness of $a_n,b_n$ is
known, `PS6` passes a coefficient-defect equation rather than a closed
normalized equation.

For localized variables, compactness of $W_n=\chi V_n$ is not enough by
itself. The branch must have the above compactness package for $V_n$ on a
neighborhood of $\operatorname{supp}\chi$, so that the source terms involving
$V_n$, $\nabla V_n$, and $P_n$ can be passed to the limit. The localized source
sequence must also satisfy either

$$
\mathcal E_{\chi,n}
\quad\hbox{bounded in}\quad L^q(I;H^{-m}(B_{2R})),
\qquad q>1,
$$

or a stated uniform-integrability hypothesis in $L^1(I;H^{-m}(B_{2R}))$. With
only an $L^1$ bound, the source is passed only distributionally, possibly with
a measure-valued defect. The divergence defects satisfy

$$
g_n=V_n\cdot\nabla\chi
\quad\hbox{bounded in}\quad L^2(I\times B_{2R}).
$$

### Objects Inspected

The proof inspects the local energy bounds, pressure oscillation bounds,
negative-Sobolev time derivative, normalized equation, and, when localization
is active, the source $\mathcal E_{\chi,n}$ and divergence defect $g_n$.

### Dependencies Used

The compactness upper-bound package must be present in the branch record before
this node runs. The entry and pressure-gauge routes come from `D_E` and `PS4`,
the normalized equation and derivative bound come from `PS5`, and the compact
activity lower bound used later comes from `PS8`. No endpoint theorem is used
in this node.

### Local Obstruction Predicate

$P_{\mathrm{PS6}}$ holds when the compactness topology needed for passing to a
limit is missing: an energy bound, pressure bound, time-derivative bound,
source bound, or divergence-defect estimate has not been established on the
compact cylinder being used.

### Local Lemmas to Prove

**Lemma PS6.1 -- Compactness from the NS3D bounds.**
Under the standing assumptions, after passing to a subsequence,

$$
V_n\to V
\quad\hbox{strongly in}\quad
L^2(I;L^2(B_R))
\quad\hbox{and}\quad
L^3(I\times B_R),
$$

while

$$
\nabla V_n\rightharpoonup\nabla V\quad\hbox{in }L^2(I\times B_R),
\qquad
P_n^R:=P_n-(P_n)_{B_R}(t)\rightharpoonup P^R
\quad\hbox{in }L^{3/2}(I\times B_R).
$$

For nested balls $B_R\subset B_S$, the limiting pressure representatives are
compatible:

$$
P^R=P^S-(P^S)_{B_R}(t).
$$

Thus the pressure passed forward is a compatible local
$L^{3/2}$ pressure class modulo functions of time, not a single global
pressure unless a global gauge has separately been fixed. In the localized
branch, the same strong convergence holds for $W_n=\chi V_n$ with
$W=\chi V$, but only because $V_n\to V$ strongly near
$\operatorname{supp}\chi$.

**Proof.** The bounds place $V_n$ in
$L^\infty_I L^2(B_{2R})\cap L^2_IH^1(B_{2R})$ and place
$\partial_\tau V_n$ in $L^1_IH^{-m}(B_R)$. The compact embedding
$H^1(B_R)\Subset L^2(B_R)$ and the continuous embedding
$L^2(B_R)\hookrightarrow H^{-m}(B_R)$ allow Aubin--Lions--Simon to give
strong compactness in $L^2_I L^2(B_R)$. The energy bounds also give the
standard three-dimensional parabolic interpolation bound

$$
V_n\quad\hbox{bounded in}\quad L^{10/3}(I\times B_R).
$$

After passing to the $L^2$-strong subsequence, the differences $V_n-V$ are
uniformly bounded in $L^{10/3}(I\times B_R)$. Interpolating between $L^2$ and
$L^{10/3}$ gives

$$
\|V_n-V\|_{L^3(I\times B_R)}
\le
C
\|V_n-V\|_{L^2(I\times B_R)}^{1/6}
\|V_n-V\|_{L^{10/3}(I\times B_R)}^{5/6}
\to0.
$$

Weak compactness in $L^2$ gives the gradient limit. Reflexivity of
$L^{3/2}$ gives, for each ball $B_R$, a subsequential weak limit
$P_n^R\rightharpoonup P^R$ after subtracting the spatial mean. If
$B_R\subset B_S$, then for every $n$,

$$
P_n^R=P_n^S-(P_n^S)_{B_R}(t).
$$

The averaging map is continuous from $L^{3/2}(B_S)$ to functions of time on
the compact interval, so passing to the weak limit gives
$P^R=P^S-(P^S)_{B_R}(t)$. A diagonal extraction over a countable ball
exhaustion produces compatible pressure representatives. Finally,
$W_n=\chi V_n\to\chi V$ strongly in $L^3$ on the localized cylinder because
$V_n\to V$ strongly on a neighborhood of $\operatorname{supp}\chi$.

**Lemma PS6.2 -- Passage to the unlocalized limiting equation.**
If the unlocalized equation from `PS5` is the active equation, then the limit
$(V,P)$, where $P$ denotes the compatible local pressure class from
Lemma PS6.1, satisfies in distributions on $B_R\times I$,

$$
\partial_\tau V+(V\cdot\nabla)V+\nabla P-\Delta V
+a(V+Y\cdot\nabla V)+b\cdot\nabla V=0,
\qquad
\nabla\cdot V=0,
$$

with the pressure representative fixed modulo functions of time.

**Proof.** Test the equation for $(V_n,P_n)$ against
$\varphi\in C_c^\infty(B_R\times I)$. The linear terms pass by weak
convergence. The nonlinear term passes because
$V_n\to V$ strongly in $L^3_{\rm loc}$, hence
$V_n\otimes V_n\to V\otimes V$ strongly in $L^{3/2}_{\rm loc}$. The pressure
term passes by weak convergence of the compatible mean-subtracted pressure
representative on the ball containing $\operatorname{supp}\varphi$.

For the drift terms, if $a_n,b_n$ are independent of $n$, weak convergence of
$V_n$ in the local energy spaces is enough. If
$a_n\to a$ and $b_n\to b$ strongly in $L^2(I)$, then

$$
a_nV_n\to aV,\qquad b_nV_n\to bV
$$

in distributions because $V_n$ is bounded in $L^\infty_I L^2(B_R)$ and
converges strongly in $L^2(I\times B_R)$. For the derivative drift terms,
use the distributional identities from `PS5`,

$$
\langle Y\cdot\nabla V_n,\varphi\rangle
=-\int V_n\cdot(3\varphi+Y\cdot\nabla\varphi),
\qquad
\langle b_n\cdot\nabla V_n,\varphi\rangle
=-\int V_n\cdot(b_n\cdot\nabla\varphi),
$$

and the same coefficient convergence. The divergence constraint passes by
distributional convergence.

If no coefficient convergence is available, the same limiting process produces
only a defect-drift equation: the weak limits of
$a_n(V_n+Y\cdot\nabla V_n)$ and $b_n\cdot\nabla V_n$ are recorded as drift
defects. In that case this lemma does not output a closed normalized equation.

**Lemma PS6.3 -- Localized limits retain their source and divergence defects.**
If the compactness variable is $W_n=\chi V_n$, then after passing to a
subsequence there are a distributional source $\mathcal E_\chi$ and a
divergence defect $g$ such that

$$
g_n\rightharpoonup g
\quad\hbox{in }L^2(I\times B_{2R}),
$$

and the limit satisfies

$$
\nabla\cdot W=g.
$$

Moreover $W$ solves the localized controlled-source equation obtained from
Lemma PS5.4 with $\mathcal E_\chi$ and $g$ retained. It is an ordinary
divergence-free Navier--Stokes profile only on subregions where $\chi=1$ or
after a later node proves that the localized defects vanish.

If the source sequence is bounded in $L^q(I;H^{-m}(B_{2R}))$ for some $q>1$,
or is uniformly integrable in $L^1(I;H^{-m}(B_{2R}))$, then
$\mathcal E_{\chi,n}$ has a subsequential weak limit in the corresponding
function class. With only an $L^1$ bound, `PS6` records only distributional
convergence after subsequence extraction, possibly with a measure-valued
source defect.

**Proof.** The bound on $g_n=V_n\cdot\nabla\chi$ gives weak compactness in
$L^2_{\rm loc}$. Since $W_n\to W$ distributionally and
$\nabla\cdot W_n=g_n$, passing to the limit gives $\nabla\cdot W=g$.

For the source, weak compactness in $L^1$ is not automatic. If the branch has
an $L^q$ bound with $q>1$, reflexivity gives a weakly convergent subsequence.
If the branch has uniform integrability in $L^1$, Dunford--Pettis gives weak
compactness in $L^1$. Without either input, the localized source is passed
only against smooth compactly supported tests and any non-compact part is
recorded as a source defect. Testing the localized equation and using Lemma
PS6.1 for the velocity terms gives the controlled-source limiting equation.
No term supported on the cutoff annulus disappears in this passage; each is
represented in $\mathcal E_\chi$, $g$, or an explicitly named source defect.

### Specific Estimate

The decisive compactness estimate is

$$
V_n\to V\quad\hbox{or}\quad W_n\to W
\qquad\hbox{strongly in }L^3_{\rm loc},
$$

with weak pressure convergence in $L^{3/2}_{\rm loc}$. This is the topology
used by `PS7` to pass the local energy inequality and by `PS8` to pass the
local activity lower bound.

If drift coefficients are present, the decisive closure condition is

$$
a_n\to a,\qquad b_n\to b
\quad\hbox{strongly on the compact time window},
$$

or else the output is a coefficient-defect equation. If localized sources are
present, an $L^q$ bound with $q>1$ or uniform integrability is required before
the source is passed as an $L^1H^{-m}$ object.

### Practical Verification Steps

1. Fix a compact cylinder $B_R\times I$ strictly inside the normalized window.
2. Verify the uniform energy, pressure, and time-derivative bounds on a larger
   cylinder.
3. Apply Aubin--Lions--Simon to obtain strong local velocity convergence.
4. Upgrade strong $L^2$ convergence to strong $L^3$ convergence using the
   uniform $L^{10/3}$ bound.
5. Extract compatible weak pressure representatives after subtracting local
   spatial means.
6. Verify convergence of drift coefficients, or record a coefficient-defect
   equation.
7. In localized variables, prove compactness of $V_n$ near
   $\operatorname{supp}\chi$ and pass $\mathcal E_{\chi,n}$ only in the
   function class justified by the source compactness input.

## Estimate Step $B_{\mathrm{PS6}}$

The estimate step is Lemmas PS6.1--PS6.3: compactness from energy, pressure,
and time-derivative bounds, followed by distributional passage to the
unlocalized or localized limiting equation.

## Failure Case

Failure name: limiting-profile compactness failure.

Analytic meaning: the branch has normalized equations, but the proof has not
established enough compactness to extract a limiting NS3D profile or
controlled localized limit.

## Refinement Step

Allowed refinements:

1. shrink to a compact subcylinder;
2. subtract spatial pressure means on the selected balls;
3. keep cutoff sources and divergence defects instead of discarding them;
4. return to `PS5` for the missing time-derivative or source estimate;
5. add coefficient convergence for $a_n,b_n$, or route the coefficient defect.

Progress measure: one missing compactness input is supplied on a fixed compact
cylinder, or the branch is routed as a compactness defect.

## Data Passed Forward

The next proof step is `PS7`. The data passed forward are

$$
\Gamma_{\mathrm{PS6}}
=
\Gamma_{\mathrm{PS5}}
\cup
\{\text{limit }V\text{ or }W=\chi V,\
V_n\to V\text{ near }\operatorname{supp}\chi\text{ in }L^3_{\rm loc},\
P^R\text{ compatible local pressure class modulo time functions},\
\text{closed equation or coefficient-defect equation},\
\mathcal E_\chi,\ g\text{ and any source defect retained if localized}\}.
$$

---

# 13. `PS7` -- Suitability and Admissibility of the Limit

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

For the Type I branch, work with the ancient limit $(U,Q)$ on
$\mathbb R^3\times(-\infty,0)$. For the represented Type II branch, work on a
compact renormalized cylinder $Q_R^J$ with variables $(V,P)$ and modulation
coefficients $a,b$.

### Standing Assumptions

The incoming hypotheses state that the approximating sequence is suitable, the pressure is normalized as
in `PS4`, and the convergence topologies from `PS6` hold on every compact
subcylinder under consideration.

### Objects Inspected

Inspect the local energy inequality term by term:

$$
\int |v|^2\phi,\qquad
\iint |\nabla v|^2\phi,\qquad
\iint |v|^2(\partial_t\phi+\Delta\phi),\qquad
\iint (|v|^2+2\pi)v\cdot\nabla\phi.
$$

In represented variables also inspect the additional drift contributions
coming from $a(V+Y\cdot\nabla V)$ and $b\cdot\nabla V$.

### Dependencies Used

The limiting profile comes from `PS6`; the pressure normalization comes from
`PS4`; the normalized equation comes from `PS5`; the local energy inequality
for approximating solutions comes from `D_E`.

### Local Obstruction Predicate

$P_{\mathrm{PS7}}$ holds if any term in the local energy inequality cannot be
passed to the limit, if the represented frame has not transported the physical
suitability inequality, or if cutoff-localized variables are being used as an
ordinary suitable solution on a test region where cutoff source or divergence
defects are present.

### Local Lemmas to Prove

**Lemma PS7.1 -- Stability of suitability under local compactness.**
Let $(v_n,\pi_n)$ be suitable on a cylinder $Q$, with

$$
v_n\to v\quad\text{strongly in }L^3_{\rm loc}(Q),
\qquad
\pi_n-(\pi_n)_{B_R}(t)\rightharpoonup \pi^R
\quad\text{weakly in }L^{3/2}_{\rm loc}(Q),
$$

and

$$
\nabla v_n\rightharpoonup \nabla v
\quad\text{weakly in }L^2_{\rm loc}(Q).
$$

Assume the local pressure representatives $\pi^R$ are compatible modulo
functions of time, and write $\pi$ for that pressure class. If $(v,\pi)$
solves NS distributionally, then $(v,\pi)$ is suitable on $Q$.

**Proof.** Write the local energy inequality for $v_n,\pi_n$ in
distributional form using a nonnegative
$\phi\in C_c^\infty(Q)$. The terms containing $|v_n|^2$ converge strongly in
$L^1_{\rm loc}$ by strong $L^2_{\rm loc}$ convergence. The cubic flux
$|v_n|^2v_n$ converges strongly in $L^1_{\rm loc}$ by strong
$L^3_{\rm loc}$ convergence. Before passing the pressure flux to the limit,
subtract any time-dependent pressure representative on the ball containing
$\operatorname{supp}\phi$. This changes the flux by

$$
\int a_n(t)v_n\cdot\nabla\phi\,dx\,dt=0
$$

because $\nabla\cdot v_n=0$ and $\phi$ is compactly supported. Thus only the
mean-subtracted pressure representatives appear; after this subtraction,
rename them $\pi_n$ on the ball containing $\operatorname{supp}\phi$. The
pressure flux converges because $\pi_n\rightharpoonup\pi$ in $L^{3/2}$ and
$v_n\cdot\nabla\phi\to v\cdot\nabla\phi$ strongly in $L^3$; explicitly,

$$
\int(\pi_n-\pi)v\cdot\nabla\phi\to0,
$$

and

$$
\left|\int(\pi_n-\pi)(v_n-v)\cdot\nabla\phi\right|
\le
C\|\pi_n-\pi\|_{L^{3/2}(\operatorname{supp}\phi)}
\|v_n-v\|_{L^3(\operatorname{supp}\phi)}\to0,
$$

while

$$
\int\pi(v_n-v)\cdot\nabla\phi\to0.
$$

The first limit uses weak convergence against the fixed $L^3$ test factor
$v\cdot\nabla\phi$; the next two limits use Holder's inequality, the
boundedness of $\pi_n$ in $L^{3/2}$, and strong $L^3$ convergence of $v_n$.
Finally,

$$
\iint |\nabla v|^2\phi
\le
\liminf_{n\to\infty}\iint |\nabla v_n|^2\phi
$$

by weak lower semicontinuity, since $\phi\ge0$. Passing to the limit gives the
distributional local energy inequality for $(v,\pi)$. Time cutoff
approximations recover the endpoint form for a.e. times.

**Lemma PS7.2 -- Transport of suitability in represented variables.**
Let $(u,p)$ be suitable on a physical cylinder and let

$$
Y=\frac{x-X(\tau)}{\Lambda(\tau)},\qquad
u(x,t)=\Lambda(\tau)^{-1}V(Y,\tau),\qquad
p(x,t)=\Lambda(\tau)^{-2}P(Y,\tau),
$$

with $dt/d\tau=\Lambda(\tau)^2$, $\Lambda>0$, and
$X,\Lambda\in W^{1,1}_{\rm loc}$. Then $(V,P)$ satisfies the represented local
energy inequality on compact renormalized windows. The additional terms are
exactly those generated by

$$
a(\tau)=-\frac{\Lambda'(\tau)}{\Lambda(\tau)},
\qquad
b(\tau)=-\frac{X'(\tau)}{\Lambda(\tau)}.
$$

**Proof.** Pull a nonnegative test function $\phi(Y,\tau)$ back to the
physical cylinder by the affine map. Applying the physical local energy
inequality and changing variables gives the usual local energy terms for
$V,P$, plus the drift terms obtained from
$\partial_t\phi((x-X)/\Lambda,\tau(t))$. The identities are the same chain
rule computations used in `PS5`: the velocity, pressure, and dissipation terms
all carry the common parabolic factor, while differentiating the moving frame
produces only the $a(V+Y\cdot\nabla V)$ and $b\cdot\nabla V$ contributions.
There are no boundary terms because the pulled-back test is compactly
supported inside the physical cylinder. Since $a,b$ depend only on time and
are locally integrable on the compact window, the drift integrals are finite
under the local energy and pressure bounds.

Equivalently, for every nonnegative compactly supported test function $\phi$
and a.e. $\tau_1<\tau_2$ inside the represented window, the represented local
energy inequality is

$$
\begin{aligned}
&\int \frac{|V|^2}{2}\phi(\tau_2)
+\int_{\tau_1}^{\tau_2}\int |\nabla V|^2\phi \\
&\le
\int \frac{|V|^2}{2}\phi(\tau_1)
+\int_{\tau_1}^{\tau_2}\int
\frac{|V|^2}{2}(\partial_\tau\phi+\Delta\phi) \\
&\quad
+\int_{\tau_1}^{\tau_2}\int
\left(\frac{|V|^2}{2}+P\right)V\cdot\nabla\phi \\
&\quad
+\int_{\tau_1}^{\tau_2}\int
\frac{a}{2}|V|^2\phi
+\int_{\tau_1}^{\tau_2}\int
\frac{|V|^2}{2}(aY+b)\cdot\nabla\phi .
\end{aligned}
$$

The last line is the contribution of the drift
$aV+(aY+b)\cdot\nabla V$. It follows from

$$
\int \left(aV+(aY+b)\cdot\nabla V\right)\cdot V\phi
=
-\int \frac{a}{2}|V|^2\phi
-\int \frac{|V|^2}{2}(aY+b)\cdot\nabla\phi,
$$

using $\nabla\cdot(aY+b)=3a$. Moving this drift contribution to the right side
gives the displayed signs. This is represented drift-suitability, not ordinary
Navier--Stokes suitability unless $a=b=0$ on the window.

**Lemma PS7.3 -- Localized limits are admissible only with their defects.**
Let $W=\chi V$ be a cutoff-localized limit produced by `PS6`, with limiting
source $\mathcal E_\chi$ and divergence defect

$$
\nabla\cdot W=g=V\cdot\nabla\chi
$$

in the convergence class proved there. Then:

1. on every compact cylinder whose spatial support lies in $\{\chi=1\}$, the
   localized limit agrees with the unlocalized represented profile, so the
   represented drift-suitability statement from Lemma PS7.2 applies;
2. on a test cylinder meeting the cutoff annulus
   $\operatorname{supp}\nabla\chi$, the pair $(W,\chi P)$ is not an ordinary
   suitable Navier--Stokes profile unless the source and divergence defect
   vanish on that cylinder;
3. if the cutoff annulus is used later, the data passed to later steps retain
   $\mathcal E_\chi$, $g$, and their convergence/boundedness classes as part
   of the admissibility data.

**Proof.** If a test function is supported where $\chi=1$, then
$W=V$, $\nabla\chi=0$, $\Delta\chi=0$, $\mathcal E_\chi=0$, and $g=0$ on the
support. The localized equation and local energy inequality therefore reduce
exactly to the represented equation and represented local energy inequality
handled in Lemma PS7.2.

If the test support meets $\operatorname{supp}\nabla\chi$, Lemma PS5.4 shows
that the localized equation contains the nonzero source vector
$\mathcal E_\chi$ and the divergence identity becomes
$\nabla\cdot W=g$ rather than zero. These terms change the integration by
parts used in the unforced local energy inequality and in the pressure flux.
Thus the ordinary suitability conclusion is not logically available from the
localized variables alone. The correct admissibility statement is the
controlled localized system: the distributional equation from `PS6`, the
source bounds from `PS5`, and the defect convergence from `PS6` are retained
and routed to the Type II state or defect nodes. Treating this localized
system as an
ordinary suitable profile would discard exactly the cutoff terms that `PS5`
and `PS6` were built to track.

### Specific Estimate

The decisive passage-to-the-limit estimate is

$$
\pi_n\rightharpoonup \pi\text{ in }L^{3/2},\qquad
v_n\to v\text{ in }L^3
\quad\Longrightarrow\quad
\pi_n v_n\cdot\nabla\phi\to\pi v\cdot\nabla\phi.
$$

This is the pressure-flux term where the compactness topology is most
important. For localized variables, the equally decisive check is

$$
\operatorname{supp}\phi\subset\{\chi=1\}
\quad\text{or}\quad
(\mathcal E_\chi,g)\text{ are retained as admissibility data}.
$$

There is no intermediate option in which cutoff terms are simply ignored.

### Practical Verification Steps

1. Fix a nonnegative local energy test function.
2. Write the approximating local energy inequality.
3. Pass the linear, quadratic, cubic, pressure, and dissipation terms to the
   limit.
4. In represented variables, pull back the test function and compute the drift
   terms.
5. If a cutoff was used, decide whether the test region lies in $\{\chi=1\}$
   or intersects the cutoff annulus.
6. Use ordinary suitability only in unlocalized or cutoff-inactive regions;
   otherwise retain the localized source and divergence-defect data.

## Estimate Step $B_{\mathrm{PS7}}$

The estimate step is Lemma PS7.1 in the Type I branch, Lemma PS7.2 in the
represented Type II branch, and Lemma PS7.3 whenever cutoff-localized variables
are used.

## Failure Case

Failure name: admissibility-inheritance failure.

Analytic meaning: the limit exists as a distributional solution but has not
yet been proved to belong to the local class required by later regularity,
Liouville, or branch-exclusion arguments.

## Refinement Step

Allowed refinements:

1. shrink to a compact subcylinder;
2. strengthen convergence by returning to `PS6`;
3. repair pressure normalization by returning to `PS4`;
4. include missing represented drift terms explicitly;
5. move the test region into $\{\chi=1\}$ or retain the cutoff source and
   divergence-defect data explicitly.

Progress measure: strict cylinder shrinkage or explicit repair of one missing
local energy term.

## Data Passed Forward

The next proof step is `PS8`. The data passed forward are exactly one of the
following admissibility records:

1. an ordinary suitable Navier--Stokes limit;
2. a represented drift-suitable limit satisfying the drifted local energy
   inequality from Lemma PS7.2;
3. a controlled localized limit with $\mathcal E_\chi$, $g=V\cdot\nabla\chi$,
   and their compactness or defect classes retained.

In record form,

$$
\Gamma_{\mathrm{PS7}}
=
\Gamma_{\mathrm{PS6}}
\cup
\{\text{ordinary suitability, drift-suitability, or controlled localized admissibility},
(\mathcal E_\chi,g)\text{ retained wherever cutoff defects are active}\}.
$$

---

# 14. `PS8` -- Activity and Nontriviality

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is the limiting profile $(U,Q)$ or $(V,P)$ from `PS6` and `PS7`.
The lower bound is measured on a compact cylinder where strong convergence is
verified.

### Standing Assumptions

The incoming hypotheses contain a compact activity packet

$$
Q_{\mathrm{act}}\Subset\Omega_{\mathrm{norm}},
\qquad
\int_{Q_{\mathrm{act}}}|V_n|^3\ge\eta_1>0
$$

for every $n$, and strong convergence on that same compact cylinder from
`PS6`. If the compactness variable is $W_n=\chi V_n$, then the branch also
contains the activity-compatible cutoff condition

$$
\chi\equiv1
\quad\hbox{on the spatial support of }Q_{\mathrm{act}},
$$

or a separately proved lower bound for $W_n$ on a compact activity cylinder.
If the only available lower bound is on the terminal cylinder
$B_1\times(-1,0)$, this node cannot run; the branch must return to `PS3` and
prove no-escape to move velocity activity to a compact negative-time cylinder.

### Objects Inspected

Inspect the compact cylinder $Q_{\mathrm{act}}$, the velocity lower bound
$\int_{Q_{\mathrm{act}}}|V_n|^3\ge\eta_1$, any cutoff-retention condition, and
the strong convergence $V_n\to V$ or $W_n\to W$ in
$L^3(Q_{\mathrm{act}})$.

### Dependencies Used

The compact velocity lower bound comes from `PS1` and `PS3`; activity-compatible
cutoff retention comes from `PS5`; compactness comes from `PS6`;
admissibility comes from `PS7`.

### Local Obstruction Predicate

$P_{\mathrm{PS8}}$ holds if the profile is zero or if the proof cannot
transfer positive concentration from the approximating sequence to the limit.

### Local Lemmas to Prove

**Lemma PS8.1 -- Strong convergence preserves positive velocity activity.**
Let $v_n\to v$ strongly in $L^3(Q_{\mathrm{act}})$ and

$$
\int_{Q_{\mathrm{act}}}|v_n|^3\,dx\,dt\ge\eta_1>0
$$

for all $n$. Then

$$
\int_{Q_{\mathrm{act}}}|v|^3\,dx\,dt\ge\eta_1.
$$

**Proof.** Strong convergence in $L^3$ implies
$\|v_n-v\|_{L^3(Q_{\mathrm{act}})}\to0$. Hence

$$
\left|
\|v_n\|_{L^3(Q_{\mathrm{act}})}
-\|v\|_{L^3(Q_{\mathrm{act}})}
\right|
\le
\|v_n-v\|_{L^3(Q_{\mathrm{act}})}
\to0.
$$

Cubing the convergent norms gives

$$
\int_{Q_{\mathrm{act}}}|v_n|^3
\to
\int_{Q_{\mathrm{act}}}|v|^3.
$$

Taking limits in the lower bound gives the result.

**Lemma PS8.2 -- Nonvanishing in centered Type I variables.**
Let $(U,Q)$ be an ancient Type I limit with compact velocity activity and set

$$
V(y,\tau)=\sqrt{-t}\,U(y\sqrt{-t},t),
\qquad
\tau=-\log(-t).
$$

If $\int_{Q_{\mathrm{act}}}|U|^3\,dx\,dt\ge\eta_1$ on a compact negative-time cylinder
$Q_{\mathrm{act}}$, then there are a compact cylinder $K$ in $(y,\tau)$ variables and
$\eta_0>0$ such that

$$
\iint_K |V(y,\tau)|^3\,dy\,d\tau\ge\eta_0.
$$

**Proof.** On compact subsets of $t<0$, the change of variables
$x=e^{-\tau/2}y$, $t=-e^{-\tau}$ has Jacobian

$$
dx\,dt=e^{-5\tau/2}\,dy\,d\tau.
$$

Since $U=e^{\tau/2}V$, one has

$$
|U|^3\,dx\,dt=e^{-\tau}|V|^3\,dy\,d\tau.
$$

On the compact image of $Q_{\mathrm{act}}$, the weight $e^{-\tau}$ is bounded above and
below by positive constants. Hence the positive lower bound transfers to
centered variables with a possibly different constant.

**Lemma PS8.3 -- Pressure-based zero-limit test is only a conditional backup.**
Let $V_n\to0$ strongly in $L^3(Q_r)$ and assume the normalized pressures
satisfy the strong convergence

$$
P_n-(P_n)_{B_r}(t)
\to0
\quad\text{strongly in }L^{3/2}(Q_r).
$$

Then

$$
\int_{Q_r}|V_n|^3+
\int_{Q_r}|P_n-(P_n)_{B_r}(t)|^{3/2}
\to0.
$$

Therefore a fixed positive $C+D$ lower bound on $Q_r$ rules out the zero
limit. Weak pressure convergence is not enough for this test, and velocity
convergence to zero does not by itself force the harmonic pressure remainder
to vanish.

**Proof.** The velocity term vanishes by strong $L^3$ convergence. For the
pressure, the displayed strong $L^{3/2}$ convergence gives the conclusion
directly. One possible way to prove that strong pressure convergence is a
local decomposition from `PS4` in which the Calderon--Zygmund part tends to
zero and the harmonic remainder is separately shown to converge to zero modulo
time functions. Without that additional pressure input, this lemma is not
available. The main nonvanishing route in `PS8` is Lemma PS8.1, using compact
velocity activity.

### Specific Estimate

The decisive estimate is the norm convergence

$$
v_n\to v\text{ in }L^3(Q_{\mathrm{act}})
\quad\Longrightarrow\quad
\int_{Q_{\mathrm{act}}}|v|^3=\lim_{n\to\infty}\int_{Q_{\mathrm{act}}}|v_n|^3.
$$

Pressure mass is not used for the main nontriviality proof. It can be used
only under the strong pressure convergence hypothesis of Lemma PS8.3.

### Practical Verification Steps

1. Identify the compact cylinder carrying positive concentration.
2. Verify strong $L^3$ convergence on that cylinder.
3. If the compactness variable is localized, verify that the cutoff is
   identically one on the active region or reprove activity for the localized
   variable.
4. Pass the velocity lower bound to the limit.
5. If using centered variables, transfer the lower bound through the
   self-similar change of variables.
6. Use pressure mass only as a conditional backup after strong pressure
   convergence has been proved.

## Estimate Step $B_{\mathrm{PS8}}$

The estimate step is the lower-bound passage in Lemmas PS8.1--PS8.3.

## Failure Case

Failure name: vanishing-profile failure.

Analytic meaning: the compactness process produced the zero profile or failed
to preserve the concentration lower bound.

## Refinement Step

Allowed refinements:

1. return to `PS3` to choose a compact concentration window;
2. return to `PS6` to strengthen convergence on that window;
3. return to `PS4` to strengthen pressure stability;
4. select a different active frame in a Type II branch.

Progress measure: active-window selection or explicit zero-profile
contradiction.

## Data Passed Forward

The next proof step is `PS9` in the Type I branch and `PS10` in the Type II
branch. The data passed forward are

$$
\Gamma_{\mathrm{PS8}}
=
\Gamma_{\mathrm{PS7}}
\cup
\{K\Subset\Omega_{\rm norm},\ \eta_*>0,\
\iint_K|V|^3\ge\eta_*,\
\text{nonzero profile or retained active frame}\}.
$$

---

# 15. `PS9` -- Type I Ancient-Profile Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknowns are $(U,Q)$ in ancient variables and $(V,\Pi)$ in centered
variables. The equation is NS for $(U,Q)$ and the centered self-similar NS
equation for $(V,\Pi)$.

### Standing Assumptions

The incoming hypotheses contain the local Type I bound near the singular point,
compactness through `PS6`, suitability through `PS7`, and nonvanishing through
`PS8`.

The Type I bound is the original-variable statement that there are
$\rho>0$ and $M<\infty$ such that

$$
\operatorname*{ess\,sup}_{T-\rho^2<t<T}
\sqrt{T-t}\,\|u(t)\|_{L^\infty(B_\rho(x_*))}\le M.
$$

### Objects Inspected

Inspect:

$$
\|U(t)\|_{L^\infty(\mathbb R^3)}\le M(-t)^{-1/2},
\qquad
\|V\|_{L^\infty(\mathbb R^3\times\mathbb R)}\le M,
$$

the centered equation, and local energy/pressure bounds for $(V,\Pi)$ on
compact cylinders.

### Dependencies Used

The Type I alternative comes from `PS3`; compactness from `PS6`; suitability
from `PS7`; nontriviality from `PS8`; the centered equation from `PS5`.

### Local Obstruction Predicate

$P_{\mathrm{PS9}}$ holds when the Type I limit has not been verified as an
ancient suitable, bounded, nonzero profile in centered variables.

### Local Lemmas to Prove

**Lemma PS9.1 -- Diagonal Type I ancient extraction and inherited bound.**
After passing to one diagonal subsequence, the Type I rescalings converge on
every compact cylinder in $\mathbb R^3\times(-\infty,0)$ to an ancient limit
$(U,Q)$ satisfying, for every compact interval
$I=[-S,-\sigma]\Subset(-\infty,0)$,

$$
\|U\|_{L^\infty(\mathbb R^3\times I)}
\le M\sigma^{-1/2}.
$$

**Proof.** Fix $R,S\in\mathbb N$ and
$\sigma=1/j$ with $j\in\mathbb N$. On the compact normalized cylinder

$$
B_R\times[-S,-\sigma],
$$

the physical image of the rescaled sequence lies inside
$B_\rho(x_*)\times(T-\rho^2,T)$ for all sufficiently large $k$. Hence the
Type I estimate gives

$$
|u_k(y,s)|
=\lambda_k|u(x_*+\lambda_ky,T+\lambda_k^2s)|
\le M(-s)^{-1/2}
\le M\sigma^{-1/2}
$$

for a.e. $(y,s)\in B_R\times[-S,-\sigma]$. On this compact cylinder, `PS6`
gives a subsequential limit. Extract successively over the countable family

$$
R\in\mathbb N,\qquad S\in\mathbb N,\qquad \sigma\in\{1,1/2,1/3,\ldots\},
$$

and take the diagonal subsequence. The resulting limit $(U,Q)$ is defined on
all of $\mathbb R^3\times(-\infty,0)$ in the sense of compatible compact
cylinder restrictions. The a.e. convergence obtained from strong local
$L^3$ convergence gives

$$
|U(y,s)|\le M(-s)^{-1/2}
$$

on each compact cylinder, hence on
$\mathbb R^3\times(-\infty,0)$ after countable exhaustion. In particular, on
$I=[-S,-\sigma]$ the essential supremum is bounded by $M\sigma^{-1/2}$. This
is a diagonal compact-cylinder conclusion, not a global compactness statement
at spatial infinity.

**Lemma PS9.2 -- Smoothness of the Type I ancient profile.**
The ancient profile $U$ is smooth on
$\mathbb R^3\times(-\infty,0)$, and the centered profile $V$ is smooth on
$\mathbb R^3\times\mathbb R$ after fixing a pressure gauge.

**Proof.** On each compact cylinder
$Q\Subset\mathbb R^3\times(-\infty,0)$, Lemma PS9.1 gives $U\in L^\infty(Q)$.
Thus $U\in L^s_tL^r_x(Q)$ for finite $r,s$ with
$2/s+3/r<1$. Serrin interior regularity gives local Holder regularity.
Interior Stokes estimates applied to

$$
\partial_tU-\Delta U+\nabla Q=-\nabla\cdot(U\otimes U)
$$

then bootstrap the solution to $C^\infty$ on smaller cylinders. The pressure
is recovered locally from
$-\Delta Q=\partial_i\partial_j(U_iU_j)$ plus a harmonic function; elliptic
interior estimates make it smooth after fixing a time-dependent gauge. The
centered change of variables is smooth for $t<0$, so $(V,\Pi)$ is smooth on
$\mathbb R^3\times\mathbb R$ after a pressure gauge is fixed.

**Lemma PS9.3 -- Bounded centered nonzero profile.**
The centered profile satisfies

$$
\|V\|_{L^\infty(\mathbb R^3\times\mathbb R)}\le M
$$

and has a compact-cylinder lower bound

$$
\iint_K |V|^3\,dy\,d\tau\ge\eta_0>0.
$$

**Proof.** The bound follows from
$V(y,\tau)=\sqrt{-t}\,U(y\sqrt{-t},t)$ and Lemma PS9.1. The lower bound is
Lemma PS8.2.

**Lemma PS9.4 -- Local energy and pressure bounds in centered variables.**
For every compact cylinder $B_R\times I$ in $(y,\tau)$ variables,

$$
\int_I\int_{B_R}(|V|^2+|\nabla V|^2)\,dy\,d\tau<\infty,
\qquad
\Pi-b(\tau)\in L^{3/2}(B_R\times I)
$$

for a suitable pressure gauge $b(\tau)$.

**Proof.** Pull the compact $(y,\tau)$ cylinder back to a compact
$(x,t)$ cylinder with $t<0$. On that compact set, the Jacobian and all powers
of $-t$ entering

$$
V=(-t)^{1/2}U,\qquad
\nabla_yV=(-t)\nabla_xU,\qquad
\Pi=(-t)Q
$$

are bounded above and below by positive constants. The local energy,
dissipation, and pressure bounds for $(U,Q)$ therefore transfer to
$(V,\Pi)$ after subtracting the corresponding time-dependent pressure mean.

**Lemma PS9.5 -- Type I branch data.**
Define the Type I branch data
$\mathcal R_{\mathrm{TI}}(V,\Pi)$ to consist exactly of the data returned by
`PS9`:

1. the domain statement
   $(V,\Pi)$ is defined on $\mathbb R^3\times\mathbb R$ modulo addition of a
   function of $\tau$ to $\Pi$;
2. the centered equation

   $$
   \partial_\tau V+(V\cdot\nabla)V+\nabla\Pi-\Delta V
   +\frac12V+\frac12y\cdot\nabla V=0,\qquad
   \nabla\cdot V=0;
   $$

3. the pressure relation

   $$
   -\Delta\Pi=\partial_i\partial_j(V_iV_j)
   $$

   in distributions after fixing the pressure gauge locally in time;
4. the bounds

   $$
   \|V\|_{L^\infty(\mathbb R^3\times\mathbb R)}\le M,
   \qquad
   V\in L^\infty_{\tau,\mathrm{loc}}L^2_{y,\mathrm{loc}}
   \cap L^2_{\tau,\mathrm{loc}}\dot H^1_{y,\mathrm{loc}},
   \qquad
   \Pi-b(\tau)\in L^{3/2}_{\mathrm{loc}};
   $$

5. the inherited centered drift local energy inequality on every compact
   centered cylinder, with the same pressure gauge as in item 4;
6. a nontriviality witness: one compact cylinder
   $K\Subset\mathbb R^3\times\mathbb R$ and one number $\eta_0>0$ such that

   $$
   \iint_K |V|^3\,dy\,d\tau\ge\eta_0;
   $$

7. the normalization data used to produce the centered profile: the singular point,
   the scaling sequence, the centered time coordinate, and the pressure mean
   convention.

If Lemmas PS9.1--PS9.4 hold, then $\mathcal R_{\mathrm{TI}}(V,\Pi)$ contains
the Type I data used by the downstream Type I class nodes. If any item is
missing, the downstream Type I class nodes do not use this package.

**Proof.** Items 1 and 7 are determined by the Type I normalization selected
in `PS3`, the pressure gauge from `PS4`, and the
compactness extraction from `PS6`. Without the domain, scale, time coordinate,
and pressure convention, later Type I nodes cannot
distinguish a genuine centered ancient profile from a locally represented
Type II window or from a profile with an unfixed pressure representative.

Item 2 is exactly Lemma PS5.2 applied to the smooth ancient limit from
Lemma PS9.2. Since Lemma PS9.2 upgrades the profile to smoothness on compact
cylinders, the equation is available both distributionally and classically.
The divergence-free condition is inherited under the same change of variables,
so no extra compatibility condition is introduced at this stage.

Item 3 follows by taking divergence of the equation in item 2 and using
$\nabla\cdot V=0$. The drift terms contribute no pressure source:

$$
\nabla\cdot V=0,\qquad
\nabla\cdot(y\cdot\nabla V)
=y\cdot\nabla(\nabla\cdot V)+\nabla\cdot V=0,
$$

and
$\nabla\cdot((V\cdot\nabla)V)=\partial_i\partial_j(V_iV_j)$. Hence the
pressure is determined by the Calderon--Zygmund source modulo addition of a
function of $\tau$. Fixing the gauge from item 4 makes the
pressure term in the local energy inequality unambiguous.

Item 4 is the union of Lemmas PS9.1, PS9.3, and PS9.4. The global
$L^\infty$ bound comes from the Type I scale-invariant estimate. The local
energy and pressure spaces transfer from the suitable ancient limit because
the centered change of variables has a Jacobian bounded above and below on
each compact cylinder. The pressure statement is intentionally local and
mean-subtracted; no global pressure integrability at spatial infinity is being
assumed.

Item 5 is the represented drift-suitability conclusion from `PS7` transported
through the same smooth centered change of variables. On each compact cylinder
the pullback of a nonnegative centered test function is an admissible
nonnegative test function for the ancient variables. The drift and Jacobian
factors are exactly those already present in the centered equation, so the
local energy inequality closes with the pressure representative from item 4
and does not create an untracked error term. This is not an ordinary
Navier--Stokes local energy inequality in the centered variables; it is the
local energy inequality for the centered equation.

Item 6 is Lemma PS9.3, which imports the nonvanishing lower bound from `PS8`.
Because the lower bound holds on a fixed compact cylinder in centered
coordinates, downstream Type I analysis may use nontriviality without
reopening the concentration extraction or the scale-selection argument.

The downstream Type I class steps use only the centered equation,
divergence-free condition, pressure gauge, boundedness, local
energy/suitability, and nonzero witness. Every one of those objects is present
in $\mathcal R_{\mathrm{TI}}(V,\Pi)$. Conversely, if any one of the listed
entries is absent, the downstream hypotheses are not all available, so the
Type I handoff has not been proved.

### Specific Estimate

The decisive estimate is the inherited Type I bound

$$
|V(y,\tau)|
=\sqrt{-t}\,|U(y\sqrt{-t},t)|
\le M.
$$

This node does not prove stationarity in $\tau$, whole-space
$L^3(\mathbb R^3)$ bounds, tightness, spatial decay, compact orbit structure,
or finite energy at infinity. Any downstream theorem requiring one of those
hypotheses must obtain it from a later node.

### Practical Verification Steps

1. Verify the Type I alternative from `PS3`.
2. Extract the ancient limit by diagonal compact-cylinder extraction.
3. Use the admissibility class from `PS7`.
4. Prove the inherited ancient Type I bound.
5. Apply Serrin regularity and Stokes bootstrapping on compact cylinders.
6. Transform to centered variables and prove boundedness, local energy, and
   nonvanishing.
7. Assemble the branch data $\mathcal R_{\mathrm{TI}}(V,\Pi)$ and
   pass the normalized Type I profile to the Type I class nodes only after
   every item in the package is present.

## Estimate Step $B_{\mathrm{PS9}}$

The estimate step is the inherited-bound, regularity, centered-variable, and
branch-routing package in Lemmas PS9.1--PS9.5.

## Failure Case

Failure name: Type I profile-admissibility failure.

Analytic meaning: an active limit exists, but it has not been verified as the
bounded smooth nonzero centered ancient profile required for Type I class
analysis.

## Refinement Step

Allowed refinements:

1. shrink compact cylinders;
2. fix a centered pressure gauge;
3. return to `PS7` if suitability is missing;
4. return to `PS8` if nontriviality is missing.

Progress measure: compact-cylinder shrinkage or fixation of one pressure
gauge.

## Data Passed Forward

The next proof steps are the Type I class nodes. The data passed forward are

$$
\Gamma_{\mathrm{PS9}}
=
\Gamma_{\mathrm{PS8}}
\cup
\{(V,\Pi)\text{ normalized nonzero bounded Type I ancient profile},\,
\mathcal R_{\mathrm{TI}}(V,\Pi)\text{ Type I branch data},\
\text{no stationarity, whole-space }L^3\text{, decay, tightness, or compact orbit claim}\}.
$$

---

# 16. `PS10` -- Type II Local Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

Work near $z_*=(x_*,T)$ with a suitable weak solution and the local CKN
quantities. The branch is local: it is defined by positive concentration and
failure of a local Type I bound in every backward neighborhood of $z_*$.

### Standing Assumptions

The raw Type II branch-entry hypotheses contain the singular-entry conclusion
from `C_mu`, the two concentration lower bounds from `PS1`, and the negation
of the local Type I alternative from `PS3`. Compactness, admissibility, and
nontrivial limiting profiles from `PS6`--`PS8` are optional additional data;
they are not required to enter the raw Type II branch.

### Objects Inspected

Inspect the concentration sequence $r_n$, the velocity and combined CKN lower
bounds, the original-variable local rate quantity
$\sqrt{T-t}\|u(t)\|_{L^\infty(B_\rho(x_*))}$, and any separately selected
rate-burst points or scales.

### Dependencies Used

Positive concentration comes from `C_mu` and `PS1`; rate classification comes
from `PS3`; suitability and pressure normalization come from `D_E` and `PS4`
when compact or represented windows are used later.

### Local Obstruction Predicate

$P_{\mathrm{PS10}}$ holds if the proof cannot verify that the active
concentration sequence lies in the Type II alternative.

### Local Lemmas to Prove

**Lemma PS10.1 -- Local Type II sequence criterion.**
Let $x_*\in\Sigma(T)$ and let $r_n\downarrow0$ satisfy

$$
C(u;z_*,r_n)\ge\varepsilon_v,
\qquad
C(u;z_*,r_n)+D(p;z_*,r_n)\ge\varepsilon_0,
$$

and no local Type I bound holds at $z_*$. Then $\{r_n\}$ is a positive local
Type II concentration sequence.

**Proof.** The verified data are

$$
r_n\downarrow0,\qquad
C(u;z_*,r_n)\ge\varepsilon_v,
\qquad
C(u;z_*,r_n)+D(p;z_*,r_n)\ge\varepsilon_0,
$$

and

$$
\forall \rho>0,\quad
\operatorname*{ess\,sup}_{T-\rho^2<t<T}
\sqrt{T-t}\,\|u(t)\|_{L^\infty(B_\rho(x_*))}=\infty.
$$

These displayed statements are the complete raw local Type II predicate used
by the branch: positive velocity concentration, positive combined CKN
concentration on shrinking cylinders, and failure of every local Type I bound
at $z_*$. The lemma is only a branch-entry criterion. It does not exclude the
Type II branch and does not assert existence of a compact limiting profile.
It identifies the exact data that the later Type II state-space nodes must
discharge.

**Lemma PS10.2 -- Rescaling of a local Type II sequence.**
For

$$
u_n(y,s)=r_nu(x_*+r_ny,T+r_n^2s),
\qquad
p_n(y,s)=r_n^2p(x_*+r_ny,T+r_n^2s),
$$

the Type II concentration lower bound becomes a fixed-cylinder bound on
$Q_1$:

$$
\int_{Q_1}|u_n|^3\,dy\,ds\ge\varepsilon_v,
$$

and

$$
\int_{Q_1}
\left(|u_n|^3+|p_n-(p_n)_{B_1}|^{3/2}\right)\,dy\,ds
\ge\varepsilon_0.
$$

**Proof.** With $x=x_*+r_ny$ and $t=T+r_n^2s$,

$$
dy\,ds=r_n^{-5}\,dx\,dt,\qquad
|u_n|^3=r_n^3|u|^3,\qquad
|p_n-(p_n)_{B_1}|^{3/2}
=r_n^3|p-p_{B_{r_n}(x_*)}|^{3/2}.
$$

Here

$$
(p_n)_{B_1}(s)=r_n^2p_{B_{r_n}(x_*)}(T+r_n^2s),
$$

so the pressure mean-subtraction is exactly the rescaled physical mean.
Therefore

$$
\int_{Q_1}\left(|u_n|^3+|p_n-(p_n)_{B_1}|^{3/2}\right)\,dy\,ds
=C(u;z_*,r_n)+D(p;z_*,r_n)\ge\varepsilon_0.
$$

The same change of variables gives

$$
\int_{Q_1}|u_n|^3\,dy\,ds=C(u;z_*,r_n)\ge\varepsilon_v.
$$

**Lemma PS10.3 -- Scale identity for the non-Type-I branch condition.**
For every selected scale $r_n$ and every fixed normalized ball $B_R$,
the Type I rate quantity transforms by the exact identity

$$
\sqrt{-s}\,\|u_n(s)\|_{L^\infty(B_R)}
=\sqrt{T-t}\,
\|u(t)\|_{L^\infty(B_{Rr_n}(x_*))},
\qquad t=T+r_n^2s.
$$

**Proof.** For $t=T+r_n^2s$ and a fixed ball $B_R$ in rescaled variables,

$$
\sqrt{-s}\,\|u_n(s)\|_{L^\infty(B_R)}
=\sqrt{T-t}\,
\|u(t)\|_{L^\infty(B_{Rr_n}(x_*))}.
$$

Thus a physical local Type I bound at $z_*$ transports to uniform bounds on
every fixed normalized compact backward cylinder once $n$ is large enough
that $B_{Rr_n}(x_*)$ lies inside the physical Type I ball. This proves the
direction needed for consistency: if the branch had a physical Type I bound,
the selected Type II rescalings would inherit it.

The converse must be stated with its quantifiers intact. A bound on one
chosen normalized cylinder, or even on one selected scale sequence, controls
only the shrinking physical cylinders
$B_{Rr_n}(x_*)\times(T-r_n^2R^2,T)$. It does not by itself give a fixed
physical radius $\rho>0$ on which the Type I predicate holds for all times
near $T$. Therefore `PS10` does not infer non-Type-I status from the
rescaled sequence alone. It uses the original quantified negation from
`PS3`,

$$
\forall \rho>0,\quad
\operatorname*{ess\,sup}_{T-\rho^2<t<T}
\sqrt{T-t}\,\|u(t)\|_{L^\infty(B_\rho(x_*))}=\infty,
$$

and uses the identity above only to check that all normalized windows are
consistent with the Type II branch hypotheses. If a later argument produces a genuine
fixed-radius Type I bound, the branch must return to `PS3` and be reclassified
instead of silently remaining in `PS10`.

**Lemma PS10.4 -- Type II branch data.**
Define the raw Type II branch data
$\mathcal R_{\mathrm{II}}(z_*,\{r_n\})$ to consist exactly of the data
returned by `PS10`:

1. the singular point $z_*=(x_*,T)$ and a sequence $r_n\downarrow0$;
2. the positive local velocity and CKN lower bounds

   $$
   C(u;z_*,r_n)\ge\varepsilon_v,
   \qquad
   C(u;z_*,r_n)+D(p;z_*,r_n)\ge\varepsilon_0;
   $$

3. the original-variable non-Type-I predicate from Lemma PS10.3;
4. the normalized variables

   $$
   u_n(y,s)=r_nu(x_*+r_ny,T+r_n^2s),
   \qquad
   p_n(y,s)=r_n^2p(x_*+r_ny,T+r_n^2s),
   $$

   together with their mean-subtracted pressure convention;
5. the fixed-cylinder lower bounds from Lemma PS10.2, including the velocity
   lower bound;
6. suitability on compact truncated normalized windows and the pressure mean
   convention inherited from `D_E`, `Rec_N`, and `PS4` whenever pressure is
   used on such windows;
7. the Type II scale-state table $\mathfrak S_{\mathrm{II}}$ from `PS3`,
   with unresolved entries assigned to `PS11`, `PS16`--`PS20`, `PS28`, or
   `PS34` as appropriate;
8. a statement that the concentration scales $r_n$ are not being identified
   with Type II rate-burst scales unless a later selection proves that
   identification.

If later Type II analysis needs actual rate-burst points, it must select a
separate packet

$$
(x_j,t_j),\qquad
t_j\uparrow T,\qquad
x_j\to x_*,
\qquad
\sqrt{T-t_j}\,|u(x_j,t_j)|\to\infty
$$

in an appropriate representative or essential-supremum sense from the
original non-Type-I predicate. This packet is not automatic from the CKN
concentration scales.

If compactness and activity have already been produced by `PS6`--`PS8`, the
branch may also carry a compact Type II profile packet containing the limit,
active frame, pressure representative, source/divergence defects, and
nonzero witness. If not, `PS10` passes only the raw branch above, with
compactness marked as an unresolved obligation rather than a proved limit.

**Proof.** Items 1--3 are exactly the branch predicate from Lemma PS10.1:
shrinking scales, positive velocity concentration, positive combined CKN
concentration, and the quantified failure of a local Type I bound. Items 4
and 5 are Lemma PS10.2, including the pressure mean transformation and the
separate velocity lower bound. Item 6 is the minimal local suitability and
pressure convention needed to use the normalized windows later; it is not a
claim that compactness has already been proved. Item 7 is the scale-state
table created in `PS3`; it prevents `PS10` from
prematurely choosing compact single-core, cascade, finite-cost, radiation, or
residual behavior. Item 8 prevents a later Type II proof from silently using
the CKN concentration scales as rate-burst scales.

These items are also exhaustive for this node. Later Type II nodes need only
the positive sequence, the non-Type-I branch status, the normalized
pressure/suitability package, and the table identifying which scale or
state-space alternatives remain. A missing item means the Type II branch has
not been entered with enough data to run the Type II assembly, so successful
handoff would be logically stronger than the proved hypotheses.

### Specific Estimate

The decisive branch-entry assertion is

$$
C(u;z_*,r_n)\ge\varepsilon_v,
\qquad
C(u;z_*,r_n)+D(p;z_*,r_n)\ge\varepsilon_0
\quad\text{and}\quad
\neg\exists(\rho,M):
\operatorname*{ess\,sup}_{T-\rho^2<t<T}
\sqrt{T-t}\|u(t)\|_{L^\infty(B_\rho(x_*))}\le M.
$$

The non-Type-I status is imported from the original quantified predicate. It
is not inferred from one rescaled sequence or one normalized compact window.

### Practical Verification Steps

1. Use the positive concentration sequence.
2. Verify that the local Type I alternative from `PS3` is false.
3. Rescale the sequence to fixed cylinders.
4. Record both fixed-cylinder lower bounds, velocity and combined CKN.
5. Assemble $\mathcal R_{\mathrm{II}}(z_*,\{r_n\})$.
6. Keep concentration scales separate from any later rate-burst points.
7. Pass the raw branch to the Type II state-space checks only after the
   package contains the pressure convention, compact-window suitability, and
   scale-table data. Attach compact-profile data only if `PS6`--`PS8` actually
   produced them.

## Estimate Step $B_{\mathrm{PS10}}$

The estimate step is the local dichotomy, scaling verification, and
branch-data assembly in Lemmas PS10.1--PS10.4.

## Failure Case

Failure name: Type II branch-entry failure.

Analytic meaning: the proof has not established the defining properties of a
local Type II concentration sequence.

## Refinement Step

Allowed refinements:

1. pass to a subsequence of Type II scales;
2. return to `PS3` if the local rate alternative was not fixed;
3. return to `PS1` if the fixed-cylinder concentration lower bound is missing;
4. select an active represented window for later Type II nodes.

Progress measure: subsequence selection or fixed declaration of Type I versus
Type II.

## Data Passed Forward

The next proof step is `PS11`. The data passed forward are

$$
\Gamma_{\mathrm{PS10}}
=
\Gamma_{\mathrm{PS3}}
\cup
\{\text{raw positive local Type II concentration sequence},\,
\mathcal R_{\mathrm{II}}(z_*,\{r_n\})\text{ raw Type II branch data},\
\text{optional compact Type II profile packet only if supplied by }PS6\text{--}PS8\}.
$$

---

# 17. `PS11` -- Type II Scale-Cascade and Scale-Collapse Alternatives

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

Work on selected Type II windows in represented variables. The unknowns are
the active profiles, their parabolic frames, their retained activity records,
and the logarithmic scale derivative. A frame is always recorded as

$$
\mathfrak F_n^j=(x_n^j,t_n^j,\lambda_n^j),
\qquad
\lambda_n^j>0,
$$

with coordinates

$$
y=\frac{x-x_n^j}{\lambda_n^j},
\qquad
s=\frac{t-t_n^j}{(\lambda_n^j)^2}.
$$

If all selected frames are terminal, the branch may state $t_n^j=T$ for every
$j$. Once new cores are extracted from compact cylinders, different center
times may appear and must be retained.

### Standing Assumptions

The incoming hypotheses contain positive local Type II concentration from `PS10`,
the `PS10` branch data and scale-state table, local suitability, pressure
control from earlier nodes, and selected active frames when more than one
concentration core is present. Lower concentration bounds are never treated as
upper compactness bounds; any use of `PS6` below requires the `PS6`
compactness package to be present.

### Objects Inspected

Inspect parabolic active-frame parameters, time-slice or window core-mass
lower bounds, logarithmic drift $d(\tau)$, compact-window estimates for the
represented velocities and pressures, and the unresolved entries of the Type
II scale-state table.

### Dependencies Used

Type II branch data come from `PS10`;
represented variables and modulation come from `PS4` and `PS5`; compactness
and admissibility come from `PS6` and `PS7`; activity comes from `PS8`.

### Local Obstruction Predicate

$P_{\mathrm{PS11}}$ holds when the Type II branch has not been assigned to one
of the explicit scale alternatives needed by the later state-space analysis.

### Local Lemmas to Prove

**Lemma PS11.1 -- Exhaustive pairwise parabolic-frame partition.**
Let two active frames be

$$
\mathfrak F_n^i=(x_n^i,t_n^i,\lambda_n^i),
\qquad
\mathfrak F_n^j=(x_n^j,t_n^j,\lambda_n^j).
$$

After passing to a subsequence, exactly one ordered alternative occurs:

1. parabolic separation:

   $$
   \frac{|x_n^i-x_n^j|}{\max(\lambda_n^i,\lambda_n^j)}
   +
   \frac{|t_n^i-t_n^j|}
   {\max((\lambda_n^i)^2,(\lambda_n^j)^2)}
   \to\infty;
   $$

2. comparable parabolic frame:

   $$
   0<\liminf_n\frac{\lambda_n^i}{\lambda_n^j}
   \le
   \limsup_n\frac{\lambda_n^i}{\lambda_n^j}<\infty,
   $$

   and the normalized spatial and temporal offsets are bounded;

3. same-point parabolic scale cascade:

   the normalized spatial and temporal offsets are bounded, but

   $$
   \frac{\lambda_n^i}{\lambda_n^j}\to0
   \quad\text{or}\quad
   \frac{\lambda_n^i}{\lambda_n^j}\to\infty;
   $$

4. comparable scale with unbounded normalized time shift:

   the scale ratio and normalized spatial offset are bounded, while

   $$
   \frac{|t_n^i-t_n^j|}{(\lambda_n^i)^2}\to\infty
   $$

   after replacing $(\lambda_n^i)^2$ by a comparable squared scale if needed.

The fourth case is a time-hull or terminal-translation branch, not automatic
invisibility.

**Proof.** Pass to a subsequence so that the scale ratio converges in
$[0,\infty]$, the normalized spatial offset either remains bounded or tends to
$\infty$, and the normalized temporal offset either remains bounded or tends to
$\infty$. If the combined normalized space-time separation tends to $\infty$,
the pair is parabolically separated unless the only unbounded part is the
normalized time shift with comparable scales and bounded spatial offset; that
case is item 4. If the normalized space-time offsets remain bounded, a
positive finite scale-ratio limit gives item 2, and a zero or infinite
scale-ratio limit gives item 3. The ordered cases are mutually exclusive and
exhaust all subsequential possibilities.

If a later argument classifies a family rather than a pair, `PS11` requires an
incoming finite-family hypothesis or a finite budget. A sufficient budget is:
each active frame carries at least $\varepsilon_{\rm sd}>0$ of local CKN mass
in selected cylinders whose enlargements have overlap at most $N_0$, and the
containing window has CKN mass at most $M_{\rm win}$. Then

$$
N\varepsilon_{\rm sd}\le C N_0M_{\rm win}.
$$

Without such a budget, this node classifies finite subfamilies only.

**Lemma PS11.2 -- Logarithmic scale identity.**
Let $\Lambda(\tau)>0$ be absolutely continuous on $[\tau_1,\tau_2]$ and set
$\ell(\tau)=\log\Lambda(\tau)$. If

$$
d(\tau)=\ell'(\tau)
$$

in $L^1_{\rm loc}$, then

$$
\log\frac{\Lambda(\tau_2)}{\Lambda(\tau_1)}
=
\int_{\tau_1}^{\tau_2}d(\tau)\,d\tau.
$$

**Proof.** Set $\ell=\log\Lambda$. The hypothesis
$\ell\in AC_{\rm loc}$ and $d=\ell'$ in $L^1_{\rm loc}$ gives, for
$\tau_1<\tau_2$ in the interval,

$$
\ell(\tau_2)-\ell(\tau_1)=\int_{\tau_1}^{\tau_2}\ell'(\tau)\,d\tau
=\int_{\tau_1}^{\tau_2}d(\tau)\,d\tau.
$$

Substituting $\ell=\log\Lambda$ gives the displayed identity.

**Definitions -- retained core mass.**
For a represented core define

$$
M_R(\tau)=\int_{B_R}|V(Y,\tau)|^3\,dY.
$$

The core has **time-slice retained mass** on $[\tau_0,\infty)$ if there are
$R>0$ and $m_0>0$ such that

$$
M_R(\tau)\ge m_0
$$

for a.e. $\tau\ge\tau_0$. It has **window-retained mass** if there are
$R,L,m_0>0$ and intervals $I_j=[\tau_j,\tau_j+L]$, $\tau_j\to\infty$, such
that

$$
\int_{I_j}M_R(\tau)\,d\tau\ge m_0.
$$

Time-slice retained mass implies window-retained mass. Window-retained mass
does not imply time-slice retained mass and cannot be used as a pointwise
lower bound.

**Lemma PS11.3 -- Collapse plus time-slice retained core forces infinite
weighted drift cost.**
Let $\Lambda>0$ be absolutely continuous on $[\tau_0,\infty)$, set

$$
d(\tau)=\frac{d}{d\tau}\log\Lambda(\tau),
\qquad
d_-(\tau)=\max\{-d(\tau),0\},
$$

and assume

$$
\Lambda(\tau_j)\to0
\qquad
\text{for some }\tau_j\to\infty.
$$

Assume also that

$$
M_R(\tau)\ge m_0>0
\quad\text{for a.e. }\tau\ge\tau_0.
$$

Then

$$
\int_{\tau_0}^{\infty}d_-(\tau)M_R(\tau)\,d\tau=\infty.
$$

**Proof.** For every finite $T>\tau_0$, Lemma PS11.2 gives

$$
\log\Lambda(T)-\log\Lambda(\tau_0)
=
\int_{\tau_0}^{T}d(\tau)\,d\tau
\ge
-\int_{\tau_0}^{T}d_-(\tau)\,d\tau .
$$

If $\int_{\tau_0}^{\infty}d_-(\tau)\,d\tau<\infty$, then
$\log\Lambda(T)$ is bounded below uniformly for all $T\ge\tau_0$. This
contradicts $\log\Lambda(\tau_j)\to-\infty$. Hence

$$
\int_{\tau_0}^{\infty}d_-(\tau)\,d\tau=\infty.
$$

Multiplying by the time-slice retained lower bound gives

$$
\int_{\tau_0}^{\infty}d_-(\tau)M_R(\tau)\,d\tau
\ge
m_0\int_{\tau_0}^{\infty}d_-(\tau)\,d\tau
=\infty.
$$

**Lemma PS11.3b -- Window activity needs synchronized drift.**
Assume there exist $R,L,m_0>0$ and intervals
$I_j=[\tau_j,\tau_j+L]$, $\tau_j\to\infty$, such that

$$
\int_{I_j}M_R(\tau)\,d\tau\ge m_0.
$$

This does not by itself imply

$$
\int d_-(\tau)M_R(\tau)\,d\tau=\infty.
$$

The implication is valid if, in addition, there is $\kappa>0$ such that

$$
\int_{I_j}d_-(\tau)M_R(\tau)\,d\tau\ge\kappa
$$

on infinitely many pairwise disjoint intervals $I_j$.

**Proof.** Without synchronization, the activity may lie on windows where
$d_-$ is arbitrarily small while scale collapse occurs elsewhere. Under the
displayed synchronized lower bound, summing over infinitely many disjoint
intervals gives

$$
\int d_-(\tau)M_R(\tau)\,d\tau
\ge
\sum_j\int_{I_j}d_-(\tau)M_R(\tau)\,d\tau
\ge
\sum_j\kappa
=\infty.
$$

**Lemma PS11.4 -- Finite-cost scale collapse is excluded only under the
matching retained-mass hypothesis.**
If

$$
\int_{\tau_0}^{\infty}d_-(\tau)\,d\tau<\infty,
$$

then $\liminf_{\tau\to\infty}\Lambda(\tau)>0$. If only

$$
\int_{\tau_0}^{\infty}d_-(\tau)M_R(\tau)\,d\tau<\infty
$$

is known, the same conclusion requires either the time-slice lower bound of
Lemma PS11.3 or the synchronized window lower bound of Lemma PS11.3b.

**Proof.** The unweighted statement follows directly from the proof of Lemma
PS11.3: finite $\int d_-$ bounds $\log\Lambda$ below. If
$M_R\ge m_0$ a.e. on the terminal ray, then the weighted finite-cost condition
implies finite unweighted cost by division by $m_0$. In the window-retained
case there is no such pointwise division; the synchronized lower bound is the
extra hypothesis that allows the weighted cost to control the collapsing
windows.

**Lemma PS11.5 -- Ordered scale-collapse state-space alternatives.**
For a represented Type II branch with scale $\Lambda$ and active core record,
apply the following ordered Boolean partition.

1. **No genuine collapse:**

   $$
   \inf_{\tau\ge\tau_0}\Lambda(\tau)>0.
   $$

2. **Genuine collapse:**

   $$
   \liminf_{\tau\to\infty}\Lambda(\tau)=0.
   $$

   Inside this case, split into finite or infinite negative-drift cost:

   $$
   \int_{\tau_0}^{\infty}d_-(\tau)\,d\tau<\infty
   \quad\text{or}\quad
   \int_{\tau_0}^{\infty}d_-(\tau)\,d\tau=\infty.
   $$

3. Inside infinite cost, **thick windows** exist if there are $L,\kappa>0$ and
   pairwise disjoint intervals $I_j$ of length $L$ such that

   $$
   \int_{I_j}d_-(\tau)\,d\tau\ge\kappa.
   $$

   If no such $L,\kappa$ exist, the branch is **thin-drift**.

4. On thick windows, test coefficient convergence in the topology required by
   `PS6`, such as strong $L^2_{\rm loc}$ convergence, or
   $L^1_{\rm loc}$ convergence plus uniform integrability. Failure is an
   **autonomous-modulation defect**.

5. If coefficient convergence holds, test the full `PS6` compactness package.
   Failure is a **compactness defect**.

6. If coefficient convergence holds, compactness holds, and activity survives
   on the selected windows, then and only then pass a **nonzero autonomous
   reduced limit**.

**Proof.** The first two alternatives are the law of excluded middle for
$\inf_{\tau\ge\tau_0}\Lambda(\tau)>0$. In the collapse case, the extended
nonnegative number $\int d_-$ is finite or infinite. In the infinite case,
fixed-length intervals with uniformly positive negative-drift cost either
exist or do not; this gives thick windows or thin-drift. The remaining tests
are exactly the hypotheses needed by `PS6`: coefficient convergence, compact
upper bounds, and retained activity. A failed test records the corresponding
defect. Only the last case has all ingredients needed to extract a nonzero
autonomous reduced limit.

### Specific Estimate

The decisive estimate is conditional on time-slice retained mass:

$$
\Lambda(\tau_j)\to0,\quad
M_R(\tau)\ge m_0\text{ a.e. on a terminal ray}
\quad\Longrightarrow\quad
\int d_-(\tau)M_R(\tau)\,d\tau=\infty.
$$

Window-integrated activity can replace this only after the synchronized-drift
condition of Lemma PS11.3b has been proved.

### Practical Verification Steps

1. List all active parabolic frames $(x_n^j,t_n^j,\lambda_n^j)$.
2. Read the unresolved entries of $\mathfrak S_{\mathrm{II}}$ from
   $\mathcal R_{\mathrm{II}}(z_*,\{r_n\})$.
3. Verify a finite active-frame budget before classifying more than a finite
   subfamily.
4. Pass to subsequences and classify pairwise parabolic frame relations.
5. If there is a single retained represented scale, compute
   $\ell=\log\Lambda$ and $d=\ell'$.
6. Check whether retained core mass is time-slice retained or only
   window-retained.
7. Apply Lemma PS11.3 only under time-slice retained mass; use Lemma PS11.3b
   for window mass only if synchronized drift has been verified.
8. Classify the branch as finite-cost excluded, thin-drift, modulation defect,
   compactness defect, thick autonomous, multibubble/cascade, or single-core.

## Estimate Step $B_{\mathrm{PS11}}$

The estimate step is the combination of parabolic active-frame partition,
logarithmic scale identity, retained-mass drift estimate, and ordered
state-space stratification in Lemmas PS11.1--PS11.5.

## Failure Case

Failure name: unresolved Type II scale dynamics.

Analytic meaning: the local Type II branch has positive concentration, but its
parabolic frame behavior or retained-mass/drift synchronization has not been
reduced to a single-core, multibubble, cascade, finite-cost, rough-core, or
residual alternative.

## Refinement Step

Allowed refinements:

1. pass to subsequences of active parabolic frames;
2. prove a finite active-frame budget;
3. group comparable parabolic frames into compound profiles;
4. select innermost frames in a same-point cascade;
5. choose thick autonomous windows only when negative drift and activity are
   synchronized;
6. assign compactness defects back to `PS6` or to the defect-audit branch.

Progress measure: finite active-frame reduction, strict scale-ratio
subsequence selection, or explicit defect identification.

## Data Passed Forward

The data passed forward are

$$
\Gamma_{\mathrm{PS11}}
=
\Gamma_{\mathrm{PS10}}
\cup
\{\text{classified Type II scale alternative},
\text{ parabolic frame records},
\text{ retained-mass status},
\text{ discharged or routed entries of }\mathfrak S_{\mathrm{II}}\}.
$$

The next node depends on the selected alternative: radiation/escape goes to
`PS16`, rough-core loss to `PS17`, multibubble or separated active frames to
`PS18`, terminal decoupling to `PS20`, finite-cost transition behavior to
`PS28`, and residual scale defects to `PS34`.

---

# Type II Local Replacement Block `TII0`--`TII16`

This block is the Type II replacement for every route that would otherwise
ask for a whole-space critical norm, a global profile-mass budget, a global
pressure tail, or a terminal whole-space profile decomposition. It is entered
only from the local Type II packet of `PS10` and the scale-state classification
of `PS11`.

The organizing rule is

$$
\text{global Type II quantity}
\quad\leadsto\quad
\text{compact-window local check}
\quad\leadsto\quad
\text{local state-space alternative}.
$$

No node in this block assumes

$$
\sup_n\|u_n\|_{L^3(\mathbb R^3)}<\infty,
\qquad
\sum_j\|\phi^j\|_{L^3(\mathbb R^3)}^3<\infty,
\qquad
p=\mathcal R_i\mathcal R_j(u_i u_j)
$$

as an entry hypothesis. Such statements may appear only if a later endpoint
theorem requires them and a previous local node has actually proved them.

Each `TII` subnode follows the same verification format as the main nodes,
compressed to avoid duplicating hypotheses already stated in `PS10`--`PS11`:
analytic setting and unknowns, standing assumptions or local check, objects
inspected when they differ from the setting, proof or estimate, failure route,
refinement route when applicable, and data passed forward. A `TII` output is
never a proof object by itself; it is only valid with the displayed local
checks and the inherited `PS10`--`PS11` branch data.

## `TII0` -- Local Type II Entry Packet

### Analytic Setting and Unknowns

The unknowns are the selected terminal point $z_*=(x_*,T)$, shrinking
entry-admissible radii $r_n\downarrow0$, and the normalized pair

$$
u_n(y,s)=r_nu(x_*+r_ny,T+r_n^2s),
\qquad
p_n(y,s)=r_n^2p(x_*+r_ny,T+r_n^2s).
$$

### Local Check

The incoming record must contain

$$
C(u;z_*,r_n)+D(p;z_*,r_n)\ge\eta_0,
\qquad
C(u;z_*,r_n)\ge\eta_v,
$$

and the original-variable non-Type-I predicate

$$
\forall \rho>0,\quad
\operatorname*{ess\,sup}_{T-\rho^2<t<T}
\sqrt{T-t}\|u(t)\|_{L^\infty(B_\rho(x_*))}=\infty .
$$

The predicate remains in original variables; it is not inferred from one
normalized sequence.

### Output

The node passes

$$
\Gamma_{\mathrm{TII0}}
=
\{z_*,r_n,u_n,p_n,\ C+D\ge\eta_0,\ C\ge\eta_v,
\text{non-Type-I predicate}\}.
$$

## `TII1` -- Compact-Window Upper-Bound Package

### Analytic Setting and Unknowns

Fix a compact normalized window

$$
Q=B_R\times I\Subset\mathbb R^3\times(-\infty,0)
$$

and a slightly larger compact cylinder $Q^+$.

### Local Check

The Type II branch may use compactness on $Q$ only if it has proved

$$
A(u_n;Q^+)+E(u_n;Q^+)+C(u_n;Q^+)+D(p_n;Q^+)\le M(Q).
$$

Here the notation means the corresponding unscaled compact-cylinder
energy, enstrophy, cubic velocity, and mean-subtracted pressure quantities on
$Q^+$. These are local upper bounds and are independent of the lower
concentration inequalities.

### Alternatives

If the package holds, the branch may pass to local compactness and repaired
pressure gauges. If it fails, the branch routes to `TII8` as rough-core or
compact CKN failure. A lower CKN concentration bound is never counted as this
upper-bound package.

### Output

$$
\Gamma_{\mathrm{TII1}}
=
\Gamma_{\mathrm{TII0}}
\cup
\{\text{compact-window package on each selected }Q
\text{ or rough-core failure}\}.
$$

## `TII2` -- Local Active-Window Budget

### Analytic Setting and Unknowns

Fix a compact analysis window

$$
K\Subset B_{R_*}\times I_*
$$

and a slightly larger compact window

$$
K^+=B_{R^*}\times I^*,
\qquad
K\Subset K^+,
$$

where $B_{R_*}\Subset B_{R^*}$. Cover $K$ by finitely many bounded-overlap
parabolic cylinders

$$
\{Q_\rho(z^\ell)\}_{\ell=1}^{N(K,\rho)}.
$$

### Local Check

A cylinder is active at threshold $\eta$ if

$$
\rho^{-2}\iint_{Q_\rho(z^\ell)}|V_n|^3
+
\rho^{-2}\iint_{Q_\rho(z^\ell)}
|P_n-(P_n)_{B_\rho}|^{3/2}
\ge\eta.
$$

If the larger compact window has bounded CKN mass

$$
\iint_{K^+}|V_n|^3
+
\int_{I^*}\int_{B_{R^*}}
|P_n-(P_n)_{B_{R^*}}(\tau)|^{3/2}
\le M_K,
$$

then bounded overlap and the pressure mean comparison give

$$
N_{\rm active}(K,\eta)
\le
C(K,\rho)\frac{M_K}{\eta}.
$$

### Proof

Sum the active-cylinder inequalities over the selected subfamily. The velocity
terms are controlled by bounded overlap. For pressure, set
$q_n=P_n-(P_n)_{B_{R^*}}(\tau)$. On each selected ball,

$$
P_n-(P_n)_{B_\rho}=q_n-(q_n)_{B_\rho},
$$

and Jensen gives the local mean comparison. Summing over the bounded-overlap
cover yields the displayed count.

### Failure

If $M_K$ is not bounded, no global mass budget may be substituted. The branch
routes to `TII8`.

### Output

$$
\Gamma_{\mathrm{TII2}}
=
\Gamma_{\mathrm{TII1}}
\cup
\{\text{finite active cylinders on every compact analysis window}\}.
$$

## `TII3` -- Local Repaired Single-Core Gauge

### Local Check

On a retained active cylinder, choose local center-scale parameters by compactly
supported moment constraints

$$
\mathcal F(\lambda,c;V)=0.
$$

The repaired gauge is accepted only if the constraint map, topology, and
Jacobian are fixed and

$$
\det\partial_{(\lambda,c)}\mathcal F\ne0
$$

with a quantitative lower bound after shrinking the window. Then
$\lambda(\tau)$ and $x_c(\tau)$ are absolutely continuous, and

$$
V(y,\tau)=\lambda(\tau)
u(x_c(\tau)+\lambda(\tau)y,t(\tau)),
$$

$$
P(y,\tau)=\lambda(\tau)^2
p(x_c(\tau)+\lambda(\tau)y,t(\tau))+\pi(\tau).
$$

### Alternatives and Output

If the gauge is nondegenerate, pass the represented local variables and
coefficients forward. If the gauge is degenerate, route to `TII9` or `TII11`.

$$
\Gamma_{\mathrm{TII3}}
=
\Gamma_{\mathrm{TII2}}
\cup
\{\lambda,x_c,a,b,\text{ or gauge degeneracy}\}.
$$

## `TII4` -- Local Pressure Replacement

### Local Check

For compact balls $B_r\Subset B_R$, write

$$
P_n=P_{{\rm loc},n}+H_n,
$$

where

$$
-\Delta P_{{\rm loc},n}
=
\partial_i\partial_j(\zeta V_{n,i}V_{n,j}),
\qquad
\zeta\equiv1\text{ on }B_r.
$$

Calderon--Zygmund gives

$$
\|P_{{\rm loc},n}\|_{L^{3/2}(B_R)}
\lesssim
\|V_n\|_{L^3(B_R)}^2.
$$

The harmonic part is controlled on $B_r$ only by a larger-ball pressure
oscillation bound and harmonic interior estimates. Velocity alone does not
control $H_n$.

### Failure and Output

If the harmonic remainder is not controlled, route to `TII8` or the local
pressure-defect coordinate of `PS30`. Otherwise pass

$$
\Gamma_{\mathrm{TII4}}
=
\Gamma_{\mathrm{TII3}}
\cup
\{P=P_{\rm loc}+H,\text{ compact-ball pressure bounds}\}.
$$

## `TII5` -- Local Represented Type II Equation

### Local Check

In the repaired gauge, the represented equation must be derived
distributionally as

$$
\partial_\tau V+(V\cdot\nabla)V+\nabla P
=
\nu\Delta V
+a(\tau)(V+y\cdot\nabla V)
+b(\tau)\cdot\nabla V,
\qquad
\nabla\cdot V=0.
$$

The coefficient topology must be stated on the compact window: either
$a,b\in L^\infty(I)$, or the exact weaker topology required later by `PS6`
and the time-derivative estimates.

### Output

If coefficients are controlled,

$$
\Gamma_{\mathrm{TII5}}
=
\Gamma_{\mathrm{TII4}}
\cup
\{\text{represented local equation},a,b\}.
$$

If they are not controlled, route to `TII13`.

## `TII6` -- Compact-Window Compactness

### Local Check

On every compact window, verify

$$
V_n\text{ bounded in }L^\infty_tL^2_x\cap L^2_tH^1_x,
$$

$$
P_n-(P_n)_{B_R}\text{ bounded in }L^{3/2},
\qquad
\partial_\tau V_n\text{ bounded in }L^1H^{-m},\ m\ge3.
$$

Then `PS6` gives

$$
V_n\to V\quad\text{strongly in }L^3_{\rm loc},
\qquad
P_n-(P_n)_{B_R}\rightharpoonup P
\quad\text{weakly in }L^{3/2}_{\rm loc}.
$$

### Failure and Output

Failure routes to `TII8` or `TII9`. Success passes

$$
\Gamma_{\mathrm{TII6}}
=
\Gamma_{\mathrm{TII5}}
\cup
\{\text{compact-window limit}\}.
$$

## `TII7` -- Compact Single-Core Zero-Dissipation Test

### Local Check

Assume a single retained core has compact-window bounds, a repaired gauge,
bounded or otherwise admissible coefficients, and

$$
\int_{Q_\rho}|\nabla V_n|^2\to0.
$$

Then `PS21.3` gives spatial means $c_n(s)$ with

$$
V_n-c_n(s)\to0\quad\text{strongly in }L^3_{\rm loc}.
$$

If the local mean is removed or converges to zero, the retained velocity lower
bound is contradicted. If a nonzero spatial constant remains, the branch is a
regular background or modulation datum, not a Type II singular core. If only
combined $C+D$ mass remains, pressure convergence must also be proved; otherwise
route to the local pressure-defect coordinate.

### Output

$$
\Gamma_{\mathrm{TII7}}
=
\Gamma_{\mathrm{TII6}}
\cup
\{\text{single-core branch excluded or not a single singular core}\}.
$$

## `TII8` -- Rough-Core Caccioppoli Redirect

### Local Check

On compact cylinders, use

$$
\mathcal A_{J'}(R)+\mathcal H_{J'}(R)
\le
C\left(1+\mathcal C_J(2R)+\mathcal D_J(2R)\right).
$$

Thus loss of compact $H^1$ control forces compact CKN failure. A finite
parabolic cover then extracts a new local active core by the pressure mean
comparison used in `PS17.3`.

This redirect applies to the divergence-free represented velocity. If a cutoff
variable is used, the cutoff source and divergence defect must be kept in the
local energy inequality; otherwise the ordinary pressure-mean cancellation is
not available.

### Output

$$
\Gamma_{\mathrm{TII8}}
=
\Gamma_{\rm in}
\cup
\{\text{rough core returns to active CKN extraction}\}.
$$

## `TII9` -- Local Active-Frame Partition

### Local Check

Every active frame is parabolic,

$$
\mathfrak F_n^j=(x_n^j,t_n^j,\lambda_n^j).
$$

Pairs are classified as parabolically separated, comparable parabolic,
same-point scale cascade, or comparable scale with unbounded normalized time
shift exactly as in Lemma PS11.1. Large normalized time shift is routed to a
time-hull branch unless a separate decay or invisibility theorem applies.

### Output

$$
\Gamma_{\mathrm{TII9}}
=
\Gamma_{\mathrm{TII8}}
\cup
\{\text{separated / comparable / cascade / time-hull classification}\}.
$$

## `TII10` -- Local Radiation and Invisibility

### Local Check

A nonselected component may be discarded in a selected compact frame only if
it is locally invisible in $L^3$ and its mixed stress and pressure source are
invisible in the topology needed by `TII12`. If it carries a local CKN lower
bound after recentering, it becomes a new active frame. If velocity vanishes
but pressure or source control is missing, it is a defect, not radiation.

### Output

$$
\Gamma_{\mathrm{TII10}}
=
\Gamma_{\mathrm{TII9}}
\cup
\{\text{invisible / recentered / diffuse / pressure-source defect}\}.
$$

## `TII11` -- Same-Point Compound and Cascade Reduction

### Local Check

Comparable same-point active frames are grouped into one local compound core.
Same-point scale-separated frames are routed to scale-cascade analysis. A
compound object is only a local selected branch containing all non-negligible
same-scale components; it is not assumed to be a global Navier--Stokes
profile.

### Output

$$
\Gamma_{\mathrm{TII11}}
=
\Gamma_{\mathrm{TII10}}
\cup
\{\text{compound core or scale cascade}\}.
$$

## `TII12` -- Local Terminal Decoupling

### Local Check

For a selected core $U_n$ and local remainder $S_n$, prove on compact windows

$$
S_n\to0\quad\text{in }L^3_{\rm loc},
$$

$$
U_n\otimes S_n+S_n\otimes U_n+S_n\otimes S_n
\to0
\quad\text{in }L^{3/2}_{\rm loc}.
$$

For pressure, use compact-ball decomposition. The local Calderon--Zygmund
part is controlled by the local mixed stress. Any exterior source that is
invisible in the selected compact ball contributes only a harmonic pressure
there, whose oscillation is controlled by larger compact-ball pressure
oscillation. If the source does not become harmonic or invisible, it is a new
local observer branch.

The selected component $U_n$ is not assumed to be a suitable weak solution by
itself. Suitability of the selected limit is inherited from the full suitable
sequence after the remainder, mixed stresses, pressure contributions, and
cutoff commutators vanish in the stated compact-window topologies.

### Output

$$
\Gamma_{\mathrm{TII12}}
=
\Gamma_{\mathrm{TII11}}
\cup
\{\text{selected local equation decoupled through the full suitable sequence}\}.
$$

## `TII13` -- Local Scale-Collapse Stratification

### Local Check

For the repaired local scale $\lambda(\tau)$ set

$$
a(\tau)=-\partial_\tau\log\lambda(\tau).
$$

On every selected terminal window,

$$
\log\frac{\lambda(\tau_2)}{\lambda(\tau_1)}
=
-\int_{\tau_1}^{\tau_2}a(\tau)\,d\tau.
$$

The branch is assigned to exactly one of: finite scale-collapsing cost, finite
absolute scale cost, thin-drift defect, autonomous-modulation defect,
compactness defect, or nonzero autonomous reduced limit. Finite cost
contradicts genuine collapse with retained core; the other cases are local
state-space alternatives, not global failures.

### Output

$$
\Gamma_{\mathrm{TII13}}
=
\Gamma_{\rm in}
\cup
\{\text{scale-collapse alternative}\}.
$$

## `TII14` -- Local Scale-Rigid Terminal State

### Local Check

Scale-rigid states are tested only through localized weights such as

$$
G_{\nu,R}(y)=\chi_R(y)G_\nu(y).
$$

The localized virial identity has the form

$$
\frac{d}{d\tau}\int |V|^2G_{\nu,R}
=
-\text{coercive terms}
+\text{boundary/cutoff/pressure/modulation defects}.
$$

The node checks whether the defects are locally absorbable. If so, the
coercive term forces the declared local weighted activity to vanish and
contradicts retained activity. If a swirl or circulation quantity is used, its
definition, axis, and weighted identity must already be present in the branch
record. If defects are not absorbable, route to scale-rigidity or multibubble
obligations.

### Output

$$
\Gamma_{\mathrm{TII14}}
=
\Gamma_{\mathrm{TII13}}
\cup
\{\text{scale-rigid state eliminated or routed}\}.
$$

## `TII15` -- Local Type II State-Space Decomposition

### Statement

Every positive local Type II concentration sequence is assigned by the ordered
local tests to exactly one clean alternative or to an explicitly named
obligation. The clean alternatives are compact single-core zero-dissipation,
multibubble/cascade/gauge degeneracy, rough-core loss, a scale-collapse
alternative, or a scale-rigid terminal state. The named obligations include
pressure, source, compactness, modulation, cascade, decoupling, and terminal
defects.

### Proof

`TII3` handles nondegenerate single-core gauges. Gauge failure gives the
multibubble or degeneracy route. `TII8` handles rough-core loss. `TII13`
handles scale collapse. `TII14` handles scale-rigid terminal states. Whenever
one of the required local estimates is missing, the missing estimate is
entered into the Type II obligation ledger. The ordered tests are exhaustive
by construction and use only compact-window objects.

### Output

$$
\Gamma_{\mathrm{TII15}}
=
\Gamma_{\mathrm{TII0}}
\cup
\{\text{local Type II state-space decomposition and obligation ledger}\}.
$$

## `TII16` -- Local Type II Closure Ledger

### Statement

No positive local Type II concentration sequence is excluded at this node
unless the local state-space alternatives have all been discharged and the
Type II obligation ledger is empty.

### Proof Routing

The compact single-core branch is closed by `TII7`. Multibubble, cascade, and
gauge-degenerate branches are handled by `TII9`--`TII12` or routed as explicit
obligations if a local decoupling estimate is missing. Rough-core loss returns
to active-core extraction by `TII8`. Finite-cost collapse is excluded by
`TII13`. Scale-rigid terminal states are excluded or routed by `TII14`. Hence
only when the obligation ledger is empty,

$$
\text{positive local Type II concentration}\Rightarrow\bot .
$$

### Output

$$
\Gamma_{\mathrm{TII16}}
=
\Gamma_{\mathrm{TII15}}
\cup
\{\text{local Type II branch excluded if the ledger is empty, or exact local obligation listed}\}.
$$

This is the only admissible substitute for a global Type II endpoint row. If
any pressure, source, compactness, cascade, modulation, or terminal obligation
remains, the Type II branch is routed by that obligation and is not closed.

---

# 18. `PS12` -- Stationary Ancient-Profile Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a smooth bounded centered ancient profile $(V,\Pi)$. The
stationary branch means

$$
\partial_\tau V=0
\quad\text{in }\mathcal D'(\mathbb R^3\times\mathbb R).
$$

Since `PS9` supplies smoothness, this is equivalent, after redefining on a
null set, to $V(y,\tau)=W(y)$ pointwise.

### Standing Assumptions

The incoming hypotheses contain the normalized Type I ancient-profile package from `PS9` and the
nonvanishing estimate from `PS8`.

### Objects Inspected

Inspect whether $\partial_\tau V=0$ in distributions and whether
$W\in L^3(\mathbb R^3)$.

### Dependencies Used

The centered equation comes from `PS5` and `PS9`; nontriviality comes from
`PS8`; smoothness comes from `PS9`.

### Local Obstruction Predicate

$P_{\mathrm{PS12}}$ is the exact conjunction

$$
\left[\partial_\tau V=0\text{ in }\mathcal D'\right]
\wedge
\left[W\in L^3(\mathbb R^3)\right].
$$

Only this stationary whole-space $L^3$ branch is excluded here. Stationary
non-$L^3$ profiles, nonstationary compact dynamics, periodic-in-time profiles,
and general bounded centered ancient profiles are routed forward.

### Local Lemmas to Prove

**Lemma PS12.1 -- Stationary reduction.**
If $V(y,\tau)=W(y)$ in distributions, then there is a pressure representative
$\Pi_W(y)$, unique modulo an additive constant, such that $(W,\Pi_W)$
satisfies

$$
(W\cdot\nabla)W+\nabla\Pi_W
=\Delta W-\frac12(W+y\cdot\nabla W),
\qquad
\nabla\cdot W=0.
$$

**Proof.** Substituting $\partial_\tau V=0$ and $V=W$ into the centered
equation gives, on every ball $B_R$,

$$
\nabla \Pi_W^{(R)}
=\Delta W-\frac12(W+y\cdot\nabla W)-(W\cdot\nabla)W
$$

in distributions. The right-hand side is independent of $\tau$ and is smooth.
On overlaps $B_R\cap B_S$,

$$
\nabla(\Pi_W^{(R)}-\Pi_W^{(S)})=0,
$$

so the local representatives differ by constants. Choosing one additive
constant convention patches them into a global pressure representative
$\Pi_W$, unique modulo an additive constant. Equivalently, one may fix a time
$\tau_0$ and set $\Pi_W(y)=\Pi(y,\tau_0)$ in any compatible local gauge; for
a.e. $\tau$ the difference $\Pi(\cdot,\tau)-\Pi_W$ has zero spatial gradient
and is only a function of $\tau$. Subtracting that function from the pressure
gauge leaves the equation unchanged and gives the displayed stationary
self-similar equation.

**Lemma PS12.2 -- Stationary $L^3$ rigidity.**
If $W\in L^3(\mathbb R^3)$ is a smooth stationary self-similar solution of the
equation in Lemma PS12.1, then $W\equiv0$.

**Proof.** The endpoint rigidity theorem used here states:
a smooth solution $W\in L^3(\mathbb R^3)$ of

$$
(W\cdot\nabla)W+\nabla\Pi_W
=\Delta W-\frac12(W+y\cdot\nabla W),\qquad
\nabla\cdot W=0
$$

on $\mathbb R^3$ is identically zero. The hypotheses in the lemma are exactly
smoothness, the displayed stationary self-similar equation, divergence-free
condition, and whole-space $L^3$ integrability; hence the conclusion is
$W\equiv0$.

**Lemma PS12.3 -- Stationary branch contradiction.**
A nonzero normalized profile cannot satisfy the hypotheses of Lemma PS12.2.

**Proof.** Lemma PS12.2 gives $W\equiv0$. Then
$V(y,\tau)=W(y)$ vanishes on every compact cylinder, contradicting the
nonvanishing compact-cylinder lower bound from `PS8`. If the proof has
stationarity but not $W\in L^3(\mathbb R^3)$, this contradiction is not
available; the stationary non-$L^3$ case is routed to the later
structured/residual branches.

### Specific Estimate

The decisive condition is the stationary integrability hypothesis

$$
W\in L^3(\mathbb R^3).
$$

It is exactly the critical integrability required by stationary rigidity.

### Practical Verification Steps

1. Test whether $\partial_\tau V=0$.
2. Fix the pressure gauge so the stationary pressure gradient is represented.
3. Verify $W\in L^3(\mathbb R^3)$.
4. Apply stationary rigidity.
5. Compare the zero conclusion with the nonvanishing lower bound from `PS8`.

## Estimate Step $B_{\mathrm{PS12}}$

The estimate step is the stationary reduction plus the stationary $L^3$
rigidity theorem.

## Failure Case

Failure name: stationary-profile obstruction.

Analytic meaning: the branch appears stationary, but the exact rigidity
hypotheses have not been verified.

## Refinement Step

Allowed refinements:

1. pass to a time-translation limit to produce a stationary hull element;
2. prove or disprove $L^3$ integrability;
3. assign nonstationary compact dynamics to `PS13`;
4. assign stationary non-$L^3$ structured cases to `PS23`--`PS25`.

Progress measure: stationarity or integrability status is fixed.

## Data Passed Forward

The data passed forward are

$$
\Gamma_{\mathrm{PS12}}
=
\Gamma_{\rm in}
\cup
\{\text{stationary }L^3\text{ branch excluded},
\text{ or stationary non-}L^3\text{ branch routed},
\text{ or nonstationary branch routed to }PS13\}.
$$

---

---

# 19. `PS13` -- Compact Orbit or Compact Hull Branch

For `PS13`--`PS22`, every named record, status, witness, route, theorem match,
or obligation must be derived inside the proof from one of the following:
a previously constructed NS3D object, an estimate proved in the current node,
the negation of a stated estimate, a pressure or compactness representative
fixed by an earlier node, or an endpoint theorem whose hypotheses are matched
line by line. The words "record", "status", and "route" are bookkeeping for
those proved items only. They do not create an independent mathematical
object, and they cannot be used to close or reroute a branch before the stated
estimate or theorem-hypothesis match has been verified.

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

Work with bounded smooth centered solutions of the centered NS equation. The
unknown is the entire time-translation trajectory. This node distinguishes the
local smooth hull from the mild hull: local smooth compactness is a compact
PDE statement, while mildness is a global Duhamel statement and is not inferred
from local smooth convergence.

### Standing Assumptions

The incoming hypotheses contain the boundedness and smoothness from `PS9`.

### Objects Inspected

Inspect $\Theta_sV$ on compact cylinders, the local pressure oscillation
needed for interior estimates, the Duhamel/mildness status, and recurrence of
the compact activity witness.

### Dependencies Used

Boundedness and smoothness come from `PS9`; stationarity status comes from
`PS12`.

### Local Obstruction Predicate

$P_{\mathrm{PS13}}$ holds if the local compactness needed to form a compact
trajectory hull is missing.

### Local Lemmas to Prove

**Theorem PS13.A -- Local bounded-centered regularity input.**
Let $(V,\Pi)$ solve the centered equation on

$$
B_{2R}\times I^*
$$

with

$$
\|V\|_{L^\infty(B_{2R}\times I^*)}\le M
$$

and with local pressure oscillation bound

$$
\|\Pi-(\Pi)_{B_{2R}}(\tau)\|_{L^{3/2}(B_{2R}\times I^*)}\le P_R.
$$

Then, for every compact
$K\Subset B_{2R}\times I^*$ and every integer $m\ge0$,

$$
\|V\|_{C^m(K)}\le C(m,K,R,M,P_R).
$$

If a branch wants constants depending only on $m,K,R,M$, it must supply a
separate theorem or pressure-control mechanism eliminating the $P_R$
dependence.

**Proof.** This is the local parabolic regularity theorem used as an external
input in this node. The $L^\infty$ velocity bound controls the convection
coefficients locally, and the displayed pressure oscillation controls the
pressure force modulo functions of time. Interior Stokes and bootstrap
estimates then give the stated derivative bounds. The theorem is stated with
$P_R$ in the hypotheses to avoid hiding the harmonic pressure remainder.

**Lemma PS13.1 -- Local smooth compactness of bounded centered ancient
families.**
Let $\{(V_n,P_n)\}$ be bounded smooth ancient centered solutions on
$\mathbb R^3\times\mathbb R$ with

$$
\sup_n \|V_n\|_{L^\infty(\mathbb R^3\times\mathbb R)}\le M<\infty .
$$

Assume that on every compact cylinder $K\Subset\mathbb R^3\times\mathbb R$
there is a larger cylinder $K^*$ and a pressure oscillation bound

$$
\sup_n
\|P_n-(P_n)_{B_{2R}}(\tau)\|_{L^{3/2}(K^*)}<\infty
$$

for the corresponding ball. Then, after subtracting spatial pressure means on
balls, a subsequence converges locally smoothly to a bounded smooth centered
solution $(V,P)$ with $\|V\|_{L^\infty}\le M$.

**Proof.** Fix $K\Subset K^*$. The hypotheses and Theorem PS13.A give uniform
$C^m(K)$ bounds for $V_n$ for every $m$. Arzela--Ascoli gives a subsequence
converging in $C^m(K)$ for each fixed $m$. A diagonal extraction over a
countable compact exhaustion gives convergence in $C^\infty_{\rm loc}$.

The centered equation and divergence constraint pass to the limit by local
smooth convergence. Pressures are handled as local representatives modulo
functions of time. On overlaps, the recovered pressure gradients agree because
both equal

$$
-\partial_\tau V+\Delta V-\frac12y\cdot\nabla V-\frac12V
-(V\cdot\nabla)V.
$$

Thus the pressure representatives patch as a compatible local pressure class.
The weak-* lower semicontinuity of the $L^\infty$ norm gives
$\|V\|_{L^\infty}\le M$.

This lemma proves compactness of the smooth local hull only. It does not prove
that the limit is mild.

**Lemma PS13.1b -- Mildness passes only through a global Duhamel gate.**
Assume, in addition to Lemma PS13.1, that the physical pullbacks of $V_n$
satisfy the mild Duhamel formula on every finite terminal slab and that the
global Duhamel terms are controlled uniformly so that the heat and bilinear
terms pass to the limit, including their spatial tails. Then the limit in
Lemma PS13.1 is mild. Without this global Duhamel passage, the output is only a
smooth local hull element.

**Proof.** The mild formula is an integral equation on the whole spatial
domain:

$$
u_n(t)=e^{(t-s)\Delta}u_n(s)
-\int_s^t e^{(t-\sigma)\Delta}
\mathbb P\nabla\cdot(u_n\otimes u_n)(\sigma)\,d\sigma .
$$

Local smooth convergence passes the integrand on compact sets. The stated
global tail control is exactly what lets the heat term and bilinear term pass
on $\mathbb R^3$ and then along the ancient time interval. If that control is
absent, the local PDE convergence cannot be upgraded to a global mild
integral equation.

**Lemma PS13.2 -- Smooth and mild trajectory hulls.**
For a bounded smooth centered ancient solution $V$ satisfying the pressure
oscillation hypotheses of Lemma PS13.1 on every compact cylinder, the smooth
local hull

$$
\mathcal H_{\rm sm}(V)
=
\overline{\{\Theta_sV:s\in\mathbb R\}}^{C^\infty_{\rm loc}}
$$

is compact in the local smooth topology and invariant under all time
translations. A mild hull $\mathcal H_{\rm mild}(V)$ is defined only when the
global Duhamel gate of Lemma PS13.1b is verified for the orbit and its limits.

**Proof.** Let $\Theta_sV(\tau)=V(\tau+s)$. Every translate is again a bounded
smooth centered solution with the same $L^\infty$ bound and with the pressure
translated and re-gauged by spatial mean subtraction. For an arbitrary sequence
$s_n\in\mathbb R$, Lemma PS13.1 gives a subsequence, still denoted $s_n$, and
a bounded smooth centered solution $\widetilde V$ such that
$\Theta_{s_n}V\to\widetilde V$ in $C^\infty_{\rm loc}$. The metric

$$
d(f,g)=\sum_{m=1}^{\infty}2^{-m}
\frac{\|f-g\|_{C^m(\mathcal C_m)}}{1+\|f-g\|_{C^m(\mathcal C_m)}}
$$

on a compact exhaustion $\{\mathcal C_m\}$ induces the local smooth topology. If
$(W_n)\subset\mathcal H_{\rm sm}(V)$, choose orbit points $\Theta_{s_n}V$ with
$d(W_n,\Theta_{s_n}V)<1/n$. A subsequence of $\Theta_{s_n}V$ converges by the
first paragraph, and the corresponding $W_n$ converge to the same limit.
Thus the hull is sequentially compact; the displayed metric makes the topology
metrizable, so the hull is compact.

The identity $\Theta_t\Theta_sV=\Theta_{t+s}V$ maps the orbit onto itself.
If $W_n\to W$ locally smoothly, then $\Theta_tW_n\to\Theta_tW$ locally
smoothly because each compact time interval is shifted to another compact time
interval. Hence

$$
\Theta_t\mathcal H_{\rm sm}(V)=\mathcal H_{\rm sm}(V)
$$

for every $t\in\mathbb R$.

If the global Duhamel gate is part of the branch record and is stable under
the same time translations and local-smooth limits, Lemma PS13.1b applies to
each convergent orbit sequence. The subset of hull elements satisfying the
mild formulation is then compact and invariant. If the gate is absent, no
mildness conclusion is passed.

**Lemma PS13.3 -- Minimal compact invariant subset.**
Every nonempty compact invariant hull contains a nonempty compact minimal
invariant subset.

**Proof.** Partially order nonempty compact invariant subsets by inclusion.
The intersection of any decreasing chain is nonempty by compactness, compact,
and invariant. Zorn's lemma gives a minimal element.

**Lemma PS13.4 -- Invariant probability measures on compact hulls.**
Every nonempty compact invariant subset
$K\subset\mathcal H_{\rm sm}(V)$, and likewise every compact invariant subset
of $\mathcal H_{\rm mild}(V)$ when the mild hull exists, carries a
time-translation-invariant Borel probability measure.

**Proof.** Fix $W\in K$ and define the Krylov--Bogolyubov averages

$$
\mu_T=\frac1T\int_0^T\delta_{\Theta_sW}\,ds .
$$

The space $K$ is compact and metrizable, so the probability measures
$\{\mu_T:T>0\}$ are tight and have weak subsequential limits. Let
$\mu_{T_j}\rightharpoonup\mu$. For every continuous function
$F:K\to\mathbb R$ and every fixed $t\in\mathbb R$,

$$
\int_K F(\Theta_tZ)\,d\mu_T(Z)-\int_KF(Z)\,d\mu_T(Z)
=
\frac1T\int_T^{T+t}F(\Theta_sW)\,ds
-\frac1T\int_0^tF(\Theta_sW)\,ds ,
$$

with the intervals oriented in the same way when $t<0$. The absolute value is bounded by
$2|t|\|F\|_\infty/T$, which tends to zero along $T_j\to\infty$. Passing to the
weak limit gives

$$
\int_KF(\Theta_tZ)\,d\mu(Z)=\int_KF(Z)\,d\mu(Z),
$$

so $\mu$ is invariant under every time translation.

**Lemma PS13.5 -- Nonzero hull elements require activity recurrence.**
The compact activity witness from `PS8`,

$$
\iint_K|V|^3\ge\eta_*>0,
$$

does not by itself imply that every time-translation limit is nonzero. The
branch may pass a nonzero compact hull or a nonzero minimal component only if
it also proves activity recurrence: there are a compact set $K_0$, a number
$\eta>0$, and a relatively dense set of times $S\subset\mathbb R$ such that

$$
\iint_{K_0}|\Theta_sV|^3\ge\eta
\qquad (s\in S).
$$

If instead there are $s_j$ such that $\Theta_{s_j}V\to0$ locally smoothly, or
if all nonzero activity leaves every fixed compact set along some sequence,
the branch is activity-escaping and is routed to `PS14`, `PS16`, or `PS34`.

**Proof.** Local smooth convergence implies strong $L^3$ convergence on every
fixed compact cylinder. Therefore a sequence with the displayed recurrent
lower bound can only converge to a hull element carrying at least $\eta$ of
$L^3$ mass on $K_0$. Conversely, the original single-time compact witness is
not invariant under arbitrary time translation; time translates may move the
activity away from the fixed compact set. Thus nonzero minimal compact
invariant data require the recurrent lower bound, while loss of the lower
bound along time translations is an escaping-activity alternative rather than
a contradiction.

### Specific Estimate

The decisive local estimate is the pressure-aware interior parabolic bound

$$
\sup_n\|V_n\|_{L^\infty(Q_{2R})}\le M
\quad\Longrightarrow\quad
\sup_n\|V_n\|_{C^m(Q_R)}\le C_{m,R,M,P_R},
$$

where $P_R$ is the local pressure oscillation bound from Theorem PS13.A.

### Practical Verification Steps

1. Verify boundedness of the centered profile.
2. Apply local parabolic estimates to all time translates.
3. Use diagonal Arzela--Ascoli compactness.
4. Define the trajectory hull with the locally smooth metric.
5. Record whether the global Duhamel gate is verified; only then form the
   mild hull.
6. Verify activity recurrence before passing nonzero minimal compact
   invariant data.
7. Verify invariance and extract minimal compact invariant subsets only with
   the stated activity status.
8. If `PS29` needs statistical rigidity data, form invariant probability
   measures by Krylov--Bogolyubov averaging.

## Estimate Step $B_{\mathrm{PS13}}$

The estimate step is the pressure-aware local smooth compactness theorem, the
smooth-hull construction, the conditional mildness gate, and the
activity-recurrence test.

## Failure Case

Failure name: compact-hull construction failure.

Analytic meaning: compact trajectory dynamics is unavailable because the
time-translation orbit has not been proved compact in the local smooth
topology.

## Refinement Step

Allowed refinements:

1. restrict to bounded ancient profiles with pressure oscillation control;
2. shrink compact cylinders in the local smooth topology;
3. pass to time-translation subsequences;
4. prove the global Duhamel gate if endpoint mildness is needed;
5. prove activity recurrence before claiming a nonzero compact hull;
6. assign noncompact or escaping terminal behavior to `PS14` or
   radiation/escape nodes.

Progress measure: time-translation subsequence extraction or explicit
noncompactness classification.

## Data Passed Forward

The data passed forward are

$$
\Gamma_{\mathrm{PS13}}
=
\Gamma_{\mathrm{PS9}}
\cup
\{\mathcal H_{\rm sm}(V)\text{ compact and invariant},
\text{ mildness status},
\text{ activity-recurrence status},
\text{ minimal/invariant measure data only when their hypotheses hold}\}.
$$

---

# 20. `PS14` -- Terminal Residual Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a bounded centered ancient profile observed along terminal
time-translation and recentering sequences. Pure time translations preserve
the centered equation. Spatial recenterings do not: if

$$
V_n(Y,\tau)=V(Y+y_n,\tau+\tau_n),
$$

then the centered drift becomes

$$
\frac12(Y+y_n)\cdot\nabla V_n.
$$

Thus spatial recentering requires coefficient control. If $|y_n|\to\infty$,
the branch is routed to exterior/radiation analysis unless a further
renormalization supplies controlled coefficients.

### Standing Assumptions

The incoming record contains local smoothness, retained activity, and an
explicit terminal sequence.

### Objects Inspected

Inspect localized $L^3$ mass, pressure-normalized $D$ quantities, and local
energy bounds on recentered terminal cylinders.

### Dependencies Used

The profile comes from `PS9`; compactness and hull structure from `PS13`;
retained activity from `PS8`; pressure gauges from `PS4`.

### Local Obstruction Predicate

$P_{\mathrm{PS14}}$ holds if terminal behavior has not been reduced to one of
the explicit local terminal alternatives.

### Local Lemmas to Prove

**Lemma PS14.1 -- Terminal compact extraction with coefficient control.**
Let $(V_n,P_n)$ be a terminal sequence obtained either by pure time translation
or by spatial recentering. Suppose that, on every compact cylinder
$K\Subset\mathbb R^3\times\mathbb R$, the sequence satisfies the compactness
package of `PS6` or the smooth compactness package of `PS13`, and that the
transformed drift coefficients converge locally in a topology sufficient to
pass the equation.

For pure time translations,

$$
c_n(Y,\tau)=\frac12Y.
$$

For spatial recenterings,

$$
c_n(Y,\tau)=\frac12(Y+y_n).
$$

Therefore this lemma applies to spatial recenterings only if $y_n$ is bounded
after passing to a subsequence, or if an additional renormalization supplies a
controlled coefficient limit. Then a subsequence converges locally smoothly,
or at least in the `PS6` suitable topology, to a limit solving the
corresponding drifted equation.

**Proof.** Fix $K=B_R\times I$ and enlarge to
$K^*=B_{2R}\times I^*$ with $I\Subset I^*$. The compactness package gives
local velocity compactness, pressure oscillation bounds after subtracting
spatial means, and the local energy inequality or drifted local energy
inequality. The pressure gauge

$$
\int_{B_{2R}}P_n(y,\tau)\,dy=0
$$

removes time-dependent pressure constants. Pure time translations preserve the
centered equation exactly. A spatial recentering changes the drift from
$\frac12Y\cdot\nabla$ to $\frac12(Y+y_n)\cdot\nabla$; if $y_n\to y_\infty$,
the coefficients converge on compact sets and the drifted equation passes to
the limit. If $|y_n|\to\infty$, the coefficient is unbounded on fixed
$Y$-balls and this compact extraction is unavailable; the sequence is routed
to exterior/radiation analysis. Under the stated coefficient control, diagonal
compactness over compact cylinders gives a terminal profile, and every term in
the equation and local energy inequality is stable under the recorded
convergence.

**Lemma PS14.2 -- Maximal separated concentration recentering.**
Fix a retained threshold $\varepsilon_{\rm sd}>0$ and a compact terminal
window $\mathcal W$ on which the total pressure-normalized CKN mass is bounded
by $M_{\rm term}$. Assume the selected unit cylinders lie in a fixed spatial
ball $\mathcal B$ inside $\mathcal W$ and that their enlarged versions have
overlap bounded by $N_0$. For any terminal sequence, there is a finite maximal
separated family of $\varepsilon_{\rm sd}$-concentrating recentering
sequences, with

$$
N\le \frac{C N_0M_{\rm term}}{\varepsilon_{\rm sd}}.
$$

After excluding fixed parabolic neighborhoods of the selected concentrating
cylinders, every residual unit terminal cylinder outside those neighborhoods
has CKN mass below
$\varepsilon_{\rm sd}$, unless the residual sequence belongs to the diffuse
exterior alternative.

**Proof.** Select recentering cylinders greedily. Once $N$ separated unit
terminal cylinders each carry CKN mass at least $\varepsilon_{\rm sd}$, bounded
overlap of their enlarged cylinders and the velocity part of the window budget
control the velocity contribution to the count. For pressure, set

$$
q_n=P_n-(P_n)_{\mathcal B}(\tau).
$$

On each selected ball $B_1^j$,

$$
P_n-(P_n)_{B_1^j}(\tau)=q_n-(q_n)_{B_1^j}(\tau),
$$

and Jensen gives

$$
\int_{B_1^j}|P_n-(P_n)_{B_1^j}(\tau)|^{3/2}
\le
C\int_{B_1^j}|q_n|^{3/2}.
$$

Summing in space-time and using the bounded overlap gives

$$
N\varepsilon_{\rm sd}\le C N_0M_{\rm term}.
$$

Hence the greedy process stops after finitely many selections. The window is
fixed before this count: if a candidate cylinder leaves the chosen compact
terminal window, it is not charged to this finite budget and is instead tested
by the exterior-escape alternative. The resulting family is maximal by
construction. If, outside fixed parabolic neighborhoods of the selected
cylinders, another unit terminal cylinder carried CKN mass at least
$\varepsilon_{\rm sd}$ along a subsequence, it could be added without
violating separation, contradicting maximality. The only remaining way for
critical mass to persist without such a unit cylinder is diffuse exterior
concentration.

**Lemma PS14.3 -- Terminal exhaustion into the six residual alternatives.**
Every terminal concentration sequence belongs to exactly one of the following
ordered alternatives:

$$
\mathcal S_{\rm one},\quad
\mathcal S_{\rm locvan},\quad
\mathcal S_{\rm ext},\quad
\mathcal S_{\rm diff},\quad
\mathcal S_{\rm noncomp},\quad
\mathcal S_{\rm sep}.
$$

Here $\mathcal S_{\rm noncomp}$ is loss of local compactness;
$\mathcal S_{\rm one}$ is one bounded retained concentration profile;
$\mathcal S_{\rm sep}$ is a finite separated family of retained concentration
profiles; $\mathcal S_{\rm ext}$ is exterior escape of the concentrating
recentering; $\mathcal S_{\rm diff}$ is diffuse exterior concentration; and
$\mathcal S_{\rm locvan}$ is local CKN vanishing after the selected
concentrating cylinders have been geometrically excluded.

**Proof.** Start with the distinguished terminal concentration sequence. If the
local compactness, pressure gauge, or suitability package fails on a fixed
terminal cylinder, the sequence is in $\mathcal S_{\rm noncomp}$. Assume no
such loss occurs. If the sequence is $\varepsilon_{\rm sd}$-concentrating,
Lemma PS14.2 inserts it into the maximal separated concentrating family up to
equivalence. A bounded representative with exactly one bounded concentrating
class gives $\mathcal S_{\rm one}$. More than one bounded separated class gives
$\mathcal S_{\rm sep}$. If the concentrating representative leaves every
bounded terminal frame, it gives $\mathcal S_{\rm ext}$.

It remains to consider the geometric residual

$$
\mathcal W\setminus\bigcup_{j=1}^N\mathcal N_j,
$$

where $\mathcal N_j$ are fixed parabolic neighborhoods of the selected
concentrating cylinders. By maximality, this residual has local CKN mass below
$\varepsilon_{\rm sd}$ on each fixed unit terminal cylinder. If no critical
mass remains on expanding regions, this is $\mathcal S_{\rm locvan}$. If
critical mass persists only on expanding regions while every fixed unit
cylinder stays below threshold, this is $\mathcal S_{\rm diff}$. The cases are
mutually exclusive because the tests are applied in the displayed order: first
compactness, then bounded concentration, then escaping concentration, then
locally vanishing versus diffuse exterior mass. They are exhaustive because
every terminal sequence either has a compactness failure, has an
above-threshold unit cylinder after recentering, or has no such unit cylinder.

**Lemma PS14.4 -- Nonactive alternatives cannot close the retained compact
activity branch.**
Assume the incoming branch contains a fixed compact retained velocity lower
bound

$$
\liminf_n\iint_K |V_n|^3\,dy\,d\tau\ge \eta_K>0
$$

on a compact terminal cylinder $K$. Then local vanishing, exterior escape, and
diffuse exterior concentration cannot be the sole explanation of this fixed
compact lower bound in the original compact frame. Local compactness failure
is not excluded here; it is routed to `PS17`.

**Proof.** Cover $K$ by finitely many unit terminal cylinders. One cylinder
carries velocity mass at least a positive fraction of $\eta_K$; choosing
$\varepsilon_{\rm sd}$ below the corresponding finite-cover fraction makes
local vanishing impossible on all cylinders covering $K$. Exterior escape
describes mass whose centers leave every bounded terminal frame, so it cannot
be the sole source of a lower bound on the fixed compact cylinder $K$ in the
original frame. Diffuse exterior concentration has every fixed unit cylinder
below threshold while mass persists only on expanding exterior regions; this
contradicts the above-threshold cylinder found in the finite cover of $K$.
Local compactness failure is different: it can occur and is not resolved by
this lemma. The branch records the exact missing compactness item and routes
to `PS17`.

**Lemma PS14.5 -- Terminal routing preserves the next-node hypotheses.**
Each terminal alternative produced by Lemma PS14.3 supplies the exact input
data required by its routed node: a single compact active profile for the
single-core route, a separated finite family for `PS18`--`PS20`, an escaping
frame with a CKN lower bound for `PS16`, a compactness-loss record for `PS17`,
or an explicit residual obligation for `PS34`.

**Proof.** In the single-core alternative, Lemma PS14.1 gives a terminal
profile with local suitability, pressure gauge, and retained compact CKN mass,
so the branch is a valid single active input. In the separated alternative,
Lemma PS14.2 gives finitely many separated $\varepsilon_{\rm sd}$-active
terminal cylinders inside the fixed terminal window; their pairwise parabolic
separation and pressure-normalized local CKN lower bounds are precisely the
frame data inspected by `PS18` and the packet nodes. In the exterior-escape
alternative, the concentration center leaves every bounded terminal frame, so
`PS16` receives both the escaping geometry and the displayed scale-invariant
lower bound needed either to prove invisibility or to recenter. In the
noncompact alternative, the missing item is one of the local compactness,
pressure, or suitability estimates on a fixed compact terminal cylinder; that
is exactly the rough-core or compact-cylinder failure audited by `PS17`. In
the locally vanishing and diffuse alternatives, Lemma PS14.4 either discharges
the alternative as the sole carrier of retained compact mass or leaves a
residual exterior/diffuse obligation whose defining property is the absence of
above-threshold fixed unit cylinders. That is the residual input required by
`PS34`. No route changes the pressure convention: every target node receives
the same spatial-mean pressure normalization recorded in this node.

### Specific Estimate

The decisive local estimate is the concentration threshold selection:

$$
\iint_{Q_1(y_n,\tau_n)}
\left(
|V_n|^3+
|P_n-(P_n)_{B_1(y_n)}(\tau)|^{3/2}
\right)\,dy\,d\tau
\ge\varepsilon_{\rm sd}
$$

for a recentered terminal cylinder, or else the retained compact activity is
not carried by that scale/location and the sequence is assigned to
$\mathcal S_{\rm locvan}$ or $\mathcal S_{\rm diff}$.

### Practical Verification Steps

1. Choose the terminal time or recentering sequence.
2. Test local compactness, pressure gauges, and suitability on fixed terminal
   cylinders.
3. Search for unit-scale CKN mass above $\varepsilon_{\rm sd}$.
4. Build a maximal separated family of retained recenterings.
5. Decide exactly one of
   $\mathcal S_{\rm one}$, $\mathcal S_{\rm locvan}$,
   $\mathcal S_{\rm ext}$, $\mathcal S_{\rm diff}$,
   $\mathcal S_{\rm noncomp}$, or $\mathcal S_{\rm sep}$.
6. Use Lemma PS14.4 to discharge nonactive alternatives as sole carriers of
   the retained compact mass.
7. Verify by Lemma PS14.5 that the selected route carries the pressure gauge,
   compactness status, and retained lower bound required by the next node.

## Estimate Step $B_{\mathrm{PS14}}$

The estimate step is terminal compact extraction, maximal concentration
recentering, six-way terminal exhaustion, nonactive-discharge verification, and
route-admissibility verification in Lemmas PS14.1--PS14.5.

## Failure Case

Failure name: terminal residual classification failure.

Analytic meaning: retained activity persists along terminal sequences but has
not been assigned to exactly one of the six terminal alternatives, or a
nonactive alternative is still being treated as the sole carrier of compact
retained mass.

## Refinement Step

Allowed refinements:

1. pass to terminal subsequences;
2. recenter at concentration packets;
3. extract terminal profiles;
4. assign compactness failure to `PS17`;
5. assign separated families to `PS18`--`PS20`.

Progress measure: terminal subsequence extraction or finite packet selection
above a fixed threshold.

The routing is part of the conclusion. The ordered alternatives are sent as
follows:

$$
\mathcal S_{\rm one}\to\text{single active branch},
\qquad
\mathcal S_{\rm sep}\to PS18\text{--}PS20,
$$

$$
\mathcal S_{\rm ext}\to PS16,
\qquad
\mathcal S_{\rm noncomp}\to PS17,
$$

and

$$
\mathcal S_{\rm locvan},\mathcal S_{\rm diff}
\to PS34
$$

unless Lemma PS14.4 has already discharged them as sole carriers of the
retained compact mass. This prevents the terminal node from merely naming an
alternative without recording the next proof obligation.

## Data Passed Forward

The data passed forward are

$$
\Gamma_{\mathrm{PS14}}
=
\Gamma_{\mathrm{PS13}}
\cup
\{\mathcal S_{\rm one}\text{ or }\mathcal S_{\rm sep}\text{ or }
\mathcal S_{\rm ext}\text{ or }\mathcal S_{\rm diff}\text{ or }
\mathcal S_{\rm locvan}\text{ or }\mathcal S_{\rm noncomp},
\text{ with required witnesses}\}.
$$

Here $\mathcal S_{\rm one}$ passes a compact active profile;
$\mathcal S_{\rm sep}$ passes finitely many separated active cylinders and the
finite budget; $\mathcal S_{\rm ext}$ passes escaping centers and a retained
lower bound in that escaping frame; $\mathcal S_{\rm noncomp}$ passes the
exact missing compactness item; and $\mathcal S_{\rm diff}$ passes the diffuse
tail witness with the below-threshold fixed-cylinder condition.

---

# 21. `PS15` -- Uniform Tightness Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a bounded centered ancient profile $V(y,\tau)$.

### Standing Assumptions

The incoming record contains normalized nonvanishing from `PS8` and the Type I profile package from
`PS9`.

### Objects Inspected

Inspect the $L^3$ tail integral outside $B_R$ uniformly for all $\tau$ when
the branch claims tightness. If tightness fails, inspect the escaping
time-space witnesses and their covariant observer cylinders. The node does
not attempt to estimate the whole exterior tail after failure; it records the
local state-space data required by `ST1`--`ST20`.

### Dependencies Used

The profile comes from `PS9`; nonvanishing from `PS8`; compact hull/tail
limits, when used, come from `PS13`.

### Local Obstruction Predicate

$P_{\mathrm{PS15}}$ holds when uniform $L^3$ tightness is present and the
finite-shift mildness gate is verified, because that branch is excluded by
the endpoint ancient $L^3$ theorem. If uniform tightness is absent,
`PS15` does not become a demand for a global tail estimate. It produces an
escaping local observer witness and routes the branch to the local residual
state-space closure `ST0`--`ST20`.

The external endpoint input used in this node is the following precise
Albritton--Barker theorem input:

$$
\begin{gathered}
u\text{ is a bounded mild ancient Navier--Stokes solution on }
\mathbb R^3\times(-\infty,0),\\
\exists\,s_k\downarrow-\infty
\quad
\sup_k\|u(s_k)\|_{L^3(\mathbb R^3)}<\infty
\end{gathered}
\quad\Longrightarrow\quad
u\equiv0 .
$$

Thus `PS15` must verify the whole-space domain, mild Duhamel formulation,
boundedness on each shifted terminal slab, and the backward sequence of
bounded $L^3$ norms. Local smoothness or compact-cylinder $L^3$ control alone
does not trigger the theorem.

The theorem is applied only after fixing a finite physical terminal time
$T<0$ and shifting the interval $(-\infty,T]$ to terminal time $0$. The node
does not require the self-similar pullback to be bounded up to the original
time $0$; it requires boundedness on each shifted ancient slab
$\mathbb R^3\times(-\infty,0]$ obtained from a fixed $T<0$. This distinction
is essential because the factor $(-t)^{-1/2}$ in the physical pullback may
blow up as $t\uparrow0$ even when $V$ is bounded.

### Local Lemmas to Prove

**Mildness gate for endpoint use.**
The Albritton--Barker endpoint theorem is applied only if the physical
pullback

$$
u(x,t)=(-t)^{-1/2}
V\left(\frac{x}{\sqrt{-t}},-\log(-t)\right)
$$

is known to be a mild ancient Navier--Stokes solution on every finite terminal
slab $\mathbb R^3\times(-\infty,T]$, $T<0$. Equivalently, after shifting
$T$ to $0$, the function

$$
u_T(x,s)=u(x,s+T),
\qquad s<0,
$$

must satisfy the Duhamel formula

$$
u_T(t)=e^{(t-s)\Delta}u_T(s)
-\int_s^t
e^{(t-\sigma)\Delta}\mathbb P\nabla\cdot
(u_T\otimes u_T)(\sigma)\,d\sigma
$$

for all $s<t<0$ in the topology required by the endpoint theorem. If this
gate is not verified, `PS15` records a mildness obstruction rather than an
endpoint contradiction.

**Lemma PS15.1 -- Closure of uniform tightness.**
If $V_n\to V$ locally smoothly, each $V_n$ satisfies

$$
\sup_{\tau}\int_{|y|>R}|V_n(y,\tau)|^3\,dy\le\omega(R),
\qquad
\omega(R)\to0,
$$

and the profiles have a common $L^\infty_\tau L^3_y$ bound

$$
\sup_n\sup_\tau\|V_n(\tau)\|_{L^3(\mathbb R^3)}\le L,
$$

then $V$ satisfies the same global $L^3$ bound and the same tightness estimate.

**Proof.** Fix $\tau$. For every $R_1<\infty$, local smooth convergence gives

$$
\int_{B_{R_1}}|V(y,\tau)|^3\,dy
=
\lim_{n\to\infty}\int_{B_{R_1}}|V_n(y,\tau)|^3\,dy
\le L^3 .
$$

Letting $R_1\to\infty$ and using monotone convergence gives
$\|V(\tau)\|_{L^3}\le L$, uniformly in $\tau$.

For the tail, fix $1\le R<R_1$. On the annulus
$A_{R,R_1}=\{R<|y|<R_1\}$,

$$
\int_{A_{R,R_1}}|V(y,\tau)|^3\,dy
=
\lim_{n\to\infty}
\int_{A_{R,R_1}}|V_n(y,\tau)|^3\,dy
\le \omega(R).
$$

Monotone convergence as $R_1\to\infty$ gives

$$
\int_{|y|>R}|V(y,\tau)|^3\,dy\le\omega(R).
$$

The bound is independent of $\tau$, so taking the supremum in time proves the
claim.

**Lemma PS15.2 -- Tightness gives the endpoint $L^3$ sequence, conditional on
mildness.**
Assume that $V$ is bounded, uniformly $L^3$-tight, and that the physical
pullback satisfies the mildness gate on every finite terminal slab. Then for
every $T<0$, the shifted solution

$$
u_T(x,s)=u(x,s+T)
$$

is a bounded mild ancient solution on $\mathbb R^3\times(-\infty,0]$, and
there is a sequence $s_k\downarrow-\infty$ such that

$$
\sup_k\|u_T(s_k)\|_{L^3(\mathbb R^3)}<\infty.
$$

**Proof.** Let $M=\|V\|_{L^\infty(\mathbb R^3\times\mathbb R)}$. Uniform
tightness gives $R_0$ such that

$$
\sup_{\tau}\int_{|y|>R_0}|V(y,\tau)|^3\,dy\le1.
$$

Therefore, for every $\tau$,

$$
\int_{\mathbb R^3}|V(y,\tau)|^3\,dy
\le M^3|B_{R_0}|+1.
$$

Choose $\tau_k=-k$ and $t_k=-e^k$. Then $t_k\downarrow-\infty$ and
$\tau_k=-\log(-t_k)$. Critical $L^3$ invariance gives

$$
\|u(t_k)\|_{L^3(\mathbb R^3)}
=
\|V(\tau_k)\|_{L^3(\mathbb R^3)}
\le
\left(M^3|B_{R_0}|+1\right)^{1/3}.
$$

For fixed $T<0$, set $s_k=t_k-T$. Then $s_k\downarrow-\infty$ and

$$
\|u_T(s_k)\|_{L^3}
=
\|u(t_k)\|_{L^3}
$$

is uniformly bounded. Also, for $s<0$, $s+T<T<0$, so

$$
|u_T(x,s)|\le (-T)^{-1/2}M.
$$

The mildness gate supplies the Duhamel formula. Thus the endpoint theorem's
bounded mild ancient solution and backward $L^3$ sequence hypotheses are
verified.

**Lemma PS15.3 -- Uniformly tight nonzero ancient profiles are excluded.**
A bounded centered ancient profile satisfying uniform $L^3$ tightness and the
nonvanishing normalization cannot occur.

**Proof.** By Lemma PS15.2, after restricting the physical pullback to any
finite terminal slab and translating that slab to terminal time $0$, the
Albritton--Barker theorem applies to a bounded mild ancient solution with a
sequence $s_k\downarrow-\infty$ satisfying

$$
\sup_k\|u_T(s_k)\|_{L^3(\mathbb R^3)}<\infty .
$$

Therefore $u_T\equiv0$ on each translated slab. Since $T<0$ was arbitrary,
$u\equiv0$ for all $t<0$, and the self-similar change of variables gives
$V\equiv0$. This contradicts the retained local nonvanishing normalization
from `PS8`.

If the mildness gate in Lemma PS15.2 is missing, the endpoint theorem cannot
be applied; the branch records a mildness obstruction for the endpoint
$L^3$ theorem instead of a contradiction.

**Lemma PS15.4 -- Failure of tightness produces a recorded exterior witness.**
If uniform $L^3$ tightness fails, then there are
$\eta_{\rm tail}>0$, radii $R_k\to\infty$, and centered times $\tau_k$ such
that

$$
\int_{|y|>R_k}|V(y,\tau_k)|^3\,dy\ge \eta_{\rm tail}.
$$

After passing to a subsequence, this witness is converted into local observer
data. One may still route a verified active exterior core to `PS16`, or a
verified finite separated family to `PS18`--`PS20`, but the generic
non-tight branch is sent to `ST0`--`ST20`. This lemma itself produces only
the exterior witness displayed above and the obligation to run those local
state-space tests.

**Proof.** The negation of uniform tightness is the existence of
$\eta_{\rm tail}>0$ such that for every $R$ there is a time $\tau_R$ with

$$
\int_{|y|>R}|V(y,\tau_R)|^3\,dy\ge \eta_{\rm tail}.
$$

Choosing $R_k\uparrow\infty$ and setting $\tau_k=\tau_{R_k}$ gives the
displayed witness. No localization conclusion follows from this tail inequality
alone. Each possible route requires an additional check written in the target
node. If a sequence of compact terminal cylinders inside the exterior region
has a scale-invariant CKN lower bound, then covariant recentering at those
cylinders gives the active exterior input for `PS16` and the active-locus
input for `ST5`. If the above-threshold cylinders can be chosen as a finite
parabolically separated family within a compact terminal window with a
recorded CKN budget, they give the finite packet input for `PS18`--`PS20` and
the finite-family input for `ST15`. If the exterior lower bound persists
through expanding regions while every fixed compact observer cylinder is below
the retained threshold, then the branch satisfies the diffuse-defect
hypothesis used by `ST8`--`ST11`. If none of these verification conditions has
been proved, the conclusion of `PS15` is only the displayed exterior witness
together with a missing local route-data item; it is not a completed
classification and it is not a global estimate obligation.

### Local Replacement Subnodes `PS15a`--`PS15n`

The non-tight side of `PS15` is implemented by the following local subnodes.
They are the explicit front end of the `ST0`--`ST20` state-space closure and
do not assume a global tail estimate. Each subnode is written in the same
compressed verification format used for local branches elsewhere in the file:
input, local check or dichotomy, output, and an implicit failure route. The
failure route is always the named missing item in the output line, such as a
local compactness defect, pressure defect, recurrence obligation,
finite-family obligation, critical-tail obligation, or mildness gap. No
subnode closes a branch merely by naming a state.

The last three subnodes are not ordinary local-estimate checks. `PS15l` is an
endpoint-sequence assembly ledger: it derives the backward whole-space
$L^3$ sequence only after every local exterior-core and diffuse-tail
alternative has been closed. `PS15m` is a Duhamel/mildness residual ledger: it
checks the integral formulation by eliminating local near-field, far-field,
pressure, gauge, and parasitic residuals. `PS15n` is only the imported
Albritton--Barker endpoint theorem application after `PS15l` and `PS15m` have
already supplied the exact hypotheses.

**`PS15a` -- Exterior witness extraction.**
Input: a bounded centered ancient profile with retained compact activity. If
tightness fails, Lemma PS15.4 gives

$$
\eta_{\rm tail}>0,\qquad R_k\to\infty,\qquad \tau_k,
\qquad
\int_{|y|>R_k}|V(y,\tau_k)|^3\,dy\ge\eta_{\rm tail}.
$$

Output:

$$
\Gamma_{\mathrm{PS15a}}
=
\Gamma_{\rm in}\cup\{\eta_{\rm tail},R_k,\tau_k\}.
$$

**`PS15b` -- Active exterior core versus diffuse tail.**
Set

$$
m_k=\sup_{|x|>R_k}\int_{B_1(x)}|V(y,\tau_k)|^3\,dy.
$$

If $\limsup_k m_k>0$, pass to a subsequence and choose
$\varepsilon_*>0$ and $x_k$ with $|x_k|>R_k$ such that

$$
\int_{B_1(x_k)}|V(y,\tau_k)|^3\,dy\ge\varepsilon_*.
$$

This is an active exterior core. If $m_k\to0$ while the exterior lower bound
persists, the branch is a diffuse exterior tail. Output:

$$
\Gamma_{\mathrm{PS15b}}
=
\Gamma_{\mathrm{PS15a}}
\cup
\{\text{active exterior core or diffuse exterior tail}\}.
$$

**`PS15c` -- Covariant exterior recentering.**
For an active exterior core, use the covariant observer recentering

$$
V_k^{\rm ext}(y,s)
=
V(y+e^{s/2}x_k,\tau_k+s),
$$

not a raw spatial translation. On every compact observer cylinder, verify
boundedness, local pressure gauges, local suitability, and retention of the
active unit cylinder in a fixed compact observer window. The transformed
equation must also be derived with every drift, cutoff, and pressure-gauge
term recorded. If the recentering produces an uncontrolled drift or a
coefficient outside the compactness topology, the output is an observer or
modulation defect, not a compact exterior profile. Output:

$$
\Gamma_{\mathrm{PS15c}}
=
\Gamma_{\mathrm{PS15b}}
\cup
\{V_k^{\rm ext},\text{ compact-window active core}\}.
$$

**`PS15d` -- Compact-window extraction of the exterior core.**
For $V_k^{\rm ext}$, test the same compact-window package as `PS6`: local
energy, enstrophy, cubic velocity, pressure oscillation, and time-derivative
control on $Q^+\supset Q$. If the package holds, extract a retained exterior
profile. If it fails, route to rough-core or local pressure defect extraction.
Output:

$$
\Gamma_{\mathrm{PS15d}}
=
\Gamma_{\mathrm{PS15c}}
\cup
\{\text{compact exterior profile or local defect}\}.
$$

**`PS15e` -- Diffuse exterior defect compactification.**
For the diffuse alternative, normalize the escaping exterior density on a
finite positive exterior region $E_k$:

$$
\lambda_k
=
\frac{|V(y,\tau_k)|^3\,dy\!\restriction E_k}
{\int_{E_k}|V(y,\tau_k)|^3\,dy}.
$$

Pass to a weak limit in the observer compactification. Output:

$$
\Gamma_{\mathrm{PS15e}}
=
\Gamma_{\mathrm{PS15b}}
\cup
\{\lambda,\text{ diffuse exterior state}\}.
$$

**`PS15f` -- Diffuse tail trichotomy.**
The diffuse state is assigned to exactly one of: regenerated local activity,
affine/parasitic lower stratum, or critical diffuse tail. The first returns to
active-frame extraction, the second exits through the lower-strata ledger, and
the third enters critical-tail compactification. Output:

$$
\Gamma_{\mathrm{PS15f}}
=
\Gamma_{\mathrm{PS15e}}
\cup
\{\text{regenerated activity / lower stratum / critical diffuse tail}\}.
$$

**`PS15g` -- Critical-tail compactification and rigidity.**
A critical diffuse tail is compactified using only local boundedness, local
pressure gauges, observer-space compactification, and local weak compactness.
It may close only through a verified local rigidity or activity-regeneration
implication; otherwise it records a critical-tail rigidity obligation. Output:

$$
\Gamma_{\mathrm{PS15g}}
=
\Gamma_{\mathrm{PS15f}}
\cup
\{\text{critical tail closed, activity regenerated, or rigidity obligation}\}.
$$

**`PS15h` -- Active successor relation.**
Define $U\mathcal R_\eta W$ only when $W$ is a retained active covariant tail
descendant of $U$ in the local state-space topology. Escaping witnesses that
are not compact in that topology return to `PS15e`--`PS15g`. Output:

$$
\Gamma_{\mathrm{PS15h}}
=
\Gamma_{\rm in}
\cup
\{\mathcal R_\eta\text{ on retained local active states}\}.
$$

**`PS15i` -- No infinite active descendant chain.**
An infinite chain for $\mathcal R_\eta$ gives a compact path space and a
shift-invariant probability measure. The branch closes only if the
recurrent-core rigidity implication forces the recurrent core into a lower
closed stratum. Otherwise it records a recurrent-core obligation. Output:

$$
\Gamma_{\mathrm{PS15i}}
=
\Gamma_{\mathrm{PS15h}}
\cup
\{\text{no infinite active chain or recurrent-core obligation}\}.
$$

**`PS15j` -- No finite separated retained family.**
Finite retained descendants are classified by parabolic frame geometry:
separated, comparable, or same-point cascade. The branch closes only if the
no-separated-family implication is verified; otherwise it records a
finite-family obligation. Output:

$$
\Gamma_{\mathrm{PS15j}}
=
\Gamma_{\mathrm{PS15i}}
\cup
\{\text{no finite separated retained family or finite-family obligation}\}.
$$

**`PS15k` -- Terminal indecomposability.**
If there is no active exterior core, no diffuse exterior tail, no infinite
active successor chain, no finite separated retained family, and no local
compactness or pressure defect, the retained profile is terminally
indecomposable. If any one of those routes is still open, terminal
indecomposability is not available and the branch must follow the named
obligation. Output:

$$
\Gamma_{\mathrm{PS15k}}
=
\Gamma_{\mathrm{PS15j}}
\cup
\{\text{terminally indecomposable retained profile}\}.
$$

**`PS15l` -- Endpoint sequence ledger from indecomposability.**
This node does not ask for the backward $L^3$ sequence as a primitive
whole-space estimate. It is an assembly ledger. The only admissible proof is:
assume the endpoint sequence is absent, convert that absence into exterior
mass on larger and larger shells, split that exterior mass into active local
cores or diffuse local tails, and use the already closed local alternatives
`PS15b`--`PS15k` to rule out both.

Concretely, if no sequence $\tau_k\to-\infty$ satisfies

$$
\sup_k\|U(\cdot,\tau_k)\|_{L^3(\mathbb R^3)}<\infty,
$$

then the $L^3$ norm is unbounded along every sufficiently far backward tail.
Since the branch entering `PS15l` is a bounded centered ancient profile, for a
chosen exhaustion $R_k\to\infty$ the interior contribution obeys

$$
\int_{|y|\le R_k}|U(y,\tau)|^3\,dy
\le
\|U\|_{L^\infty}^3 |B_{R_k}|.
$$

Choose $\tau_k\to-\infty$ so that

$$
\|U(\cdot,\tau_k)\|_{L^3(\mathbb R^3)}^3
>
\|U\|_{L^\infty}^3 |B_{R_k}|+1.
$$

Then

$$
\int_{|y|>R_k}|U(y,\tau_k)|^3\,dy\ge1.
$$

The unit-ball supremum of this exterior mass either produces an active
exterior core or tends to zero and produces a diffuse exterior tail, both
contradicting terminal indecomposability. Therefore

$$
\exists\,\tau_k\to-\infty
\quad
\sup_k\|U(\cdot,\tau_k)\|_{L^3(\mathbb R^3)}<\infty.
$$

If any part of the local ledger is still open, this node does not output the
sequence. The output is instead the precise open obligation: active exterior
core, diffuse tail, recurrence, finite separated family, pressure defect,
compactness defect, or terminal indecomposability gap.

Output:

$$
\Gamma_{\mathrm{PS15l}}
=
\Gamma_{\mathrm{PS15k}}
\cup
\{\tau_k\to-\infty,\ \sup_k\|U(\tau_k)\|_{L^3}<\infty
\text{ or exact endpoint-sequence obligation}\}.
$$

**`PS15m` -- Duhamel/mildness residual ledger.**
For

$$
u(x,t)=(-t)^{-1/2}
U\left(\frac{x}{\sqrt{-t}},-\log(-t)\right)
$$

and each fixed $T<0$, $u^T(x,s)=u(x,T+s)$ may be sent to the endpoint theorem
only if it is a bounded mild ancient solution and satisfies the Duhamel
formula in the endpoint topology. Bounded smoothness and local compactness do
not imply this gate.

Thus `PS15m` must verify the Duhamel residual by a ledger rather than by a
single global assertion. For every finite terminal slab and all admissible
times $s<t<0$, write the residual

$$
\mathcal D_{s,t}
=
u^T(t)-e^{(t-s)\Delta}u^T(s)
+
\int_s^t e^{(t-\sigma)\Delta}
\mathbb P\nabla\cdot(u^T\otimes u^T)(\sigma)\,d\sigma .
$$

The branch must prove that $\mathcal D_{s,t}=0$ in the endpoint topology by
checking, separately, the near-field nonlinear term, far-field shell terms,
pressure-projection compatibility, gauge terms, commutators, parasitic or
affine lower-stratum components, and the limit as the ancient lower time is
taken. Each residual either vanishes by a local estimate, is absorbed by a
proved local compactness/tail statement, or is routed as an exact mildness
obligation.

If the Duhamel ledger has an open entry, this node records a mildness gap
instead of an endpoint-ready solution. Output:

$$
\Gamma_{\mathrm{PS15m}}
=
\Gamma_{\mathrm{PS15l}}
\cup
\{\text{finite-shift bounded mild ancient pullback or exact mildness gap}\}.
$$

**`PS15n` -- Imported endpoint theorem application.**
If `PS15l` supplies the backward sequence-$L^3$ and `PS15m` supplies the
finite-shift bounded mild ancient pullback, the Albritton--Barker endpoint
theorem gives $u^T\equiv0$ for each $T<0$. Hence $U\equiv0$, contradicting
retained compact activity. If either the backward sequence, the mild Duhamel
gate, whole-space domain status, boundedness, or active-attainment transfer is
missing, this node records the exact endpoint theorem gap and routes to
`PS31`; it does not close the branch. Output:

$$
\Gamma_{\mathrm{PS15n}}
=
\Gamma_{\mathrm{PS15m}}
\cup
\{\text{local non-tight terminal branch excluded, or exact endpoint theorem gap}\}.
$$

Thus `PS15a`--`PS15n` is the explicit local replacement for using global
uniform tightness as an entry assumption. The chain may close only when every
local compactness, pressure, recurrence, finite-family, and mildness
obligation has been discharged.

### Optional Global Shortcut and Local Assembly Route

There is an optional global shortcut. If the branch already proves the uniform
tail bound

$$
\sup_{\tau\in\mathbb R}\int_{|y|>R}|V(y,\tau)|^3\,dy\to0
\qquad (R\to\infty).
$$

then `PS15` may recover a global $L^\infty_\tau L^3_y$ bound and proceed to
the endpoint theorem after the Duhamel/mildness gate is also verified. This is
not the default local sieve route.

If this estimate is unavailable or false, the decisive output is instead the
local residual witness

$$
\eta_{\rm tail}>0,\qquad R_k\to\infty,\qquad \tau_k,
\qquad
\int_{|y|>R_k}|V(y,\tau_k)|^3\,dy\ge\eta_{\rm tail},
$$

together with the covariant observer routing status required by
`ST1`--`ST20`.

### Practical Verification Steps

1. If a theorem or prior estimate already proves uniform tightness, record the
   induced global
   $L^\infty_\tau L^3_y$ bound.
2. If tightness holds, verify the physical pullback is bounded mild ancient on
   each finite terminal-time restriction.
3. Use $L^3$ critical-norm invariance on times $t_k\downarrow-\infty$.
4. Apply the endpoint ancient $L^3$ theorem only after its hypotheses match.
5. Compare the zero conclusion with retained activity.
6. If tightness is unavailable or fails, do not try to prove one global tail
   estimate at this node. Record the exterior witness of Lemma PS15.4, convert it
   into covariant observer data through `PS15a`--`PS15d`, and route diffuse or
   generic non-tight branches through `PS15e`--`PS15n`, equivalently through
   `ST0`--`ST20`. Route to `PS16` or `PS18`--`PS20` only when the active-core
   or finite-separated-family hypotheses of those nodes have already been
   verified.

## Estimate Step $B_{\mathrm{PS15}}$

The estimate step has two modes. In the imported global mode, it is the
tightness-to-endpoint-$L^3$ argument in Lemmas PS15.1--PS15.3, together with a
separately verified Duhamel/mildness gate. In the local sieve mode, it is only
the non-tightness witness extraction in Lemma PS15.4 followed by
`PS15a`--`PS15n` or `ST0`--`ST20`. Later routes are not used unless their own
local lower bounds, finite-budget arguments, diffuse-residual tests, and
residual-ledger entries have been verified.

## Failure Case

Failure name: unresolved tightness or observer-routing branch.

Analytic meaning: either the record asserts spatial tightness but lacks the
exact tail estimate or mild Duhamel gate, or tightness fails but the escaping
tail witness has not yet been converted into a covariant local state-space
branch.

## Refinement Step

Allowed refinements:

1. strengthen tail estimates;
2. verify the mild Duhamel gate on finite terminal slabs;
3. pass to hull limits using `PS13`;
4. assign verified active exterior packets to `PS16`;
5. assign verified finite separated packets to `PS18`--`PS20`;
6. send the generic non-tight residual witness to `PS15a`--`PS15n` and the
   corresponding `ST0`--`ST20` state-space closure.

Progress measure: tail modulus is fixed, or non-tightness is represented by a
covariant observer-state witness.

## Data Passed Forward

If tightness holds and mildness is verified, `PS15` excludes the branch. If
tightness holds but mildness is not verified, the data passed forward record a
mildness obstruction. If tightness fails, the next proof step is the explicit
local chain `PS15a`--`PS15n` together with the state-space residual block
`ST0`--`ST20`, except for subbranches already verified as active exterior or
finite separated packets. The data passed forward are

$$
\Gamma_{\mathrm{PS15}}
=
\Gamma_{\mathrm{PS9}}
\cup
\{\text{endpoint exclusion, or mildness obstruction, or }
\eta_{\rm tail},R_k,\tau_k,
\text{ route status: active exterior / finite separated / local residual state},
\Gamma_{\mathrm{PS15a}}\to\cdots\to\Gamma_{\mathrm{PS15n}}
\text{ on the local non-tight closure path}\}.
$$

---

# 22. `PS16` -- Radiation or Escaping-Profile Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknowns are rescaled suitable weak solutions or profile components in
selected concentration frames, together with pressure representatives normalized
by subtracting spatial means.

### Standing Assumptions

The incoming record states that the incoming branch has positive local CKN mass and that every selected
frame satisfies the local energy inequality and pressure compactness already
verified in `PS7`.

### Objects Inspected

Inspect profile components in $L^3_{\rm loc}$, mixed stresses in
$L^{3/2}_{\rm loc}$, pressure oscillations in $L^{3/2}_{\rm loc}$ after
subtracting spatial means, and the corresponding pressure forces in the local
dual topology of the represented equation.

### Dependencies Used

Positive activity comes from `PS8`; frame parameters come from `PS2` and
`PS11`; non-tightness comes from `PS15`; pressure stability comes from `PS4`.

### Local Obstruction Predicate

$P_{\mathrm{PS16}}$ holds only if a profile component has nonzero retained mass
while being invisible in all selected compact frames already registered.

### Local Lemmas to Prove

**Lemma PS16.1 -- Escaping or strict outer critical rescalings vanish
locally.**
Let $\phi\in L^3(\mathbb R^3)$ and

$$
W_n(y)=\rho_n\phi(z_n+\rho_n y).
$$

If either $\rho_n\to0$, or

$$
\operatorname{dist}(z_n+\rho_nB_R,K)\to\infty
\qquad\text{for every compact }K\Subset\mathbb R^3,
$$

then

$$
\|W_n\|_{L^3(B_R)}\to0.
$$

**Proof.** By the critical change of variables,

$$
\|W_n\|_{L^3(B_R)}^3
=
\int_{z_n+\rho_nB_R}|\phi(x)|^3\,dx.
$$

If $\rho_n\to0$, the sets $z_n+\rho_nB_R$ have measure
$\rho_n^3|B_R|\to0$, and absolute continuity of the integral of
$|\phi|^3$ gives convergence to zero uniformly in the centers. If the sets
escape to infinity, choose a compact set $K$ so that

$$
\int_{\mathbb R^3\setminus K}|\phi|^3<\varepsilon.
$$

For large $n$, $z_n+\rho_nB_R\subset\mathbb R^3\setminus K$, hence the same
integral is below $\varepsilon$.

**Lemma PS16.1b -- Uniform-in-time spatial invisibility.**
Let $I$ be a compact time interval, let $\sigma_n(I)\subset K$ for a compact
time interval $K$, and assume

$$
\{|\phi(\cdot,\sigma)|^3:\sigma\in K\}
$$

is uniformly integrable in space. For example, this holds when
$\phi\in C(K;L^3(\mathbb R^3))$. If

$$
W_n(y,s)=\rho_n\phi(z_n+\rho_n y,\sigma_n(s))
$$

and either $\rho_n\to0$ or $z_n+\rho_n B_R$ escapes every compact set
uniformly for $s\in I$, then

$$
\|W_n\|_{L^\infty(I;L^3(B_R))}\to0,
\qquad
\|W_n\|_{L^3(B_R\times I)}\to0.
$$

**Proof.** For every $s\in I$,

$$
\|W_n(s)\|_{L^3(B_R)}^3
=
\int_{z_n+\rho_nB_R}
|\phi(x,\sigma_n(s))|^3\,dx.
$$

Uniform integrability in $\sigma\in K$ gives uniform convergence to zero when
the integration sets shrink or escape. Taking the supremum in $s$ gives the
$L^\infty_sL^3_y$ estimate, and the spacetime estimate follows. Mere
boundedness in $L^\infty_tL^3_x$ is not enough unless uniform spatial
integrability on $K$ has also been proved.

**Lemma PS16.2 -- Separated active frames are invisible in each other locally.**
If

$$
\frac{|x_n^q-x_n^p|}{\lambda_n^p}\to\infty
$$

and $\phi^q\in L^3(\mathbb R^3)$, then the $q$-profile observed in the
$p$-frame satisfies $W_n\to0$ in $L^3(B_R)$ for every finite $R$.

**Proof.** Set

$$
z_n=\frac{x_n^p-x_n^q}{\lambda_n^q},
\qquad
\rho_n=\frac{\lambda_n^p}{\lambda_n^q}.
$$

Then $W_n(y)=\rho_n\phi^q(z_n+\rho_n y)$ and

$$
\frac{|z_n|}{\rho_n}
=
\frac{|x_n^q-x_n^p|}{\lambda_n^p}\to\infty.
$$

If $\rho_n\to0$, Lemma PS16.1 gives local vanishing. Otherwise pass to a
subsequence with $\rho_n\ge\rho_0>0$. For $y\in B_R$,

$$
|z_n+\rho_n y|
\ge \rho_n\left(\frac{|z_n|}{\rho_n}-R\right)\to\infty,
$$

so the sets $z_n+\rho_nB_R$ escape every compact subset. Lemma PS16.1 again
applies. The remaining regime $\rho_n\to\infty$ with bounded
$|z_n|/\rho_n$ is not a separated-frame case; it is a same-point inner-scale
cascade and is routed to `PS11` or `PS18`.

**Lemma PS16.3 -- Strict outer scales vanish in an innermost frame.**
If several profiles have the same physical center and

$$
\frac{\lambda_n^1}{\lambda_n^j}\to0
$$

for an innermost scale $\lambda_n^1$, then every outer profile $\phi^j\in L^3$
converges to zero in $L^3(B_R)$ when viewed in the $\lambda_n^1$-frame.
This lemma is used only after the selected frame has been chosen innermost
among the same-point cascade, or after all stricter inner scales have been
included in the selected compound core.

**Proof.** In the innermost variables the outer contribution has the form

$$
W_n^j(y)=
\frac{\lambda_n^1}{\lambda_n^j}
\phi^j\left(z_n^j+\frac{\lambda_n^1}{\lambda_n^j}y\right).
$$

Apply Lemma PS16.1 with
$\rho_n=\lambda_n^1/\lambda_n^j\to0$. Non-$L^3$ ambient components, such as
constant or affine backgrounds, are not covered by this lemma. They must be
recorded separately as modulation or Galilean data before any removal is
claimed. Strict inner scales do not generally vanish in an outer frame; they
can concentrate there and must be retained or routed to the cascade analysis.

**Spacetime form used in applications.**
The preceding three lemmas are applied on compact time intervals in the
selected frame. Thus the actual estimate recorded for a profile evolution
$\phi^j(\cdot,\sigma)$ is

$$
\left\|
\frac{\lambda_n^p}{\lambda_n^q}
\phi^j\left(
\frac{x_n^p-x_n^q}{\lambda_n^q}
+\frac{\lambda_n^p}{\lambda_n^q}y,\sigma_n(\tau)
\right)
\right\|_{L^\infty_I L^3(B_R)}
\to0
$$

or the corresponding $L^3_{y,\tau}(B_R\times I)$ convergence, depending on
the profile package available. The pressure contribution generated by such an
invisible component is not considered removed until its mixed source is also
small in $L^1_I L^{3/2}_{\rm loc}$ or has been placed in the pressure-tail
record used later by `PS20`. This prevents a velocity-only radiation
estimate from silently closing a pressure defect.

**Lemma PS16.4 -- Escaping mass either vanishes or recenters.**
Let a component be invisible in the selected frame and satisfy a fixed
local lower bound

$$
\limsup_n
\left[
\rho_n^{-2}\iint_{Q_{\rho_n}(z_n)}|u_n|^3
+\rho_n^{-2}\iint_{Q_{\rho_n}(z_n)}
|p_n-(p_n)_{B_{\rho_n}}|^{3/2}
\right]>0.
$$

Then rescaling around $(z_n,\rho_n)$ produces a new active frame with the same
local suitability and pressure-gauge structure as the selected frame.

**Proof.** Choose a subsequence on which the displayed scale-invariant
quantity is bounded below by $\eta>0$. The parabolic rescaling at
$(z_n,\rho_n)$ preserves the CKN quantity on $Q_1$. The local energy inequality
is invariant under the same scaling. The pressure representative is transformed
by the Navier--Stokes pressure scaling and then re-gauged by subtracting spatial
means on the unit ball, so the pressure oscillation term is unchanged. Hence
the rescaled sequence is admissible for `PS2`--`PS8` and carries positive local
activity.

**Lemma PS16.5 -- Radiation removal has complete pressure/source status.**
After testing an escaping or radiative component, and only when the required
estimate or lower bound in the relevant item has been proved, exactly one of
the following statuses is recorded:

1. the component is locally invisible in the selected frame, and its mixed
   stress and pressure source are also invisible in the topology needed by
   `PS20`;
2. the component carries a positive scale-invariant CKN lower bound and is
   recentered as a new active frame by Lemma PS16.4;
3. the velocity is locally small but the associated pressure source, cutoff
   source, or local harmonic pressure remainder is not controlled, in which
   case the branch is a named defect routed to `PS30`;
4. the remaining mass is exterior and diffuse with no above-threshold compact
   cylinder, in which case the branch is routed to the residual complement
   `PS34`.

If the available topology does not decide any of these verified alternatives,
the missing estimate remains an explicit unresolved estimate and the component
is not removed.

**Proof.** The velocity alternatives are determined by Lemmas PS16.1--PS16.4:
local vanishing, separation/outer-scale vanishing, or positive-mass
recentering. For a velocity-vanishing component, inspect the tensor products
with the selected component and the pressure source generated by those tensors.
If the mixed tensor and pressure-tail record match the topology used in
`PS20`, the component is invisible as a term in the local equation. If the
velocity estimate is present but the pressure or cutoff estimate is missing,
the radiation-removal estimate is unavailable and the branch is routed to
`PS30`. Finally, if no fixed compact cylinder has
above-threshold mass but the exterior witness from `PS15` or `PS14` remains
positive, the branch is diffuse residual mass and belongs to `PS34`. These
four statuses are mutually exclusive by the ordered tests and exhaustive for
every radiative component
whose local vanishing, CKN lower-bound, pressure-source, cutoff-source, and
diffuse-tail predicates are all decidable in the recorded topology. If one of
those predicates is not decidable, the proof does not assert invisibility or a
residual classification.

### Specific Estimate

The decisive local estimate is

$$
\left\|
\frac{\lambda_n^p}{\lambda_n^q}
\phi^q\left(
\frac{x_n^p-x_n^q}{\lambda_n^q}
+\frac{\lambda_n^p}{\lambda_n^q}y
\right)
\right\|_{L^3(B_R)}
\to0
$$

whenever the $q$-component is separated from the $p$-frame or belongs to a
strict outer scale.

### Practical Verification Steps

1. List all retained profile frames and their scale-center ratios.
2. For each nonselected component, prove either separation, strict outer-scale
   behavior, or comparability.
3. Apply Lemmas PS16.1--PS16.3 to remove invisible components locally.
4. If invisible mass still carries a CKN lower bound, recenter it by Lemma
   PS16.4 and return it to the active-frame analysis.
5. Verify the complete pressure/source status in Lemma PS16.5.

## Estimate Step $B_{\mathrm{PS16}}$

The estimate step is the $L^3_{\rm loc}$ invisibility calculation, the
associated pressure-stability estimate, and the complete radiation status
record of Lemma PS16.5.

## Failure Case

Failure name: unresolved radiative concentration.

Analytic meaning: a component escapes selected compact frames but still appears
to carry scale-invariant local CKN mass.

## Refinement Step

Allowed refinements:

1. recenter at the escaping CKN mass;
2. add a separated active frame;
3. group comparable same-point frames before retesting;
4. send a nonvanishing exterior component to `PS18`.

Progress measure: the number of unassigned positive-mass components decreases,
or a new active frame is explicitly registered.

## Data Passed Forward

The next proof step is `PS17`. The data passed forward are

$$
\Gamma_{\mathrm{PS16}}
=
\Gamma_{\mathrm{PS15}}
\cup
\{\text{radiative components invisible, recentered, or unresolved}\}.
$$

---

# 23. `PS17` -- Rough-Core Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a suitable weak solution in a selected physical or renormalized
frame. If modulation is present, the drift coefficients $a(\tau)$ and
$b(\tau)$ are bounded on $J$.

### Standing Assumptions

The lemma hypotheses contain

$$
\mathcal C_J(2R)+\mathcal D_J(2R)<\infty
$$

and the local energy inequality on $B_{2R}\times J$.

### Objects Inspected

Inspect local kinetic energy, local enstrophy, pressure oscillation, convection
flux, and modulation terms produced by the selected frame.

### Dependencies Used

Local suitability comes from `PS7`; compact-frame selection comes from `PS16`;
pressure normalization comes from `PS4`; modulation equations come from `PS5`.

### Local Obstruction Predicate

$P_{\mathrm{PS17}}$ holds when compact-cylinder $H^1$ control fails despite the
claimed finite compact CKN package.

The Caccioppoli estimate below is conditional on the displayed compact
package. If a pressure oscillation term, local energy inequality, interior
time truncation, or bounded modulation coefficient is absent, the node does
not absorb the failure into the constant. The missing item remains an explicit
pressure/modulation defect routed to `PS30`, and the branch does not use the
Caccioppoli estimate until it is supplied.

### Local Lemmas to Prove

**Lemma PS17.1 -- Compact-cylinder Caccioppoli estimate.**
Let $J'\Subset J$ and let the local energy inequality hold on
$B_{2R}\times J$.
If

$$
\mathcal C_J(2R)+\mathcal D_J(2R)<\infty,
$$

then

$$
\mathcal A_{J'}(R)+\mathcal H_{J'}(R)
\le
C\left(1+\mathcal C_J(2R)+\mathcal D_J(2R)\right),
$$

where $C$ depends on $R$, $J$, $J'$, the viscosity, and the bounded modulation
coefficients.

**Proof.** Let

$$
\delta=\operatorname{dist}(J',\partial J)>0.
$$

Choose $\eta\in C_c^\infty(J)$ and $\zeta\in C_c^\infty(B_{2R})$ with

$$
0\le\eta,\zeta\le1,\qquad
\eta\equiv1\text{ on }J',\qquad
\zeta\equiv1\text{ on }B_R,
$$

and

$$
\|\eta'\|_\infty\le C\delta^{-1},\qquad
\|\nabla\zeta\|_\infty\le CR^{-1},\qquad
\|\Delta\zeta\|_\infty\le CR^{-2}.
$$

Use

$$
\phi(y,\tau)=\eta(\tau)\zeta(y)^2
$$

in the local energy inequality. Choose $\tau_-<\inf J'$ and
$\tau_+>\sup J'$ in $J$ with $\eta(\tau_\pm)=0$. Applying the local energy
inequality on $(\tau_-,\tau_+)$ gives

$$
\nu\iint_{B_{2R}\times J}\phi|\nabla V|^2
\le
|R_\tau|+|R_\Delta|+|R_{\rm conv}|+|R_{\rm pres}|+|R_{\rm mod}|,
$$

where $R_\tau$ and $R_\Delta$ are the time and Laplacian cutoff terms,
$R_{\rm conv}$ is the cubic convection flux, $R_{\rm pres}$ is the pressure
flux with $P-(P)_{B_{2R}}(\tau)$, and $R_{\rm mod}$ contains bounded
scale/translation modulation terms. The same estimate applied on
$(\tau_-,\sigma)$ for a.e. $\sigma\in J'$ leaves
$\frac12\int_{B_R}|V(\sigma)|^2$ on the left and therefore controls
$\mathcal A_{J'}(R)$ as well.

Set

$$
C_3=\mathcal C_J(2R),\qquad D_{3/2}=\mathcal D_J(2R),\qquad
L_2=\iint_{B_{2R}\times J}|V|^2 .
$$

By Holder on the finite cylinder,

$$
L_2\le |B_{2R}\times J|^{1/3}C_3^{2/3}
\le C(R,|J|)(1+C_3).
$$

Thus

$$
|R_\tau|+|R_\Delta|
\le C(R,|J|,\delta,\nu)(1+C_3).
$$

The convection term satisfies

$$
|R_{\rm conv}|
\le C R^{-1}\iint_{B_{2R}\times J}|V|^3
\le C R^{-1}C_3.
$$

For the pressure flux, subtracting $(P)_{B_{2R}}(\tau)$ is legitimate because

$$
\int_{B_{2R}}(P)_{B_{2R}}(\tau)V\cdot\nabla\phi
=
-(P)_{B_{2R}}(\tau)\int_{B_{2R}}\phi\,\nabla\cdot V=0.
$$

Holder's inequality gives

$$
\iint |P-(P)_{B_{2R}}|\,|V|\,|\nabla\phi|
\le
C_R\mathcal D_J(2R)^{2/3}\mathcal C_J(2R)^{1/3}.
$$

Young's inequality bounds this by $C_R(C_3+D_{3/2})$.

The modulation terms have the schematic form

$$
\iint a(\tau)|V|^2(\phi+y\cdot\nabla\phi)
+
\iint b(\tau)\cdot\nabla\phi\,|V|^2 .
$$

On $B_{2R}$, $|y|\le2R$ and $|\nabla\phi|\le CR^{-1}$, while
$a,b$ are bounded on $J$. Hence

$$
|R_{\rm mod}|
\le
C(R,|J|,\|a\|_{L^\infty(J)},\|b\|_{L^\infty(J)})(1+C_3).
$$

Collecting the bounds and using $\phi\equiv1$ on $B_R\times J'$ gives
$\mathcal H_{J'}(R)\le C(1+C_3+D_{3/2})$. Repeating the same estimates with
terminal time $\sigma\in J'$ and taking the essential supremum gives the
corresponding bound for $\mathcal A_{J'}(R)$.

**Lemma PS17.2 -- Rough-core loss forces compact CKN failure.**
If $\mathcal A_{J'}(R)+\mathcal H_{J'}(R)$ is unbounded along a sequence while
the assumptions of Lemma PS17.1 remain valid with uniformly bounded constants,
then $\mathcal C_J(2R)+\mathcal D_J(2R)$ is unbounded along that sequence.

**Proof.** Lemma PS17.1 gives constants $C=C(R,J,J')$ such that

$$
\mathcal A_{J'}(R)+\mathcal H_{J'}(R)
\le C\left(1+\mathcal C_J(2R)+\mathcal D_J(2R)\right).
$$

Uniform boundedness of $\mathcal C_J(2R)+\mathcal D_J(2R)$ therefore implies
uniform boundedness of $\mathcal A_{J'}(R)+\mathcal H_{J'}(R)$. The verified
failure of the latter along a sequence forces
$\mathcal C_J(2R)+\mathcal D_J(2R)\to\infty$ along that sequence.

**Lemma PS17.3 -- Compact CKN failure produces a new concentration core.**
If $\mathcal C_J(2R)+\mathcal D_J(2R)\to\infty$, then there are centers
$z_n$, radii $\rho_n$, and rescaled suitable weak solutions with

$$
\rho_n^{-2}\iint_{Q_{\rho_n}(z_n)}|u_n|^3
+
\rho_n^{-2}\iint_{Q_{\rho_n}(z_n)}
|p_n-(p_n)_{B_{\rho_n}}|^{3/2}
\ge\eta
$$

for some $\eta>0$.

**Proof.** Work inside a fixed compact cylinder
$B_{2R}\times J$ and restrict to parabolic subcylinders whose double
enlargement remains inside this cylinder. For each $n$, choose a cylinder
$Q_{\rho_n}(z_n)$ on which the local scale-invariant CKN quantity is at least
one half of the supremum over this admissible family. If the supremum tends to
infinity, such a cylinder certainly has value at least any fixed
$\eta>0$ for all large $n$. If the supremum is merely bounded below by a
positive amount after a dyadic localization, choose $\eta$ below that amount.

The chosen cylinders inherit suitability from the outer cylinder and use the
pressure representative

$$
p_n-(p_n)_{B_{\rho_n}(x_n)}(t)
$$

on their own spatial balls. Parabolic rescaling maps
$Q_{\rho_n}(z_n)$ to $Q_1$, preserves the local energy inequality, and
preserves the displayed CKN quantity exactly. The rescaled pressure is then
renormalized by subtracting its spatial mean on $B_1$. Thus the extracted core
has the same admissibility and pressure-gauge package required by `PS2`--`PS8`,
not merely a formal lower bound.

### Specific Estimate

The decisive local estimate is

$$
\mathcal A_{J'}(R)+\mathcal H_{J'}(R)
\le
C\left(1+\mathcal C_J(2R)+\mathcal D_J(2R)\right).
$$

### Practical Verification Steps

1. Fix $B_R\times J'\Subset B_{2R}\times J$.
2. Choose cutoffs and insert them into the local energy inequality.
3. Estimate convection, pressure, cutoff, and modulation terms.
4. If the estimate fails, identify which compact CKN term is unbounded.
5. Extract a new CKN concentration core and assign it to the multibubble
   analysis.

## Estimate Step $B_{\mathrm{PS17}}$

The estimate step is the compact-cylinder Caccioppoli inequality.

## Failure Case

Failure name: unresolved rough local core.

Analytic meaning: the branch lacks local $H^1$ control on compact cylinders and
has not been converted into a CKN concentration core.

## Refinement Step

Allowed refinements:

1. enlarge the outer cylinder;
2. improve the pressure gauge;
3. extract the new concentration core from the unbounded CKN quantity;
4. assign the new core to `PS18`.

Progress measure: either compact $H^1$ control is obtained, or a new active core
is named.

## Data Passed Forward

The next proof step is `PS18`. The data passed forward are

$$
\Gamma_{\mathrm{PS17}}
=
\Gamma_{\mathrm{PS16}}
\cup
\{\text{compact }H^1\text{ control or new CKN core}\}.
$$

---

# 24. `PS18` -- Multicenter or Multibubble Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknowns are finitely or countably many candidate active profiles in
critical variables, each divergence-free and equipped with a local pressure
representative.

### Standing Assumptions

The incoming record states that every registered active frame has a fixed
positive local critical lower bound and that the selected compact analysis
window carries a finite compact-window CKN budget in the sense of
Lemma PS18.1.

### Objects Inspected

Inspect the active mass record, pairwise frame geometry, compactness of the
selected cores, and the mixed terms between distinct profiles.

### Dependencies Used

Active mass comes from `PS8`, recentering from `PS16`, rough-core extraction
from `PS17`, and scale classification from `PS11`.

### Local Obstruction Predicate

$P_{\mathrm{PS18}}$ holds when the single-core reduction is not sufficient
because at least two active cores remain after local invisibility reductions.

### Local Lemmas to Prove

**Lemma PS18.1 -- Local finite active count above threshold.**
Fix a compact analysis window

$$
K\Subset B_{R_*}\times I_*
$$

and a slightly larger compact window

$$
K^+=B_{R^*}\times I^*,
\qquad
K\Subset K^+,
\qquad
B_{R_*}\Subset B_{R^*}.
$$

Cover $K$ by parabolic cylinders $Q_\rho(z^\ell)$ whose doubled cylinders
remain in $K^+$ and whose overlap is bounded by $N_0$. Declare a cylinder
active when

$$
\rho^{-2}\iint_{Q_\rho(z^\ell)}|V_n|^3
+
\rho^{-2}\iint_{Q_\rho(z^\ell)}
|P_n-(P_n)_{B_\rho}|^{3/2}
\ge\eta_*.
$$

Assume the compact-window budget

$$
\iint_{K^+}|V_n|^3
+
\int_{I^*}\int_{B_{R^*}}
|P_n-(P_n)_{B_{R^*}}(\tau)|^{3/2}
\le M_K.
$$

Then the number of active cylinders in $K$ is bounded by

$$
N_{\rm active}(K,\eta_*)
\le
C\,N_0\,\frac{M_K}{\eta_*}.
$$

This is the Type II counting mechanism. A global profile budget

$$
\sum_j\|\phi^j\|_{L^3(\mathbb R^3)}^3<\infty
$$

may be used only in a separate branch that explicitly supplies such a global
profile decomposition; it is not an entry assumption for the local Type II
path.

**Proof.** Sum the active-cylinder inequalities over the selected active
subfamily. The velocity contributions are bounded by the overlap constant.
For pressure, set

$$
q_n=P_n-(P_n)_{B_{R^*}}(\tau).
$$

On every selected ball,

$$
P_n-(P_n)_{B_\rho}=q_n-(q_n)_{B_\rho},
$$

and Jensen's inequality gives

$$
\int_{B_\rho}|q_n-(q_n)_{B_\rho}|^{3/2}
\le
C\int_{B_\rho}|q_n|^{3/2}.
$$

After integrating in time and summing, bounded overlap controls the sum by
$C N_0M_K$. Dividing by $\eta_*$ gives the count.

**Lemma PS18.2 -- Pairwise parabolic active-frame classification.**
After passing to a subsequence, each pair of active frames

$$
\mathfrak F_n^i=(x_n^i,t_n^i,\lambda_n^i),
\qquad
\mathfrak F_n^j=(x_n^j,t_n^j,\lambda_n^j)
$$

falls into exactly one of the following ordered classes:

1. parabolically separated:

   $$
   \frac{|x_n^i-x_n^j|}{\max(\lambda_n^i,\lambda_n^j)}
   +
   \frac{|t_n^i-t_n^j|}
   {\max((\lambda_n^i)^2,(\lambda_n^j)^2)}
   \to\infty;
   $$

2. comparable parabolic frame: scale ratios are bounded above and below by
   positive constants, and normalized space-time offsets are bounded;

3. same-point scale cascade: normalized space-time offsets are bounded, but
   $\lambda_n^i/\lambda_n^j\to0$ or $\lambda_n^i/\lambda_n^j\to\infty$;

4. comparable scale with unbounded normalized time shift: scales and spatial
   offsets are comparable, but

   $$
   \frac{|t_n^i-t_n^j|}{(\lambda_n^i)^2}\to\infty
   $$

   after replacing $(\lambda_n^i)^2$ by a comparable squared scale if needed.

The last case is a time-hull or terminal-translation branch. It is not
automatic local invisibility.

**Proof.** Lemma PS18.1 gives a finite active list on the selected compact
window. Pass to one subsequence on which every scale ratio converges in
$[0,\infty]$ and every normalized spatial and temporal offset either stays
bounded or tends to infinity. The ordered alternatives above exhaust these
subsequential limits. Parabolic separation permits invisibility only after the
spacetime invisibility hypotheses of `PS16` have been checked.

**Lemma PS18.3 -- Same-point comparable profiles form one compound core.**
If a group has bounded normalized space-time offsets and comparable scales,
then its finite sum is a single divergence-free compound local core in a common
parabolic frame. The compound core may be removed only if the exact
perturbative criterion being invoked is verified. For a CKN removal this means
a spacetime pressure-aware smallness condition, not merely a small spatial
$L^3$ norm.

**Proof.** Choose one representative
$(x_n^{j_k},t_n^{j_k},\lambda_n^{j_k})$ for the comparable class. After
passing to the subsequence already fixed in Lemma PS18.2,

$$
\rho_j=\lim_n\frac{\lambda_n^j}{\lambda_n^{j_k}}\in(0,\infty),
\qquad
z_j=\lim_n\frac{x_n^j-x_n^{j_k}}{\lambda_n^{j_k}}\in\mathbb R^3,
\qquad
\theta_j=\lim_n
\frac{t_n^j-t_n^{j_k}}{(\lambda_n^{j_k})^2}
\in\mathbb R .
$$

In the representative frame the class is represented by the finite parabolic
sum

$$
\Phi^k(y,s)=
\sum_{j\in J_k}\rho_j^{-1}
\phi^j\left(\frac{y-z_j}{\rho_j},
\frac{s-\theta_j}{\rho_j^2}\right).
$$

Finiteness of $J_k$ comes from Lemma PS18.1. Each summand is divergence-free,
and divergence commutes with translation and dilation, so $\Phi^k$ is
divergence-free. Local critical spacetime norms are controlled under the
displayed bounded changes of variables. If the branch verifies

$$
\iint_Q|\Phi^k|^3
+
\iint_Q|\Pi^k-(\Pi^k)_B|^{3/2}
<
\varepsilon_{\rm CKN}
$$

on the same compact cylinder and pressure gauge used by the branch, then the
compound class is perturbative by the local CKN criterion and is removed by
`PS21`. If the theorem invoked is instead a global mild small-data theorem,
the branch must verify that the compound object is genuinely global initial
data in that theorem's topology. Without such a theorem-matching smallness
record, $\Phi^k$ is retained as a single active compound core.

This grouping is only a geometric and critical-norm reduction. It does not
claim that the finite sum $\Phi^k$ by itself solves Navier--Stokes with a
closed pressure. The compound core carries the common frame, divergence-free
condition, pressure gauge, and interaction estimates forward to `PS19` and
`PS20`; the standalone local equation is obtained only after terminal
decoupling removes all mixed stresses and pressure sources.

**Lemma PS18.4 -- Minimal bad-configuration extraction.**
If the branch is not a single retained core after applying Lemmas PS18.1--PS18.3
and the invisibility lemmas of `PS16`, then it contains a same-point compound
core, a same-point scale cascade, separated active centers, or a
gauge-degenerate active selection.

**Proof.** Start with the finite active list supplied by Lemma PS18.1 and
classify every pair by Lemma PS18.2. Comparable parabolic equivalence classes
are replaced by compound cores using Lemma PS18.3. A parabolically separated
pair remains a separated active branch unless its contribution is locally
invisible in the selected compact frame; the invisible case was already
removed or recentered by `PS16`. A same-point pair with scale ratio tending to
$0$ or $\infty$ is precisely a scale cascade. A comparable-scale pair with
unbounded normalized time shift is a time-hull or terminal-translation branch.
After these reductions, if the branch is still not a single nondegenerate
selected core, the only unassigned obstruction is nonuniqueness or degeneracy
in the center-scale selection of the selected core. This is recorded as the
gauge-degenerate active selection.

### Specific Estimate

The decisive counting estimate is the compact-window budget

$$
N_{\rm active}(K,\eta_*)\eta_*
\le
C N_0M_K,
$$

where $M_K$ is the compact-window CKN mass and $N_0$ is the bounded-overlap
constant.

### Practical Verification Steps

1. Fix an activity threshold $\eta_*$ below the retained lower bound.
2. Count all active cylinders by compact-window CKN mass and bounded overlap.
3. Classify every pair by parabolic center, time, and scale ratios.
4. Group comparable same-point frames into compound cores.
5. Remove locally invisible separated or outer-scale components using `PS16`.
6. Assign the resulting finite active packet to `PS19`.

## Estimate Step $B_{\mathrm{PS18}}$

The estimate step is the finite active-count bound and pairwise parameter
classification.

## Failure Case

Failure name: unresolved multicore concentration.

Analytic meaning: more than one active core remains, but the pairwise
separation, grouping, or gauge-degenerate alternative has not been identified.

## Refinement Step

Allowed refinements:

1. lower the active threshold and repeat the finite count;
2. pass to subsequences to settle all pairwise ratios;
3. group comparable same-point profiles;
4. recenter parabolically separated positive-mass components.

Progress measure: the unclassified active-frame list is replaced by a finite
classified packet.

## Data Passed Forward

The next proof step is `PS19`. The data passed forward are

$$
\Gamma_{\mathrm{PS18}}
=
\Gamma_{\mathrm{PS17}}
\cup
\{\text{single core, finite local packet, same-point cascade,
parabolically separated frames, time-hull branch, or gauge degeneracy}\}.
$$

---

# 25. `PS19` -- Finite Packet Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a finite local packet of active parabolic cylinders or local
profile pieces in critical variables. A countable global profile expansion is
used only in a separate theorem-specific branch that explicitly supplies that
object. Each retained local packet component is divergence-free, or carries a
recorded cutoff/divergence defect, and each pressure is represented modulo a
time-dependent gauge by local spatial-mean normalization.

### Standing Assumptions

The incoming record states that `PS18` has produced a finite local active list
on the selected compact window by the compact-window CKN budget. If a global
profile expansion is present, it is recorded as an optional theorem-specific
package, not as a Type II entry assumption.

### Objects Inspected

Inspect thresholded active masses, packet ordering, critical tails, mixed
velocity stresses, and pressure sources generated by profile interactions.

### Dependencies Used

Finite counting comes from `PS18`; local invisibility comes from `PS16`;
compact $H^1$ control comes from `PS17`; pressure gauges come from `PS4`.

### Local Obstruction Predicate

$P_{\mathrm{PS19}}$ holds when no finite packet has yet been selected or when
the selected packet has unresolved local interactions.

### Local Lemmas to Prove

**Lemma PS19.1 -- Finite local packet selection.**
For a fixed compact analysis window and threshold $\eta$, the packet
$\mathcal P_\eta$ of active cylinders supplied by Lemma PS18.1 is finite. The
nonselected part may be used as a perturbative remainder only in one of the
following explicitly recorded modes.

1. **Fixed-packet mode.** There is one finite packet $\mathcal P$ such that
   the local remainder $S_n$ satisfies

   $$
   S_n\to0
   \quad\text{in the exact local topology required by }PS20.
   $$

2. **Two-limit packet mode.** For every $\varepsilon>0$ there is a finite
   packet $\mathcal P_\varepsilon$ such that

   $$
   \limsup_{n\to\infty}
   \|S_n^\varepsilon\|_{\mathcal X_{\rm loc}}
   \le\varepsilon,
   $$

   where $\mathcal X_{\rm loc}$ is the topology used to estimate mixed
   stresses and pressure sources in `PS20`. Conclusions then have the order
   $\lim_{\varepsilon\downarrow0}\limsup_{n\to\infty}$ unless a diagonal
   packet is explicitly selected.

If only velocity smallness is known but pressure/source readiness is missing,
the packet is not decoupled; it is routed to the local pressure or source
defect coordinates.

**Proof.** Finiteness of $\mathcal P_\eta$ is exactly Lemma PS18.1. The two
remainder modes are definitions of the topology available for the local
packet. In fixed-packet mode the selected core is stable along the original
subsequence. In two-limit mode the packet may depend on $\varepsilon$, so all
estimates must keep the two limits in the displayed order or perform a
diagonal selection that records the selected core. Neither mode follows from a
global summability assertion unless the branch separately supplies the
unconditional convergence of the profile tail in the local topology
$\mathcal X_{\rm loc}$.

**Lemma PS19.2 -- The discarded tail is perturbative after finite
truncation.**
Let $r_n$ be the nonselected local remainder in either fixed-packet mode or
two-limit packet mode. If the packet has been chosen so that

$$
\|r_n\|_{L^\infty_I L^3(B_{2R})}\le\varepsilon_*
\quad\text{for all large }n,
$$

where $\varepsilon_*$ is below the local critical small-data threshold, then
the discarded tail is perturbative and

$$
\|r_n\otimes r_n\|_{L^1_I L^{3/2}(B_R)}
\le
|I|\,\varepsilon_*^2 .
$$

**Proof.** The displayed $L^\infty_I L^3(B_{2R})$ smallness is an explicit
packet-readiness input. It is not inferred from the finite active count.
Once the displayed smallness is verified, Holder's inequality on the compact
cylinder gives

$$
\|r_n\otimes r_n\|_{L^1_I L^{3/2}(B_R)}
\le
|I|\,\|r_n\|_{L^\infty_I L^3(B_R)}^2
\le
|I|\,\varepsilon_*^2.
$$

The discarded tail is a perturbative error contribution in the selected local
equation. It is a perturbative solution only if the branch separately proves
that the tail itself solves the relevant Navier--Stokes or forced
Navier--Stokes equation.

The topology in the displayed smallness is part of the packet data. If the
profile package supplies only $L^3_{I,x}(B_{2R}\times I)$ smallness, the node
may still pass a mixed-stress estimate to `PS20`, but it may not claim the
stronger $L^\infty_I L^3_x$ packet readiness. In that case the packet data
state the weaker topology explicitly, and `PS20` must use the matching
$L^{3/2}_{I,x}$ estimates rather than the $L^\infty_I L^3_x$ shorthand.

**Lemma PS19.3 -- Packet ordering by active geometry.**
After passing to a subsequence, the finite packet is ordered into
same-point comparable groups, same-point scale chains, parabolically separated
groups, and comparable-scale groups with unbounded normalized time shift.

**Proof.** Since the packet is finite, apply the pairwise classification of
Lemma PS18.2 to all pairs simultaneously. Make a graph whose vertices are
packet elements and whose edges connect same-point comparable-scale pairs.
Connected components of this graph are compound-core groups. The remaining
edges are classified as same-point cascade, parabolic separation, or
comparable scale with unbounded normalized time shift.
Because the graph is finite, this sorting is stable after passing to the
already chosen subsequence.

**Lemma PS19.4 -- Decoupling preparation.**
For a selected packet element or compound core $U_n$, the sum $S_n$ of all
packet components classified as separated, strict outer-scale, or finite-tail
perturbative satisfies

$$
\|S_n\|_{L^\infty_I L^3(B_{2R})}\to0
$$

after grouping comparable same-point elements into compound cores.

**Proof.** Separated components vanish in the selected frame by Lemma PS16.2.
Strict outer-scale components vanish by Lemma PS16.3 only in an innermost
selected frame, or after all stricter inner scales have been included in the
selected compound or cascade core. Strict inner scales are never discarded from
an outer selected frame. The finite-tail
remainder is small by Lemma PS19.2, and the truncation tolerance is chosen to
go to zero along the terminal subsequence if an actual limit is needed.
Comparable same-point terms are never placed in $S_n$; they are grouped into
the selected compound core before this estimate is tested. Since only finitely
many nonselected packet components remain, the triangle inequality gives
$S_n\to0$ in $L^\infty_I L^3(B_{2R})$.

### Specific Estimate

The decisive estimate preparing `PS20` is

$$
\|U_n\otimes S_n+S_n\otimes U_n+S_n\otimes S_n\|_{L^1_I L^{3/2}(B_R)}
\to0.
$$

### Practical Verification Steps

1. Choose an activity threshold $\eta$ and a perturbative tail tolerance
   $\varepsilon$.
2. Select $\mathcal P_{\eta,\varepsilon}$ and prove it is finite.
3. Group same-point comparable packet elements.
4. Assign parabolically separated and admissible strict outer-scale components
   to $S_n$ in the selected frame, after checking the direction of the cascade.
5. Prove $S_n\to0$ in the exact local critical topology needed by `PS20`,
   preferably $L^\infty_I L^3(B_{2R})$ and otherwise a recorded weaker
   $L^3_{I,x}$ topology with matching mixed-stress estimates.
6. Record the packet as admissible input for terminal decoupling.

## Estimate Step $B_{\mathrm{PS19}}$

The estimate step is finite thresholding plus mixed-stress control.

## Failure Case

Failure name: unresolved infinite or interacting packet.

Analytic meaning: infinitely many above-threshold profiles remain, or a finite
packet has mixed stresses that are not locally small in the selected frame.

## Refinement Step

Allowed refinements:

1. adjust the threshold $\eta$;
2. group comparable same-point profiles;
3. add a missing active frame;
4. assign a non-small tail to `PS34` if it cannot be represented by named
   profiles.

Progress measure: the packet becomes finite and ordered, or the residual tail is
named explicitly.

## Data Passed Forward

The next proof step is `PS20`. The data passed forward are

$$
\Gamma_{\mathrm{PS19}}
=
\Gamma_{\mathrm{PS18}}
\cup
\{\mathcal P_\eta\text{ or }\mathcal P_\varepsilon,\ \text{packet mode},
\text{packet ordering},\ \text{interaction and pressure-source readiness record}\}.
$$

---

# 26. `PS20` -- Terminal Decoupling Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknowns are $U_n$, $S_n$, their pressures, and the localized equations on
$B_R\times I$ in the selected frame.

### Standing Assumptions

The lemma hypotheses contain

$$
\sup_n\|U_n\|_{L^\infty_I L^3(B_{2R})}<\infty,
\qquad
\|S_n\|_{L^\infty_I L^3(B_{2R})}\to0,
$$

and that pressure representatives are normalized by spatial means on balls.
For mixed-pressure decoupling the hypotheses additionally include the
compact-ball pressure decomposition record described below, including control
of the harmonic remainder on a larger ball. No global exterior pressure tail
is used on the Type II local path.

### Objects Inspected

Inspect $\nabla\cdot S_n$, $\mathcal T_n$, pressure sources from
$\mathcal T_n$, cutoff commutators, and the selected CKN mass.

### Dependencies Used

Packet selection comes from `PS19`; local pressure stability from `PS4`;
renormalized equations from `PS5`; compact-cylinder control from `PS17`.

### Local Obstruction Predicate

$P_{\mathrm{PS20}}$ holds when a term containing $S_n$ remains nonzero in the
local equation for the selected frame.

### Local Lemmas to Prove

**Lemma PS20.1 -- Divergence is preserved under packet splitting.**
If each packet component and the remainder are divergence-free, then
$\nabla\cdot S_n=0$ in distributions.

**Proof.** Divergence commutes with translations, dilations, finite sums, and
distributional limits. Therefore each summand of $S_n$ is divergence-free and
so is $S_n$.

**Lemma PS20.2 -- Mixed stresses vanish.**
Under the standing assumptions, or under the direct mixed-stress convergence
obligation supplied by a weaker `PS19` packet topology,

$$
\mathcal T_n\to0
\quad\text{in }L^1_I L^{3/2}(B_R).
$$

**Proof.** In the $L^\infty_I L^3$ packet topology, Holder's inequality gives

$$
\|U_n\otimes S_n\|_{L^1_I L^{3/2}(B_R)}
\le
|I|\|U_n\|_{L^\infty_I L^3(B_R)}
\|S_n\|_{L^\infty_I L^3(B_R)}\to0.
$$

The same estimate applies to $S_n\otimes U_n$, and

$$
\|S_n\otimes S_n\|_{L^1_I L^{3/2}(B_R)}
\le
|I|\|S_n\|_{L^\infty_I L^3(B_R)}^2\to0.
$$

If `PS19` supplies a weaker topology, these Holder estimates are replaced by
the direct convergence of $\mathcal T_n$ in
$L^1_I L^{3/2}(B_R)$. The proof uses whichever of these two convergence
routes is actually supplied; it does not infer the strong packet topology from
a weaker remainder estimate.

**Lemma PS20.3 -- Mixed pressure oscillations and forces vanish locally.**
Let $B_R\Subset B_{\Theta R}$, $\Theta>1$, and let
$\pi_n^{\rm mix}$ be decomposed on $B_{\Theta R}$ as

$$
\pi_n^{\rm mix}
=
\pi_{n,\rm loc}^{\rm mix}+h_n^{\rm mix},
$$

where

$$
-\Delta\pi_{n,\rm loc}^{\rm mix}
=
\partial_i\partial_j(\zeta\mathcal T_{n,ij}),
\qquad
\zeta\equiv1\text{ on }B_R,
$$

and $h_n^{\rm mix}$ is harmonic on $B_R$. Assume

$$
\mathcal T_n\to0
\quad\text{in }L^1_I L^{3/2}(B_{\Theta R})
$$

and the harmonic remainder has the larger-ball oscillation control

$$
\|h_n^{\rm mix}-(h_n^{\rm mix})_{B_{\Theta R}}\|
_{L^1_I L^{3/2}(B_{\Theta R})}\to0
$$

or an equivalent local pressure-decomposition estimate. Then

$$
\|\pi_n^{\rm mix}-(\pi_n^{\rm mix})_{B_R}\|_{L^1_I L^{3/2}(B_R)}
\to0.
$$

The pressure force vanishes in

$$
L^1_IW^{-1,3/2}(B_R),
$$

and also in $L^1_IH^{-m}(B_R)$ for every $m\ge3$. It may be treated as an
$H^{-1}$ term only if an additional pressure improvement, such as
$\pi_n^{\rm mix}\in L^2$, has been proved.

**Proof.** Calderon--Zygmund boundedness gives

$$
\|\pi_{n,\rm loc}^{\rm mix}\|_{L^1_I L^{3/2}(B_R)}
\le
C\|\mathcal T_n\|_{L^1_I L^{3/2}(B_{\Theta R})}
\to0.
$$

For the harmonic part, the assumed larger-ball oscillation bound and the
pressure mean comparison imply

$$
\|h_n^{\rm mix}-(h_n^{\rm mix})_{B_R}\|_{L^1_I L^{3/2}(B_R)}
\to0.
$$

Combining the local and harmonic estimates gives the displayed pressure
oscillation convergence. For a test vector $\varphi\in C_c^\infty(B_R)$,

$$
\langle\nabla\pi_n^{\rm mix},\varphi\rangle
=
-\int_{B_R}
(\pi_n^{\rm mix}-(\pi_n^{\rm mix})_{B_R})
\nabla\cdot\varphi .
$$

This is the $W^{-1,3/2}$ pressure-force topology. Since
$H^m_0(B_R)\hookrightarrow W^{1,3}(B_R)$ for $m\ge3$, the same convergence
holds in $H^{-m}$. No conclusion is drawn from local velocity convergence
alone; if the harmonic remainder control is missing, the branch is routed to
the local pressure-defect audit.

**Lemma PS20.4 -- Cutoff commutators vanish.**
Let $\chi_R\in C_c^\infty(B_{2R})$ be one on $B_R$. Every localization
commutator containing $S_n$ tends to zero in the same local dual topology used
for the represented equation.

**Proof.** The commutators are finite sums of terms supported where derivatives
of $\chi_R$ are nonzero, with factors $S_n$, $\mathcal T_n$, or
$\pi_n^{\rm mix}$. The fixed cutoff constants convert these terms into the
norms controlled by Lemmas PS20.2 and PS20.3. Terms with $S_n$ vanish in local
$L^3$, tensor terms vanish in $L^{3/2}$, and pressure cutoff terms vanish
through pressure oscillation after subtracting spatial means. Hence every
commutator containing a nonselected component tends to zero.

**Lemma PS20.5 -- Selected positive mass persists.**
If the full packet has positive selected CKN mass and $S_n\to0$ in local $L^3$
with mixed pressure oscillations and pressure forces vanishing, then the
selected component $U_n$ retains the same positive local mass up to an error
tending to zero.

**Proof.** The $L^3$ part follows from

$$
\left|\|U_n+S_n\|_{L^3(B_R)}-\|U_n\|_{L^3(B_R)}\right|
\le
\|S_n\|_{L^3(B_R)}.
$$

The pressure part is stable by Lemma PS20.3 and the corresponding pure
$S_n$ pressure estimate. Therefore subtracting $S_n$ cannot remove the retained
positive lower bound.

The pure $S_n$ pressure estimate is not implicit. It is either supplied by the
same packet-tail topology that gives $S_n\otimes S_n\to0$ in
$L^1_I L^{3/2}_{\rm loc}$ together with the compact-ball harmonic-remainder
control of Lemma PS20.3, or it is a separate pressure-defect obligation. In
the supplied case, the proof of Lemma PS20.3 applies with $\mathcal T_n$
replaced by $S_n\otimes S_n$ and shows that the pressure oscillation generated
by the discarded part vanishes after subtracting spatial means. If this pure
discarded-pressure estimate is missing, the selected positive mass cannot be
transferred from $V_n$ to $U_n$; the branch is routed to `PS30` as a pressure
defect.

**Lemma PS20.6 -- The selected limit is the full suitable limit after
decoupling.**
Assume Lemmas PS20.1--PS20.5 and the local compactness package of `PS17` for
the full suitable sequence $V_n=U_n+S_n$. Then a subsequence of the full
sequence converges to a suitable weak limit $(U,P_U)$. The same velocity $U$
is the selected-component limit because $S_n\to0$ locally in $L^3$. The limit
satisfies the standalone local Navier--Stokes equation, local energy
inequality, pressure convention, and retained positive CKN lower bound, with
no remaining source term from $S_n$.

**Proof.** Write the weak equation for the full suitable sequence
$V_n=U_n+S_n$ against a compactly supported divergence-free test field
$\varphi$ in $B_R\times I$. The linear terms split exactly. The nonlinear
term is

$$
\int (V_n\otimes V_n):\nabla\varphi
=
\int (U_n\otimes U_n):\nabla\varphi
+
\int \mathcal T_n:\nabla\varphi .
$$

The second integral tends to zero by Lemma PS20.2. For non-divergence-free
localized energy test fields, the pressure contribution created by
$\mathcal T_n$ tends to zero by Lemma PS20.3, and all cutoff terms containing
$S_n$ vanish by Lemma PS20.4. The remaining terms pass to the limit by the
local compactness and pressure convergence of the full suitable sequence and
the pressure gauge from `PS4`. Suitability of the limit follows from the
stability theorem applied to $V_n$, not from a local energy inequality for
$U_n$ alone. Since $U_n=V_n-S_n$ and $S_n\to0$ locally in $L^3$, the selected
component has the same velocity limit. Lemma PS20.5 supplies the retained
positive CKN lower bound. Hence the selected limit is an admissible local NS3D
branch only after full-sequence suitability and decoupling have both been used.

### Specific Estimate

The decisive estimate is

$$
\|\mathcal T_n\|_{L^1_I L^{3/2}(B_R)}
+
\|\pi_n^{\rm mix}-(\pi_n^{\rm mix})_{B_R}\|_{L^1_I L^{3/2}(B_R)}
\to0.
$$

For the pressure term this shorthand includes the compact-ball decomposition
from Lemma PS20.3: near-field Calderon--Zygmund convergence on
$B_{\Theta R}$ and harmonic-remainder oscillation control on the larger ball.
If that local harmonic control is missing, only the near-field pressure has
been decoupled and the branch must be routed to the pressure-defect audit
`PS30`.

### Practical Verification Steps

1. Decompose $V_n=U_n+S_n$ on $B_{2R}\times I$.
2. Verify $S_n\to0$ in $L^\infty_I L^3(B_{2R})$, or record a direct
   $L^1_I L^{3/2}$ mixed-stress convergence estimate from `PS19`.
3. Estimate all mixed stresses by Holder's inequality or by the direct
   mixed-stress record.
4. Apply compact-ball pressure reconstruction to mixed sources, including the
   larger-ball harmonic-remainder oscillation record.
5. Check cutoff commutators in the represented-equation dual topology, and in
   the compatible local energy space when the $H^1$ upgrade is being used.
6. Verify the selected CKN lower bound persists.

## Estimate Step $B_{\mathrm{PS20}}$

The estimate step is mixed-stress and mixed-pressure convergence.

## Failure Case

Failure name: terminal interaction defect.

Analytic meaning: nonselected components do not vanish from the local equation
of the selected active frame.

## Refinement Step

Allowed refinements:

1. add the interacting component to the selected compound core;
2. refine the pressure decomposition;
3. enlarge the packet and repeat `PS19`;
4. assign unresolved interaction defects to `PS30`.

Progress measure: every nonselected component is either absorbed into the
selected core or shown to vanish in the selected local equation.

## Data Passed Forward

The next proof step is `PS21`. The data passed forward are

$$
\Gamma_{\mathrm{PS20}}
=
\Gamma_{\mathrm{PS19}}
\cup
\{\text{terminal decoupling, full-sequence suitable limit, and selected positive branch}\}.
$$

---

# 27. `PS21` -- Small-Data or Perturbative Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is either a suitable weak solution on a compact physical cylinder
or a bounded ancient profile in centered variables.

### Standing Assumptions

The incoming record states that the selected branch has no unresolved interaction terms and satisfies
the local energy inequality with pressure normalized by spatial means.

### Objects Inspected

Inspect local $L^3$ velocity, local $L^{3/2}$ pressure oscillation,
$L^\infty$ size of centered profiles, and local dissipation.

### Dependencies Used

The selected branch comes from `PS20`; suitability from `PS7`; pressure gauges
from `PS4`; positive retained activity from `PS8`.

### Local Obstruction Predicate

$P_{\mathrm{PS21}}$ holds when the branch satisfies the hypotheses of the
perturbative exclusion theorem. The theorem conclusion is then compared with
the retained singular-profile lower bound.

There are three separate perturbative gates:

$$
\begin{array}{c|c|c}
\text{gate} & \text{hypothesis} & \text{conclusion}\\
\hline
\text{CKN} &
C(U;z_0,r)+D(P;z_0,r)<\varepsilon_{\rm CKN} &
\text{local boundedness}\\
\text{centered mild} &
\|V\|_{L^\infty(\mathbb R^3\times\mathbb R)}<\varepsilon_\infty
\text{ plus bounded mild formulation} &
V\equiv0\\
\text{compact Type II} &
\int_{Q_\rho}|\nabla V_n|^2\to0
\text{ plus compact convergence and velocity activity, or retained }C+D
\text{ plus pressure convergence} &
\text{zero-dissipation single-core closure}
\end{array}
$$

The node may close a branch only through the row whose hypotheses are actually
verified. The rows are not interchangeable: CKN needs pressure smallness,
centered mild Liouville needs the projected mild semigroup estimate, and
zero-dissipation needs compact convergence plus persistent selected velocity
concentration, or a combined $C+D$ lower bound together with a separate proof
that the pressure oscillation vanishes.

The CKN row is especially narrow. Local boundedness of a limit profile is not
by itself a contradiction: a nonzero smooth local profile may still carry
$L^3$ mass. CKN smallness closes only a selected singular-core cylinder, or a
cylinder on which the same pressure-normalized CKN quantity also has a
retained lower bound contradicting the smallness.

### Local Lemmas to Prove

**Lemma PS21.1 -- CKN smallness excludes local singularity.**
There is $\varepsilon_{\rm CKN}>0$ such that if

$$
C(U;z_0,r)+D(P;z_0,r)<\varepsilon_{\rm CKN},
$$

then $U$ is locally bounded in $Q_{r/2}(z_0)$. This is an exclusion only if
the branch record also says that $Q_r(z_0)$ is the selected singular-core
cylinder, or if the same cylinder carries a retained lower bound

$$
C(U;z_0,r)+D(P;z_0,r)\ge\eta_0
$$

with $\eta_0\ge\varepsilon_{\rm CKN}$.

**Proof.** The theorem input used here is the local regularity statement

$$
C(U;z_0,r)+D(P;z_0,r)<\varepsilon_{\rm CKN}
\Longrightarrow
U\in L^\infty(Q_{r/2}(z_0)).
$$

The pressure representative in $D$ is
$P-(P)_{B_r}(t)$, and subtracting this spatial mean preserves $\nabla P$ in
the Navier--Stokes equation. The displayed smallness assumption is exactly the
left-hand side of the theorem input, so local boundedness follows on
$Q_{r/2}(z_0)$. If $z_0$ is the selected singular point in the branch, this
contradicts the singular-core assumption. If the same cylinder also carries a
retained lower bound $\eta_0\ge\varepsilon_{\rm CKN}$ for the same CKN
quantity, the lower and upper hypotheses are incompatible. In every other
case the conclusion is only a regular subregion; remaining activity must be
returned to active-frame selection.

**Lemma PS21.2 -- Small bounded mild centered ancient profiles vanish.**
There is $\varepsilon_\infty>0$, chosen below the constant threshold in the
proof, such that every bounded mild ancient solution of the centered
Navier--Stokes equation on
$\mathbb R^3\times\mathbb R$ satisfying

$$
\|V\|_{L^\infty(\mathbb R^3\times\mathbb R)}
\le\varepsilon_\infty
$$

is identically zero.

**Proof.** Write the projected centered equation as

$$
\partial_\tau V=LV-\mathbb P\nabla\cdot(V\otimes V),
\qquad
L=\Delta-\frac12y\cdot\nabla-\frac12.
$$

The centered Stokes semigroup satisfies

$$
\|e^{sL}f\|_{L^\infty}\le e^{-s/2}\|f\|_{L^\infty},
$$

and

$$
\|e^{sL}\mathbb P\nabla\cdot F\|_{L^\infty}
\le
C e^{-s/2}(1-e^{-s})^{-1/2}\|F\|_{L^\infty}.
$$

This is an Oseen-kernel estimate for the combined operator
$e^{sL}\mathbb P\nabla\cdot$, not an assertion that the Helmholtz projection
is bounded on $L^\infty$ at a fixed time. Equivalently, it is an explicit
bounded-mild theorem input for the centered Stokes evolution. If this combined
semigroup estimate or the corresponding mild formulation is absent, the proof
does not use small $L^\infty$ alone as a Liouville theorem. In that case the
output is only "small bounded smooth profile, mildness unresolved."

The mild formulation used here is the identity

$$
V(\tau)
=
e^{TL}V(\tau-T)
-
\int_0^T
e^{\sigma L}\mathbb P\nabla\cdot
(V\otimes V)(\tau-\sigma)\,d\sigma
$$

in $L^\infty$, for every $T>0$. The kernel in the second semigroup estimate is
integrable for $\sigma>0$, and

$$
A
=
C\int_0^\infty e^{-\sigma/2}(1-e^{-\sigma})^{-1/2}\,d\sigma
<\infty.
$$

Choose once and for all

$$
0<\varepsilon_\infty<\frac1A.
$$

Applying the mild formula on $[\tau-T,\tau]$ gives

$$
\|V(\tau)\|_\infty
\le e^{-T/2}M+A M^2,
\qquad
M=\|V\|_{L^\infty}.
$$

Taking the supremum in $\tau$ and then letting $T\to\infty$ gives
$M\le AM^2$. Since $M\le\varepsilon_\infty<1/A$, the only possibility is
$M=0$.

**Lemma PS21.3 -- Zero localized dissipation gives a mean-subtracted
alternative.**
Let $(V_n,P_n)$ have the `PS6` compactness package on $Q_\rho$. Assume

$$
\int_{Q_\rho}|\nabla V_n|^2\,dy\,ds\to0.
$$

Then there are spatial means

$$
c_n(s)=(V_n)_{B_\rho}(s)
$$

such that

$$
V_n-c_n(s)\to0
\quad\text{strongly in }L^3_{\rm loc}(Q_\rho).
$$

If the selected normalization removes the mean, or if $c_n\to0$ in the
topology used on the selected subcylinder, then $V_n\to0$ strongly in
$L^3_{\rm loc}$. Any retained velocity lower bound

$$
\int_{Q_{r_*}}|V_n|^3\ge\eta_v>0
$$

on a compact $Q_{r_*}\Subset Q_\rho$ is then contradicted. If only a combined
lower bound

$$
C_{V_n}(r_*)+D_{P_n}(r_*)\ge\eta_0
$$

is retained, the branch closes only after one also proves

$$
D_{P_n}(r_*)\to0.
$$

If this pressure convergence is missing, the remaining branch is a
pressure-only or harmonic-pressure defect routed to `PS30`. If the means
$c_n(s)$ converge to a nonzero spatially constant velocity, the branch is a
regular background or modulation datum, not a singular Type II core.

**Proof.** For a.e. $s$, Poincare's inequality on $B_\rho$ gives

$$
\|V_n(\cdot,s)-c_n(s)\|_{L^2(B_\rho)}
\le
C\rho\|\nabla V_n(\cdot,s)\|_{L^2(B_\rho)}.
$$

Integrating in $s$ gives strong convergence of $V_n-c_n(s)$ to zero in
$L^2(Q_\rho)$. The compactness package gives $V_n$ uniformly bounded in
$L^\infty_sL^2_y\cap L^2_sH^1_y$, hence uniformly bounded in
$L^{10/3}(Q_\rho)$. The means are also harmless: by Jensen,

$$
|c_n(s)|
\le |B_\rho|^{-1/2}\|V_n(\cdot,s)\|_{L^2(B_\rho)},
$$

so $c_n(s)$ is uniformly bounded on finite cylinders by the
$L^\infty_sL^2_y$ part of the compactness package. Thus
$V_n-c_n(s)$ is uniformly bounded in $L^{10/3}$ on compact subcylinders.
For $Q'\Subset Q_\rho$,

$$
\|V_n-c_n\|_{L^3(Q')}
\le
\|V_n-c_n\|_{L^2(Q')}^{1/6}
\|V_n-c_n\|_{L^{10/3}(Q')}^{5/6}.
$$

The first factor tends to zero and the second is uniformly bounded, proving
strong $L^3_{\rm loc}$ convergence of $V_n-c_n(s)$ to zero.

If the selected gauge or modulation removes $c_n$, or if $c_n\to0$, the
velocity itself converges strongly to zero on compact subcylinders. A retained
velocity activity lower bound on the same subcylinder is therefore impossible.
For a combined $C+D$ lower bound, velocity vanishing is not enough:
pressure oscillation may persist through a harmonic remainder. The conclusion
$D_{P_n}(r_*)\to0$ must be supplied by strong pressure convergence, a
pressure-decay inequality, or a pressure decomposition with controlled
harmonic remainder. Without that input, the pressure branch is routed rather
than closed.

### Specific Estimate

The decisive estimate is either

$$
C(U;z_0,r)+D(P;z_0,r)<\varepsilon_{\rm CKN}
$$

or

$$
\|V\|_{L^\infty}<\varepsilon_\infty.
$$

In a compact Type II single-core window, the additional perturbative closure
test is

$$
\int_{Q_\rho}|\nabla V_n|^2\,dy\,ds\to0
\quad\text{with persistent velocity activity, or with separately proved }
D_{P_n}(r_*)\to0.
$$

### Practical Verification Steps

1. Fix the local cylinder or centered ancient profile to be tested.
2. Normalize the pressure by subtracting spatial means.
3. Compute the selected critical norm.
4. Apply only the perturbative row whose hypotheses are exactly verified.
5. For CKN, close only a selected singular-core cylinder or a same-cylinder
   lower-bound contradiction.
6. For zero dissipation, use velocity activity first; use combined $C+D$ only
   after pressure convergence is proved.

## Estimate Step $B_{\mathrm{PS21}}$

The estimate step is the critical smallness-to-regularity or small ancient
Liouville argument.

## Failure Case

Failure name: unresolved perturbative branch.

Analytic meaning: the velocity appears small, but the pressure, gauge, or
critical topology does not match a perturbative theorem listed in the node
record.

## Refinement Step

Allowed refinements:

1. improve pressure normalization;
2. shrink the cylinder;
3. pass from local smallness to CKN smallness;
4. assign pressure or cutoff defects to `PS30`.

Progress measure: the smallness hypothesis is either verified exactly or
declared absent.

## Data Passed Forward

The next proof step is `PS22`. The data passed forward are

$$
\Gamma_{\mathrm{PS21}}
=
\Gamma_{\mathrm{PS20}}
\cup
\{\text{CKN-small singular core excluded},
\text{ or small bounded mild ancient profile excluded},
\text{ or zero-dissipation velocity-active core excluded},
\text{ or regular background / pressure defect / mildness gap routed forward}\}.
$$

---

# 28. `PS22` -- Stationary Critical-Norm Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a smooth bounded ancient centered profile $(V,P)$ satisfying

$$
\partial_\tau V-\Delta V+\frac12y\cdot\nabla V+\frac12V
+(V\cdot\nabla)V+\nabla P=0,
\qquad
\nabla\cdot V=0.
$$

This node applies only to this exact centered equation. It does not apply to a
forced equation, a cutoff-localized equation, a divergence-defect equation, or
a represented equation with modulation terms unless all additional drift,
source, cutoff, pressure, and divergence-defect terms have vanished or have
been removed by a theorem-preserving normalization.

### Standing Assumptions

The incoming record states that the profile is nonzero by retained activity and is not already excluded
by `PS21`.

### Objects Inspected

Inspect $\partial_\tau V$, the pressure gradient, the spatial profile $W$, and
$\|W\|_{L^3(\mathbb R^3)}$.

### Dependencies Used

The centered equation comes from `PS5`; smoothness from `PS7`; nontriviality
from `PS8`; tight or tail information used to prove $L^3$ comes from
`PS15`.

### Local Obstruction Predicate

$P_{\mathrm{PS22}}$ holds when the selected branch is an $L^3$ stationary
centered profile, because such a profile is excluded by the stationary
self-similar Liouville theorem.

The stationarity test is exact:

$$
\partial_\tau V=0
\quad\text{in }\mathcal D'(\mathbb R^3\times\mathbb R),
$$

or, equivalently after smoothness is known, pointwise on compact cylinders. An
omega-limit, time-average, almost-periodic, or statistically stationary object
is not treated as stationary at this node; those cases are routed to the
compact-hull and Lyapunov/statistical branch `PS29` unless a later theorem
upgrades them to exact stationarity.

If only a hull element is stationary, this node applies to that hull element,
not automatically to the original trajectory. The retained activity must be
shown to be actively attained by the stationary hull element before a zero
conclusion can contradict the branch.

### Local Lemmas to Prove

**Lemma PS22.1 -- Stationarity gives the elliptic Leray profile equation.**
Assume the incoming equation is the exact centered equation, with no residual
source, cutoff term, divergence defect, pressure defect, or modulation drift.
If $V(y,\tau)=W(y)$, then after subtracting a time-dependent pressure gauge,
$W$ and $\Pi$ solve

$$
-\Delta W+\frac12y\cdot\nabla W+\frac12W
+(W\cdot\nabla)W+\nabla\Pi=0,
\qquad
\nabla\cdot W=0.
$$

**Proof.** Substitute $V=W$ into the centered equation and use
$\partial_\tau V=0$. The vector field

$$
F(y)=
\Delta W-\frac12(W+y\cdot\nabla W)-(W\cdot\nabla)W
$$

is independent of $\tau$, while the equation gives
$\nabla_yP(\cdot,\tau)=F$ for a.e. $\tau$. Fix one time $\tau_0$ and define
$\Pi(y)=P(y,\tau_0)$. Then
$\nabla_y(P(\cdot,\tau)-\Pi)=0$, so
$P(y,\tau)-\Pi(y)=c(\tau)$ is spatially constant. Subtracting $c(\tau)$ is an
allowed pressure gauge and yields the displayed stationary Leray profile
equation.

If the incoming equation contains a residual force, cutoff source, divergence
defect, pressure defect, or modulation term
$a(\tau)(V+y\cdot\nabla V)+b(\tau)\cdot\nabla V$, the same substitution gives
a different elliptic equation. In that case this lemma is not invoked until
those terms have been proved zero or removed by a theorem-preserving
normalization.

**Lemma PS22.2 -- The physical pullback is backward self-similar.**
If $V(y,\tau)=W(y)$, then

$$
U(x,t)=(-t)^{-1/2}W\left(\frac{x}{\sqrt{-t}}\right)
$$

with pressure

$$
p(x,t)=(-t)^{-1}\Pi\left(\frac{x}{\sqrt{-t}}\right)
$$

is a backward self-similar Navier--Stokes solution on $t<0$.

**Proof.** Use

$$
y=\frac{x}{\sqrt{-t}},\qquad \tau=-\log(-t),\qquad
U(x,t)=(-t)^{-1/2}V(y,\tau).
$$

For $V(y,\tau)=W(y)$, the identities

$$
\partial_tU=(-t)^{-3/2}\left(\frac12W+\frac12y\cdot\nabla W\right),
\quad
(U\cdot\nabla)U=(-t)^{-3/2}(W\cdot\nabla W),
\quad
\Delta_xU=(-t)^{-3/2}\Delta_yW,
\quad
\nabla_xp=(-t)^{-3/2}\nabla_y\Pi
$$

reduce the physical Navier--Stokes equation on $t<0$ to the stationary Leray
profile equation in Lemma PS22.1. Thus the pullback is backward
self-similar.

**Lemma PS22.3 -- Stationary $L^3$ profiles vanish.**
If $W\in L^3(\mathbb R^3)$ solves the stationary Leray profile equation, then
$W\equiv0$.

**Proof.** The theorem input is: every
$W\in L^3(\mathbb R^3)$ solving the stationary Leray profile equation

$$
-\Delta W+(W\cdot\nabla)W+\nabla P
+\frac12 y\cdot\nabla W+\frac12W=0,\qquad
\nabla\cdot W=0
$$

on $\mathbb R^3$ in distributions, with pressure defined modulo constants and
locally integrable, is zero. The lemma hypotheses are exactly the equation,
divergence constraint, smoothness/local pressure reconstruction, and
whole-space $L^3$ integrability required by this input. A merely local
$L^3_{\rm loc}$ bound is not enough; the whole-space $L^3$ fact must be
attached to this stationary profile itself, for example by uniform tightness,
weighted decay, or an exact endpoint premise. If the branch lacks that tail
source, the proof cannot apply this theorem. Hence, in the verified stationary
critical-norm branch, $W\equiv0$.

**Lemma PS22.4 -- Zero contradicts retained activity.**
For the original stationary branch, if $W=0$, then the pressure-normalized
local CKN mass of the centered profile is zero on every compact cylinder,
contradicting the retained positive lower bound from `PS8`. For a stationary
hull element, the same contradiction is available only after active attainment
has been verified.

**Proof.** With $W=0$, the pressure gradient is zero by the equation and the
pressure is spatially constant after gauge normalization. Both the velocity
$L^3$ term and the pressure oscillation term vanish.

If $W$ is only a stationary hull element, the contradiction is valid only when
the branch has active attainment: there are time shifts along which the
trajectory converges strongly enough to $W$ on a compact cylinder carrying the
retained lower bound. Without active attainment, $W=0$ may be an unrelated hull
limit and does not contradict the original trajectory's activity.

**Lemma PS22.5 -- Stationary branch status is exhaustive.**
After the stationarity and critical-norm checks, the selected branch has
exactly one of the following statuses:

1. $\partial_\tau V\ne0$ in distributions, so the branch is not stationary and
   passes to the later structured or residual checks;
2. $\partial_\tau V=0$ and $W\in L^3(\mathbb R^3)$, so Lemmas PS22.1--PS22.4
   exclude the branch;
3. $\partial_\tau V=0$ but the whole-space $L^3$ hypothesis is missing, so the
   branch is a stationary non-critical-norm obligation routed to `PS25`,
   `PS31`, or `PS34`;
4. a stationary hull element is present but active attainment is missing, so
   the branch is an attainability gap routed forward;
5. stationarity itself is undecided, so the missing compactness or convergence
   input remains an explicit stationarity gap.

**Proof.** The distribution $\partial_\tau V$ is either zero, nonzero, or not
decidable from the recorded topology. In the zero case, the pressure-gauge
reduction of Lemma PS22.1 turns the profile into a stationary Leray solution.
The whole-space $L^3$ predicate is then either verified or missing. If it is
verified, Lemma PS22.3 gives $W=0$. Lemma PS22.4 contradicts retained activity
only for the original stationary branch or for an actively attained stationary
hull element. If $L^3$ is missing, the stationary Liouville theorem is
unavailable. If active attainment is missing, the zero conclusion cannot be
compared with the branch activity. These alternatives are disjoint and exhaust
all possible outcomes of the stationary critical-norm check.

### Specific Estimate

The decisive verification is

$$
\partial_\tau V=0,
\qquad
\|V(\cdot,\tau)\|_{L^3(\mathbb R^3)}=\|W\|_{L^3}<\infty.
$$

It also includes verification that the equation is the exact centered equation
and that retained activity is attached to the stationary profile being
excluded.

### Practical Verification Steps

1. Prove $\partial_\tau V=0$ in distributions and then classically by
   smoothness.
2. Verify that no drift, cutoff source, divergence defect, pressure defect, or
   residual force remains.
3. Fix the pressure gauge so $P(y,\tau)=\Pi(y)$ up to time constants.
4. Verify $W\in L^3(\mathbb R^3)$ using tail information attached to $W$.
5. If $W$ is a hull element, prove active attainment of the retained lower
   bound.
6. Apply the stationary $L^3$ Liouville theorem.
7. Compare the zero conclusion with retained activity.
8. If any gate is missing, record the exact status in Lemma PS22.5 rather than
   applying the stationary theorem.

## Estimate Step $B_{\mathrm{PS22}}$

The estimate step is the verification of stationarity and $L^3$ integrability,
followed by the stationary Liouville theorem and the exhaustive status split
of Lemma PS22.5.

## Failure Case

Failure name: stationary profile without critical-norm closure.

Analytic meaning: the profile is stationary, but the exact $L^3$ hypothesis
needed for the stationary Liouville theorem has not been proved.

## Refinement Step

Allowed refinements:

1. prove a tail estimate using `PS15`;
2. pass to a compact hull limit using `PS13`;
3. assign non-$L^3$ stationary behavior to `PS25` or `PS34`;
4. audit pressure defects in `PS30`.

Progress measure: stationarity and critical integrability are either both
verified or a named residual stationary class is recorded.

## Data Passed Forward

The next proof step is `PS23`. The data passed forward are

$$
\Gamma_{\mathrm{PS22}}
=
\Gamma_{\mathrm{PS21}}
\cup
\{\text{exact stationary }L^3\text{ centered branch excluded},
\text{ or stationary non-}L^3\text{ branch routed},
\text{ or nonstationary branch routed},
\text{ or stationarity/active-attainment undecided}\}.
$$

---

# 29. `PS23` -- Symmetry-Class Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a smooth bounded ancient physical solution $U$ or its centered
representative $V$, together with the pressure gradient.

### Standing Assumptions

The incoming record states that the profile is nonzero, suitable, and not already perturbative or
stationary $L^3$.

### Objects Inspected

Inspect the transformed symmetry group in the selected frame, limiting
axis/period data, swirl, circulation $\Gamma$, weighted local energy, and the
pressure-gradient compatibility with the limiting symmetry.

### Dependencies Used

The smooth ancient profile comes from `PS7`; nontriviality from `PS8`;
stationary branches have been checked by `PS22`; tail and weighted information
comes from `PS15` in the tightness branch.

### Local Obstruction Predicate

$P_{\mathrm{PS23}}$ holds when a recognized exact structured class is present
and the corresponding structured Navier--Stokes theorem hypotheses are
verified.

### Local Lemmas to Prove

**Lemma PS23.1 -- The transformed symmetry group must be tracked.**
Suppose the original solution is invariant under a physical-space group $G$:

$$
u(gx,t)=Dg\,u(x,t).
$$

In a parabolic frame $x=x_n+\lambda_ny$, the rescaled field is invariant under

$$
G_n=
\left\{
y\mapsto
\lambda_n^{-1}\bigl(g(x_n+\lambda_n y)-x_n\bigr):g\in G
\right\}.
$$

An endpoint theorem using symmetry may be invoked only after $G_n$ converges,
in the relevant local sense, to the exact symmetry group required by that
theorem.

**Proof.** For

$$
u_n(y,s)=\lambda_n u(x_n+\lambda_n y,t_n+\lambda_n^2s)
$$

and

$$
h_n(y)=\lambda_n^{-1}\bigl(g(x_n+\lambda_n y)-x_n\bigr),
$$

one has $x_n+\lambda_nh_n(y)=g(x_n+\lambda_n y)$. Therefore

$$
u_n(h_n(y),s)
=
\lambda_nu(g(x_n+\lambda_n y),t_n+\lambda_n^2s)
=Dg\,u_n(y,s).
$$

The rescaled solution is invariant under $G_n$, not necessarily under the
original group $G$. Scaling, recentering, and limiting can move, degenerate, or
erase the visible symmetry, so the limiting group must be recorded.

For axisymmetry about a physical axis $\mathcal A$, record

$$
d_n=\operatorname{dist}(x_n,\mathcal A),
\qquad
\frac{d_n}{\lambda_n}.
$$

If $d_n/\lambda_n\to d_*<\infty$, a limiting axis survives; if
$d_n/\lambda_n\to0$, it passes through the frame origin; if
$d_n/\lambda_n\to\infty$, the axis escapes and no axisymmetric endpoint
theorem is available without a separate limiting-symmetry argument.

For physical periodicity with period $\ell$, record

$$
\ell_n=\frac{\ell}{\lambda_n}.
$$

If $\ell_n\to\ell_*\in(0,\infty)$, finite-period periodicity survives. If
$\ell_n\to\infty$, periodicity disappears locally. If $\ell_n\to0$, a
homogenization or strong-convergence argument is required before any reduced
equation is claimed.

**Lemma PS23.2 -- Axisymmetric no-swirl branch supplies a structured endpoint
record.**
If a smooth ancient solution is axisymmetric in the selected limiting frame,
the limiting axis is recorded, the domain contains the axis when the endpoint
theorem requires it, and $U_\theta=0$ relative to that limiting axis with the
appropriate regularity at $r=0$, then the branch supplies the exact structural
hypotheses needed for the no-swirl regularity endpoint. The branch is not
closed until `PS31` and `PS32` match and apply that endpoint theorem.

**Proof.** In the no-swirl class the vorticity equation reduces to a scalar
parabolic equation for $\omega_\theta/r$ with no vortex-stretching term of the
three-dimensional type. The local record therefore contains the following
endpoint data:

$$
\mathcal L_{\partial_\theta}U=0,\qquad U_\theta=0,\qquad
\nabla\cdot U=0,
$$

together with the limiting axis, axis regularity at $r=0$ when the axis is in
the domain, the solution class, pressure convention, domain, and local energy
information inherited from `PS7` and `PS30`. These are exactly the items that
`PS31` must compare with the selected axisymmetric no-swirl theorem. If the
axis has escaped in the selected frame, this lemma does not supply no-swirl
endpoint data.
If the theorem hypotheses match, `PS32` translates its regularity conclusion
back to the singular-entry variables and closes the branch only when the
regularity conclusion applies to the same selected active branch carrying the
retained singular-core activity. If the theorem applies only to a regular
subregion, the remaining activity is returned to active-frame selection. If
any theorem hypothesis is missing, the conclusion of this lemma is the missing
hypothesis, not a closure claim.

**Lemma PS23.3 -- Controlled circulation supplies scalar endpoint hypotheses
data.**
If the axisymmetric swirl satisfies a verified bound such as

$$
\Gamma=rU_\theta\in L^\infty_tL^q_x
$$

for an admissible $q$, or a pointwise bound

$$
\sup r^\gamma |U_\theta|<\infty,
$$

then, relative to the recorded limiting axis with $r$ equal to distance from
that axis, the branch records the swirl equation

$$
\partial_t\Gamma+U\cdot\nabla\Gamma
=
\Delta\Gamma-\frac{2}{r}\partial_r\Gamma
$$

and sends the displayed quantitative bound to the axisymmetric Liouville or
regularity theorem hypotheses in `PS31`.

**Proof.** The displayed scalar equation follows from the angular component of
Navier--Stokes in cylindrical coordinates around the limiting axis. The
operator has the absorbing structure at that axis. The branch record contains
the exact exponent $q$ or weight $\gamma$, the time interval, the domain, and
the behavior at the axis $r=0$. If the selected frame does not contain a
limiting axis, or if the axis has escaped, this scalar endpoint is not
available. `PS31` compares
these data with the selected controlled-swirl theorem. Only after that match,
and only when the theorem conclusion covers the selected active branch itself,
does `PS32` use the theorem conclusion, such as regularity, vanishing, or a
constant ancient state, to close the retained singular branch. Without the
exact exponent, axis condition, pressure convention, solution class, and
same-branch activity link, the controlled-swirl theorem is not invoked as an
exclusion.

**Lemma PS23.4 -- Weighted decay is a structured hypothesis.**
If

$$
\sup_\tau\|V(\tau)\|_{L^\infty(\mathbb R^3)}\le M
$$

and

$$
\sup_\tau\int_{\mathbb R^3}(1+|y|^2)^m|V(y,\tau)|^2\,dy<\infty
$$

for an admissible $m>0$, then the branch has uniform critical tails:

$$
\sup_\tau\int_{|y|>R}|V(y,\tau)|^3\,dy
\le
M(1+R^2)^{-m}
\sup_\tau\int_{\mathbb R^3}(1+|y|^2)^m|V(y,\tau)|^2\,dy
\to0.
$$

The branch is assigned to the weighted-decay endpoint hypotheses or to the
uniform-tightness endpoint only after the boundedness or another explicit
interpolation input has been verified.

**Proof.** On $|y|>R$,

$$
|V|^3\le \|V(\tau)\|_\infty |V|^2
$$

and

$$
|V|^2\le(1+R^2)^{-m}(1+|y|^2)^m|V|^2.
$$

Combining the inequalities and taking the supremum in $\tau$ gives the
displayed estimate. Weighted $L^2$ alone does not give a critical $L^3$ tail.

### Specific Estimate

The decisive estimates are the exact symmetry identity

$$
\mathcal L_X U=0
$$

for the selected infinitesimal symmetry generator $X$, and the quantitative
bound on $\Gamma$ or the weighted norm required by the selected structured
theorem.

### Practical Verification Steps

1. State the physical symmetry group and compute the transformed groups $G_n$
   in the selected frame.
2. For axisymmetry, record $\operatorname{dist}(x_n,\mathcal A)/\lambda_n$ and
   the limiting axis status.
3. For periodicity, record $\ell/\lambda_n$ and the limiting period regime.
4. If axisymmetric, write the velocity in cylindrical components around the
   limiting axis.
5. Verify the no-swirl, controlled-swirl, periodic-circulation, or weighted
   hypothesis exactly in the selected frame.
6. Apply a structured theorem only after the limiting symmetry matches its
   hypotheses.

## Estimate Step $B_{\mathrm{PS23}}$

The estimate step verifies the symmetry identity and the quantitative
structured bound.

## Failure Case

Failure name: unresolved structured profile.

Analytic meaning: exact symmetry appears present, but the scalar, swirl,
periodic, or weighted estimate needed for endpoint routing is missing.

## Refinement Step

Allowed refinements:

1. pass to the quotient under the exact symmetry;
2. derive the swirl or weighted equation;
3. assign coherent time-dependent symmetry to `PS24`;
4. assign hidden normalization defects to `PS26`.

Progress measure: the structured theorem's hypotheses are either matched or the
branch is assigned to a more precise structured node.

## Data Passed Forward

The next proof step is `PS24`. The data passed forward are

$$
\Gamma_{\mathrm{PS23}}
=
\Gamma_{\mathrm{PS22}}
\cup
\{\text{transformed symmetry group }G_n,
\text{ limiting symmetry status},
\text{ axis/period/weight data},
\text{ structured theorem hypotheses matched, absent, or unresolved}\}.
$$

---

# 30. `PS24` -- Relative Equilibrium or Coherent-Structure Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a smooth ancient centered profile whose time dependence is
represented by a finite-dimensional group action on a fixed spatial profile.

### Standing Assumptions

The incoming record states that the profile is nonzero, non-perturbative, and
not already routed through an exact static symmetry theorem.

### Objects Inspected

Inspect the modulation generator, the co-moving profile, pressure gauge,
vorticity, Bernoulli-type quantities, and annular fluxes created by the
co-moving transport terms.

### Dependencies Used

The centered equation comes from `PS5`; pressure gauge from `PS4`; structured
symmetry information from `PS23`; retained activity from `PS8`.

### Local Obstruction Predicate

$P_{\mathrm{PS24}}$ holds when all time dependence is carried by a symmetry
action and the resulting co-moving equation remains outside the conclusions of
the preceding structured theorems.

### Local Lemmas to Prove

**Lemma PS24.1 -- Rotational relative equilibria satisfy a co-rotating
stationary equation.**
If

$$
V(y,\tau)=Q(\tau)W(Q(\tau)^Ty),
\qquad
Q'(\tau)=\Omega_{\rm sp} Q(\tau),
$$

with $Q(\tau)\in SO(3)$, $\Omega_{\rm sp}^T=-\Omega_{\rm sp}$, and
$\Omega_{\rm sp}$ constant, define the body-frame generator

$$
\Omega
=Q(\tau)^T\Omega_{\rm sp}Q(\tau).
$$

This matrix is independent of $\tau$. After setting $z=Q(\tau)^Ty$, the
profile $W$ satisfies

$$
-\Delta W+\frac12z\cdot\nabla W+\frac12W
+(W\cdot\nabla)W
+\Omega W-(\Omega z)\cdot\nabla W+\nabla\Pi=0,
\qquad
\nabla\cdot W=0.
$$

**Proof.** First,

$$
\frac{d}{d\tau}(Q^T\Omega_{\rm sp}Q)
=
Q^T\Omega_{\rm sp}^T\Omega_{\rm sp}Q
+
Q^T\Omega_{\rm sp}^2Q
=0,
$$

because $\Omega_{\rm sp}^T=-\Omega_{\rm sp}$. Thus
$\Omega=Q^T\Omega_{\rm sp}Q$ is constant and skew-symmetric. Differentiate
$z=Q(\tau)^Ty$ to get $\partial_\tau z=-\Omega z$.
The velocity derivative transforms as

$$
Q^T\partial_\tau V
=
\Omega W-(\Omega z)\cdot\nabla W.
$$

The remaining terms are orthogonally covariant:

$$
Q^T\Delta_y V=\Delta_zW,
\qquad
Q^T((V\cdot\nabla_y)V)=(W\cdot\nabla_z)W,
$$

$$
Q^T(y\cdot\nabla_yV)=z\cdot\nabla_zW,
\qquad
Q^T\nabla_yP=\nabla_z\Pi,
$$

where $\Pi(z,\tau)=P(Q(\tau)z,\tau)$. Substituting these identities into the
centered Navier--Stokes equation and multiplying by $Q(\tau)^T$ gives the
displayed equation with $\nabla_z\Pi(\cdot,\tau)$. Since every velocity and
drift term in that equation is independent of $\tau$, the pressure gradient is
independent of $\tau$. Fixing one time and subtracting the remaining
time-dependent spatial constant gives a pressure representative $\Pi(z)$ and
the displayed stationary co-rotating equation.

If the body-frame generator $Q^TQ'$ is not constant, the co-rotating frame
contains time-dependent modulation terms and the profile is not stationary in
this sense. That case is routed to modulation/coherent-defect analysis with
the extra terms retained.

**Lemma PS24.2 -- Physical Galilean normalization and centered translations
are different.**
The physical Galilean transform

$$
\widetilde u(x,t)=u(x-ct,t)+c,
\qquad
\widetilde p(x,t)=p(x-ct,t)
$$

preserves the physical Navier--Stokes equation. A centered-variable
translation

$$
V(y,\tau)=W(y-a(\tau))
$$

does not behave as an ordinary Galilean symmetry of the centered equation; it
creates the additional transport coefficient

$$
\left(\frac12a(\tau)-a'(\tau)\right)\cdot\nabla W.
$$

**Proof.** For the physical transform, one has

$$
\partial_t\widetilde u
=
(\partial_tu)(x-ct,t)-c\cdot\nabla u(x-ct,t),
$$

and

$$
(\widetilde u\cdot\nabla)\widetilde u
=(u+c)\cdot\nabla u(x-ct,t).
$$

The two terms involving $c\cdot\nabla u$ cancel, so the material derivative
pulls back exactly. The pressure gradient and Laplacian also transform by
pullback, hence Navier--Stokes is preserved. No pressure term $-c\cdot x$ is
added; that term would change the pressure gradient and break the
cancellation.

For centered variables, set $z=y-a(\tau)$ and $V(y,\tau)=W(z)$. Then

$$
\partial_\tau V=-a'(\tau)\cdot\nabla W,
\qquad
\frac12y\cdot\nabla V
=
\frac12(z+a(\tau))\cdot\nabla W.
$$

Thus the centered equation contains the extra transport
$\left(\frac12a(\tau)-a'(\tau)\right)\cdot\nabla W$. If this coefficient is
zero, the translation may correspond to a change of self-similar center. If it
is nonzero but controlled, it is a coherent modulation term. If it is
uncontrolled, the branch is routed to modulation defect. Centered translations
are not identified with Galilean invariance unless the branch has been pulled
back to physical variables and the exact Galilean transform above has been
verified.

**Lemma PS24.3 -- Non-small co-moving flux creates retained local activity.**
Let $F_{\rm coh}$ denote the additional transport or Coriolis-type term in the
co-moving stationary equation. Assume the localized energy identity on
$A_R\times I$ is available, all pressure, cutoff, and source terms except
$F_{\rm coh}$ are controlled, and the annulus has a finite parabolic cover
with constant $C_{\rm cover}$. If the localized annular flux satisfies

$$
\left|\iint_{A_R\times I} F_{\rm coh}\cdot W\,\chi_R\right|
\ge \eta
$$

with $\eta>C_{\rm cover}\varepsilon_{\rm CKN}$, then the branch contains a
retained local concentration component and is assigned to active-frame
analysis rather than a coherent-structure endpoint route.

**Proof.** Insert the co-moving equation into the localized energy identity on
the annulus. Cover $A_R\times I$ by finitely many parabolic cylinders whose
enlargements remain in the controlled region. If every cylinder in the cover
satisfies

$$
C+D<\varepsilon_{\rm CKN},
$$

and all pressure, cutoff, and source terms except $F_{\rm coh}$ are controlled,
then the perturbative estimates from `PS21` bound the coherent flux by
$C_{\rm cover}\varepsilon_{\rm CKN}$, with the constant depending only on the
cover, cutoff, and coefficient bounds. This finite-cover perturbative flux
estimate is part of the lemma's hypothesis; it is not inferred from flux
smallness alone. Therefore, if the flux is bounded below by
$\eta>C_{\rm cover}\varepsilon_{\rm CKN}$, at least one cylinder in the cover
has non-small CKN quantity. Rescaling that cylinder gives an active local
concentration frame. If any pressure, cutoff, or modulation term in the
localized energy identity is uncontrolled, or if the finite-cover flux
estimate has not been proved in the required topology, the conclusion is not
active-frame routing; the missing or uncontrolled term is entered into the
`PS30` defect vector.

**Lemma PS24.4 -- Small co-moving flux produces a stationary or perturbative
endpoint candidate.**
If the coherent transport flux is small on every localization annulus and the
remaining profile satisfies the stationary or perturbative endpoint hypotheses,
then the branch is assigned to the `PS21` or `PS22` endpoint hypotheses.

**Proof.** With the extra co-moving transport controlled as a perturbative
forcing term in the exact topology required by the endpoint theorem, the local
equation is a small perturbation of the stationary centered equation. In local
compactness arguments this topology is usually $L^1_tW^{-1,3/2}_x$ or
$L^1_tH^{-m}_x$ with $m\ge3$, not automatically $H^{-1}$. The data passed
forward are the data package

$$
(\text{co-moving equation},\ \|F_{\rm coh}\|_{\mathcal X_{\rm endpoint}},
\text{pressure gauge},\text{smallness or stationarity hypothesis}).
$$

`PS31` must still verify that this package matches the perturbative or
stationary endpoint theorem. If the match is complete, `PS32` extracts the
regularity or zero-profile contradiction. If the match is incomplete, the
missing smallness, tail, pressure, or topology hypothesis is recorded rather
than hidden inside the phrase "small co-moving flux."

### Specific Estimate

The decisive local estimate is the annular flux bound

$$
\left|\iint_{A_R\times I} F_{\rm coh}\cdot W\,\chi_R\right|
$$

relative to the perturbative threshold and the retained local CKN lower bound.

### Practical Verification Steps

1. Identify the finite-dimensional group action and its generator.
2. Derive the co-moving profile equation.
3. Normalize the pressure in the co-moving frame.
4. Estimate added transport terms in the exact endpoint topology, typically
   $L^1W^{-1,3/2}$ or $L^1H^{-m}$, or by annular flux.
5. In the small-flux case, assign the branch to a perturbative or stationary
   endpoint hypotheses.
6. If the flux is non-small, register the induced active concentration branch.

## Estimate Step $B_{\mathrm{PS24}}$

The estimate step is the co-moving equation and local coherent-flux estimate.

## Failure Case

Failure name: unresolved relative equilibrium.

Analytic meaning: the time dependence is coherent, but the co-moving
transport terms are not controlled by the recorded local estimates.

## Refinement Step

Allowed refinements:

1. change to the exact co-moving frame;
2. absorb physical constant drifts by a verified Galilean normalization;
3. assign non-small flux to active-frame analysis;
4. assign degenerate stationary-family behavior to `PS25`.

Progress measure: the coherent structure is either stationary/perturbative,
active, or assigned to a degenerate structured class.

## Data Passed Forward

The next proof step is `PS25`. The data passed forward are

$$
\Gamma_{\mathrm{PS24}}
=
\Gamma_{\mathrm{PS23}}
\cup
\{\text{group action }Q(\tau)\text{ or translation/modulation }a(\tau),
\text{ co-moving equation with all extra terms},
\text{ coherent-flux status},
\text{ small-force topology},
\text{ active-frame routing if flux is non-small}\}.
$$

---

# 31. `PS25` -- Degenerate Structured Direction Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a stationary centered profile or an ancient profile whose hull
contains an actively attained stationary profile.

### Standing Assumptions

The incoming record states that all previous named endpoint routes or
reductions have been recorded and that the branch still carries positive
pressure-normalized local CKN mass.

### Objects Inspected

Inspect the stationary profile equation, the topology and pressure gauge of a
stationary family, the tangent vectors $Z_j$, the linearized operator, true
symmetry tangents, modulation/frame directions, and active attainment sequence.

### Dependencies Used

Stationary checks come from `PS22`; hull compactness from `PS13`; coherent
structure reduction from `PS24`; active mass from `PS8`.

### Local Obstruction Predicate

$P_{\mathrm{PS25}}$ holds when a stationary-family or kernel direction has not
been classified as a true symmetry, modulation/frame direction, genuine
stationary-family tangent, unresolved kernel direction, or actively attained
endpoint profile.

### Local Lemmas to Prove

**Lemma PS25.1 -- A stationary family gives linearized stationary solutions.**
Assume $a\mapsto W_a$, $a\in\mathbb R^m$, is $C^1$ in a topology strong enough
to differentiate the stationary centered equation, for example
$C^2_{\rm loc}$, and that $a\mapsto\Pi_a$ is differentiable modulo constants.
If $a\mapsto(W_a,\Pi_a)$ is a stationary family and $W=W_0$, then each tangent
vector

$$
Z_j=\partial_{a_j}W_a|_{a=0}
$$

satisfies

$$
-\Delta Z_j+\frac12y\cdot\nabla Z_j+\frac12Z_j
+(Z_j\cdot\nabla)W+(W\cdot\nabla)Z_j+\nabla\pi_j=0,
\qquad
\nabla\cdot Z_j=0.
$$

**Proof.** The topology assumption permits differentiating the Laplacian,
centered drift, pressure gradient, and product $(W_a\cdot\nabla)W_a$ locally.
The pressure-gauge assumption gives a derivative
$\pi_j=\partial_{a_j}\Pi_a|_{a=0}$ modulo constants. Differentiating the
stationary centered equation with respect to $a_j$ at $a=0$ gives the
displayed linearized equation, and differentiating $\nabla\cdot W_a=0$ gives
$\nabla\cdot Z_j=0$.

**Lemma PS25.2 -- True symmetry directions and modulation directions are
different.**
True symmetries of the stationary centered Leray equation include rotations

$$
W_Q(y)=QW(Q^Ty),
\qquad Q\in SO(3),
$$

with pressure $\Pi_Q(y)=\Pi(Q^Ty)$, and pressure gauges
$\Pi\mapsto\Pi+\text{constant}$. Their tangents are symmetry degeneracies.
Ordinary translations, dilations, Galilean boosts, and moving-center
normalizations are not automatic symmetries of the fixed stationary centered
equation; they are frame or modulation directions unless one proves that the
differentiated family remains inside the exact stationary solution set.

**Proof.** Rotations commute with $\Delta$, preserve $y\cdot\nabla$, preserve
divergence, and rotate the nonlinear and pressure-gradient terms covariantly.
Pressure gauges do not change the velocity equation. In contrast, translating
$W(y)$ to $W(y-a)$ changes the centered drift term into one containing
$\frac12a\cdot\nabla W$, and dilating $W$ changes the fixed coefficient in the
centered drift. Thus translations and dilations are not symmetries of the
stationary centered equation. They may appear as modulation or frame
directions, but then the extra equation terms must be retained and routed to
`PS24` or `PS26`.

**Lemma PS25.3 -- Exact motion along a stationary family is stationary only
modulo true symmetries.**
If an ancient trajectory satisfies $V(\tau)=W_{a(\tau)}$ with
$a\in W^{1,1}_{\rm loc}$ for a stationary family whose parametrization is an
immersion modulo pressure gauges and true velocity symmetries, then
$a'(\tau)=0$ for a.e. $\tau$ after quotienting those true symmetries, and $V$
is stationary in the quotient.

**Proof.** Substitute $V(\tau)=W_{a(\tau)}$ into the centered equation. Since
each $W_a$ solves the stationary equation, the only remaining term is

$$
\sum_{j=1}^m a_j'(\tau)\partial_{a_j}W_{a(\tau)}.
$$

The immersion hypothesis says that this tangent combination vanishes only when
$a'(\tau)=0$, after quotienting pressure gauges and true velocity symmetries.
If the tangent combination lies in a rotation generator, the branch is a
relative equilibrium handled by `PS24`. If it lies in a frame/modulation
direction, the equation is not the exact stationary equation and is routed to
`PS26`.

**Lemma PS25.4 -- Active attainment is required for stationary endpoint use.**
If a compact hull contains a stationary profile lying on a nontrivial
stationary family, then it can be sent to endpoint contradiction only in the
actively attained case. Active attainment means there are time shifts
$\tau_n$, a compact cylinder $K$, and $\eta>0$ such that

$$
V(\cdot,\cdot+\tau_n)\to W
\quad\text{locally smoothly}
$$

and

$$
\iint_K|V(y,\tau+\tau_n)|^3\,dy\,d\tau\ge\eta.
$$

Then strong convergence gives

$$
\iint_K|W|^3\,dy\,d\tau\ge\eta.
$$

If active attainment and the exact stationary endpoint hypotheses hold, the
profile is routed to `PS22`/`PS31`. If the hull contains the stationary profile
without activity, there is no contradiction. If activity is present but an
endpoint hypothesis is missing, the branch is routed to `PS33` or the relevant
tail/attainability node.

**Proof.** The first two displayed conditions define convergence and the lower
bound on the same compact cylinder. Strong local convergence passes the lower
bound to $W$. If $W$ also satisfies the exact stationary equation, compatible
pressure gauge, whole-space $L^3$ or other endpoint tail hypothesis, and
normalization requirements, the endpoint theorem can be matched downstream. If
any endpoint hypothesis is missing, the profile is not silently excluded. If
activity is missing, hull containment alone is insufficient for contradiction.

### Specific Estimate

The decisive verification is the linearized stationary equation for each
$Z_j$ and
the classification

$$
Z_j\in
\{\text{true symmetry tangent},
\text{ modulation/frame tangent},
\text{ genuine stationary-family tangent},
\text{ unresolved kernel direction}\}.
$$

No Fredholm, spectral, finite-dimensionality, isolation, or removability claim
is made unless a separate theorem supplies it.

### Practical Verification Steps

1. Produce the stationary-family parametrization.
2. Verify differentiability topology and compatible pressure gauges.
3. Differentiate the stationary profile equation.
4. Classify each tangent as true symmetry, modulation/frame direction, genuine
   stationary-family tangent, or unresolved kernel direction.
5. Verify active attainment of the stationary profile by local CKN mass before
   using endpoint contradiction.
6. Assign stationary $L^3$ cases to `PS22` and nonclosed attainability cases to
   `PS33` through the later endpoint checks.

## Estimate Step $B_{\mathrm{PS25}}$

The estimate step is the linearized stationary equation plus active-attainment
verification.

## Failure Case

Failure name: unresolved degenerate structured direction.

Analytic meaning: a nontrivial stationary-family or linearized kernel appears,
but it is not classified as a true symmetry, modulation direction, genuine
family tangent, unresolved kernel, stationary endpoint datum, or endpoint
attainability problem.

## Refinement Step

Allowed refinements:

1. quotient out true symmetry-generated tangent vectors;
2. strengthen hull convergence to active attainment;
3. prove the stationary profile satisfies the `PS22` critical norm;
4. route modulation/frame tangents to `PS24` or `PS26`;
5. assign nonclosed attainability to `PS33`.

Progress measure: every degenerate tangent direction is assigned to true
symmetry, modulation, genuine-family, unresolved-kernel, stationary, or
endpoint-realization category.

## Data Passed Forward

The next proof step is `PS26`. The data passed forward are

$$
\Gamma_{\mathrm{PS25}}
=
\Gamma_{\mathrm{PS24}}
\cup
\{\text{stationary family topology and pressure gauge},
\text{ linearized stationary equations for }Z_j,
\text{ classification of each }Z_j:
\text{ true symmetry / modulation direction / genuine family / unresolved kernel},
\text{ active attainment status},
\text{ endpoint hypothesis status}\}.
$$

---

# 32. `PS26` -- Symmetry Action and Normalization Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a selected profile modulo the Navier--Stokes symmetry
group, together with a pressure class modulo time-dependent functions.

### Standing Assumptions

The incoming record states that the profile is smooth locally and carries retained activity, and that
all physical structured branches up to `PS25` have been classified.

### Objects Inspected

Inspect center, scale, rotation, drift, pressure mean, and the orthogonality
conditions used to fix them.

### Dependencies Used

Center-scale data come from `PS2` and `PS3`; pressure normalization from `PS4`;
modulation terms from `PS5`; symmetry tangent data from `PS25`.

### Local Obstruction Predicate

$P_{\mathrm{PS26}}$ holds when a remaining symmetry action changes the
coordinates of the profile without changing the physical Navier--Stokes
branch.

### Local Lemmas to Prove

**Lemma PS26.1 -- Navier--Stokes transformations preserve the local equation.**
Translations, rotations, parabolic scalings, and Galilean transforms preserve
the incompressible Navier--Stokes equation after the corresponding pressure
change; adding a function of time to the pressure does not change the velocity
equation.

**Proof.** Apply the chain rule. Rotations preserve divergence, Laplacian, and
the tensor contraction in the convection term. Parabolic scaling preserves the
relative scaling of $\partial_tu$, $\Delta u$, $(u\cdot\nabla)u$, and
$\nabla p$. The standard Galilean transform preserves the equation with the
pressure pulled back as in Lemma PS24.2; no linear pressure term is needed.
Pressure additions depending only on time have zero spatial gradient.

**Lemma PS26.2 -- Nondegenerate slices fix the local group action.**
If the slice Jacobian is invertible at a profile $V$, then nearby group
parameters are uniquely determined by the conditions $\Phi_\alpha(g\cdot V)=0$.

**Proof.** Let

$$
F(g,V)=(\Phi_\alpha(g\cdot V))_{\alpha=1}^m.
$$

The verified slice condition is

$$
\det D_gF(e,V)\ne0.
$$

The finite-dimensional implicit-function theorem gives a neighborhood of
$V$ and a unique $C^1$ parameter map $V'\mapsto g(V')$ satisfying
$F(g(V'),V')=0$. Thus nearby group parameters are determined by the slice
conditions, and no additional normalization freedom remains inside that
neighborhood.

**Lemma PS26.3 -- Pressure gauges are fixed by spatial mean normalization.**
On every ball $B_R$, the condition

$$
\int_{B_R}P(y,\tau)\,dy=0
$$

fixes the pressure representative up to choices on different balls and removes
the time-dependent additive ambiguity on that ball.

**Proof.** Let $P_1-P_2=c(\tau)$ and impose

$$
\int_{B_R}P_1(y,\tau)\,dy=\int_{B_R}P_2(y,\tau)\,dy=0.
$$

Subtracting the two identities gives $c(\tau)|B_R|=0$ for a.e. $\tau$, hence
$c(\tau)=0$ a.e.

**Lemma PS26.4 -- Normalization defects return to the generating node.**
If the hidden parameter is a center, scale, pressure, rotation, or Galilean
drift, then the branch is not a new PDE case; it is assigned respectively to `PS2`,
`PS3`, `PS4`, `PS23`, or `PS24` for the corresponding normalization.

**Proof.** Each listed defect is one of the previously declared
coordinate choices. Repeating the corresponding local selection changes the
representation but not the underlying solution.

### Specific Estimate

The decisive verification is the slice nondegeneracy condition

$$
\left|\det\left(\partial_{g_\beta}\Phi_\alpha(g\cdot V)\right)\right|
\ge c_0>0
$$

on the selected local branch.

### Practical Verification Steps

1. List the remaining symmetry parameters.
2. State the slice functionals used to fix them.
3. Prove the slice Jacobian is bounded away from zero.
4. Normalize the pressure by spatial means.
5. Assign any failed slice to the node that chooses that coordinate.

## Estimate Step $B_{\mathrm{PS26}}$

The estimate step is the slice nondegeneracy and pressure-mean verification.

## Failure Case

Failure name: hidden normalization defect.

Analytic meaning: the apparent branch changes under a symmetry action rather
than under genuine Navier--Stokes dynamics.

## Refinement Step

Allowed refinements:

1. choose a different nondegenerate slice;
2. recenter or rescale the active frame;
3. reset the pressure mean;
4. assign rotational or Galilean defects to `PS23` or `PS24`.

Progress measure: the group parameter is fixed or assigned to its generating
normalization node.

## Data Passed Forward

The next proof step is `PS27`. The data passed forward are

$$
\Gamma_{\mathrm{PS26}}
=
\Gamma_{\mathrm{PS25}}
\cup
\{\text{normalization slice and pressure gauge fixed}\}.
$$

---

# 33. `PS27` -- Symmetry-Breaking Stability Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is the perturbation $w$ to a structured profile in a normalized
local frame.

### Standing Assumptions

The incoming record states that $W$ is already classified by `PS23`--`PS26` and that $w$ is small in the
local norm used for the stability estimate.

### Objects Inspected

Inspect $w$, its vorticity, the projected linearized equation, nonlinear
commutators, and the orthogonality conditions against symmetry modes.

### Dependencies Used

The structured profile comes from `PS23` or `PS24`; quotient conditions from
`PS26`; defect assignment from `PS30` if stress or pressure terms remain.

### Local Obstruction Predicate

$P_{\mathrm{PS27}}$ holds when $w$ remains nonzero and no local coercive
estimate forces decay, regularity, or assignment to a named defect.

### Local Lemmas to Prove

**Lemma PS27.1 -- Transverse perturbation equation.**
After writing $V=W+w$, the perturbation satisfies

$$
\partial_\tau w
=
L_Ww-\mathbb P\nabla\cdot(w\otimes w)+F_{\rm mod},
$$

where $L_W$ is the linearized centered Navier--Stokes operator around $W$ and
$F_{\rm mod}$ contains only normalized modulation errors.

**Proof.** Substitute $V=W+w$ into the centered equation and subtract the
equation for $W$. The linear terms in $w$ form $L_Ww$, the quadratic remainder
is $-\mathbb P\nabla\cdot(w\otimes w)$, and the remaining terms come from
time-dependent frame parameters.

**Lemma PS27.2 -- Coercive transverse gap produces perturbative control.**
The lemma hypotheses contain a local coercive functional $\mathcal N(w)$ and constants
$\kappa>0$, $C<\infty$ such that

$$
\frac{d}{d\tau}\mathcal N(w)
+\kappa\mathcal N(w)
\le
C\mathcal N(w)^{3/2}+\|F_{\rm mod}\|_{H^{-1}}^2
$$

on the selected window. The branch records thresholds
$\delta_N,\delta_F>0$ such that

$$
\sup_I\mathcal N(w)\le\delta_N,\qquad
\|F_{\rm mod}\|_{L^2_IH^{-1}_x}\le\delta_F
$$

imply that $w$ decays or remains perturbative.

**Proof.** Choose
$\delta_N=(\kappa/(2C))^2$. Then
$C\mathcal N(w)^{3/2}\le(\kappa/2)\mathcal N(w)$ whenever
$\mathcal N(w)\le\delta_N$. The differential inequality becomes

$$
\frac{d}{d\tau}\mathcal N(w)+\frac{\kappa}{2}\mathcal N(w)
\le \|F_{\rm mod}\|_{H^{-1}}^2.
$$

Gronwall's inequality controls $\mathcal N(w)$ by its initial size and the
modulation-error norm. Under the displayed thresholds, the transverse
component remains in the perturbative regime. The branch is excluded only
after the perturbative endpoint hypotheses,
including the equation, pressure, boundary, and source conditions, are matched
in `PS31` and contradicted in `PS32`. If those endpoint conditions are not
available, this lemma supplies a controlled transverse estimate but not a final
exclusion.

**Lemma PS27.3 -- Failure of the gap is a named defect.**
If the transverse estimate fails because $F_{\rm mod}$, pressure, stress, or
frequency terms do not vanish, then the branch is not an unclassified
symmetry-breaking branch; it is assigned to `PS30`.

**Proof.** Each failure term is a concrete distributional residue in the
perturbation equation. The defect audit records exactly such residues by
channel.

### Specific Estimate

The decisive estimate is

$$
\frac{d}{d\tau}\mathcal N(w)
+\kappa\mathcal N(w)
\le
C\mathcal N(w)^{3/2}+\mathrm{Err}(\tau),
\qquad
\int_I\mathrm{Err}(\tau)\,d\tau\ll1.
$$

### Practical Verification Steps

1. Fix the quotient orthogonality conditions.
2. Derive the perturbation equation.
3. Identify the coercive functional.
4. Prove the linearized gap on the transverse subspace.
5. Absorb nonlinear and modulation errors.
6. Assign any unabsorbed term to the defect audit.

## Estimate Step $B_{\mathrm{PS27}}$

The estimate step is the transverse coercive stability inequality.

## Failure Case

Failure name: uncontrolled symmetry-breaking mode.

Analytic meaning: after quotienting symmetries, a nonzero transverse component
survives without perturbative control.

## Refinement Step

Allowed refinements:

1. strengthen the quotient orthogonality;
2. change the coercive functional;
3. absorb modulation errors through `PS26`;
4. assign remaining residues to `PS30`.

Progress measure: the transverse mode is either controlled or converted into a
named defect.

## Data Passed Forward

The next proof step is `PS28`. The data passed forward are

$$
\Gamma_{\mathrm{PS27}}
=
\Gamma_{\mathrm{PS26}}
\cup
\{\text{transverse stability status}\}.
$$

---

# 34. `PS28` -- Transition-Action or Finite-Cost Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a modulated Type II profile with scale $\lambda(\tau)$ and
selected nonzero core on renormalized windows.

### Standing Assumptions

The incoming record states that the scale variable is positive and absolutely continuous on selected
windows, and that the branch has genuine collapse if
$\lambda(\tau_j)\to0$ along a terminal sequence.

### Objects Inspected

Inspect $\log\lambda$, $a(\tau)$, selected-window thickness, compactness
estimates, and autonomous modulation limits.

### Dependencies Used

Type II scale data come from `PS10` and `PS11`; normalized modulation from
`PS26`; compactness defects are assigned to `PS30`.

### Local Obstruction Predicate

$P_{\mathrm{PS28}}$ holds when finite modulation cost is asserted together with
genuine scale collapse and positive selected mass.

### Local Lemmas to Prove

**Lemma PS28.1 -- Genuine scale collapse has infinite logarithmic variation.**
If $\lambda(\tau)>0$ is absolutely continuous, $\lambda(\tau_0)>0$, and
$\lambda(\tau_j)\to0$, then

$$
\int_{\tau_0}^{\tau_j}|a(\tau)|\,d\tau
\ge
\left|\log\lambda(\tau_j)-\log\lambda(\tau_0)\right|
\to\infty.
$$

**Proof.** Since $a=-\partial_\tau\log\lambda$, the inequality is the total
variation bound for an absolutely continuous function.

**Lemma PS28.2 -- Finite-cost dynamics cannot realize the genuine-collapse
predicate.**
If $\mathcal C_{[\tau_0,\infty)}<\infty$, then $\log\lambda(\tau)$ has a finite
limit along terminal times. Hence genuine collapse $\lambda\to0$ is impossible;
any claimed nonzero collapsed core contradicts the scale-collapse predicate
through Lemma PS28.1. If a positive selected core remains but the scale has a
positive terminal limit, the branch is no longer a finite-cost collapse branch;
it is a scale-rigid terminal branch routed to `PS29`.

**Proof.** Finite $L^1$ norm of $a$ makes $\log\lambda$ a Cauchy function on
terminal tails. It therefore converges to a finite number and cannot tend to
$-\infty$. Since $\lambda>0$, genuine collapse would require
$\log\lambda(\tau_j)\to-\infty$ along the selected terminal sequence, which is
incompatible with finite variation. The only remaining finite-cost possibility
is a noncollapsed terminal scale, and that is precisely the scale-rigid data
sent forward rather than silently excluded.

**Lemma PS28.3 -- Non-finite-cost exits are named alternatives.**
If finite cost fails, then the branch is classified as thin-drift,
autonomous-modulation defect, compactness defect, or scale-rigid terminal
state according to whether selected fixed-length windows, autonomous limits, or
compact estimates fail.

**Proof.** On selected terminal windows either one has enough thickness and
compact estimates to pass to an autonomous limit, or the failure is exactly
loss of window thickness, modulation convergence, or compactness. These are
the named local alternatives.

### Specific Estimate

The decisive estimate is

$$
\int_{\tau_0}^{\tau_1}|a(\tau)|\,d\tau
\ge
\left|\log\frac{\lambda(\tau_1)}{\lambda(\tau_0)}\right|.
$$

### Practical Verification Steps

1. Define the scale $\lambda$ and modulation coefficient $a$.
2. Prove absolute continuity of $\log\lambda$ on selected windows.
3. Compute or bound the cost $\int|a|$.
4. Compare finite cost with the claimed terminal scale limit.
5. Assign non-finite-cost exits to the named defect or scale-rigid alternatives.

## Estimate Step $B_{\mathrm{PS28}}$

The estimate step is the logarithmic variation inequality.

## Failure Case

Failure name: unresolved scale-transition action.

Analytic meaning: the scale trajectory lacks enough regularity or window data
to decide finite cost, collapse, or autonomous-limit behavior.

## Refinement Step

Allowed refinements:

1. select thicker terminal windows;
2. improve modulation convergence;
3. assign compactness loss to `PS30`;
4. assign scale-rigid terminal states to `PS29`.

Progress measure: the finite-cost-collapse predicate is discharged or replaced
by one named scale-transition alternative.

## Data Passed Forward

The next proof step is `PS29`. The data passed forward are

$$
\Gamma_{\mathrm{PS28}}
=
\Gamma_{\mathrm{PS27}}
\cup
\{\text{scale-transition cost status}\}.
$$

---

# 35. `PS29` -- Lyapunov, Monotonicity, or Statistical-Rigidity Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a compact family of ancient profiles or terminal scale-rigid
states with a continuous local flow.

### Standing Assumptions

The incoming record contains compactness in the local smooth topology and pressure normalization on
compact balls.

### Objects Inspected

Inspect the candidate energy $E$, dissipation $D$, local virial identity,
pressure terms, cutoff errors, and invariant probability measures on $K$.

### Dependencies Used

Compactness comes from `PS13`; stationarity checks from `PS22`; virial or
scale-rigid structure from `PS28`; pressure defects are assigned to `PS30`.

### Local Obstruction Predicate

$P_{\mathrm{PS29}}$ holds when no verified monotone or statistical rigidity
functional is recorded for the terminal branch.

### Local Lemmas to Prove

**Lemma PS29.1 -- Invariant measures annihilate dissipation.**
If $\mu$ is an invariant Borel probability measure on $K$, then

$$
\int_KD(W)\,d\mu(W)=0.
$$

**Proof.** Integrate the Lyapunov identity with respect to $\mu$. Invariance
gives

$$
\int_KE(\Theta_tW)\,d\mu(W)=\int_KE(W)\,d\mu(W).
$$

Fubini's theorem and invariance also give

$$
\int_K\int_0^tD(\Theta_sW)\,ds\,d\mu(W)
=
t\int_KD(W)\,d\mu(W).
$$

Thus the last integral is zero.

**Lemma PS29.2 -- Zero averaged dissipation supports the measure on
$\{D=0\}$.**
If $D\ge0$ is continuous and $\int_KD\,d\mu=0$, then
$\mu(K\setminus\{D=0\})=0$.

**Proof.** For every $n$, the closed set $\{D\ge1/n\}$ has measure zero
because

$$
0=\int_KD\,d\mu\ge \frac1n\,\mu(\{D\ge1/n\}).
$$

The complement of $\{D=0\}$ is the union of these sets.

**Lemma PS29.3 -- Zero dissipation supplies stationary endpoint data when $D$
is coercive.**
If $D(W)=0$ implies $\partial_\tau W=0$ or the stationary centered equation,
then every invariant measure is supported on stationary profiles.

**Proof.** The coercivity condition verified in this node is

$$
D(W)=0
\Longrightarrow
\partial_\tau W=0
\quad\text{or}\quad
-\Delta W+(W\cdot\nabla)W+\nabla P
-\frac12y\cdot\nabla W-\frac12W=0.
$$

The invariant-measure identity in Lemma PS29.2 gives
$D(W)=0$ for $\mu$-almost every profile. Substituting this equality into the
displayed coercivity implication shows that the support of $\mu$ is contained
in the stationary zero set. This is an endpoint datum: it may be used for
endpoint exclusion only after `PS31` verifies the stationary theorem
hypotheses, including pressure representative, whole-space or local regime,
and critical-norm data.

**Lemma PS29.4 -- Gaussian virial identities supply zero-mass endpoint data.**
For a scale-rigid terminal state, if a Gaussian virial identity gives

$$
\frac{d}{d\tau}\int |V|^2G_\nu
\le
-\kappa\int |V|^2G_\nu
-\kappa\int |\nabla V|^2G_\nu
$$

after pressure and convection terms are absorbed, then the Gaussian velocity
mass vanishes on ancient terminal trajectories. This conclusion is an endpoint
input for `PS31`/`PS32`; it is a contradiction only if the branch also carries a
matched retained-positive-mass normalization.

**Proof.** Integrate the differential inequality backward on arbitrarily long
time intervals. Boundedness of the weighted mass and Gronwall's inequality
force the mass at any fixed time to be zero. The absorption hypothesis is part
of the lemma: if pressure, convection, cutoff, or modulation remainders are not
controlled in the displayed topology, the branch is routed to `PS30` instead
of using the virial identity.

### Specific Estimate

The decisive estimate is either the Lyapunov identity

$$
E(\Theta_tW)-E(W)
=
-\int_0^tD(\Theta_sW)\,ds
$$

or a local virial inequality with a coercive negative right-hand side.

### Practical Verification Steps

1. Define the compact hull and local flow.
2. State the functional $E$ and dissipation $D$.
3. Prove continuity of $E,D$ on the hull.
4. Derive the exact local identity or inequality.
5. Average over invariant measures or integrate the virial inequality.
6. Assign missing pressure or cutoff absorption to `PS30`.

## Estimate Step $B_{\mathrm{PS29}}$

The estimate step is the Lyapunov/statistical identity or virial inequality.

## Failure Case

Failure name: missing rigidity functional.

Analytic meaning: compact terminal dynamics remain, but no verified monotone,
coercive, or statistical estimate supplies stationary, zero-dissipation, or
zero-mass endpoint data.

## Refinement Step

Allowed refinements:

1. construct a local Lyapunov functional;
2. prove virial pressure absorption;
3. strengthen compactness of the hull;
4. assign missing absorption terms to `PS30`.

Progress measure: the terminal dynamics either supply stationary,
zero-dissipation, or zero-mass endpoint data, or are represented by named
defects.

## Data Passed Forward

The next proof step is `PS30`. The data passed forward are

$$
\Gamma_{\mathrm{PS29}}
=
\Gamma_{\mathrm{PS28}}
\cup
\{\text{rigidity theorem hypotheses verified, absent, or defective}\}.
$$

---

# 36. `PS30` -- Defect Audit Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknowns are all sequences and profiles produced by the proof, together
with their pressures, nonlinear stresses, and localization errors.

### Standing Assumptions

The incoming record states that every previous node has supplied its local estimates and that all
defects are evaluated on compact cylinders with pressure normalized by spatial
means.

### Objects Inspected

Inspect weak convergence of $u_n$, strong convergence in $L^3_{\rm loc}$,
convergence of $u_n\otimes u_n$, pressure oscillation convergence on compact
cylinders after local gauge subtraction, cutoff commutators, high-frequency
projections, modulation coefficients, and every defect that must be handed to
the local residual state space.

### Dependencies Used

Measure defects use `C_mu`, `PS1`, and `PS16`; stress defects use `PS6`,
`PS19`, and `PS20`; pressure defects use `PS4`; cutoff defects use `PS17` and
`PS20`; frequency and scale defects use `PS11` and `PS28`; rigidity defects use
`PS29`.

### Local Obstruction Predicate

$P_{\mathrm{PS30}}$ holds when some meaningful defect channel has no
conclusion compatible with the endpoint theorem hypotheses or with the local
state-space residual theorem. A pressure defect is not a demand for a global
pressure representative; it is a local gauge or compact-window pressure
failure unless an endpoint theorem explicitly requires more.

### Local Lemmas to Prove

**Lemma PS30.1 -- Strong local $L^3$ convergence removes Reynolds defects.**
If $u_n\to u$ strongly in $L^3(Q_R)$, then

$$
u_n\otimes u_n\to u\otimes u
\quad\text{in }L^{3/2}(Q_R).
$$

**Proof.** Use

$$
u_n\otimes u_n-u\otimes u
=(u_n-u)\otimes u_n+u\otimes(u_n-u)
$$

and Holder's inequality with the boundedness of $u_n$ in $L^3(Q_R)$.

**Lemma PS30.2 -- Measure defects are either absent or produce a core.**
If a nonnegative concentration measure $\mu$ associated with $|u_n|^3$ or
local CKN mass has positive mass on a compact cylinder, then a parabolic
blow-up around a density point produces a positive local concentration core.
If $\mu=0$, the measure defect is absent.

**Proof.** At a density point, choose radii on which the normalized mass is
bounded below. Parabolic rescaling preserves the local CKN normalization and
produces the active core. If no such density point exists, the measure is zero.

**Lemma PS30.3 -- Pressure defects are controlled by local pressure
decomposition.**
If the tensor source $u_n\otimes u_n$ converges in $L^{3/2}_{\rm loc}$, then
the pressure oscillations converge locally in $L^{3/2}$ after subtracting
spatial means, up to harmonic pressure terms controlled on smaller balls.

**Proof.** Decompose pressure into the Calderon--Zygmund part generated by
$\partial_i\partial_j(u_{n,i}u_{n,j})$ and a harmonic part. The singular
integral is continuous on $L^{3/2}$ locally after cutoff. Interior harmonic
estimates control the harmonic oscillation on smaller balls.

**Lemma PS30.4 -- Cutoff and artificial-boundary defects are controlled only
in the recorded topology.**
If the localized fields are bounded in $L^\infty_tL^3_x$, their mixed stresses
vanish in the branch topology, and the pressure source has a recorded annular
pressure convergence or pressure-decomposition estimate, then fixed-cutoff
commutators are either absent on cylinders contained in $\{\chi=1\}$ or are
controlled in the source topology recorded for the branch. A fixed
$L^1_tH^{-1}_x$ control is sufficient only for endpoint theorems whose
registered source topology accepts that space.

**Proof.** A cutoff commutator has the form

$$
(\nabla\chi)\cdot(u_n\otimes w_n),\qquad
(\Delta\chi)w_n,\qquad
(\nabla\chi)(P_n-\bar P_n),
$$

or a finite sum of such terms, with $\chi\in C_c^\infty$. The coefficients
$\nabla\chi$ and $\Delta\chi$ are bounded and are supported on the cutoff
annulus. On a cylinder contained in $\{\chi=1\}$ all displayed cutoff
commutators are absent. On a cylinder meeting the cutoff annulus, the
commutators are real source terms: products involving a vanishing velocity
component are controlled by the recorded mixed-stress convergence, while
$(\nabla\chi)(P_n-\bar P_n)$ is controlled only by the recorded annular
pressure estimate or pressure decomposition. The conclusion is therefore
topology-relative: the commutator is marked vanishing or absorbed only in the
source topology later registered for the endpoint theorem; otherwise the
uncontrolled term remains in the `PS30` defect vector.

**Lemma PS30.5 -- Frequency defects either vanish or define a new scale.**
Let $P_{\ge N}$ be a Littlewood--Paley projection on a compact window. If

$$
\lim_{N\to\infty}\limsup_n\|P_{\ge N}u_n\|_{L^3(Q_R)}=0,
$$

then no high-frequency defect remains. If the limit is positive, selecting a
frequency scale where the tail is nonzero produces a new concentration scale
for `PS11`.

**Proof.** The vanishing statement is the definition of compactness in the
critical local topology. If it fails, choose $N_n\to\infty$ with a fixed lower
bound on the projected norm. The associated physical length scale $N_n^{-1}$
is a new scale carrying positive critical mass.

**Lemma PS30.6 -- Modulation defects are named coefficient failures.**
If modulation coefficients fail to converge or remain bounded on selected
windows, the defect is exactly a scale, drift, or normalization defect already
assigned to `PS11`, `PS26`, or `PS28`.

**Proof.** Modulation coefficients are derivatives of selected center, scale,
rotation, or drift parameters. Failure of their boundedness or convergence is a
failure of the corresponding parameter selection or transition-cost estimate.

**Lemma PS30.7 -- Pressure-tail defects are local gauge states.**
If a branch has local pressure oscillation bounds

$$
P_n-a_{n,Q}(t)\rightharpoonup P
\quad\text{in }L^{3/2}(Q)
$$

on every compact observer cylinder $Q$, then pressure is admissible for the
local residual state space even if no global pressure representative on
$\mathbb R^3$ has been fixed. If such compact-window gauge bounds fail on a
selected observer cylinder, the failure is recorded as a local pressure
compactness defect for `ST4` and `ST6`.

**Proof.** The local energy inequality, local CKN quantities, and centered
equation use either $\nabla P$ or pressure oscillations modulo functions of
time. Subtracting $a_{n,Q}(t)$ leaves those quantities unchanged on $Q$ and is
stable under restriction to smaller cylinders. Hence compact-window
$L^{3/2}$ weak control is exactly the pressure datum required by
$\mathfrak X_M$ in `ST2`. A missing whole-space Riesz representative affects
only endpoint theorems whose registry entry in `PS31` demands that
representative. It is not a residual obstruction for the local state-space
closure.

**Lemma PS30.8 -- Residual defects have local destinations.**
If a defect channel remains after the audit, it must be assigned to one of the
following local residual destinations:

$$
\text{active local concentration},\quad
\text{diffuse exterior defect},\quad
\text{critical-tail boundary state},\quad
\text{local pressure compactness failure},\quad
\text{local coefficient or source failure}.
$$

The first three are routed to `ST5`--`ST11`, pressure compactness failure is
routed to `ST4` and `ST6`, and coefficient/source failures are routed to the
earlier node that owns the missing transformed equation or to `PS33` if
realization is the issue.

**Proof.** The defect vector is meaningful only when each entry has a target
predicate. A local concentration defect has positive mass on a compact
observer cylinder and therefore belongs to the active-locus extraction.
A diffuse defect is precisely mass that survives through escaping observer
windows while subthreshold on each unit cylinder. A critical tail is a
boundary state of the observer compactification. A pressure failure on a
compact cylinder is local because gauges are defined cylinderwise. Any
remaining coefficient or source term belongs to the equation construction or
to the provenance graph, not to a global tail estimate.

### Specific Estimate

The decisive vector statement is

$$
\mathbf d
\in
\{\mathrm{absent},\mathrm{absorbed},\mathrm{recentered},
\mathrm{unresolved},\mathrm{not\ applicable}\}^7
$$

with no blank entry.

### Practical Verification Steps

1. Build the defect vector in the displayed order.
2. Verify strong $L^3$ convergence or record a stress defect.
3. Verify pressure convergence after spatial-mean normalization.
4. Check measure concentration and recenter any positive defect measure.
5. Check pressure only through local gauges on every compact observer
   cylinder unless a registered endpoint theorem requires a global
   representative.
6. Check cutoff commutators on every compact cylinder used later.
7. Check frequency tails or assign a new scale.
8. Check modulation coefficients and assign parameter failures.
9. Record unknown residues only if they are explicit local distributions or
   local residual-state witnesses for `ST0`--`ST20`.

## Estimate Step $B_{\mathrm{PS30}}$

The estimate step is the channel-by-channel local compactness, pressure, and
commutator audit.

## Failure Case

Failure name: incomplete defect vector.

Analytic meaning: a term appears in a limit equation, endpoint hypothesis, or
local energy identity without being absent, absorbed, recentered, or named.

## Refinement Step

Allowed refinements:

1. add a missing concentration scale;
2. improve strong convergence;
3. refine pressure decomposition;
4. shrink or enlarge cutoffs;
5. name an explicit local residual distribution and assign it to the
   state-space residual block `ST0`--`ST20` or to `PS34` only as
   set-theoretic bookkeeping for that routed branch.

Progress measure: each refinement fills one previously blank defect-vector
entry.

## Data Passed Forward

The next proof step is `ST0` for any generic residual branch with local
state-space destinations. Nonresidual endpoint branches may continue directly
to `PS31`. The data passed forward are

$$
\Gamma_{\mathrm{PS30}}
=
\Gamma_{\mathrm{PS29}}
\cup
\left\{
\begin{array}{l}
\mathbf d\text{ complete},\\
\text{all pressure entries are local-gauge controlled, endpoint-specific,}\\
\text{or routed as local pressure compactness defects to }ST4/ST6,\\
\text{all residual entries have local destinations in }ST0\text{--}ST20
\end{array}
\right\}.
$$

---

# 37. `ST0` -- Lower-Strata Ledger

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a bounded centered profile $(U,\Pi)$ or a terminal sequence
$(U_n,\Pi_n)$ that has reached the residual state-space audit after the
earlier profile, structure, defect, and endpoint-routing nodes have been run.

### Standing Assumptions

The incoming record contains retained compact CKN activity, local suitability,
local pressure gauges, and a complete defect vector from `PS30`. The record
also contains the status of every lower class already treated by the sieve.

### Objects Inspected

Inspect membership in the lower strata whose status is already known or whose
open theorem/realization route is already owned by an earlier node:

$$
\begin{array}{l}
\text{small amplitude},\quad
\text{stationary }L^3,\quad
\text{uniformly }L^3\text{-tight},\\
\text{closed structured or decay classes},\quad
\text{axisymmetric bounded-circulation classes},\\
\text{rotational relative equilibria},\quad
\text{degenerate stationary-hull profiles},\\
\text{affine/parasitic constant profiles}.
\end{array}
$$

### Dependencies Used

Smallness comes from `PS21`; stationary branches from `PS22`; structured,
axisymmetric, rotational, perturbative, and degenerate stationary-hull
branches from `PS23`--`PS29`; tight branches from `PS15`; defect status from
`PS30`. If a lower stratum has an endpoint theorem or realization issue that
has not yet been matched, `ST0` records the lower-stratum route and exits the
generic residual block; it does not wait for, or assume, a later `PS31`--`PS33`
closure record.

### Local Obstruction Predicate

$P_{\mathrm{ST0}}$ holds if the residual block is entered before the proof has
recorded whether the incoming profile is already in a lower stratum with a
closed status or an owned open obligation. The residual block may not reprove
lower cases or count a lower case again as generic residual.

### Local Lemmas to Prove

**Lemma ST0.1 -- Closed lower strata are absorbing exits.**
If the incoming profile belongs to a lower stratum whose realized admissible
part has already been excluded or proved nonattainable, then the residual
state-space block stops for that branch and records the corresponding
exclusion source. If it belongs to a lower stratum whose endpoint or
realization status is still open, the residual block also stops, but records
the open lower-stratum obligation rather than treating the branch as generic
residual.

**Proof.** The lower-strata ledger stores, for each class $\mathcal L_j$, the
predicate defining the class, its proof source, and its status. If
$(U,\Pi)\in\mathcal L_j$ and the status is closed, then the branch has already
received either an endpoint contradiction or a nonattainability proof. If the
status is open, the correct next obligation is the theorem or realization
route attached to $\mathcal L_j$. In both cases the profile is not a generic
residual profile. Running a residual argument on the same branch would
duplicate the class rather than refine the proof.

**Lemma ST0.2 -- Generic residual means outside the routed lower ledger.**
Define

$$
\mathcal L_j^{\rm routed}
=
\left\{
\begin{array}{l}
\mathcal L_j\text{ with closed status, or}\\
\mathcal L_j\text{ with an open endpoint/realization obligation owned by}\\
\text{its lower-stratum node}
\end{array}
\right\},
\qquad
\mathcal R_{\rm loc}
=
\mathfrak X_M
\setminus
\bigcup_j \mathcal L_j^{\rm routed}.
$$

A profile enters `ST1` only if it lies in $\mathcal R_{\rm loc}$ and carries
retained compact activity.

**Proof.** This is the ordered subtraction convention used in `PS34`. The
residual class is not a vague complement; it is the exact complement of all
lower predicates whose status already has an owner. Closed lower predicates
exit with their existing exclusion source. Open lower predicates exit with
their theorem or realization obligation. Only profiles outside this routed
ledger enter the generic residual state space.

### Specific Estimate

The decisive check is the lower-strata membership vector

$$
\ell(U)
=
\left(
\mathbf 1_{U\in\mathcal L_j},
\mathrm{status}(\mathcal L_j),
\mathrm{source}(\mathcal L_j)
\right)_j
$$

with no blank entry.

### Practical Verification Steps

1. List every lower stratum available before the residual block.
2. Attach the defining predicate, proof source, and status to each entry.
3. If a closed lower predicate holds, route to that existing exclusion.
4. If an open lower predicate holds, route to the theorem or realization gap
   that owns it.
5. If no lower predicate holds, declare the profile generic residual and pass
   it to `ST1`.

## Estimate Step $B_{\mathrm{ST0}}$

The estimate step is ordered lower-strata ledger verification. No PDE estimate
is introduced at this node.

## Failure Case

Failure name: lower-strata ledger gap.

Analytic meaning: the branch is being called generic residual while a lower
class has not been checked, or while an open lower-class status is being
silently treated as excluded.

## Refinement Step

Allowed refinements:

1. add the missing lower-stratum predicate;
2. reroute the branch to the lower node that owns it;
3. split a mixed-status class into closed and open subbranches;
4. update `PS34` so the same class is not counted twice.

Progress measure: the membership vector $\ell(U)$ has no blank entries.

## Data Passed Forward

On the generic residual path, the next proof step is `ST1`. Lower-stratum
matches exit through their owning lower node or obligation ledger. The data
passed forward on the generic residual path are

$$
\Gamma_{\mathrm{ST0}}
=
\Gamma_{\mathrm{PS30}}
\cup
\left\{
\begin{array}{l}
\mathcal R_{\rm loc}\text{ defined by ordered routed lower-strata subtraction},\\
(U,\Pi)\in\mathcal R_{\rm loc},\\
\text{retained compact activity},\\
\text{lower-strata ledger with sources}
\end{array}
\right\}.
$$

---

# 38. `ST1` -- Covariant Observer Calculus

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a bounded centered profile $(U,\Pi)$ and the family of local
observer transforms used to recenter escaping activity without breaking the
centered equation.

### Standing Assumptions

The incoming profile solves the centered equation locally,

$$
\partial_\tau U+(U\cdot\nabla)U+\nabla\Pi-\Delta U
+\frac12U+\frac12y\cdot\nabla U=0,
\qquad \nabla\cdot U=0,
$$

with pressure defined modulo functions of $\tau$.

### Objects Inspected

Inspect the covariant centered translations

$$
(T_aU)(y,\tau)=U(y+e^{\tau/2}a,\tau),
$$

time translations

$$
(\Theta_sU)(y,\tau)=U(y,\tau+s),
$$

and observer recenterings

$$
\mathscr T_{(x,\sigma)}U(y,s)
=
U(y+e^{s/2}x,\sigma+s).
$$

The pressure is transformed by composition and by any affine correction needed
for the affine/parasitic normalization of `ST3`; time-dependent gauge terms
remain quotiented out.

### Dependencies Used

The centered equation comes from `PS5`; pressure gauges from `PS4` and
`PS30`; local suitability from `PS7`; retained activity from `PS8`;
terminal recentering warnings from `PS14`.

### Local Obstruction Predicate

$P_{\mathrm{ST1}}$ holds if a branch uses raw spatial translations as if they
preserved the centered equation. Raw translations create the wrong
$\frac12y\cdot\nabla U$ drift and cannot be used in an endpoint-ready residual
branch.

### Local Lemmas to Prove

**Lemma ST1.1 -- Covariant translations preserve the centered equation.**
If $(U,\Pi)$ solves the centered equation, then

$$
U_a(y,\tau)=U(y+e^{\tau/2}a,\tau),
\qquad
\Pi_a(y,\tau)=\Pi(y+e^{\tau/2}a,\tau)
$$

solves the same centered equation, modulo the same time-dependent pressure
gauge.

**Proof.** Set $z=y+e^{\tau/2}a$. Then

$$
\partial_\tau U_a
=
(\partial_\tau U)(z,\tau)
+\frac12 e^{\tau/2}a\cdot\nabla U(z,\tau),
$$

while

$$
\frac12y\cdot\nabla U_a
+\frac12 e^{\tau/2}a\cdot\nabla U(z,\tau)
=
\frac12z\cdot\nabla U(z,\tau).
$$

All other terms are invariant under spatial composition. Substitution gives
the centered equation for $U_a$.

**Lemma ST1.2 -- Observer recentering preserves local suitability.**
If $(U,\Pi)$ is locally suitable on compact centered cylinders, then
$\mathscr T_{(x,\sigma)}(U,\Pi)$ is locally suitable on every compact
observer cylinder whose image is a compact centered cylinder.

**Proof.** The observer map is a smooth parabolic change of variables on each
compact window and is an exact symmetry of the centered equation by
Lemma ST1.1 and time translation. Pulling test functions through the map gives
the local energy inequality with the pressure gauge transformed by
composition and by a function of time.

### Specific Estimate

The decisive identity is

$$
\partial_s\mathscr T_{(x,\sigma)}U
+\frac12 y\cdot\nabla\mathscr T_{(x,\sigma)}U
=
\left(\partial_\tau U+\frac12 z\cdot\nabla U\right)(z,\sigma+s),
\quad z=y+e^{s/2}x.
$$

### Practical Verification Steps

1. Write every recentering in covariant observer form.
2. Check the pressure gauge after composition.
3. Verify that compact observer cylinders map to compact centered cylinders.
4. Recompute the centered drift if any raw translate appears.
5. Route noncovariant formulas back to this node before compactness is used.

## Estimate Step $B_{\mathrm{ST1}}$

The estimate step is the chain-rule verification of the covariant observer
symmetry and the local suitability pullback.

## Failure Case

Failure name: noncovariant recentering.

Analytic meaning: an escaping branch has been recentered by a map that changes
the centered equation but the new drift or pressure term has not been
recorded.

## Refinement Step

Allowed refinements:

1. replace raw translations by $T_a$ or $\mathscr T_{(x,\sigma)}$;
2. add the missing drift if the branch intentionally uses a noncovariant
   frame;
3. shrink the compact observer cylinder so the map is admissible;
4. return to `PS5` if the transformed equation has changed.

Progress measure: every observer transform is exact or its defect is recorded.

## Data Passed Forward

The next proof step is `ST2`. The data passed forward are

$$
\Gamma_{\mathrm{ST1}}
=
\Gamma_{\mathrm{ST0}}
\cup
\left\{
Z=\mathbb R^3\times\mathbb R,\quad
T_a,\quad \Theta_s,\quad \mathscr T_{(x,\sigma)},
\quad \text{covariant observer gauges}
\right\}.
$$

---

# 39. `ST2` -- Terminal Local State Space

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is the compact local state space replacing global compactness
assumptions in the residual branch.

### Standing Assumptions

The residual profile is bounded by a fixed $M$, solves the centered equation
locally, is locally suitable on compact terminal cylinders, and has pressure
oscillation representatives modulo time-dependent gauges.

### Objects Inspected

Define $\mathfrak X_M$ to be the set of pairs $(U,\Pi)$ modulo
time-dependent pressure gauges such that

$$
\|U\|_{L^\infty(Z)}\le M,
$$

$(U,\Pi)$ solves the centered equation locally, $(U,\Pi)$ is locally suitable
on every compact terminal cylinder, and

$$
\Pi-c_Q(\tau)\in L^{3/2}(Q)
$$

for every compact observer cylinder $Q\Subset Z$ and some gauge $c_Q(\tau)$.
The smooth branch topology is

$$
U_n\to U\quad\text{in }C^\infty_{\rm loc},
\qquad
\Pi_n-c_{n,Q}(\tau)\rightharpoonup \Pi
\quad\text{in }L^{3/2}(Q)
$$

on every compact $Q$. If a sequence reaches this node only with the suitable
weak compactness package from `PS6`, the velocity topology is first the
recorded strong $L^3_{\rm loc}$ suitable topology with weak
$L^{3/2}_{\rm loc}$ pressure. It may be upgraded to $C^\infty_{\rm loc}$ only
after the bounded local regularity input has been applied on the compact
window. The node must record which of these two topologies is being used.

### Dependencies Used

Bounded centered profiles come from `PS9`; smooth local compactness from
`PS13` or the local compactness package of `PS6`; pressure gauges from `PS4`
and `PS30`; local suitability from `PS7`; covariant observers from `ST1`.

### Local Obstruction Predicate

$P_{\mathrm{ST2}}$ holds if the residual branch invokes compactness without
specifying the compact-window topology, pressure quotient, or local suitability
class.

### Local Lemmas to Prove

**Lemma ST2.1 -- Compact-window closure of $\mathfrak X_M$.**
Every sequence in $\mathfrak X_M$ with the recorded compact-window pressure
gauges and either the smooth bounded-profile package or the `PS6` suitable
compactness package has a subsequence converging on any fixed compact observer
cylinder in the corresponding local topology.

**Proof.** In the smooth bounded-profile branch, uniform boundedness, the
centered equation, and local pressure-gauge bounds give local parabolic
regularity on compact subcylinders, hence $C^\infty_{\rm loc}$ subsequential
compactness. In the suitable branch, the output is only the strong
$L^3_{\rm loc}$ and weak pressure topology already justified in `PS6` until a
separate bounded-regularity upgrade is recorded. Pressure oscillation bounds
give weak compactness in $L^{3/2}$ after subtracting local gauges in either
case. Diagonal extraction over a compact exhaustion gives the stated
compact-window topology. No global compactness is used.

**Lemma ST2.2 -- No global norm is part of the state-space definition.**
Membership in $\mathfrak X_M$ requires no global $L^3$ bound, no uniform
tightness modulus, and no whole-space pressure representative.

**Proof.** Each defining item is checked on compact observer cylinders or is
the global boundedness inherited from the Type I profile. The pressure is
quotiented by local time-dependent gauges. Whole-space integrability appears
only later if an endpoint theorem requires it, and in the residual branch it
is recovered at `ST17` rather than assumed here.

### Specific Estimate

The decisive compactness statement is

$$
\forall K\Subset Z,\qquad
\sup_n\|U_n\|_{L^\infty(K)}+
\sup_n\|\Pi_n-c_{n,K}(\tau)\|_{L^{3/2}(K)}<\infty
$$

with local suitability on $K$, plus an explicit topology label:
$C^\infty_{\rm loc}$ after bounded regularity, or the `PS6`
strong-$L^3_{\rm loc}$/weak-pressure topology before that upgrade.

### Practical Verification Steps

1. Fix the bound $M$ and the compact exhaustion of observer space.
2. Normalize pressure separately on each compact cylinder.
3. Verify the centered equation and local suitability after observer
   recentering.
4. Extract subsequences only on compact windows and diagonalize.
5. Keep the pressure quotient in every later state-space map.

## Estimate Step $B_{\mathrm{ST2}}$

The estimate step is compact-window parabolic compactness plus local
pressure-gauge weak compactness.

## Failure Case

Failure name: missing local state-space topology.

Analytic meaning: the branch has a residual profile but lacks the topology or
pressure gauge needed to pass to local observer limits.

## Refinement Step

Allowed refinements:

1. shrink to compact observer cylinders;
2. subtract local pressure gauges;
3. downgrade smooth compactness to the `PS6` suitable topology when needed;
4. return to `PS30` if a local pressure bound is missing.

Progress measure: every compact observer window has a recorded state-space
compactness package.

## Data Passed Forward

The next proof step is `ST3`. The data passed forward are

$$
\Gamma_{\mathrm{ST2}}
=
\Gamma_{\mathrm{ST1}}
\cup
\{\mathfrak X_M,\ \text{local topology},\
\text{compactness on compact observer windows}\}.
$$

---

# 40. `ST3` -- Affine/Parasitic Lower Stratum and Non-Affine Activity

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a local observer state in $\mathfrak X_M$ that may contain a
constant or affine parasitic component rather than genuine singular activity.

### Standing Assumptions

The branch is outside the lower-strata ledger of `ST0` except possibly for
affine/parasitic behavior revealed only after local observer normalization.

### Objects Inspected

Inspect the affine centered normalization

$$
(S_cU)(y,\tau)=U(y-2c,\tau)-c,
$$

with the pressure transformed by the corresponding affine linear correction.
On a compact observer cylinder $Q$, define

$$
\operatorname{osc}_3(U;Q)
=
\inf_{c\in\mathbb R^3}\iint_Q |U-c|^3\,dy\,d\tau .
$$

### Dependencies Used

The affine transform is checked against the centered equation from `PS5`;
local pressure gauges come from `PS4` and `ST2`; retained activity comes from
`PS8`; lower-strata status comes from `ST0`.

### Local Obstruction Predicate

$P_{\mathrm{ST3}}$ holds if the branch treats a constant or affine parasitic
state as a genuine residual singular profile.

### Local Lemmas to Prove

**Lemma ST3.1 -- Affine/parasitic states are lower strata.**
If on every retained compact observer cylinder

$$
\operatorname{osc}_3(U;Q)=0,
$$

then $U$ is locally constant in space on the retained observer component, and
the branch belongs to the affine/parasitic lower stratum after the pressure
linear correction.

**Proof.** Vanishing oscillation on a cylinder means $U=c_Q$ almost
everywhere there. Overlapping cylinders force the constants to agree on the
connected retained component. Substitution into the centered equation leaves
only an affine pressure gradient, which is exactly the pressure correction in
$S_c$.

**Lemma ST3.2 -- Non-affine activity is a local observable.**
If the branch is not affine/parasitic, then there exist a compact observer
cylinder $Q$ and $\eta>0$ such that

$$
\Phi(U;Q):=\operatorname{osc}_3(U;Q)\ge\eta.
$$

**Proof.** If all compact-cylinder oscillations were zero, Lemma ST3.1 would
place the branch in the lower stratum. The negation gives a compact cylinder
with positive oscillation; reduce the positive value to a rational threshold
$\eta$ for the ledger.

### Specific Estimate

The decisive local activity estimate is

$$
\Phi(U;Q)\ge\eta
$$

on a specified compact observer cylinder, unless the affine/parasitic lower
stratum status is recorded.

### Practical Verification Steps

1. Apply the affine/parasitic normalization if a constant component is present.
2. Compute local oscillation on the retained observer cylinders.
3. If all oscillations vanish, route to the affine/parasitic lower stratum.
4. If some oscillation is positive, record $Q$ and $\eta$.
5. Use oscillation, not global tail mass, as the residual activity observable.

## Estimate Step $B_{\mathrm{ST3}}$

The estimate step is the local oscillation test and affine pressure
normalization.

## Failure Case

Failure name: parasitic activity ambiguity.

Analytic meaning: the branch carries nonzero-looking tail data, but the proof
has not separated genuine local oscillation from an affine or constant
parasitic mode.

## Refinement Step

Allowed refinements:

1. subtract the best local constant;
2. apply the affine pressure correction;
3. shrink to a retained compact cylinder with positive oscillation;
4. route zero-oscillation cases to the affine lower stratum.

Progress measure: the branch receives either affine lower-stratum status or a
positive non-affine local activity threshold.

## Data Passed Forward

If non-affine activity is recorded, the next proof step is `ST4`. If the
affine/parasitic lower stratum is detected, the branch exits through `ST0`.
The data passed forward on the non-affine path are

$$
\Gamma_{\mathrm{ST3}}
=
\Gamma_{\mathrm{ST2}}
\cup
\{\Phi(U;Q)\ge\eta\}.
$$

---

# 41. `ST4` -- Local CKN and Oscillation Measures

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a terminal sequence $(U_n,\Pi_n)$ in $\mathfrak X_M$ and the
local measures used to detect active, diffuse, or pressure-compactness
residual behavior.

### Standing Assumptions

The branch has covariant observer coordinates, local pressure gauges, and
non-affine local activity unless it has already exited through `ST3`.

### Objects Inspected

For compact observer windows define local CKN measures. The pressure gauge is
part of the compact-window datum: for $K\Subset Z$ choose an enlarged window
$K^+$ and a gauge $a_{n,K^+}(\tau)$, and set

$$
d\mu_{n,K}
=
\left(
|U_n|^3+|\Pi_n-a_{n,K^+}(\tau)|^{3/2}
\right)\,dy\,d\tau,
$$

on $K$. These are compatible local measures on selected compact windows, not
a single global pressure measure. Also fix a bounded-overlap covariant
unit-cylinder cover
$\{\mathcal Q_1^{\rm cov}(z_k)\}_{k\in I}$ with locally finite index set.
For $E\subset K$, set

$$
I_K(E)
=
\{k\in I:\ z_k\in K,\ \mathcal Q_1^{\rm cov}(z_k)\cap E\ne\emptyset\},
$$

and define

$$
\mu_{n,K}^{\rm osc}(E)
=
\sum_{k\in I_K(E)}
\operatorname{osc}_3
\left(U_n;\mathcal Q_1^{\rm cov}(z_k)\right).
$$

### Dependencies Used

Observer cylinders come from `ST1`; state-space gauges from `ST2`; oscillation
from `ST3`; pressure compactness from `PS30`.

### Local Obstruction Predicate

$P_{\mathrm{ST4}}$ holds if local activity or pressure is measured without a
compact-window gauge, or if a local pressure compactness failure is hidden as
a global pressure-tail problem.

### Local Lemmas to Prove

**Lemma ST4.1 -- Compact windows have finite local mass.**
For every compact observer window $K\Subset Z$,

$$
\sup_n\mu_{n,K^+}(K^+)<\infty
$$

after enlarging $K$ to a fixed compact $K^+$ used for local estimates.

**Proof.** The velocity term is bounded by $M^3|K^+|$. The pressure term is
bounded by the local pressure-gauge estimate in $\mathfrak X_M$ or by the
`PS30` local pressure audit. The bounded-overlap cover converts cylinder
oscillation sums into a finite multiple of the $L^3$ bound on a fixed
enlarged compact window containing all cylinders that meet $K$.

**Lemma ST4.2 -- Local pressure failure has a local residual destination.**
If the estimate in Lemma ST4.1 fails because no pressure gauge has compact
$L^{3/2}$ control on a cylinder, the branch is assigned to the local
noncompactness remainder in `ST6`, not to a global pressure-tail estimate.

**Proof.** The failure occurs on a named compact observer cylinder and hence
is a local compactness failure. It prevents the state-space topology of `ST2`
on that window. No statement about pressure on the complement of the cylinder
is involved.

### Specific Estimate

The decisive estimate is

$$
\forall K\Subset Z,\qquad
\sup_n\mu_{n,K}(K)+\sup_n\mu_{n,K}^{\rm osc}(K)<\infty.
$$

### Practical Verification Steps

1. Choose compact observer windows and their bounded-overlap unit-cylinder
   covers.
2. Select local pressure gauges on each enlarged window.
3. Bound the velocity and pressure pieces of $\mu_{n,K}$ locally.
4. Bound the oscillation measure by the same compact-window control.
5. Route missing pressure compactness to `ST6`.

## Estimate Step $B_{\mathrm{ST4}}$

The estimate step is local finite-mass control for CKN and oscillation
measures.

## Failure Case

Failure name: local measure/gauge failure.

Analytic meaning: a compact observer window lacks the pressure or oscillation
control needed to define local active and diffuse states.

## Refinement Step

Allowed refinements:

1. choose a smaller compact observer window;
2. subtract a compatible local pressure gauge;
3. replace CKN mass by oscillation mass if pressure is not needed for the
   branch;
4. route genuine local pressure compactness failure to `ST6`.

Progress measure: every compact observer window has a finite local measure or
a named local noncompactness defect.

## Data Passed Forward

The next proof step is `ST5`. The data passed forward are

$$
\Gamma_{\mathrm{ST4}}
=
\Gamma_{\mathrm{ST3}}
\cup
\{\mu_{n,K},\ \mu_{n,K}^{\rm osc}\text{ for }K\Subset Z,\
\text{local finite mass on compact observer windows}\}.
$$

---

# 42. `ST5` -- Active-Locus Extraction on Compact Observer Windows

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is the compact-window active set of a terminal sequence measured
by the local CKN or oscillation mass.

### Standing Assumptions

The sequence has local finite measure on compact observer windows and a
retained activity threshold.

### Objects Inspected

For an exhaustion $K_m\Subset Z$ and threshold $\eta>0$, define

$$
A_{n,m}^{\eta}
=
\{z\in K_m:\ M_n(z)\ge\eta\},
$$

where $M_n(z)$ is the local CKN or oscillation mass on
$\mathcal Q_1^{\rm cov}(z)$.

### Dependencies Used

Measures come from `ST4`; observer cylinders from `ST1`; retained compact
activity from `PS8`; compactness of closed subsets of $K_m$ from the
Hausdorff--Fell topology.

### Local Obstruction Predicate

$P_{\mathrm{ST5}}$ holds if the proof claims tail activity without extracting
active sets on compact observer windows or without a threshold.

### Local Lemmas to Prove

**Lemma ST5.1 -- Compact-window active sets have subsequential limits.**
For each fixed $m$ and rational $\eta>0$, after passing to a subsequence,

$$
A_{n,m}^{\eta}\to A_m^\eta
$$

in the Hausdorff--Fell topology on closed subsets of $K_m$.

**Proof.** After replacing $A_{n,m}^\eta$ by its closure, the hyperspace of
closed subsets of compact $K_m$ is compact. Diagonal extraction over the
countable set of pairs $(m,\eta)$ gives compatible limits.

**Lemma ST5.2 -- Active retained loci give local profile descendants.**
If $z_n\in A_{n,m}^{\eta}$ with $z_n\to z\in A_m^\eta$, then covariant
recenterings at $z_n$ have a subsequential local state-space limit with
positive local activity, unless `ST6` records local compactness failure.

**Proof.** The definition of $A_{n,m}^\eta$ supplies a lower bound on a fixed
covariant unit cylinder. `ST2` gives compactness on compact observer windows
provided pressure gauges and the local topology are available. Lower
semicontinuity of the local mass or oscillation gives positive retained
activity in the limit.

### Specific Estimate

The decisive extraction is

$$
A_{n,m}^{\eta}\to A_m^\eta
\quad
\text{for all }m\in\mathbb N,\ \eta\in\mathbb Q_+.
$$

### Practical Verification Steps

1. Choose the compact exhaustion $K_m$.
2. Choose rational thresholds.
3. Define $M_n(z)$ using CKN or oscillation mass on covariant unit cylinders.
4. Extract Hausdorff--Fell limits on each $K_m$.
5. Record every positive retained active locus and its recentering sequence.

## Estimate Step $B_{\mathrm{ST5}}$

The estimate step is compact active-set extraction and retained local profile
recovery.

## Failure Case

Failure name: missing active-locus extraction.

Analytic meaning: the branch speaks of escaping or retained activity but has
not produced compact-window active sets or thresholds.

## Refinement Step

Allowed refinements:

1. lower the threshold to a rational value;
2. enlarge the compact observer window;
3. switch from CKN mass to oscillation mass if pressure is not compact;
4. send compactness failure to `ST6`.

Progress measure: every activity claim has a compact-window active set or a
named remainder.

## Data Passed Forward

The next proof step is `ST6`. The data passed forward are

$$
\Gamma_{\mathrm{ST5}}
=
\Gamma_{\mathrm{ST4}}
\cup
\{A_m^\eta,\ \text{active retained profile loci and recenterings}\}.
$$

---

# 43. `ST6` -- Paired Terminal Decomposition

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is the residual remainder after active neighborhoods have been
removed from compact observer windows.

### Standing Assumptions

Active loci have been extracted for all compact windows and thresholds.

### Objects Inspected

After deleting fixed covariant neighborhoods of $A_m^\eta$, classify the
remaining sequence as one of

$$
\mathcal R_{\rm van},\qquad
\mathcal R_{\rm diff},\qquad
\mathcal R_{\rm noncomp},
$$

meaning local vanishing, diffuse exterior concentration, or loss of local
compactness.

### Dependencies Used

Active loci come from `ST5`; local finite mass from `ST4`; local state-space
compactness from `ST2`; pressure compactness defects from `PS30`.

### Local Obstruction Predicate

$P_{\mathrm{ST6}}$ holds if a residual remainder is left without being named
as vanishing, diffuse, or locally noncompact.

### Local Lemmas to Prove

**Lemma ST6.1 -- Ordered active/remainder decomposition.**
For every compact $K_m$ and threshold $\eta$, the sequence decomposes into
active neighborhoods of $A_m^\eta$ plus a remainder on which every covariant
unit cylinder has mass below $\eta$, unless local compactness fails.

**Proof.** By definition, outside a small neighborhood of the Hausdorff--Fell
limit of $A_{n,m}^\eta$, all remaining unit cylinders eventually fail the
$\eta$ lower bound. If this assertion fails, there is a new sequence of
points with mass at least $\eta$, and compactness of $K_m$ produces a point in
$A_m^\eta$, contradicting that the point was outside the active neighborhood.
If the local topology needed for this argument fails, the branch is
$\mathcal R_{\rm noncomp}$.

**Lemma ST6.2 -- The residual alternatives are exhaustive.**
The remainder either vanishes on compact observer windows, persists only
through escaping windows with subthreshold unit mass, or lacks local
state-space compactness.

**Proof.** If compact-window mass tends to zero, it is
$\mathcal R_{\rm van}$. If compact-window mass is controlled and nonzero mass
survives only outside every compact set while unit cylinders remain
subthreshold, it is $\mathcal R_{\rm diff}$. The only remaining possibility is
failure of the compact-window topology, pressure gauge, or equation/source
compactness, which is $\mathcal R_{\rm noncomp}$.

### Specific Estimate

The decisive remainder statement is

$$
\sup_{z\in K_m\setminus N_\rho(A_m^\eta)}
M_n(z)<\eta
\quad\text{eventually},
$$

unless the branch is explicitly marked $\mathcal R_{\rm noncomp}$.

### Practical Verification Steps

1. Remove covariant neighborhoods of active sets.
2. Check subthreshold unit-cylinder mass on the complement.
3. Test whether the complement vanishes on compact windows.
4. If mass escapes every compact window, label it diffuse.
5. If compactness or pressure gauges fail locally, label it noncompact.

## Estimate Step $B_{\mathrm{ST6}}$

The estimate step is active-neighborhood removal and exhaustive remainder
classification.

## Failure Case

Failure name: unpaired residual remainder.

Analytic meaning: active loci have been found, but the leftover sequence has
no local status.

## Refinement Step

Allowed refinements:

1. adjust the active threshold;
2. shrink or enlarge active neighborhoods;
3. add a missing pressure-gauge compactness check;
4. route escaping subthreshold mass to `ST8`.

Progress measure: every remainder is vanishing, diffuse, or locally
noncompact.

## Data Passed Forward

The next proof step is `ST7`. The data passed forward are

$$
\Gamma_{\mathrm{ST6}}
=
\Gamma_{\mathrm{ST5}}
\cup
\{\text{paired active-locus/remainder decomposition},
\mathcal R_{\rm van},\mathcal R_{\rm diff},\mathcal R_{\rm noncomp}\}.
$$

---

# 44. `ST7` -- Nonactive Sole-Carrier Discharge

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a branch in which the retained singular activity is claimed to
be carried only by local vanishing, diffuse exterior behavior, or local
noncompactness.

### Standing Assumptions

The singular-entry packet supplies a fixed compact retained activity lower
bound

$$
\iint_{Q_{\rm act}} |U_n|^3\,dy\,d\tau\ge\eta_*,
$$

or the corresponding pressure-normalized CKN lower bound.

### Objects Inspected

Inspect a finite covariant unit-cylinder cover of $Q_{\rm act}$ and the
remainder status from `ST6`.

### Dependencies Used

Retained activity comes from `PS8`; finite covering from compactness of
$Q_{\rm act}$; local measures from `ST4`; paired decomposition from `ST6`.

### Local Obstruction Predicate

$P_{\mathrm{ST7}}$ holds if the proof allows the distinguished compact
retained mass to disappear into a nonactive or escaping remainder without a
finite covering check.

### Local Lemmas to Prove

**Lemma ST7.1 -- Compact retained activity forces an active local carrier.**
If $Q_{\rm act}$ is covered by $N$ covariant unit cylinders and

$$
\iint_{Q_{\rm act}} |U_n|^3\ge\eta_*,
$$

then at least one cylinder in the cover carries mass at least
$\eta_*/N$.

**Proof.** Sum the local masses over the finite cover. Bounded overlap changes
only the constant. If every cylinder had mass below $\eta_*/N$ with the
appropriate overlap factor, the total mass on $Q_{\rm act}$ would be below
$\eta_*$.

**Lemma ST7.2 -- Pure nonactive remainders cannot be sole carriers.**
The alternatives $\mathcal R_{\rm van}$ and $\mathcal R_{\rm diff}$ cannot
alone carry the retained compact activity on $Q_{\rm act}$. A local
noncompactness claim must be recorded as an `ST6` defect rather than as a
closed branch.

**Proof.** Local vanishing contradicts Lemma ST7.1 on the finite cover.
Diffuse exterior concentration escapes every compact set, while
$Q_{\rm act}$ is fixed compact. Local noncompactness is a named defect and
therefore cannot be counted as the carrier of a completed retained profile.

### Specific Estimate

The decisive finite-cover estimate is

$$
\max_{1\le j\le N}
\iint_{\mathcal Q_j}|U_n|^3
\ge
c(N)\eta_*.
$$

### Practical Verification Steps

1. Fix the compact retained activity cylinder.
2. Cover it by finitely many covariant unit cylinders.
3. Compare the lower bound with the cover constant.
4. Reject pure local vanishing and pure diffuse exterior carriers.
5. Route local noncompactness to its owning defect node.

## Estimate Step $B_{\mathrm{ST7}}$

The estimate step is the finite covering argument on the retained compact
activity cylinder.

## Failure Case

Failure name: retained mass without active carrier.

Analytic meaning: the proof has a compact activity lower bound but no active
unit cylinder or local defect explaining it.

## Refinement Step

Allowed refinements:

1. refine the finite cover;
2. lower the active threshold according to the cover constant;
3. identify the local pressure or compactness defect;
4. return to `ST5` to extract the active locus.

Progress measure: the retained compact mass is attached to an active local
carrier or to an explicit local defect.

## Data Passed Forward

The next proof step is `ST8`. The data passed forward are

$$
\Gamma_{\mathrm{ST7}}
=
\Gamma_{\mathrm{ST6}}
\cup
\{\text{nonactive alternatives cannot be sole retained-mass carriers}\}.
$$

---

# 45. `ST8` -- Diffuse-Defect Compactness

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is subthreshold activity that escapes every compact observer
window but retains positive total oscillation or CKN mass.

### Standing Assumptions

The remainder is $\mathcal R_{\rm diff}$: there are escaping finite-measure
observer sets $E_n\subset Z$ such that

$$
0<m_0\le\mu_{n,K_n}^{\rm osc}(E_n)<\infty,
\qquad
\sup_{k\in I_{K_n}(E_n)}
\operatorname{osc}_3(U_n;\mathcal Q_1^{\rm cov}(z_k))\to0.
$$

Here $K_n$ is a compact observer window containing $E_n$; the windows
$K_n$ escape every fixed compact subset of $Z$. If the natural escaping set
has infinite oscillation mass, first restrict it to a finite union of
covariant unit cylinders carrying mass in $[m_0,2m_0]$.

The same diffuse state-space route is allowed for the time-slice exterior
measure produced in `ST17`:

$$
d\nu_n(y)=\mathbf 1_{\{|y|>R_n\}}|U(y,\tau_n)|^3\,dy,
$$

after decomposing it by covariant unit balls at centered time $\tau_n$ and
normalizing a finite positive truncation. A later node may thicken a finite
truncation to short observer cylinders if a spacetime CKN measure is needed,
but the diffuse contradiction itself does not require a uniform thickening of
infinitely many subthreshold balls.

### Objects Inspected

Normalize the diffuse measure

$$
\lambda_n
=
\frac{\mu_{n,K_n}^{\rm osc}\lfloor E_n}
{\mu_{n,K_n}^{\rm osc}(E_n)}.
$$

Choose a compactification $\overline Z$ of observer space compatible with
covariant observer translations and extract weak limits of $\lambda_n$.
For a time-slice diffuse input, first embed the normalized ball measure at
observer time $s=0$ into $\overline Z$ and attach the same local velocity hull
data obtained by covariant recentering.

### Dependencies Used

Diffuse status comes from `ST6`; local finite measures from `ST4`; observer
space from `ST1`; finite-cover discharge from `ST7`.

### Local Obstruction Predicate

$P_{\mathrm{ST8}}$ holds if diffuse exterior concentration is recorded as an
unresolved global tail instead of being compactified as a local observer
boundary state.

### Local Lemmas to Prove

**Lemma ST8.1 -- Diffuse measures compactify to observer-boundary states.**
After passing to a subsequence,

$$
\lambda_n\rightharpoonup\lambda
$$

as probability measures on $\overline Z$, and

$$
\lambda(\overline Z)=1,\qquad
\lambda(B)=0
\quad\text{for every }B\Subset Z.
$$

**Proof.** The finite-measure truncation makes each $\lambda_n$ a probability.
Viewing it as a probability on compact $\overline Z$, weak compactness gives
a subsequential limit. Since $E_n$ escapes every compact subset of $Z$, for
any $B\Subset Z$ one has $\lambda_n(B)=0$ for large $n$ after enlarging $B$
slightly. Hence $\lambda(B)=0$. The time-slice variant is the same argument
with $E_n$ contained in the slice $s=0$ after covariant recentering; compact
boundary support records the escaping spatial coordinate, while local hull
coordinates record the recentered state.

**Lemma ST8.2 -- Diffuse defects remain local state objects.**
The diffuse state records observer-boundary support and local hull data; it
does not require a global $L^3$ tail estimate.

**Proof.** The normalized measures use only local oscillation on covariant
unit cylinders, or local time-slice $L^3$ mass decomposed by covariant unit
balls, together with compactness of probability measures on $\overline Z$.
Velocity hull data are extracted from local observer recenterings in
$\mathfrak X_M$. No integral over the whole spatial complement is estimated
as a closing hypothesis; the exterior mass is only a witness used to create a
local observer-boundary state.

### Specific Estimate

The decisive compactness statement is

$$
\lambda_n\rightharpoonup\lambda\in\mathcal P(\overline Z),
\qquad
\operatorname{supp}\lambda\subset\partial\overline Z.
$$

### Practical Verification Steps

1. Verify escaping support of $E_n$.
2. If needed, truncate to a finite union of escaping unit cylinders with
   mass in a fixed positive finite range.
3. Normalize the oscillation or local CKN measure on $E_n$.
4. Choose the observer compactification.
5. Extract the probability-measure limit.
6. Record boundary support and the associated local velocity hull.

## Estimate Step $B_{\mathrm{ST8}}$

The estimate step is probability compactness of diffuse observer measures.

## Failure Case

Failure name: uncompactified diffuse defect.

Analytic meaning: mass escapes every compact observer window but has not been
turned into a state-space object.

## Refinement Step

Allowed refinements:

1. replace CKN mass by oscillation mass if pressure gauges are insufficient;
2. adjust the compactification;
3. pass to a subsequence;
4. return to `ST6` if the support is not actually escaping.

Progress measure: diffuse mass becomes a probability state on the observer
boundary.

## Data Passed Forward

The next proof step is `ST9`. The data passed forward are

$$
\Gamma_{\mathrm{ST8}}
=
\Gamma_{\mathrm{ST7}}
\cup
\{\mathfrak D_M,\ \lambda,\ \text{diffuse-defect state}\}.
$$

---

# 46. `ST9` -- Diffuse-Defect Recurrence and Trichotomy

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is the compact diffuse-defect state space $\mathfrak D_M$ and the
action of the covariant observer group on it.

### Standing Assumptions

A nonzero diffuse-defect state has been produced by `ST8`, and the chosen
observer compactification carries a verified continuous action of the
covariant observer group on the compact diffuse-defect state space.

### Objects Inspected

Let

$$
G=\mathbb R^3\rtimes\mathbb R
$$

denote the covariant observer group generated by $T_a$ and $\Theta_s$.
For a Følner net $F_\alpha\subset G$, form

$$
\mathbb P_\alpha
=
\frac1{|F_\alpha|}
\int_{F_\alpha}\delta_{g\lambda}\,dg.
$$

### Dependencies Used

Observer group from `ST1`; diffuse compactification from `ST8`; local state
space from `ST2`; lower-strata ledger from `ST0` and `ST3`.

### Local Obstruction Predicate

$P_{\mathrm{ST9}}$ holds if a diffuse boundary state is left as an unresolved
tail instead of being assigned to recurrence, lower stratum, or critical-tail
compactification.

### Local Lemmas to Prove

**Lemma ST9.1 -- Diffuse states admit invariant probability measures.**
The Følner averages $\mathbb P_\alpha$ have weak limit points, and every such
limit is $G$-invariant on the compact diffuse state space.

**Proof.** The state space is compact and the verified observer action is
continuous, so the probability measures on it are weakly compact and the
pushforward map is continuous. Amenability of the locally compact group
$G=\mathbb R^3\rtimes\mathbb R$ gives Følner sets. The usual
Krylov--Bogolyubov/Følner argument shows that every weak limit is invariant
under each fixed group element. If the compactification does not carry this
continuous action, the output is a diffuse recurrence gap rather than an
invariant measure.

**Lemma ST9.2 -- Diffuse recurrence has three ordered outcomes.**
Every recurrent diffuse state either regenerates local activity in
$\mathfrak X_M$, falls into the affine/parasitic lower stratum, or is a
critical diffuse tail.

**Proof.** If a recurrent translate has positive unit-cylinder oscillation,
it regenerates local activity and returns to `ST5`. If all recurrent
oscillation vanishes, `ST3` identifies the affine/parasitic stratum. The
remaining case has nonzero boundary-supported diffuse mass with no active
unit cylinder and no affine collapse; this is, by definition, the critical
diffuse tail sent to `ST10`.

### Specific Estimate

The decisive recurrence output is

$$
\mathbb P\in\mathcal P(\mathfrak D_M),
\qquad
g_\#\mathbb P=\mathbb P
\quad(g\in G),
$$

plus exactly one trichotomy label.

### Practical Verification Steps

1. Verify the compact diffuse state space and group action.
2. Choose Følner sets.
3. Average Dirac masses over observer translates.
4. Extract an invariant probability measure.
5. Apply the ordered trichotomy: local activity, affine lower stratum, or
   critical diffuse tail.

## Estimate Step $B_{\mathrm{ST9}}$

The estimate step is amenable-group averaging and trichotomy assignment.

## Failure Case

Failure name: diffuse recurrence gap.

Analytic meaning: a diffuse boundary state survives without an invariant
state or ordered route.

## Refinement Step

Allowed refinements:

1. repair the compact state-space topology;
2. restrict the group action to the verified covariant observer subgroup;
3. rerun the oscillation test of `ST3`;
4. route nonzero critical boundary data to `ST10`.

Progress measure: every diffuse state is recurrently classified.

## Data Passed Forward

If the diffuse trichotomy produces a critical tail, the next proof step is
`ST10`. If it regenerates local activity, the branch returns to `ST5`. If it
falls into the affine/parasitic lower stratum, the branch exits through
`ST3`/`ST0`. The data passed forward are

$$
\Gamma_{\mathrm{ST9}}
=
\Gamma_{\mathrm{ST8}}
\cup
\{\text{diffuse trichotomy: activity / affine lower stratum / critical tail}\}.
$$

---

# 47. `ST10` -- Critical-Tail Compactification

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a critical diffuse tail that has survived recurrence without
regenerating active unit-cylinder mass and without affine collapse.

### Standing Assumptions

The branch is in the critical-tail case of `ST9`.

### Objects Inspected

Define a compact critical-tail state space $\mathfrak T_M$. A state
$\mathcal T\in\mathfrak T_M$ records:

$$
\begin{array}{l}
\text{barycentric velocity data},\\
\text{Young-measure or coherent tail data},\\
\text{logarithmic radial dynamics},\\
\text{local pressure/gauge data},\\
\text{observer-boundary support}.
\end{array}
$$

### Dependencies Used

Diffuse boundary support comes from `ST8`; recurrence from `ST9`; pressure
gauges from `ST2` and `ST4`; lower-strata exclusions from `ST0`.

### Local Obstruction Predicate

$P_{\mathrm{ST10}}$ holds if the proof names a critical tail but does not put
it in a compact state space with local pressure and observer-boundary data.

### Local Lemmas to Prove

**Lemma ST10.1 -- Critical diffuse tails have compactification limits.**
Every critical diffuse tail has a subsequential limit

$$
\mathcal T\in\mathfrak T_M.
$$

**Proof.** Barycentric velocity data are bounded by $M$. Young-measure data are
compact by weak-* compactness of probability-valued measures on compactified
local ranges. Logarithmic radial dynamics are recorded modulo the observer
compactification, and local pressure gauges are compact in the weak
$L^{3/2}_{\rm loc}$ quotient. Tychonoff and diagonal extraction over compact
observer windows give a limit in $\mathfrak T_M$.

**Lemma ST10.2 -- Critical-tail states are local records.**
The state $\mathcal T$ contains no global pressure norm and no global
$L^3$ tail bound as a hypothesis.

**Proof.** Each coordinate of $\mathfrak T_M$ is defined from local observer
cylinders, weak local measures, or boundary support in $\overline Z$. The
whole-space tail is represented by the boundary state, not estimated as an
integral over $\mathbb R^3$.

### Specific Estimate

The decisive compactness statement is

$$
\mathcal T_n\to\mathcal T
\quad\text{in }\mathfrak T_M.
$$

### Practical Verification Steps

1. List the critical-tail coordinates.
2. Verify compactness of each coordinate.
3. Preserve pressure gauges in the local quotient.
4. Record observer-boundary support.
5. Pass the realized critical-tail state to `ST11`.

## Estimate Step $B_{\mathrm{ST10}}$

The estimate step is compactification of the critical diffuse tail into
$\mathfrak T_M$.

## Failure Case

Failure name: unregistered critical tail.

Analytic meaning: a critical diffuse tail is named but lacks compact
state-space coordinates.

## Refinement Step

Allowed refinements:

1. add the missing Young-measure coordinate;
2. add the missing logarithmic radial coordinate;
3. repair local pressure-gauge data;
4. return to `ST9` if the state actually regenerates local activity.

Progress measure: the tail is represented by a point of $\mathfrak T_M$.

## Data Passed Forward

The next proof step is `ST11`. The data passed forward are

$$
\Gamma_{\mathrm{ST10}}
=
\Gamma_{\mathrm{ST9}}
\cup
\{\mathcal T\in\mathfrak T_M\}.
$$

---

# 48. `ST11` -- Realized Critical-Tail Rigidity

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a realized critical-tail state $\mathcal T\in\mathfrak T_M$.

### Standing Assumptions

The critical tail is realized by the original residual branch through the
observer compactification and is outside the routed lower-strata ledger unless
this node routes it there.

### Objects Inspected

Inspect the ordered critical-tail types:

$$
\text{Young critical tail},\quad
\text{coherent homogeneous critical tail},\quad
\text{coherent log-periodic critical tail},\quad
\text{coherent aperiodic critical tail}.
$$

### Dependencies Used

Critical-tail compactification from `ST10`; recurrence from `ST9`; lower
strata from `ST0` and `ST3`; structured branch nodes `PS23`--`PS29`; and the
critical-tail rigidity records verified in this node for the Young,
homogeneous, log-periodic, and aperiodic cases.

### Local Obstruction Predicate

$P_{\mathrm{ST11}}$ holds if a realized critical-tail state survives without
being routed to local activity, lower structured strata, or a rigidity
contradiction.

### Local Lemmas to Prove

**Lemma ST11.1 -- Realized critical tails have ordered type.**
Every $\mathcal T\in\mathfrak T_M$ belongs to one of the Young, coherent
homogeneous, coherent log-periodic, or coherent aperiodic critical-tail types.

**Proof.** The Young/coherent split is determined by whether the tail state is
represented by a non-atomic Young measure or by a single coherent tail
representative. For coherent tails, the logarithmic radial dynamics is fixed,
periodic, or neither. These alternatives are mutually exhaustive.

**Lemma ST11.2 -- Critical-tail discharge requires a rigidity record.**
For each realized critical-tail type, the node must record one of the
following verified outcomes:

$$
\text{local activity regenerated},\qquad
\text{membership in a closed lower structured stratum},\qquad
\text{zero/affine collapse}.
$$

If the corresponding rigidity statement for a Young, homogeneous,
log-periodic, or aperiodic tail has not been proved or registered as an
available theorem, the branch is not discharged and is passed forward as a
critical-tail rigidity obligation.

**Proof.** Positive observer-translate oscillation is a direct local check and
returns the branch to `ST5`. Affine collapse is the lower-stratum test of
`ST3`. Every remaining nonactive critical-tail type requires a specific
rigidity implication identifying it with a closed structured class or with the
zero state. Without that implication, classification by type is only
bookkeeping and does not exclude the branch. With the rigidity record
verified, the state is no longer a generic residual state.

### Specific Estimate

The decisive output is

$$
\mathcal T
\longrightarrow
\left\{
\begin{array}{l}
\text{local activity regenerated and routed to }ST5,\\
\text{closed lower structured stratum with source recorded},\\
\text{zero or affine/parasitic state},\\
\text{or explicit critical-tail rigidity obligation}
\end{array}
\right.
$$

### Practical Verification Steps

1. Decide Young versus coherent.
2. For coherent states, decide homogeneous, log-periodic, or aperiodic radial
   dynamics.
3. Test for local activity in observer translates.
4. For nonactive states, attach the rigidity theorem or proof source for the
   selected type.
5. Route activity to `ST5`.
6. Route verified nonactive structured states to the lower-strata ledger.
7. If the type-specific rigidity source is missing, record a critical-tail
   rigidity obligation rather than closing the branch.

## Estimate Step $B_{\mathrm{ST11}}$

The estimate step is realized critical-tail classification and local discharge.

## Failure Case

Failure name: surviving critical-tail state or missing rigidity source.

Analytic meaning: a critical-tail compactification point has no local activity
route, no lower-stratum route, or no verified rigidity implication for its
type.

## Refinement Step

Allowed refinements:

1. add the missing critical-tail type predicate;
2. test observer translates for oscillation;
3. register or prove the corresponding structured rigidity theorem;
4. return to `ST10` if compactification data are incomplete.

Progress measure: every critical-tail state is discharged, returns to
active-locus extraction, or is recorded as an explicit rigidity obligation.

## Data Passed Forward

If the critical-tail state is discharged into a lower stratum, the branch
exits through the lower-strata ledger. If it regenerates local activity, it
returns to `ST5`. If it produces a rigidity obligation, that obligation is
entered into $\mathrm{Obl}_{\mathrm{ST}}$. Only discharged residual states
continue to `ST12`. The data passed forward are

$$
\Gamma_{\mathrm{ST11}}
=
\Gamma_{\mathrm{ST10}}
\cup
\{\text{critical tail discharged, local activity regenerated, or explicit }
\text{critical-tail rigidity obligation}\}.
$$

---

# 49. `ST12` -- Descendant Heredity

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a retained covariant tail descendant $W$ of a residual state
$U$.

### Standing Assumptions

$U\in\mathfrak X_M$ is residual, has retained activity, and is outside the
routed lower-strata ledger of `ST0`.

### Objects Inspected

Inspect descendant limits of the form

$$
W=\lim_{n\to\infty}\mathscr T_{(x_n,\sigma_n)}U
$$

in the local state-space topology, with retained local activity in the
observer frame.

### Dependencies Used

Observer covariance from `ST1`; compact state space from `ST2`; activity
semicontinuity from `ST5`; lower-strata ledger from `ST0`; pressure gauges
from `ST4`.

### Local Obstruction Predicate

$P_{\mathrm{ST12}}$ holds if a descendant is used later without proving that
it remains in the residual universe or falls into a known lower stratum.

### Local Lemmas to Prove

**Lemma ST12.1 -- Descendants inherit the local state-space class.**
If $U\in\mathfrak X_M$ and $W$ is a covariant observer limit of $U$, then
$W\in\mathfrak X_M$.

**Proof.** Covariant observers preserve the centered equation and local
suitability by `ST1`. The $L^\infty$ bound is unchanged. Pressure gauges pass
by the local quotient topology of `ST2`.

**Lemma ST12.2 -- Descendants are residual or lower-stratum.**
If $U$ is residual and $W$ is a retained active descendant, then either $W$ is
outside the routed lower-strata ledger and hence residual, or the first lower
predicate that holds for $W$ routes it to the lower-strata ledger.

**Proof.** Apply the membership vector of `ST0` to $W$. Heredity of the
state-space class makes $W$ admissible for the same ledger. Retained activity
passes by lower semicontinuity of the local mass or oscillation.

### Specific Estimate

The decisive heredity statement is

$$
U\in\mathfrak X_M,\quad
W=\lim\mathscr T_{(x_n,\sigma_n)}U
\quad\Longrightarrow\quad
W\in\mathfrak X_M
$$

with retained activity or a lower-stratum exit.

### Practical Verification Steps

1. Write the descendant as a covariant observer limit.
2. Verify local state-space compactness and pressure gauges.
3. Pass retained activity by semicontinuity.
4. Run the lower-strata ledger on the descendant.
5. Declare residual heredity only after the ledger excludes lower exits.

## Estimate Step $B_{\mathrm{ST12}}$

The estimate step is observer-limit compactness and lower-strata heredity.

## Failure Case

Failure name: descendant status gap.

Analytic meaning: a tail descendant is used in an active chain or recurrence
argument without residual/lower-stratum status.

## Refinement Step

Allowed refinements:

1. repair the observer limit;
2. add a missing pressure gauge;
3. re-run the lower-strata ledger;
4. discard descendants without retained activity from active-chain arguments.

Progress measure: every retained descendant is residual or lower-stratum.

## Data Passed Forward

The next proof step is `ST13`. The data passed forward are

$$
\Gamma_{\mathrm{ST12}}
=
\Gamma_{\mathrm{ST11}}
\cup
\{\text{descendant heredity}\}.
$$

---

# 50. `ST13` -- Active Successor Relation and Path-Space Recurrence

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is the active successor graph of retained residual descendants.

### Standing Assumptions

Descendant heredity has been proved and lower-stratum exits have been removed.

### Objects Inspected

For a registered compact observer cylinder $Q_\eta$ and threshold $\eta>0$,
define

$$
\mathfrak A_{\eta,Q_\eta}
=
\{U\in\mathfrak X_M:\Phi(U;Q_\eta)\ge\eta\}.
$$

Define $U\,\mathcal R_\eta\,V$ if $V$ is a retained active covariant tail
descendant of $U$ carrying the same registered activity observable, with no
intervening diffuse, critical-tail, or local noncompactness defect. If a
witnessing observer sequence escapes the local state-space topology, it is
routed back to `ST6`--`ST11` and is not counted as an edge of
$\mathcal R_\eta$.

### Dependencies Used

Activity from `ST3` and `ST5`; heredity from `ST12`; compact state space from
`ST2`; observer covariance from `ST1`.

### Local Obstruction Predicate

$P_{\mathrm{ST13}}$ holds if an infinite active descendant chain is allowed
without a compact path-space or recurrence measure.

### Local Lemmas to Prove

**Lemma ST13.1 -- The active successor relation is closed.**

$$
\mathcal R_\eta
\subset
\mathfrak A_{\eta,Q_\eta}\times\mathfrak A_{\eta,Q_\eta}
$$

is closed in the local state-space topology.

**Proof.** Let $U_j\to U$, $V_j\to V$, and
$U_j\mathcal R_\eta V_j$. The descendant relation is represented by covariant
observer sequences whose witnesses are retained in the local state-space
topology by definition of the edge. If those witnesses escaped compactness,
the branch would have been routed to `ST6`--`ST11` instead of producing an
edge. Under retained compactness, diagonal extraction passes the descendant
limit to $V$. Local activity is lower semicontinuous at threshold $\eta$ after
reducing to a rational subthreshold if necessary; in the strong local
topology the fixed-cylinder oscillation observable is continuous. Thus the
closed relation is recorded at the slightly reduced threshold and
$U,V\in\mathfrak A_{\eta,Q_\eta}$.

**Lemma ST13.2 -- Infinite active chains produce recurrent path measures.**
If

$$
U_0\mathcal R_\eta U_1\mathcal R_\eta U_2\cdots,
$$

then the path space

$$
\mathscr P_\eta
=
\{(W_j)_{j\ge0}: W_j\mathcal R_\eta W_{j+1}\}
$$

is compact and shift-invariant, and Krylov--Bogolyubov averaging gives a
shift-invariant probability measure.

**Proof.** Compactness follows from compactness of
$\mathfrak A_{\eta,Q_\eta}$ and closedness of $\mathcal R_\eta$ in the product
topology. The left shift maps paths to paths. Averaging the orbit of any
infinite path under the shift and extracting a weak limit gives a
shift-invariant probability measure.

### Specific Estimate

The decisive recurrence output is

$$
\mathbb Q\in\mathcal P(\mathscr P_\eta),
\qquad
S_\#\mathbb Q=\mathbb Q,
$$

or else no infinite active chain exists.

### Practical Verification Steps

1. Fix the activity threshold $\eta$.
2. Register the compact activity cylinder $Q_\eta$ and define
   $\mathfrak A_{\eta,Q_\eta}$ and $\mathcal R_\eta$.
3. Prove closedness of the successor relation, with escaping witnesses routed
   back to the diffuse/noncompact nodes.
4. If an infinite chain exists, build the compact path space.
5. Average under the shift to obtain recurrence.

## Estimate Step $B_{\mathrm{ST13}}$

The estimate step is closed-relation compactness and shift recurrence.

## Failure Case

Failure name: active chain without recurrence.

Analytic meaning: the proof permits infinitely many retained active
descendants but does not extract a recurrent core.

## Refinement Step

Allowed refinements:

1. lower the activity threshold slightly;
2. repair descendant heredity;
3. add missing pressure-gauge compactness for path limits;
4. prove that the active chain terminates.

Progress measure: either an invariant path measure is produced or infinite
active chains are excluded.

## Data Passed Forward

The next proof step is `ST14`. The data passed forward are

$$
\Gamma_{\mathrm{ST13}}
=
\Gamma_{\mathrm{ST12}}
\cup
\{\text{compact active path space or no infinite active chain}\}.
$$

---

# 51. `ST14` -- Recurrent-Core Rigidity and No Infinite Active Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a compact recurrent active tail core produced by `ST13`.

### Standing Assumptions

There is a shift-invariant active path measure or a compact tail-minimal
recurrent core $\mathcal M\subset\mathfrak A_{\eta,Q_\eta}$ outside the routed
lower-strata ledger.

### Objects Inspected

Inspect local observables at the observer origin,

$$
u_i(W)=W_i(0,0),
\qquad
p_i(W)=\partial_i P_W(0,0),
$$

after choosing local pressure gauges on a fixed compact cylinder.
Point observables are allowed only on branches whose local topology has been
upgraded to smooth bounded-profile compactness. If the branch still carries
only the suitable `PS6` topology, replace them by fixed mollified
compact-cylinder observables and record the mollifier scale in the constants
ledger.

### Dependencies Used

Recurrence from `ST13`; pressure gauges from `ST2` and `ST4`; affine lower
stratum from `ST3`; centered equation from `PS5`; and a recurrent-core
rigidity record whose hypotheses are checked in this node.

### Local Obstruction Predicate

$P_{\mathrm{ST14}}$ holds if a recurrent active tail core remains outside
lower strata, or if the proof invokes recurrent-core rigidity without
checking the invariant-measure identities, pressure gauges, and affine
normalization it requires.

### Local Lemmas to Prove

**Lemma ST14.1 -- Recurrent-core averages satisfy local invariant identities.**
A compact recurrent core carries an invariant probability measure $\nu$ such
that the averaged centered equation at the observer origin is valid for the
local observables.

**Proof.** The invariant measure comes from the path-space recurrence of
`ST13` and the projection to the zero coordinate. Smooth local state-space
topology permits evaluation of velocity and locally gauged pressure gradients
at $(0,0)$. If only suitable compactness is available, the same identity is
first tested against the fixed mollified compact-cylinder observables; a
pointwise identity is not asserted until the bounded-regularity upgrade from
`ST2` has been recorded. Integrating the centered equation against $\nu$ gives
the averaged identity in the verified observable class.

**Lemma ST14.2 -- Recurrent active cores require a rigidity implication.**
The recurrent core is excluded only after verifying the following local
rigidity implication:

$$
\left[
\begin{array}{l}
\mathcal M\subset\mathfrak A_{\eta,Q_\eta}\text{ compact and tail-minimal},\\
\nu\text{ invariant under the verified observer dynamics},\\
\text{local origin observables and pressure gradients are gauge-compatible},\\
\text{the averaged centered equation holds}
\end{array}
\right]
\Longrightarrow
\mathcal M\text{ lies in the affine/parasitic lower stratum}.
$$

If this implication is not available, the infinite active branch is not
excluded; it is recorded as a recurrent-core rigidity obligation.

**Proof.** Lemma ST14.1 supplies the invariant-measure identities and the
well-defined local observables. The displayed rigidity implication is the
separate mathematical step that turns those identities into affine/parasitic
collapse. Once it is verified, `ST3` routes the core to the lower stratum,
contradicting the assumption that the active recurrent core is residual. If
the implication is missing, the averaged identities alone are not a
contradiction.

### Specific Estimate

The decisive check is the pair consisting of the local invariant averaged
identity and the verified recurrent-core rigidity implication yielding
affine/parasitic status.

### Practical Verification Steps

1. Project the invariant path measure to a recurrent core measure.
2. Fix local pressure gauges near the observer origin.
3. Justify pointwise observables from local smooth topology, or use mollified
   compact-cylinder observables in the suitable topology.
4. Average the centered equation.
5. Verify the recurrent-core rigidity implication.
6. Route the resulting affine/constant core to `ST3` and exclude infinite
   active branches.
7. If the implication is missing, record a recurrent-core rigidity obligation.

## Estimate Step $B_{\mathrm{ST14}}$

The estimate step is local invariant-measure averaging and affine rigidity.

## Failure Case

Failure name: recurrent-core rigidity gap.

Analytic meaning: an infinite active descendant branch has produced a
recurrent core that has not been collapsed to a lower stratum.

## Refinement Step

Allowed refinements:

1. strengthen the local topology to justify origin observables;
2. choose compatible local pressure gauges;
3. restrict to a tail-minimal recurrent component;
4. register any missing recurrent-core rigidity theorem in `PS31`.

Progress measure: recurrent cores are lower-stratum, impossible, or recorded
as explicit rigidity obligations.

## Data Passed Forward

The next proof step is `ST15`. The data passed forward are

$$
\Gamma_{\mathrm{ST14}}
=
\Gamma_{\mathrm{ST13}}
\cup
\{\text{no infinite active descendant branch, or recurrent-core rigidity obligation}\}.
$$

---

# 52. `ST15` -- No Finite Separated Retained Profile Family

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a finite family of separated retained active profiles that
might survive after infinite active branches are excluded.

### Standing Assumptions

There is no infinite active descendant branch, or any recurrent-core
obligation has been separately recorded; diffuse critical tails have been
discharged or assigned rigidity obligations; and descendant heredity holds.

### Objects Inspected

Inspect finitely many separated covariant observer cylinders

$$
\mathcal Q_1^{\rm cov}(z_{n,j}),
\qquad 1\le j\le N,
$$

with uniform lower local CKN or oscillation bounds and pairwise separation.

### Dependencies Used

Finite packet information from `PS18`--`PS20` when present; active loci from
`ST5`; descendant heredity from `ST12`; no infinite active branches from
`ST14`; diffuse exclusion from `ST8`--`ST11`.

### Local Obstruction Predicate

$P_{\mathrm{ST15}}$ holds if a finite separated retained family is left as a
surviving residual object, or if such a family is excluded without the
separated-family exclusion check below.

### Local Lemmas to Prove

**Lemma ST15.1 -- A nontrivial retained tail limit splits a finite family.**
If one retained profile has a nontrivial separated tail limit, then after
covariant recentering the branch produces two separated active descendants.

**Proof.** The original retained cylinder and the escaping retained tail
cylinder have disjoint covariant neighborhoods for large $n$. Local compactness
on each cylinder gives two active state-space limits, and the separation
persists in observer coordinates.

**Lemma ST15.2 -- Finite separated families require a no-separated-family
record.**
A finite separated retained family is excluded only after the branch verifies
the local no-separated-family implication:

$$
\left[
\begin{array}{l}
\text{finite covariantly separated retained active profiles},\\
\text{descendant heredity for every member},\\
\text{no diffuse, critical-tail, or local noncompact component},\\
\text{no infinite active branch}
\end{array}
\right]
\Longrightarrow
\text{contradiction}.
$$

If this implication is not verified, the family is recorded as a finite
separated-family obligation.

**Proof.** Lemma ST15.1 shows that any member with a retained separated tail
splits the family and starts an active successor. Iterating either reaches an
infinite active branch, handled by `ST14`, or leaves a finite terminal
separated family with no diffuse, critical-tail, or noncompact component. The
displayed no-separated-family implication is the remaining local exclusion
needed for that terminal finite case. Without it, termination of the
iteration is not itself a contradiction.

### Specific Estimate

The decisive local separation record is

$$
\operatorname{dist}_{\rm par}
(\mathcal Q_1^{\rm cov}(z_{n,i}),
\mathcal Q_1^{\rm cov}(z_{n,j}))\to\infty
\quad(i\ne j),
$$

with a fixed lower activity bound on each cylinder.

### Practical Verification Steps

1. List the separated active cylinders.
2. Extract local profiles on each cylinder.
3. Apply descendant heredity to every profile.
4. Check whether any member has a retained tail successor.
5. Use `ST14` and diffuse-tail discharge to remove nonterminal alternatives.
6. Verify the no-separated-family implication for the terminal finite case.
7. If the implication is missing, record a finite separated-family obligation.

## Estimate Step $B_{\mathrm{ST15}}$

The estimate step is separated-cylinder profile extraction and finite-family
elimination.

## Failure Case

Failure name: finite separated family survives or lacks exclusion source.

Analytic meaning: a finite packet of active residual profiles remains without
being made into an active chain, diffuse state, or lower stratum.

## Refinement Step

Allowed refinements:

1. verify pairwise covariant separation;
2. extract the missing descendant;
3. route diffuse members to `ST8`;
4. lower the activity threshold and rerun `ST13`.

Progress measure: finite separated families are excluded, transformed into
already handled alternatives, or recorded as explicit obligations.

## Data Passed Forward

The next proof step is `ST16`. The data passed forward are

$$
\Gamma_{\mathrm{ST15}}
=
\Gamma_{\mathrm{ST14}}
\cup
\{\text{finite separated retained families excluded, or finite-family obligation}\}.
$$

---

# 53. `ST16` -- Terminal Indecomposability

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is any retained residual profile that survives all active,
diffuse, critical-tail, infinite-chain, and finite-family exclusions.

### Standing Assumptions

Nodes `ST5`--`ST15` have been completed with no remaining diffuse-tail,
critical-tail, recurrent-core, finite-family, or local noncompactness
obligation on the branch being declared terminal.

### Objects Inspected

A retained profile is terminally indecomposable if it has no retained
concentrating tail limit, no diffuse exterior concentration, and no escaping
recentering with local compactness or pressure failure.

### Dependencies Used

Active/remainder decomposition from `ST6`; nonactive discharge from `ST7`;
diffuse and critical-tail closure from `ST8`--`ST11`; heredity and recurrence
from `ST12`--`ST14`; finite-family exclusion from `ST15`.

### Local Obstruction Predicate

$P_{\mathrm{ST16}}$ holds if a residual profile is declared terminal without
checking every decomposition channel, or while one of the explicit obligations
from `ST11`, `ST14`, or `ST15` remains open.

### Local Lemmas to Prove

**Lemma ST16.1 -- Surviving retained residual profiles are single.**
After `ST15` closes without a finite-family obligation, no finite separated
retained family remains. Therefore any surviving retained residual profile has
only one active component.

**Proof.** Two or more separated active components form a finite separated
family. `ST15` excludes such families only when its no-separated-family record
has been verified; otherwise `ST16` is not allowed to run. Nonseparated
components belong to the same local active component by the active-locus
decomposition.

**Lemma ST16.2 -- Surviving single profiles are terminally indecomposable.**
The remaining single retained profile has no retained tail successor, no
diffuse exterior concentration, and no local noncompactness defect.

**Proof.** A retained tail successor would generate an active descendant and
hence either an infinite active branch or a finite separated family, both
excluded. Diffuse exterior concentration is excluded by `ST8`--`ST11`.
Local compactness or pressure failure is a named `ST6` noncompactness defect
and cannot be part of a completed terminal profile.

### Specific Estimate

The decisive ordered consequence is

$$
\text{surviving residual}
\Longrightarrow
\text{single terminally indecomposable retained profile}.
$$

### Practical Verification Steps

1. Check that mixed compact--diffuse states were discharged.
2. Check that compact--noncompact states were assigned as defects.
3. Check that infinite active chains were excluded with no recurrent-core
   rigidity obligation.
4. Check that finite separated families were excluded with no finite-family
   obligation.
5. Record the terminally indecomposable profile.

## Estimate Step $B_{\mathrm{ST16}}$

The estimate step is ordered consequence bookkeeping from the previous ST
nodes.

## Failure Case

Failure name: premature terminal profile.

Analytic meaning: a residual profile is being treated as indecomposable while
some active, diffuse, separated, or noncompact channel has not been closed.

## Refinement Step

Allowed refinements:

1. rerun active-locus extraction;
2. rerun diffuse compactification;
3. rerun active successor recurrence;
4. rerun finite-family exclusion;
5. name the remaining local noncompactness defect.

Progress measure: every decomposition channel is closed before
indecomposability is declared.

## Data Passed Forward

The next proof step is `ST17`. The data passed forward are

$$
\Gamma_{\mathrm{ST16}}
=
\Gamma_{\mathrm{ST15}}
\cup
\{\text{single terminally indecomposable retained profile}\}.
$$

---

# 54. `ST17` -- Backward Sequence-$L^3$ from Terminal Indecomposability

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a single terminally indecomposable retained centered profile
$(U,\Pi)$.

### Standing Assumptions

The profile is bounded, belongs to $\mathfrak X_M$, is retained, and is
terminally indecomposable in the sense of `ST16`.

### Objects Inspected

Inspect backward centered time slices $\tau\to-\infty$ and exterior unit-ball
local $L^3$ masses.

### Dependencies Used

Terminal indecomposability from `ST16`; observer recentering from `ST1`;
local compactness from `ST2`; diffuse exclusion from `ST8`--`ST11`.

### Local Obstruction Predicate

$P_{\mathrm{ST17}}$ holds if the proof assumes a global backward
$L^3$ sequence instead of deriving it from terminal indecomposability.

### Local Lemmas to Prove

**Lemma ST17.1 -- Absence of a bounded sequence creates local tail alternatives.**
If there is no sequence $\tau_k\to-\infty$ with

$$
\sup_k\|U(\cdot,\tau_k)\|_{L^3(\mathbb R^3)}<\infty,
$$

then one can choose $R_n\to\infty$ first and then choose
$\tau_n\to-\infty$ so negative that

$$
\int_{\mathbb R^3}|U(y,\tau_n)|^3\,dy
>
M^3|B_{R_n}|+1.
$$

Since $|U|\le M$, this implies

$$
\int_{|y|>R_n}|U(y,\tau_n)|^3\,dy\ge1.
$$

Define

$$
m_n
=
\sup_{|x|>R_n}
\int_{B_1(x)}|U(y,\tau_n)|^3\,dy .
$$

Either $\limsup_n m_n>0$, producing a local tail core at some positive
threshold $\varepsilon_{\rm tail}$, or $m_n\to0$, producing diffuse exterior
concentration.

**Proof.** If no bounded sequence exists, then
$\|U(\cdot,\tau)\|_{L^3(\mathbb R^3)}\to\infty$ as
$\tau\to-\infty$. Fix any increasing sequence $R_n\to\infty$. For each $n$,
choose $\tau_n$ so negative that the integral of $|U|^3$ is larger than
$M^3|B_{R_n}|+1$. The interior contribution on $B_{R_n}$ is at most
$M^3|B_{R_n}|$, so the exterior integral is at least $1$. The dichotomy for
$m_n$ is the elementary split between a positive unit-ball concentration and
vanishing unit-ball supremum with nonzero exterior mass. If
$\limsup_n m_n>0$, pass to a subsequence and choose a rational
$\varepsilon_{\rm tail}>0$ below that limsup.

**Lemma ST17.2 -- Local tail alternatives contradict indecomposability.**
The local tail core alternative contradicts terminal indecomposability by
producing a retained concentrating tail limit or a local compactness failure.
The diffuse alternative contradicts terminal indecomposability by producing
diffuse exterior concentration. In the local tail-core alternative the
time-slice lower bound is first thickened to a short covariant observer
cylinder using bounded local regularity; the length of the short interval is
recorded in the constants ledger. In the diffuse alternative one records a
normalized escaping time-slice diffuse measure as an `ST8` input. If a later
argument specifically needs spacetime CKN mass, it first restricts to a
finite positive truncation and thickens only that finite truncation.

**Proof.** In the local tail core case choose $x_n$ with

$$
\int_{B_1(x_n)}|U(y,\tau_n)|^3\,dy\ge\varepsilon_{\rm tail}.
$$

Covariantly recenter

$$
U_n(y,s)
=
U(y+e^{s/2}x_n,\tau_n+s).
$$

By bounded local regularity, the lower bound persists, after reducing the
constant, on a fixed short observer cylinder around $(0,0)$. If the
recentered sequence is compact in $\mathfrak X_M$, a retained tail descendant
is obtained, contradicting `ST16`. If it is not compact, `ST16` is
contradicted by a local compactness or pressure failure. In the case
$m_n\to0$, exterior mass persists while every unit ball on the time slice is
subthreshold. Normalizing a finite positive exterior truncation gives the
time-slice diffuse state accepted by `ST8`; terminal indecomposability
excludes that state as diffuse exterior concentration.

### Specific Estimate

The decisive conclusion is

$$
\exists\,\tau_k\to-\infty
\quad
\sup_k\|U(\cdot,\tau_k)\|_{L^3(\mathbb R^3)}<\infty.
$$

This is an output of local state-space exclusion, not an a priori global
estimate.

### Practical Verification Steps

1. Assume no bounded backward $L^3$ sequence.
2. Choose exterior mass witnesses $(\tau_n,R_n)$.
3. Compute the unit-tail supremum $m_n$.
4. If $m_n$ has a positive limsup, choose a positive rational
   $\varepsilon_{\rm tail}$, thicken the time-slice lower bound to an observer
   cylinder, covariantly recenter, and contradict indecomposability.
5. If $m_n\to0$, identify diffuse exterior concentration and contradict
   indecomposability.
6. Record the resulting bounded backward $L^3$ sequence.

## Estimate Step $B_{\mathrm{ST17}}$

The estimate step is the local tail-core/diffuse dichotomy forcing the
backward sequence-$L^3$ bound.

## Failure Case

Failure name: unforced endpoint $L^3$ sequence.

Analytic meaning: the endpoint theorem is being prepared with a backward
$L^3$ sequence that has not been derived from the local residual state-space
alternatives.

## Refinement Step

Allowed refinements:

1. complete terminal indecomposability in `ST16`;
2. identify the positive unit-tail core;
3. identify the diffuse exterior state;
4. repair the covariant recentering from `ST1`.

Progress measure: absence of the $L^3$ sequence is converted into a forbidden
local tail alternative.

## Data Passed Forward

The next proof step is `ST18`. The data passed forward are

$$
\Gamma_{\mathrm{ST17}}
=
\Gamma_{\mathrm{ST16}}
\cup
\left\{
\tau_k\to-\infty,\quad
\sup_k\|U(\cdot,\tau_k)\|_{L^3(\mathbb R^3)}<\infty
\right\}.
$$

---

# 55. `ST18` -- Parasitic-Free Mildness Inheritance

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is the physical pullback of a retained terminal profile or of any
retained covariant tail descendant, active-tree limit, or recurrent-core
limit that survived lower-stratum removal.

### Standing Assumptions

The affine/parasitic lower stratum has been removed, the profile is bounded in
$\mathfrak X_M$, the centered equation is exact in covariant observer
coordinates, local pressure gauges are compatible, and the branch must verify
the finite-shift mildness gate below. Bounded local smoothness alone is not
counted as mildness until the Duhamel formula has been proved in the endpoint
topology.

### Objects Inspected

For

$$
u(x,t)
=
(-t)^{-1/2}
U\left(\frac{x}{\sqrt{-t}},-\log(-t)\right),
$$

and each fixed $T<0$, define

$$
u^T(x,s)=u(x,T+s),
\qquad s<0.
$$

### Dependencies Used

Affine/parasitic removal from `ST3`; observer covariance from `ST1`; centered
equation from `PS5`; local pressure gauges from `ST2`; boundedness from
`ST2`; sequence-$L^3$ from `ST17`.

### Local Obstruction Predicate

$P_{\mathrm{ST18}}$ holds if the endpoint theorem is invoked before proving
that the physical pullback is parasitic-free and bounded mild ancient on every
finite terminal shift.

### Local Lemmas to Prove

**Lemma ST18.1 -- Finite-shift pullbacks are bounded ancient solutions.**
For every fixed $T<0$, the pullback $u^T$ is bounded on
$\mathbb R^3\times(-\infty,0)$.

**Proof.** If $\|U\|_\infty\le M$ and $s<0$, then $T+s<T<0$, so

$$
|u^T(x,s)|\le (-T)^{-1/2}M.
$$

The centered-to-physical change of variables from `PS5` gives the
Navier--Stokes equation on the shifted ancient interval.

**Lemma ST18.2 -- Parasitic-free normalized profiles inherit mildness only
through the mildness gate.**
After affine/parasitic lower-stratum removal, the finite-shift physical
pullback may be passed to the endpoint theorem only if it satisfies the
Duhamel formula required by that theorem:

$$
u^T(t)=e^{(t-s)\Delta}u^T(s)
-\int_s^t e^{(t-\sigma)\Delta}
\mathbb P\nabla\cdot(u^T\otimes u^T)(\sigma)\,d\sigma
$$

for all $s<t<0$ in the registered endpoint topology.

**Proof.** The centered-to-physical transform and Lemma ST18.1 give a bounded
ancient distributional solution on every finite-shift slab. The additional
mildness gate is the verification that no affine/parasitic forcing, harmonic
pressure drift, or non-decaying pressure representative remains in the
projected equation and that the heat-kernel identity holds in the topology
used by the endpoint theorem. Any residual affine forcing would be a lower
stratum from `ST3`. If the heat-kernel identity or pressure compatibility is
not proved, this lemma outputs a mildness inheritance gap rather than an
endpoint-ready branch.

### Specific Estimate

The decisive endpoint-class statement is

$$
u^T\text{ is a bounded mild ancient solution on }
\mathbb R^3\times(-\infty,0)
\quad(T<0).
$$

This statement includes the Duhamel formula and pressure-projection
compatibility, not merely bounded distributional ancientness.

### Practical Verification Steps

1. Confirm affine/parasitic lower modes have been removed or normalized.
2. Pull the centered equation back to physical variables.
3. Fix $T<0$ before estimating the bound.
4. Verify the pressure gauge is compatible with the mild formulation.
5. Prove the Duhamel formula on the shifted ancient interval.

## Estimate Step $B_{\mathrm{ST18}}$

The estimate step is finite-shift boundedness and mildness inheritance for
the physical pullback.

## Failure Case

Failure name: mildness inheritance gap.

Analytic meaning: the branch has the local sequence-$L^3$ output but does not
yet satisfy the bounded mild ancient solution hypothesis of the endpoint
theorem.

## Refinement Step

Allowed refinements:

1. remove affine/parasitic modes through `ST3`;
2. fix a finite terminal shift $T<0$;
3. repair pressure-gauge compatibility;
4. prove the Duhamel formula in the endpoint topology.

Progress measure: the physical pullback is endpoint-ready on every finite
terminal shift.

## Data Passed Forward

If the mildness gate is verified, the next proof step is `ST19`. If the
Duhamel or pressure-projection compatibility check fails, the branch is not
endpoint-ready and the mildness inheritance gap is entered into
$\mathrm{Obl}_{\mathrm{ST}}$. The data passed forward on the successful path
are

$$
\Gamma_{\mathrm{ST18}}
=
\Gamma_{\mathrm{ST17}}
\cup
\{\text{finite-shift bounded mild ancient pullback}\}.
$$

---

# 56. `ST19` -- Endpoint Liouville after Local Sequence-$L^3$

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is the endpoint-ready physical pullback of a terminally
indecomposable retained residual profile.

### Standing Assumptions

`ST17` supplies a backward centered sequence with bounded $L^3$ norm, and
`ST18` supplies a parasitic-free bounded mild ancient physical pullback on
each finite terminal shift.

### Objects Inspected

Under the physical pullback, set

$$
t_k=-e^{-\tau_k}.
$$

Then $\tau_k\to-\infty$ gives $t_k\to-\infty$, and critical invariance gives

$$
\|u(\cdot,t_k)\|_{L^3(\mathbb R^3)}
=
\|U(\cdot,\tau_k)\|_{L^3(\mathbb R^3)}.
$$

### Dependencies Used

Sequence-$L^3$ from `ST17`; mildness from `ST18`; the named
Albritton--Barker endpoint ancient theorem as an external endpoint input;
retained activity from `PS8`.

### Local Obstruction Predicate

$P_{\mathrm{ST19}}$ holds if the Albritton--Barker endpoint theorem is
applied without the locally derived sequence-$L^3$ input, finite-shift
mildness, or retained-activity contradiction.

### Local Lemmas to Prove

**Lemma ST19.1 -- The endpoint theorem applies to the finite-shift pullback.**
For every fixed $T<0$, the shifted solution $u^T$ is bounded mild ancient and
has a sequence $s_k=t_k-T\to-\infty$ with

$$
\sup_k\|u^T(\cdot,s_k)\|_{L^3(\mathbb R^3)}<\infty.
$$

**Proof.** Mildness and boundedness come from `ST18`. Critical $L^3$
invariance and `ST17` give the displayed bound.

**Lemma ST19.2 -- Endpoint zero contradicts retained activity.**
The Albritton--Barker endpoint theorem gives $u^T\equiv0$ for each fixed
$T<0$, hence $U\equiv0$. This contradicts retained compact activity

$$
\iint_Q |U|^3+
|\Pi-c_Q(\tau)|^{3/2}\,dy\,d\tau
\ge\varepsilon_*.
$$

**Proof.** The endpoint theorem gives zero on each shifted ancient slab.
Since $T<0$ is arbitrary, $u\equiv0$ for all physical times $t<0$. The
self-similar transform is invertible for $t<0$, so $U\equiv0$. The zero
profile has zero velocity activity and zero pressure oscillation in the local
gauge, contradicting the retained lower bound.

### Specific Estimate

The decisive endpoint chain is

$$
\sup_k\|U(\tau_k)\|_{L^3}<\infty
\Longrightarrow
\sup_k\|u^T(s_k)\|_{L^3}<\infty
\Longrightarrow
u^T\equiv0
\Longrightarrow
U\equiv0,
$$

contradicting retained local activity.

### Practical Verification Steps

1. Convert $\tau_k$ to physical times $t_k$.
2. Fix a finite terminal shift $T<0$.
3. Verify $s_k=t_k-T\to-\infty$.
4. Apply critical $L^3$ invariance.
5. Apply the endpoint theorem.
6. Compare the zero conclusion with the retained activity cylinder.

## Estimate Step $B_{\mathrm{ST19}}$

The estimate step is endpoint theorem application after the locally forced
sequence-$L^3$ and mildness inputs.

## Failure Case

Failure name: endpoint residual mismatch.

Analytic meaning: the terminal residual profile has not been matched to the
Albritton--Barker theorem or the zero conclusion has not been tied to the
retained activity witness.

## Refinement Step

Allowed refinements:

1. rerun `ST17` for the sequence-$L^3$ output;
2. rerun `ST18` for finite-shift mildness;
3. repair critical norm invariance bookkeeping;
4. identify the retained activity cylinder and pressure gauge.

Progress measure: the endpoint theorem gives a Type C zero/activity
contradiction.

## Data Passed Forward

The next proof step is `ST20`. The data passed forward are

$$
\Gamma_{\mathrm{ST19}}
=
\Gamma_{\mathrm{ST18}}
\cup
\{\text{terminally indecomposable retained profile excluded}\}.
$$

---

# 57. `ST20` -- Local Residual Closure Theorem

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a bounded centered Seregin profile or terminal residual state
$(U,\Pi)\in\mathfrak X_M$ with retained compact concentration and outside the
routed lower-strata ledger.

### Standing Assumptions

The hypotheses are strictly local except for boundedness of the centered
profile:

$$
(U,\Pi)\in\mathfrak X_M,\qquad
\|U\|_{L^\infty}\le M,
$$

local centered equation, local suitability, local pressure gauges on compact
observer cylinders, retained compact CKN or oscillation activity, and
membership outside the routed lower-strata ledger. The closure assertion is
made only after the local ST-obligation ledger is empty.

No global $L^3$ bound, no uniform tightness estimate, and no whole-space
pressure representative is assumed.

### Objects Inspected

Inspect the full residual closure chain:

$$
ST5\text{--}ST6,\quad
ST7,\quad
ST8\text{--}ST11,\quad
ST12\text{--}ST14,\quad
ST15,\quad
ST16,\quad
ST17,\quad
ST18,\quad
ST19,
\quad
\mathrm{Obl}_{\mathrm{ST}}.
$$

### Dependencies Used

All previous ST nodes, the centered equation and local suitability from
`PS5`--`PS7`, retained activity from `PS8`, local pressure gauges from `PS4`
and `PS30`, and the named Albritton--Barker endpoint ancient theorem used in
`ST19`. `PS31` later audits the theorem match; it is not an input to the
proof of `ST20`.

### Local Obstruction Predicate

$P_{\mathrm{ST20}}$ holds if the generic residual class is declared excluded
while any ST obligation remains open, if it is left as open after all
state-space alternatives have been exhausted, or if its theorem statement
contains any global tightness, global $L^3$, or global pressure-tail
hypothesis.

### Local Lemmas to Prove

**Lemma ST20.1 -- Local residual alternatives are exhaustive.**
Every retained generic residual profile has an active-locus/remainder
decomposition. The remainder is vanishing, diffuse, or locally noncompact.
Diffuse and critical-tail alternatives are either discharged or recorded in
$\mathrm{Obl}_{\mathrm{ST}}$; active descendants cannot form infinite chains
or finite separated retained families unless the corresponding rigidity or
finite-family obligation is recorded.

**Proof.** Active-locus extraction is `ST5`, paired decomposition is `ST6`,
nonactive sole-carrier discharge is `ST7`, diffuse closure is `ST8`--`ST9`,
critical-tail closure is `ST10`--`ST11`, descendant heredity is `ST12`,
infinite active-branch exclusion is `ST13`--`ST14`, and finite-family
exclusion is `ST15`. The conclusion is valid only when
$\mathrm{Obl}_{\mathrm{ST}}=\emptyset$, including no local noncompactness,
diffuse recurrence, critical-tail rigidity, recurrent-core rigidity, or
finite-family obligation.

**Lemma ST20.2 -- The only surviving residual is endpoint-ready and excluded.**
Any surviving residual profile is terminally indecomposable by `ST16`.
`ST17` forces a backward sequence-$L^3$ bound, `ST18` gives parasitic-free
finite-shift mildness, and `ST19` applies the endpoint ancient theorem to
contradict retained activity.

**Proof.** This is the ordered chain

$$
\text{surviving residual}
\Rightarrow
\text{terminally indecomposable}
\Rightarrow
\exists \tau_k\to-\infty:
\sup_k\|U(\tau_k)\|_{L^3}<\infty
$$

followed by finite-shift mildness and the Albritton--Barker zero conclusion.
The zero conclusion contradicts the compact retained activity lower bound.
The implication is closed only when `ST18` has produced the finite-shift
bounded mild ancient pullback rather than a mildness inheritance gap.

### Specific Estimate

The decisive theorem statement is

$$
\left[
\begin{array}{l}
(U,\Pi)\in\mathfrak X_M,\quad \|U\|_\infty\le M,\\
\text{local suitability and local pressure gauges},\\
\text{retained compact activity},\\
(U,\Pi)\notin\bigcup_j\mathcal L_j^{\rm routed},\\
\mathrm{Obl}_{\mathrm{ST}}=\emptyset
\end{array}
\right]
\Longrightarrow
\text{contradiction}.
$$

### Practical Verification Steps

1. Run `ST5`--`ST6` to obtain active-locus plus residual decomposition.
2. Use `ST7` to discharge pure nonactive carriers.
3. Verify that no local compactness or pressure noncompactness defect remains
   from `ST6`.
4. Use `ST8`--`ST11` to discharge diffuse and critical-tail boundary states.
5. Verify that no diffuse compactification, diffuse recurrence, or
   critical-tail rigidity obligation remains from `ST8`--`ST11`.
6. Use `ST12`--`ST14` to exclude infinite active descendant branches.
7. Verify that no recurrent-core rigidity obligation remains from `ST14`.
8. Use `ST15` to exclude finite separated retained families.
9. Verify that no finite-family obligation remains from `ST15`.
10. Use `ST16` to reduce to one terminally indecomposable retained profile.
11. Use `ST17` to force the backward sequence-$L^3$ bound.
12. Use `ST18` to inherit parasitic-free mildness and verify that no
    mildness inheritance gap remains.
13. Use `ST19` to apply endpoint Liouville and contradict retained activity.

## Estimate Step $B_{\mathrm{ST20}}$

The estimate step is the assembly of the local residual state-space closure.
It is not a global tail estimate.

## Failure Case

Failure name: unclosed local residual state.

Analytic meaning: one of the local state-space alternatives has not been
discharged, an ST obligation remains open, the mildness gate has not been
passed, or the residual theorem has been stated with an illicit global input.

## Refinement Step

Allowed refinements:

1. identify the first missing ST node in the chain;
2. repair the local pressure-gauge or observer-state data;
3. rerun lower-strata ledger subtraction;
4. prove or explicitly record the missing local compactness, diffuse,
   critical-tail, recurrent-core, finite-family, or mildness obligation;
5. rerun endpoint matching in `PS31` using only local hypotheses;
6. if the endpoint theorem itself is unavailable, record that theorem status
   in `PS31` rather than reopening the residual class.

Progress measure: the generic residual class has status
$\mathrm{excluded}$ with source `ST20`.

## Data Passed Forward

The next proof step is `PS31`. The data passed forward are

$$
\Gamma_{\mathrm{ST20}}
=
\Gamma_{\mathrm{ST19}}
\cup
\left\{
\begin{array}{l}
\text{generic residual class empty},\\
\mathrm{status}(\mathcal R_{\rm loc})=\mathrm{excluded},\\
\mathrm{Obl}_{\mathrm{ST}}=\emptyset
\end{array}
\right\}.
$$

---

# 58. `PS31` -- Endpoint Hypothesis Verification

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is the selected endpoint candidate together with an admissibility
record for applying it: CKN regularity, Serrin regularity, small-data
theory, stationary $L^3$ Liouville, endpoint ancient $L^3$ Liouville,
structured Liouville, Type II local exclusion, or residual closure.

### Standing Assumptions

The incoming record states that every defect channel has a conclusion and no
unnamed distributional term appears in the limit equation. If a channel is
marked unresolved, the endpoint theorem must either list that unresolved
object as an explicit hypothesis or the branch is blocked at this node. For
the generic residual branch, the incoming record is the local theorem output
of `ST20`.

### Objects Inspected

Inspect solution class, pressure normalization, domain, boundary condition,
trace class, flattening map, boundary local energy inequality, boundary
smallness criterion, endpoint source topology, critical norm, compactness
topology, decay or tightness when the selected theorem requires it, symmetry
hypotheses, positivity normalization, and the local residual closure theorem
`ST20`.

### Dependencies Used

The earlier estimates feed `PS31`; `PS30` is decisive because endpoint
hypotheses cannot be checked while the defect vector is incomplete. The
generic residual endpoint is supplied by `ST20`, which is decisive because it
replaces global tightness, global $L^3$, and global pressure-tail obligations
by local state-space hypotheses. The local Type II endpoint is supplied by
`TII16` only when its Type II obligation ledger is empty. In that case it is
decisive because it replaces global Type II profile budgets, whole-space tail
estimates, and global pressure reconstructions by compact-window Type II
state-space alternatives. If the ledger is nonempty, the corresponding entries
are routed as explicit obligations and no Type II endpoint theorem is
available for that branch.

### Endpoint Theorem Registry

Endpoint matching is recorded in a theorem registry

$$
\mathscr T=\{T_\beta\}_{\beta\in B}.
$$

Each entry is a tuple

$$
T_\beta
=
\left(
\mathrm{status},
\mathrm{equation},
\mathrm{domain},
\mathrm{time\ interval},
\mathrm{solution\ class},
\mathrm{pressure\ convention},
\mathrm{boundary\ record},
\mathrm{source\ topology},
\mathrm{norms},
\mathrm{quantifiers},
\mathrm{constants},
\mathrm{conclusion}
\right).
$$

The status field belongs to

$$
\{\mathrm{established},\ \mathrm{proved\ earlier},\
\mathrm{conditional},\ \mathrm{conjectural},\ \mathrm{open}\}.
$$

`PS31` may pass $T_\beta$ to `PS32` only in one of the following cases:

1. $\mathrm{status}(T_\beta)\in
   \{\mathrm{established},\mathrm{proved\ earlier}\}$;
2. $\mathrm{status}(T_\beta)=\mathrm{conditional}$ and the entire downstream
   theorem statement is explicitly tagged as conditional on $T_\beta$.

An entry with status $\mathrm{conjectural}$ or $\mathrm{open}$ cannot close a
branch. A branch relying on such an entry is routed to `PS33` or `PS34` as an
explicit endpoint obligation.

The boundary record field is empty only for interior or whole-space
theorems. For boundary theorems it records the boundary condition, trace
class, flattening map, boundary local energy inequality, pressure convention,
and boundary smallness criterion. The source topology field records the space
$Y_{T_\beta}$ used later by `Bound_B`.

The residual closure entry is

$$
T_{\mathrm{ST20}}
=
\left(
\begin{array}{l}
\mathrm{proved\ earlier},\\
\text{centered Navier--Stokes in covariant observer variables},\\
Z=\mathbb R^3\times\mathbb R\text{ locally on compact observer cylinders},\\
\mathbb R\text{ in centered time},\\
\mathfrak X_M\text{ with retained compact activity},\\
\text{pressure modulo local time-dependent gauges},\\
\emptyset,\quad \text{no source term},\\
\text{local }C^\infty_{\rm loc}\text{ velocity and weak }L^{3/2}_{\rm loc}
\text{ pressure topology},\\
\|U\|_{L^\infty}\le M,\quad
\text{local suitability},\quad
\text{local pressure gauges},\\
\text{outside the routed lower-strata ledger},\\
\mathrm{Obl}_{\mathrm{ST}}=\emptyset,\\
\text{generic residual class empty}
\end{array}
\right).
$$

The first field $\mathrm{proved\ earlier}$ is assigned only when the
`ST20` output ledger records

$$
\mathrm{Obl}_{\mathrm{ST}}
=
\emptyset,
$$

where $\mathrm{Obl}_{\mathrm{ST}}$ contains every local state-space gap that
would block the `ST20` chain: missing local state-space topology or pressure
gauge from `ST2`--`ST4`, active/remainder or local noncompactness defects
from `ST5`--`ST6`, diffuse compactification or recurrence gaps from
`ST8`--`ST9`, critical-tail rigidity obligations from `ST11`,
recurrent-core rigidity obligations from `ST14`, finite-family obligations
from `ST15`, and the mildness inheritance gap from `ST18`. If this set is
nonempty, $T_{\mathrm{ST20}}$ is not a closing theorem entry; the nonempty
elements of $\mathrm{Obl}_{\mathrm{ST}}$ are routed as explicit obligations.

The hypothesis list for $T_{\mathrm{ST20}}$ is exactly

$$
(U,\Pi)\in\mathfrak X_M,\quad
\|U\|_{L^\infty}\le M,\quad
\text{local suitability},\quad
\text{local pressure gauges},\quad
\text{retained compact CKN or oscillation activity},\quad
(U,\Pi)\notin\bigcup_j\mathcal L_j^{\rm routed},\quad
\mathrm{Obl}_{\mathrm{ST}}=\emptyset.
$$

It contains no global $L^3$ bound, no uniform tightness hypothesis, and no
whole-space pressure representative.

The local Type II closure entry is

$$
T_{\mathrm{TII16}}
=
\left(
\begin{array}{l}
\mathrm{proved\ earlier},\\
\text{Navier--Stokes in local Type II represented variables},\\
\text{compact normalized observer windows},\\
\text{selected Type II windows below the terminal face},\\
\text{positive local Type II concentration sequence},\\
\text{pressure modulo compact-ball spatial means},\\
\emptyset\text{ after all local source/defect coordinates are discharged},\\
\text{local }L^3\text{ velocity and }L^{3/2}\text{ pressure compactness},\\
C+D\ge\eta_0,\quad C\ge\eta_v,\\
\text{non-Type-I predicate in original variables},\\
\mathrm{Obl}_{\mathrm{TII}}=\emptyset,\\
\text{local Type II branch excluded}
\end{array}
\right).
$$

The hypothesis list for $T_{\mathrm{TII16}}$ is exactly the raw positive local
Type II packet from `TII0`, compact-window suitability and pressure data from
`TII1`--`TII6`, the local state-space decomposition from `TII15`, and

$$
\mathrm{Obl}_{\mathrm{TII}}=\emptyset
$$

after `TII16`. It contains no global profile-mass
sum, no uniform tightness hypothesis, no whole-space pressure representative,
and no global far-field pressure-tail estimate.

If $\mathrm{Obl}_{\mathrm{TII}}\ne\emptyset$, the status field of
$T_{\mathrm{TII16}}$ is not $\mathrm{proved\ earlier}$ for that branch. The
nonempty elements of $\mathrm{Obl}_{\mathrm{TII}}$ are routed to `PS30`,
`PS34`, or the named local Type II refinement node.

### Local Obstruction Predicate

$P_{\mathrm{PS31}}$ holds when an endpoint theorem is being invoked with a
hypothesis, theorem status, variable convention, pressure gauge, domain, time
interval, or constant dependence not actually verified in the branch.

### Local Lemmas to Prove

**Lemma PS31.1 -- Endpoint record is complete.**
For every endpoint theorem $T_\beta$ selected by the branch, each hypothesis
$H_j\in\mathcal H(T_\beta)$ must be supplied by a record

$$
\mathfrak h_{T_\beta}(H_j)
=
(\mathrm{status}_j,\mathrm{source}_j,\mathrm{variables}_j,\mathrm{gauge}_j),
$$

where

$$
\mathrm{status}_j\in
\{\mathrm{proved},\mathrm{not\ applicable},\mathrm{missing}\}.
$$

Here $\mathrm{source}_j$ names the node or theorem proving the hypothesis,
$\mathrm{variables}_j$ records whether the hypothesis is in physical,
rescaled, centered, or co-moving variables, and $\mathrm{gauge}_j$ records the
pressure and normalization convention. The endpoint record is complete
only if no applicable hypothesis has status $\mathrm{missing}$ and all
variable, gauge, domain, time, quantifier, and constant fields match the entry
$T_\beta$.

**Proof.** Write

$$
\mathcal H(T_\beta)=\{H_1,\ldots,H_m\}.
$$

If an applicable hypothesis has $\mathrm{status}_j=\mathrm{missing}$, the
theorem hypothesis has not been proved from the branch data. If the source is
present but in a different variable frame, pressure gauge, domain, time
interval, or constant regime, the statement proved by the source is not the
statement required by $T_\beta$. Conversely, if every applicable hypothesis is
proved with matching source, variables, gauge, domain, time, quantifiers, and
constants, the endpoint theorem receives exactly its hypotheses.

**Lemma PS31.2 -- Pressure conventions must match endpoint norms.**
Pressure matching has three distinct cases:

1. if the endpoint theorem uses only $\nabla P$, time-dependent gauges do not
   matter;
2. if the endpoint theorem uses local oscillations
   $P-(P)_{B_R}(t)$, spatial-mean gauges match;
3. if the endpoint theorem uses an actual pressure representative, a global
   pressure norm, or a Riesz-transform normalization such as
   $P=\mathcal R_i\mathcal R_j(U_iU_j)$, then the branch must prove that exact
   representative. Local mean-subtracted pressures are not sufficient unless
   the theorem explicitly quotients by that gauge.

**Proof.** For $P'=P+a(t)$,

$$
\nabla P'=\nabla P,
\qquad
P'-(P')_{B_r}=P-P_{B_r}.
$$

Thus gradient data and local oscillation data are invariant under the
corresponding allowed gauges. However,

$$
\|P+a(t)\|_{L^{3/2}(\mathbb R^3\times I)}
$$

and the identity

$$
P=\mathcal R_i\mathcal R_j(U_iU_j)
$$

are not invariant under arbitrary additions $a(t)$ unless the theorem
explicitly modded out that freedom. Therefore a theorem requiring a pressure
representative requires a record for that representative, not merely a
local oscillation record.

**Lemma PS31.3 -- Solution classes are matched by verified implication.**
For an endpoint theorem whose hypotheses require a smooth ancient solution,
mild solution, suitable weak solution, bounded mild ancient solution, or strong
critical convergence, the branch must prove a named implication

$$
\mathrm{recorded\ class}\Longrightarrow\mathrm{endpoint\ class}.
$$

Equality of labels is neither necessary nor sufficient.

**Proof.** Some recorded classes are stronger than endpoint classes, but only
after the auxiliary structures have been checked. For example,

$$
C^\infty_{\rm loc}+L^\infty+\text{mild Duhamel}
\Longrightarrow
\text{bounded mild ancient}.
$$

Likewise a smooth mild ancient solution with pressure satisfying the local
energy inequality gives a suitable weak solution on compact cylinders.
However,

$$
C^\infty_{\rm loc}+L^\infty
\not\Longrightarrow
\text{mild ancient}
$$

unless the Duhamel formula and its time interval are separately proved. The
endpoint class is therefore admissible exactly when the implication and all
auxiliary pressure and local-energy requirements have named sources.

**Lemma PS31.4 -- Missing hypotheses become explicit obligations.**
The value

$$
\mathfrak h_{T_\beta}(H_j)
=
(\mathrm{missing},\mathrm{source}_j,\mathrm{variables}_j,\mathrm{gauge}_j)
$$

blocks endpoint application; the missing item is assigned to the proof step
that can supply it or to `PS33` as a precise open attainability or exclusion
item.

**Proof.** The endpoint-admissibility predicate is

$$
\forall H_j\in\mathcal H(T_\beta),\qquad
\mathrm{status}_j\ne\mathrm{missing}.
$$

The displayed predicate fails as soon as one hypothesis has missing status.
Therefore the theorem is not applied, and the unresolved entry remains a named
regularity, decay, pressure, compactness, or attainability item.

**Lemma PS31.5 -- Theorem status controls branch closure.**
An endpoint theorem with status $\mathrm{established}$ or
$\mathrm{proved\ earlier}$ may be used in an unconditional branch exclusion.
A conditional theorem may be used only in a conclusion explicitly conditional
on that theorem. A conjectural or open theorem cannot close a branch.

**Proof.** A branch is closed only by statements already available to the
proof, or by statements that the proof has declared as assumptions of a
conditional theorem. A conjectural or open statement has neither property.
Using it as an exclusion would replace a missing theorem by a hidden axiom.

**Lemma PS31.6 -- Quantifiers and constants are part of theorem matching.**
Endpoint matching requires the branch to prove the theorem hypotheses with the
same quantifier order and with constants in the admissible range recorded in
$T_\beta$. In particular, a theorem of the form

$$
\forall M\ \exists \varepsilon(M)>0\quad H(M,\varepsilon)\Rightarrow C
$$

cannot be applied with an $\varepsilon$ chosen before $M$, and a theorem whose
conclusion holds on $Q_{\theta r}$ cannot be compared with a lower bound on a
different cylinder unless the constants ledger records the inclusion and the
losses.

**Proof.** Changing quantifier order changes the mathematical statement: an
epsilon depending on a later bound is not available uniformly before that
bound is fixed. Likewise, endpoint conclusions are scale- and cylinder-specific
statements. The contradiction step compares a theorem conclusion with a
branch lower bound only after the constants ledger verifies that the radius,
shrink factor, threshold, covering constant, and normalization are the same or
are related by proved monotonicity. Hence quantifiers and constants are not
cosmetic metadata; they are hypotheses of the endpoint application.

**Lemma PS31.7 -- The `ST20` theorem has local hypotheses only.**
When the selected theorem is $T_{\mathrm{ST20}}$, `PS31` checks only the local
state-space hypotheses displayed above and the emptiness of
$\mathrm{Obl}_{\mathrm{ST}}$. A branch may not add global
$L^3$ boundedness, uniform tightness, global pressure-tail compactness, or a
Riesz-transform pressure representative to the hypothesis list.

**Proof.** `ST20` proves residual emptiness by deriving the endpoint
sequence-$L^3$ bound in `ST17` from terminal indecomposability and by using
local pressure gauges throughout `ST2`--`ST6`. Adding a global tail or pressure
representative to $T_{\mathrm{ST20}}$ would change the theorem from local
state-space closure into the global estimate machine that the residual block
was built to avoid. The emptiness of $\mathrm{Obl}_{\mathrm{ST}}$ is not a
global estimate; it is the statement that the local ST subtheorems needed by
`ST20` have actually closed. Therefore global statements can appear only as
outputs or as hypotheses of other endpoint theorems, never as hypotheses of
the residual closure theorem.

### Specific Estimate

The decisive verification is

$$
\mathrm{status}(T_\beta)\in
\{\mathrm{established},\mathrm{proved\ earlier}\}
\quad\text{or}\quad
\text{the proof is explicitly conditional on }T_\beta,
$$

and, for every applicable $H_j\in\mathcal H(T_\beta)$,

$$
\mathfrak h_{T_\beta}(H_j)
=
(\mathrm{proved},\mathrm{source}_j,\mathrm{variables}_j,\mathrm{gauge}_j)
$$

with matching variables, gauge, domain, boundary record, source topology,
time interval, quantifiers, and constants.

For the residual branch, the decisive verification is instead the local
hypothesis map

$$
\mathfrak h_{T_{\mathrm{ST20}}}
=
\left\{
\begin{array}{l}
(U,\Pi)\in\mathfrak X_M,\\
\|U\|_{L^\infty}\le M,\\
\text{local suitability},\\
\text{local pressure gauges},\\
\text{retained compact activity},\\
\text{outside the routed lower-strata ledger},\\
\mathrm{Obl}_{\mathrm{ST}}=\emptyset
\end{array}
\right\}
$$

with every entry proved by `ST0`--`ST20` and no global tail hypothesis.

### Practical Verification Steps

1. Select the endpoint theorem entry $T_\beta\in\mathscr T$ for the branch.
2. List all theorem hypotheses verbatim in mathematical form.
3. Attach each hypothesis to a record
   $(\mathrm{status},\mathrm{source},\mathrm{variables},\mathrm{gauge})$.
4. Verify theorem status, pressure representative, boundary record, source
   topology, topology implication, domain, time interval, quantifiers, and
   constants.
5. If the selected theorem is `ST20`, verify only the local state-space
   hypothesis list, check that $\mathrm{Obl}_{\mathrm{ST}}=\emptyset$, and
   check that no global $L^3$, tightness, or pressure-tail hypothesis has
   been inserted.
6. Record missing, conjectural, or open items as explicit obligations; do not
   apply the theorem to close the branch.

## Estimate Step $B_{\mathrm{PS31}}$

The estimate step is theorem-hypothesis matching, not a new Navier--Stokes
estimate.

## Failure Case

Failure name: endpoint hypothesis mismatch.

Analytic meaning: the branch is being passed to a theorem whose hypotheses have
not been proved from the local estimates.

## Refinement Step

Allowed refinements:

1. prove the missing estimate at its source node;
2. select a theorem with matching hypotheses;
3. strengthen pressure or topology convergence;
4. assign unresolved theorem applicability to `PS33`.

Progress measure: the hypothesis map changes from missing to proved or to a
named obligation.

## Data Passed Forward

The next proof step is `PS32`. The data passed forward are

$$
\Gamma_{\mathrm{PS31}}
=
\Gamma_{\mathrm{PS30}}
\cup
\Gamma_{\mathrm{ST20}}^{\rm res}
\cup
\left\{
\begin{array}{l}
T_\beta\in\mathscr T,\quad
\mathrm{status}(T_\beta)\in
\{\mathrm{established},\mathrm{proved\ earlier}\}
\text{ or declared conditional},\\
\mathrm{Obl}_{\mathrm{ST}}=\emptyset
\text{ when }T_\beta=T_{\mathrm{ST20}},\\
\mathfrak h_{T_\beta}\text{ complete with sources},\\
\text{variable, gauge, domain, boundary, source topology, time, quantifier,}\\
\text{and constant matching verified}
\end{array}
\right\}.
$$

Here $\Gamma_{\mathrm{ST20}}^{\rm res}=\Gamma_{\mathrm{ST20}}$ on the
residual-closure branch and is empty on nonresidual endpoint branches.

If any item is missing, mismatched, conjectural, or open, `PS31` routes the
branch to `PS33` or `PS34`; it does not pass an endpoint exclusion to `PS32`.

---

# 59. `PS32` -- Endpoint Exclusion Theorem Application

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is the endpoint branch whose hypotheses exactly match a recorded
Navier--Stokes theorem.

### Standing Assumptions

The incoming record states that the theorem is applicable exactly as checked
in `PS31`, including theorem status, variable frame, pressure gauge, domain,
time interval, and constants.

### Objects Inspected

Inspect the theorem conclusion, local regularity implication, zero-profile
implication, branch-emptiness implication, retained activity witness, and the
object to which the theorem conclusion applies.

### Dependencies Used

The theorem match comes from `PS31`; positive activity from `PS8`; local
singularity from `C_mu`; pressure and defect closure from `PS30`.

### Local Obstruction Predicate

$P_{\mathrm{PS32}}$ holds when the endpoint conclusion is insufficient to
contradict the retained branch after object, frame, gauge, and prelimit
transfer have been checked.

### Local Lemmas to Prove

Endpoint contradictions have three admissible types.

**Type A -- direct physical regularity.** The endpoint theorem applies
directly to the original pair $(u,p)$ on a physical cylinder $Q_r(z_*)$ and
gives local boundedness or Holder regularity in $Q_{r/2}(z_*)$.

**Type B -- quantitative prelimit contradiction.** The endpoint theorem
applies to a limit profile and gives a quantitative conclusion stable under
the convergence used to obtain the profile, for example

$$
C(V;Q)+D(P;Q)<\frac12\varepsilon_{\rm CKN}.
$$

Strong velocity convergence and pressure convergence then imply, for large
$n$,

$$
C(V_n;Q)+D(P_n;Q)<\varepsilon_{\rm CKN},
$$

contradicting the retained lower bound on the same cylinder.

**Type C -- zero or rigidity contradiction.** The endpoint theorem gives
$V\equiv0$ for the same endpoint object that carries a compact activity
witness

$$
\iint_K |V|^3\ge\eta_*>0.
$$

**Lemma PS32.1 -- Regularity conclusions contradict singular entry only after
transfer to the original branch.**
If an endpoint theorem gives local boundedness of the original solution near
$z_*=(x_*,T)$, then $x_*\notin\Sigma(T)$, contradicting the singular-entry
branch. If the theorem gives local boundedness only for a rescaled limit
profile, no contradiction follows unless the branch also proves a quantitative
prelimit transfer yielding CKN smallness or boundedness for the original
solution on a physical cylinder ending at $z_*$.

**Proof.** The singular set is defined by failure of local boundedness of the
original solution on physical cylinders. Thus Type A contradicts singular
entry directly. A limit profile is only a limit of rescaled objects; its
smoothness by itself does not imply regularity of the original solution.
Type B supplies the missing implication by transferring a quantitative
smallness or boundedness statement back to sufficiently large prelimit
cylinders, where the CKN regularity criterion contradicts the retained
singular concentration.

**Lemma PS32.2 -- Zero-profile conclusions contradict retained activity.**
If the endpoint theorem gives $V\equiv0$ for the same endpoint object, same
frame, same time interval, and same pressure gauge that carries

$$
\iint_K |V|^3\ge\eta_*>0,
$$

then the branch is contradictory.

**Proof.** The zero velocity has zero $L^3$ activity on every compact set, so
it cannot satisfy the displayed positive activity bound. If the activity bound
belongs only to a different sequence, a different hull element, or a different
frame, the contradiction is unavailable until active attainment is transferred
to the endpoint object. A combined velocity-pressure lower bound may be used
only when the endpoint conclusion also forces the pressure oscillation to
vanish in the same gauge.

**Lemma PS32.3 -- Empty-branch conclusions remove the branch.**
If the endpoint theorem states that no object satisfying the verified
hypotheses exists, then the selected branch is impossible.

**Proof.** The branch hypothesis vector proves exactly those hypotheses; the
theorem denies existence of such an object.

**Lemma PS32.4 -- Endpoint conclusion must match the retained branch.**
`PS32` may conclude contradiction only if the endpoint theorem's conclusion is
about the same object, same frame, same time interval, and same pressure gauge
as the retained activity or singular-entry condition.

**Proof.** Endpoint theorems are statements about a specified object in a
specified coordinate frame and gauge. Applying a theorem to a stationary hull
element, for example, does not exclude a different active hull element unless
the branch proves they coincide or transfers the activity and conclusion
between them. Without that identification, the theorem has excluded a
different object.

**Lemma PS32.5 -- Quantitative prelimit transfer is a separate record.**
For a Type B contradiction, the branch must record a transfer record

$$
\mathcal Q_{\rm trans}
=
\left(
Q,\ Q_n,\ \Phi_n,\ \mathrm{topology},\ \delta,\ \varepsilon_{\rm CKN},
\mathrm{source}
\right),
$$

where $\Phi_n$ maps the endpoint cylinder $Q$ to the physical or rescaled
prelimit cylinder $Q_n$, the convergence topology implies convergence of the
CKN quantities on $Q$, and

$$
C(V;Q)+D(P;Q)\le\varepsilon_{\rm CKN}-2\delta
\quad\Longrightarrow\quad
C(V_n;Q_n)+D(P_n;Q_n)<\varepsilon_{\rm CKN}
$$

for all sufficiently large $n$.

**Proof.** Type B uses a limiting theorem to contradict a prelimit lower
bound. This requires more than qualitative convergence. The map $\Phi_n$
identifies the exact cylinders, the topology gives convergence of both the
velocity and pressure terms in the scale-invariant quantities, and the margin
$2\delta$ absorbs convergence errors and all constant losses. Without such a
margin and cylinder identification, smallness of the limit may coexist with
large prelimit concentration on the cylinder where singularity was retained.

**Lemma PS32.6 -- Non-exclusion becomes a realization question.**
If the theorem applies but permits a nonzero branch, or if the conclusion
applies to a different object without active-attainment transfer, the remaining
issue is whether such a branch is realized by a Navier--Stokes blow-up
sequence; this is the task of `PS33`.

**Proof.** The branch record contains two completed fields:

$$
\mathfrak h_{T_\beta}(H_j)\ne\mathrm{missing}\quad(1\le j\le m),
\qquad
T_\beta(\text{branch})=\text{nonzero admissible or unmatched object}.
$$

Thus theorem matching and theorem application have both been evaluated. The
remaining unevaluated predicate is the existence of a suitable NS3D blow-up
sequence converging to the active object, or a transfer from the theorem object
to the active object, which is the realization predicate checked in `PS33`.

**Lemma PS32.7 -- Exclusion labels require proof records, not statuses alone.**
Whenever `PS32` outputs $\mathrm{excluded}$, it also outputs an exclusion
record

$$
\mathcal R_{\rm excl}^{\rm PS32}
=
\left(
\mathrm{type},\ T_\beta,\ \mathrm{theorem\ status},\
\mathrm{object},\ \mathrm{frame},\ \mathrm{gauge},\
\mathrm{time\ interval},\ \mathrm{activity\ or\ singularity\ target},\
\mathcal Q_{\rm trans}
\right),
$$

where $\mathrm{type}\in\{\mathrm{A},\mathrm{B},\mathrm{C}\}$ and
$\mathcal Q_{\rm trans}$ is required exactly for Type B. The theorem status
is either established, proved earlier, or explicitly conditional as recorded
in `PS31`.

**Proof.** The word $\mathrm{excluded}$ is ambiguous unless the contradiction
mechanism and target object are recorded. Type A needs the physical
singularity target, Type B needs the quantitative transfer record, and Type C
needs active attainment for the same endpoint object. The exclusion record stores
these data in one row, so later nodes can audit that the branch was closed by
a valid contradiction rather than by a theorem label.

### Specific Estimate

The decisive comparison is

$$
\text{endpoint conclusion}
\quad\Longrightarrow\quad
\neg\{\text{retained singular activity or singular entry}\},
$$

by Type A, Type B, or Type C above.

### Practical Verification Steps

1. Apply the endpoint theorem.
2. Classify the endpoint conclusion as Type A, Type B, Type C, or
   non-excluding.
3. Verify same object, frame, gauge, time interval, and pressure convention.
4. For Type B, prove the quantitative prelimit transfer to the original branch.
5. If a contradiction follows, build $\mathcal C_{\rm excl}^{\rm PS32}$.
6. If no contradiction follows, record the precise non-excluded branch or the
   missing active-attainment transfer.

## Estimate Step $B_{\mathrm{PS32}}$

The estimate step is the contradiction extraction from the endpoint theorem.

## Failure Case

Failure name: endpoint non-exclusion.

Analytic meaning: all hypotheses match a theorem, but the theorem's conclusion
does not eliminate the retained local branch.

## Refinement Step

Allowed refinements:

1. choose a stronger endpoint theorem whose hypotheses are recorded;
2. add a missing endpoint hypothesis through `PS31`;
3. prove quantitative prelimit transfer;
4. prove active-attainment transfer to the endpoint object;
5. assign the non-excluded branch to realization analysis in `PS33`.

Progress measure: the endpoint conclusion either contradicts retained activity
or becomes a precise realizable-branch question.

## Data Passed Forward

The next proof step is `PS33`. The data passed forward are

$$
\Gamma_{\mathrm{PS32}}
=
\Gamma_{\mathrm{PS31}}
\cup
\left\{
\begin{array}{l}
\text{endpoint contradiction achieved by Type A, Type B, or Type C with }
\mathcal C_{\rm excl}^{\rm PS32},\\
\text{or endpoint theorem applies but does not exclude the active branch},\\
\text{or endpoint conclusion applies to a different object and requires transfer}
\end{array}
\right\}.
$$

The status $\mathrm{excluded}$ is passed forward only in the first case.

---

# 60. `PS33` -- Realization or Admissible Counterexample Check

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is not a formal profile alone, but a profile together with an
admissible sequence of suitable weak solutions producing it.

### Standing Assumptions

The incoming record states that the branch has not been excluded by the
selected endpoint theorem, or that a claimed exclusion requires a missing
active-attainment or prelimit-transfer record.

### Objects Inspected

Inspect the construction sequence, local energy inequality, pressure gauges,
convergence topology, activity lower bound, and compatibility with the defect
vector.

### Dependencies Used

Every preceding estimate contributes either a universal necessary condition
for realization or a branch-specific condition for membership in a named
class. These two roles are kept separate.

### Local Obstruction Predicate

$P_{\mathrm{PS33}}$ holds when a formal branch is treated as relevant without
an admissible provenance sequence, or when a non-excluded branch lacks either
a proof that it is generated by the original blow-up sequence or a proof that
it cannot be so generated.

### Provenance Graph

Every branch reaching `PS33` carries a provenance graph

$$
\mathcal P_{\rm prov}.
$$

The nodes of this graph are transformations used to produce the branch:

$$
\text{subsequence extraction},\quad
\text{rescaling},\quad
\text{recentering},\quad
\text{time translation},\quad
\text{pressure gauge},\quad
\text{cutoff},\quad
\text{limit passage},\quad
\text{endpoint theorem}.
$$

Each edge is marked either as an actual operation on the original suitable
sequence or as a formal candidate only. If all edges are actual operations on
the original sequence, the branch is realized by provenance. If some edge is
formal only, realization is unresolved unless an attaining sequence is
constructed. If a necessary condition is violated, the branch is proved
nonattainable.

### Local Lemmas to Prove

**Lemma PS33.1 -- Realization requires all prior verified conditions.**
If a branch is realized by an admissible blow-up sequence, then the sequence
must satisfy the entry, concentration, compactness, pressure, defect, and
endpoint-hypothesis conclusions recorded in `C_mu` and `PS1`--`PS32`.

**Proof.** Each conclusion is a necessary condition imposed by the construction
of the branch. Dropping any one changes the local PDE problem or the endpoint
class.

**Lemma PS33.2 -- Universal and branch-specific necessary conditions have
different consequences.**
Let

$$
\mathcal N_{\rm univ}
=
\{\text{conditions necessary for every admissible realization}\}
$$

and

$$
\mathcal N_{\rm branch}
=
\{\text{conditions necessary only for this named branch}\}.
$$

A violation of $\mathcal N_{\rm univ}$ gives proved nonattainability. A
violation of $\mathcal N_{\rm branch}$ means the object is not in that named
branch and must be rerouted. A violation of a chosen gauge gives proved
nonattainability only if no allowed symmetry or pressure normalization repairs
it.

**Proof.** Universal necessary conditions are imposed by the original suitable
weak sequence and its admissible blow-up operations. Violating one contradicts
the existence of any admissible realization. Branch-specific conditions define
membership in a particular class; failing one removes that class label but may
leave another admissible branch. Gauge choices are representatives of an
equivalence class until a theorem requires an exact representative.

**Lemma PS33.3 -- Provenance decides realized, formal, and undecided status.**
If all edges of $\mathcal P_{\rm prov}$ are actual operations on the original
sequence, then the branch has status $\mathrm{realized}$. If a necessary
universal condition is violated, the branch has status
$\mathrm{proved\ nonattainable}$. If the branch is a candidate object with
only formal edges and no attaining sequence, it has status
$\mathrm{formal\ only}$. If the record proves neither realization nor
nonattainability, it has status $\mathrm{undecided}$.

**Proof.** Actual provenance is precisely an admissible construction from the
original sequence. Violation of a universal necessary condition forbids every
such construction. A purely formal edge records a candidate in the limiting
PDE class but not a limit of the original NS3D branch. If none of these tests
settles the issue, the realization predicate is undecided.

**Lemma PS33.4 -- Formal profiles without sequences are obligations, not
counterexamples.**
A profile solving a limiting equation but lacking a suitable weak
approximating sequence is not an admissible NS3D obstruction; it is a missing
attainability theorem.

**Proof.** The blow-up analysis studies limits of actual NS3D solutions. A
formal limiting solution not obtained from such a sequence does not contradict
regularity of the original equation.

**Lemma PS33.5 -- Realized non-excluded profiles are surviving branches.**
If a non-excluded profile is actually produced by the original blow-up
sequence, then the proof cannot close unless a later node excludes it or proves
it incompatible with the original singular-entry assumptions.

**Proof.** A realized non-excluded profile is not a formal artifact; it is an
admissible limit of the branch generated from the alleged singular point.
Since neither endpoint exclusion nor proved nonattainability has removed it,
it remains a genuine surviving local branch.

### Specific Estimate

The decisive verification is the existence or nonexistence of an admissible
sequence satisfying

$$
u_n^{(z_n,r_n)}\to V
$$

in the branch topology with all previous local estimates, together with a
provenance graph whose edges are actual operations on the original sequence.

### Practical Verification Steps

1. Build the provenance graph $\mathcal P_{\rm prov}$.
2. Mark every edge as actual on the original sequence or formal only.
3. Separate $\mathcal N_{\rm univ}$ from $\mathcal N_{\rm branch}$.
4. Check local energy, pressure conventions, convergence topology, and
   retained activity against the graph.
5. Assign exactly one status:
   $\mathrm{realized}$, $\mathrm{proved\ nonattainable}$,
   $\mathrm{formal\ only}$, or $\mathrm{undecided}$.

## Estimate Step $B_{\mathrm{PS33}}$

The estimate step is verification of the realization package or a proved
nonattainability contradiction.

## Failure Case

Failure name: undecided branch realization.

Analytic meaning: a formal branch survives endpoint exclusion, and the record
lacks either an attaining NS3D sequence or a proved nonattainability proof.

## Refinement Step

Allowed refinements:

1. prove nonattainability from prior verified conditions;
2. construct an admissible sequence;
3. repair a gauge or symmetry normalization;
4. reroute an object that fails only a branch-specific condition;
5. assign undecided attainability to `PS34` as residual.

Progress measure: the branch becomes proved nonattainable, realized, or part
of the exact residual complement.

## Data Passed Forward

The next proof step is `PS34`. The data passed forward are

$$
\Gamma_{\mathrm{PS33}}
=
\Gamma_{\mathrm{PS32}}
\cup
\left\{
\begin{array}{l}
\mathcal P_{\rm prov},\\
\mathrm{realization\ status}\in
\{\mathrm{realized},\mathrm{proved\ nonattainable},
\mathrm{formal\ only},\mathrm{undecided}\},\\
\text{source of proved nonattainability if applicable}
\end{array}
\right\}.
$$

Only $\mathrm{proved\ nonattainable}$ closes a candidate at `PS33`.
A $\mathrm{realized}$ non-excluded branch is passed forward as a live
obstruction. A $\mathrm{formal\ only}$ or $\mathrm{undecided}$ branch is not an
admissible counterexample, but it remains an attainability obligation until
`PS33` proves nonattainability or constructs admissible provenance.

---

# 61. `PS34` -- Residual Complement Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is an admissible local profile or local branch after every named
PDE alternative has been checked.

### Standing Assumptions

The incoming record states that each named branch predicate has a yes/no/inc
conclusion, every realization issue has been recorded, and the generic local
residual branch has been routed through `ST0`--`ST20` when its local
hypotheses are present.

### Objects Inspected

Inspect the list of named predicates, their overlaps, residual membership, and
all unresolved endpoint, attainability, and nonattainability-source
obligations.

### Dependencies Used

All previous branch classifications feed `PS34`; `PS33` supplies the last
non-exclusion status; `ST20` supplies the closure status for the generic local
residual class.

### Local Obstruction Predicate

$P_{\mathrm{PS34}}$ holds when there is no exact set-theoretic residual
identity for the remaining branches, or when classification completeness is
being treated as exclusion completeness, or when an unsourced
nonattainability label is counted as closed, or when a branch satisfying the
local hypotheses of `ST20` is left as an open generic residual.

### Candidate and Admissible Branch Universes

First define the candidate universe

$$
\mathcal S_{\rm cand}
=
\left\{
\begin{array}{l}
\text{all formal or realized local objects named by the branch predicates,}\\
\text{endpoint candidates, residual witnesses, or limiting equations}
\end{array}
\right\}.
$$

The admissible local universe is the realized-provenance subcollection

$$
\mathcal S_{\rm loc}
=
\left\{
\begin{array}{l}
V\in\mathcal S_{\rm cand}: V\text{ is actually obtainable from the}\\
\text{singular-entry branch by the allowed operations}
\end{array}
\right\}.
$$

Membership in $\mathcal S_{\rm loc}$ records equation class, domain, time
interval, pressure convention, defect-vector status, activity witness, and
realization/provenance status. A candidate with only claimed, formal, or
refuted provenance is not in $\mathcal S_{\rm loc}$ unless `PS33` supplies an
actual attaining sequence. This keeps the realized universe separate from the
bookkeeping universe of formal candidates.

The proved nonattainability ledger is

$$
\mathcal N_{\rm att}^{0}
=
\left\{
V\in\mathcal S_{\rm cand}:
\mathrm{status}(V)=\mathrm{proved\ nonattainable}
\right\}.
$$

Its unsourced subledger is

$$
\mathcal O_{\rm src}
=
\left\{
V\in\mathcal N_{\rm att}^{0}:
\text{the `PS33' source or failed necessary condition is missing}
\right\}.
$$

An element of $\mathcal N_{\rm att}^{0}$ is a closed candidate only when it is
not in $\mathcal O_{\rm src}$. These are candidates, not realized local
objects. If a profile is recorded both in $\mathcal S_{\rm loc}$ and in
$\mathcal N_{\rm att}^{0}$, the ledger is inconsistent: the proof must return
to `PS33` and decide whether the provenance is actual or impossible.

The unsettled attainability-obligation set is

$$
\mathcal O_{\rm att}
=
\left\{
V\in\mathcal S_{\rm cand}:
\mathrm{status}(V)\in
\{\mathrm{formal\ only},\mathrm{undecided}\}
\right\}.
$$

The set $\mathcal O_{\rm att}$ is not an admissible counterexample set, but it
is a proof-obligation set. The proof cannot close while
$\mathcal O_{\rm att}\ne\emptyset$, because one of its elements may still be
made admissible by a later attaining-sequence construction. The proof also
cannot close while $\mathcal O_{\rm src}\ne\emptyset$, because an unsupported
nonattainability label has not yet been proved.

Define the union of named classes by

$$
\mathcal U_{\rm named}
=
\bigcup_{\alpha\in A}\mathcal U_\alpha.
$$

Let $A_{\rm excl}\subset A$ be the set of indices whose realized admissible
part is excluded by `PS32`:

$$
\alpha\in A_{\rm excl}
\quad\Longleftrightarrow\quad
\mathcal U_\alpha\cap\mathcal S_{\rm loc}
\text{ has status }\mathrm{excluded}.
$$

Named candidates proved nonattainable by `PS33` are recorded in
$\mathcal N_{\rm att}^{0}$ instead. They close a formal or claimed branch by
showing that it cannot enter $\mathcal S_{\rm loc}$ only when they are not in
$\mathcal O_{\rm src}$.
Define the preliminary named excluded union

$$
\mathcal U_{\rm excl}^{0}
=
\left(\bigcup_{\alpha\in A_{\rm excl}}\mathcal U_\alpha\right)
\cap\mathcal S_{\rm loc}.
$$

There are two different complements before residual closure:

$$
\mathcal R_{\rm named}
=
\mathcal S_{\rm loc}\setminus\mathcal U_{\rm named},
\qquad
\mathcal R_{\rm open}^{0}
=
\mathcal S_{\rm loc}\setminus\mathcal U_{\rm excl}^{0}.
$$

The first is the unnamed residual among admissible local objects. The second
is the not-yet-excluded admissible local branch set before applying the local
residual closure theorem.

Let

$$
\mathcal R_{\rm ST20}
=
\left\{
V\in\mathcal R_{\rm named}:
V\text{ satisfies the local hypotheses of }T_{\mathrm{ST20}}
\text{ and }\mathrm{Obl}_{\mathrm{ST}}(V)=\emptyset
\right\}.
$$

The residual ST-obligation set is

$$
\mathcal O_{\rm ST}
=
\left\{
V\in\mathcal R_{\rm named}:
V\text{ satisfies the local residual hypotheses but }
\mathrm{Obl}_{\mathrm{ST}}(V)\ne\emptyset
\right\}.
$$

After `ST20` and the `PS31` theorem match for $T_{\mathrm{ST20}}$, this
subclass has status $\mathrm{excluded}$. The excluded realized universe is

$$
\mathcal U_{\rm excl}
=
\mathcal U_{\rm excl}^{0}\cup\mathcal R_{\rm ST20}.
$$

The final open realized complement is

$$
\mathcal R_{\rm open}
=
\mathcal S_{\rm loc}\setminus\mathcal U_{\rm excl}.
$$

The proof can close only if

$$
\mathcal R_{\rm open}=\emptyset
\qquad\text{and}\qquad
\mathcal O_{\rm ST}=\emptyset
\qquad\text{and}\qquad
\mathcal O_{\rm att}=\emptyset
\qquad\text{and}\qquad
\mathcal O_{\rm src}=\emptyset.
$$

### Local Lemmas to Prove

**Lemma PS34.1 -- Named complement gives classification, not exclusion.**
Every admissible local profile belongs either to $\mathcal U_{\rm named}$ or
to $\mathcal R_{\rm named}$.

**Proof.** The residual class is declared by the displayed formula

$$
\mathcal R_{\rm named}
=\mathcal S_{\rm loc}\setminus\mathcal U_{\rm named}.
$$

For $V\in\mathcal S_{\rm loc}$, membership in $\mathcal U_{\rm named}$ gives a
named branch alternative, and nonmembership gives
$V\in\mathcal R_{\rm named}$ by the displayed set identity. This proves
coverage by named-or-residual classes. It does not prove that any class is
empty or excluded until the class is included in $\mathcal U_{\rm excl}$; for
the generic residual this happens through $\mathcal R_{\rm ST20}$.

**Lemma PS34.2 -- Exclusion complement records live branches.**
The open complement

$$
\mathcal R_{\rm open}
=
\mathcal S_{\rm loc}\setminus\mathcal U_{\rm excl}
$$

is empty if and only if every actually obtainable local branch is excluded by
a `PS32` contradiction or by the locally matched `ST20` residual closure.
Proved nonattainability closes candidates before they enter
$\mathcal S_{\rm loc}$. The full local audit is closed only when, in
addition, every nonattainability claim has its `PS33` source and
$\mathcal O_{\rm att}=\emptyset$, equivalently
$\mathcal O_{\rm ST}=\emptyset$, $\mathcal O_{\rm att}=\emptyset$, and
$\mathcal O_{\rm src}=\emptyset$.

**Proof.** By definition, $\mathcal U_{\rm excl}$ contains only realized
objects in $\mathcal S_{\rm loc}$ closed by an endpoint contradiction or by
the residual closure theorem `ST20`. Thus
$V\notin\mathcal U_{\rm excl}$ exactly means that the actually obtainable
local object has not been excluded. Therefore
$\mathcal R_{\rm open}=\emptyset$ exactly when no realized nonclosed branch
remains. Candidate branches in $\mathcal O_{\rm att}$ are not closed either:
they lack both an attaining sequence and a proved nonattainability record.
They remain open obligations because a later attaining-sequence construction
could move them into $\mathcal S_{\rm loc}$ as realized branches. Candidate
branches in $\mathcal N_{\rm att}^{0}$ are closed only because `PS33` supplies
the separate necessary-condition contradiction; they are not counted as
realized exclusions. If a candidate lies in $\mathcal O_{\rm src}$, that
contradiction has not been supplied and the candidate remains open.
Branches in $\mathcal O_{\rm ST}$ are realized residual branches whose local
hypotheses have been identified but whose ST subtheorems have not all closed;
they are therefore open local PDE obligations until
$\mathrm{Obl}_{\mathrm{ST}}$ is emptied.

**Lemma PS34.3 -- Overlaps among named branches create no residual class.**
If a profile belongs to more than one named branch, it is still in
$\mathcal U_{\rm named}$ and is not an unnamed residual.

**Proof.** Membership in a union requires membership in at least one branch.
Multiple memberships do not create a new complement class.

**Lemma PS34.4 -- Ordered subtraction gives disjoint reporting.**
If a disjoint record is desired, define

$$
\widetilde{\mathcal U}_1=\mathcal U_1,\qquad
\widetilde{\mathcal U}_j
=
\mathcal U_j\setminus\bigcup_{i<j}\mathcal U_i.
$$

Then the disjoint classes have the same union as the original named classes.

**Proof.** The construction removes only elements already assigned to earlier
classes, so the union is unchanged and the resulting classes are disjoint.

**Lemma PS34.5 -- Unresolved named classes remain open.**
Membership in a named class is not an exclusion unless that class status is
$\mathrm{excluded}$ for the realized part, or unless `PS33` separately proves
that the named candidate is nonattainable.

**Proof.** A status such as $\mathrm{inc}$, $\mathrm{undecided}$,
$\mathrm{formal\ only}$, $\mathrm{realized\ nonexcluded}$, or
$\mathrm{unresolved}$ records that the branch has been classified, not that it
has been eliminated. Such a branch lies in $\mathcal U_{\rm named}$ but remains in
$\mathcal R_{\rm open}$ unless it also lies in $\mathcal U_{\rm excl}$.
If the branch lacks admissible provenance, it lies instead in
$\mathcal O_{\rm att}$ until `PS33` decides realization or nonattainability.
A named candidate with a proved nonattainability record lies in
$\mathcal N_{\rm att}^{0}$; this closes the candidate but does not convert it
into a realized excluded branch. If its source is missing, it lies in
$\mathcal O_{\rm src}$ and is not closed.

**Lemma PS34.6 -- Residual obligations require witnesses.**
Every residual item must be recorded as a concrete mathematical proposition
with a witness.

**Proof.** A residual label without a witness is only a name for ignorance and
does not define a subset of $\mathcal S_{\rm loc}$. Examples of admissible
residual witnesses are

$$
\mathcal R_{\rm tail}:\quad
\exists \eta>0,\ R_k\to\infty,\ \tau_k
\text{ such that }
\int_{|y|>R_k}|V(y,\tau_k)|^3\ge\eta,
$$

or

$$
\mathcal R_{\rm pressure}:\quad
\text{the harmonic pressure remainder lacks local }L^{3/2}\text{ compactness}.
$$

Each displayed proposition specifies the property by which an object belongs
to the residual set. Under the corrected architecture, these witnesses are
not requests for global estimates: tail witnesses are routed through
`ST1`--`ST20`, and pressure witnesses are local gauge or compact-window
defects routed through `ST4` and `ST6`.

**Lemma PS34.7 -- Generic local residual is closed by `ST20`.**
If $V\in\mathcal R_{\rm named}$ satisfies the local residual hypotheses

$$
V\in\mathfrak X_M,\quad
\|V\|_{L^\infty}\le M,\quad
\text{local suitability},\quad
\text{local pressure gauges},\quad
\text{retained compact activity},\quad
V\notin\bigcup_j\mathcal L_j^{\rm routed},\quad
\mathrm{Obl}_{\mathrm{ST}}(V)=\emptyset,
$$

then $V\in\mathcal R_{\rm ST20}\subset\mathcal U_{\rm excl}$.

**Proof.** These are exactly the hypotheses of the theorem entry
$T_{\mathrm{ST20}}$ registered in `PS31`. `ST20` proves the generic residual
class empty by local state-space stratification and endpoint sequence
recovery once the local ST obligation ledger is empty. Therefore a realized
residual branch with these hypotheses is excluded, not open. If
$\mathrm{Obl}_{\mathrm{ST}}(V)\ne\emptyset$, the branch is not placed in
$\mathcal R_{\rm ST20}$; the nonempty obligation is recorded in the open
residual ledger. No global $L^3$, tightness, or whole-space pressure
hypothesis is added in this classification step.

### Specific Estimate

The decisive bookkeeping identities are

$$
\mathcal S_{\rm loc}
=
\mathcal U_{\rm named}\cup\mathcal R_{\rm named},
\qquad
\mathcal U_{\rm excl}
=
\mathcal U_{\rm excl}^{0}\cup\mathcal R_{\rm ST20},
\qquad
\mathcal R_{\rm open}
=
\mathcal S_{\rm loc}\setminus\mathcal U_{\rm excl}.
$$

The decisive closure condition is

$$
\mathcal R_{\rm open}=\emptyset,\qquad
\mathcal O_{\rm ST}=\emptyset,\qquad
\mathcal O_{\rm att}=\emptyset,\qquad
\mathcal O_{\rm src}=\emptyset.
$$

### Practical Verification Steps

1. Define $\mathcal S_{\rm cand}$ explicitly.
2. Define $\mathcal S_{\rm loc}$ by admissible provenance and compute
   $\mathcal O_{\rm att}$.
3. List every named branch predicate and status.
4. Record $\mathcal N_{\rm att}^{0}$ for candidates proved nonattainable by
   `PS33`.
5. Compute $\mathcal O_{\rm src}$ for unsourced nonattainability labels.
6. Define $\mathcal U_{\rm named}$ and $\mathcal U_{\rm excl}^{0}$
   separately.
7. Define $\mathcal R_{\rm named}$ and $\mathcal R_{\rm open}$ by set
   subtraction.
8. Test the generic residual component against the local hypotheses of
   `ST20`; include the matched component in $\mathcal R_{\rm ST20}$ with
   status $\mathrm{excluded}$ only when $\mathrm{Obl}_{\mathrm{ST}}=\emptyset$.
9. Put local residual components with nonempty $\mathrm{Obl}_{\mathrm{ST}}$
   into $\mathcal O_{\rm ST}$.
10. Record all residual, attainability, and source obligations as explicit PDE
   statements with witnesses.

## Estimate Step $B_{\mathrm{PS34}}$

The estimate step is set-theoretic and predicate verification, not a new PDE
estimate.

## Failure Case

Failure name: ill-defined residual complement or source ledger.

Analytic meaning: a branch remains outside named classes without a precise
predicate or obligation, or a nonattainability label lacks the `PS33` source
needed to close it.

## Refinement Step

Allowed refinements:

1. add the missing predicate;
2. define ordered subtraction;
3. move an unresolved theorem to `PS31` or `PS33`;
4. split a named branch into excluded and open subclasses;
5. attach a mathematical witness to each residual or attainability obligation;
6. attach the missing `PS33` source to a nonattainability claim or move it to
   the open source ledger.

Progress measure: the residual class becomes exact, the `ST20`-eligible
generic residual component is excluded, every nonempty ST obligation is
visible, and every nonattainability label is either sourced or explicitly
open.

## Data Passed Forward

The next proof step is `PS35`. The data passed forward are

$$
\Gamma_{\mathrm{PS34}}
=
\Gamma_{\mathrm{PS33}}
\cup
\left\{
\begin{array}{l}
\mathcal S_{\rm cand}\text{ precisely defined},\\
\mathcal S_{\rm loc}\text{ precisely defined},\\
\mathcal O_{\rm att}\text{ computed},\\
\mathcal N_{\rm att}^{0}\text{ computed},\\
\mathcal O_{\rm src}\text{ computed},\\
\mathcal O_{\rm ST}\text{ computed},\\
\mathcal U_{\rm named},\quad
\mathcal U_{\rm excl}^{0},\quad
\mathcal R_{\rm ST20},\quad
\mathcal U_{\rm excl},\\
\mathcal R_{\rm named}
=\mathcal S_{\rm loc}\setminus\mathcal U_{\rm named},\\
\mathcal R_{\rm open}
=\mathcal S_{\rm loc}\setminus\mathcal U_{\rm excl},\\
\text{explicit witness for every residual, attainability, or source item}
\end{array}
\right\}.
$$

---

# 62. `PS35` -- Case-Decomposition Completeness Check

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is an arbitrary admissible local profile generated by the NS3D
blow-up procedure.

### Standing Assumptions

The incoming record states that the candidate universe, admissible universe,
named residual complement, realized-open complement, attainability-obligation
set, local ST-obligation set, nonattainability ledger, and
unsourced-nonattainability set have all been defined exactly in `PS34`.

### Objects Inspected

Inspect the complete branch list: Type I, Type II, compact, stationary, tight,
radiative, rough, multicore, finite packet, terminal, perturbative, structured,
scale-transition, rigidity, defect, endpoint, realization, and residual
classes, together with the `PS33` source for every claimed
nonattainability label.

### Dependencies Used

The nodes `C_mu`, `PS1`--`PS34`, and `ST0`--`ST20` contribute one or more
branch predicates or exclusions, and `PS34` distinguishes named coverage from
actual closure.

### Local Obstruction Predicate

$P_{\mathrm{PS35}}$ holds if an admissible local profile is not assigned to a
named or residual status, if an attainability obligation is hidden, or if an
open branch, local ST obligation, unsourced nonattainability label, or merely
classified branch is treated as excluded.

### Local Lemmas to Prove

Define coverage completeness by

$$
\mathbf C_{\rm cover}:\quad
\forall V\in\mathcal S_{\rm loc},\quad
V\in\mathcal U_{\rm named}
\text{ or }
V\in\mathcal R_{\rm named}.
$$

Define realized-branch exclusion by

$$
\mathbf C_{\rm real}:\quad
\forall V\in\mathcal S_{\rm loc},\quad
V\in\mathcal U_{\rm excl}.
$$

Define full exclusion completeness by

$$
\mathbf C_{\rm excl}:\quad
\mathbf C_{\rm real}
\quad\text{and}\quad
\mathcal O_{\rm ST}=\emptyset
\quad\text{and}\quad
\mathcal O_{\rm att}=\emptyset
\quad\text{and}\quad
\mathcal O_{\rm src}=\emptyset.
$$

Thus

$$
\mathbf C_{\rm real}\quad\Longleftrightarrow\quad
\mathcal R_{\rm open}=\emptyset
$$

and

$$
\mathbf C_{\rm excl}\quad\Longleftrightarrow\quad
\mathcal R_{\rm open}=\emptyset,\quad
\mathcal O_{\rm ST}=\emptyset,\quad
\mathcal O_{\rm att}=\emptyset,\quad
\mathcal O_{\rm src}=\emptyset.
$$

**Lemma PS35.1 -- Every profile is classified by named predicates or residual
membership.**
The predicate $\mathbf C_{\rm cover}$ holds.

**Proof.** `PS34` defines

$$
\mathcal U_{\rm named}=\bigcup_{\alpha\in A}\mathcal U_\alpha,
\qquad
\mathcal R_{\rm named}
=\mathcal S_{\rm loc}\setminus\mathcal U_{\rm named}.
$$

For every $V\in\mathcal S_{\rm loc}$, either
$V\in\mathcal U_{\rm named}$ or $V\notin\mathcal U_{\rm named}$. In the second
case, the complement identity gives $V\in\mathcal R_{\rm named}$.

**Lemma PS35.2 -- Completed decomposition preserves endpoint obligations.**
If a branch is excluded, the exclusion status is recorded. If a branch is
not excluded, its endpoint, realization, residual, attainability, or
nonattainability-source obligation is recorded.

**Proof.** Nodes `PS31`--`PS34` attach theorem, realization, and residual
statuses to every branch that reaches them. `PS33` supplies the
nonattainability source when the status is $\mathrm{proved\ nonattainable}$;
without that source the item is recorded in $\mathcal O_{\rm src}$ and
remains open.

**Lemma PS35.3 -- No silent branch remains, but open branches may remain.**
If `PS30` gives a complete defect vector and `PS34` gives an exact residual
complement, then every local profile branch has a named status. Therefore no
silent branch remains. However, if any branch status is

$$
\mathrm{realized\ nonexcluded},\quad
\mathrm{undecided},\quad
\mathrm{formal\ only},\quad
\mathrm{residual\ not\ closed\ by\ }ST20,\quad
\mathrm{ST\ obligation\ open},\quad
\mathrm{unresolved},
$$

and not $\mathrm{excluded}$ or $\mathrm{proved\ nonattainable}$, then the local
contradiction is not complete.

**Proof.** A silent branch is either an unrecorded defect or an element outside
the named/residual coverage. The first case contradicts the complete defect
vector from `PS30`; the second contradicts $\mathbf C_{\rm cover}$ from
`PS34`. This proves classification coverage only. A branch with one of the
displayed open statuses is recorded. If it has admissible provenance, it is not
in $\mathcal U_{\rm excl}$, so it remains in $\mathcal R_{\rm open}$. If its
provenance is unsettled, it remains in $\mathcal O_{\rm att}$. In either case
it cannot be used as a contradiction.

**Lemma PS35.4 -- Exclusion report separates closed and open statuses.**
Define

$$
\mathcal E_{\rm report}
=
\left\{
(\mathcal U_\alpha,\mathrm{status}_\alpha,\mathrm{witness}_\alpha):
\alpha\in A
\right\}
\cup
\left\{
(\mathcal R_i,\mathrm{obligation}_i,\mathrm{witness}_i)
\right\}
\cup
\left\{
(O_j,\mathrm{attainability\ obligation}_j,\mathrm{witness}_j)
\right\}
\cup
\left\{
(N_k,\mathrm{nonattainability\ source\ status}_k,\mathrm{failed\ necessary\ condition}_k)
\right\}.
$$

The allowed statuses are

$$
\mathrm{excluded},\quad
\mathrm{excluded\ by\ }ST20,\quad
\mathrm{proved\ nonattainable},\quad
\mathrm{realized\ nonexcluded},\quad
\mathrm{formal\ only},\quad
\mathrm{undecided},\quad
\mathrm{residual\ not\ closed\ by\ }ST20,\quad
\mathrm{ST\ obligation\ open}.
$$

The status $\mathrm{excluded}$ is closed for realized elements of
$\mathcal S_{\rm loc}$; the status $\mathrm{excluded\ by\ }ST20$ is the
closed status of the generic local residual row only when
$\mathcal O_{\rm ST}=\emptyset$. The status
$\mathrm{proved\ nonattainable}$ is closed for candidates in
$\mathcal N_{\rm att}^{0}$ only when the displayed
`PS33` source and failed necessary condition are recorded, equivalently when
the candidate is not in $\mathcal O_{\rm src}$. All other statuses are open
unless a later theorem handles them.

**Proof.** The status $\mathrm{excluded}$ records a valid `PS32`
contradiction for an actually obtainable branch. The status
$\mathrm{excluded\ by\ }ST20$ records the local residual closure theorem and
its `PS31` hypothesis match, including the empty ST-obligation ledger. The
status
$\mathrm{proved\ nonattainable}$ records a valid `PS33` non-realization proof
with a named failed necessary condition, so it closes the candidate before it
can enter $\mathcal S_{\rm loc}$. Each other status either has an actual
non-excluded profile, lacks an attaining sequence, lacks a nonattainability
proof, or is a residual witness. None is a contradiction without an additional
theorem.

**Lemma PS35.5 -- Handoff to boundary compatibility is conditional.**
If $\mathcal R_{\rm open}=\emptyset$, $\mathcal O_{\rm ST}=\emptyset$,
$\mathcal O_{\rm att}=\emptyset$, and $\mathcal O_{\rm src}=\emptyset`,
`PS35` passes a no-local-profile conclusion to `Bound_partial`. If any of
these checks fails, it passes the open branch, ST-obligation,
attainability-obligation, or unsourced nonattainability list for compatibility
checking only, not as a local contradiction.

**Proof.** Boundary compatibility cannot eliminate an unresolved interior
branch unless it supplies a new estimate. Therefore the handoff must preserve
whether the local audit has already excluded every branch or merely
classified them with explicit obligations.

### Specific Estimate

The decisive statements are

$$
\mathbf C_{\rm cover}\text{ true},
\qquad
\mathbf C_{\rm excl}
\Longleftrightarrow
\mathcal R_{\rm open}=\emptyset
\text{ and }
\mathcal O_{\rm ST}=\emptyset
\text{ and }
\mathcal O_{\rm att}=\emptyset
\text{ and }
\mathcal O_{\rm src}=\emptyset.
$$

### Practical Verification Steps

1. List all branch predicates in order.
2. Confirm each predicate has a conclusion.
3. Build $\mathcal E_{\rm report}$ with a witness for every status.
4. Compute $\mathcal R_{\rm ST20}$, $\mathcal R_{\rm open}$, and
   $\mathcal O_{\rm ST}$.
5. Compute $\mathcal O_{\rm att}$ and $\mathcal O_{\rm src}$ and verify both
   are empty before closure.
6. Verify that the generic residual row, if present, is in
   $\mathcal R_{\rm ST20}$ or is listed as an explicit open residual
   obligation.
7. Pass either the no-local-profile conclusion or the open-obligation list to
   the compatibility nodes.

## Estimate Step $B_{\mathrm{PS35}}$

The estimate step is coverage verification plus exclusion-status reporting.

## Failure Case

Failure name: incomplete or nonclosed local branch decomposition.

Analytic meaning: there is an admissible profile branch not represented by the
named alternatives or residual complement, or there is a represented branch
whose status is open or whose nonattainability source is missing but is being
treated as closed.

## Refinement Step

Allowed refinements:

1. add the missing branch predicate;
2. update the residual complement;
3. rerun the defect audit for the missing branch;
4. rerun endpoint matching if the new branch has an exclusion theorem.

Progress measure: the uncovered profile is either named or added to the exact
residual complement, every realized open branch is either excluded or
preserved as an explicit obligation, and every nonattainability claim is
backed by its `PS33` source.

## Data Passed Forward

The next proof step is `Bound_partial`. The data passed forward are

$$
\Gamma_{\mathrm{PS35}}
=
\Gamma_{\mathrm{PS34}}
\cup
\left\{
\begin{array}{l}
\mathbf C_{\rm cover}\text{ true},\\
\mathcal E_{\rm report}\text{ complete},\\
\mathcal R_{\rm open}\text{ computed},\\
\mathcal R_{\rm ST20}\text{ computed},\\
\mathcal O_{\rm ST}\text{ computed},\\
\mathcal O_{\rm att}\text{ computed},\\
\mathcal N_{\rm att}^{0}\text{ computed},\\
\mathcal O_{\rm src}\text{ computed},\\
\mathbf C_{\rm excl}\Longleftrightarrow
\mathcal R_{\rm open}=\emptyset,\quad
\mathcal O_{\rm ST}=\emptyset,\quad
\mathcal O_{\rm att}=\emptyset,\quad
\mathcal O_{\rm src}=\emptyset
\end{array}
\right\}.
$$

The next node receives either

$$
\text{local realized branches excluded, ST obligations empty, and nonattainability sources verified},
$$

or

$$
\text{local branch decomposition complete but open obligations remain}.
$$

---

# 63. `Bound_partial` -- Boundary or Physical-Domain Compatibility

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a suitable weak solution on either $\mathbb R^3$ or a physical
domain $\Omega$, together with every physical cylinder and rescaled frame used
downstream.

### Standing Assumptions

The incoming record states that all local profile branches have been
classified with an exclusion report, or that the no-local-profile conclusion is
active. Classification alone is not a contradiction: the terminal block may
close only with a verified negative branch record, or else it must report the
remaining open obligations.

### Objects Inspected

Inspect the boundary status of every active cylinder and frame, the boundary
condition, trace spaces, pressure representative, flattening map, boundary
local energy inequality, finite-entry record, and flux terms in the local
energy inequality.

### Dependencies Used

The domain comes from `H0`; terminal finite entry comes from `D_E`, `PS0`, and
`PS1`; cutoff commutators come from `PS17`, `PS20`, and `PS30`; endpoint
registration and application come from `PS31` and `PS32`; completed profile
status comes from `PS35`.

### Local Obstruction Predicate

$P_{\mathrm{Bound\_partial}}$ holds if a physical boundary, boundary-visible
frame, uncontrolled boundary scaling, or missing terminal finite-entry record
enters a cylinder used by the proof without the record required by the
endpoint theorem.

### Local Lemmas to Prove

**Lemma Bound_partial.1 -- Every active cylinder has a boundary-status
record.**
For every physical cylinder or active frame used by the proof, record exactly
one of

$$
\mathrm{interior},\qquad
\mathrm{whole\ space},\qquad
\mathrm{boundary\ visible},\qquad
\mathrm{boundary\ unresolved}.
$$

For a physical cylinder, the interior record is

$$
\operatorname{dist}(B_r(x_0),\partial\Omega)>0.
$$

For a rescaled frame

$$
x=x_n+\lambda_n y,
$$

the scale-sensitive boundary coordinate is

$$
d_n^{\partial}
=
\frac{\operatorname{dist}(x_n,\partial\Omega)}{\lambda_n}.
$$

If $d_n^{\partial}\to\infty$, the profile has an interior whole-space limit. If
$d_n^{\partial}\to d_*\in[0,\infty)$ along the declared subsequence, a
boundary or half-space limit may be visible. If no controlled subsequential
behavior of $d_n^{\partial}$ is recorded, the branch has a named
boundary-geometry defect.

**Proof.** For a physical cylinder, the condition
$\operatorname{dist}(B_r(x_0),\partial\Omega)>0$ gives an open collar between
the support of every spatial test function and $\partial\Omega$. Thus the
distributional equation and local energy inequality can be tested with
functions compactly supported in $\Omega$, and spatial integration by parts
produces only interior distributional terms. No boundary trace or boundary
flux is used.

For a rescaled frame, a point at bounded $y$-distance from the origin
corresponds to $x_n+\lambda_n y$ in physical variables. The physical distance
to $\partial\Omega$ seen at unit scale is therefore exactly the physical
distance divided by $\lambda_n$, namely $d_n^\partial$. If
$d_n^\partial\to\infty$, then every fixed $y$-ball is eventually contained in
$\Omega$ after rescaling, so the limiting frame is interior and whole-space on
every compact set. If $d_n^\partial\to d_*<\infty$, some fixed-scale compact
sets may see the boundary, so an interior theorem is not verified. If no
subsequence or diagonal extraction controls $d_n^\partial$, the proof does not
know whether a fixed endpoint cylinder is interior or boundary-visible; this is
exactly the named boundary-geometry defect.

**Lemma Bound_partial.2 -- Whole-space profiles have no physical boundary.**
For $\Omega=\mathbb R^3$, every finite cylinder is interior.

**Proof.** When $\Omega=\mathbb R^3$, the boundary set is empty and every
finite ball $B_R$ is an admissible spatial support for compactly supported
tests. In the rescaled variables, the physical domains remain all of
$\mathbb R^3$ for every $n$, so there is no boundary ratio to control and no
boundary trace can enter the local energy inequality. The record is
therefore $\mathrm{whole\ space}$ for every active cylinder and frame.

**Lemma Bound_partial.3 -- Boundary branches require registered endpoint
theorems.**
If a boundary-visible cylinder or frame appears, the exact boundary regularity
theorem must already be registered through `PS31` and applied through `PS32`.
The boundary endpoint record contains

$$
\text{boundary condition},\quad
\text{trace class},\quad
\text{pressure convention},\quad
\text{flattening map},\quad
\text{boundary local energy inequality},\quad
\text{boundary smallness criterion}.
$$

For no-slip NS3D, for instance, the record states whether

$$
u|_{\partial\Omega}=0
$$

holds in the required trace sense and whether the pressure is controlled in
the boundary cylinder.

**Proof.** A boundary-visible cylinder has test functions whose supports may
meet $\partial\Omega$ after flattening or rescaling. The integration-by-parts
identity then uses boundary traces and the boundary version of the local
energy inequality, not merely the interior distributional equation. The
endpoint theorem must know the boundary condition, the trace class in which it
holds, the pressure representative or oscillation convention, and the
flattening map used to transfer the estimate to a model boundary cylinder.
These data are not hypotheses of the interior CKN theorem. Therefore a
boundary-visible branch can be closed only by a boundary endpoint theorem
registered in `PS31` with those hypotheses and applied in `PS32`; otherwise
the endpoint invocation is a theorem-hypothesis mismatch rather than a
contradiction.

**Lemma Bound_partial.4 -- Interior shrinking and terminal entry have separate
records.**
Shrinking to an interior cylinder is allowed only when the target point or
selected active frame is strictly interior at the relevant scale. If
$x_0\in\partial\Omega$, no backward cylinder centered at $x_0$ is fully
interior, so the branch must be treated as a boundary branch. A top-time
terminal face is not a spatial boundary term, but every terminal branch still
requires the finite-entry record

$$
A+C+D+E<\infty
$$

from `D_E`, `PS0`, and `PS1`.

**Proof.** If $x_0$ has positive distance from $\partial\Omega$, one may choose
$r<\operatorname{dist}(x_0,\partial\Omega)$ and obtain a strictly interior
backward cylinder. If $x_0\in\partial\Omega$, every centered ball
$B_r(x_0)$ intersects the complement of $\Omega$, so no reduction to an
interior cylinder is possible without changing the target point or frame. The
branch must therefore retain its boundary status and use a boundary endpoint
record.

For terminal time, the obstruction is different. Backward cylinders
$B_r(x_0)\times(t_0-r^2,t_0)$ do not create a spatial boundary face at
$t=t_0$, so no physical boundary flux is added to the local energy identity.
However, the quantities $A$, $C$, $D$, and $E$ must be finite on the entry
cylinder before CKN smallness, normalization, or profile extraction is
well-defined. Thus terminal-time branches are harmless only after the
finite-entry record has been supplied; without it they are finite-entry
obstructions, not interior branches.

### Specific Estimate

The decisive interior and frame conditions are

$$
\operatorname{dist}(B_r(x_0),\partial\Omega)>0.
$$

and

$$
\frac{\operatorname{dist}(x_n,\partial\Omega)}{\lambda_n}
\to\infty
$$

for an interior whole-space limit. Boundary-visible limits require the
boundary endpoint record, and uncontrolled ratios are named
boundary-geometry defects.

The boundary/entry obstruction ledger is

$$
\mathcal O_{\partial}
=
\left\{
\begin{array}{l}
\text{active cylinders or frames with }\mathrm{boundary\ unresolved},\\
\text{boundary-visible branches missing the registered boundary endpoint theorem},\\
\text{terminal branches missing the finite-entry record}
\end{array}
\right\}.
$$

### Practical Verification Steps

1. Identify the physical domain.
2. Build the boundary-status ledger for every active cylinder and frame.
3. For rescaled frames, record the subsequential behavior of
   $\operatorname{dist}(x_n,\partial\Omega)/\lambda_n$.
4. If whole-space or interior, record the whole-space or interior record.
5. If boundary-visible, attach the `PS31` boundary endpoint theorem record
   and the `PS32` application.
6. If terminal time is involved, attach the finite-entry record from `D_E`,
   `PS0`, and `PS1`.
7. Assign missing boundary hypotheses or uncontrolled boundary scaling to a
   named boundary-geometry defect.
8. Verify $\mathcal O_{\partial}=\emptyset$ before treating boundary and
   terminal compatibility as closed.

## Estimate Step $B_{\mathrm{Bound\_partial}}$

The estimate step is the interior-cylinder, scale-sensitive frame,
boundary-endpoint, or terminal-entry verification.

## Failure Case

Failure names: unresolved boundary compatibility; boundary-geometry defect;
missing boundary endpoint theorem; terminal finite-entry obstruction.

Analytic meaning: a proof cylinder or frame touches or sees a physical
boundary without the estimates required by the endpoint theorem, or a terminal
branch lacks the finite-entry record required to enter the profile machinery.

## Refinement Step

Allowed refinements:

1. shrink to an interior cylinder only when the target point or selected active
   frame is strictly interior at the relevant scale;
2. add boundary trace and pressure estimates;
3. switch to a boundary regularity theorem registered in `PS31`;
4. pass the boundary theorem through `PS32`;
5. close or explicitly list terminal finite-entry obstructions;
6. assign boundary defects to `PS30` or `PS31`.

Progress measure: every cylinder and frame is whole-space, interior,
boundary-visible with a verified boundary endpoint theorem, or explicitly
marked by a named boundary/entry defect.

## Data Passed Forward

The next proof step is `Bound_B`. The data passed forward are

$$
\Gamma_{\mathrm{Bound\_partial}}
=
\Gamma_{\mathrm{PS35}}
\cup
\left\{
\begin{array}{l}
\text{boundary status of every active cylinder/frame},\\
\text{interior or whole-space record},\\
\text{or boundary theorem record},\\
\mathcal O_{\partial}\text{ empty or explicit boundary/entry defect list}
\end{array}
\right\}.
$$

A boundary-visible branch is closed only if the correct boundary endpoint
theorem has already been verified through `PS31` and applied through `PS32`.
If $\mathcal O_{\partial}\ne\emptyset$, the terminal block reports the
corresponding boundary or finite-entry obligation rather than a contradiction.

---

# 64. `Bound_B` -- Forcing, Lower-Order, or Cutoff-Source Compatibility

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is the selected localized or renormalized velocity equation,
together with the source ledger for the endpoint theorem being applied.

### Standing Assumptions

The incoming record states that the physical equation is unforced and all
additional terms are created only by localization, pressure normalization,
tail removal, divergence correction, or modulation. No such term may be
dropped unless it is absent on the endpoint cylinder, vanishes in the endpoint
source topology, is absorbed by a verified estimate, or is explicitly included
in the endpoint theorem.

### Objects Inspected

Inspect the source ledger

$$
\mathcal F
=
\{F^{\rm cut},F^{\rm press},F^{\rm mod},F^{\rm div},F^{\rm tail},F^{\rm force}\},
$$

the endpoint source topology $Y_{T_\beta}(Q)$, and every right-hand term in
the localized or normalized equation.

### Dependencies Used

Cutoff terms come from `PS17` and `PS20`; modulation terms from `PS5`,
`PS26`, and `PS28`; pressure terms from `PS4` and `PS30`; endpoint source
topologies and equation classes come from `PS31`.

### Local Obstruction Predicate

$P_{\mathrm{Bound\_B}}$ holds when a source term remains unresolved: it is not
absent, not vanishing in $Y_{T_\beta}$, not absorbed in the required topology,
and not included in the endpoint theorem's equation.

### Local Lemmas to Prove

**Lemma Bound_B.1 -- Physical forcing is absent.**
For the incompressible unforced NS3D equation, the physical source term is
identically zero and the source ledger entry $F^{\rm force}$ is marked
$\mathrm{absent}$.

**Proof.** The equation in `H0` is
$\partial_tu+(u\cdot\nabla)u+\nabla p=\Delta u$, with no prescribed external
force on the right-hand side. Hence every nonzero right-hand term appearing
after this point must come from a proof operation: cutoff localization,
pressure normalization, divergence correction, tail truncation, or modulation.
The physical forcing entry is therefore not estimated; it is identified with
zero before localization and marked $\mathrm{absent}$ in $\mathcal F$.

**Lemma Bound_B.2 -- Source convergence uses the endpoint topology.**
For each endpoint theorem $T_\beta$, `PS31` specifies a source space

$$
Y_{T_\beta}(Q).
$$

A source marked $\mathrm{vanishing}$ must satisfy

$$
\|F_n\|_{Y_{T_\beta}(Q_R)}\to0
$$

on every endpoint cylinder $Q_R$. Ordinary local compactness may use, for
example,

$$
L^1_tW^{-1,3/2}_x(Q_R)
$$

or

$$
L^1_tH^{-m}_x(Q_R),\qquad m\ge3,
$$

while an energy-based theorem may use $L^2_tH^{-1}_x(Q_R)$ only when the
source terms actually belong to that space.

**Proof.** Let $\varphi$ be an admissible test object for the endpoint theorem
$T_\beta$. The source contribution in the weak formulation has the form

$$
\langle F_n,\varphi\rangle_{Y_{T_\beta},Y_{T_\beta}' }.
$$

If $\|F_n\|_{Y_{T_\beta}(Q_R)}\to0$ and the test norm is bounded in the dual
space required by $T_\beta$, this contribution vanishes and the source does
not appear in the endpoint equation. Conversely, convergence in some unrelated
space, such as a fixed $L^1_tH^{-1}_x$, does not imply that the pairing used
by $T_\beta$ vanishes unless the registry supplies the corresponding
embedding. The proof must therefore use the topology in the endpoint registry,
and the source is removable exactly in that topology.

**Lemma Bound_B.3 -- Cutoff sources vanish only where the cutoff is constant.**
If $W=\chi V$, then terms such as

$$
(V\cdot\nabla\chi)V,\qquad
P\nabla\chi,\qquad
2\nabla\chi\cdot\nabla V,\qquad
V\Delta\chi
$$

are supported on $\operatorname{supp}\nabla\chi$. They are absent on an
endpoint cylinder only when that cylinder lies inside $\{\chi=1\}$. Otherwise
they remain explicit source terms and must be controlled in
$Y_{T_\beta}$.

**Proof.** Applying the equation for $V$ to $W=\chi V$ differentiates the
cutoff whenever a derivative or transport operator lands on $\chi$. These
terms are products of $V$, $P$, or $\nabla V$ with $\nabla\chi$ or
$\Delta\chi$. Since the derivative factors vanish exactly on the set where
$\chi$ is locally constant, the commutators are zero on any endpoint cylinder
contained in $\{\chi=1\}$. On a cylinder meeting the cutoff annulus,
$\nabla\chi$ and $\Delta\chi$ are bounded coefficients but the products are
not zero. They must therefore remain in $\mathcal F$ and be estimated in
$Y_{T_\beta}$, including the pressure term if present.

**Lemma Bound_B.4 -- Pressure cutoff terms require pressure-tail control.**
A pressure source such as $(P_n-\bar P_n)\nabla\chi$ is compatible only if the
pressure defect coordinate from `PS30` is marked absent or absorbed in the
same source topology required here. This may be verified by

$$
P_n-\bar P_n\to0
\quad\text{in }L^{3/2}
\quad\text{on the cutoff annulus},
$$

or by a pressure decomposition showing that the Calderon-Zygmund and harmonic
pressure pieces vanish or are absorbed in $Y_{T_\beta}$.

**Proof.** The term $(P_n-\bar P_n)\nabla\chi$ is supported where
$\nabla\chi\ne0$, and $\nabla\chi$ contributes only a fixed bounded multiplier.
Thus its size in any reasonable negative or distributional source topology is
controlled by the pressure oscillation on the annulus, not by velocity
compactness on the endpoint core. If `PS30` has marked the pressure defect
absent, the required pressure convergence or decomposition is already
available and the source can be marked $\mathrm{vanishing}$ or
$\mathrm{absorbed}$. If the pressure defect remains, the cutoff pressure term
may carry a nonzero harmonic or Calderon-Zygmund tail into the endpoint
equation. In that case it is compatible only if the endpoint theorem explicitly
includes that source; otherwise the ledger entry is unresolved.

**Lemma Bound_B.5 -- Modulation terms may define the endpoint equation.**
Suppose the normalized equation contains

$$
a_n(\tau)(V_n+y\cdot\nabla V_n)
+
b_n(\tau)\cdot\nabla V_n.
$$

If $a_n\to a$ and $b_n\to b$, the limiting equation has drift coefficients
$a,b$. This is compatible if the endpoint theorem is registered for that
drifted equation. If the endpoint theorem is ordinary NS3D, then compatibility
requires $a=0$ and $b=0$ and the residual modulation source must vanish in
$Y_{T_\beta}$.

**Proof.** Testing against a compact smooth divergence-free test function
$\varphi$ gives contributions of the form

$$
\int a_n(\tau)\langle V_n+y\cdot\nabla V_n,\varphi\rangle\,d\tau
+
\int b_n(\tau)\cdot\langle \nabla V_n,\varphi\rangle\,d\tau .
$$

After the compactness and weak convergence already recorded in the branch, the
limits of these pairings are the same expressions with $a,b,V$ in place of
$a_n,b_n,V_n$, plus any residual that must vanish in $Y_{T_\beta}$. If
$a$ or $b$ is nonzero, the limiting weak equation contains the corresponding
dilation or translation drift. It is therefore not the ordinary NS3D endpoint
equation unless $a=b=0$. A drifted endpoint theorem may still apply, but only
when that theorem is explicitly registered with the drift in its equation
field.

**Lemma Bound_B.6 -- The source ledger is the pass/fail object.**
Each entry of $\mathcal F$ receives one of

$$
\mathrm{absent},\quad
\mathrm{vanishing},\quad
\mathrm{included\ in\ theorem},\quad
\mathrm{absorbed},\quad
\mathrm{unresolved}.
$$

`Bound_B` passes exactly when no entry is marked $\mathrm{unresolved}$.

Define the unresolved source-obstruction ledger by

$$
\mathcal O_{\mathcal F}
=
\{F\in\mathcal F:\mathrm{status}(F)=\mathrm{unresolved}\}.
$$

Thus source compatibility is closed exactly when
$\mathcal O_{\mathcal F}=\emptyset$.

**Proof.** For each source entry there are four ways it can fail to obstruct
the endpoint theorem. If it is $\mathrm{absent}$, it does not occur in the
localized equation on the endpoint cylinder. If it is $\mathrm{vanishing}$, its
weak contribution tends to zero in $Y_{T_\beta}$. If it is
$\mathrm{absorbed}$, a recorded estimate places it under a smallness or
compactness hypothesis of the endpoint theorem. If it is
$\mathrm{included\ in\ theorem}$, the endpoint theorem is for the equation that
contains that term. These alternatives exhaust the verified non-obstructive
cases. Any source outside them changes the equation seen by the endpoint
theorem without a matching hypothesis, so it is precisely an unresolved source
defect.

### Specific Estimate

For every source marked $\mathrm{vanishing}$, the decisive estimate is

$$
\|F_n\|_{Y_{T_\beta}(Q_R)}\to0
$$

where $Y_{T_\beta}$ is specified in the endpoint theorem registry. Sources
marked $\mathrm{included\ in\ theorem}$ must appear in the registered endpoint
equation, and sources marked $\mathrm{absorbed}$ must cite the absorption
estimate in that same topology.

### Practical Verification Steps

1. Write the localized equation explicitly.
2. Build the ledger $\mathcal F$ with entries for cutoff, pressure,
   modulation, divergence, tail, and physical forcing.
3. Read $Y_{T_\beta}$ and the endpoint equation from the `PS31` theorem
  registry.
4. Mark physical forcing absent.
5. Mark cutoff terms absent only when the endpoint cylinder lies inside
   $\{\chi=1\}$; otherwise estimate them in $Y_{T_\beta}$.
6. Check pressure cutoff terms against the `PS30` pressure defect coordinate.
7. Mark modulation terms as vanishing, theorem-included, absorbed, or
   unresolved according to the endpoint equation.
8. Compute $\mathcal O_{\mathcal F}$ and pass as closed only if
   $\mathcal O_{\mathcal F}=\emptyset$.

## Estimate Step $B_{\mathrm{Bound\_B}}$

The estimate step is source ledger verification in the endpoint-specific
topology $Y_{T_\beta}$.

## Failure Case

Failure name: unresolved local source term.

Analytic meaning: the branch equation differs from the registered endpoint
equation by a term that is not absent, not vanishing, not absorbed, and not
included in the endpoint theorem.

## Refinement Step

Allowed refinements:

1. move the endpoint cylinder inside $\{\chi=1\}$ when justified;
2. change pressure decomposition;
3. refine modulation;
4. strengthen source convergence in $Y_{T_\beta}$;
5. register a drifted or forced endpoint theorem in `PS31`;
6. assign source defects to `PS30` or `PS31`.

Progress measure: every source is absent, vanishing in $Y_{T_\beta}$,
absorbed, theorem-included, or named as an unresolved source defect.

## Data Passed Forward

The next proof step is `Bound_Sigma`. The data passed forward are

$$
\Gamma_{\mathrm{Bound\_B}}
=
\Gamma_{\mathrm{Bound\_partial}}
\cup
\left\{
\begin{array}{l}
\mathcal F\text{ source ledger},\\
Y_{T_\beta}\text{ endpoint source topology},\\
\text{all sources absent, vanishing, absorbed, or theorem-included},\\
\mathcal O_{\mathcal F}\text{ empty or explicit unresolved source defect list}
\end{array}
\right\}.
$$

If $\mathcal O_{\mathcal F}\ne\emptyset$, the terminal block reports the
corresponding source defect rather than treating the endpoint equation as
matched.

---

# 65. `Bound_Sigma` -- Sufficiency of Input Data and Selected Objects

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The objects inspected are the NS3D hypotheses already assembled for the final
local exclusion, not a new PDE solution. This node is the final no-hidden-object
audit before the local branch record is connected to the target theorem.

### Standing Assumptions

The incoming record states that local branch decomposition, boundary
compatibility, and source compatibility have been checked, with any open
branches preserved as explicit obligations rather than erased.

### Objects Inspected

Inspect every selected mathematical object used in the endpoint implication,
its source node, branch condition, subsequence, frame, pressure gauge,
representative identity, and compatibility status. Also inspect the constants
and inequalities used in the final contradiction.

### Dependencies Used

Every previous node contributes to the proof record matrix
$\mathfrak S_T$ and constants ledger $\mathcal K$.

### Local Obstruction Predicate

$P_{\mathrm{Bound\_\Sigma}}$ holds when a final proof step refers to an object
that was not constructed, was constructed on an incompatible subsequence, is
used outside its active branch condition, has an incompatible frame or pressure
gauge, or relies on unverified constants.

### Local Lemmas to Prove

**Lemma Bound_Sigma.1 -- Data completeness uses the endpoint record
matrix.**
For each endpoint application, define

$$
\mathfrak S_T
=
\left\{
(X,\mathrm{branch},\mathrm{node},\mathrm{subsequence},\mathrm{frame},
\mathrm{gauge},\mathrm{status})
\right\}.
$$

The record is finite for each endpoint application, or countable with an
explicit exhaustion and diagonal subsequence. Completeness means

$$
\forall X\in\mathfrak S_T,\qquad
X\text{ has a construction node and compatibility status.}
$$

**Proof.** Fix one endpoint application $T$. Its statement mentions a
definite set of solution objects, domains, frames, gauges, limits, source
terms, and constants. Each such object is entered as a row of $\mathfrak S_T$.
If the endpoint application is finite, checking all rows proves that no object
in the endpoint implication is anonymous. If the endpoint application is built
by exhaustion, for example over $R=1,2,\ldots$, completeness is checked at
each finite level and the matrix must also name the diagonal subsequence that
survives all levels. Thus every object used at the endpoint has both a
construction node and a compatibility status before the final implication is
allowed to cite it.

**Lemma Bound_Sigma.2 -- Subsequence lineage is common.**
Every object has a field $\mathrm{subseq}(X)$. All objects used in the same
final implication must be built along a common subsequence or a declared
diagonal subsequence. For example,

$$
V_{n_k}\to V,\qquad
P_{n_k}\rightharpoonup P,\qquad
a_{n_k}\to a,\qquad
b_{n_k}\to b
$$

must all hold for the same $n_k$, unless the matrix declares the diagonal
subsequence that makes them simultaneous.

**Proof.** Suppose the velocity limit is constructed along $n_k$ but the
pressure or modulation coefficient is constructed only along another
subsequence $m_j$. Without a declared common refinement, there is no single
sequence of approximate solutions for which all limits appear simultaneously.
The endpoint equation would then combine objects that were never obtained from
one prelimit branch. Requiring a common subsequence, or explicitly recording a
diagonal extraction $n_{k_\ell}$, ensures that every convergence statement in
the endpoint equation is true along the same indices and hence defines one
admissible limiting branch.

**Lemma Bound_Sigma.3 -- Branch-conditional objects are used only on active
branches.**
Each row of $\mathfrak S_T$ carries a branch condition. The final implication
may use an object only when its branch condition is active. A boundary pressure
estimate, for example, is required in a boundary-visible branch and irrelevant
in a verified interior branch.

**Proof.** A row whose branch condition is inactive is not false; it is simply
irrelevant to the branch currently being closed. For example, an interior
branch has no boundary pressure term, so a boundary pressure estimate should
not be required there. Conversely, a boundary-visible branch cannot borrow an
interior-only pressure record. The branch condition column prevents both
errors: it suppresses irrelevant hypotheses and forces every active endpoint
hypothesis to have a record in the branch where it is actually used.

**Lemma Bound_Sigma.4 -- Representative compatibility records all operations.**
If the same object is produced in two nodes, the later node must either use the
same representative or record the transformation between representatives.
Allowed recorded operations include scalings, translations, rotations,
Galilean transforms, pressure gauges, subsequence restrictions, diagonal
extractions, time translations, cutoff localizations, and frame-to-physical
pullbacks.

For every duplicated object, record the exact identity, for example

$$
V_n(y,s)=r_nu(x_*+r_ny,T+r_n^2s),
$$

or

$$
P^R=P-(P)_{B_R}(t).
$$

**Proof.** The final contradiction often compares a lower bound or defect in
one representation with an endpoint conclusion in another. This comparison is
valid only if the record gives the identity connecting those representations.
A scaling changes the cylinder size and the normalization of velocity and
pressure; a time translation changes the time interval; a pressure gauge
changes the representative but not the gradient; a cutoff changes the equation
by adding sources; and a subsequence restriction changes the lineage of every
limit. Recording the exact formula for each duplicated object makes these
changes auditable and guarantees that the endpoint theorem and the prelimit
contradiction refer to the same branch object.

**Lemma Bound_Sigma.5 -- Constants and inequalities are verified.**
The constants ledger

$$
\mathcal K
=
\{\varepsilon_0,\varepsilon_v,\eta_*,M,R,\sigma,\kappa,\ldots\}
$$

contains every constant used in the final contradiction and verifies the
needed ordering, such as

$$
\eta_* >0,\qquad
\eta_0\ge\varepsilon_{\rm CKN},\qquad
C+D<\varepsilon_{\rm CKN}/2.
$$

It is not enough to record separately that a lower bound and a smallness
theorem exist; the ledger must verify that their constants are compatible.

**Proof.** The endpoint contradiction is not produced by the existence of a
lower bound and a smallness theorem separately. It is produced by an
incompatible chain of inequalities in one normalization. For example, a
concentration lower bound $\eta_*$ contradicts a CKN smallness conclusion only
after the ledger proves that $\eta_*$ is measured on the same cylinder and
that the upper bound is below the same threshold. The ledger records all
thresholds, losses, covering constants, shrink factors, and chosen scales.
Checking their order prevents a proof from silently using an epsilon smaller
than the one actually available or a radius outside the range where the
estimate was proved.

### Specific Estimate

The decisive statement is

$$
\forall X\in\mathfrak S_T,\qquad
X\text{ has a prior construction node and compatibility status.}
$$

with common-subsequence, active-branch, representative, gauge, and constants
compatibility verified.

The insufficiency-obstruction ledger is

$$
\mathcal O_{\Sigma}
=
\left\{
\begin{array}{l}
\text{missing construction rows in }\mathfrak S_T,\\
\text{incompatible subsequence or diagonal lineage},\\
\text{inactive-branch object use},\\
\text{frame, pullback, or pressure-gauge mismatch},\\
\text{unverified constants or inequality ordering}
\end{array}
\right\}.
$$

The data sufficiency audit is closed exactly when
$\mathcal O_{\Sigma}=\emptyset$.

### Practical Verification Steps

1. Build $\mathfrak S_T$ for each endpoint theorem application.
2. Attach each object to the node that constructs it.
3. Record the active branch condition for every row.
4. Record $\mathrm{subseq}(X)$ and verify common or diagonal subsequence
   lineage.
5. Check frame, time-translation, cutoff, pullback, and pressure-gauge
   compatibility.
6. Record exact representative identities for duplicated objects.
7. Build $\mathcal K$ and verify the constants are in the right order for the
   contradiction being used.
8. Assign missing or inconsistent objects to $\mathcal O_{\Sigma}$ and pass as
   closed only if $\mathcal O_{\Sigma}=\emptyset$.

## Estimate Step $B_{\mathrm{Bound\_\Sigma}}$

The estimate step is proof record matrix and constants ledger
verification.

## Failure Case

Failure name: insufficient analytic data.

Analytic meaning: the final implication uses a mathematical object not
constructed by the local proof, constructed on the wrong subsequence, used in
the wrong branch, stated in an incompatible frame or gauge, or paired with
unverified constants.

## Refinement Step

Allowed refinements:

1. add the missing construction;
2. reconcile representatives through `PS26`;
3. declare the diagonal subsequence;
4. add missing pressure, scale, frame, time-translation, cutoff, or pullback
   data;
5. add the constants ledger or repair the constants;
6. rerun endpoint matching.

Progress measure: every object in $\mathfrak S_T$ becomes constructed,
branch-active, subsequence-compatible, frame/gauge-compatible, and verified by
the constants ledger.

## Data Passed Forward

The next proof step is `GC_T`. The data passed forward are

$$
\Gamma_{\mathrm{Bound\_\Sigma}}
=
\Gamma_{\mathrm{Bound\_B}}
\cup
\left\{
\begin{array}{l}
\mathfrak S_T\text{ proof record matrix complete},\\
\text{common subsequence/diagonal verified},\\
\text{frame and pressure gauge compatibility verified},\\
\mathcal K\text{ constants ledger verified},\\
\mathcal O_{\Sigma}\text{ empty or explicit insufficiency defect list}
\end{array}
\right\}.
$$

If $\mathcal O_{\Sigma}\ne\emptyset$, the terminal block reports the missing
object, lineage, frame, gauge, or constants obligation instead of closing the
target theorem.

---

# 66. `GC_T` -- Local Compatibility with the Target Regularity Statement

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a hypothetical target singular point of a suitable weak
solution, together with the declared target regularity statement.

### Standing Assumptions

The incoming record states that all local branches generated from such a
singular point have been classified and their endpoint, realization, or
residual statuses recorded in the exclusion report. This node distinguishes
classification coverage from exclusion completeness.

### Objects Inspected

Inspect the target declaration, finite-entry record, local singularity
criterion, positive concentration sequence, branch decomposition, open branch
record, and final status of every branch.

### Dependencies Used

Finite entry comes from `D_E`, `PS0`, and `PS1`; singular entry comes from
`C_mu` and normalized concentration from `PS1`; branch completeness and open
branch status come from `PS35`; nonattainability provenance comes from `PS33`;
boundary and source compatibility come from `Bound_partial` and `Bound_B`;
data sufficiency comes from `Bound_Sigma`; endpoint theorem status comes from
`PS31`.

### Local Obstruction Predicate

$P_{\mathrm{GC\_T}}$ holds if a target singular point lacks an admissible
finite-entry route into the profile machinery, if the declared target theorem
is broader than the local machinery covers, or if the final open-obligation
set defined below satisfies
$\mathfrak O_{\rm final}\ne\emptyset$.

### Local Lemmas to Prove

**Lemma GC_T.1 -- Singular entry generates concentration.**
Let $z_0=(x_0,t_0)$ be an interior point, or a terminal point with a verified
finite-entry radius. Suppose $z_0$ is singular in the sense that $u$ is not
locally bounded in any admissible backward cylinder. Then for every
sufficiently small entry-admissible radius $r$,

$$
C(u;z_0,r)+D(p;z_0,r)\ge\varepsilon_0.
$$

If the velocity-only criterion is used, then also

$$
C(u;z_0,r)\ge\varepsilon_v.
$$

**Proof.** Work only with radii for which the backward cylinder is admissible
and the finite-entry quantities are defined. The contrapositive of the CKN
epsilon-regularity theorem at $z_0$ states

$$
\text{if }C(u;z_0,r)+D(p;z_0,r)<\varepsilon_0
\text{ on an entry-admissible cylinder, then }z_0
\text{ is regular in a smaller cylinder.}
$$

If a singular $z_0$ had one sufficiently small entry-admissible radius with
$C(u;z_0,r)+D(p;z_0,r)<\varepsilon_0$, the theorem would give local boundedness
in a smaller admissible cylinder, contradicting the definition of singularity.
Therefore every sufficiently small entry-admissible radius satisfies the
displayed lower bound. If the proof chooses the velocity-only criterion, the
same argument is applied with the velocity threshold $\varepsilon_v$: a radius
with $C(u;z_0,r)<\varepsilon_v$ would imply regularity, so singularity forces
$C(u;z_0,r)\ge\varepsilon_v$ on every sufficiently small admissible radius.

If the target singular point lacks any finite-entry radius, then it does not
enter the `PS0`--`PS35` profile machinery. It is a finite-entry obstruction and
must be either excluded by a separate global-energy/pressure argument or listed
in $\mathcal O_{\rm target}$ as an open finite-entry obligation.

**Lemma GC_T.2 -- Target declaration controls the local-to-global scope.**
The record declares

$$
\mathcal T_{\rm target}
=
\begin{array}{c}
\text{interior regularity, terminal regularity, boundary regularity,}\\
\text{or whole-space blow-up exclusion}
\end{array}.
$$

If the target theorem is only interior regularity, then `GC_T` quantifies only
over points with positive distance from the parabolic boundary. If the target
includes boundary points, `Bound_partial` must have supplied the boundary
endpoint records. If the target is a global finite-time blow-up
exclusion, the record must include the global-to-local selection principle

$$
\text{finite-time singularity}
\Rightarrow
\exists x_*\text{ such that }(x_*,T)\in\Sigma(T),
$$

together with finite entry at that point.

**Proof.** The local profile machinery begins with an admissible local
singular-entry packet. For an interior target theorem, positive distance from
the parabolic boundary supplies the admissible cylinders. For a boundary
target theorem, admissibility also requires the boundary records from
`Bound_partial`. For a terminal theorem, the backward cylinders must carry the
finite-entry record. For a global finite-time blow-up statement, a further
selection principle is needed to produce a point $x_*$ at time $T$ to which the
local machinery applies. Since these hypotheses are different, the final
implication may quantify only over the target class whose bridge has been
recorded.

**Lemma GC_T.3 -- Local concentration enters the profile decomposition.**
The positive concentration packet constructed from an admissible singular
entry enters `C_mu` and `PS1`--`PS35` and is assigned to one admissible branch
or the named residual complement.

**Proof.** By Lemma GC_T.1, singularity supplies a finite-entry concentration
packet at arbitrarily small admissible scales. `Rec_N` records the selected
entry point and scales. `C_mu` turns the lower bound into the original-scale
packet used by the sieve. `PS1` normalizes the packet on a fixed cylinder, and
`PS2` records the center and parabolic scaling so that the normalized object
has admissible provenance from the original solution. `PS3` then sends the
packet into the local Type I predicate or its Type II negation, and the later
profile nodes refine those alternatives. Thus the resulting admissible local
profile lies in $\mathcal S_{\rm loc}$, and `PS35` gives

$$
\mathcal S_{\rm loc}
=
\mathcal U_{\rm named}\cup\mathcal R_{\rm named}.
$$

Hence the concentration packet is not merely named informally: it is assigned
by the recorded decomposition to a named branch or to the named residual
complement, with its open/closed status recorded separately in
$\mathcal E_{\rm report}$.

**Lemma GC_T.4 -- Exclusion completeness is required.**
Branch coverage alone gives only

$$
\mathcal S_{\rm loc}
=
\mathcal U_{\rm named}\cup\mathcal R_{\rm named}.
$$

Let $\mathfrak L$ denote the provenance-audited branch universe assembled from
realized local branches, `PS32` excluded branches, and the `PS33`
proved-nonattainability ledger. Define

$$
\mathfrak L_{\rm open}
=
\left\{
B\in\mathfrak L:
\mathrm{status}(B)\notin
\{\mathrm{excluded},\mathrm{proved\ nonattainable}\}
\right\}
$$

and

$$
\mathcal O_{\rm src}
=
\left\{
B\in\mathcal N_{\rm att}^{0}:
\text{the `PS33' source or failed necessary condition is missing}
\right\}.
$$

The branch-record part of target regularity is

$$
\mathfrak O_{\rm branch}
=
\mathfrak L_{\rm open}\cup\mathcal O_{\rm ST}
\cup\mathcal O_{\rm att}\cup\mathcal O_{\rm src}
=\emptyset.
$$

Equivalently, for the branch ledger alone,

$$
\mathcal R_{\rm open}=\emptyset
\quad\text{and}\quad
\mathcal O_{\rm ST}=\emptyset
\quad\text{and}\quad
\mathcal O_{\rm att}=\emptyset
\quad\text{and}\quad
\mathcal O_{\rm src}=\emptyset.
$$

The compatibility obstruction ledger is

$$
\mathcal O_{\rm target}
=
\left\{
\begin{array}{l}
\text{finite-entry obstruction not closed by a separate argument},\\
\text{target declaration broader than the local machinery},\\
\text{missing global-to-local selection bridge for a global target}
\end{array}
\right\},
$$

$$
\mathcal O_T
=
\left\{
T_\beta\in\mathscr T:
T_\beta\text{ is needed for closure and has status }
\mathrm{conjectural}\text{ or }\mathrm{open},
\text{ or has undeclared conditional status}
\right\},
$$

and

$$
\mathfrak O_{\rm compat}
=
\mathcal O_{\partial}
\cup\mathcal O_{\mathcal F}
\cup\mathcal O_{\Sigma}
\cup\mathcal O_{\rm target}
\cup\mathcal O_T.
$$

Here $\mathcal O_{\partial}$ records boundary and terminal-entry defects from
`Bound_partial`, $\mathcal O_{\mathcal F}$ records unresolved source defects
from `Bound_B`, $\mathcal O_{\Sigma}$ records missing-object, lineage, gauge,
or constants defects from `Bound_Sigma`, $\mathcal O_{\rm target}$ records a
target declaration, finite-entry, or global-to-local bridge gap, and
$\mathcal O_T$ records endpoint theorem gaps such as a theorem with status
$\mathrm{conjectural}$ or $\mathrm{open}$ being needed for closure.

Thus the actual target-closure condition is

$$
\mathfrak O_{\rm final}
=
\mathfrak O_{\rm branch}
\cup\mathfrak O_{\rm compat}
=\emptyset.
$$

A branch marked $\mathrm{proved\ nonattainable}$ closes only if `PS33` proved
nonattainability from necessary conditions. The statuses
$\mathrm{formal\ only}$, $\mathrm{undecided}$, and $\mathrm{unresolved}$ do not
close a target regularity theorem. Purely formal profiles with no closed
provenance decision remain in $\mathcal O_{\rm att}$. The condition
$\mathcal O_{\rm ST}=\emptyset$ says every local state-space subtheorem needed
for `ST20` has closed. The condition
$\mathcal O_{\rm src}=\emptyset$ says that every nonattainability label has an
actual `PS33` necessary-condition proof. The condition
$\mathfrak O_{\rm compat}=\emptyset$ says that no boundary, source, data,
target-entry, or endpoint-theorem gap is still needed for the final
implication.

**Proof.** Lemma GC_T.3 assigns a singular-entry packet to some
$B\in\mathcal S_{\rm loc}$. Because the packet comes from the original
solution, the assigned branch enters the provenance-audited universe unless it
is only a formal candidate whose attainability is still undecided; that latter
case is recorded in $\mathcal O_{\rm att}$. If the record only says that an
audited $B$ belongs to a named class, then the packet is classified but not
contradicted. If the realized branch $B$ has status $\mathrm{excluded}$,
`PS32` supplies an endpoint contradiction for that same branch. If a proposed
branch alternative has status $\mathrm{proved\ nonattainable}$, `PS33`
supplies a necessary-condition failure showing that no admissible sequence
from the singular packet can realize that alternative. These are the only
recorded statuses that negate, respectively, a realized branch or a candidate
alternative. Therefore the final target argument needs every realized branch
to be excluded, every nonattainability label to be backed by its `PS33`
source, $\mathcal O_{\rm ST}=\emptyset$ so no local state-space subtheorem is
still open, and $\mathcal O_{\rm att}=\emptyset$ so no unsettled candidate can
later enter the audited universe. The compatibility nodes must also have
empty obstruction ledgers. Equivalently, the single terminal obstruction set
$\mathfrak O_{\rm final}$ is empty: no open audited branch, no open ST
obligation, no unsourced nonattainability claim, no open attainability
obligation, and no boundary, source, data, target-entry, or endpoint-theorem
gap remains.

**Lemma GC_T.5 -- Excluding realized branches and sourcing nonattainability
excludes the singular point.**
If the target declaration applies to $z_0$, the singular point has admissible
finite entry or a closed finite-entry obstruction, and
$\mathfrak O_{\rm final}=\emptyset$, then the initial assumption
$z_0\in\Sigma$ is impossible. If $\mathfrak O_{\rm final}\ne\emptyset$, the
output is target compatibility incomplete: an open local branch, open ST
obligation, open attainability obligation, unsourced nonattainability label,
or compatibility defect remains.

**Proof.** Assume for contradiction that $z_0\in\Sigma$ and that the target
declaration applies. If finite entry is not available, the hypothesis of the
lemma says that the finite-entry obstruction has already been closed by a
separate argument; otherwise Lemma GC_T.1 supplies a concentration packet.
Lemma GC_T.3 then assigns that packet to an admissible branch

$$
B\in
\mathfrak L\cap\mathcal S_{\rm loc}
\subset
\mathcal U_{\rm named}\cup\mathcal R_{\rm named}.
$$

The verified branch-status record assigns each element of
$\mathfrak L$ one of the closed statuses
$\mathrm{excluded}$ or
$\mathrm{proved\ nonattainable}$ exactly when
$\mathcal R_{\rm open}=\emptyset$ and $\mathcal O_{\rm src}=\emptyset$. The
condition $\mathcal O_{\rm ST}=\emptyset$ ensures that residual closure has no
open local state-space subtheorem. The condition
$\mathcal O_{\rm att}=\emptyset$ ensures that no
undecided candidate is still available as the branch assigned to the packet.
Together these four checks give $\mathfrak O_{\rm branch}=\emptyset$. The
compatibility audits from `Bound_partial`, `Bound_B`, `Bound_Sigma`, `GC_T`,
and `PS31` give $\mathfrak O_{\rm compat}=\emptyset$. These two statements
are exactly $\mathfrak O_{\rm final}=\emptyset$.
If the assigned branch is excluded, the `PS32` record applies to the branch
generated from the packet and gives a direct endpoint, prelimit, or rigidity
contradiction. If a proposed branch alternative is proved nonattainable, the
`PS33` provenance record says that no admissible sequence with the recorded
necessary conditions can produce it; such an alternative cannot be the branch
generated by the sequence just constructed from $z_0$. Each remaining possible
assignment is therefore contradicted or removed. If an open branch, ST
obligation, attainability obligation, unsourced nonattainability label,
boundary defect, source defect, missing object, target-entry gap, or endpoint
theorem gap remains, the proof must report target compatibility incomplete
instead of concluding regularity.

### Specific Estimate

The singular-entry implication is

$$
z_0\in\Sigma
\Longrightarrow
C(u;z_0,r)+D(p;z_0,r)\ge\varepsilon_0
$$

for every sufficiently small entry-admissible radius $r$. The decisive closing
condition is

$$
\mathfrak O_{\rm final}
=
\mathfrak O_{\rm branch}\cup\mathfrak O_{\rm compat}
=\emptyset,
$$

equivalently

$$
\mathcal R_{\rm open}=\emptyset,\qquad
\mathcal O_{\rm ST}=\emptyset,\qquad
\mathcal O_{\rm att}=\emptyset,\qquad
\mathcal O_{\rm src}=\emptyset,\qquad
\mathfrak O_{\rm compat}=\emptyset.
$$

### Practical Verification Steps

1. Declare $\mathcal T_{\rm target}$.
2. Check that the target point lies in the declared target class.
3. Verify finite entry, or close/list the finite-entry obstruction.
4. Produce the singular-entry concentration packet.
5. Run the completed local branch decomposition.
6. Check that every realized admissible branch is $\mathrm{excluded}$ and
   every proved nonattainability label has a `PS33` source.
7. Check that $\mathcal O_{\rm ST}=\emptyset$ for residual components closed
   through `ST20`.
8. Check boundary, source, data, target-entry, and endpoint theorem ledgers
   and build $\mathfrak O_{\rm compat}$.
9. Build
   $\mathfrak O_{\rm final}
   =\mathfrak O_{\rm branch}\cup\mathfrak O_{\rm compat}$.
10. Conclude target regularity only if $\mathfrak O_{\rm final}=\emptyset$.
11. If this check fails, output the open branch, ST obligation, attainability
    obligation, unsourced nonattainability label, or compatibility defect list
    rather than a contradiction.

## Estimate Step $B_{\mathrm{GC\_T}}$

The estimate step is the finite-entry CKN singular-entry implication plus the
negative branch-record check.

## Failure Case

Failure names: target-compatibility gap; finite-entry obstruction; open local
branch.

Analytic meaning: the local decomposition does not cover every singular point
needed for the target regularity statement, the point does not enter the
finite-entry profile machinery, or coverage exists but exclusion completeness
fails.

## Refinement Step

Allowed refinements:

1. strengthen singular-entry construction;
2. prove a finite-entry bridge;
3. add missing local branch predicates;
4. refine the target regularity statement;
5. add a global-to-local selection principle if the target is global;
6. return to `PS35` or `PS33` for open branch closure or nonattainability
   sourcing.

Progress measure: every target singular point either enters the completed
finite-entry local decomposition and has no open branch, or is reported as a
named finite-entry or target-compatibility obstruction.

## Data Passed Forward

The next proof step is `FinalExcl`. The data passed forward are

$$
\Gamma_{\mathrm{GC\_T}}
=
\Gamma_{\mathrm{Bound\_\Sigma}}
\cup
\left\{
\begin{array}{l}
\mathcal T_{\rm target}\text{ declared},\\
\text{singular point has admissible finite entry or finite-entry obstruction is closed},\\
\text{singular point generates a }PS0\text{ concentration packet},\\
\mathfrak O_{\rm final}=\emptyset,\\
\mathcal R_{\rm open}=\emptyset,\\
\mathcal O_{\rm ST}=\emptyset,\\
\mathcal O_{\rm att}=\emptyset,\\
\mathcal O_{\rm src}=\emptyset,\\
\mathfrak O_{\rm compat}=\emptyset,\\
\text{therefore target singular point impossible}
\end{array}
\right\}.
$$

If $\mathfrak O_{\rm final}\ne\emptyset$, equivalently if
$\mathcal R_{\rm open}\ne\emptyset$, $\mathcal O_{\rm ST}\ne\emptyset$,
$\mathcal O_{\rm att}\ne\emptyset$, or $\mathcal O_{\rm src}\ne\emptyset$, or
$\mathfrak O_{\rm compat}\ne\emptyset$, `GC_T` instead passes forward

$$
\text{target compatibility incomplete: open branch, ST, attainability, source, boundary, data, target, or endpoint obligations remain}.
$$

---

# 67. `FinalExcl` -- Final Local Singularity Exclusion Record

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is the final branch universe generated from the original
singular-entry packet by admissible operations, together with all exclusion,
nonattainability, endpoint-theorem, and open-obligation records.

### Standing Assumptions

The incoming record states that the local-to-target implication has been
verified in `GC_T` only under a negative branch record. Naming or classifying
all branches is weaker than excluding them. The terminal block outputs exactly
one of the following:

$$
\boxed{
\begin{array}{c}
\text{Every admissible local singular branch is excluded or proved}\\
\text{nonattainable, with all `PS33' sources recorded,}\\
\text{hence the target point is regular.}
\end{array}
}
$$

or

$$
\boxed{
\begin{array}{c}
\text{The proof is complete up to the following explicit open branch}\\
\text{obligations.}
\end{array}
}
$$

### Objects Inspected

Inspect every branch status, endpoint theorem application, residual class, and
realization decision; the `ST20` residual closure record; the `PS32`
contradiction record for every excluded branch; the `PS33` provenance record
for every proved nonattainable branch; and the status of every endpoint
theorem used.

### Dependencies Used

All preceding estimates contribute to the final branch status. Endpoint
contradiction records come from `PS32`; nonattainability provenance comes
from `PS33`; generic residual closure comes from `ST20`; branch coverage and
residual bookkeeping come from `PS35`; target compatibility comes from
`GC_T`.

### Local Obstruction Predicate

$P_{\mathrm{FinalExcl}}$ holds if some branch has status
$\mathrm{realized\ nonexcluded}$, $\mathrm{undecided}$,
$\mathrm{unresolved}$, $\mathrm{residual\ not\ closed\ by\ }ST20$, or
$\mathrm{ST\ obligation\ open}$, or
$\mathrm{formal\ but\ possibly\ attainable}$, or if an allegedly closed
branch lacks its `PS32` or `PS33` record, or if
$\mathcal O_{\rm ST}\ne\emptyset$, $\mathcal O_{\rm att}\ne\emptyset$,
$\mathcal O_{\rm src}\ne\emptyset$, or $\mathfrak O_{\rm compat}\ne\emptyset$.

### Local Lemmas to Prove

**Lemma FinalExcl.1 -- The final branch universe has audited provenance.**
Let $\mathfrak L$ be the set of all branch alternatives assigned to the
singular-entry packet before final status evaluation:

$$
B\in\mathfrak L
\quad\Longleftrightarrow\quad
B\text{ has realized, claimed, or still-unsettled provenance from the original singular-entry sequence}.
$$

The unsourced nonattainability set is

$$
\mathcal O_{\rm src}
=
\left\{
B\in\mathcal N_{\rm att}^{0}:
\text{the `PS33' source or failed necessary condition is missing}
\right\}.
$$

The endpoint theorem gap set is

$$
\mathcal O_T
=
\left\{
T_\beta\in\mathscr T:
T_\beta\text{ is needed for closure and has status }
\mathrm{conjectural}\text{ or }\mathrm{open},
\text{ or has undeclared conditional status}
\right\}.
$$

The compatibility-obligation set is

$$
\mathfrak O_{\rm compat}
=
\mathcal O_{\partial}
\cup\mathcal O_{\mathcal F}
\cup\mathcal O_{\Sigma}
\cup\mathcal O_{\rm target}
\cup\mathcal O_T,
$$

where the five terms record, respectively, boundary/entry defects, unresolved
source terms, missing proof objects or constants, target finite-entry or
global-to-local bridge gaps, and endpoint theorem status gaps.

The open part of the audited branch universe is

$$
\mathfrak L_{\rm open}
=
\left\{
B\in\mathfrak L:
\mathrm{status}(B)\notin
\{\mathrm{excluded},\mathrm{proved\ nonattainable}\}
\right\}.
$$

Purely formal profiles are never counted as closed merely because they are
named. They are either outside $\mathfrak L$ and listed in
$\mathcal O_{\rm att}$, or included in $\mathfrak L$ only as open
formal-but-possibly-attainable obligations unless `PS33` proves that they are
realized by an admissible sequence or proves that they are nonattainable.

**Proof.** `PS35` gives coverage of the realized local universe

$$
\mathcal S_{\rm loc}
=
\mathcal U_{\rm named}\cup\mathcal R_{\rm named},
$$

and separately records formal, undecided, and proved-nonattainable candidates
in $\mathcal O_{\rm att}$ and $\mathcal N_{\rm att}^{0}$. `FinalExcl`
assembles these records into the final audit universe $\mathfrak L$ so that no
candidate is silently counted as either excluded or irrelevant. It keeps a
branch $B$ in $\mathfrak L$ exactly when the record supplies either the chain

$$
\text{singular entry packet}
\longrightarrow
\text{admissible operations}
\longrightarrow
B.
$$

or an explicit unresolved or failed edge in that chain. If the record proves
that a purely formal profile cannot have such a chain, it is retained only as a
closed $\mathrm{proved\ nonattainable}$ record. If the record does not decide
whether such a chain exists, the profile is listed in $\mathcal O_{\rm att}$
or, if it has already been tied to the branch universe, as a
$\mathrm{formal\ but\ possibly\ attainable}$ open record. Thus
$\mathfrak L$ contains exactly the provenance-audited branch alternatives that
the hypothetical singular point could generate unless a closing record negates
them. The realized/admissible component of $\mathfrak L$ agrees with
$\mathcal S_{\rm loc}$, so

$$
\mathcal R_{\rm open}=\emptyset
$$

closes the realized-branch component. The full terminal open-obligation set is

$$
\mathfrak O_{\rm final}
=
\mathfrak L_{\rm open}\cup\mathcal O_{\rm ST}\cup\mathcal O_{\rm att}
\cup\mathcal O_{\rm src}\cup\mathfrak O_{\rm compat}.
$$

Thus the final theorem closes only when $\mathfrak O_{\rm final}=\emptyset$,
not merely when the realized-branch complement $\mathcal R_{\rm open}$ is
empty.

**Lemma FinalExcl.2 -- Closed and open statuses are disjoint.**
Define

$$
\mathrm{Closed}
=
\{\mathrm{excluded},\mathrm{proved\ nonattainable}\},
$$

and

$$
\mathrm{Open}
=
\{
\mathrm{realized\ nonexcluded},
\mathrm{undecided},
\mathrm{unresolved},
\mathrm{residual\ not\ closed\ by\ }ST20,
\mathrm{ST\ obligation\ open},
\mathrm{formal\ but\ possibly\ attainable}
\}.
$$

The final exclusion condition is

$$
\forall B\in\mathfrak L,\qquad
\mathrm{status}(B)\in\mathrm{Closed},
\qquad
\mathcal O_{\rm att}=\emptyset,
\qquad
\mathcal O_{\rm ST}=\emptyset,
\qquad
\mathcal O_{\rm src}=\emptyset,
\qquad
\mathfrak O_{\rm compat}=\emptyset,
$$

equivalently

$$
\mathfrak O_{\rm final}
=
\mathfrak L_{\rm open}\cup\mathcal O_{\rm ST}\cup\mathcal O_{\rm att}
\cup\mathcal O_{\rm src}\cup\mathfrak O_{\rm compat}
=
\emptyset.
$$

**Proof.** A branch status is closed only when it negates the branch as a
possible singular mechanism. The status $\mathrm{excluded}$ negates the branch
by an endpoint contradiction. The status $\mathrm{proved\ nonattainable}$
negates it by a provenance contradiction. Each open status fails to provide
one of these negations: a realized nonexcluded branch is still present,
undecided and unresolved branches lack a decision, residual branches not
matched by `ST20` have not been reduced to closed subclasses, and
formal-but-possibly-attainable branches have not been ruled out by `PS33`.
The generic local residual branch is closed only when it carries the
`ST20` status

$$
\mathrm{status}(\mathcal R_{\rm loc})=\mathrm{excluded}.
$$

Therefore the set
$\mathfrak L_{\rm open}$ is empty exactly when every branch generated by the
singular-entry packet has a closing record. The additional condition
$\mathcal O_{\rm ST}=\emptyset$ says there is no residual branch whose local
state-space subtheorems are still open. The condition
$\mathcal O_{\rm att}=\emptyset$ says there is no unresolved candidate that
might still become generated by that packet. The condition
$\mathcal O_{\rm src}=\emptyset$ ensures that every nonattainability item
counted as closed has the required `PS33` proof. The condition
$\mathfrak O_{\rm compat}=\emptyset$ ensures that no boundary, source,
missing-object, target-entry, or endpoint theorem gap remains outside the
branch-status ledger. With these conventions, the terminal closure condition is
$\mathfrak O_{\rm final}=\emptyset$.

**Lemma FinalExcl.3 -- Excluded branches carry contradiction type.**
Every branch with status $\mathrm{excluded}$ carries a `PS32` contradiction
record of one of the following types:

$$
\mathrm{Type\ A}:\quad
\text{direct physical regularity};
$$

$$
\mathrm{Type\ B}:\quad
\text{quantitative prelimit contradiction};
$$

$$
\mathrm{Type\ C}:\quad
\text{zero/rigidity contradiction with active attainment}.
$$

**Proof.** The label $\mathrm{excluded}$ is not a proof by itself. In Type A,
the endpoint theorem gives regularity of the original physical target or of a
pullback cylinder, contradicting the singular-entry assumption. In Type B, the
endpoint theorem yields a quantitative smallness or decay estimate that, when
pulled back through the verified frame and constants ledger, contradicts the
positive prelimit concentration. In Type C, the endpoint theorem or rigidity
statement forces a zero or rigid profile while the attainment record says that
the branch carries active nonzero mass. These three mechanisms cover the
allowed `PS32` contradiction routes. For the generic residual branch, the
`ST20` record supplies the local Type C zero/activity contradiction through
`ST17`--`ST19`; `PS31` registers it as the residual closure theorem and
`PS32` records the matching contradiction. Recording the type identifies the
exact logical contradiction used to close the branch.

**Lemma FinalExcl.4 -- Proved nonattainability carries provenance.**
Every branch with status $\mathrm{proved\ nonattainable}$ carries a
provenance record $\mathcal P_{\rm prov}$ from `PS33`, including the
exact necessary condition that fails, such as pressure-gauge incompatibility,
impossible boundary geometry, an inadmissible defect vector, or absence of any
admissible subsequence realizing the profile.

**Proof.** To close a branch by nonattainability, `PS33` must prove that every
admissible sequence realizing that branch would satisfy a necessary condition
$N(B)$. The provenance record then records that the candidate branch
violates $N(B)$. For instance, the pressure gauge may be incompatible with the
endpoint pressure class, the boundary ratio may have no admissible geometric
limit, the defect vector may violate a required sign or vanishing condition, or
no common subsequence may realize all selected objects. The contradiction is
between admissible provenance and the failure of $N(B)$; without this recorded
necessary condition, nonattainability has not been proved.

**Lemma FinalExcl.5 -- Endpoint theorem status controls final theorem status.**
`FinalExcl` is unconditional only if every endpoint theorem used has status
$\mathrm{established}$ or $\mathrm{proved\ earlier}$. If an endpoint theorem
used has status $\mathrm{conditional}$, the final conclusion is conditional on
that theorem. If an endpoint theorem has status $\mathrm{conjectural}$ or
$\mathrm{open}$, it cannot be used to close the final exclusion.

**Proof.** Each excluded branch depends on at least one endpoint theorem in the
registry. If all such theorems are established or proved earlier in the same
development, the branch contradictions are unconditional. If a branch uses a
conditional theorem, the implication closing that branch is conditional, and
the conjunction closing all branches is conditional on that theorem. If a
branch uses a conjectural or open theorem, the branch has not been closed by a
proved implication at all, so it must be moved to the open-obligation list
rather than counted as excluded. The final theorem status is therefore the
maximum logical weakness among the endpoint theorems actually used.

**Lemma FinalExcl.6 -- Complete negative record excludes local singularity.**
If $\mathfrak O_{\rm final}=\emptyset$, and every closed branch has the
required `PS32` or `PS33` record, then no admissible local singular
profile remains.

**Proof.** Let $B\in\mathfrak L$ be arbitrary. Since
$\mathfrak O_{\rm final}=\emptyset$, Lemma FinalExcl.2 gives
$\mathrm{status}(B)\in\mathrm{Closed}$. If
$\mathrm{status}(B)=\mathrm{excluded}$, Lemma FinalExcl.3 supplies the
`PS32` contradiction type and hence an explicit contradiction for that branch.
If $\mathrm{status}(B)=\mathrm{proved\ nonattainable}$, Lemma FinalExcl.4
supplies the failed necessary condition from `PS33`, so $B$ cannot lie in the
realized component of $\mathfrak L$ generated by the original sequence. If the
singular-entry packet were assigned to that alternative, the assignment would
contradict the `PS33` provenance record. Thus no arbitrary $B\in\mathfrak L$
can remain as a possible local singular profile.

**Lemma FinalExcl.7 -- Local exclusion implies regularity at the target point.**
If no admissible local singular profile remains and `GC_T` has verified the
target declaration and finite-entry bridge, then the target point is regular.

**Proof.** `GC_T` proves the target bridge

$$
\{\mathfrak O_{\rm final}=\emptyset\}
\Longrightarrow
z_0\notin\Sigma.
$$

Lemma FinalExcl.6 supplies the hypothesis on the left-hand side by closing
every branch in the admissible universe generated by the singular-entry
packet. Hence the hypothetical target singular point cannot exist. In the
language of the target theorem, $z_0\notin\Sigma$ means the target point is
regular in the declared class. The conclusion is unconditional or conditional
according to Lemma FinalExcl.5.

**Lemma FinalExcl.8 -- Open branches are exact obligations.**
For every branch with status $\mathrm{realized\ nonexcluded}$,
$\mathrm{undecided}$, $\mathrm{unresolved}$,
$\mathrm{residual\ not\ closed\ by\ }ST20$,
$\mathrm{ST\ obligation\ open}$, or
$\mathrm{formal\ but\ possibly\ attainable}$, the final conclusion is that
branch and its named theorem gap, local ST subtheorem gap, construction gap,
finite-entry gap, boundary defect, source defect, missing-object defect,
endpoint theorem gap, or exclusion-estimate gap.

**Proof.** By Lemma FinalExcl.1, every branch in $\mathfrak L$ has realized,
claimed, failed, or still-unsettled provenance from the singular-entry packet.
By Lemma FinalExcl.2, a branch is open exactly when its status is not one of
the two closed statuses. The branch record also carries the reason it failed
to close: missing endpoint theorem, missing construction, unresolved source,
boundary defect, missing finite entry, unresolved residual, or failed
constants/object verification. Therefore
$\mathfrak L_{\rm open}\ne\emptyset$ is not a vague failure of the proof; it is
an explicit finite or declared countable list of remaining PDE obligations.
The same applies to every element of $\mathcal O_{\rm ST}$: it is recorded
with the residual branch, the missing ST subtheorem, and the witness produced
by the state-space stratification.
The same applies to every element of $\mathcal O_{\rm att}$: it is recorded
with the candidate profile, missing provenance edge, and the theorem or
construction needed to decide attainability. Every element of
$\mathcal O_{\rm src}$ is recorded with the candidate branch and the missing
`PS33` source or failed necessary condition needed before the branch may be
counted as proved nonattainable. Every element of
$\mathfrak O_{\rm compat}$ is recorded with its source ledger:
$\mathcal O_{\partial}$ for boundary or terminal-entry defects,
$\mathcal O_{\mathcal F}$ for unresolved equation sources,
$\mathcal O_{\Sigma}$ for missing objects, lineage, gauges, or constants,
$\mathcal O_{\rm target}$ for target-scope or global-to-local gaps, and
$\mathcal O_T$ for endpoint theorem status gaps.

**Lemma FinalExcl.9 -- The generic residual row is closed by `ST20`.**
If the final branch report contains the generic local residual class
$\mathcal R_{\rm loc}$ with the local hypotheses registered in `PS31` and
$\mathrm{Obl}_{\mathrm{ST}}=\emptyset$, then
its final status is

$$
\mathrm{status}(\mathcal R_{\rm loc})
=
\mathrm{excluded}
\quad\text{with source }ST20.
$$

It is not listed in $\mathfrak L_{\rm open}$.

**Proof.** `ST20` proves that no bounded centered residual profile in
$\mathfrak X_M$ with local suitability, local pressure gauges, retained
compact activity, and outside-routed-lower-ledger status can exist. `PS31` verifies
that this theorem has local hypotheses only and that the ST obligation ledger
is empty. Therefore the generic residual row has a closing theorem source.
If $\mathrm{Obl}_{\mathrm{ST}}\ne\emptyset$, the row is instead recorded in
$\mathcal O_{\rm ST}$. Listing a closed residual row as merely
$\mathrm{residual}$ would discard the `ST20` contradiction and would re-open
the global-estimate gap that the state-space block was inserted to close.

### Specific Estimate

The decisive final condition is

$$
\mathfrak O_{\rm final}
=
\mathfrak L_{\rm open}\cup\mathcal O_{\rm ST}\cup\mathcal O_{\rm att}
\cup\mathcal O_{\rm src}\cup\mathfrak O_{\rm compat}
=\emptyset,
\qquad\text{equivalently}\qquad
\forall B\in\mathfrak L,\quad
\mathrm{status}(B)\in
\{\mathrm{excluded},\mathrm{proved\ nonattainable}\}
\quad\text{with }
\mathrm{status}(\mathcal R_{\rm loc})=\mathrm{excluded}
\text{ by }ST20\text{ when present}
\quad\text{and}\quad
\mathcal O_{\rm ST}=\emptyset
\quad\text{and}\quad
\mathcal O_{\rm att}=\emptyset
\quad\text{and}\quad
\mathcal O_{\rm src}=\emptyset
\quad\text{and}\quad
\mathfrak O_{\rm compat}=\emptyset.
$$

### Practical Verification Steps

1. Build $\mathfrak L$ from branches with admissible provenance from the
   original singular-entry packet.
2. Split any branch class with mixed statuses into uniform-status subclasses.
3. Put every status in $\mathrm{Closed}$ or $\mathrm{Open}$.
4. For every excluded branch, attach the `PS32` contradiction type A, B, or C.
5. For the generic residual row, attach the `ST20` exclusion source, the
   local hypothesis match from `PS31`, and the check
   $\mathrm{Obl}_{\mathrm{ST}}=\emptyset$.
6. For every proved nonattainable branch, attach the `PS33` provenance
   record $\mathcal P_{\rm prov}$ and failed necessary condition.
7. Add every residual component with nonempty $\mathrm{Obl}_{\mathrm{ST}}$ to
   $\mathcal O_{\rm ST}$.
8. Add every formal-only or undecided candidate to $\mathcal O_{\rm att}$ and
   every unsourced nonattainability label to $\mathcal O_{\rm src}$.
9. Build $\mathfrak O_{\rm compat}$ from boundary/entry, source, data,
   target-bridge, and endpoint theorem ledgers.
10. Check endpoint theorem statuses: established, proved earlier, conditional,
   conjectural, or open; put closure-blocking theorem gaps in $\mathcal O_T$.
11. If $\mathfrak O_{\rm final}=\emptyset$, apply `GC_T` and report theorem
   status proved or conditional.
12. If $\mathfrak O_{\rm final}\ne\emptyset$, report the explicit open list and
   mark the final theorem open.

## Estimate Step $B_{\mathrm{FinalExcl}}$

The estimate step is final branch universe assembly, closed/open status
splitting, contradiction/provenance verification, and endpoint theorem status
checking.

## Failure Case

Failure name: remaining local singular branch.

Analytic meaning: a branch in the exhaustive local decomposition remains
attainable and not excluded by the verified endpoint theorems, or an allegedly
closed branch lacks its contradiction or nonattainability record, or a
boundary, source, data, target-entry, or endpoint theorem compatibility
obligation remains, or a local ST obligation remains open.

## Refinement Step

Allowed refinements:

1. prove the missing endpoint theorem;
2. prove nonattainability with `PS33` provenance;
3. refine the residual complement;
4. repair the `PS32` contradiction record;
5. rerun `ST0`--`ST20` if the generic residual row lacks its local closure
   source or has a nonempty ST obligation ledger;
6. rerun endpoint matching and realization checks;
7. downgrade the final theorem to conditional or open when endpoint theorem
   status requires it.

Progress measure: every remaining obstruction is reduced to a named theorem or
construction problem, and every closed branch has either a `PS32`
contradiction record or a `PS33` nonattainability record.

## Data Passed Forward

This is a terminal node. The data passed forward are

$$
\Gamma_{\mathrm{FinalExcl}}
=
\Gamma_{\mathrm{GC\_T}}
\cup
\left\{
\begin{array}{l}
\mathfrak L\text{ final branch universe},\\
\mathfrak O_{\rm final}=\emptyset\text{ or explicit open list},\\
\mathrm{status}(\mathcal R_{\rm loc})=\mathrm{excluded}
\text{ by }ST20\text{ when the generic residual row is present and }
\mathcal O_{\rm ST}=\emptyset,\\
\mathcal O_{\rm ST},\ \mathcal O_{\rm att},\ \mathcal O_{\rm src},\text{ and }
\mathfrak O_{\rm compat}\text{ empty or explicitly listed},\\
\mathcal N_{\rm att}^{0}\text{ sourced whenever counted as closed},\\
\text{for every closed branch: exclusion or nonattainability record},\\
\text{endpoint theorem status report},\\
\text{final theorem status: proved / conditional / open}
\end{array}
\right\}.
$$
