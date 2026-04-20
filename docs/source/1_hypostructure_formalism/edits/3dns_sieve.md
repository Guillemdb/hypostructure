# Three-Dimensional Navier--Stokes Local Blow-Up Analysis

This document records a step-by-step local blow-up analysis for the
three-dimensional incompressible Navier--Stokes equations in standard PDE
terminology. Each proof step is a local lemma or local estimate. The named
steps

$$
H0,\ D_E,\ Rec_N,\ C_\mu,\ PS1,\ldots
$$

are only identifiers for the order of the argument; the mathematical content is
given by the stated estimates, compactness statements, and reductions.

This file treats the pre-entry data, the profile analysis, the compatibility
checks, and the final local exclusion record:

$$
H0,\ D_E,\ Rec_N,\ C_\mu,\ PS1,\ldots,\ PS35,\ Bound\_\partial,\ Bound_B,\ Bound_\Sigma,\ GC_T,\ FinalExcl.
$$

Each named step is a local verification problem. An exclusion is asserted only
where the selected endpoint theorem appears together with the hypotheses
proved at that step. A full NS3D local exclusion is used only when its full
theorem hypotheses appear in the same local branch data.

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
hypothesis, decay hypothesis, or exact endpoint premise. If the branch supplies
only compact-cylinder bounds, then the endpoint theorem is not applicable and
the missing tail or decay statement is recorded as an explicit obligation.

In particular, the following local bounds do not imply any whole-space bound
unless a tail condition is also present:

$$
\sup_{n}\|u_n\|_{L^3(Q_R)}<\infty,
\qquad
\sup_{n}\left(A(u_n;0,R)+E(u_n;0,R)+C(u_n;0,R)+D(p_n;0,R)\right)<\infty,
$$

for fixed $R<\infty$. They give compactness only after localization to smaller
cylinders.

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

Inspect the $L^3$ tail integral outside $B_R$ uniformly for all $\tau$.

### Dependencies Used

The profile comes from `PS9`; nonvanishing from `PS8`; compact hull/tail
limits, when used, come from `PS13`.

### Local Obstruction Predicate

$P_{\mathrm{PS15}}$ holds when uniform $L^3$ tightness is present, because
that branch is excluded by the endpoint ancient $L^3$ theorem.

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

After passing to a subsequence, this witness may be routed to one of the
next exterior mechanisms, but only after the corresponding verification is
proved:
a recentered active exterior core for `PS16`, a finite separated family for
`PS18`--`PS20`, or a diffuse residual tail for `PS34`. This lemma itself
produces only the exterior witness displayed above and the obligation to run
those later tests.

**Proof.** The negation of uniform tightness is the existence of
$\eta_{\rm tail}>0$ such that for every $R$ there is a time $\tau_R$ with

$$
\int_{|y|>R}|V(y,\tau_R)|^3\,dy\ge \eta_{\rm tail}.
$$

Choosing $R_k\uparrow\infty$ and setting $\tau_k=\tau_{R_k}$ gives the
displayed witness. No localization conclusion follows from this tail inequality
alone. Each possible route requires an additional check written in the target
node. If a sequence of compact terminal cylinders inside the exterior region
has a scale-invariant CKN lower bound, then recentering at those cylinders
gives the active exterior input for `PS16`. If the above-threshold cylinders
can be chosen as a finite parabolically separated family within a compact
terminal window with a recorded CKN budget, they give the finite packet input
for `PS18`--`PS20`. If the exterior lower bound persists through expanding
regions while every fixed compact cylinder is below the retained threshold,
then the branch satisfies the diffuse-residual hypothesis used by `PS34`.
If none of these verification conditions has been proved, the conclusion of
`PS15` is only the displayed exterior witness together with a missing
route-data item; it is not a completed classification.

### Specific Estimate

The decisive estimate is the uniform tail bound

$$
\sup_{\tau\in\mathbb R}\int_{|y|>R}|V(y,\tau)|^3\,dy\to0
\qquad (R\to\infty).
$$

### Practical Verification Steps

1. Estimate the spatial $L^3$ tails of $V$ uniformly in $\tau$.
2. Record the induced global $L^\infty_\tau L^3_y$ bound.
3. If tightness holds, verify the physical pullback is bounded mild ancient on
   each finite terminal-time restriction.
4. Use $L^3$ critical-norm invariance on times $t_k\downarrow-\infty$.
5. Apply the endpoint ancient $L^3$ theorem.
6. Compare the zero conclusion with retained activity.
7. If tightness fails, record the exterior witness of Lemma PS15.4. Route it
   only after the active-core, finite-separated-family, or diffuse-residual
   verification required by the target node has been proved.

## Estimate Step $B_{\mathrm{PS15}}$

The estimate step is the tightness-to-endpoint-$L^3$ argument in Lemmas
PS15.1--PS15.3, together with the non-tightness witness extraction in
Lemma PS15.4. Later routes are not used in this estimate unless their
own local lower bounds, finite-budget arguments, or diffuse-residual tests
have been verified.

## Failure Case

Failure name: unresolved tightness branch.

Analytic meaning: the record asserts spatial tightness but lacks the exact tail
estimate, the mild Duhamel gate, or another endpoint $L^3$ input.

## Refinement Step

Allowed refinements:

1. strengthen tail estimates;
2. verify the mild Duhamel gate on finite terminal slabs;
3. pass to hull limits using `PS13`;
4. assign failed tightness to `PS16`.

Progress measure: tail modulus is fixed, or non-tightness is recorded.

## Data Passed Forward

If tightness holds and mildness is verified, `PS15` excludes the branch. If
tightness holds but mildness is not verified, the data passed forward record a
mildness obstruction. If tightness fails, the next proof step is `PS16` only
after the exterior route status has been determined. The data passed forward
are

$$
\Gamma_{\mathrm{PS15}}
=
\Gamma_{\mathrm{PS9}}
\cup
\{\text{endpoint exclusion, or mildness obstruction, or }
\eta_{\rm tail},R_k,\tau_k,
\text{ route status: active exterior / finite separated / diffuse / unresolved}\}.
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
converges to zero in $L^3(B_R)$ when viewed in the $\lambda_n^1$-frame. Constant
ambient velocities are incorporated into the translation modulation.

**Proof.** In the innermost variables the outer contribution has the form

$$
W_n^j(y)=
\frac{\lambda_n^1}{\lambda_n^j}
\phi^j\left(z_n^j+\frac{\lambda_n^1}{\lambda_n^j}y\right).
$$

Apply Lemma PS16.1 with
$\rho_n=\lambda_n^1/\lambda_n^j\to0$. The only possible nondecaying first-order
term is a constant velocity, and Navier--Stokes is invariant under Galilean
translation of the local frame; that term is therefore recorded in the
translation modulation rather than as a radiative component.

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
   source, or far-field tail is not controlled, in which case the branch is a
   named defect routed to `PS30`;
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

The incoming record states that every registered active frame has a fixed positive local critical lower
bound and that the total local critical mass in the selected analysis window is
finite.

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

**Lemma PS18.1 -- Finite active count above threshold.**
Assume the active profile package has finite critical mass

$$
\sum_{j\in J}\|\phi^j\|_{L^3(\mathbb R^3)}^3\le M_*<\infty
$$

and that a profile is active only when

$$
\|\phi^j\|_{L^3(\mathbb R^3)}\ge \eta_*>0.
$$

Then there are at most $M_*\eta_*^{-3}$ active profiles.

**Proof.** For every finite active subfamily $J_0\subset J$,

$$
\#J_0\,\eta_*^3
\le
\sum_{j\in J_0}\|\phi^j\|_{L^3}^3
\le
\sum_{j\in J}\|\phi^j\|_{L^3}^3
\le M_*.
$$

Taking the supremum over finite subfamilies gives
$\#J\le M_*\eta_*^{-3}$. This proof uses only nonnegativity and finite critical
mass; asymptotic orthogonality is needed later for interaction estimates, not
for the count.

**Lemma PS18.2 -- Pairwise active-frame classification.**
After passing to a subsequence, each pair of active frames
$\mathfrak F^i=(x_n^i,\lambda_n^i)$ and
$\mathfrak F^j=(x_n^j,\lambda_n^j)$ falls into exactly one of the following
ordered classes:

1. separated point:

   $$
   \frac{|x_n^i-x_n^j|}{\max(\lambda_n^i,\lambda_n^j)}\to\infty;
   $$

2. same-point comparable scale:

   $$
   \frac{|x_n^i-x_n^j|}{\max(\lambda_n^i,\lambda_n^j)}=O(1),
   \qquad
   0<\liminf_n\frac{\lambda_n^i}{\lambda_n^j}
   \le
   \limsup_n\frac{\lambda_n^i}{\lambda_n^j}<\infty;
   $$

3. same-point cascade:

   $$
   \frac{|x_n^i-x_n^j|}{\max(\lambda_n^i,\lambda_n^j)}=O(1),
   \qquad
   \frac{\lambda_n^i}{\lambda_n^j}\to0
   \quad\text{or}\quad
   \frac{\lambda_n^i}{\lambda_n^j}\to\infty .
   $$

If none of these classes is visible in a selected compact frame, then one of
the two contributions is locally invisible there and has already been handled
by `PS16`.

**Proof.** Because the active set is finite by Lemma PS18.1, pass to one
subsequence on which every scale ratio
$\lambda_n^i/\lambda_n^j$ converges in $[0,\infty]$ and every normalized center
distance

$$
\frac{|x_n^i-x_n^j|}{\max(\lambda_n^i,\lambda_n^j)}
$$

either stays bounded or tends to infinity. If the normalized center distance
tends to infinity, the frames are separated. If it stays bounded and the scale
ratio has a finite positive limit, the pair is same-point comparable-scale. If
it stays bounded and the scale ratio tends to $0$ or $\infty$, the pair is a
same-point cascade. These alternatives exhaust all subsequential limits. When
the selected frame sees neither bounded same-point geometry nor separated
positive mass, the local invisibility estimates of `PS16` remove that
contribution from the selected compact equation.

**Lemma PS18.3 -- Same-point comparable profiles form one compound core.**
If a group has same center and comparable scales, then its finite sum is a
single divergence-free compound profile in a common scale. If the compound
profile is below the perturbative critical threshold, it is removed by local
small-data theory; otherwise it is retained as one active core.

**Proof.** Choose one representative
$(x_n^{j_k},\lambda_n^{j_k})$ for the comparable class. After passing to the
subsequence already fixed in Lemma PS18.2,

$$
\rho_j=\lim_n\frac{\lambda_n^j}{\lambda_n^{j_k}}\in(0,\infty),
\qquad
z_j=\lim_n\frac{x_n^j-x_n^{j_k}}{\lambda_n^{j_k}}\in\mathbb R^3 .
$$

In the representative frame the class is represented by the finite
$L^3_\sigma$ sum

$$
\Phi^k(y)=
\sum_{j\in J_k}\rho_j^{-1}
\phi^j\left(\frac{y-z_j}{\rho_j}\right).
$$

Finiteness of $J_k$ comes from Lemma PS18.1. Each summand is divergence-free,
and divergence commutes with translation and dilation, so $\Phi^k$ is
divergence-free. Critical $L^3$ norms are invariant under the displayed
bounded changes of variables. If $\|\Phi^k\|_{L^3}$ is below the local
small-data threshold, the compound class is perturbative and is removed by
`PS21`; otherwise $\Phi^k$ is retained as a single active compound core.

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
classify every pair by Lemma PS18.2. Comparable same-point equivalence classes
are replaced by compound cores using Lemma PS18.3. A pair with separated
physical points remains a separated-center branch unless its contribution is
locally invisible in the selected compact frame; the invisible case was already
removed or recentered by `PS16`. A same-point pair with scale ratio tending to
$0$ or $\infty$ is precisely a scale cascade. After these reductions, if the
branch is still not a single nondegenerate selected core, the only unassigned
obstruction is nonuniqueness or degeneracy in the center-scale selection of the
selected core. This is recorded as the gauge-degenerate active selection.

### Specific Estimate

The decisive counting estimate is

$$
N_{\eta_*}\eta_*^3
\le
\sum_{j=1}^{N_{\eta_*}}m_j
\le M_*,
$$

where $m_j=\|\phi^j\|_{L^3}^3$ is the critical mass of the $j$-th active
profile.

### Practical Verification Steps

1. Fix an activity threshold $\eta_*$ below the retained lower bound.
2. Count all frames with local CKN mass at least the corresponding threshold,
   equivalently profiles with $m_j\ge\eta_*^3$ in the active package.
3. Classify every pair by center and scale ratios.
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
4. recenter separated positive-mass components.

Progress measure: the unclassified active-frame list is replaced by a finite
classified packet.

## Data Passed Forward

The next proof step is `PS19`. The data passed forward are

$$
\Gamma_{\mathrm{PS18}}
=
\Gamma_{\mathrm{PS17}}
\cup
\{\text{single core, finite packet, cascade, separated centers, or gauge degeneracy}\}.
$$

---

# 25. `PS19` -- Finite Packet Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a finite or countable profile expansion in critical variables,
with each profile divergence-free and each pressure represented modulo a
time-dependent gauge.

### Standing Assumptions

The incoming record states that total retained critical mass is finite and the active-frame partition of
`PS18` has been completed.

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

**Lemma PS19.1 -- Finite packet selection with small summable tail.**
For each $\eta>0$, the set $\mathcal P_\eta$ of profiles with retained mass at
least $\eta$ is finite. Moreover, if

$$
\sum_{j\in J}m_j<\infty,
$$

then for every $\varepsilon>0$ there is a finite packet
$\mathcal P_{\eta,\varepsilon}\supset\mathcal P_\eta$ such that

$$
\sum_{j\notin\mathcal P_{\eta,\varepsilon}}m_j<\varepsilon .
$$

**Proof.** The above-threshold count follows from

$$
\#\mathcal P_\eta\,\eta
\le
\sum_{j\in\mathcal P_\eta}m_j
\le
\sum_{j\in J}m_j<\infty .
$$

For the small tail, enumerate the countable complement of $\mathcal P_\eta$ as
$\{j_1,j_2,\dots\}$. Since the critical masses are summable, choose $N$ so that

$$
\sum_{k>N}m_{j_k}<\varepsilon .
$$

Then

$$
\mathcal P_{\eta,\varepsilon}
=
\mathcal P_\eta\cup\{j_1,\dots,j_N\}
$$

is finite and has the required tail bound.

**Lemma PS19.2 -- The discarded tail is perturbative after finite
truncation.**
Let $r_n$ be the sum of all profiles and remainders not included in
$\mathcal P_{\eta,\varepsilon}$. If the finite truncation has been chosen so
that

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

**Proof.** The finite critical-mass tail bound from Lemma PS19.1 is converted,
using the profile-decomposition remainder estimate in the active package, into
the displayed $L^\infty_I L^3(B_{2R})$ smallness. This conversion is an
explicit input: summability of the numbers $m_j$ alone is not enough unless
the decomposition also supplies unconditional convergence of the profile tail
in the selected local critical topology. If that tail estimate is absent, the
tail is not treated as perturbative. Once the displayed smallness is verified,
Holder's inequality on
the compact cylinder gives

$$
\|r_n\otimes r_n\|_{L^1_I L^{3/2}(B_R)}
\le
|I|\,\|r_n\|_{L^\infty_I L^3(B_R)}^2
\le
|I|\,\varepsilon_*^2.
$$

The local small-data stability theorem then treats the tail as a perturbative
solution on the selected window.

The topology in the displayed smallness is part of the packet data. If the
profile package supplies only $L^3_{I,x}(B_{2R}\times I)$ smallness, the node
may still pass a mixed-stress estimate to `PS20`, but it may not claim the
stronger $L^\infty_I L^3_x$ packet readiness. In that case the packet data
state the weaker topology explicitly, and `PS20` must use the matching
$L^{3/2}_{I,x}$ estimates rather than the $L^\infty_I L^3_x$ shorthand.

**Lemma PS19.3 -- Packet ordering by active geometry.**
After passing to a subsequence, the finite packet is ordered into
same-point comparable groups, same-point scale chains, and separated-center
groups.

**Proof.** Since the packet is finite, apply the pairwise classification of
Lemma PS18.2 to all pairs simultaneously. Make a graph whose vertices are
packet elements and whose edges connect same-point comparable-scale pairs.
Connected components of this graph are compound-core groups. The remaining
edges are classified as either same-point cascade or separated-center.
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
Strict outer-scale components vanish by Lemma PS16.3. The finite-tail
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
4. Assign separated and outer-scale components to $S_n$ in the selected frame.
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
\{\mathcal P_{\eta,\varepsilon},\ \text{packet ordering},\ \text{interaction record}\}.
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
kernel-tail pressure record described above, or an equivalent pressure
payload from the terminal profile decomposition.

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

**Lemma PS20.3 -- Mixed pressure oscillations and forces vanish.**
Let $\pi_n^{\rm mix}$ solve locally

$$
-\Delta\pi_n^{\rm mix}
=
\partial_i\partial_j(\mathcal T_{n,ij})
$$

with spatial means subtracted. Then

$$
\|\pi_n^{\rm mix}-(\pi_n^{\rm mix})_{B_R}\|_{L^1_I L^{3/2}(B_R)}
\to0,
$$

and the pressure force vanishes in the represented-equation dual topology,
namely $L^1_IW^{-1,3/2}(B_R)$; when the compact-cylinder $L^2$ energy upgrade
is invoked, the same term is controlled in the local $H^{-1}$ space used by
the energy formulation.

**Proof.** Localize to $B_{2R}$ and split the pressure into a near-field
Calderon--Zygmund part, an intermediate annular part, and a far-field harmonic
part. For the local part,

$$
\pi_{n,\rm loc}^{\rm mix}
=
R_iR_j(\chi_{2R}\mathcal T_{n,ij}),
$$

where $\chi_{2R}$ is one on $B_{3R/2}$. Calderon--Zygmund boundedness gives

$$
\|\pi_{n,\rm loc}^{\rm mix}\|_{L^1_I L^{3/2}(B_{3R/2})}
\le C
\|\mathcal T_n\|_{L^1_I L^{3/2}(B_{2R})}
\to0
$$

by Lemma PS20.2. For the remaining pressure, fix $A>4R$ and write the source
outside $B_{2R}$ as the sum of the annulus $B_A\setminus B_{2R}$ and the
exterior $\mathbb R^3\setminus B_A$. On the annulus, Lemma PS20.2 applied on
$B_A$ gives

$$
\|\mathcal T_n\|_{L^1_I L^{3/2}(B_A\setminus B_{2R})}\to0
$$

for fixed $A$, so smooth kernel bounds give vanishing harmonic pressure
oscillation on $B_R$ from that annular source.

For the exterior source, subtract the kernel value at the center of $B_R$.
The Calderon--Zygmund kernel difference satisfies

$$
|K(x-y)-K(0-y)|\le C_R |y|^{-4},
\qquad x\in B_R,\ |y|>A .
$$

Holder's inequality gives the uniform tail bound

$$
\int_{|y|>A}|K(x-y)-K(0-y)|\,|\mathcal T_n(y,\tau)|\,dy
\le
C_R A^{-3}
\|\mathcal T_n(\tau)\|_{L^{3/2}(\mathbb R^3\setminus B_A)}
$$

whenever the global or annular $L^{3/2}$ pressure-tail record is present.
After integration in time, the exterior oscillation is bounded by a quantity
that tends to zero as $A\to\infty$, uniformly in $n$. Taking first
$n\to\infty$ for fixed $A$ and then $A\to\infty$ proves the local pressure
oscillation convergence in $L^1_IL^{3/2}(B_R)$.

For a test vector $\varphi\in C_c^\infty(B_R)$,

$$
\langle\nabla\pi_n^{\rm mix},\varphi\rangle
=
-\int_{B_R}
(\pi_n^{\rm mix}-(\pi_n^{\rm mix})_{B_R})
\nabla\cdot\varphi .
$$

This is exactly the $W^{-1,3/2}$ pressure-force topology. In applications that
use the compact-cylinder $H^1$ energy upgrade, the same pressure force is
paired only with the admissible localized test fields from that energy
formulation and is controlled in the corresponding $H^{-1}$ error space.

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
$L^1_I L^{3/2}_{\rm loc}$ together with the far-field tail record, or it
is a separate pressure-defect obligation. In the supplied case, the proof of
Lemma PS20.3 applies with $\mathcal T_n$ replaced by $S_n\otimes S_n$ and
shows that the pressure oscillation generated by the discarded part vanishes
after subtracting spatial means. If this pure discarded-pressure estimate is
missing, the selected positive mass cannot be transferred from $V_n$ to
$U_n$; the branch is routed to `PS30` as a pressure defect.

**Lemma PS20.6 -- The selected limit solves the standalone local equation.**
Assume Lemmas PS20.1--PS20.5 and the local compactness package of `PS17`.
Then every locally convergent subsequence of the selected components has a
limit $(U,P_U)$ satisfying the same local Navier--Stokes equation, local energy
inequality, pressure convention, and retained positive CKN lower bound as the
original selected branch, with no remaining source term from $S_n$.

**Proof.** Write the weak equation for
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
$S_n$ vanish by Lemma PS20.4. The remaining terms pass to the selected limit
by the local compactness and pressure convergence already recorded in `PS17`
and the pressure gauge from `PS4`. Lemma PS20.5 supplies the retained positive
CKN lower bound. Hence the selected limit is not merely a formal component of
the packet; it is an admissible local NS3D branch in its own right.

### Specific Estimate

The decisive estimate is

$$
\|\mathcal T_n\|_{L^1_I L^{3/2}(B_R)}
+
\|\pi_n^{\rm mix}-(\pi_n^{\rm mix})_{B_R}\|_{L^1_I L^{3/2}(B_R)}
\to0.
$$

For the pressure term this shorthand includes the two-scale tail limit

$$
\lim_{A\to\infty}\limsup_n
\int_I
A^{-3}\|\mathcal T_n(\tau)\|_{L^{3/2}(\mathbb R^3\setminus B_A)}
\,d\tau=0,
$$

together with local convergence of $\mathcal T_n$ on each fixed $B_A$. If this
tail statement is missing, only the near-field pressure has been decoupled and
the branch must be routed to the pressure-defect audit `PS30`.

### Practical Verification Steps

1. Decompose $V_n=U_n+S_n$ on $B_{2R}\times I$.
2. Verify $S_n\to0$ in $L^\infty_I L^3(B_{2R})$, or record a direct
   $L^1_I L^{3/2}$ mixed-stress convergence estimate from `PS19`.
3. Estimate all mixed stresses by Holder's inequality or by the direct
   mixed-stress record.
4. Apply local pressure reconstruction to mixed sources, including the
   far-field kernel-tail estimate record.
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
\{\text{terminal decoupling and selected positive branch}\}.
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
\text{not a Type II single core}
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
There is $\varepsilon_\infty>0$ such that every bounded mild ancient solution
of the centered Navier--Stokes equation on
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

The kernel is integrable for $s>0$. Applying the mild formula on
$[\tau-T,\tau]$ gives

$$
\|V(\tau)\|_\infty
\le e^{-T/2}M+A M^2,
\qquad
M=\|V\|_{L^\infty}.
$$

Letting $T\to\infty$ gives $M\le AM^2$. If
$M<1/A$, then $M=0$.

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
$L^2(Q_\rho)$. The compactness package gives a uniform
$L^{10/3}(Q_\rho)$ bound. Interpolating between strong $L^2$ convergence and
the uniform $L^{10/3}$ bound yields strong $L^3_{\rm loc}$ convergence of
$V_n-c_n(s)$ to zero.

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
back to the singular-entry variables and contradicts retained singular
activity. If any theorem hypothesis is missing, the conclusion of this lemma is
the missing hypothesis, not a closure claim.

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
these data with the selected controlled-swirl theorem. Only after that match
does `PS32` use the theorem conclusion, such as regularity, vanishing, or a
constant ancient state, to contradict retained singular activity. Without the
exact exponent, axis condition, pressure convention, and solution class, the
controlled-swirl theorem is not invoked.

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
Q'(\tau)=\Omega Q(\tau),
$$

with $Q(\tau)\in SO(3)$, $\Omega^T=-\Omega$, and $\Omega$ constant, then,
after setting $z=Q(\tau)^Ty$, the profile $W$ satisfies

$$
-\Delta W+\frac12z\cdot\nabla W+\frac12W
+(W\cdot\nabla)W
+\Omega W-(\Omega z)\cdot\nabla W+\nabla\Pi=0,
\qquad
\nabla\cdot W=0.
$$

**Proof.** Differentiate $z=Q(\tau)^Ty$ to get $\partial_\tau z=-\Omega z$.
The Laplacian, divergence, convection term, and pressure gradient are invariant
under the orthogonal map $Q(\tau)$. Substituting into the centered
Navier--Stokes equation and multiplying by $Q(\tau)^T$ gives the displayed
equation.

If $\Omega=\Omega(\tau)$ is not constant, the co-rotating frame contains
time-dependent modulation terms and the profile is not stationary in this
sense. That case is routed to modulation/coherent-defect analysis with the
extra terms retained.

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
co-moving stationary equation. If its localized annular flux satisfies

$$
\left|\iint_{A_R\times I} F_{\rm coh}\cdot W\,\chi_R\right|
\ge \eta
$$

on a fixed annulus $A_R$, then the branch contains a retained local
concentration component and is assigned to active-frame analysis rather than a
coherent-structure endpoint route.

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
cover, cutoff, and coefficient bounds. Therefore, if the flux is bounded below
by $\eta>C_{\rm cover}\varepsilon_{\rm CKN}$, at least one cylinder in the
cover has non-small CKN quantity. Rescaling that cylinder gives an active local
concentration frame. If any pressure, cutoff, or modulation term in the
localized energy identity is uncontrolled, the conclusion is not active-frame
routing; the uncontrolled term is entered into the `PS30` defect vector.

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
2. absorb constant drifts by Galilean normalization;
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
stationary family, the tangent vector $Z$, the linearized operator, true
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

**Lemma PS25.1 -- A stationary family gives a linearized stationary solution.**
Assume $a\mapsto W_a$ is $C^1$ in a topology strong enough to differentiate
the stationary centered equation, for example $C^2_{\rm loc}$, and that
$a\mapsto\Pi_a$ is differentiable modulo constants. If
$a\mapsto(W_a,\Pi_a)$ is a stationary family, then
$Z=\partial_aW_a|_{a=0}$ satisfies

$$
-\Delta Z+\frac12y\cdot\nabla Z+\frac12Z
+(Z\cdot\nabla)W+(W\cdot\nabla)Z+\nabla\pi=0,
\qquad
\nabla\cdot Z=0.
$$

**Proof.** The topology assumption permits differentiating the Laplacian,
centered drift, pressure gradient, and product $(W_a\cdot\nabla)W_a$ locally.
The pressure-gauge assumption gives a derivative
$\pi=\partial_a\Pi_a|_{a=0}$ modulo constants. Differentiating the stationary
centered equation at $a=0$ gives the displayed linearized equation, and
differentiating $\nabla\cdot W_a=0$ gives $\nabla\cdot Z=0$.

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
If an ancient trajectory satisfies $V(\tau)=W_{a(\tau)}$ for a stationary
family whose parametrization is an immersion modulo pressure gauges and true
velocity symmetries, then $a'(\tau)=0$ after quotienting those true symmetries,
and $V$ is stationary in the quotient.

**Proof.** Substitute $V(\tau)=W_{a(\tau)}$ into the centered equation. Since
each $W_a$ solves the stationary equation, the only remaining term is

$$
a'(\tau)\partial_aW_{a(\tau)}.
$$

The immersion hypothesis says that this tangent vanishes only when
$a'(\tau)=0$, after quotienting pressure gauges and true velocity symmetries.
If the tangent is a rotation generator, the branch is a relative equilibrium
handled by `PS24`. If it is a frame/modulation direction, the equation is not
the exact stationary equation and is routed to `PS26`.

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

The decisive verification is the linearized stationary equation for $Z$ and
the classification

$$
Z\in
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
\text{ linearized stationary equation for }Z,
\text{ classification of }Z:
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
convergence of $u_n\otimes u_n$, pressure oscillation convergence, cutoff
commutators, high-frequency projections, and modulation coefficients.

### Dependencies Used

Measure defects use `C_mu`, `PS1`, and `PS16`; stress defects use `PS6`,
`PS19`, and `PS20`; pressure defects use `PS4`; cutoff defects use `PS17` and
`PS20`; frequency and scale defects use `PS11` and `PS28`; rigidity defects use
`PS29`.

### Local Obstruction Predicate

$P_{\mathrm{PS30}}$ holds when some meaningful defect channel has no conclusion
compatible with the endpoint theorem hypotheses.

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

**Lemma PS30.4 -- Cutoff and artificial-boundary defects vanish under compact
control.**
If the localized fields are bounded in $L^\infty_tL^3_x$ and their mixed
stresses vanish in $L^1_tL^{3/2}_x$, then all commutators generated by fixed
cutoffs vanish in $L^1_tH^{-1}_x$ whenever at least one factor is a vanishing
component.

**Proof.** A cutoff commutator has the form

$$
(\nabla\chi)\cdot(u_n\otimes w_n),\qquad
(\Delta\chi)w_n,\qquad
(\nabla\chi)(P_n-\bar P_n),
$$

or a finite sum of such terms, with $\chi\in C_c^\infty$. The coefficients
$\nabla\chi$ and $\Delta\chi$ are bounded on the compact cylinder. The
recorded convergence of $w_n$, mixed stresses, or pressure sources to zero in
the stated spaces gives convergence of each displayed commutator to zero in
$L^1_tH^{-1}_x$ on compact balls.

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
5. Check cutoff commutators on every compact cylinder used later.
6. Check frequency tails or assign a new scale.
7. Check modulation coefficients and assign parameter failures.
8. Record unknown residues only if they are explicit distributions.

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
5. name an explicit residual distribution and assign it to `PS34`.

Progress measure: each refinement fills one previously blank defect-vector
entry.

## Data Passed Forward

The next proof step is `PS31`. The data passed forward are

$$
\Gamma_{\mathrm{PS30}}
=
\Gamma_{\mathrm{PS29}}
\cup
\{\mathbf d\text{ complete}\}.
$$

---

# 37. `PS31` -- Endpoint Hypothesis Verification

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is the selected endpoint candidate: CKN regularity, Serrin
regularity, small-data theory, stationary $L^3$ Liouville, endpoint ancient
$L^3$ Liouville, structured Liouville, Type II local exclusion, or residual
closure.

### Standing Assumptions

The incoming record states that every defect channel has a conclusion and no
unnamed distributional term appears in the limit equation. If a channel is
marked unresolved, the endpoint theorem must either list that unresolved
object as an explicit hypothesis or the branch is blocked at this node.

### Objects Inspected

Inspect solution class, pressure normalization, domain, boundary condition,
critical norm, compactness topology, decay or tightness, symmetry hypotheses,
and positivity normalization.

### Dependencies Used

The earlier estimates feed `PS31`; `PS30` is decisive because endpoint
hypotheses cannot be checked while the defect vector is incomplete.

### Local Obstruction Predicate

$P_{\mathrm{PS31}}$ holds when an endpoint theorem is being invoked with a
hypothesis not actually proved in the branch.

### Local Lemmas to Prove

**Lemma PS31.1 -- Hypothesis map is complete.**
For every endpoint theorem $T$ selected by the branch, each hypothesis
$H_j\in\mathcal H(T)$ must be supplied by a preceding Navier--Stokes estimate,
or else shown to be irrelevant to the selected theorem.

**Proof.** Write the theorem hypotheses as

$$
\mathcal H(T)=\{H_1,\ldots,H_m\}.
$$

For each $j$, define

$$
\mathfrak h_T(H_j)\in
\{\text{proved by an earlier estimate},\ \text{not applicable with reason},\
\text{missing}\}.
$$

The theorem hypotheses are complete precisely when no value equals `missing`.
A value `missing` identifies the exact theorem hypothesis not proved by the
branch data.

**Lemma PS31.2 -- Pressure conventions must match endpoint norms.**
For an endpoint theorem whose hypotheses contain pressure oscillations, the
branch pressure representative normalized by spatial means is admissible for
that theorem. For an endpoint theorem whose hypotheses contain only
$\nabla p$, time-dependent pressure gauges leave the theorem data unchanged.

**Proof.** For $P'=P+a(t)$,

$$
\nabla P'=\nabla P,
\qquad
P'-(P')_{B_r}=P-P_{B_r}.
$$

Thus both possible pressure data used by endpoint theorems, $\nabla P$ and
the oscillation $P-P_{B_r}$, are unchanged by the allowed gauge. The branch
pressure convention is therefore matched to the theorem norm by the displayed
identities.

**Lemma PS31.3 -- Convergence topology must supply the endpoint class.**
For an endpoint theorem whose hypotheses require a smooth ancient solution,
mild solution, suitable weak solution, or strong critical convergence, the
corresponding compactness node supplies exactly that class.

**Proof.** The branch record lists its solution class as one of

$$
C^\infty_{\rm loc}\text{ ancient},\qquad
\text{mild ancient},\qquad
\text{suitable weak},\qquad
\text{strongly convergent in the critical local norm}.
$$

The compactness steps `PS6`, `PS7`, `PS13`, `PS20`, and `PS30` specify which
entry in this list has been proved. The endpoint theorem is admissible exactly
when its required class equals the recorded entry.

**Lemma PS31.4 -- Missing hypotheses become explicit obligations.**
The value $\mathfrak h_T(H_j)=\text{missing}$ blocks endpoint application; the
missing item is assigned to the proof step that can supply it or to `PS33` as a
precise open attainability or exclusion item.

**Proof.** The endpoint-admissibility predicate is

$$
\forall H_j\in\mathcal H(T),\qquad
\mathfrak h_T(H_j)\ne\text{missing}.
$$

The displayed predicate fails as soon as one hypothesis has value `missing`.
Therefore the theorem is not applied, and the unresolved entry remains a named
regularity, decay, pressure, compactness, or attainability item.

### Specific Estimate

The decisive verification is

$$
\mathfrak h_T(H_j)=\text{proved}
\quad\text{for every applicable }H_j\in\mathcal H(T).
$$

### Practical Verification Steps

1. Select the endpoint theorem for the branch.
2. List all theorem hypotheses verbatim in mathematical form.
3. Attach each hypothesis to the estimate that proves it.
4. Verify pressure and convergence conventions.
5. Record missing hypotheses as explicit obligations.

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
\{\mathfrak h_T\text{ complete}\}.
$$

---

# 38. `PS32` -- Endpoint Exclusion Theorem Application

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is the endpoint branch whose hypotheses exactly match a recorded
Navier--Stokes theorem.

### Standing Assumptions

The incoming record states that the theorem is applicable exactly as checked in `PS31`.

### Objects Inspected

Inspect the theorem conclusion, local regularity implication, zero-profile
implication, or branch-emptiness implication.

### Dependencies Used

The theorem match comes from `PS31`; positive activity from `PS8`; local
singularity from `C_mu`; pressure and defect closure from `PS30`.

### Local Obstruction Predicate

$P_{\mathrm{PS32}}$ holds when the endpoint conclusion is insufficient to
contradict the retained branch.

### Local Lemmas to Prove

**Lemma PS32.1 -- CKN-type endpoint conclusions contradict singular entry.**
If the endpoint theorem gives boundedness or Holder regularity in a smaller
cylinder around the selected point, then the singular-entry conclusion of
`C_mu` is false.

**Proof.** The singular set was defined by failure of local boundedness on
every cylinder. Regularity on one smaller cylinder excludes membership in that
set.

**Lemma PS32.2 -- Zero-profile conclusions contradict retained activity.**
If the endpoint theorem gives $V\equiv0$ or $W\equiv0$, then the
pressure-normalized local CKN lower bound from `PS8` is impossible.

**Proof.** The zero velocity has zero velocity contribution. The pressure
gradient is zero in the endpoint equation. Choosing the spatial-mean pressure
gauge gives zero pressure oscillation on compact balls.

**Lemma PS32.3 -- Empty-branch conclusions remove the branch.**
If the endpoint theorem states that no object satisfying the verified
hypotheses exists, then the selected branch is impossible.

**Proof.** The branch hypothesis vector proves exactly those hypotheses; the
theorem denies existence of such an object.

**Lemma PS32.4 -- Non-exclusion becomes a realization question.**
If the theorem applies but permits a nonzero branch, the remaining issue is
whether such a branch is realized by a Navier--Stokes blow-up sequence; this is
the task of `PS33`.

**Proof.** The branch record contains two completed fields:

$$
\mathfrak h_T(H_j)\ne\text{missing}\quad(1\le j\le m),
\qquad
T(\text{branch})=\text{nonzero admissible object}.
$$

Thus theorem matching and theorem application have both been evaluated. The
remaining unevaluated predicate is the existence of a suitable NS3D blow-up
sequence converging to that admissible object, which is the realization
predicate checked in `PS33`.

### Specific Estimate

The decisive comparison is

$$
\text{endpoint conclusion}
\quad\Longrightarrow\quad
\neg\{\text{retained singular activity}\}.
$$

### Practical Verification Steps

1. Apply the endpoint theorem.
2. Translate its conclusion into the selected variables.
3. Compare with the retained CKN lower bound or singular-entry condition.
4. If no contradiction follows, record the precise non-excluded branch.

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
3. assign the non-excluded branch to realization analysis in `PS33`.

Progress measure: the endpoint conclusion either contradicts retained activity
or becomes a precise realizable-branch question.

## Data Passed Forward

The next proof step is `PS33`. The data passed forward are

$$
\Gamma_{\mathrm{PS32}}
=
\Gamma_{\mathrm{PS31}}
\cup
\{\text{endpoint conclusion and contradiction status}\}.
$$

---

# 39. `PS33` -- Realization or Admissible Counterexample Check

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is not a formal profile alone, but a profile together with an
admissible sequence of suitable weak solutions producing it.

### Standing Assumptions

The incoming record states that the branch has not been excluded by the selected endpoint theorem.

### Objects Inspected

Inspect the construction sequence, local energy inequality, pressure gauges,
convergence topology, activity lower bound, and compatibility with the defect
vector.

### Dependencies Used

Every preceding estimate contributes a necessary condition for realization;
failure of any condition gives non-attainability.

### Local Obstruction Predicate

$P_{\mathrm{PS33}}$ holds when the branch is not merely formal but actually
arises as a local blow-up limit from NS3D.

### Local Lemmas to Prove

**Lemma PS33.1 -- Realization requires all prior verified conditions.**
If a branch is realized by an admissible blow-up sequence, then the sequence
must satisfy the entry, concentration, compactness, pressure, defect, and
endpoint-hypothesis conclusions recorded in `C_mu` and `PS1`--`PS32`.

**Proof.** Each conclusion is a necessary condition imposed by the construction
of the branch. Dropping any one changes the local PDE problem or the endpoint
class.

**Lemma PS33.2 -- Failure of any necessary verified condition gives
non-attainability.**
If one prior necessary verified condition cannot be satisfied by the proposed
branch, then no admissible NS3D blow-up sequence realizes that branch.

**Proof.** Lemma PS33.1 requires every realization sequence to satisfy the
verified condition. The proposed branch violates that condition, so no
admissible realization sequence exists.

**Lemma PS33.3 -- Formal profiles without sequences are obligations, not
counterexamples.**
A profile solving a limiting equation but lacking a suitable weak
approximating sequence is not an admissible NS3D obstruction; it is a missing
attainability theorem.

**Proof.** The blow-up analysis studies limits of actual NS3D solutions. A
formal limiting solution not obtained from such a sequence does not contradict
regularity of the original equation.

### Specific Estimate

The decisive verification is the existence or nonexistence of an admissible
sequence satisfying

$$
u_n^{(z_n,r_n)}\to V
$$

in the branch topology with all previous local estimates.

### Practical Verification Steps

1. State the proposed realization sequence.
2. Check local energy and pressure conventions.
3. Check convergence topology and retained activity.
4. Check every prior verified condition against the sequence.
5. If no sequence is recorded, record the exact missing construction or
   non-attainability theorem.

## Estimate Step $B_{\mathrm{PS33}}$

The estimate step is verification of the realization package or a
non-attainability contradiction.

## Failure Case

Failure name: undecided branch realization.

Analytic meaning: a formal branch survives endpoint exclusion, and the record
lacks either an attaining NS3D sequence or a non-attainability proof.

## Refinement Step

Allowed refinements:

1. prove non-attainability from prior verified conditions;
2. construct an admissible sequence;
3. add a missing local estimate;
4. assign undecided attainability to `PS34` as residual.

Progress measure: the branch becomes non-attainable, realized, or part of the
exact residual complement.

## Data Passed Forward

The next proof step is `PS34`. The data passed forward are

$$
\Gamma_{\mathrm{PS33}}
=
\Gamma_{\mathrm{PS32}}
\cup
\{\text{realization status}\}.
$$

---

# 40. `PS34` -- Residual Complement Branch

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is an admissible local profile or local branch after every named
PDE alternative has been checked.

### Standing Assumptions

The incoming record states that each named branch predicate has a yes/no/inc conclusion and every
realization issue has been recorded.

### Objects Inspected

Inspect the list of named predicates, their overlaps, residual membership, and
all unresolved endpoint or attainability obligations.

### Dependencies Used

All previous branch classifications feed `PS34`; `PS33` supplies the last
non-exclusion status.

### Local Obstruction Predicate

$P_{\mathrm{PS34}}$ holds when there is no exact set-theoretic residual
identity for the remaining branches.

### Local Lemmas to Prove

**Lemma PS34.1 -- Complement identity.**
Every admissible local profile belongs either to $\mathcal U$ or to
$\mathcal R_{\rm loc}$.

**Proof.** The residual class is declared by the displayed formula

$$
\mathcal R_{\rm loc}
=\mathcal S_{\rm loc}\setminus\mathcal U.
$$

For $V\in\mathcal S_{\rm loc}$, membership in $\mathcal U$ gives the named
branch alternative, and nonmembership in $\mathcal U$ gives
$V\in\mathcal R_{\rm loc}$ by the displayed set identity.

**Lemma PS34.2 -- Overlaps among named branches create no residual class.**
If a profile belongs to more than one named branch, it is still in
$\mathcal U$ and is not residual.

**Proof.** Membership in a union requires membership in at least one branch.
Multiple memberships do not create a new complement class.

**Lemma PS34.3 -- Ordered subtraction gives disjoint reporting.**
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

### Specific Estimate

The decisive statement is the exact identity

$$
\mathcal S_{\rm loc}
=
\mathcal U\cup\mathcal R_{\rm loc}.
$$

### Practical Verification Steps

1. Define $\mathcal S_{\rm loc}$ explicitly.
2. List every named branch predicate.
3. Define $\mathcal U$ as their union.
4. Define the residual complement by set subtraction.
5. Record all residual obligations as explicit PDE statements.

## Estimate Step $B_{\mathrm{PS34}}$

The estimate step is set-theoretic and predicate verification, not a new PDE
estimate.

## Failure Case

Failure name: ill-defined residual complement.

Analytic meaning: a branch remains outside named classes without a precise
predicate or obligation.

## Refinement Step

Allowed refinements:

1. add the missing predicate;
2. define ordered subtraction;
3. move an unresolved theorem to `PS31` or `PS33`;
4. repeat the complement identity.

Progress measure: the residual class becomes exact.

## Data Passed Forward

The next proof step is `PS35`. The data passed forward are

$$
\Gamma_{\mathrm{PS34}}
=
\Gamma_{\mathrm{PS33}}
\cup
\{\mathcal R_{\rm loc}\text{ exact}\}.
$$

---

# 41. `PS35` -- Case-Decomposition Completeness Check

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is an arbitrary admissible local profile generated by the NS3D
blow-up procedure.

### Standing Assumptions

The incoming record states that the residual complement has been defined exactly in `PS34`.

### Objects Inspected

Inspect the complete branch list: Type I, Type II, compact, stationary, tight,
radiative, rough, multicore, finite packet, terminal, perturbative, structured,
scale-transition, rigidity, defect, endpoint, realization, and residual
classes.

### Dependencies Used

The nodes `C_mu` and `PS1`--`PS34` contribute one or more branch predicates or
exclusions.

### Local Obstruction Predicate

$P_{\mathrm{PS35}}$ holds if an admissible local profile is not assigned to any
named branch and is not in the residual complement.

### Local Lemmas to Prove

**Lemma PS35.1 -- Every profile is classified by named predicates or residual
membership.**
For every $V\in\mathcal S_{\rm loc}$, either $V\in\mathcal U_\alpha$ for some
$\alpha$, or $V\in\mathcal R_{\rm loc}$.

**Proof.** `PS34` defines

$$
\mathcal U=\bigcup_{\alpha\in A}\mathcal U_\alpha,
\qquad
\mathcal R_{\rm loc}=\mathcal S_{\rm loc}\setminus\mathcal U.
$$

For every $V\in\mathcal S_{\rm loc}$, either
$V\in\mathcal U$ or $V\notin\mathcal U$. In the first case, the union identity
gives at least one $\alpha\in A$ with $V\in\mathcal U_\alpha$; in the second
case, the complement identity gives $V\in\mathcal R_{\rm loc}$.

**Lemma PS35.2 -- Completed decomposition preserves endpoint obligations.**
If a branch is excluded, the exclusion status is recorded. If a branch is
not excluded, its endpoint, realization, or residual obligation is recorded.

**Proof.** Nodes `PS31`--`PS34` attach theorem, realization, and residual
statuses to every branch that reaches them.

**Lemma PS35.3 -- No silent branch remains after full-pass audit.**
Because `PS30` gives a complete defect vector and `PS34` gives an exact
residual complement, every limit equation term and branch predicate has a
named status.

**Proof.** A silent branch is either an unrecorded defect or an element outside
the residual complement. The first case contradicts the complete defect vector
from `PS30`; the second contradicts the complement identity from `PS34`.

### Specific Estimate

The decisive statement is the coverage identity

$$
\forall V\in\mathcal S_{\rm loc},\qquad
V\in\bigcup_{\alpha\in A}\mathcal U_\alpha
\quad\text{or}\quad
V\in\mathcal R_{\rm loc}.
$$

### Practical Verification Steps

1. List all branch predicates in order.
2. Confirm each predicate has a conclusion.
3. Confirm the residual complement identity.
4. Confirm no defect-vector entry is blank.
5. Pass the completed case decomposition to the compatibility nodes.

## Estimate Step $B_{\mathrm{PS35}}$

The estimate step is coverage verification.

## Failure Case

Failure name: incomplete local branch decomposition.

Analytic meaning: there is an admissible profile branch not represented by the
named alternatives or residual complement.

## Refinement Step

Allowed refinements:

1. add the missing branch predicate;
2. update the residual complement;
3. rerun the defect audit for the missing branch;
4. rerun endpoint matching if the new branch has an exclusion theorem.

Progress measure: the uncovered profile is either named or added to the exact
residual complement.

## Data Passed Forward

The next proof step is `Bound_partial`. The data passed forward are

$$
\Gamma_{\mathrm{PS35}}
=
\Gamma_{\mathrm{PS34}}
\cup
\{\text{local branch decomposition complete}\}.
$$

---

# 42. `Bound_partial` -- Boundary or Physical-Domain Compatibility

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a suitable weak solution on either $\mathbb R^3$ or a local
domain $\Omega$.

### Standing Assumptions

The incoming record states that all local profile branches have been classified or the no-profile conclusion
is active.

### Objects Inspected

Inspect the cylinder location, boundary conditions, trace spaces, pressure
representative, and flux terms in the local energy inequality.

### Dependencies Used

The domain comes from `H0`; cutoff commutators come from `PS17`, `PS20`, and
`PS30`; completed profile status comes from `PS35`.

### Local Obstruction Predicate

$P_{\mathrm{Bound\_partial}}$ holds only if a physical boundary or coupling
enters a cylinder used by the proof without a verified boundary estimate.

### Local Lemmas to Prove

**Lemma Bound_partial.1 -- Interior cylinders have no physical boundary term.**
For $Q_r(z_0)\Subset\Omega\times I$, all test functions used in the local
energy inequality are chosen compactly supported in the domain, and no
physical boundary integral appears.

**Proof.** Choose spatial cutoffs supported in $B_r(x_0)\Subset\Omega$. The
distributional equation and local energy inequality are tested with compactly
supported functions, so integration by parts produces no boundary term.

**Lemma Bound_partial.2 -- Whole-space profiles have no physical boundary.**
For $\Omega=\mathbb R^3$, every finite cylinder is interior.

**Proof.** Compact subsets of $\mathbb R^3$ have positive distance from the
empty boundary.

**Lemma Bound_partial.3 -- Boundary branches require trace and flux estimates.**
If a physical boundary is present, then the proof must verify the boundary
condition, trace class, pressure compatibility, and boundary local energy
inequality on boundary cylinders.

**Proof.** Boundary local regularity theorems require boundary condition,
trace, pressure, and boundary local energy hypotheses that are absent from the
interior CKN theorem. Applying an interior theorem across a boundary is
therefore a hypothesis mismatch and is assigned back to `PS31`.

### Specific Estimate

The decisive condition in the interior case is

$$
\operatorname{dist}(B_r(x_0),\partial\Omega)>0.
$$

### Practical Verification Steps

1. Identify the physical domain.
2. Check whether each proof cylinder is interior.
3. If whole-space or interior, record no physical boundary term.
4. If boundary cylinders occur, state the boundary condition and trace
   estimates needed.
5. Assign missing boundary hypotheses to `PS31` or boundary-specific analysis.

## Estimate Step $B_{\mathrm{Bound\_partial}}$

The estimate step is the interior-cylinder or boundary-trace verification.

## Failure Case

Failure name: unresolved boundary compatibility.

Analytic meaning: a proof cylinder touches a physical boundary without the
boundary estimates required by the endpoint theorem.

## Refinement Step

Allowed refinements:

1. shrink to an interior cylinder;
2. add boundary trace estimates;
3. switch to a boundary regularity theorem;
4. assign boundary defects to `PS30`.

Progress measure: every cylinder is either interior or covered by a boundary
theorem.

## Data Passed Forward

The next proof step is `Bound_B`. The data passed forward are

$$
\Gamma_{\mathrm{Bound\_partial}}
=
\Gamma_{\mathrm{PS35}}
\cup
\{\text{boundary compatibility status}\}.
$$

---

# 43. `Bound_B` -- Forcing, Lower-Order, or Cutoff-Source Compatibility

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is the selected localized or renormalized velocity equation.

### Standing Assumptions

The incoming record states that the physical equation is unforced and all additional terms are created
only by localization or normalization.

### Objects Inspected

Inspect all terms on the right side of the localized equation and the endpoint
space in which they must vanish or be controlled.

### Dependencies Used

Cutoff terms come from `PS17` and `PS20`; modulation terms from `PS5`,
`PS26`, and `PS28`; pressure terms from `PS4` and `PS30`.

### Local Obstruction Predicate

$P_{\mathrm{Bound\_B}}$ holds when a source term remains in the equation but
does not appear among the endpoint theorem hypotheses.

### Local Lemmas to Prove

**Lemma Bound_B.1 -- Physical forcing is absent.**
For the incompressible unforced NS3D equation, the physical source term is
identically zero.

**Proof.** The equation in `H0` is
$\partial_tu+(u\cdot\nabla)u+\nabla p=\Delta u$.

**Lemma Bound_B.2 -- Cutoff sources are compact commutators.**
If $\chi$ is fixed and smooth, then commutators involving $\nabla\chi$ and
$\Delta\chi$ are controlled by the local norms of velocity, stress, and
pressure already audited in `PS30`.

**Proof.** Fixed cutoff derivatives are bounded coefficients. Multiplication
by them maps the compact local spaces used in `PS30` into $H^{-1}$ or
$L^{3/2}$ on smaller cylinders.

**Lemma Bound_B.3 -- Vanishing modulation errors do not alter the endpoint
equation.**
If modulation errors tend to zero in $L^1_tH^{-1}_x$ on every endpoint
cylinder, then they do not alter the limiting endpoint equation.

**Proof.** Testing the equation against compact smooth divergence-free
functions shows the modulation error contributes a term bounded by its
$L^1H^{-1}$ norm times the test-function norm.

### Specific Estimate

The decisive estimate is

$$
\|F_n\|_{L^1_tH^{-1}_x(Q_R)}\to0
$$

for every artificial source not included in the endpoint theorem.

### Practical Verification Steps

1. Write the localized equation explicitly.
2. List every right-hand source term.
3. Mark physical forcing absent.
4. Estimate cutoff, pressure, and modulation sources.
5. Assign any nonvanishing source to `PS30` or `PS31`.

## Estimate Step $B_{\mathrm{Bound\_B}}$

The estimate step is source convergence in the endpoint topology.

## Failure Case

Failure name: unresolved local source term.

Analytic meaning: the branch equation differs from unforced NS3D by a term not
absorbed in the endpoint theorem.

## Refinement Step

Allowed refinements:

1. improve cutoff estimates;
2. change pressure decomposition;
3. refine modulation;
4. assign source defects to `PS30`.

Progress measure: every source is absent, vanishing, absorbed, or named.

## Data Passed Forward

The next proof step is `Bound_Sigma`. The data passed forward are

$$
\Gamma_{\mathrm{Bound\_B}}
=
\Gamma_{\mathrm{Bound\_partial}}
\cup
\{\text{source compatibility status}\}.
$$

---

# 44. `Bound_Sigma` -- Sufficiency of Input Data and Selected Objects

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The objects inspected are the NS3D hypotheses already assembled for the final
local exclusion, not a new PDE solution.

### Standing Assumptions

The incoming record states that local branch decomposition, boundary
compatibility, and source compatibility have been checked.

### Objects Inspected

Inspect every selected mathematical object used in the final implication and
its source node.

### Dependencies Used

Every previous node contributes to $\mathfrak S$.

### Local Obstruction Predicate

$P_{\mathrm{Bound\_\Sigma}}$ holds when a final proof step refers to an object
that was not constructed or was constructed inconsistently.

### Local Lemmas to Prove

**Lemma Bound_Sigma.1 -- Data completeness.**
If every entry of $\mathfrak S$ is assigned to the step that constructs it, then no
unconstructed object is used in the final implication.

**Proof.** The record is finite. Checking every entry proves completeness.

**Lemma Bound_Sigma.2 -- Compatibility of duplicated objects.**
If the same object is produced in two nodes, the later node must either use the
same representative or record the transformation between representatives.

**Proof.** For NS3D the only allowed changes are the declared scalings,
translations, rotations, Galilean transforms, and pressure gauges. Each has
already been checked in `PS26`.

### Specific Estimate

The decisive statement is

$$
\forall X\in\mathfrak S,\qquad
X\text{ has a prior construction node and compatibility status.}
$$

### Practical Verification Steps

1. Build the sufficiency record.
2. Attach each object to the node that constructs it.
3. Check representative compatibility.
4. Assign missing objects to their construction nodes.

## Estimate Step $B_{\mathrm{Bound\_\Sigma}}$

The estimate step is record verification.

## Failure Case

Failure name: insufficient analytic data.

Analytic meaning: the final implication uses a mathematical object not
constructed by the local proof.

## Refinement Step

Allowed refinements:

1. add the missing construction;
2. reconcile representatives through `PS26`;
3. add missing pressure or scale data;
4. rerun endpoint matching.

Progress measure: every object in $\mathfrak S$ becomes constructed and
compatible.

## Data Passed Forward

The next proof step is `GC_T`. The data passed forward are

$$
\Gamma_{\mathrm{Bound\_\Sigma}}
=
\Gamma_{\mathrm{Bound\_B}}
\cup
\{\mathfrak S\text{ complete}\}.
$$

---

# 45. `GC_T` -- Local Compatibility with the Target Regularity Statement

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is a hypothetical singular point of a suitable weak solution.

### Standing Assumptions

The incoming record states that all local branches generated from such a singular point have been
classified and their endpoint or residual statuses recorded.

### Objects Inspected

Inspect the local singularity criterion, positive concentration sequence,
branch decomposition, and final status of every branch.

### Dependencies Used

Singular entry comes from `C_mu` and normalized concentration from `PS1`;
branch completeness from `PS35`;
data sufficiency from `Bound_Sigma`.

### Local Obstruction Predicate

$P_{\mathrm{GC\_T}}$ holds if a singular point could fail to generate any of
the branches checked by the local analysis.

### Local Lemmas to Prove

**Lemma GC_T.1 -- Singular points generate local concentration.**
If $z_0$ is a singular point of a suitable weak solution, then there exists a
sequence of scales $r_n\downarrow0$ with positive local CKN concentration.

**Proof.** The contrapositive of CKN regularity at $z_0$ states

$$
z_0\in\Sigma
\Longrightarrow
\limsup_{r\downarrow0}
\{C(u;z_0,r)+D(p;z_0,r)\}\ge\varepsilon_0.
$$

Choose a sequence $r_n\downarrow0$ along which the limsup is attained up to a
factor $1/2$. Then
$C(u;z_0,r_n)+D(p;z_0,r_n)\ge\varepsilon_0/2$, which is the required positive
local CKN concentration.

**Lemma GC_T.2 -- Local concentration enters the profile decomposition.**
The positive concentration sequence constructed from a singular point enters
`C_mu` and `PS1`--`PS35` and is assigned to one named branch or the residual
complement.

**Proof.** `Rec_N` selects the singular entry point, `C_mu` constructs the
original-scale concentration sequence, `PS1` normalizes it on the fixed
cylinder, `PS2` fixes the center and parabolic scaling, and `PS3` separates
the local Type I predicate from its local Type II negation. The resulting
admissible local profile lies in
$\mathcal S_{\rm loc}$, and `PS35` gives

$$
\mathcal S_{\rm loc}
=
\left(\bigcup_{\alpha\in A}\mathcal U_\alpha\right)
\cup\mathcal R_{\rm loc}.
$$

Hence the concentration sequence is assigned to a named branch or to the
residual complement.

**Lemma GC_T.3 -- Excluding every admissible branch excludes the singular
point.**
If every branch produced by Lemma GC_T.2 is excluded or declared
non-attainable, then the initial assumption $z_0\in\Sigma$ is impossible.

**Proof.** Lemma GC_T.2 gives an admissible branch

$$
B\in
\left(\bigcup_{\alpha\in A}\mathcal U_\alpha\right)
\cup\mathcal R_{\rm loc}.
$$

The verified branch-status record assigns each element of this set one of the
statuses `excluded` or `non-attainable`. The first status contradicts the
endpoint conclusion for that branch; the second contradicts existence of the
NS3D blow-up sequence that produced $B$. Both alternatives contradict the
initial membership $z_0\in\Sigma$.

### Specific Estimate

The decisive implication is

$$
z_0\in\Sigma
\Longrightarrow
\exists\text{ admissible branch in }
\left(\bigcup_{\alpha\in A}\mathcal U_\alpha\right)\cup\mathcal R_{\rm loc}.
$$

### Practical Verification Steps

1. Start from the recorded singular-point hypothesis.
2. Produce a positive local concentration sequence.
3. Run the completed local branch decomposition.
4. Check the status of every branch.
5. Conclude regularity only if no attainable branch remains.

## Estimate Step $B_{\mathrm{GC\_T}}$

The estimate step is the CKN singular-entry implication.

## Failure Case

Failure name: target-compatibility gap.

Analytic meaning: the local decomposition does not cover every singular point
needed for the target regularity statement.

## Refinement Step

Allowed refinements:

1. strengthen singular-entry construction;
2. add missing local branch predicates;
3. refine the target regularity statement;
4. return to `PS35`.

Progress measure: every target singular point enters the completed local
decomposition.

## Data Passed Forward

The next proof step is `FinalExcl`. The data passed forward are

$$
\Gamma_{\mathrm{GC\_T}}
=
\Gamma_{\mathrm{Bound\_\Sigma}}
\cup
\{\text{local-to-target implication verified}\}.
$$

---

# 46. `FinalExcl` -- Final Local Singularity Exclusion Record

## Implementation and Verification in NS3D Terms

### Analytic Setting and Unknowns

The unknown is the finite branch-status record for all admissible local profile
branches.

### Standing Assumptions

The incoming record states that the local-to-target implication has been verified in `GC_T`.

### Objects Inspected

Inspect every branch status, endpoint theorem application, residual class, and
realization decision.

### Dependencies Used

All preceding estimates contribute to the final branch status.

### Local Obstruction Predicate

$P_{\mathrm{FinalExcl}}$ holds if some branch has status realized or remains an
attainable non-excluded obstruction.

### Local Lemmas to Prove

**Lemma FinalExcl.1 -- Complete negative record excludes local singularity.**
If every branch in $\mathfrak L$ has status excluded or nonattainable, then no
admissible local singular profile remains.

**Proof.** `PS35` gives that the branch universe is

$$
\left(\bigcup_{\alpha\in A}\mathcal U_\alpha\right)
\cup\mathcal R_{\rm loc}.
$$

Before `FinalExcl`, heterogeneous named classes are split into uniform-status
subclasses. Hence the condition that every subclass has status
`excluded` or `nonattainable` implies that every admissible branch $B$ in the
displayed universe has one of those two statuses. The first status contradicts
the endpoint theorem conclusion matched in `PS31` and applied in `PS32`; the
second status contradicts the existence of a suitable NS3D blow-up sequence
realizing $B$ as checked in `PS33`. Therefore no branch recorded in
$\mathfrak L$ remains as an admissible local singular profile.

**Lemma FinalExcl.2 -- Local exclusion implies regularity at the target point.**
If no admissible local singular profile remains, then the target point is
regular.

**Proof.** Lemma GC_T.3 proves the implication

$$
\{\text{all admissible branches excluded or nonattainable}\}
\Longrightarrow
z_0\notin\Sigma.
$$

The conclusion of Lemma FinalExcl.1 supplies the hypothesis on the left-hand
side, so the target point is regular.

**Lemma FinalExcl.3 -- Undecided or realized branches are exact obligations.**
For every branch with status undecided or realized, the final conclusion is that branch
and its named theorem gap, construction gap, or exclusion-estimate gap.

**Proof.** The record contains every branch and all endpoint/realization
statuses. Therefore any nonnegative conclusion is already localized to a
specific PDE obligation.

### Specific Estimate

The decisive final condition is

$$
\forall \alpha\in A,\qquad
\mathrm{status}_\alpha\in\{\mathrm{excluded},\mathrm{nonattainable}\},
\qquad
\mathrm{status}_{\rm res}\in\{\mathrm{excluded},\mathrm{nonattainable}\}.
$$

### Practical Verification Steps

1. Build the final branch record.
2. Split any branch class with mixed statuses into uniform-status subclasses.
3. Confirm each subclass has one allowed status.
4. If all statuses are excluded or nonattainable, apply `GC_T`.
5. If a branch is realized or undecided, report it as the exact remaining PDE
   obligation.

## Estimate Step $B_{\mathrm{FinalExcl}}$

The estimate step is final branch status assembly.

## Failure Case

Failure name: remaining local singular branch.

Analytic meaning: a branch in the exhaustive local decomposition remains
attainable and not excluded by the verified endpoint theorems.

## Refinement Step

Allowed refinements:

1. prove the missing endpoint theorem;
2. prove non-attainability;
3. refine the residual complement;
4. rerun endpoint matching and realization checks.

Progress measure: every remaining obstruction is reduced to a named theorem or
construction problem.

## Data Passed Forward

This is a terminal node. The data passed forward are

$$
\Gamma_{\mathrm{FinalExcl}}
=
\Gamma_{\mathrm{GC\_T}}
\cup
\{\mathfrak L\text{ final}\}.
$$
