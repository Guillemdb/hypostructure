# Paper I Roadmap: Type I Blow-up Limits and the Centered Ancient Equation

## Target Paper

**Proposed title.**  *Type I Blow-up Limits for Navier-Stokes and the Centered
Self-similar Ancient Equation*

**Purpose.**  Prove that a genuine Type I singularity produces a nonzero
bounded ancient solution of the centered self-similar Navier-Stokes equation,
and prove that this centered equation has no Type II-style scale-cascade
symmetry.

This document gives the exact lemma and theorem sequence for the first paper.
It is written in traditional PDE-paper order.

---

## Main Theorems of Paper I

### Theorem A: Type I Blow-up Limit

Let $u$ be a suitable Leray-Hopf solution of the three-dimensional
Navier-Stokes equations on $\mathbb{R}^3\times[0,T^*)$ with a genuine Type I
singularity at $(x_*,T^*)$.  Then there exist scales
$\lambda_k\downarrow0$ such that
$$
u_k(x,t)
=
\lambda_k u(x_*+\lambda_k x,T^*+\lambda_k^2t)
$$
converges, after passing to a subsequence and choosing pressure gauges, to an
ancient suitable weak solution $U$ on
$$
\mathbb{R}^3\times(-\infty,0).
$$
Moreover
$$
\sup_{t<0}\sqrt{-t}\,
\|U(t)\|_{L^\infty(\mathbb{R}^3)}<\infty.
$$

### Theorem B: Nontriviality of the Ancient Limit

The scales in Theorem A can be chosen so that the limit is nonzero.  More
precisely, the renormalized representative
$$
V(y,\tau)=\sqrt{-t}\,U(y\sqrt{-t},t),
\qquad \tau=-\log(-t),
$$
satisfies a fixed local lower bound of the form
$$
\sup_{\tau\in\mathbb{R}}\int_{B_{R_0}}|V(y,\tau)|^3\,dy
\ge \eta_0
$$
for some $R_0,\eta_0>0$ determined by the singularity normalization.

### Theorem C: Centered Renormalized Ancient Equation

The field $V$ solves
$$
\partial_\tau V+(V\cdot\nabla)V+\nabla\Pi
=
\Delta V-\frac12(V+y\cdot\nabla V),
\qquad \nabla\cdot V=0,
$$
on $\mathbb{R}^3\times\mathbb{R}$ and satisfies
$$
\|V\|_{L^\infty_{\tau,y}}<\infty.
$$

### Theorem D: No Type II Cascade Symmetry

The centered equation in Theorem C is not invariant under any positive
nontrivial scaling
$$
V(y,\tau)\mapsto \alpha V(\alpha y,\alpha^2\tau),
\qquad \alpha\ne1,
$$
nor under constant translations in $y$.  Therefore centered Type I ancient
limits do not possess the same same-point scale-cascade structure used in Type
II blow-up analysis.

---

## Section 1: Preliminaries and Normalizations

### Definition 1.1: Suitable Leray-Hopf Solution

State the precise solution class used in the paper:

- $u\in L^\infty_tL^2_x\cap L^2_t\dot H^1_x$ locally up to $T^*$;
- $\nabla\cdot u=0$ in distributions;
- $(u,p)$ solve Navier-Stokes distributionally;
- $(u,p)$ satisfy the local energy inequality;
- the pressure is taken modulo functions of time.

### Definition 1.2: Type I Singular Point

Define a Type I singular point $(x_*,T^*)$ by:
$$
\limsup_{t\uparrow T^*}
\sqrt{T^*-t}\,
\|u(t)\|_{L^\infty(\mathbb{R}^3)}
=M<\infty,
$$
and $(x_*,T^*)$ is not a regular point.

Also record the useful enlarged bound: after replacing $M$ by $2M+1$, there is
$t_0<T^*$ such that
$$
\|u(t)\|_{L^\infty(\mathbb{R}^3)}
\le
\frac{M}{\sqrt{T^*-t}},
\qquad t_0<t<T^*.
$$

### Definition 1.3: Parabolic Cylinders and Scale-invariant Quantities

For $z_0=(x_0,t_0)$ define
$$
Q_r(z_0)=B_r(x_0)\times(t_0-r^2,t_0).
$$
Introduce the CKN scale-invariant quantities
$$
C(u;z_0,r)
=
r^{-2}\iint_{Q_r(z_0)}|u|^3\,dx\,dt,
$$
and
$$
D(p;z_0,r)
=
r^{-2}\iint_{Q_r(z_0)}
|p-(p)_{B_r}(t)|^{3/2}\,dx\,dt.
$$

### Lemma 1.4: Scaling of the Equations and Local Energy Inequality

For $\lambda>0$ define
$$
u^\lambda(x,t)=\lambda u(x_*+\lambda x,T^*+\lambda^2t),
$$
$$
p^\lambda(x,t)=\lambda^2p(x_*+\lambda x,T^*+\lambda^2t).
$$
Then $(u^\lambda,p^\lambda)$ is again a suitable weak solution on its rescaled
domain, and the local energy inequality is preserved under this scaling.

**Use.**  This is the formal basis for every later compactness statement.

---

## Section 2: Rescaled Sequence and Uniform Bounds

### Lemma 2.1: Exhaustion of the Ancient Time Domain

Let $\lambda_k\downarrow0$ and define
$$
u_k(x,t)=\lambda_k u(x_*+\lambda_k x,T^*+\lambda_k^2t).
$$
Then $u_k$ is defined on
$$
\mathbb{R}^3\times(-T^*/\lambda_k^2,0),
$$
and these domains exhaust
$$
\mathbb{R}^3\times(-\infty,0)
$$
as $k\to\infty$.

### Lemma 2.2: Inherited Type I Bound

For every fixed $t<0$ and all sufficiently large $k$,
$$
\|u_k(t)\|_{L^\infty(\mathbb{R}^3)}
\le
\frac{M}{\sqrt{-t}}.
$$
Consequently, for every compact interval
$I\Subset(-\infty,0)$,
$$
\sup_k\|u_k\|_{L^\infty(\mathbb{R}^3\times I)}
\le
C_I M.
$$

**Use.**  Gives local boundedness away from the terminal time $t=0$.

### Lemma 2.3: Uniform Local Kinetic Energy Bounds Away from $t=0$

For every compact cylinder
$$
Q=B_R\times[-S,-\sigma]
\Subset
\mathbb{R}^3\times(-\infty,0),
\qquad 0<\sigma<S<\infty,
$$
there is a constant $C=C(R,S,\sigma,M)$ such that
$$
\sup_k\|u_k\|_{L^\infty_tL^2_x(Q)}
\le C.
$$

**Proof input.**  Use Lemma 2.2 and the finite volume of $B_R$.

### Lemma 2.4: Uniform Local Dissipation Bounds

For every compact cylinder
$$
Q=B_R\times[-S,-\sigma]\Subset
\mathbb{R}^3\times(-\infty,0),
$$
there is $C=C(R,S,\sigma,M)$ such that
$$
\sup_k\|\nabla u_k\|_{L^2(Q)}\le C.
$$

**Proof input.**  Apply the local energy inequality to $u_k$ with a cutoff
equal to $1$ on $Q$ and supported in a slightly larger compact cylinder.  Bound
the transport and pressure terms using Lemma 2.2 and the pressure estimates in
Lemma 2.5.

### Lemma 2.5: Local Pressure Decomposition and Bounds

For each compact cylinder
$$
Q=B_R\times[-S,-\sigma],
$$
choose the pressure gauge so that
$$
p_k=q_k+h_k,
$$
where
$$
-\Delta q_k=\partial_i\partial_j(u_{k,i}u_{k,j})
$$
locally in a larger ball, and $h_k$ is harmonic in space.  Then
$$
\sup_k\|p_k\|_{L^{3/2}(Q)}\le C(R,S,\sigma,M)
$$
after subtracting a function of time if necessary.

**Proof input.**  Calderón-Zygmund estimates for $q_k$ and harmonic estimates
for $h_k$, using the local $L^\infty$ bound on $u_k$.

### Lemma 2.6: Local Time Derivative Bound

For every compact cylinder $Q\Subset\mathbb{R}^3\times(-\infty,0)$,
$$
\partial_t u_k
=
\Delta u_k-(u_k\cdot\nabla)u_k-\nabla p_k
$$
is uniformly bounded in a negative Sobolev space, for example
$$
\partial_tu_k
\quad\text{bounded in}\quad
L^{3/2}_tW^{-1,3/2}_x(Q).
$$

**Use.**  This is the Aubin-Lions compactness input if one does not use a
pure local-regularity compactness argument.

### Proposition 2.7: Local Compactness of the Rescaled Sequence

After passing to a subsequence,
$$
u_k\to U
$$
strongly in $L^q_{\mathrm{loc}}(\mathbb{R}^3\times(-\infty,0))$ for each
finite $q$, and locally uniformly after using parabolic regularity away from
$t=0$.  Also
$$
p_k\rightharpoonup P
$$
locally in $L^{3/2}$ after fixing gauges.

**Proof input.**  Lemmas 2.2--2.6 plus Aubin-Lions; local boundedness then
upgrades convergence by parabolic regularity on compact subcylinders.

---

## Section 3: The Ancient Suitable Weak Limit

### Lemma 3.1: Distributional Limit Solves Navier-Stokes

The limit $(U,P)$ satisfies
$$
\partial_tU+(U\cdot\nabla)U+\nabla P=\Delta U,
\qquad
\nabla\cdot U=0
$$
in distributions on $\mathbb{R}^3\times(-\infty,0)$.

**Proof input.**  Strong local convergence of $u_k$ and weak convergence of
$p_k$.

### Lemma 3.2: Stability of the Local Energy Inequality

The limit $(U,P)$ satisfies the local energy inequality on
$\mathbb{R}^3\times(-\infty,0)$.

**Proof input.**  Lower semicontinuity of the dissipation term,
strong local convergence of $u_k$, and weak convergence of $p_k$ in
$L^{3/2}_{\mathrm{loc}}$.

### Lemma 3.3: Inheritance of the Type I Bound

For every $t<0$,
$$
\|U(t)\|_{L^\infty(\mathbb{R}^3)}
\le
\frac{M}{\sqrt{-t}},
$$
in the appropriate essential supremum sense.  Equivalently,
$$
\sup_{t<0}\sqrt{-t}\,\|U(t)\|_{L^\infty}<\infty.
$$

**Proof input.**  Lemma 2.2 and local weak-star or locally uniform convergence.

### Proposition 3.4: Ancient Suitable Weak Limit

The pair $(U,P)$ is an ancient suitable weak solution on
$$
\mathbb{R}^3\times(-\infty,0),
$$
and it satisfies the Type I ancient bound from Lemma 3.3.

**Use.**  This proves Theorem A except for nontriviality.

---

## Section 4: Nontriviality of the Ancient Limit

This is the delicate section.  It must prevent the rescaling procedure from
producing the zero ancient solution.

### Lemma 4.1: Epsilon Regularity Criterion

There exists $\varepsilon_{\mathrm{CKN}}>0$ such that if
$$
C(u;z_0,r)+D(p;z_0,r)
\le \varepsilon_{\mathrm{CKN}}
$$
for some parabolic cylinder $Q_r(z_0)$, then $u$ is regular in a smaller
cylinder, for example $Q_{r/2}(z_0)$.

**Use.**  This is the contrapositive tool: singularity forces a scale-invariant
lower bound.

### Lemma 4.2: Concentration Forced by Singularity

If $(x_*,T^*)$ is singular, then for every sufficiently small $r>0$,
$$
C(u;(x_*,T^*),r)+D(p;(x_*,T^*),r)
\ge \varepsilon_{\mathrm{CKN}}.
$$

**Proof.**  Immediate from Lemma 4.1 by contraposition.

### Lemma 4.3: Choice of Normalizing Scales

There exists a sequence $\lambda_k\downarrow0$ such that the rescaled sequence
satisfies a fixed lower bound on a unit cylinder:
$$
\iint_{Q_1(0,0)}|u_k|^3\,dx\,dt
+
\iint_{Q_1(0,0)}
|p_k-(p_k)_{B_1}(t)|^{3/2}\,dx\,dt
\ge c_0>0.
$$

**Proof input.**  Take $\lambda_k=r_k$ in Lemma 4.2 and use scale invariance of
$C$ and $D$.

### Lemma 4.4: Velocity Nontriviality Alternative

After possibly adjusting the normalization, the lower bound in Lemma 4.3 can
be converted into a velocity lower bound on a compact cylinder away from
$t=0$:
$$
\iint_{B_{R_0}\times[-S_0,-\sigma_0]}|u_k|^3\,dx\,dt
\ge c_1>0.
$$

**Reason.**  If all velocity mass vanished on every compact subcylinder away
from $t=0$, pressure lower bounds alone would be harmonic-gauge artifacts or
would contradict the local pressure decomposition and epsilon regularity.

**Remark.**  This is the step that must be written most carefully.  One can
avoid pressure-only issues by using a velocity-only regularity criterion if
one chooses that route instead.

### Lemma 4.5: Passage of the Lower Bound to the Limit

The strong local convergence from Proposition 2.7 gives
$$
\iint_{B_{R_0}\times[-S_0,-\sigma_0]}|U|^3\,dx\,dt
\ge c_1>0.
$$
In particular $U\not\equiv0$.

### Proposition 4.6: Nonzero Ancient Limit

The ancient suitable weak solution $U$ obtained in Proposition 3.4 is nonzero.
Equivalently, its renormalized representative $V$ satisfies
$$
\sup_{\tau\in\mathbb{R}}
\int_{B_{R_0}}|V(y,\tau)|^3\,dy
\ge \eta_0
$$
for suitable $R_0,\eta_0>0$.

**Use.**  This proves Theorem B.

---

## Section 5: Smoothness Away from the Terminal Time

### Lemma 5.1: Local Serrin Regularity from the Type I Bound

On every compact cylinder
$$
Q\Subset\mathbb{R}^3\times(-\infty,0),
$$
the ancient limit satisfies
$$
U\in L^\infty(Q).
$$
Therefore $U$ is smooth on $Q$ by standard local regularity theory for bounded
solutions.

### Lemma 5.2: Higher Derivative Bounds

For every compact cylinder $Q\Subset\mathbb{R}^3\times(-\infty,0)$ and every
integer $m\ge0$,
$$
\|\nabla^m U\|_{L^\infty(Q)}
+
\|\nabla^m P\|_{L^\infty(Q)}
\le C_{Q,m}.
$$

**Use.**  Justifies the classical change of variables in Section 6.

---

## Section 6: Centered Self-similar Variables

### Lemma 6.1: Change of Variables

Define
$$
y=\frac{x}{\sqrt{-t}},
\qquad
\tau=-\log(-t),
\qquad
V(y,\tau)=\sqrt{-t}\,U(y\sqrt{-t},t).
$$
Then $t\in(-\infty,0)$ corresponds to $\tau\in\mathbb{R}$.

### Lemma 6.2: Derivation of the Centered Equation

The field $V$ satisfies
$$
\partial_\tau V+(V\cdot\nabla)V+\nabla\Pi
=
\Delta V-\frac12(V+y\cdot\nabla V),
\qquad
\nabla\cdot V=0.
$$

The pressure rescales as
$$
\Pi(y,\tau)=(-t)P(y\sqrt{-t},t)
$$
up to functions of $\tau$.

### Lemma 6.3: Uniform Renormalized Boundedness

The Type I ancient bound gives
$$
\|V\|_{L^\infty_{\tau,y}}\le M.
$$

### Lemma 6.4: Nontriviality in Renormalized Variables

The nontriviality lower bound from Proposition 4.6 transfers to a fixed
renormalized local lower bound:
$$
\sup_{\tau\in\mathbb{R}}\int_{B_{R_0}}|V(y,\tau)|^3\,dy
\ge \eta_0.
$$

### Proposition 6.5: Centered Ancient Renormalized Solution

The renormalized field $V$ is a nonzero bounded classical ancient solution of
the centered self-similar Navier-Stokes equation on
$$
\mathbb{R}^3\times\mathbb{R}.
$$

**Use.**  This proves Theorem C.

---

## Section 7: No Type II Cascade Symmetry

### Lemma 7.1: Failure of Scaling Invariance

Let $V$ solve the centered equation.  Define
$$
\widetilde V(y,\tau)=\alpha V(\alpha y,\alpha^2\tau).
$$
Then the transport, pressure, and diffusion terms scale like $\alpha^3$, while
the drift term
$$
-\frac12(V+y\cdot\nabla V)
$$
scales like $\alpha$.  Hence $\widetilde V$ solves the same equation only for
the trivial positive scaling $\alpha=1$.

### Lemma 7.2: Failure of Translation Invariance

For a constant vector $a\in\mathbb{R}^3$, define
$$
\widetilde V(y,\tau)=V(y-a,\tau).
$$
Then
$$
y\cdot\nabla \widetilde V(y,\tau)
\ne
(y-a)\cdot\nabla V(y-a,\tau)
$$
unless $a=0$ or the profile is degenerate.  Thus constant translations in
$y$ are not symmetries of the centered equation.

### Proposition 7.3: No Same-point Scale-cascade Structure

The centered Type I ancient equation has no intrinsic same-point scale-cascade
symmetry.  Multi-scale spatial features may occur inside a single profile, but
they are not a cascade of rescaled copies governed by the Type II
camera-on-innermost argument.

**Use.**  This proves Theorem D and fixes the conceptual foundation for the
next papers.

---

## Exact Dependency Chain

The proof of Paper I should be written in the following order:

1. Definition 1.1: suitable Leray-Hopf solution.
2. Definition 1.2: Type I singular point and enlarged Type I bound.
3. Definition 1.3: cylinders and CKN quantities.
4. Lemma 1.4: scaling of solutions and local energy inequality.
5. Lemma 2.1: rescaled domains exhaust the ancient domain.
6. Lemma 2.2: inherited Type I $L^\infty$ bound.
7. Lemma 2.3: local kinetic energy bounds.
8. Lemma 2.5: local pressure decomposition and pressure bounds.
9. Lemma 2.4: local dissipation bounds.
10. Lemma 2.6: local time derivative bounds.
11. Proposition 2.7: compactness of the rescaled sequence.
12. Lemma 3.1: distributional limit solves Navier-Stokes.
13. Lemma 3.2: local energy inequality passes to the limit.
14. Lemma 3.3: Type I bound passes to the limit.
15. Proposition 3.4: ancient suitable weak limit.
16. Lemma 4.1: epsilon regularity criterion.
17. Lemma 4.2: singularity forces scale-invariant concentration.
18. Lemma 4.3: choose normalizing scales.
19. Lemma 4.4: convert concentration to a velocity lower bound.
20. Lemma 4.5: pass the lower bound to the limit.
21. Proposition 4.6: nonzero ancient limit.
22. Lemma 5.1: local smoothness away from $t=0$.
23. Lemma 5.2: higher derivative and pressure bounds.
24. Lemma 6.1: define centered self-similar variables.
25. Lemma 6.2: derive the centered equation.
26. Lemma 6.3: uniform renormalized boundedness.
27. Lemma 6.4: nontriviality in renormalized variables.
28. Proposition 6.5: nonzero bounded centered ancient solution.
29. Lemma 7.1: no scaling symmetry.
30. Lemma 7.2: no translation symmetry.
31. Proposition 7.3: no Type II same-point cascade structure.
32. Theorem A: Type I blow-up limit.
33. Theorem B: nontriviality.
34. Theorem C: centered renormalized ancient equation.
35. Theorem D: no Type II cascade symmetry.

---

## Delicate Points to Settle Before Writing

### Nontriviality route

The most delicate step is Lemma 4.4.  There are two possible routes:

1. Use the standard CKN quantity $C+D$ and prove that pressure-only
   concentration cannot survive if velocity vanishes locally.
2. Use a velocity-only regularity criterion, if the paper chooses to import
   one, and avoid pressure-only concentration altogether.

The second route is cleaner if the chosen velocity-only criterion has exactly
the hypotheses needed here.

### Time intervals near $t=0$

Compactness is first proved on subcylinders compactly contained in
$$
\mathbb{R}^3\times(-\infty,0),
$$
not on cylinders touching $t=0$.  Any normalization using $Q_1(0,0)$ should be
converted to a lower bound on a cylinder away from $t=0$ before passing to the
ancient limit.

### Pressure gauges

The pressure should always be stated modulo functions of time.  Every pressure
bound must specify the chosen gauge, usually by subtracting spatial averages
over a ball.

### Centering

The first paper should keep the blow-up center fixed at $x_*$.  If one later
recenters around points $x_k\to x_*$, that should be stated explicitly as a
separate normalization.  The centered renormalized equation itself is not
translation invariant.

---

## Expected Output of Paper I

At the end of Paper I, the series should have the following PDE object:

There exists a nonzero bounded smooth solution
$$
V:\mathbb{R}^3\times\mathbb{R}\to\mathbb{R}^3
$$
of
$$
\partial_\tau V+(V\cdot\nabla)V+\nabla\Pi
=
\Delta V-\frac12(V+y\cdot\nabla V),
\qquad \nabla\cdot V=0,
$$
with
$$
\|V\|_{L^\infty_{\tau,y}}<\infty
$$
and a fixed local nontriviality lower bound.

The paper also establishes that this centered equation has no nontrivial
scaling symmetry and no constant spatial translation symmetry.  Thus the next
papers must study compact ancient dynamics and extremizers, not Type II
same-point cascades.
