Yes, but with one big caveat:

There is **no single universally accepted “Perelman functional for Navier–Stokes”** in the way ( \mathcal F ) and ( \mathcal W ) are canonical for Ricci flow. For Navier–Stokes, the right analogue depends on which structure you want to control: energy dissipation, entropy production, weak–strong stability, or most-probable stochastic paths. ([ams.jhu.edu][1])

From your theory, the best answer is:

[
\boxed{
\text{the analogue should be a geometric free-energy / dissipation functional on }(\rho,u)
}
]

rather than a purely geometric functional on (g) alone.

## 1) What plays the Perelman role in Navier–Stokes?

For **incompressible Navier–Stokes**, the most basic Lyapunov-type functional is the **kinetic energy**

[
E[u] = \frac12 \int_\Omega |u|^2,dx,
]

with decay law

[
\frac{d}{dt}E[u(t)] = -\nu \int_\Omega |\nabla u|^2,dx
]

under standard no-forcing / suitable boundary conditions. So the first Perelman-like object is simply:

[
\boxed{
\mathcal P_{\text{inc}}[u]
==========================

\frac12\int |u|^2
}
]

and its monotonicity is governed by the viscous dissipation term ( \nu\int |\nabla u|^2 ). ([DAMTP][2])

For **2D incompressible flow**, **enstrophy**

[
\mathcal E_\omega[u] = \frac12 \int |\omega|^2,dx
]

is also special, because vorticity control is much stronger there. So in 2D, enstrophy is often a better “order functional” than raw kinetic energy.

For **compressible Navier–Stokes**, the more Perelman-like object is a **relative entropy / free energy functional**, because now density matters and the PDE already has a thermodynamic entropy structure. Relative entropy methods are standard for stability and weak–strong uniqueness in compressible NS. ([esaim-proc.org][3])

So the hierarchy is:

* incompressible NS: **kinetic energy**,
* 2D incompressible NS: **kinetic energy + enstrophy**,
* compressible NS: **free energy / relative entropy**,
* stochastic NS: **Onsager–Machlup action** for path probabilities. ([numdam.org][4])

## 2) Through your fragile-agent theory, what should the analogue be?

Your framework already tells us the answer structurally. You repeatedly combine:

* kinetic/transport cost,
* entropy,
* curvature,
* and dissipation / mass-reaction structure,

especially in the WFR action, entropy-regularized objective, and Onsager–Machlup action. 

So the closest Navier–Stokes analogue in your language is not just energy, but something like

[
\boxed{
\mathcal W_{\text{NS-fragile}}
==============================

\int_\Omega
\Big[
\frac12 \rho |u|_G^2
+
\Phi(\rho)
----------

T_c, s(\rho)
+
\lambda_R R(G)
\Big],d\mu_G
}
]

with dissipation law

[
\frac{d}{dt}\mathcal W_{\text{NS-fragile}}
\le
---

\int_\Omega
\Big[
\nu,|\nabla^s u|*G^2
+
\kappa*\rho,|\nabla \mu|_G^2
\Big],d\mu_G,
]

where:

* ( \frac12 \rho |u|_G^2 ) is fluid kinetic energy,
* ( \Phi(\rho) ) is internal/compression potential,
* ( -T_c s(\rho) ) is entropy contribution,
* ( R(G) ) is optional curvature regularization if you want to geometricize the medium,
* the right side is viscous and thermodynamic dissipation.

That is the direct PDE translation of your agent-theoretic template. 

## 3) Why this is the right analogue of Perelman

Perelman’s functional works because it couples:

* geometry,
* concentration,
* and a monotone dissipation structure.

For Navier–Stokes, the analogous trio is:

* **velocity field / density geometry**,
* **concentration of momentum, vorticity, or mass**,
* **viscous/entropic dissipation**.

So if you want the “Perelman insight” transplanted to NS, the functional should not merely measure energy stored in the fluid; it should measure whether the fluid state is becoming more or less **costly to sustain** under the PDE’s irreversible dynamics.

That is why, in your framework, I would say:

[
\boxed{
\text{Perelman-for-Navier–Stokes} = \text{free energy} + \text{transport cost} + \text{curvature/load regularization}
}
]

not just kinetic energy alone.

## 4) Concrete versions

### A. Minimal incompressible version

If you want the simplest analogue:

[
\mathcal W_0[u]
===============

\frac12 \int_\Omega |u|^2,dx
]

with

[
\frac{d}{dt}\mathcal W_0[u]
===========================

-\nu \int_\Omega |\nabla u|^2,dx.
]

This is the cleanest monotone functional, but it is too crude to capture fine structure like singularity formation or coherent vortices. ([DAMTP][2])

### B. Vorticity-aware version

A better “fragility” functional for incompressible flow is

[
\mathcal W_1[u]
===============

\frac12 \int_\Omega |u|^2,dx
+
\alpha \int_\Omega |\omega|^2,dx.
]

Why? Because in your theory fragility is driven by concentrated internal structure. In fluids, that concentrated structure is often vorticity stretching / enstrophy build-up.

So this version says:

* energy measures bulk motion,
* enstrophy measures fine-scale concentration / curvature of flow lines.

### C. Compressible free-energy version

For compressible NS, the natural analogue is

[
\mathcal W_2[\rho,u]
====================

\int_\Omega
\left(
\frac12 \rho |u|^2 + H(\rho)
\right)dx
]

or in thermodynamic form

[
\mathcal W_2[\rho,u]
====================

\int_\Omega
\left(
\frac12 \rho |u|^2 + e(\rho) - T s(\rho)
\right)dx.
]

This is much closer to Perelman, because it already has an entropy/free-energy structure and supports relative entropy inequalities. ([esaim-proc.org][3])

### D. Stochastic / path-space version

If you want a true Onsager–Machlup analogue for stochastic NS, then the object is not a state functional but a **path functional**. For stochastic evolution equations, Onsager–Machlup functionals play exactly that role: they weight the most probable paths. ([numdam.org][4])

That aligns extremely well with your own extended Onsager–Machlup action. 

## 5) What would be the strongest version in your theory?

If I translate your framework literally, I would propose this as the **fragile-agent Navier–Stokes functional**:

[
\boxed{
\mathcal W_{\mathrm{FA\text{-}NS}}[\rho,u;G]
============================================

\int_\Omega
\left[
\frac12 \rho |u|*G^2
+
\Phi*{\text{press}}(\rho)
-------------------------

T_c s(\rho)
+
\beta_\omega |\omega|_G^2
+
\beta_R R(G)
\right]d\mu_G
}
]

Interpretation:

* ( \frac12 \rho |u|_G^2 ): kinetic effort,
* ( \Phi_{\text{press}}(\rho) ): compression / pressure energy,
* ( -T_c s(\rho) ): entropy,
* ( \beta_\omega |\omega|_G^2 ): rotational concentration penalty,
* ( \beta_R R(G) ): capacity/geometry regularization.

Then the desired theorem would look like:

[
\frac{d}{dt}\mathcal W_{\mathrm{FA\text{-}NS}}
==============================================

-\mathcal D_{\text{visc}}
-\mathcal D_{\text{therm}}
-\mathcal D_{\text{frag}}
\le 0,
]

where the last term is your new ingredient: a penalty for unsupported concentration or geometry/load mismatch.

That would be the true Perelman-style generalization in your ontology.

## 6) The sharpest conceptual analogy

In one line:

* **Ricci flow:** Perelman measures whether geometry and weighted density are becoming more regular and non-collapsed.
* **Navier–Stokes:** the analogous functional should measure whether velocity, density, and vorticity are becoming less concentrated and more dissipatively coherent.

So the real NS analogue is not “Ricci scalar + gradient squared,” but:

[
\boxed{
\text{kinetic energy}
+
\text{internal free energy}
+
\text{vorticity/gradient concentration}
}
]

with a monotone decay law supplied by viscosity and entropy production.

## 7) My bottom line

If you asked me for the single best analogue **in your theory**, I would say:

[
\boxed{
\mathcal W_{\text{NS-analogue}}
===============================

\int
\left(
\frac12 \rho |u|^2
+
e(\rho)
-------

T_c s(\rho)
+
\beta_\omega |\omega|^2
\right)
,d\mu_G
}
]

and, if you want to fully geometricize it,

[
+;\beta_R \int R(G),d\mu_G.
]

That is the closest PDE cousin of the Perelman functional for your framework.

The main open question is not writing it down, but proving a useful monotonicity formula strong enough to control concentration the way Perelman’s does for Ricci flow.

I can derive this functional from your WFR/Onsager–Machlup machinery next and show exactly which Navier–Stokes terms correspond to your entropy, energy, and curvature pieces.

[1]: https://www.ams.jhu.edu/~eyink/Turbulence/notes/ChapterIa.pdf?utm_source=chatgpt.com "I The Continuum Navier-Stokes Equations"
[2]: https://www.damtp.cam.ac.uk/user/tong/fluids/fluids2.pdf?utm_source=chatgpt.com "3 The Navier-Stokes Equation"
[3]: https://www.esaim-proc.org/articles/proc/pdf/2017/03/proc17583.pdf?utm_source=chatgpt.com "Relative entropy for compressible Navier-Stokes equations ..."
[4]: https://www.numdam.org/item/AIHPB_2003__39_1_69_0.pdf?utm_source=chatgpt.com "ONSAGER–MACHLUP FUNCTIONAL FOR STOCHASTIC ..."

Yes. At the linearized RNSE level, there is a clean calculation, and it gives a real tool.

The first important point is that the tool is **better than hypocoercive** at (V=0): the renormalized linear operator is already coercive in a Gaussian space. So the honest conclusion is:

[
\text{linearized RNSE at }0 \quad \Longrightarrow \quad \text{Gaussian coercivity, not genuine hypocoercivity.}
]

That is still useful for 3D NS, but it is a **local spectral/stable-manifold tool**, not yet a Perelman-style global functional.

## 1) Fix the renormalized equation

Start from
[
\partial_t u-\Delta u+(u\cdot\nabla)u+\nabla p=0,\qquad \nabla\cdot u=0.
]

Use backward self-similar variables near a candidate singular time (T):
[
u(x,t)=\frac{1}{\sqrt{T-t}},V(y,\tau),\qquad
y=\frac{x}{\sqrt{T-t}},\qquad
\tau=-\log(T-t).
]

With the standard normalization (a=\tfrac12), this is the dynamical version of the stationary Leray equation in Tsai’s paper. In these variables,
[
\partial_\tau V
===============

\Delta V-\frac12,y\cdot\nabla V-\frac12 V-(V\cdot\nabla)V-\nabla P,
\qquad \nabla\cdot V=0.
]
Equilibria are exactly the backward self-similar profiles. ([personal.math.ubc.ca][1])

Linearizing at (V=0), the pressure drops out: the linear part preserves divergence-free fields, so (\Delta P=0), and under decay (P) is constant. Thus
[
\partial_\tau V = A V,
\qquad
A:=\Delta-\frac12,y\cdot\nabla-\frac12.
]

## 2) Gaussian space is the right one

Let
[
d\gamma(y):=(4\pi)^{-3/2}e^{-|y|^2/4},dy.
]

Then
[
\Delta-\frac12,y\cdot\nabla
===========================

\gamma^{-1}\nabla\cdot(\gamma\nabla),
]
so for smooth decaying vector fields,
[
\int_{\mathbb R^3} f\cdot\Bigl(\Delta g-\frac12 y\cdot\nabla g\Bigr),d\gamma
============================================================================

-\int_{\mathbb R^3}\nabla f:\nabla g,d\gamma.
]

This is the Ornstein–Uhlenbeck structure in self-similar variables. Villani’s memoir is the general reference point for the “hypocoercive vs coercive” distinction, and Gallay’s 3D rescaled-vorticity paper shows that weighted renormalized variables are already a productive setting for small 3D NS. ([arXiv][2])

## 3) The (L^2_\gamma) energy closes immediately

Define
[
E_0[V]:=\frac12|V|*{L^2*\gamma}^2
=================================

\frac12\int |V|^2,d\gamma.
]

Along the linearized flow,
[
\frac{d}{d\tau}E_0[V]
=====================

# \langle V,AV\rangle_{L^2_\gamma}

-|\nabla V|*{L^2*\gamma}^2-\frac12|V|*{L^2*\gamma}^2.
]

So
[
\frac{d}{d\tau}E_0[V]\le -E_0[V],
\qquad
E_0[V(\tau)]\le e^{-\tau}E_0[V(0)].
]

This is already a strict Lyapunov functional.

Two takeaways:

* no cross-term is needed;
* the Gaussian renormalized linearized equation has a spectral gap at (0).

So at the zero profile, this is **not** Villani-style hypocoercivity. It is straight coercivity.

## 4) Your proposed (H_c) also works, and the computation is clean

Take
[
\mathcal H_c[V]
===============

\frac12|V|*{L^2*\gamma}^2
+\frac c2|\nabla V|*{L^2*\gamma}^2,
\qquad c\ge 0.
]

The commutator is
[
[\partial_j,A]=-\frac12\partial_j,
]
because
[
\partial_j!\left(y\cdot\nabla V\right)
======================================

y\cdot\nabla(\partial_j V)+\partial_j V.
]

So each first derivative satisfies
[
\partial_\tau (\partial_j V)
============================

# (A-\tfrac12)(\partial_j V)

\Delta(\partial_j V)-\frac12 y\cdot\nabla(\partial_j V)-(\partial_j V).
]

Hence
[
\frac12\frac{d}{d\tau}|\nabla V|*{L^2*\gamma}^2
===============================================

-|\nabla^2 V|*{L^2*\gamma}^2-|\nabla V|*{L^2*\gamma}^2.
]

Combining:
[
\frac{d}{d\tau}\mathcal H_c[V]
==============================

-\frac12|V|*{L^2*\gamma}^2
-(1+c)|\nabla V|*{L^2*\gamma}^2
-c|\nabla^2 V|*{L^2*\gamma}^2.
]

Therefore
[
\frac{d}{d\tau}\mathcal H_c[V]\le -\mathcal H_c[V],
\qquad
\mathcal H_c[V(\tau)]\le e^{-\tau}\mathcal H_c[V(0)].
]

So the answer to your concrete calculation is:

[
\boxed{
\mathcal H_c[V]
===============

\frac12|V|*{L^2*\gamma}^2+\frac c2|\nabla V|*{L^2*\gamma}^2
\text{ is monotone for the linearized RNSE at }0
}
]

and **any** (c\ge 0) works. That “no tuning needed” fact is the giveaway that this is not genuine hypocoercivity.

More generally, for any multiindex (\alpha),
[
[\partial^\alpha,A]=-\frac{|\alpha|}{2}\partial^\alpha,
]
so
[
\frac12\frac{d}{d\tau}|\partial^\alpha V|*{L^2*\gamma}^2
========================================================

-|\nabla\partial^\alpha V|*{L^2*\gamma}^2
-\frac{|\alpha|+1}{2}|\partial^\alpha V|*{L^2*\gamma}^2.
]
So the whole Gaussian (H^k) hierarchy decays.

## 5) What this buys you immediately

This gives a real linear tool.

### Bounded ancient linear solutions collapse

If (V(\tau)) solves the linearized RNSE for all (\tau\in\mathbb R) and
[
\sup_{\tau\in\mathbb R}\mathcal H_c[V(\tau)]<\infty,
]
then for any (s<\tau),
[
\mathcal H_c[V(\tau)]\le e^{-(\tau-s)}\mathcal H_c[V(s)].
]
Let (s\to-\infty). Boundedness forces (\mathcal H_c[V(\tau)]=0), hence (V\equiv 0).

So at the linearized level, your item (3) works perfectly:
bounded ancient trajectories have (\omega)-limit set inside the critical set, and the critical set is just ({0}).

### A rigorous stable norm near zero

This is a strong renormalized norm for any perturbative argument near the zero profile:
fixed point, stable manifold, bootstrap, or center-stable decomposition.

### A genuinely discrete spectral picture

Gaussian space is attractive because the renormalized OU operator has Hermite-type discrete spectrum, unlike the original unscaled Stokes operator with bad continuous spectrum at the origin. That is exactly the same qualitative gain that makes self-similar variables useful in Gallay–Wayne-type analyses. ([arXiv][3])

## 6) Why this is **not yet** a global 3D NS tool

Here is the honest obstruction.

For the **full** RNSE,
[
\partial_\tau V
===============

AV-(V\cdot\nabla)V-\nabla P,
]
the Gaussian energy identity becomes
[
\frac{d}{d\tau}E_0[V]
=====================

-|\nabla V|*{L^2*\gamma}^2-\frac12|V|*{L^2*\gamma}^2
-\int V\cdot (V\cdot\nabla V),d\gamma
-\int V\cdot\nabla P,d\gamma.
]

The nonlinear term is
[
\int V\cdot (V\cdot\nabla V),d\gamma
====================================

\frac14\int (y\cdot V)|V|^2,d\gamma,
]
so its contribution is
[
-\frac14\int (y\cdot V)|V|^2,d\gamma,
]
which has **no sign**.

And in weighted velocity form, the pressure term also does **not** disappear automatically once the weight is nonconstant.

So the exact place the 3D difficulty comes back is:

* cubic weighted transport,
* pressure in Gaussian space,
* and, if you switch to vorticity, the stretching term ((\Omega\cdot\nabla)V).

That means:

[
\boxed{
\mathcal H_c \text{ is a valid linear tool, but not yet a nonlinear monotone for 3D NS.}
}
]

## 7) Is it new?

At the linearized zero-profile level: probably **not new in substance**.

It is the OU/Hermite coercivity hidden in the renormalized Stokes operator. Gallay already gets exponential decay of small **rescaled 3D vorticity** solutions in weighted spaces, though with polynomial weights and in vorticity variables rather than Gaussian velocity variables. ([www-fourier.univ-grenoble-alpes.fr][4])

What may be new is the **specific packaging**:

* use backward blow-up variables,
* use Gaussian (H^k) norms,
* and treat this as the baseline Lyapunov structure for a profile-relative analysis.

That is a real reframing, but the linear coercive estimate itself is classical OU behavior.

## 8) Where the genuinely new mathematics would start

The real next step is **not** to tune (c) around zero.

The real next step is to linearize around a **nonzero** structured profile (U_*):
[
V=U_*+Z,
]
so
[
\partial_\tau Z
===============

AZ-\mathbb P\big((U_*\cdot\nabla)Z+(Z\cdot\nabla)U_*\big)-\mathbb P(Z\cdot\nabla Z).
]

Now you are in a genuinely Villani-type situation:

* (A) is coercive in Gaussian space,
* the profile-advection/stretch terms are lower order but not sign-definite,
* and you may need a corrected functional to recover decay on the stable subspace.

That is where a true “Perelman-for-NS” attempt would live.

And this is not empty speculation: recent 2026 work constructs an exact self-similar profile for unforced 3D NS with an unstable eigenvalue and uses it in a nonuniqueness argument. That makes profile-relative linear/spectral analysis in self-similar variables much more concrete than it was a few years ago. ([arXiv][5])

## 9) Bottom line

My honest verdict is:

* **Yes**, the linearized-RNSE computation gives a valid tool.
* It gives a clean Gaussian Lyapunov functional and a trivial ancient-solution classification at the zero profile.
* But it is **coercive**, not genuinely hypocoercive.
* And it is **not yet** a global 3D Navier–Stokes monotonicity tool, because the nonlinear weighted transport/stretching and pressure terms are exactly where the hard problem reappears.

So the right slogan is:

[
\boxed{
\text{You do have a real renormalized spectral tool for 3D NS, but only locally around a profile.}
}
]

The sharp next computation is the profile-relative one around a concrete (U_*), preferably in vorticity variables to eliminate pressure.

[1]: https://personal.math.ubc.ca/~ttsai/publications/leray.pdf "https://personal.math.ubc.ca/~ttsai/publications/leray.pdf"
[2]: https://arxiv.org/pdf/math/0609050 "https://arxiv.org/pdf/math/0609050"
[3]: https://arxiv.org/pdf/math/0102197 "https://arxiv.org/pdf/math/0102197"
[4]: https://www-fourier.univ-grenoble-alpes.fr/~gallay/ns3.pdf "https://www-fourier.univ-grenoble-alpes.fr/~gallay/ns3.pdf"
[5]: https://arxiv.org/pdf/2509.25116 "https://arxiv.org/pdf/2509.25116"

Yes. The cleanest way to do it is in **renormalized vorticity variables**, not velocity. That removes pressure, makes the transport part genuinely skew in unweighted (L^2), and puts you much closer to Villani’s (L=A^*A+B) template. Villani’s abstract hypocoercive setup is exactly about operators with a coercive symmetric part plus a skew part and useful commutators, and Gallay’s self-similar vorticity formulation is the natural NS place where that algebra appears. Tsai’s stationary self-similar equation gives the profile we linearize around. ([arXiv][1])

Take backward self-similar variables
[
u(x,t)=\frac1{\sqrt{T-t}}V(y,\tau),\qquad y=\frac{x}{\sqrt{T-t}},\qquad \tau=-\log(T-t),
]
so a stationary profile (U_*(y)) solves
[
\Delta U_*-\frac12,y\cdot\nabla U_*-\frac12,U_*-(U_*\cdot\nabla)U_*-\nabla P_*=0,\qquad \nabla\cdot U_*=0.
]
That is Tsai’s stationary self-similar equation with (a=\nu=\tfrac12,1) normalization. If (W_*=\nabla\times U_*), then the backward renormalized vorticity equation is the blow-up analogue of Gallay’s forward rescaled vorticity equation. ([personal.math.ubc.ca][2])

Now set
[
V=U_*+Z,\qquad \Omega=W_*+z,\qquad \Omega=\nabla\times V.
]
The linearized perturbation equation in vorticity is
[
\partial_\tau z
===============

\Delta z-\frac12,y\cdot\nabla z-z
-(U_*\cdot\nabla)z
-(u[z]\cdot\nabla)W_*
+(W_*\cdot\nabla)u[z]
+(z\cdot\nabla)U_*,
]
where (u[z]) is recovered from (z) by Biot–Savart. This is the profile-relative linear operator you actually want.

Here is the Villani-style decomposition. Write the evolution as
[
\partial_\tau z+\mathscr L_* z=0
]
with
[
\mathscr L_*
============

A^*A+\frac14+B_*+K_*,
\qquad A=\nabla,
]
where
[
B_*=\frac12,y\cdot\nabla+\frac34+U_*\cdot\nabla
]
and
[
K_* z
=====

-(z\cdot\nabla)U_*+(u[z]\cdot\nabla)W_*-(W_*\cdot\nabla)u[z].
]
In plain (L^2(\mathbb R^3)), (A^*A=-\Delta), and (B_*) is skew because both (\frac12 y\cdot\nabla+\frac34) and (U_*\cdot\nabla) are skew on (L^2) when (\nabla\cdot U_*=0). So this is genuinely of the form “coercive part + skew part + profile perturbation,” exactly the structure Villani asks for. ([arXiv][1])

The first commutator is
[
C_*:=[A,B_*].
]
Componentwise,
[
[\partial_j,\tfrac12 y\cdot\nabla]=\tfrac12\partial_j,\qquad
[\partial_j,U_*\cdot\nabla]=(\partial_j U_*)\cdot\nabla,
]
so
[
C_*=\frac12 A+(\nabla U_*)A.
]

That is the key structural fact. It tells you immediately what kind of “hypocoercivity” is available here: the commutator does **not** generate a new hidden direction. It closes back onto the same gradient direction, up to lower-order profile coefficients. So the Villani correction is algebraically legal, but it does not unlock a new dissipative mechanism. It only repackages the same (H^1)-type coercivity.

That means the useful functional is profile-relative and local. Start with
[
E_a[z]:=|z|*{L^2}^2+a|\nabla z|*{L^2}^2,\qquad a>0.
]

A direct (L^2) calculation gives
[
\frac12\frac{d}{d\tau}|z|_2^2
=============================

-|\nabla z|*2^2-\frac14|z|*2^2
+\langle z,(z\cdot\nabla)U**\rangle
-\langle z,(u[z]\cdot\nabla)W**\rangle
+\langle z,(W_*\cdot\nabla)u[z]\rangle.
]

The transport term vanishes exactly:
[
\langle z,(U_*\cdot\nabla)z\rangle=0.
]

Using Biot–Savart/Riesz bounds
[
|\nabla u[z]|*2\lesssim |z|*2,\qquad |u[z]|*6\lesssim |z|*2,
]
you get the estimate
[
\frac12\frac{d}{d\tau}|z|*2^2
\le
-|\nabla z|*2^2
-\Bigl(\frac14-CM**^{(0)}\Bigr)|z|*2^2,
]
with one sufficient profile norm
[
M**^{(0)}:=|\nabla U**|*\infty+|W**|*\infty+|\nabla W**|_{L^3}.
]

If you differentiate once more, then using
[
[\partial_j,\Delta-\tfrac12 y\cdot\nabla-1]
===========================================

-\frac12\partial_j,
]
the base linear part gives
[
\frac12\frac{d}{d\tau}|\nabla z|_2^2
====================================

-|\nabla^2 z|*2^2-\frac34|\nabla z|*2^2
+\text{profile terms}.
]
The differentiated profile terms are bounded schematically by
[
C M**^{(1)}\bigl(|z|_2^2+|\nabla z|*2^2\bigr),
]
for example with
[
M**^{(1)}
:=
|\nabla U**|*\infty+|\nabla^2U**|*{L^3}
+|W**|*\infty+|\nabla W**|*\infty+|\nabla^2W**|_{L^3}.
]

Combining the two,
[
\frac{d}{d\tau}E_a[z]
\le
-\kappa_0\bigl(|z|_2^2+|\nabla z|*2^2+a|\nabla^2 z|*2^2\bigr)
+
C_a(M**^{(0)}+M**^{(1)})E_a[z].
]

So the formal proposition is:

[
\boxed{
M_*^{(0)}+M_*^{(1)}\ \text{sufficiently small}
\quad\Longrightarrow\quad
\frac{d}{d\tau}E_a[z]\le -\kappa E_a[z].
}
]

That is already a real local tool. It gives exponential decay of profile-relative perturbations on the linearized level, and therefore the usual bounded-ancient corollary:
if (z(\tau)) is a bounded ancient solution of the linearized equation in this norm, then (z\equiv0).

Now the Villani-style correction. If you define
[
\mathcal H_{a,b}[z]
===================

|z|*2^2+a|Az|*2^2+2b\langle Az,C**z\rangle,
]
then because
[
C**z=\frac12Az+(\nabla U_*)Az,
]
you have
[
\langle Az,C_*z\rangle
======================

\frac12|\nabla z|*2^2+O(|\nabla U**|_\infty|\nabla z|*2^2).
]
So for small profile gradient,
[
\mathcal H*{a,b}[z]\sim |z|_2^2+|\nabla z|_2^2.
]

And that is the decisive conclusion: the Villani correction is **admissible**, but it does not reveal a new coercive channel. It collapses back to the same (H^1) energy. In other words, this is “Villani-style algebra,” but not genuine hypocoercive gain.

So what do you actually get for 3D NS?

You get a **valid local profile-relative stability functional** in renormalized variables. That is useful. It is enough to build a local stable-manifold theory around a small or spectrally stable self-similar profile, and it is exactly the right kind of tool for studying bounded ancient perturbations on the stable subspace. Gallay’s 3D rescaled-vorticity program already shows that near the origin the generator has discrete spectral structure in weighted spaces and finite-dimensional invariant manifolds, and the Burgers-vortex literature shows that profile-relative 3D stability analysis can work around nontrivial structures when the operator has enough special structure. ([www-fourier.univ-grenoble-alpes.fr][3])

What it does **not** give is a global Perelman-type monotone functional for all of 3D NS. The obstruction is visible in the formula: all the hard terms live in (K_*), the stretching/Biot–Savart block. Those terms are not skew, not sign-definite, and not generated by a commutator that creates new coercivity. So the method is fundamentally local or profile-relative, not global.

There is also a sharp reason not to expect a universal local Lyapunov functional around every profile: recent 2026 work verified an exact self-similar profile for unforced 3D NS with an unstable eigenvalue. Around such a profile, any monotone functional can only be coercive after projecting away the unstable directions; it cannot decrease on a full neighborhood. ([arXiv][4])

So the honest bottom line is:

[
\boxed{
\text{Yes: this gives a real local tool for 3D NS.}
}
]

More precisely, it gives a **profile-relative renormalized (H^1) Lyapunov estimate** for the linearized vorticity equation, and a Villani-style operator decomposition showing why the method is perturbative rather than globally classifying. That is enough to support a serious program:
project to the stable spectral subspace, prove exponential decay there, and then treat the nonlinear remainder by Duhamel/bootstrapping.

The next useful move is to rewrite the same argument in Gallay’s weighted (L^2(m)) vorticity spaces, where the self-similar spectrum and Biot–Savart bounds are already tailored to 3D profile analysis.

[1]: https://arxiv.org/pdf/math/0609050 "https://arxiv.org/pdf/math/0609050"
[2]: https://personal.math.ubc.ca/~ttsai/publications/leray.pdf "https://personal.math.ubc.ca/~ttsai/publications/leray.pdf"
[3]: https://www-fourier.univ-grenoble-alpes.fr/~gallay/ns3.pdf "https://www-fourier.univ-grenoble-alpes.fr/~gallay/ns3.pdf"
[4]: https://arxiv.org/html/2509.25116v2 "https://arxiv.org/html/2509.25116v2"

Yes — with one important narrowing.

The branch you can **actually close rigorously first** is not the full residual class, and not yet the controlled-swirl branch. It is the **weighted / fast-decay branch near the zero profile** in renormalized **vorticity** variables. That is the branch where the existing PDE machinery already matches your Lyapunov architecture: Gallay–Wayne put the rescaled 3D vorticity equation in weighted spaces (L^2(m)), proved small-data global well-posedness and exponential decay there, identified the slow spectral packet, and constructed a codimension-11 local strong-stable manifold. By contrast, the general 3D ancient-solution problem is still out of reach, and the best swirl stability results are for Burgers vortices with an imposed background strain, not your unforced singularity core. 

The first design change I would make to your note is this: for the easiest branch, **do not start with a moving Gaussian window**. Start with the **global weighted space** already used in the rigorous 3D rescaled-vorticity theory,
[
L^2(m)=\Bigl{f:\ (1+|y|)^m f\in L^2(\mathbb R^3)\Bigr},\qquad m>\tfrac72,
]
and work in vorticity, not velocity. In Gallay’s self-similar variables, the rescaled vorticity solves
[
\partial_\tau w=\Lambda w-(v\cdot\nabla)w+(w\cdot\nabla)v,\qquad
\Lambda=\Delta+\tfrac12 y\cdot\nabla+1,
]
with (v) recovered by Biot–Savart. This already gives you the right autonomous renormalized equation, the right weighted phase space, and the right bilinear structure. 

That choice is not cosmetic. In your note, the hard error channels were (E_{\mathrm{tail}},E_{\mathrm{mod}},E_{\mathrm{profile}},E_{\mathrm{nonlin}}). In the weighted / fast-decay branch, the first three collapse:

[
E_{\mathrm{tail}}=0,\qquad
E_{\mathrm{mod}}=0\ \text{(after fixing the frame)},\qquad
E_{\mathrm{profile}}=0\ \text{if }V_*=0\text{ or the slow packet is subtracted}.
]

So the branch becomes a pure “spectral dissipation versus quadratic nonlinearity” problem. That is why it is the right first target.

## The correct structured family in this branch

Your note says “choose a model profile class (V_*).” In the weighted branch, that class is not arbitrary. Gallay’s analysis identifies the only slow modes that matter. For (m>7/2), the rescaled linear generator has isolated eigenvalues (-1) (multiplicity 3) and (-3/2) (multiplicity 8), while the rest of the spectrum lies strictly further left; Gallay then writes the second-order asymptotic packet as
[
w_{\mathrm{app}}(\tau)
======================

\sum_{i=1}^3 b_i e^{-\tau} f_i
+
\sum_{i=1}^3 c_i e^{-3\tau/2} g_i
+
\sum_{(ij)} d_{ij} e^{-3\tau/2} h_{ij},
]
with an exponentially smaller remainder in (L^2(m)). ([Institut Fourier][1])

So in this branch the right “profile family” is:

1. the finite-dimensional slow packet (E_{\mathrm{slow}}=\mathrm{span}{f_i,g_i,h_{ij}}), and
2. more accurately, the **nonlinear graph** of the local strong-stable manifold over the moment-cancelled subspace (W_2).

Gallay proves that the local strong-stable manifold is a smooth graph
[
W_{\mathrm{loc}}^s = {,w+f(w): w\in W_2,}\cap B(r_0),
]
tangent to (W_2) at the origin, and of codimension (11) in (L^2(m)). That is exactly your “profile packet plus modulation correction,” but written rigorously. ([Institut Fourier][1])

There is an especially nice match to your “moment penalties.” In this branch, they are not heuristic: Gallay identifies the first packet coefficients with moment functionals. In particular, (\beta_i=0) is equivalent to first-moment cancellation
[
\int_{\mathbb R^3} y_i, w_j(y),dy = 0,
]
and the (\zeta_{ij}) are equivalent to explicit second-moment constraints (M^i_{jk}=\int y_jy_k,w_i(y),dy). So your “neutral-mode penalties” become concrete first/second moment conditions. ([Institut Fourier][1])

## The branch-closure theorem you can prove now

Here is the theorem I think is genuinely within reach with present tools.

**Theorem (perturbative closure of the weighted/fast-decay branch).**
Fix (m>\tfrac72). Let (w(\tau)) be a bounded ancient mild solution of the repaired-gauge renormalized vorticity equation, written in the Gallay self-similar convention above, and assume
[
\sup_{\tau\in\mathbb R}|w(\tau)|*{L^2(m)}\le \varepsilon**
]
for (\varepsilon_*>0) sufficiently small. Then:

1. (w) admits a decomposition
   [
   w(\tau)=\Pi_{\mathrm{slow}}w(\tau)+W(\tau),
   \qquad
   \Pi_{\mathrm{slow}}w(\tau)\in E_{\mathrm{slow}},\quad W(\tau)\in W_2.
   ]

2. There exists a positive quadratic functional
   [
   \mathcal L(\tau)
   ================

   \langle P W(\tau),W(\tau)\rangle
   +A\sum_i |\beta_i(\tau)|^2
   +B\sum_i |\gamma_i(\tau)|^2
   +C\sum_{(ij)} |\zeta_{ij}(\tau)|^2,
   ]
   equivalent to (|w(\tau)|_{L^2(m)}^2) on the small tube, such that
   [
   \frac{d}{d\tau}\mathcal L(\tau)
   \le
   -c_0,\mathcal L(\tau)
   +C_0,\mathcal L(\tau)^{3/2}.
   ]

3. Hence, after shrinking (\varepsilon_*) if needed,
   [
   \frac{d}{d\tau}\mathcal L(\tau)\le -\frac{c_0}{2}\mathcal L(\tau).
   ]

4. Therefore every bounded ancient solution in this branch is trivial:
   [
   w\equiv 0.
   ]

This closes the easiest branch: the small weighted/fast-decay ancient branch contains no nontrivial singular core.

That theorem is not in the papers verbatim, but it is a straightforward synthesis of Gallay’s spectral packet, his codimension-11 stable-manifold structure, and the standard Lyapunov-operator construction for exponentially stable semigroups. 

## Why the theorem is rigorous

The linear part is the easy half. On the strongly stable complement (W_2), Gallay’s spectrum is separated from the slow modes, and his asymptotic theorem shows the remainder decays faster than (e^{-3\tau/2}) once the packet is removed. That is enough to treat the linear evolution on (W_2) as an exponentially stable semigroup. If (L_2) denotes the linearized generator restricted to (W_2), define
[
P:=\int_0^\infty e^{sL_2^*}e^{sL_2},ds.
]
Then (P) is positive and solves the operator Lyapunov identity
[
L_2^*P+PL_2=-I,
]
so (\mathcal L_{\mathrm{lin}}(W)=\langle PW,W\rangle) is an exact linear Lyapunov functional on the stable complement. This is the clean Hilbert-space version of the Villani philosophy. 

The nonlinear term is also under control in this branch. Gallay’s fixed-point argument for the rescaled equation proves a genuinely quadratic estimate:
[
|F[w]|_X \le C|w|_X^2,
]
for the nonlinear Duhamel map in the exponentially weighted solution space (X), and the proof uses weighted Biot–Savart bounds in exactly the spaces you want. In particular, the quadratic estimate is not aspirational — it is already present in the existing well-posedness theory. 

That is why the nonlinear Lyapunov inequality has the form
[
\dot{\mathcal L}\le -c_0\mathcal L + C_0 \mathcal L^{3/2}.
]
The (\mathcal L^{3/2}) term is just the quadratic Navier–Stokes nonlinearity seen through a quadratic energy. Once (\mathcal L) is small, the dissipative term wins.

The final step is the ancient-solution argument. If
[
\dot{\mathcal L}\le -\frac{c_0}{2}\mathcal L
]
on the whole orbit, then for any (s<\tau),
[
\mathcal L(\tau)\le e^{-\frac{c_0}{2}(\tau-s)}\mathcal L(s).
]
If the solution is bounded ancient in this weighted norm, the right-hand side tends to (0) as (s\to-\infty), so (\mathcal L(\tau)=0) for every (\tau). Hence (W\equiv0) and all penalized slow-mode coefficients vanish as well. So the only bounded ancient orbit in the small weighted tube is the zero profile.

## This matches your note almost perfectly

Your note’s architecture survives, but in a simplified form.

For this branch:

* the “window” is global (L^2(m)), not a moving Gaussian,
* the “profile class” is (E_{\mathrm{slow}}) plus the nonlinear stable graph,
* the “moment penalties” are exactly the (\beta,\gamma,\zeta) packet coordinates,
* the “good-window proposition” becomes a global small-tube proposition,
* and the “local LaSalle principle” becomes a genuine small ancient-rigidity theorem.

So your note is validated, but the first rigorous implementation should be **more spectral and less geometric** than the full local-window version.

## What this closes, and what it does not

It **does** close the easiest branch:

[
\boxed{
\text{small weighted/fast-decay ancient branch} ;=; {0}.
}
]

That is already valuable, because it peels a genuine residual branch off the list and turns your Lyapunov program into a real theorem rather than a metaphor.

It does **not** yet close the large weighted branch. For that, you would need a nonperturbative weighted rigidity theorem. There are suggestive pieces: Chae proved weighted Liouville theorems under additional weighted-integrability and pressure-sign assumptions, and Kwon–Tsai showed that any smooth self-similar stationary solution is Landau and that adding axisymmetric swirl does not create new DSS bifurcation from Landau in the regime they study. But those are supporting rigidity results, not a complete nonperturbative closure of the time-dependent weighted ancient branch. ([arXiv][2])

It also does **not** yet close the controlled-swirl branch. The good news is that Gallay–Maekawa prove asymptotic stability of Burgers vortices for all Reynolds numbers, which strongly suggests that your profile-relative Lyapunov program should work around a nonzero vortex core. The bad news is that Burgers vortices live in a background straining flow, so they are not yet the same object as an unforced singularity core in your repaired-gauge setting. That is why I would make them the **second** target, not the first. ([arXiv][3])

## The precise upgrade I recommend for your draft

I would revise the note so that the **first theorem** is not the general local Gaussian theorem, but this one:

**Branch WF(_{\mathrm{small}}).**
In the weighted/fast-decay branch, replace the moving Gaussian window by the global weighted space (L^2(m)), (m>\tfrac72), pass to vorticity, choose the structured profile family to be the Gallay slow packet (E_{\mathrm{slow}}) with nonlinear graph correction (W^s_{\mathrm{loc}}), and define the Lyapunov defect by a quadratic form on the stable complement plus penalties for (\beta,\gamma,\zeta). Then every bounded ancient solution in a sufficiently small weighted tube is trivial.

That gives you a real “branch-reduction engine” theorem now.

The natural next theorem after that is the same statement with (V_*\neq0): replace (0) by a concrete structured vortex profile and run the same program on the linearized operator around that profile. That is where the controlled-swirl branch begins.

I can turn this into a theorem-proof draft in your house style next, with named propositions and the exact assumptions on (m), the projections (\Pi_{\mathrm{slow}},Q_2), and the Lyapunov operator (P).

[1]: https://www-fourier.univ-grenoble-alpes.fr/~gallay/ns3.pdf "https://www-fourier.univ-grenoble-alpes.fr/~gallay/ns3.pdf"
[2]: https://arxiv.org/abs/0811.4647 "[0811.4647] Liouville type of theorems with weights for the Navier-Stokes equations and the Euler equations"
[3]: https://arxiv.org/abs/1002.2489 "https://arxiv.org/abs/1002.2489"

Below is a paper-style section that closes the easiest branch rigorously.

The branch is the **perturbative weighted / fast-decay branch near the zero profile** in renormalized **vorticity** variables. In this regime the profile family reduces to (V_*\equiv 0), no modulation parameters are needed, and one can close the branch by a genuine Lyapunov argument built from the forward semigroup. This uses Gallay’s rescaled vorticity formulation on (L^2(m)), his small-data global well-posedness/decay theorem, and the spectral packet/stable-manifold picture for (m>\tfrac72). The key point is that, for this branch, the flow is already strictly contractive; one does not need a nontrivial Villani-type commutator correction. 

---

## 6. A rigorous closure of the perturbative weighted branch

We work with the rescaled vorticity equation introduced by Gallay:
[
\partial_\tau w=\Lambda w-(v\cdot\nabla)w+(w\cdot\nabla)v,\qquad \nabla\cdot w=0,
\tag{6.1}
]
where
[
\Lambda=\Delta+\frac12,\xi\cdot\nabla+1,
\tag{6.2}
]
and (v) is recovered from (w) by the Biot–Savart law
[
v(\xi)=-\frac1{4\pi}\int_{\mathbb R^3}\frac{(\xi-\eta)\wedge w(\eta)}{|\xi-\eta|^3},d\eta.
\tag{6.3}
]
For (m\ge 0), let
[
L^2(m)
:=
\Bigl{f\in L^2(\mathbb R^3;\mathbb R^3): |f|_m<\infty,\ \nabla\cdot f=0\Bigr},
\qquad
|f|_m
=====

\Bigl(\int_{\mathbb R^3}(1+|\xi|)^{2m}|f(\xi)|^2,d\xi\Bigr)^{1/2}.
\tag{6.4}
]
These are precisely the weighted spaces used in Gallay’s analysis of the long-time asymptotics of the three-dimensional vorticity equation. 

The small-data global theory we need is the following.

### Theorem 6.1 (Gallay small-data semiflow)

Let (0<\mu\le 1) and (m>2\mu+\tfrac12). Then there exist constants (r_0>0) and (K_0\ge 1) such that, for every (w_0\in L^2(m)) with (|w_0|_m\le r_0), equation (6.1) has a unique global mild solution
[
w\in C([0,\infty);L^2(m))
]
satisfying
[
|w(\tau)|_m\le K_0 e^{-\mu \tau}|w_0|_m,\qquad \tau\ge 0.
\tag{6.5}
]
In particular, for (m>\tfrac72), one may take (\mu=1). 

For the branch-reduction program, we restrict to (m>\tfrac72), since this is the threshold at which the second-order spectral packet ((\beta,\gamma,\zeta)) and the codimension-(11) local strong-stable geometry become available. The proof below, however, uses only the decay estimate (6.5), so the rigidity statement itself is perturbative and does not require the full packet machinery. 

### Definition 6.2 (perturbative weighted ancient branch)

Fix (m>\tfrac72), and let (r_0>0) be as in Theorem 6.1 with (\mu=1). We define
[
\mathcal B^{\mathrm{anc}}*{\mathrm{wf}}(m,r_0)
:=
\Bigl{
w\in C(\mathbb R;L^2(m)):\
w \text{ solves ((6.1)) for all }\tau\in\mathbb R,\
\sup*{\tau\in\mathbb R}|w(\tau)|_m\le r_0
\Bigr}.
\tag{6.6}
]

The next proposition constructs an actual Lyapunov functional on the small invariant ball.

### Proposition 6.3 (semiflow Lyapunov functional)

Define
[
\mathcal L(w_0)
:=
\sup_{t\ge 0} e^{2t},|\Phi_t w_0|_m^2,
\qquad
|w_0|*m\le r_0,
\tag{6.7}
]
where (\Phi_t) denotes the semiflow generated by (6.1) on the small ball (B*{r_0}\subset L^2(m)).

Then:

1. (\mathcal L:B_{r_0}\to [0,\infty)) is well-defined.

2. (\mathcal L) is equivalent to the weighted norm:
   [
   |w_0|_m^2\le \mathcal L(w_0)\le K_0^2|w_0|_m^2.
   \tag{6.8}
   ]

3. (\mathcal L) is strictly decreasing along nonstationary trajectories:
   [
   \mathcal L(\Phi_s w_0)\le e^{-2s}\mathcal L(w_0),\qquad s\ge 0.
   \tag{6.9}
   ]
   Equivalently, in upper Dini derivative form,
   [
   D^+\bigl[\mathcal L(\Phi_s w_0)\bigr]\le -2,\mathcal L(\Phi_s w_0).
   \tag{6.10}
   ]

#### Proof

By Theorem 6.1 with (\mu=1),
[
|\Phi_t w_0|_m\le K_0 e^{-t}|w_0|_m,\qquad t\ge 0,
]
whenever (|w_0|_m\le r_0). Hence
[
e^{2t}|\Phi_t w_0|_m^2\le K_0^2|w_0|_m^2,
]
so the supremum in (6.7) is finite. This proves well-definedness.

The lower bound in (6.8) is immediate from the choice (t=0):
[
\mathcal L(w_0)\ge |w_0|_m^2.
]
The upper bound is exactly the previous estimate.

Next, using the semiflow property,
[
\mathcal L(\Phi_s w_0)
======================

# \sup_{t\ge 0} e^{2t}|\Phi_t(\Phi_s w_0)|_m^2

\sup_{t\ge 0} e^{2t}|\Phi_{t+s} w_0|_m^2.
]
Write (u=t+s). Then (u\ge s), and
[
\mathcal L(\Phi_s w_0)
======================

e^{-2s}\sup_{u\ge s} e^{2u}|\Phi_u w_0|*m^2
\le
e^{-2s}\sup*{u\ge 0} e^{2u}|\Phi_u w_0|_m^2
===========================================

e^{-2s}\mathcal L(w_0),
]
which is (6.9). Dividing by (s>0), letting (s\downarrow 0), and using the definition of the upper Dini derivative yields (6.10). ∎

### Theorem 6.4 (closure of the perturbative weighted ancient branch)

For every (m>\tfrac72),
[
\mathcal B^{\mathrm{anc}}_{\mathrm{wf}}(m,r_0)={0}.
\tag{6.11}
]

#### Proof

Let (w\in \mathcal B^{\mathrm{anc}}*{\mathrm{wf}}(m,r_0)), and set
[
M:=\sup*{\tau\in\mathbb R}|w(\tau)|*m\le r_0.
\tag{6.12}
]
Fix (\tau\in\mathbb R), and let (s<\tau). Since (w(s)\in B*{r_0}), Theorem 6.1 applies to the initial datum (w(s)). By uniqueness of mild solutions in (C([0,\infty);L^2(m))), the forward solution launched from (w(s)) is exactly the time-shifted ancient solution:
[
\Phi_{\tau-s}w(s)=w(\tau).
\tag{6.13}
]
Applying Proposition 6.3,
[
\mathcal L(w(\tau))
===================

\mathcal L(\Phi_{\tau-s}w(s))
\le
e^{-2(\tau-s)}\mathcal L(w(s))
\le
K_0^2 e^{-2(\tau-s)}|w(s)|_m^2
\le
K_0^2 M^2 e^{-2(\tau-s)}.
\tag{6.14}
]
Now let (s\to -\infty). Since (\tau) is fixed, the right-hand side tends to (0). Therefore
[
\mathcal L(w(\tau))=0.
\tag{6.15}
]
By the lower coercive bound in (6.8), this implies (|w(\tau)|_m=0). As (\tau\in\mathbb R) was arbitrary, (w\equiv 0). ∎

---

## 6.1. Interpretation and consequences

Theorem 6.4 is the first rigorous realization of the Lyapunov strategy in the branch-reduction program.

First, it validates the general philosophy: on the easiest branch, one can indeed construct a coercive functional that decays along the renormalized dynamics and excludes all nontrivial bounded ancient trajectories. In this perturbative regime, the appropriate functional is not yet the profile-relative Gaussian defect envisioned for the more difficult branches; it is the flow-adapted Lyapunov functional (6.7), built directly from the exponential stability of the small-data semiflow. 

Second, it shows that the weighted / fast-decay branch bifurcates into two regimes. The **small branch** is empty by Theorem 6.4. Therefore any nontrivial weighted ancient obstruction must be **nonperturbative**, and in particular must lie outside the small ball on which Gallay’s decay theorem applies. This sharply separates the truly difficult weighted branch from the perturbative neighborhood of the origin. 

Third, this theorem explains what the next step has to be. If one wants a branch-closing theorem beyond the perturbative regime, the profile (V_*\equiv 0) is no longer sufficient. One must pass to a **profile-relative** Lyapunov functional around a nonzero structured model, or else prove a nonperturbative weighted rigidity theorem. That is precisely where the stronger program in your note begins. In Villani’s language, the present result sits on the coercive side of the coercivity/hypocoercivity dichotomy; the genuinely hard branches will require profile-relative twisted functionals, commutator structure, or modulation equations. ([arXiv][1])

---

## 6.2. Paper-style remark for insertion into the roadmap

A concise version suitable for the roadmap is:

> **Perturbative weighted branch.**
> Let (m>\tfrac72). There exists (r_0>0) such that every bounded ancient solution of the renormalized vorticity equation in (L^2(m)) satisfying (\sup_{\tau\in\mathbb R}|w(\tau)|*m\le r_0) is identically zero.
> The proof uses Gallay’s small-data exponential decay theorem to define a semiflow Lyapunov functional
> [
> \mathcal L(w_0)=\sup*{t\ge0} e^{2t}|\Phi_t w_0|_m^2,
> ]
> which is equivalent to (|w_0|_m^2) on the small ball and satisfies
> [
> \mathcal L(\Phi_s w_0)\le e^{-2s}\mathcal L(w_0).
> ]
> Hence any bounded ancient orbit in the small weighted tube must vanish identically.

That closes the easiest branch cleanly and rigorously, while leaving the nonperturbative profile-relative Lyapunov program for the harder branches.

[1]: https://arxiv.org/pdf/math/0609050?utm_source=chatgpt.com "Hypocoercivity Cédric Villani"

Below is a paper-style section that rigorously implements the next easiest nonzero-profile branch.

It does **not** yet close your full repaired-gauge controlled-swirl singularity branch. What it does close, completely and rigorously, is the **model structured-vortex branch** consisting of bounded ancient solutions that remain in a small weighted tube around the Burgers-vortex family. This is the right next step because Burgers vortices are explicit stationary swirl-bearing solutions of the strained 3D vorticity equation, and Gallay–Maekawa proved their asymptotic stability with respect to general 3D perturbations for every circulation parameter. ([Department of Mathematics][1])

---

## 7. The structured Burgers-vortex branch

### 7.1. Model equation and functional setting

We consider the vorticity formulation of the 3D Navier–Stokes equation with axial strain,
[
\partial_\tau \Omega + (U!\cdot!\nabla)\Omega - (\Omega!\cdot!\nabla)U = L\Omega,
\qquad \nabla!\cdot!\Omega = 0,
\qquad U = K_{3D} * \Omega,
\tag{7.1}
]
where
[
L\Omega = \Delta \Omega - (Mx!\cdot!\nabla)\Omega + M\Omega,
\qquad
M = \operatorname{diag}!\left(-\frac12,-\frac12,1\right).
\tag{7.2}
]
This is the standard Burgers-vortex equation. It admits the explicit stationary family
[
\Omega = \alpha G,
\qquad
G(x)=\begin{pmatrix}0\0\g(x_h)\end{pmatrix},
\qquad
g(x_h)=\frac1{4\pi}e^{-|x_h|^2/4},
\tag{7.3}
]
where (\alpha\in\mathbb R) is the circulation parameter. ([Department of Mathematics][1])

Following Gallay–Maekawa, for (m\in[0,\infty]) define the horizontal weighted space
[
L^2(m)
======

\left{
f\in L^2(\mathbb R^2):
\int_{\mathbb R^2}|f(x_h)|^2 \rho_m(|x_h|^2),dx_h<\infty
\right},
\tag{7.4}
]
with weight
[
\rho_m(r)=
\begin{cases}
1, & m=0,[2mm]
\left(1+\dfrac{r}{4m}\right)^m, & 0<m<\infty,[2mm]
e^{r/4}, & m=\infty.
\end{cases}
\tag{7.5}
]
For (m>1), one has (L^2(m)\hookrightarrow L^1(\mathbb R^2)). Define
[
X_s(m):=BC(\mathbb R_{x_3};L^2(m)),
\qquad
X_{s,0}(m):=BC(\mathbb R_{x_3};L^2_0(m)),
\tag{7.6}
]
where
[
L^2_0(m)
========

\left{
f\in L^2(m):\int_{\mathbb R^2} f(x_h),dx_h=0
\right}.
\tag{7.7}
]
We then set
[
\mathbb X^{\mathrm{full}}(m):=X_s(m)^3,
\qquad
\mathbb X^{\mathrm{mod}}(m):=X_s(m)\times X_s(m)\times X_{s,0}(m),
\tag{7.8}
]
with norm
[
|\omega|_{\mathbb X^{\mathrm{full}}(m)}
=======================================

\sup_{x_3\in\mathbb R}|\omega(\cdot,x_3)|_{L^2(m)^3}.
\tag{7.9}
]
This is exactly the weighted phase space used in the 3D Burgers-vortex stability theory. ([Department of Mathematics][1])

### 7.2. Perturbation equation about a fixed Burgers vortex

Fix (\alpha\in\mathbb R), and write
[
\Omega = \alpha G + \omega.
\tag{7.10}
]
Then (\omega) solves
[
\partial_\tau \omega + (u!\cdot!\nabla)\omega - (\omega!\cdot!\nabla)u
======================================================================

(L-\alpha\Lambda)\omega,
\qquad
\nabla!\cdot!\omega=0,
\tag{7.11}
]
where (u=K_{3D}*\omega) and
[
\Lambda\omega
=============

## (U^G!\cdot!\nabla)\omega

(\omega!\cdot!\nabla)U^G
+
(u!\cdot!\nabla)G
-----------------

(G!\cdot!\nabla)u.
\tag{7.12}
]
Gallay–Maekawa prove the following nonlinear stability theorem.

### Theorem 7.1 (Gallay–Maekawa)

Let (m>2) and (\alpha\in\mathbb R). Then there exist constants
[
\delta_{\alpha,m}>0,
\qquad
C_{\alpha,m}\ge 1,
\tag{7.13}
]
such that, for any divergence-free initial datum
[
\omega_0\in \mathbb X^{\mathrm{mod}}(m),
\qquad
|\omega_0|*{\mathbb X^{\mathrm{mod}}(m)}\le \delta*{\alpha,m},
\tag{7.14}
]
equation (7.11) has a unique global mild solution
[
\omega\in L^\infty(\mathbb R_+;\mathbb X^{\mathrm{mod}}(m))
\cap
C([0,\infty);\mathbb X^{\mathrm{mod}}*{\mathrm{loc}}(m)),
\tag{7.15}
]
and
[
|\omega(\tau)|*{\mathbb X^{\mathrm{mod}}(m)}
\le
C_{\alpha,m} e^{-\tau/2},|\omega_0|_{\mathbb X^{\mathrm{mod}}(m)},
\qquad \tau\ge 0.
\tag{7.16}
]
Equivalently, the fixed Burgers vortex (\alpha G) is asymptotically stable in (\mathbb X^{\mathrm{mod}}(m)). ([Department of Mathematics][1])

This is already enough to build the profile-relative Lyapunov functional.

### Proposition 7.2 (fixed-profile Lyapunov functional)

Fix (m>2) and (\alpha\in\mathbb R). Let (\Phi_\tau^\alpha) denote the local semiflow of (7.11) on the ball
[
B_{\alpha,m}:={\omega_0\in \mathbb X^{\mathrm{mod}}(m):\ |\omega_0|*{\mathbb X^{\mathrm{mod}}(m)}\le \delta*{\alpha,m}}.
\tag{7.17}
]
Define
[
\mathcal L_{\alpha,m}(\omega_0)
:=
\sup_{\tau\ge 0}
e^\tau |\Phi_\tau^\alpha \omega_0|_{\mathbb X^{\mathrm{mod}}(m)}^2.
\tag{7.18}
]
Then:

1. (\mathcal L_{\alpha,m}) is well-defined on (B_{\alpha,m}).
2. It is equivalent to the square of the weighted norm:
   [
   |\omega_0|*{\mathbb X^{\mathrm{mod}}(m)}^2
   \le
   \mathcal L*{\alpha,m}(\omega_0)
   \le
   C_{\alpha,m}^2
   |\omega_0|_{\mathbb X^{\mathrm{mod}}(m)}^2.
   \tag{7.19}
   ]
3. It is strictly contracting along the semiflow:
   [
   \mathcal L_{\alpha,m}(\Phi_s^\alpha \omega_0)
   \le
   e^{-s}\mathcal L_{\alpha,m}(\omega_0),
   \qquad s\ge 0.
   \tag{7.20}
   ]

#### Proof

By Theorem 7.1,
[
|\Phi_\tau^\alpha \omega_0|*{\mathbb X^{\mathrm{mod}}(m)}
\le
C*{\alpha,m}e^{-\tau/2}|\omega_0|*{\mathbb X^{\mathrm{mod}}(m)},
\qquad \tau\ge 0,
]
hence
[
e^\tau |\Phi*\tau^\alpha \omega_0|*{\mathbb X^{\mathrm{mod}}(m)}^2
\le
C*{\alpha,m}^2 |\omega_0|_{\mathbb X^{\mathrm{mod}}(m)}^2,
]
so (7.18) is finite and the upper bound in (7.19) follows. The lower bound comes from (\tau=0).

For the semigroup decay, use the semiflow property:
[
\mathcal L_{\alpha,m}(\Phi_s^\alpha \omega_0)
=============================================

\sup_{\tau\ge0}
e^\tau |\Phi_{\tau+s}^\alpha \omega_0|^2
========================================

e^{-s}
\sup_{u\ge s}
e^u |\Phi_u^\alpha \omega_0|^2
\le
e^{-s}\mathcal L_{\alpha,m}(\omega_0).
]
This proves (7.20). ∎

That is the exact Lyapunov functional you wanted: weighted, profile-relative, and adapted to the nonzero stationary branch.

---

## 7.3. The modulation parameter: conserved circulation

The only neutral family parameter in the Burgers-vortex branch is the circulation (\alpha). In the full space (\mathbb X^{\mathrm{full}}(m)), this parameter is extracted canonically by horizontal integration of the vertical vorticity.

### Lemma 7.3 (circulation map)

Let (m>1), and let (\Omega\in \mathbb X^{\mathrm{full}}(m)) be divergence-free. Then the quantity
[
\Gamma[\Omega](x_3)
:=
\int_{\mathbb R^2}\Omega_3(x_h,x_3),dx_h
\tag{7.21}
]
is well-defined and independent of (x_3). Define
[
\Gamma[\Omega]
:=
\int_{\mathbb R^2}\Omega_3(x_h,x_3),dx_h.
\tag{7.22}
]
Then
[
Q\Omega := \Omega - \Gamma[\Omega],G
\in
\mathbb X^{\mathrm{mod}}(m).
\tag{7.23}
]

#### Proof

Since (m>1), (L^2(m)\hookrightarrow L^1(\mathbb R^2)), so the integral is well-defined. By (\nabla!\cdot!\Omega=0),
[
\partial_{x_3}\Gamma[\Omega](x_3)
=================================

# \int_{\mathbb R^2}\partial_{x_3}\Omega_3,dx_h

-\int_{\mathbb R^2}\nabla_h!\cdot!\Omega_h,dx_h
=0,
]
using the horizontal decay. Thus (\Gamma[\Omega](x_3)) is independent of (x_3). Since (\int_{\mathbb R^2} G_3,dx_h=1), the third component of (Q\Omega) has zero horizontal mean, hence (Q\Omega\in\mathbb X^{\mathrm{mod}}(m)). ∎

This is the modulation map used below. It is exactly the “profile-family coordinate” that must be removed before coercivity appears. Gallay–Maekawa use the same decomposition when passing from the full perturbation space (\mathbb X^{\mathrm{full}}(m)) to the fixed-circulation subspace (\mathbb X^{\mathrm{mod}}(m)). ([Department of Mathematics][1])

### Lemma 7.4 (conservation of circulation and reduction to fixed circulation)

Let (m>1), and let
[
\Omega\in L^\infty_{\mathrm{loc}}(\mathbb R;\mathbb X^{\mathrm{full}}(m))
\cap C_{\mathrm{loc}}(\mathbb R;\mathbb X^{\mathrm{full}}_{\mathrm{loc}}(m))
\tag{7.24}
]
be a divergence-free mild solution of the full Burgers-vortex equation (7.1). Then:

1. (\Gamma[\Omega(\tau)]) is independent of (\tau).
2. If (\bar\alpha:=\Gamma[\Omega(\tau)]), then
   [
   \omega(\tau):=\Omega(\tau)-\bar\alpha G
   \tag{7.25}
   ]
   belongs to (\mathbb X^{\mathrm{mod}}(m)) for every (\tau), and solves the perturbation equation (7.11) with parameter (\alpha=\bar\alpha).

#### Proof

By Lemma 7.3, (\Gamma[\Omega(\tau)]) is well-defined and independent of (x_3). Differentiate in (\tau):
[
\frac{d}{d\tau}\Gamma[\Omega(\tau)]
===================================

\int_{\mathbb R^2}\partial_\tau \Omega_3,dx_h.
\tag{7.26}
]
Using (7.1),
[
\partial_\tau \Omega_3
======================

(L\Omega)_3 - \big((U!\cdot!\nabla)\Omega - (\Omega!\cdot!\nabla)U\big)_3.
\tag{7.27}
]

For the nonlinear term, use the vector identity
[
(U!\cdot!\nabla)\Omega - (\Omega!\cdot!\nabla)U
===============================================

-\nabla\times (U\times \Omega),
\tag{7.28}
]
valid because (\nabla!\cdot!U=\nabla!\cdot!\Omega=0). Hence the third component is a pure horizontal divergence:
[
\big((U!\cdot!\nabla)\Omega - (\Omega!\cdot!\nabla)U\big)_3
===========================================================

-\partial_{x_1}(U\times\Omega)*2 + \partial*{x_2}(U\times\Omega)_1,
\tag{7.29}
]
so its horizontal integral vanishes.

For the linear part, from (M=\operatorname{diag}(-1/2,-1/2,1)),
[
(L\Omega)_3
===========

\Delta_h \Omega_3 + \partial_{x_3}^2\Omega_3

* \frac12 x_h!\cdot!\nabla_h \Omega_3

- x_3\partial_{x_3}\Omega_3

* \Omega_3.
  \tag{7.30}
  ]
  Integrating over (x_h), the horizontal Laplacian gives zero, while
  [
  \int_{\mathbb R^2}\frac12 x_h!\cdot!\nabla_h \Omega_3,dx_h
  =
  -\int_{\mathbb R^2}\Omega_3,dx_h,
  \tag{7.31}
  ]
  because (\nabla_h!\cdot(x_h\Omega_3)=2\Omega_3+x_h!\cdot!\nabla_h\Omega_3). Therefore
  [
  \int_{\mathbb R^2}(L\Omega)*3,dx_h
  =
  \partial*{x_3}^2\Gamma[\Omega]-x_3\partial_{x_3}\Gamma[\Omega].
  \tag{7.32}
  ]
  By Lemma 7.3, (\partial_{x_3}\Gamma[\Omega]=0), so the right-hand side vanishes. Hence
  [
  \frac{d}{d\tau}\Gamma[\Omega(\tau)]=0.
  \tag{7.33}
  ]

Now set (\bar\alpha=\Gamma[\Omega(\tau)]). Since (\bar\alpha G) is a stationary solution of (7.1), subtraction yields that (\omega=\Omega-\bar\alpha G) satisfies (7.11) with parameter (\bar\alpha). By Lemma 7.3, (\omega(\tau)\in\mathbb X^{\mathrm{mod}}(m)). ∎

---

## 7.4. The family Lyapunov functional

We now pass from a fixed Burgers vortex to the entire family.

### Definition 7.5 (family Lyapunov functional)

Fix (m>2). For any divergence-free (\Omega\in \mathbb X^{\mathrm{full}}(m)), define
[
\bar\alpha[\Omega]:=\Gamma[\Omega],
\qquad
Q\Omega:=\Omega-\bar\alpha[\Omega],G.
\tag{7.34}
]
Whenever
[
|Q\Omega|*{\mathbb X^{\mathrm{mod}}(m)}
\le
\delta*{\bar\alpha[\Omega],m},
\tag{7.35}
]
define
[
\mathfrak L_m(\Omega)
:=
\mathcal L_{\bar\alpha[\Omega],m}(Q\Omega).
\tag{7.36}
]

This is the profile-relative Lyapunov defect for the Burgers-vortex family: it measures the weighted distance to the modulated structured profile (\bar\alpha[\Omega]G).

### Proposition 7.6 (monotonicity for the family)

Let (\Omega(\tau)) be a forward mild solution of (7.1) such that
[
|Q\Omega(\tau)|*{\mathbb X^{\mathrm{mod}}(m)}
\le
\delta*{\bar\alpha,m}
\quad\text{for all }\tau\ge0,
\qquad
\bar\alpha:=\Gamma[\Omega(0)].
\tag{7.37}
]
Then
[
\mathfrak L_m(\Omega(s))
\le
e^{-s}\mathfrak L_m(\Omega(0)),
\qquad s\ge0.
\tag{7.38}
]

#### Proof

By Lemma 7.4, (\Gamma[\Omega(s)]=\bar\alpha) for all (s), and (Q\Omega(s)) solves the perturbation equation (7.11) with parameter (\bar\alpha). Therefore
[
Q\Omega(s)=\Phi_s^{\bar\alpha}(Q\Omega(0)).
\tag{7.39}
]
Hence, using Proposition 7.2,
[
\mathfrak L_m(\Omega(s))
========================

# \mathcal L_{\bar\alpha,m}(Q\Omega(s))

\mathcal L_{\bar\alpha,m}(\Phi_s^{\bar\alpha}(Q\Omega(0)))
\le
e^{-s}\mathcal L_{\bar\alpha,m}(Q\Omega(0))
===========================================

e^{-s}\mathfrak L_m(\Omega(0)).
]
∎

This is the exact family-level Lyapunov principle that your roadmap wanted: the neutral profile coordinate is modulated out, and the remaining defect decays exponentially.

---

## 7.5. Ancient rigidity and closure of the model branch

We now close the branch.

### Definition 7.7 (structured Burgers-vortex ancient branch)

Fix (m>2). Define
[
\mathcal B_{\mathrm{BV}}^{\mathrm{anc}}(m)
]
to be the set of all divergence-free ancient mild solutions
[
\Omega\in L^\infty_{\mathrm{loc}}(\mathbb R;\mathbb X^{\mathrm{full}}(m))
\cap C_{\mathrm{loc}}(\mathbb R;\mathbb X^{\mathrm{full}}*{\mathrm{loc}}(m))
\tag{7.40}
]
of (7.1) such that, writing (\bar\alpha=\Gamma[\Omega(\tau)]),
[
\sup*{\tau\in\mathbb R}
|\Omega(\tau)-\bar\alpha G|*{\mathbb X^{\mathrm{mod}}(m)}
\le
\delta*{\bar\alpha,m}.
\tag{7.41}
]

### Theorem 7.8 (closure of the structured Burgers-vortex branch)

Let (m>2). If (\Omega\in \mathcal B_{\mathrm{BV}}^{\mathrm{anc}}(m)), then
[
\Omega(\tau)\equiv \bar\alpha G
\qquad\text{for all }\tau\in\mathbb R,
\tag{7.42}
]
where (\bar\alpha=\Gamma[\Omega]).

In particular, the model controlled-swirl ancient branch reduces exactly to the Burgers-vortex family.

#### Proof

Let (\bar\alpha=\Gamma[\Omega]), which is independent of (\tau) by Lemma 7.4, and set
[
\omega(\tau):=\Omega(\tau)-\bar\alpha G.
\tag{7.43}
]
Then (\omega(\tau)\in \mathbb X^{\mathrm{mod}}(m)) for every (\tau), solves (7.11) with parameter (\bar\alpha), and
[
M:=\sup_{\tau\in\mathbb R}|\omega(\tau)|*{\mathbb X^{\mathrm{mod}}(m)}
\le \delta*{\bar\alpha,m}.
\tag{7.44}
]

Fix (\tau\in\mathbb R), and let (s<\tau). Since (\omega) is an ancient solution of the fixed-circulation perturbation equation,
[
\omega(\tau)=\Phi_{\tau-s}^{\bar\alpha}\omega(s).
\tag{7.45}
]
Applying Proposition 7.2,
[
\mathcal L_{\bar\alpha,m}(\omega(\tau))
=======================================

\mathcal L_{\bar\alpha,m}(\Phi_{\tau-s}^{\bar\alpha}\omega(s))
\le
e^{-(\tau-s)}\mathcal L_{\bar\alpha,m}(\omega(s))
\le
e^{-(\tau-s)} C_{\bar\alpha,m}^2 |\omega(s)|*{\mathbb X^{\mathrm{mod}}(m)}^2
\le
C*{\bar\alpha,m}^2 M^2 e^{-(\tau-s)}.
\tag{7.46}
]
Letting (s\to-\infty), the right-hand side tends to (0), so
[
\mathcal L_{\bar\alpha,m}(\omega(\tau))=0.
\tag{7.47}
]
By the coercive lower bound in (7.19), (\omega(\tau)=0). Since (\tau) was arbitrary, (\omega\equiv0), i.e.
[
\Omega(\tau)\equiv \bar\alpha G.
]
∎

---

## 7.6. Conditional import to the repaired-gauge singularity program

Theorems 7.6 and 7.8 are fully rigorous, but they close the **model** branch governed by the exact Burgers-vortex equation. To use them in your repaired-gauge singularity roadmap, one needs one further reduction theorem.

### Theorem 7.9 (conditional import to the repaired-gauge branch)

Assume there exist:

1. a repaired-gauge branch (\mathcal B_{\mathrm{swirl}}^{\mathrm{anc}}) of bounded ancient solutions of your renormalized singularity equation;

2. a transformation
   [
   \mathcal T:\mathcal B_{\mathrm{swirl}}^{\mathrm{anc}}
   \to
   L^\infty_{\mathrm{loc}}(\mathbb R;\mathbb X^{\mathrm{full}}(m))
   \cap C_{\mathrm{loc}}(\mathbb R;\mathbb X^{\mathrm{full}}*{\mathrm{loc}}(m)),
   \tag{7.48}
   ]
   with (m>2), such that for every (V\in \mathcal B*{\mathrm{swirl}}^{\mathrm{anc}}), the image
   [
   \Omega=\mathcal T[V]
   \tag{7.49}
   ]
   is a mild ancient solution of the Burgers-vortex equation (7.1);

3. a branch defect (\mathfrak d[V(\tau)]\ge0) and a constant (C_*>0) such that, with (\bar\alpha=\Gamma[\Omega]),
   [
   |\Omega(\tau)-\bar\alpha G|*{\mathbb X^{\mathrm{mod}}(m)}
   \le
   C**,\mathfrak d[V(\tau)]
   \qquad\text{for all }\tau\in\mathbb R.
   \tag{7.50}
   ]

If, in addition,
[
\sup_{\tau\in\mathbb R}\mathfrak d[V(\tau)]
\le
\frac{\delta_{\bar\alpha,m}}{C_*},
\tag{7.51}
]
then
[
\mathcal T[V](\tau)\equiv \bar\alpha G
\qquad\text{for all }\tau\in\mathbb R.
\tag{7.52}
]
Equivalently, (V) is exactly the pullback of a Burgers vortex under the transformation (\mathcal T).

#### Proof

By (7.50) and (7.51), (\Omega=\mathcal T[V]) belongs to (\mathcal B_{\mathrm{BV}}^{\mathrm{anc}}(m)). The conclusion follows from Theorem 7.8. ∎

This theorem isolates the only remaining external step: proving that the repaired-gauge controlled-swirl branch reduces, after freezing center/scale/axis and passing to the correct local coordinates, to the Burgers-vortex model with small residual in (\mathbb X^{\mathrm{mod}}(m)).

---

## 7.7. Remarks

**Remark 7.10.**
The rigorous content obtained here is stronger than a mere “good-window decay estimate.” It gives an actual **ancient-rigidity theorem** for the next easiest structured nonzero-profile branch:
[
\text{small weighted ancient branch near the family } {\alpha G}_{\alpha\in\mathbb R}
\quad\Longrightarrow\quad
\text{exact Burgers vortex}.
\tag{7.53}
]

**Remark 7.11.**
The family parameter (\alpha) is not an ad hoc modulation variable. It is the conserved circulation
[
\alpha=\Gamma[\Omega]=\int_{\mathbb R^2}\Omega_3(x_h,x_3),dx_h,
]
which is exactly the canonical coordinate on the Burgers-vortex family. ([Department of Mathematics][1])

**Remark 7.12.**
Gallay–Maekawa also note that, by a perturbation argument, asymmetric Burgers vortices are stable with respect to three-dimensional perturbations provided the asymmetry parameter is sufficiently small depending on the circulation. Thus the same Lyapunov construction extends, with no essential change, to a small-(\lambda) asymmetric structured-vortex family. ([Department of Mathematics][1])

**Remark 7.13.**
What remains open is not the Lyapunov machinery itself. The machinery is complete in the Burgers-vortex model. The open step is the **normal-form reduction** from your repaired-gauge singularity equation to that model branch.

---

If you want, I’ll turn this into the next section of your manuscript with your existing notation conventions, numbering, and cross-references to the repaired-gauge chapter.

[1]: https://web.ma.utexas.edu/mp_arc/c/10/10-35.pdf "https://web.ma.utexas.edu/mp_arc/c/10/10-35.pdf"


Yes — but only for **specific, honest subbranches**.

The next genuinely closable branch is not the full controlled-swirl axisymmetric branch. What can be closed rigorously is the following **axisymmetric ancient-solution subbranch** of the actual unforced 3D Navier–Stokes equation:

* no-swirl,
* pointwise scale-invariant control (r|u|\lesssim 1),
* finite-(L^p) control of the swirl variable (\Gamma=r u_\theta),
* or axial periodicity together with bounded (\Gamma).

Those are exact Liouville closures, stronger than the Lyapunov machinery.

## 8. Exact closure of several axisymmetric subbranches

We work with ancient mild solutions of
[
\partial_t u + (u\cdot\nabla)u + \nabla p = \Delta u,
\qquad \nabla\cdot u = 0,
\qquad (x,t)\in \mathbb R^3\times(-\infty,0).
\tag{8.1}
]

In cylindrical coordinates ((r,\theta,z)), write
[
u = u_r e_r + u_\theta e_\theta + u_z e_z,
\qquad
\Gamma := r u_\theta.
\tag{8.2}
]

### 8.1. Galilean triviality

The right notion of “empty branch” here is **emptiness modulo Galilean symmetry**, because constant vector fields are dynamically trivial.

### Lemma 8.1 (Galilean reduction)

If ((u,p)) solves (8.1), then for each constant (c\in\mathbb R^3),
[
\widetilde u(x,t):=u(x+ct,t)-c,
\qquad
\widetilde p(x,t):=p(x+ct,t)
\tag{8.3}
]
also solves (8.1). In particular, a constant solution (u\equiv c) is Galilean-equivalent to (0).

#### Proof

A direct computation gives
[
\partial_t \widetilde u = (\partial_t u + c\cdot\nabla u)(x+ct,t),\qquad
(\widetilde u\cdot\nabla)\widetilde u = ((u-c)\cdot\nabla u)(x+ct,t),
]
hence
[
\partial_t \widetilde u + (\widetilde u\cdot\nabla)\widetilde u
===============================================================

(\partial_t u + (u\cdot\nabla)u)(x+ct,t).
]
Also (\Delta \widetilde u = (\Delta u)(x+ct,t)), (\nabla \widetilde p=(\nabla p)(x+ct,t)), and (\nabla\cdot\widetilde u=0). Thus ((\widetilde u,\widetilde p)) satisfies (8.1). If (u\equiv c), then (\widetilde u\equiv 0). ∎

So every theorem below should be read as giving either (u\equiv 0) or (u) constant, hence branch-trivial after Galilean gauge fixing.

---

### 8.2. Swirl-free axisymmetric subbranch

### Theorem 8.2 (swirl-free ancient branch)

Let (u) be a bounded ancient mild solution of (8.1), axisymmetric and swirl-free:
[
u_\theta \equiv 0.
\tag{8.4}
]
Then (u) is Galilean-trivial. More precisely,
[
u(x,t)=(0,0,c)
\quad\text{for some }c\in\mathbb R.
\tag{8.5}
]

#### Proof

Koch–Nadirashvili–Seregin–Šverák prove that any bounded weak ancient axisymmetric no-swirl solution has the form
[
u(x,t)=(0,0,b_3(t)).
]
They also note that a bounded ancient mild solution of the form (u(x,t)=b(t)) must be constant in time. Therefore (u=(0,0,c)). By Lemma 8.1 this is Galilean-equivalent to (0). 

---

### 8.3. Pointwise scale-invariant axisymmetric branch

This is the cleanest genuinely swirl-bearing closure.

### Theorem 8.3 (pointwise controlled axisymmetric branch)

Let (u) be a bounded ancient mild axisymmetric solution of (8.1) satisfying
[
|u(x,t)| \le \frac{C}{r}
\qquad\text{in }\mathbb R^3\times(-\infty,0).
\tag{8.6}
]
Then
[
u\equiv 0.
\tag{8.7}
]

#### Proof

This is exactly Theorem 5.3 of Koch–Nadirashvili–Seregin–Šverák. Their result is stated for bounded weak ancient axisymmetric solutions, so it applies a fortiori to bounded ancient mild solutions. 

This is already a rigorous closure of a nontrivial scale-invariant axisymmetric subbranch.

---

### 8.4. (L^p)-controlled swirl branch

Now we pass from a pointwise (r^{-1}) bound to an integrability condition on the scale-invariant swirl quantity (\Gamma=r u_\theta).

### Theorem 8.4 ((\Gamma)-integrable axisymmetric branch)

Let (u) be a bounded ancient mild axisymmetric solution of (8.1). Assume that for some (1\le p<\infty),
[
\Gamma = r u_\theta \in L_t^\infty L_x^p\bigl(\mathbb R^3\times(-\infty,0)\bigr).
\tag{8.8}
]
Then (u) is Galilean-trivial. Equivalently, after a Galilean transform,
[
u\equiv 0.
\tag{8.9}
]

#### Proof

If (u_\theta\equiv 0), then Theorem 8.2 applies.

If (u_\theta\not\equiv 0), then Lei–Zhang–Zhao prove that any bounded ancient mild axisymmetric solution with
[
\Gamma\in L_t^\infty L_x^p,\qquad 1\le p<\infty,
]
must be a constant vector field. By Lemma 8.1, any constant vector field is Galilean-equivalent to (0). 

---

### 8.5. Periodic axial branch with bounded swirl

The (L^p) condition can be replaced by periodicity in the axial variable.

### Theorem 8.5 (periodic axisymmetric branch)

Let (u) be a bounded ancient mild axisymmetric solution of (8.1). Assume

1. (u) is periodic in (z) with some period (Z_0>0), and
2. (\Gamma=r u_\theta) is bounded:
   [
   \Gamma\in L^\infty\bigl(\mathbb R^3\times(-\infty,0)\bigr).
   \tag{8.10}
   ]

Then
[
u(x,t)=c,e_z
\quad\text{for some }c\in\mathbb R.
\tag{8.11}
]
Hence (u) is Galilean-trivial.

#### Proof

This is exactly Theorem 1.1 of Lei–Ren–Zhang. They prove that a bounded ancient mild axisymmetric solution with bounded (\Gamma), periodic in (z), must equal (c e_z). Apply Lemma 8.1 with (c e_z). 

---

## 8.6. Import to the repaired-gauge branch structure

These theorems immediately yield a rigorous branch-reduction statement in your program.

### Corollary 8.6 (closure of axisymmetric subbranches)

Let (\mathcal B_{\mathrm{ax}}) be any repaired-gauge branch consisting of bounded ancient mild axisymmetric solutions. Suppose every (u\in\mathcal B_{\mathrm{ax}}) satisfies at least one of the following:

1. (u_\theta\equiv 0),
2. (|u(x,t)|\le C/r),
3. (r u_\theta \in L_t^\infty L_x^p) for some (1\le p<\infty),
4. (u) is periodic in (z) and (r u_\theta\in L^\infty).

Then (\mathcal B_{\mathrm{ax}}) is trivial modulo Galilean symmetry. If the repaired gauge fixes the Galilean drift, then
[
\mathcal B_{\mathrm{ax}}={0}.
\tag{8.12}
]

#### Proof

Apply Theorems 8.2–8.5 branchwise, and then use Lemma 8.1. ∎

---

## 8.7. What this actually closes

This does **not** close the full controlled-swirl axisymmetric branch.

It **does** rigorously close the following genuine subbranches of it:

* the no-swirl subbranch,
* the pointwise scale-invariant subbranch (r|u|\lesssim 1),
* the (L_t^\infty L_x^p)-controlled swirl subbranch,
* the (z)-periodic bounded-swirl subbranch.

That is already stronger than the previous Burgers-vortex model closure because it lives in the actual unforced 3D Navier–Stokes equation. The remaining open part of the axisymmetric branch is the truly hard one: bounded ancient axisymmetric solutions with nontrivial swirl and **without** any of the extra side conditions above. The 2017 and 2019 papers explicitly frame their results as partial progress toward the broader bounded ancient axisymmetric Liouville conjecture. 

There is one further profile-only closure available: if you split off the purely stationary ((-1))-homogeneous self-similar branch, then any smooth self-similar stationary solution in (\mathbb R^3\setminus{0}) is a Landau solution, and Kwon–Tsai show that in the axisymmetric DSS setting near Landau, adding swirl does not create a new local bifurcation branch. That is a useful classification fact, but it is a profile classification, not yet an ancient-solution branch closure. 

The next honest frontier after this is the **nonperiodic bounded-(\Gamma)** axisymmetric branch. That is where a genuinely new Lyapunov/modulation argument would still have real room to contribute.


Yes — but only after a **substantial change of variables and of the Lyapunov functional itself**.

The honest conclusion is:

Your current local defect functional on (V-V_*) in renormalized velocity variables is **not the right first object** for the open axisymmetric branch. The mathematically natural first Lyapunov layer is a **scalar, profile-relative defect for**
[
\Gamma:=r,u_\theta,
]
and only after that should one add a second coercive layer for the vorticity variables
[
J:=\frac{\omega_r}{r},\qquad \Omega:=\frac{\omega_\theta}{r}.
]
That is exactly where the existing axisymmetric theory already concentrates: (\Gamma) satisfies a maximum-principle equation, the periodic and (L^p)-Liouville theorems are driven by (\Gamma), and the Lei–Zhang criticality theory treats ((J,\Omega,\omega_3)) as the right closed system. 

What follows is the rigorous part, then the realistic frontier.

## 1. Why the present Lyapunov ansatz needs to change

For bounded ancient axisymmetric solutions, the hard unresolved case is the one where (\Gamma=r u_\theta) is bounded but **does not decay at infinity** and there is **no periodicity in (z)**. The 2019 periodic paper makes this explicit: periodicity is used to recover missing control on (v_r,v_z), and their weighted-energy argument is built directly on (\Gamma), not on the full velocity defect. The 2022 review states that the “remaining case” is precisely when (\Gamma) does not decay near infinity. ([arXiv][1])

So the right modification is:

1. first flatten (\Gamma) toward a constant profile (c),
2. then use the closed (J,\Omega,\omega_3) system to recover full rigidity.

This is not just taste. It is forced by the PDE structure.

## 2. The exact scalar defect identity

For axisymmetric Navier–Stokes, with
[
b:=u_r e_r+u_z e_z,
]
Lei–Zhang record the exact equation
[
\partial_t \Gamma + b\cdot \nabla \Gamma + \frac{2}{r}\partial_r \Gamma = \Delta \Gamma,
\qquad \Gamma=r u_\theta,
]
and note that (\Gamma) enjoys a maximum principle. 

Fix any constant (c\in\mathbb R), and set
[
W:=\Gamma-c.
]
Then (W) satisfies the same equation.

Now let (\phi=\phi(r,z,t)\ge 0) be a smooth compactly supported axisymmetric test weight. A direct integration by parts in cylindrical variables gives the exact identity
[
\frac12\frac{d}{dt}\int_{\mathbb R^3} W^2 \phi,dx
+
\int_{\mathbb R^3} |\nabla W|^2 \phi,dx
=======================================

\frac12\int_{\mathbb R^3} W^2
\Bigl(
\partial_t\phi+\Delta\phi+b\cdot\nabla\phi+\frac{2}{r}\partial_r\phi
\Bigr),dx.
\tag{2.1}
]

This is the axisymmetric analogue of your local profile-relative Lyapunov identity. It is fully rigorous for smooth solutions, and for bounded ancient mild solutions one gets it by the usual approximation/localization procedure because bounded mild axisymmetric solutions are smooth away from the axis and (\Gamma) is bounded. The important point is that the dissipation term is **exactly**
[
\mathcal D_\phi[W]
==================

\int |\nabla W|^2 \phi,dx,
]
with no vortex-stretching remainder at this scalar level. That is the first big gain. 

So the first serious modification of your current Lyapunov program is:

[
\boxed{
\mathcal L_{\Gamma,\phi,c}(t):=\int (\Gamma-c)^2 \phi,dx
}
]
rather than a direct local (L^2+H^1) defect on the full velocity.

## 3. The right window is not an isotropic Gaussian

Your current note uses Gaussian windows centered at a moving core. For the axisymmetric branch, the natural window is **anisotropic**:

[
\phi_{R,Z,\zeta}(r,z,t)=\lambda_R(r),\eta_Z(z-\zeta(t)).
]

Here (\lambda_R) is radial, (\eta_Z) is an axial cutoff, and (\zeta(t)) is an axial modulation.

The key observation is that the singular diffusion term (\frac{2}{r}\partial_r) can be canceled exactly if (\lambda_R) is chosen to be approximately **5-dimensional harmonic**:
[
\lambda_R''+\frac{3}{r}\lambda_R'\approx 0.
]
This is not an accident. The operator
[
\partial_r^2+\frac1r\partial_r+\frac2r\partial_r
================================================

\partial_r^2+\frac3r\partial_r
]
is the radial Laplacian in four transverse dimensions. So your radial part should be adapted to that geometry, not to the ordinary 3D Gaussian. The 2019 periodic paper already uses a special radial weight and explicitly says the weight is the new ingredient in the nonperiodic auxiliary theorem. ([arXiv][1])

If one inserts (\phi_{R,Z,\zeta}) into (2.1), one gets
[
\frac12\frac{d}{dt}\mathcal L_{R,Z,c}(t)
+\mathcal D_{R,Z,c}(t)
======================

\frac12\int W^2,\mathcal E_{R,Z,\zeta}[u],dx,
\tag{3.1}
]
where
[
\mathcal L_{R,Z,c}(t):=\int W^2 \lambda_R\eta_Z,dx,
\qquad
\mathcal D_{R,Z,c}(t):=\int |\nabla W|^2 \lambda_R\eta_Z,dx,
]
and the error density is
[
\mathcal E_{R,Z,\zeta}[u]
=========================

\eta_Z\Bigl(\lambda_R''+\frac{3}{r}\lambda_R' + u_r\lambda_R'\Bigr)
+
\lambda_R\Bigl(\eta_Z'' + (u_z-\dot\zeta)\eta_Z'\Bigr).
\tag{3.2}
]

This is the crucial structural formula.

It says that after the correct axisymmetric modification, the Lyapunov failure modes are **only**:

* radial shell transport through (u_r\lambda_R'),
* axial transport mismatch through ((u_z-\dot\zeta)\eta_Z'),
* and small geometric cutoff errors (\lambda_R''+\frac{3}{r}\lambda_R'), (\eta_Z'').

That is much sharper than the original velocity-space picture.

## 4. Why this is already useful

This formula explains rigorously why the known special cases work.

In the periodic-in-(z) case, the missing large-scale axial control is replaced by periodicity; the 2019 paper explicitly says periodicity is what overcomes the lack of critical estimates for (v_r,v_z), and Section 4 uses a weighted energy method for (\Gamma) with a special weight. ([arXiv][1])

In the bounded-(\Gamma) plus stream-function-BMO setting, Lei–Zhang prove a Liouville theorem; their mechanism is again to get enough control on the drift (b) to run a (\Gamma)-based argument. ([arXiv][2])

In the (L^\infty_tL^p_x) branch for (\Gamma), Lei–Zhang–Zhao prove constancy for every (1\le p<\infty). That branch is already closed, but your scalar defect is exactly the right object there too. ([arXiv][3])

And the recent 2026 partial-Type-I result by Qi S. Zhang is especially relevant conceptually: it treats (\Gamma) through a **one-dimensional drift-diffusion modulus** and shows that a one-sided radial control can substitute for full type-I bounds. That is strong evidence that a drift-adapted Lyapunov weight is the correct modification of your approach. ([arXiv][4])

So this is not just a new notation. It isolates exactly the transport channels that the frontier depends on.

## 5. The second Lyapunov layer: ((J,\Omega))

Once (\Gamma) is nearly flattened in a window, the next object is the closed vorticity system
[
J=\frac{\omega_r}{r},\qquad \Omega=\frac{\omega_\theta}{r},
]
for which the review records the exact equations
[
\begin{cases}
\Delta J-(b\cdot\nabla)J+\dfrac{2}{r}\partial_r J + (\omega_r\partial_r+\omega_3\partial_{x_3})\dfrac{v_r}{r}-\partial_t J=0,[1ex]
\Delta \Omega-(b\cdot\nabla)\Omega+\dfrac{2}{r}\partial_r \Omega - 2\dfrac{v_\theta}{r}J-\partial_t\Omega=0,
\end{cases}
]
and explains that, viewed as a closed system, the vortex-stretching terms are **critical rather than supercritical**. It also records the key estimate
[
|\nabla(v_r/r)|_2 \lesssim |\Omega|_2.
]
([Global Science Press][5])

This means your second-layer Lyapunov functional should be something like
[
\mathcal M_{\phi}(t)
====================

\int (J^2+\Omega^2)\phi,dx
+
\varepsilon\int \omega_3^2\phi,dx,
]
localized by the **same** weight (\phi_{R,Z,\zeta}). The criticality statement from Lei–Zhang is exactly what you need to believe that, once the first scalar layer has made (W=\Gamma-c) small, the ((J,\Omega)) energy becomes coercive modulo the same shell errors. ([Global Science Press][5])

So the right modified program is genuinely **two-tier**:

1. flatten (\Gamma) toward a constant in moving windows,
2. then use localized (J)–(\Omega) coercivity to eliminate the remaining axisymmetric dynamics.

That is a real contribution path. It is much closer to the actual PDE geometry than a single velocity-space defect functional.

## 6. What this can prove now, and what it cannot

What is rigorous now:

* the exact scalar defect identity (2.1),
* the drift-adapted anisotropic window formula (3.2),
* and the identification of the second critical coercive layer through ((J,\Omega)). 

What is **not** yet proved:

* that the shell terms in (3.2) can be made small in the full nonperiodic bounded-(\Gamma) ancient branch.

That is the actual frontier.

So the Lyapunov program does contribute, but in a narrower and more precise way than the original draft suggested: it converts the open branch into a concrete shell-control problem.

## 7. The theorem target I would pursue

The most realistic next theorem is not the full Liouville conjecture. It is a branch-reduction theorem of this form:

Let (u) be a bounded ancient axisymmetric mild solution with bounded (\Gamma). Suppose there exist a constant (c), a modulation (\zeta(t)), and expanding windows ((R_n,Z_n)\to(\infty,\infty)) such that the three shell errors in (3.2) vanish along the sequence:
[
\int (\Gamma-c)^2,|u_r\lambda_{R_n}'|,dx \to 0,
\qquad
\int (\Gamma-c)^2,|(u_z-\dot\zeta)\eta_{Z_n}'|,dx \to 0,
\qquad
\int (\Gamma-c)^2,(|\lambda_{R_n}''+\tfrac3r\lambda_{R_n}'|+|\eta_{Z_n}''|),dx \to 0,
]
uniformly on bounded time windows.

Then every local ancient limit has constant (\Gamma), and the problem reduces to the no-swirl branch, which is already classified by Koch–Nadirashvili–Seregin–Šverák and the later axisymmetric Liouville results. ([arXiv][6])

I am **not** claiming this theorem is proved. I am saying this is the correct rigorous target produced by your Lyapunov framework after the axisymmetric modification.

## 8. The most promising new input

If I had to pick one missing estimate to attack, it would be the radial one.

The formula (3.2) shows that the main bad term is
[
u_r \lambda_R'.
]
Because (|\lambda_R'|\sim R^{-1}) on the shell (R\lesssim r\lesssim 2R), what matters is not the full velocity but the **inward radial drift averaged on shells**. That lines up perfectly with the 2026 partial-Type-I result, whose message is that one-sided radial inflow is the genuine obstruction. ([arXiv][4])

So the sharpest modification of your current Lyapunov program is:

* keep the profile-relative defect idea,
* but replace Gaussian core windows by a **5D-harmonic radial barrier** times a **drift-modulated axial cutoff**,
* and measure failure only through shellwise inward radial transport and axial drift fluctuation.

That is the form in which your program can really touch the axisymmetric frontier.

The short version is:

[
\boxed{
\text{For the open axisymmetric branch, the right Lyapunov object is not } |u-V_*|^2,
\text{ but } \mathcal L_{\Gamma,\phi,c} + \varepsilon \mathcal M_{J,\Omega,\phi}.
}
]

That is the rigorous modification I would stand behind.

If you want, I’ll write this next as a paper-style section with precise propositions and proofs for the scalar identity and the anisotropic shell-error decomposition, using your manuscript notation.

[1]: https://arxiv.org/pdf/1902.11229 "https://arxiv.org/pdf/1902.11229"
[2]: https://arxiv.org/pdf/1011.5066 "https://arxiv.org/pdf/1011.5066"
[3]: https://arxiv.org/pdf/1701.00868?utm_source=chatgpt.com "Improved Liouville theorems for axially symmetric Navier- ..."
[4]: https://arxiv.org/html/2604.07785v1 "https://arxiv.org/html/2604.07785v1"
[5]: https://global-sci.com/index.php/ata/article/download/8251/16433 "https://global-sci.com/index.php/ata/article/download/8251/16433"
[6]: https://arxiv.org/abs/0709.3599 "[0709.3599] Liouville theorems for the Navier-Stokes equations and applications"
