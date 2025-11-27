## 0. Problem → hypostructure one-liner

**Native statement.**
Let (\Gamma \subset \mathbb R^3) be a nice closed curve. Among all integral 2-currents (T) with (\partial T = \Gamma) minimizing area (mass), any minimizer is smooth in the interior (no interior singularities).

**Hypostructure/GMT one-sentence goal:**

> Any area-minimizing 2-current (T) with boundary (\Gamma) gives rise to GMT blow-up trajectories ({T_\lambda}) near any putative singular point. Either these trajectories stay in the **safe stratum** (smooth tangent planes) or any attempted singular tangent cone would contradict the hypostructure axioms A1–A8 (mainly Northcott, compactness, and rigidity).

So we’re not trying to reprove Almgren; we’re showing how your GMT hypostructure *would* implement that proof.

---

## 1. Structural GMT ingredients for this example

### 1.1 Base space and current space

* Base space:
  (X = \mathbb R^3) with Euclidean metric.
* Current dimension:
  (k = 2).
* Ambient space of currents:
  (\mathcal X = \mathbf I_2(B_1(0))), integral 2-currents in the unit ball (we localize near a candidate singular point, translate it to the origin).

We restrict to currents with:

* (\partial T = 0) in (B_1) (we’re looking at interior points),
* (\mathbf M(T)\le M_0) (coming from global minimizing property).

Metric: **flat norm** (d_\mathcal F).

---

### 1.2 RG trajectories (blow-up sequence)

Fix a point (x_0) in the support of a minimizer (T). Define blow-ups:

* Scaling maps: (D_\lambda(x) = \lambda^{-1}(x - x_0)).
* Blow-up currents:
  [
  T_\lambda := (D_\lambda)_# T ;\text{restricted to}; B_1(0).
  ]

Think of (\lambda \downarrow 0) as the **RG scale**. A “trajectory” is the curve
(\lambda \mapsto T_\lambda) in ((\mathcal X, d_\mathcal F)).

Because (T) is area-minimizing, each (T_\lambda) is area-minimizing in (\mathbb R^3) (locally), and the **monotonicity formula** gives control of the mass ratios.

We won’t worry about full BV-in-(\lambda) formalism here — for a toy example we treat (\lambda\mapsto T_\lambda) as a “curve of approximate minimizers” with good compactness.

---

### 1.3 Height (\Phi), defect (\nu), efficiency (\Xi), recovery (R), strata

* **Height (\Phi)**: just mass in the unit ball
  [
  \Phi(T) := \mathbf M(T\llcorner B_1(0)).
  ]
  This is the simplest possible height; it has the Northcott / compactness property via Federer–Fleming.

* **Defect (\nu_T)**: “excess over a plane”

  For each plane 2-current (P) through 0 with (\mathbf M(P\llcorner B_1)=\pi),
  define the **excess**
  [
  \text{Exc}(T;P) := \mathbf M(T\llcorner B_1) - \mathbf M(P\llcorner B_1).
  ]
  Then set
  [
  \nu_T := \inf_{P \text{ plane}} \text{Exc}(T;P) ;\ge 0.
  ]
  This is our “defect”: 0 if (T) is exactly a plane, positive if it deviates.

* **Efficiency (\Xi(T))**: reverse of excess ratio

  Define
  [
  \Xi(T) := -,\nu_T
  ]
  or, more informatively,
  [
  \Xi(T) := - \frac{\nu_T}{\Phi(T)}.
  ]
  So (\Xi) is maximized (0) on planes and negative if you have excess.

* **Recovery (R(T))**: rectifiability / tilt regularity index

  For this toy case you can take (R) to be something like:
  [
  R(T) := \text{(negative of the average tilt-excess over scales)}.
  ]
  Informally: as you rescale, if the excess decays, rectifiability improves → (R) increases.

* **Stratification (\Sigma)** (minimal but meaningful):

  * (S_{\text{plane}}): tangent planes (flat multiplicity-1 planes through 0).
  * (S_{\text{smooth}}): currents given by smooth minimal graphs over a plane in (B_1).
  * (S_{\text{nonflat cone}}): homogeneous area-minimizing cones (if any).
  * (S_{\text{wild}}): arbitrary other integral currents in (\mathcal X).

In the **2D in (\mathbb R^3)** case, classical GMT tells us the only area-minimizing cones are planes, so (S_{\text{nonflat cone}}) will be empty in reality — this will be our “Branch A rigidity”.

---

## 2. Axioms A1–A8 for this toy GMT example

I’ll be concise and highlight tools.

### A1 – Height / Northcott (compactness)

**Statement here.**
On (\mathcal X), sublevel sets
[
{T\in\mathcal X : \Phi(T)\le C}
]
with uniformly bounded boundary mass are **precompact in the flat norm**.

**Soft implementation.**

* Take a sequence (T_j) with (\mathbf M(T_j)\le C) and (\mathbf M(\partial T_j)\le C').
* Apply **Federer–Fleming compactness**: there exists a subsequence (T_{j_k}\to T) in flat norm. 

Nothing else needed. This is the perfect “soft” A1: mass = height.

---

### A2 – Metric/height compatibility

**Statement.**
(\Phi) is lower semi-continuous in flat norm, and the monotonicity formula provides a simple “dissipation” along the blow-up path.

Soft facts:

* Flat convergence ⇒ lower semicontinuity of mass.
* Monotonicity formula says: for fixed minimizer (T),
  [
  \lambda \mapsto \frac{\mathbf M(T\llcorner B_\lambda(x_0))}{\pi\lambda^2}
  ]
  is nondecreasing. In rescaled form this is exactly “height nonincreasing along the RG trajectory” up to normalization.

We don’t even need a proper metric gradient-flow structure here; A2 is just “height behaves nicely under the metric and along the blow-up trajectory”.

---

### A3 – Defect–slope compatibility

**Statement (soft qualitative form).**
If a flat limit current (T) of a minimizing sequence has (\nu_T>0) (non-trivial excess away from all planes), then it can’t be stationary/minimizing without paying some “dissipation” → we can deform it to lower mass.

In classical GMT language:

* If (T) is **not** a plane but is still area-minimizing cone, you’d get a real issue.
* For 2D in (\mathbb R^3), the **rigidity theorem** says: every 2D area-minimizing cone is a plane.
* So in this dimension, **area-minimizing + conical + nonzero defect** is impossible.

So in your axiom-speak:

* Any tangent cone (C) with (\nu_C>0) cannot be “stationary” in the sense required by the geometric flow; that branch is killed by rigidity, not by a separate slope bound.

We’re using:

* monotonicity formula + classification of cones ⇒
  “(\nu_T>0) can’t coexist with being an area-minimizing tangent cone”.

---

### A4 – Safe stratum

**Choice.**
Let (S_{\text{safe}} = S_{\text{smooth}}\cup S_{\text{plane}}):

* currents that are smooth minimal graphs near 0 or exactly planes.

**Properties.**

* If the blow-up limit is in (S_{\text{plane}}), Allard’s regularity theorem implies the original minimizer is smooth near (x_0).
* Once in (S_{\text{smooth}}), there is no mechanism to leave: minimal surfaces are analytic; small perturbations that preserve area-minimizing property keep you there.

Soft tools:

* **Allard’s regularity theorem** (standard GMT heavy-ish, but here it’s the central “rigidity/regularity” input, not a structural axiom).

---

### A5 – Stiffness near equilibria (no LS overkill)

Here we can take a **very soft version**:

* Equilibria: planes (and nearby minimal graphs).
* Near a plane (P), the area functional is **strictly convex** in the graphical coordinates for small slopes.
* So we get a spectral-gap-type inequality:
  [
  \Phi(T) - \Phi(P) \gtrsim |u|_{H^1}^2 \quad \text{for } T \text{ graph of } u \text{ over } P.
  ]

That’s enough to prevent “flat energy valleys” near equilibria. No need to invoke Simon LS; standard second-variation computation suffices.

---

### A6 – Metric stiffness / no teleportation

**Statement.**
Flat norm small ⇒ support and mass distribution close:

* In particular, if a sequence (T_j\to T) in flat norm, then:

  * (\mathbf M(T_j)\to\mathbf M(T)),
  * supports converge in Hausdorff sense (locally) after throwing away small-mass pieces.

This ensures you can’t “jump” between wildly different strata without travelling a noticeable flat distance → aligns with your “no teleportation” axiom.

Soft tools:

* definition of flat norm as inf of (\mathbf M(A)+\mathbf M(B)) with (T_j-T = A+\partial B) and bounded mass of A,B,
* standard arguments in Federer/Simon’s GMT book.

---

### A7 – Structural compactness for trajectories

We need:

* From bounded mass and boundary + monotonicity, we can extract **subsequential blow-up limits** (T_{\lambda_k}\to C) in flat norm.

This is exactly:

* Federer–Fleming compactness applied to (T_{\lambda_k}),
* plus the monotonicity formula to ensure the limit is an area-minimizing cone.

So A7 is basically already built into A1 + the blow-up construction.

---

### A8 – Algebraic/analytic rigidity

Here “algebraic” just means “geometric minimal cone classification”.

In our toy 2D-in-(\mathbb R^3) case:

* **Rigidity statement**: any 2D area-minimizing cone in (\mathbb R^3) is a plane.
* This is a classical theorem (you can get it from Simons’ work and the dimension bound for singular sets).

So:

* Extremizers (minimal cones) are all in (S_{\text{plane}}).
* Therefore any blow-up limit of a minimizer is a plane → safe stratum.

That’s A8: extremizers live entirely inside the regular stratum.

---

## 3. SP1 (Recovery) and SP2 (Capacity) in this toy

For this very classical GMT example, the usual proof works almost entirely through **rigidity**, so SP1/SP2 are overkill. But we can still align them loosely:

* **SP1/Recovery**:
  The decay of excess under scaling: if the excess is small at some scale, monotonicity + Caccioppoli/tilt estimates imply it gets smaller at smaller scales. You can interpret that as “inefficiency (excess) ⇒ recovery (improving regularity)”.

* **SP2/Capacity**:
  Nonflat 2D cones in (\mathbb R^3) would force a “large” singular set in higher dimensions; in this dimension, classification says there is simply **no capacity** for nonflat cones. So this branch dies from rigidity rather than capacity exhaustion.

So for this toy, SP1/SP2 are mostly decorative; the key mechanisms are A1 (compactness), A7 (blow-up subsequences), and A8 (cone rigidity).

---

## 4. Branch logic / “no fourth option” in this example

Take a candidate blow-up trajectory ({T_\lambda}) at an interior point of a minimizer.

As (\lambda\downarrow 0), pick a subsequence (\lambda_j\to 0) with (T_{\lambda_j}\to C) (A1/A7).

**Branch A – Structured/cone branch**

* Hypothesis: limit (C) is an area-minimizing cone.
* Using A8 (rigidity of cones in 2D/3D), we get:
  (C) is a flat plane → (C\in S_{\text{plane}}).
* Then Allard’s theorem shows the original surface is smooth near (x_0). No singularity.

**Branch B – Generic/wild branch**

* Hypothesis: limit (C) is *not* an area-minimizing cone, or has nonzero defect (\nu_C>0) but somehow tries to be stationary.
* This contradicts minimality + monotonicity: if (C) is not minimizing, you can adjust at the tangent scale to lower mass, contradicting minimality of (T).
* Operationally: this branch cannot actually occur for blow-ups of minimizers.

Conclusion: **no fourth option**:

* Every blow-up limit is a plane;
* any other possibility contradicts either minimality or cone classification.

Hence no interior singularities.
