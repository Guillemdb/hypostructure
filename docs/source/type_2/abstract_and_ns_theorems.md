# Abstract Compactness/Tightness and Time-Regularity Theorems, with a 3D Navier–Stokes Instantiation

## Purpose

This note states and proves two reusable abstract results that isolate the two main functional-analytic steps in the compact Type II renormalization program:

1. **Compactness implies tightness** in a global critical norm after edge-tracking gauge fixing.
2. **Renormalized PDE plus local control implies time regularity**, hence Aubin–Lions compactness.

It then gives a concrete **3D incompressible Navier–Stokes** instantiation of both results.

The note is self-contained and independent of the earlier hypostructure language.

For the canonical repaired compact-type-II barrier statement and notation, use:

- [compact_typeII_master_note_repaired_gauge.md](compact_typeII_master_note_repaired_gauge.md)

---

# Part I. Abstract compactness-to-tightness theorem

## 1. Setup

Let \(X\) be a Banach function space of measurable vector-valued or scalar-valued functions on \(\mathbb R^d\). Assume the following two properties:

### (X1) Multiplication by cutoffs
For every \(\chi\in C_c^\infty(\mathbb R^d)\), the map
\[
f \mapsto \chi f
\]
is bounded from \(X\) to \(X\).

### (X2) Tail truncation continuity
If \(\chi_R\in C_c^\infty(\mathbb R^d)\) is a standard radial cutoff with
\[
0\le \chi_R\le 1,\qquad
\chi_R\equiv 1 \text{ on } B_R,\qquad
\chi_R\equiv 0 \text{ on } \mathbb R^d\setminus B_{2R},
\]
then
\[
\|(1-\chi_R)f\|_X \to 0
\qquad\text{as }R\to\infty
\]
for each fixed \(f\in X\).

These are satisfied by the standard spaces \(L^p(\mathbb R^d)\), \(1\le p<\infty\), and many other translation-invariant critical spaces.

---

## 2. Theorem: precompactness implies uniform tightness

### Theorem T0 (Abstract compactness implies tightness)

Let \(K\subset X\) be a precompact subset of \(X\). Then for every \(\varepsilon>0\) there exists \(R_\varepsilon>0\) such that
\[
\sup_{f\in K}\|(1-\chi_{R_\varepsilon})f\|_X < \varepsilon.
\]

Equivalently, the set \(K\) is uniformly tight in the \(X\)-norm.

### Proof

Fix \(\varepsilon>0\).

Since \(K\) is precompact in \(X\), there exist finitely many elements
\[
f_1,\dots,f_N \in K
\]
such that
\[
K \subset \bigcup_{j=1}^N B_X(f_j,\varepsilon/3).
\]

For each \(j=1,\dots,N\), property (X2) gives an \(R_j>0\) such that
\[
\|(1-\chi_{R_j})f_j\|_X < \varepsilon/3.
\]

Set
\[
R_\varepsilon := \max\{R_1,\dots,R_N\}.
\]
Since \(0\le 1-\chi_{R_\varepsilon}\le 1-\chi_{R_j}\) pointwise in the standard nested cutoff construction, and because multiplication by cutoffs is bounded, we may use the same estimate for all \(f_j\); more directly, enlarging the radius preserves tail convergence, so
\[
\|(1-\chi_{R_\varepsilon})f_j\|_X < \varepsilon/3
\qquad \text{for all }j.
\]

Now take any \(f\in K\). There exists \(j\in\{1,\dots,N\}\) such that
\[
\|f-f_j\|_X < \varepsilon/3.
\]
Then
\[
\|(1-\chi_{R_\varepsilon})f\|_X
\le
\|(1-\chi_{R_\varepsilon})(f-f_j)\|_X
+
\|(1-\chi_{R_\varepsilon})f_j\|_X.
\]

By (X1), multiplication by \(1-\chi_{R_\varepsilon}\) is bounded on \(X\). In the standard spaces \(L^p\), its operator norm is at most \(1\), so
\[
\|(1-\chi_{R_\varepsilon})(f-f_j)\|_X
\le \|f-f_j\|_X < \varepsilon/3.
\]
Thus
\[
\|(1-\chi_{R_\varepsilon})f\|_X
<
\varepsilon/3+\varepsilon/3 < \varepsilon.
\]

Since \(f\in K\) was arbitrary, the conclusion follows.

\(\square\)

---

## 3. Corollary: shell-edge gauge-fixed compact orbits are tight

### Corollary A.1

Let \(V:[\tau_0,\infty)\to X\) be an orbit such that the set
\[
K:=\{V(\tau):\tau\ge \tau_0\}
\]
is precompact in \(X\). Then \(K\) is uniformly tight in \(X\).

### Proof

Apply Theorem T0 to the set \(K\).

\(\square\)

---

## 4. Remarks

1. The theorem is purely functional-analytic. It does not use any PDE.
2. The main nontrivial hypothesis is **global precompactness in the actual norm \(X\)**, not merely local compactness.
3. If one only has local precompactness in \(L^p_{\mathrm{loc}}\), then tightness does **not** follow in general.

---

# Part II. Abstract time-regularity and Aubin–Lions compactness theorem

## 1. Functional-analytic setup

Let \(X,Z,Y\) be Banach spaces such that
\[
X \hookrightarrow\hookrightarrow Z \hookrightarrow Y,
\]
where the first embedding is compact and the second is continuous.

Let \(I=(0,T)\) be a bounded time interval.

---

## 2. Aubin–Lions compactness theorem

### Theorem B (Abstract Aubin–Lions compactness)

Suppose \((u_n)\) is a sequence such that
\[
(u_n) \text{ is bounded in } L^2(I;X),
\]
and
\[
(\partial_t u_n) \text{ is bounded in } L^2(I;Y).
\]
Then \((u_n)\) is precompact in \(L^2(I;Z)\).

### Proof

This is the classical Aubin–Lions theorem in the Hilbert/Banach setting.

\(\square\)

---

## 3. Trace-compactness version

### Theorem B.1 (Lions–Magenes trace consequence)

Assume in addition that
\[
u_n \in L^2(I;X)\cap H^1(I;Y)
\]
uniformly, with \(X\hookrightarrow\hookrightarrow Z\hookrightarrow Y\), and that
\[
u_n \to u
\quad\text{in }L^2(I;Z).
\]
Then, after passing to a subsequence if necessary, one may identify traces \(u_n(0)\) in \(Z\), and if one has uniform continuity of the trajectories in \(Z\) or the standard Lions–Magenes continuity into an intermediate Hilbert space, then
\[
u_n(0)\to u(0)
\quad\text{in }Z.
\]

### Remark

This theorem is not as black-box universal as Theorem B; one needs a specific trace theorem for the chosen triple \(X,Z,Y\). In the PDE application below, we use
\[
X=H^1(B_R),\quad Z=L^2(B_R),\quad Y=H^{-1}(B_R),
\]
for which the standard Lions–Magenes theory applies.

---

## 4. Abstract PDE-to-time-regularity theorem

We now formulate the reusable PDE template.

### Theorem C (Abstract renormalized PDE implies time regularity)

Let \(I\subset \mathbb R\) be a bounded interval, and let \(U:I\to X\) satisfy
\[
\partial_t U = \mathcal A(U) + \sum_{k=1}^m m_k(t)\,Z_k(U)
\]
in the distributional sense, where:

1. \(X\) is a Banach space,
2. \(Y\) is another Banach space,
3. \(U\) is bounded in \(L^2(I;X)\),
4. each coefficient \(m_k\in L^\infty(I)\),
5. the nonlinear/operator terms satisfy
   \[
   \|\mathcal A(U(t))\|_Y \le F(\|U(t)\|_X),
   \]
   for some continuous function \(F\),
6. the modulation directions satisfy
   \[
   \|Z_k(U(t))\|_Y \le C_k(1+\|U(t)\|_X)
   \]
   for all \(t\in I\).

Then
\[
\partial_t U \in L^2(I;Y),
\]
and
\[
\|\partial_t U\|_{L^2(I;Y)}
\le
C\Big(
\|F(\|U\|_X)\|_{L^2(I)}
+
\sum_{k=1}^m \|m_k\|_{L^\infty(I)}( |I|^{1/2} + \|U\|_{L^2(I;X)} )
\Big).
\]

### Proof

By the evolution equation,
\[
\partial_t U = \mathcal A(U) + \sum_{k=1}^m m_k(t)\,Z_k(U).
\]
Hence, using the triangle inequality in \(L^2(I;Y)\),
\[
\|\partial_t U\|_{L^2(I;Y)}
\le
\|\mathcal A(U)\|_{L^2(I;Y)}
+
\sum_{k=1}^m \|m_k Z_k(U)\|_{L^2(I;Y)}.
\]

For the first term,
\[
\|\mathcal A(U)\|_{L^2(I;Y)}
\le
\|F(\|U\|_X)\|_{L^2(I)}.
\]

For each modulation term,
\[
\|m_k Z_k(U)\|_{L^2(I;Y)}
\le
\|m_k\|_{L^\infty(I)} \|Z_k(U)\|_{L^2(I;Y)}.
\]
Using the bound on \(Z_k(U)\),
\[
\|Z_k(U)\|_{L^2(I;Y)}
\le
C_k\bigl( |I|^{1/2} + \|U\|_{L^2(I;X)} \bigr).
\]
Summing over \(k\) gives the conclusion.

\(\square\)

### Remark

The theorem is abstract and reusable; PDE applications require verifying the hypotheses on \(\mathcal A\) and \(Z_k\) in the chosen model.

---

# Part III. 3D Navier–Stokes-specific instantiation

We now instantiate the abstract theorems for the renormalized 3D incompressible Navier–Stokes equation.

## 1. Renormalized NS equation

Let \(V,P\) solve
\[
\partial_\tau V +(V\cdot\nabla)V+\nabla P
=
\nu\Delta V
+a(\tau)\bigl(V+y\cdot\nabla V\bigr)
+b(\tau)\cdot\nabla V,
\qquad \nabla\cdot V=0
\]
on \(B_{2R}\times I\), where \(I\subset\mathbb R\).

---

## 2. NS-specific compactness implies tightness

### Theorem A-NS

Let \(X=L^3(\mathbb R^3)\). Suppose the shell-gauge-fixed renormalized orbit
\[
K:=\{V(\tau):\tau\ge \tau_0\}
\subset L^3(\mathbb R^3)
\]
is precompact in \(L^3(\mathbb R^3)\). Then for every \(\varepsilon>0\), there exists \(R_\varepsilon>0\) such that
\[
\sup_{\tau\ge \tau_0}\int_{|y|>R_\varepsilon}|V(y,\tau)|^3\,dy < \varepsilon^3.
\]
Equivalently,
\[
\sup_{\tau\ge \tau_0}\|(1-\chi_{R_\varepsilon})V(\tau)\|_{L^3}<\varepsilon.
\]

### Proof

Apply Theorem T0 with \(X=L^3(\mathbb R^3)\). Properties (X1) and (X2) are standard for \(L^3\).

\(\square\)

### Scope

The theorem is valid when the shell-gauge-fixed orbit is compact in the global critical norm \(L^3\). With only local compactness, the conclusion is unavailable.

---

## 3. NS-specific time-regularity estimate

We now instantiate Theorem C with
\[
X=H^1(B_{2R}),
\qquad
Y=H^{-1}(B_R).
\]

### Theorem C-NS

Assume on \(B_{2R}\times I\):

1. \(V\) is bounded in \(L^2(I;H^1(B_{2R}))\);
2. \(a,b\in L^\infty(I)\);
3. there exists \(M_P\) such that
   \[
   \sup_{\tau\in I}\inf_{c\in\mathbb R}\|P(\tau)-c\|_{L^{3/2}(B_{2R})}\le M_P.
   \]

Then
\[
\partial_\tau V \in L^2(I;H^{-1}(B_R)),
\]
and
\[
\|\partial_\tau V\|_{L^2(I;H^{-1}(B_R))}
\le
C\Big(
\|\nabla V\|_{L^2(I;L^2(B_{2R}))}
+
\|V\|_{L^4(I;L^4(B_{2R}))}^2
+
|I|^{1/2}M_P
+
\|a\|_{L^\infty(I)}\|V+y\cdot\nabla V\|_{L^2(I;H^{-1}(B_R))}
+
\|b\|_{L^\infty(I)}\|\nabla V\|_{L^2(I;H^{-1}(B_R))}
\Big).
\]

In particular, if the right-hand side is finite, then
\[
\partial_\tau V \in L^2(I;H^{-1}(B_R)).
\]

### Proof

We estimate each term in the renormalized equation in \(H^{-1}(B_R)\).

- The Laplacian term satisfies
  \[
  \|\nu\Delta V\|_{H^{-1}(B_R)}
  \le \nu \|\nabla V\|_{L^2(B_R)}.
  \]

- The transport term satisfies
  \[
  \|(V\cdot\nabla)V\|_{H^{-1}(B_R)}
  \le
  \|V\|_{L^4(B_R)}^2,
  \]
  since
  \[
  (V\cdot\nabla)V = \nabla\cdot(V\otimes V).
  \]

- The pressure term satisfies
  \[
  \|\nabla P\|_{H^{-1}(B_R)}
  \le C(R)\inf_{c\in\mathbb R}\|P-c\|_{L^{3/2}(B_R)}.
  \]

- The scale modulation term satisfies
  \[
  \|a(\tau)(V+y\cdot\nabla V)\|_{H^{-1}(B_R)}
  \le
  |a(\tau)|\,\|V+y\cdot\nabla V\|_{H^{-1}(B_R)}.
  \]

- The translation modulation term satisfies
  \[
  \|b(\tau)\cdot\nabla V\|_{H^{-1}(B_R)}
  \le
  |b(\tau)|\,\|\nabla V\|_{H^{-1}(B_R)}.
  \]

Integrating these bounds in \(L^2(I)\) and using the triangle inequality yields the estimate.

\(\square\)

### Remark

In the Navier–Stokes instantiation, this is the NS-specific verification of the abstract time-regularity template. The required estimates are:
- \(L^2_\tau H^1_y\) from renormalized Caccioppoli,
- local pressure control in Theorem P-NS,
- bounds for the modulation directions.

---

## 4. Local pressure estimate for NS

### Theorem P-NS

Let \(V(\cdot,\tau)\in L^3(\mathbb R^3)\) be divergence free, and let \(P(\cdot,\tau)\) solve
\[
-\Delta P = \partial_i\partial_j(V_iV_j)
\quad \text{in } \mathbb R^3.
\]
Then for every \(R>0\), there exists \(c_R(\tau)\in\mathbb R\) such that
\[
\|P(\tau)-c_R(\tau)\|_{L^{3/2}(B_R)}
\le
C\Big(
\|V(\tau)\|_{L^3(B_{4R})}^2
+
R^2\sup_{x\in B_R}\int_{|z|>2R}\frac{|V(z,\tau)|^2}{|x-z|^5}\,dz
\Big).
\]

### Proof

This is exactly the local pressure estimate proved in the master note by decomposing the pressure into a local Calderón–Zygmund part and a harmonic far-field part.

\(\square\)

---

## 5. NS-specific compact barrier theorem

### Theorem Final-NS (conditional compact Type II barrier)

Assume the renormalized Navier–Stokes orbit \(V(\tau)\) satisfies:

1. **global critical compactness**
   \[
   K=\{V(\tau):\tau\ge \tau_0\}
   \]
   is precompact in \(L^3(\mathbb R^3)\);

2. **local \(L^2_\tau H^1_y\) control**
   \[
   V\in L^2_{\mathrm{loc}}\big((\tau_0,\infty);H^1_{\mathrm{loc}}(\mathbb R^3)\big);
   \]

3. **local \(L^2_\tau H^{-1}_y\) time regularity**
   \[
   \partial_\tau V\in L^2_{\mathrm{loc}}\big((\tau_0,\infty);H^{-1}_{\mathrm{loc}}(\mathbb R^3)\big);
   \]

4. **global \(L^3\)-normalization**
   \[
   \|V(\tau)\|_{L^3(\mathbb R^3)}=1 \quad \forall \tau\ge \tau_0;
   \]

5. **nonnegative renormalization cost**
   \[
   \tilde{\mathfrak D}_{R_0}(\tau)
   =
   \nu\int |\nabla V|^2\phi_{R_0}
   +
   a_+(\tau)\int |V|^2\phi_{R_0};
   \]

6. **pressure-tail control** sufficient to apply Theorem P-NS uniformly on compact time windows.

Then
\[
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau = \infty.
\]

### Proof

By Theorem A-NS, global precompactness in \(L^3\) implies uniform \(L^3\)-tightness of the orbit.

Assume for contradiction that
\[
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau<\infty.
\]
Since \(\tilde{\mathfrak D}_{R_0}\ge 0\), Lemma 6 gives a sequence \(\tau_n\to\infty\) such that
\[
\tilde{\mathfrak D}_{R_0}(\tau_n)\to 0.
\]

By the bridge lemma,
\[
\int_{B_{R_0}}|\nabla V(\tau_n)|^2\to 0.
\]

Now combine the \(L^2_\tau H^1_y\) and \(L^2_\tau H^{-1}_y\) bounds with Aubin–Lions on shifted windows to extract a subsequence such that
\[
V(\tau_n)\to V_*
\quad\text{strongly in }L^2_{\mathrm{loc}}(\mathbb R^3).
\]
Because the orbit is uniformly bounded and precompact in \(L^3(\mathbb R^3)\), the same subsequence converges strongly in \(L^3_{\mathrm{loc}}(\mathbb R^3)\).

Then the compact low-dissipation contradiction theorem applies:
- strong local compactness,
- uniform \(L^3\)-tightness,
- normalization,
- vanishing localized dissipation.

This is impossible.

Hence
\[
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty.
\]

\(\square\)

---

# Part IV. What these theorems buy you

## Abstractly

The two reusable structural results are:

1. **Global compactness in a critical norm implies uniform tightness.**
2. **Renormalized evolution with controlled nonlinear and modulation terms implies time regularity in a negative Sobolev space.**

These are not specific to Navier–Stokes, though the NS verification is.

## For NS3D

The remaining PDE-specific tasks are to justify the hypotheses of Theorem Final-NS:
- that the compact branch is compact in a global critical norm such as \(L^3\),
- that the renormalized orbit has the needed local \(H^1\) and pressure bounds,
- and that the gauge/modulation parameters are controlled (including \(a,b\) under moment transversality \(DG_{\mathrm{sc}}(V)=p\Theta_0\)).

Those are the exact remaining closure tasks.
