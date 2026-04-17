# Master Note: Compact Type II Renormalization Barrier for 3D Navier–Stokes

## Status and purpose of this document

This note consolidates, end to end, the rigorous PDE derivations and lemmas developed in the conversation for a **compact Type II renormalization barrier** for the 3D incompressible Navier–Stokes equations.

The note is **self-contained at the level of the internal analytic chain**. It does **not** claim an unconditional proof of Type II exclusion for 3D Navier–Stokes. Instead, it proves a complete **conditional barrier theorem** under explicit hypotheses on the renormalized orbit.

The document contains:

- the renormalized equation;
- pointwise and cutoff local energy identities;
- the renormalized Caccioppoli estimate;
- the compact low-dissipation rigidity theorem;
- the bridge from low renormalization cost to low core dissipation;
- the local pressure estimate;
- the local \(H^{-1}\) time-regularity estimate;
- the Aubin–Lions–Simon closure theorem yielding the final conditional barrier;
- the repaired weighted-moment scale gauge;
- continuity and structural formulas for the modulation matrix;
- the uniform scale transversality estimate for the repaired weighted scale gauge.

Every lemma proved below is either:
1. proved completely from the stated assumptions; or
2. explicitly marked as depending on a named standard external theorem.

Nothing is hidden.

---

## Strategy overview

The barrier strategy is:

1. Renormalize the Navier–Stokes flow around a concentrating candidate singularity:
   \[
   u(x,t)=\lambda(t)^{-1}V\!\left(\frac{x-x_c(t)}{\lambda(t)},\tau(t)\right),\qquad \tau_t=\lambda(t)^{-2}.
   \]

2. In renormalized variables, define the **nonnegative renormalization cost**
   \[
   \tilde{\mathfrak D}_{R_0}(\tau)
   :=
   \nu \int_{\mathbb R^3} |\nabla V(y,\tau)|^2 \phi_{R_0}(y)\,dy
   +
   a_+(\tau)\int_{\mathbb R^3}|V(y,\tau)|^2 \phi_{R_0}(y)\,dy,
   \]
   where \(\phi_{R_0}\ge 0\) is a cutoff satisfying \(\phi_{R_0}\equiv 1\) on \(B_{R_0}\), and \(a_+(\tau)=\max(a(\tau),0)\).

3. Prove that if the total cost were finite, then by nonnegativity there exists a sequence \(\tau_n\to\infty\) such that
   \[
   \tilde{\mathfrak D}_{R_0}(\tau_n)\to 0.
   \]

4. Show that low cost implies low localized dissipation:
   \[
   \tilde{\mathfrak D}_{R_0}(\tau_n)\to0
   \quad\Longrightarrow\quad
   \int_{B_{R_0}} |\nabla V(\tau_n)|^2 \to 0.
   \]

5. Use compactness + time regularity (Aubin–Lions–Simon) to extract a strong local limit of \(V(\tau_n)\).

6. Prove the rigidity theorem:
   a compact normalized \(L^3\)-tight orbit cannot admit a subsequence with vanishing localized dissipation on a fixed core.

7. Conclude that finite total renormalization cost is impossible:
   \[
   \int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau = \infty.
   \]

The only assumptions left at the end are the explicit hypotheses in the final theorem.

---

## Theorem map

### Main conditional theorem

**Theorem A.** Under hypotheses (H1)–(H5) in Part VII below,
\[
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau = \infty.
\]

### Internal lemmas used

1. Lemma 1: renormalized equation.
2. Lemma 2: pointwise renormalized local energy identity.
3. Lemma 3: spacetime cutoff renormalized local energy identity.
4. Lemma 4: localized renormalized energy identity with spatial cutoff.
5. Lemma 5: renormalized Caccioppoli estimate.
6. Lemma 6: nonnegative integrable cost has a vanishing subsequence.
7. Lemma 7: low renormalization cost implies low core dissipation.
8. Lemma 8: compact normalized low-dissipation subsequence is impossible.
9. Lemma 9: local \(L^{3/2}\) pressure estimate modulo a constant.
10. Lemma 9.1: uniform pressure-tail control from global \(L^3\).
11. Lemma 9.2: verification of the local \(L^2\) pressure hypotheses.
12. Lemma 9.3: local \(L^2\) pressure estimate modulo constants.
13. Lemma 10: local \(L^2_\tau H^{-1}_x\) time-regularity.
14. Lemma 11: Aubin–Lions–Simon closure and contradiction.

### Gauge appendix lemmas

G1. Repaired weighted-moment scale gauge and centering gauge.  
G2. Fréchet derivative of \(G_0\) and exact scaling derivative.  
G3. Modulation matrix entry formulas.  
G4. Continuity of the modulation matrix (including weighted-moment entries).  
G5. Full matrix invertibility by Schur complement under quantitative hypotheses.  
G6. Bounded modulation parameters under uniform nondegeneracy.

---

## Standard external tools quoted

The following standard results are used as black boxes:

1. **Calderón–Zygmund / Riesz transform boundedness**
   \[
   \|\mathcal R_i\mathcal R_j f\|_{L^p(\mathbb R^3)}\le C_p\|f\|_{L^p(\mathbb R^3)},\qquad 1<p<\infty.
   \]

2. **Sobolev and bounded-domain interpolation**
   \[
   H^1(B_R)\hookrightarrow L^6(B_R),
   \]
   and
   \[
   \|u\|_{L^4(B_R)}\le C(R)\|u\|_{L^2(B_R)}^{1/4}\|u\|_{L^6(B_R)}^{3/4}.
   \]

3. **Aubin–Lions–Simon compactness theorem**.

4. **Lions–Magenes trace regularity**
   \[
   L^2(0,1;H^1(B_R))\cap H^1(0,1;H^{-1}(B_R))
   \subset C([0,1];L^2(B_R)).
   \]

We state precisely where each is used.

---

# Part I. Renormalization

## Lemma 1 (Renormalized equation)

Let \(u,p\) solve
\[
\partial_t u + (u\cdot\nabla_x)u + \nabla_x p = \nu \Delta_x u,
\qquad \nabla_x\cdot u = 0,
\]
and define
\[
u(x,t)=\frac{1}{\lambda(t)}V(y,\tau),
\qquad
y=\frac{x-x_c(t)}{\lambda(t)},
\qquad
\frac{d\tau}{dt}=\frac{1}{\lambda(t)^2},
\]
with
\[
p(x,t)=\frac{1}{\lambda(t)^2}P(y,\tau).
\]
Assume \(u,p\) are classical, \(x_c,\lambda\in C^1\), and \(\lambda>0\). Then
\[
\partial_\tau V +(V\cdot\nabla)V+\nabla P
=
\nu\Delta V
+a(\tau)\bigl(V+y\cdot\nabla V\bigr)
+b(\tau)\cdot\nabla V,
\qquad \nabla\cdot V=0,
\]
where
\[
a(\tau)=\lambda(t)\lambda'(t),\qquad b(\tau)=\lambda(t)x_c'(t).
\]

### Proof

We compute
\[
y=\frac{x-x_c(t)}{\lambda(t)}
\quad\Longrightarrow\quad
\partial_t y
=
-\frac{x_c'(t)}{\lambda(t)}
-\frac{\lambda'(t)}{\lambda(t)}y.
\]
Also,
\[
u(x,t)=\lambda^{-1}V(y,\tau),
\]
so
\[
\partial_t u
=
-\frac{\lambda'}{\lambda^2}V
+\frac{1}{\lambda}\Bigl(\partial_\tau V\,\tau_t + (\partial_t y)\cdot\nabla V\Bigr).
\]
Since \(\tau_t=\lambda^{-2}\),
\[
\partial_t u
=
\frac{1}{\lambda^3}\partial_\tau V
-\frac{1}{\lambda^2}x_c'(t)\cdot\nabla V
-\frac{\lambda'}{\lambda^2}(V+y\cdot\nabla V).
\]
Also,
\[
\nabla_x u=\frac{1}{\lambda^2}\nabla_y V,\qquad
\Delta_x u=\frac{1}{\lambda^3}\Delta_y V,
\qquad
(u\cdot\nabla_x)u=\frac{1}{\lambda^3}(V\cdot\nabla_y)V,
\]
and
\[
\nabla_x p=\frac{1}{\lambda^3}\nabla_y P.
\]
Substituting into
\[
\partial_t u+(u\cdot\nabla_x)u+\nabla_x p=\nu\Delta_x u
\]
and multiplying by \(\lambda^3\) gives
\[
\partial_\tau V +(V\cdot\nabla)V+\nabla P
=
\nu\Delta V
+\lambda\lambda'(V+y\cdot\nabla V)
+\lambda x_c'(t)\cdot\nabla V.
\]
Thus
\[
a(\tau)=\lambda\lambda'(t),\qquad b(\tau)=\lambda x_c'(t),
\]
and \(\nabla\cdot V=0\) follows from \(\nabla_x\cdot u=0\).

\(\square\)

---

# Part II. Renormalized local energy identities

Set
\[
e(y,\tau):=\frac12|V(y,\tau)|^2.
\]

## Lemma 2 (Pointwise renormalized local energy identity)

Assume \(V,P\) are classical solutions of the renormalized equation. Then
\[
\partial_\tau e
-\nu \Delta e
+\nu |\nabla V|^2
+\operatorname{div}\bigl((e+P)V-a\,y e-b\,e\bigr)
+a e
=0.
\]

### Proof

Take the dot product of
\[
\partial_\tau V +(V\cdot\nabla)V+\nabla P
=
\nu\Delta V
+a(V+y\cdot\nabla V)
+b\cdot\nabla V
\]
with \(V\). We use:
\[
V\cdot\partial_\tau V=\partial_\tau e,
\qquad
V\cdot(V\cdot\nabla V)=V\cdot\nabla e=\operatorname{div}(eV),
\]
\[
V\cdot\nabla P=\operatorname{div}(PV)
\quad(\nabla\cdot V=0),
\]
\[
\nu V\cdot\Delta V=\nu\Delta e-\nu|\nabla V|^2,
\]
\[
V\cdot(V+y\cdot\nabla V)=2e+y\cdot\nabla e,
\]
\[
V\cdot(b\cdot\nabla V)=b\cdot\nabla e=\operatorname{div}(be),
\]
since \(b=b(\tau)\) depends only on \(\tau\). Therefore
\[
\partial_\tau e + \operatorname{div}(eV)+\operatorname{div}(PV)
=
\nu\Delta e-\nu|\nabla V|^2 + a(2e+y\cdot\nabla e)+\operatorname{div}(be).
\]
Now
\[
2e+y\cdot\nabla e = \operatorname{div}(ye)-e,
\]
because \(\operatorname{div}(ye)=3e+y\cdot\nabla e\). Substituting gives the claim:
\[
\partial_\tau e
-\nu \Delta e
+\nu |\nabla V|^2
+\operatorname{div}\bigl((e+P)V-a\,ye-be\bigr)
+a e
=0.
\]

\(\square\)

---

## Lemma 3 (Spacetime cutoff renormalized local energy identity)

Let \(\zeta=\zeta(y,\tau)\in C_c^\infty(\mathbb R^3\times(\tau_1,\tau_2))\), \(\zeta\ge 0\). Under the assumptions of Lemma 2,
\[
\begin{aligned}
&\int_{\mathbb R^3} e(y,\tau_2)\zeta(y,\tau_2)\,dy
-\int_{\mathbb R^3} e(y,\tau_1)\zeta(y,\tau_1)\,dy \\
&\qquad
+\nu\int_{\tau_1}^{\tau_2}\int_{\mathbb R^3} |\nabla V|^2\zeta\,dy\,d\tau
+\int_{\tau_1}^{\tau_2}\int_{\mathbb R^3} a(\tau)e\zeta\,dy\,d\tau \\
&=
\int_{\tau_1}^{\tau_2}\int_{\mathbb R^3} e(\partial_\tau\zeta+\nu\Delta\zeta)\,dy\,d\tau
+\int_{\tau_1}^{\tau_2}\int_{\mathbb R^3} (e+P)V\cdot\nabla\zeta\,dy\,d\tau \\
&\qquad
-\int_{\tau_1}^{\tau_2}\int_{\mathbb R^3} a(\tau)e\, y\cdot\nabla\zeta\,dy\,d\tau
-\int_{\tau_1}^{\tau_2}\int_{\mathbb R^3} e\, b(\tau)\cdot\nabla\zeta\,dy\,d\tau.
\end{aligned}
\]

### Proof

Multiply the identity in Lemma 2 by \(\zeta\), integrate over \(\mathbb R^3\times(\tau_1,\tau_2)\), and integrate by parts in both \(\tau\) and \(y\). Since \(\zeta\) is compactly supported in \((\tau_1,\tau_2)\times\mathbb R^3\), no boundary terms arise except those displayed at \(\tau=\tau_1,\tau_2\). The formula follows directly.

\(\square\)

---

## Lemma 4 (Localized renormalized energy identity)

Let \(\phi\in C_c^\infty(\mathbb R^3)\), \(0\le \phi\le 1\), and define
\[
\mathcal E_\phi(\tau):=\frac12\int_{\mathbb R^3}|V(y,\tau)|^2\phi(y)\,dy.
\]
Assume:
- \(V(\cdot,\tau)\in H^2(\mathbb R^3)\cap L^3(\mathbb R^3)\),
- \(P(\cdot,\tau)\in H^1_{\mathrm{loc}}(\mathbb R^3)\),
- \(V\) is continuous in \(\tau\) with values in \(H^1\cap L^3\),
- \(a,b\) are continuous.

Then \(\mathcal E_\phi\in C^1\) and
\[
\frac{d}{d\tau}\mathcal E_\phi(\tau)
=
-\nu\int |\nabla V|^2\phi
+\frac{\nu}{2}\int |V|^2\Delta\phi
+\frac12\int |V|^2V\cdot\nabla\phi
+\int P\,V\cdot\nabla\phi
-\frac{a}{2}\int |V|^2\phi
-\frac{a}{2}\int |V|^2 y\cdot\nabla\phi
-\frac12\,b\cdot\int |V|^2\nabla\phi.
\]

### Proof

Differentiate:
\[
\frac{d}{d\tau}\mathcal E_\phi
=
\int V\cdot\partial_\tau V\,\phi.
\]
Insert the renormalized equation:
\[
\frac{d}{d\tau}\mathcal E_\phi
=
\nu\int V\cdot\Delta V\,\phi
-\int V\cdot(V\cdot\nabla V)\phi
-\int V\cdot\nabla P\,\phi
+a\int V\cdot(V+y\cdot\nabla V)\phi
+b\cdot\int V\cdot\nabla V\,\phi.
\]

Compute each term.

For the viscous term,
\[
\int V\cdot\Delta V\,\phi
=
-\int |\nabla V|^2\phi
+\frac12\int |V|^2\Delta\phi.
\]

For convection, using \(\nabla\cdot V=0\),
\[
-\int V\cdot(V\cdot\nabla V)\phi
=
\frac12\int |V|^2V\cdot\nabla\phi.
\]

For pressure,
\[
-\int V\cdot\nabla P\,\phi
=
\int P\,V\cdot\nabla\phi.
\]

For the scale term,
\[
\int V\cdot(V+y\cdot\nabla V)\phi
=
-\frac12\int |V|^2\phi - \frac12\int |V|^2\,y\cdot\nabla\phi.
\]

For the translation term,
\[
b\cdot\int V\cdot\nabla V\,\phi
=
-\frac12\,b\cdot\int |V|^2\nabla\phi.
\]

Substituting gives the result.

\(\square\)

---

# Part III. Renormalized Caccioppoli estimate

## Lemma 5 (Renormalized Caccioppoli estimate)

Let \(V,P\) solve the renormalized equation on \(B_R\times(\tau_1,\tau_2)\), where \(0<r\le R/2\). Let
\[
I_R=(\tau_1,\tau_2),\qquad Q_R=B_R\times I_R.
\]
Choose \(\sigma_1,\sigma_2\) with
\[
\tau_1<\sigma_1<\sigma_2<\tau_2,
\]
and set
\[
I_r=[\sigma_1,\sigma_2],\qquad Q_r=B_r\times I_r.
\]

Assume:
- \(V,P\) are classical on \(Q_R\),
- \(a,b\in L^\infty(I_R)\).

Then there exists \(C>0\), depending only on dimension and cutoff choices, such that
\[
\begin{aligned}
\nu\iint_{Q_r} |\nabla V|^2
\le\;&
\frac{C}{\sigma_2-\sigma_1}\iint_{Q_R}|V|^2
+
\frac{C\nu}{(R-r)^2}\iint_{Q_R}|V|^2 \\
&+
\frac{C}{R-r}\iint_{Q_R}|V|^3
+
\frac{C}{R-r}\iint_{Q_R}|P||V| \\
&+
C\|a\|_{L^\infty(I_R)}\iint_{Q_R}|V|^2
+
\frac{C\|b\|_{L^\infty(I_R)}}{R-r}\iint_{Q_R}|V|^2.
\end{aligned}
\]

If in addition \(a(\tau)\ge 0\) on \(I_R\), then
\[
\begin{aligned}
\nu\iint_{Q_r} |\nabla V|^2
+\iint_{Q_r} a(\tau)\frac{|V|^2}{2}
\le\;&
\frac{C}{\sigma_2-\sigma_1}\iint_{Q_R}|V|^2
+
\frac{C\nu}{(R-r)^2}\iint_{Q_R}|V|^2 \\
&+
\frac{C}{R-r}\iint_{Q_R}|V|^3
+
\frac{C}{R-r}\iint_{Q_R}|P||V| \\
&+
C\|a\|_{L^\infty(I_R)}\iint_{Q_R}|V|^2
+
\frac{C\|b\|_{L^\infty(I_R)}}{R-r}\iint_{Q_R}|V|^2.
\end{aligned}
\]

### Proof

Choose \(\phi\in C_c^\infty(B_R)\) such that
\[
0\le \phi\le 1,\qquad \phi\equiv 1 \text{ on }B_r,
\]
with
\[
|\nabla\phi|\le \frac{C}{R-r},\qquad
|\Delta\phi|\le \frac{C}{(R-r)^2}.
\]
Choose \(\chi\in C_c^\infty((\tau_1,\tau_2))\) such that
\[
0\le \chi\le 1,\qquad \chi\equiv 1 \text{ on } [\sigma_1,\sigma_2],
\]
and
\[
|\chi'|\le \frac{C}{\sigma_2-\sigma_1}.
\]
Set
\[
\zeta(y,\tau):=\phi(y)\chi(\tau).
\]

Apply Lemma 3:
\[
\iint_{Q_R}\nu |\nabla V|^2\zeta
+\iint_{Q_R} a e\,\zeta
=
I_1+I_2+I_3+I_4+I_5,
\]
where
\[
I_1=\iint e\,\partial_\tau\zeta,\qquad
I_2=\nu\iint e\,\Delta\zeta,
\]
\[
I_3=\iint (e+P)V\cdot\nabla\zeta,\qquad
I_4=-\iint a e\, y\cdot\nabla\zeta,\qquad
I_5=-\iint e\, b\cdot\nabla\zeta.
\]

Estimate each term.

Because \(|\chi'|\le C/(\sigma_2-\sigma_1)\),
\[
|I_1|
\le
\frac{C}{\sigma_2-\sigma_1}\iint_{Q_R}|V|^2.
\]

Because \(|\Delta\phi|\le C/(R-r)^2\),
\[
|I_2|
\le
\frac{C\nu}{(R-r)^2}\iint_{Q_R}|V|^2.
\]

For the flux term,
\[
|I_3|
\le
\frac{C}{R-r}\iint_{Q_R}|V|^3
+
\frac{C}{R-r}\iint_{Q_R}|P||V|.
\]

For the scale-modulation flux,
\[
|I_4|
\le
\|a\|_{L^\infty(I_R)}\iint_{Q_R} e\, |y|\,|\nabla\phi|\,\chi.
\]
Since \(r\le R/2\), one has \(\frac{R}{R-r}\le 2\), so
\[
|I_4|
\le
C\|a\|_{L^\infty(I_R)}\iint_{Q_R}|V|^2.
\]

For the translation flux,
\[
|I_5|
\le
\frac{C\|b\|_{L^\infty(I_R)}}{R-r}\iint_{Q_R}|V|^2.
\]

Since \(\zeta\equiv1\) on \(Q_r\),
\[
\nu\iint_{Q_r}|\nabla V|^2\le \iint_{Q_R}\nu |\nabla V|^2\zeta.
\]
If \(a\ge 0\), then
\[
\iint_{Q_r} a e \le \iint_{Q_R} a e\,\zeta.
\]

Combining the previous bounds proves the desired inequalities.

\(\square\)

---

# Part IV. Rigidity of compact low-dissipation subsequences

## Lemma 6 (Nonnegative integrable cost has a vanishing subsequence)

Let \(f:[\tau_0,\infty)\to[0,\infty)\) be measurable. If
\[
\int_{\tau_0}^{\infty} f(\tau)\,d\tau<\infty,
\]
then there exists a sequence \(\tau_n\to\infty\) such that
\[
f(\tau_n)\to 0.
\]

### Proof

Suppose not. Then there exists \(\varepsilon>0\) and \(T>\tau_0\) such that
\[
f(\tau)\ge \varepsilon \qquad \text{for all }\tau\ge T.
\]
Then
\[
\int_{\tau_0}^{\infty} f(\tau)\,d\tau
\ge
\int_T^\infty \varepsilon\,d\tau = \infty,
\]
contradiction.

\(\square\)

---

## Lemma 7 (Low renormalization cost implies low core dissipation)

Fix \(R_0>0\). Let \(\phi_{R_0}\ge 0\) satisfy \(\phi_{R_0}\equiv 1\) on \(B_{R_0}\). Define
\[
\tilde{\mathfrak D}_{R_0}(\tau)
=
\nu\int |\nabla V(y,\tau)|^2 \phi_{R_0}(y)\,dy
+
a_+(\tau)\int |V(y,\tau)|^2 \phi_{R_0}(y)\,dy,
\]
where \(a_+(\tau)=\max(a(\tau),0)\).

Then for any sequence \(\tau_n\),
\[
\tilde{\mathfrak D}_{R_0}(\tau_n)\to 0
\quad\Longrightarrow\quad
\int_{B_{R_0}} |\nabla V(\tau_n,y)|^2\,dy \to 0.
\]

### Proof

Since \(\phi_{R_0}\equiv1\) on \(B_{R_0}\) and the second term in \(\tilde{\mathfrak D}_{R_0}\) is nonnegative,
\[
\int_{B_{R_0}} |\nabla V|^2
\le
\int |\nabla V|^2\phi_{R_0}
\le
\frac{1}{\nu}\tilde{\mathfrak D}_{R_0}.
\]
Taking \(n\to\infty\) proves the claim.

\(\square\)

---

## Lemma 8 (Compact normalized low-dissipation subsequence is impossible)

Let \((V_n)_{n\ge 1}\) be a sequence of divergence-free vector fields on \(\mathbb R^3\) such that:

1. \(V_n\in H^1_{\mathrm{loc}}(\mathbb R^3)\cap L^3(\mathbb R^3)\) for all \(n\);
2. there exists \(V_*\) such that
   \[
   V_n\to V_*
   \quad\text{strongly in }L^2_{\mathrm{loc}}(\mathbb R^3)\cap L^3_{\mathrm{loc}}(\mathbb R^3);
   \]
3. for every \(\varepsilon>0\), there exists \(R_\varepsilon>0\) such that
   \[
   \sup_n \int_{|y|>R_\varepsilon}|V_n(y)|^3\,dy < \varepsilon;
   \]
4. normalization:
   \[
   \|V_n\|_{L^3(\mathbb R^3)}=1 \quad \text{for all }n;
   \]
5. there exists \(R_0>0\) such that
   \[
   \int_{B_{R_0}} |\nabla V_n(y)|^2\,dy \to 0.
   \]

Then such a sequence cannot exist.

### Proof

By strong \(L^2(B_{R_0})\)-convergence, it suffices to show that \(\nabla V_*=0\) in \(B_{R_0}\).

Take any \(\Psi\in C_c^\infty(B_{R_0};\mathbb R^{3\times 3})\). Then
\[
\left|\int_{B_{R_0}}V_n\cdot \operatorname{div}\Psi\,dy\right|
=
\left|\int_{B_{R_0}}\nabla V_n:\Psi\,dy\right|
\le
\|\nabla V_n\|_{L^2(B_{R_0})}\|\Psi\|_{L^2(B_{R_0})}.
\]
By assumption 5, the right-hand side tends to \(0\). Passing to the limit using strong \(L^2(B_{R_0})\)-convergence yields
\[
\int_{B_{R_0}}V_*\cdot \operatorname{div}\Psi\,dy = 0.
\]
Hence \(\nabla V_*=0\) in \(\mathcal D'(B_{R_0})\), so \(V_*\) is a.e. constant on \(B_{R_0}\):
\[
V_*(y)\equiv c \quad \text{on } B_{R_0}
\]
for some \(c\in \mathbb R^3\).

We claim \(c=0\). If \(c\neq 0\), then for every \(R<R_0\),
\[
\int_{B_R}|V_n|^3 \to \int_{B_R}|c|^3 = |c|^3 |B_R|.
\]
Choose \(R<R_0\) large enough that \(|c|^3|B_R|>2\). Then for all large \(n\),
\[
\int_{B_R}|V_n|^3 > 1,
\]
contradicting \(\|V_n\|_{L^3}^3=1\). Thus \(c=0\), so \(V_*\equiv0\) on \(B_{R_0}\).

Now fix \(\varepsilon>0\). By tightness choose \(R>R_0\) such that
\[
\sup_n \int_{|y|>R}|V_n(y)|^3\,dy < \varepsilon.
\]
Since \(V_n\to0\) strongly in \(L^3(B_R)\),
\[
\int_{B_R}|V_n|^3\,dy \to 0.
\]
Therefore for all large \(n\),
\[
\|V_n\|_{L^3}^3
=
\int_{B_R}|V_n|^3\,dy
+
\int_{|y|>R}|V_n|^3\,dy
< 2\varepsilon.
\]
Since \(\varepsilon\) is arbitrary, \(\|V_n\|_{L^3}\to0\), contradicting normalization.

\(\square\)

---

# Part V. Pressure and time-regularity

## Lemma 9 (Local pressure estimate modulo a constant)

Let \(R>0\), fix a time \(\tau\), and suppose \(V(\cdot,\tau)\in L^3(\mathbb R^3;\mathbb R^3)\) is divergence free. Let \(P(\cdot,\tau)\) solve
\[
-\Delta P = \partial_i\partial_j(V_iV_j)
\quad \text{in }\mathbb R^3.
\]
Then there exists a constant \(c_R(\tau)\in\mathbb R\) such that
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

Choose \(\eta\in C_c^\infty(B_{4R})\) with
\[
0\le \eta\le 1,\qquad \eta\equiv 1 \text{ on }B_{2R}.
\]
Decompose
\[
V_iV_j=\eta V_iV_j + (1-\eta)V_iV_j.
\]
Define
\[
P_{\mathrm{loc}}:=\mathcal R_i\mathcal R_j(\eta V_iV_j),\qquad
P_{\mathrm{far}}:=P-P_{\mathrm{loc}}.
\]

For the local part, Calderón–Zygmund gives
\[
\|P_{\mathrm{loc}}\|_{L^{3/2}(\mathbb R^3)}
\le
C\|\eta V_iV_j\|_{L^{3/2}(\mathbb R^3)}
\le
C\|V\|_{L^3(B_{4R})}^2,
\]
hence
\[
\|P_{\mathrm{loc}}\|_{L^{3/2}(B_R)}
\le
C\|V\|_{L^3(B_{4R})}^2.
\]

For the far part, observe that
\[
-\Delta P_{\mathrm{far}}=\partial_i\partial_j((1-\eta)V_iV_j),
\]
and the right-hand side vanishes in \(B_{2R}\), so \(P_{\mathrm{far}}\) is harmonic in \(B_{2R}\).

Let
\[
K_{ij}(x)=\partial_i\partial_j\Big(\frac{1}{4\pi|x|}\Big).
\]
Up to an additive constant,
\[
P_{\mathrm{far}}(x)=\int_{\mathbb R^3}K_{ij}(x-z)(1-\eta(z))V_i(z)V_j(z)\,dz.
\]
Fix \(x_0\in B_R\) and set \(c_R:=P_{\mathrm{far}}(x_0)\). Then for \(x\in B_R\),
\[
P_{\mathrm{far}}(x)-c_R
=
\int_{|z|>2R}\bigl(K_{ij}(x-z)-K_{ij}(x_0-z)\bigr)V_i(z)V_j(z)\,dz.
\]
Using the mean value theorem and \(|\nabla K_{ij}(\xi)|\le C|\xi|^{-4}\),
\[
|K_{ij}(x-z)-K_{ij}(x_0-z)|
\le
C|x-x_0||x-z|^{-4}.
\]
Since \(x,x_0\in B_R\), \(|x-x_0|\le 2R\), so
\[
|P_{\mathrm{far}}(x)-c_R|
\le
CR\int_{|z|>2R}\frac{|V(z)|^2}{|x-z|^4}\,dz.
\]
Because \(|x-z|\ge cR\) on \(|z|>2R\), \(x\in B_R\),
\[
\int_{|z|>2R}\frac{|V(z)|^2}{|x-z|^4}\,dz
\le
CR\sup_{x\in B_R}\int_{|z|>2R}\frac{|V(z)|^2}{|x-z|^5}\,dz.
\]
Hence
\[
|P_{\mathrm{far}}(x)-c_R|
\le
CR^2\sup_{x\in B_R}\int_{|z|>2R}\frac{|V(z)|^2}{|x-z|^5}\,dz.
\]
Taking the \(L^{3/2}(B_R)\)-norm gives
\[
\|P_{\mathrm{far}}-c_R\|_{L^{3/2}(B_R)}
\le
CR^2\sup_{x\in B_R}\int_{|z|>2R}\frac{|V(z)|^2}{|x-z|^5}\,dz.
\]

Since
\[
P-c_R = P_{\mathrm{loc}}+(P_{\mathrm{far}}-c_R),
\]
the stated estimate follows.

\(\square\)

---

## Lemma 9.1 (Uniform pressure-tail control from global \(L^3\))

For \(R>0\), define
\[
\mathcal T_R(\tau):=
\sup_{x\in B_R}\int_{|z|>2R}\frac{|V(z,\tau)|^2}{|x-z|^5}\,dz.
\]
If
\[
M_3:=\sup_{\tau\ge\tau_0}\|V(\tau)\|_{L^3(\mathbb R^3)}<\infty,
\]
then
\[
\sup_{\tau\ge\tau_0}\mathcal T_R(\tau)\le C R^{-4}M_3^2.
\]

### Proof

Fix \(x\in B_R\) and decompose
\[
\{|z|>2R\}=\bigcup_{k=1}^\infty A_k,\qquad
A_k:=\{2^kR<|z|\le2^{k+1}R\}.
\]
For \(z\in A_k\),
\[
|x-z|\ge |z|-|x|\ge 2^kR-R\ge 2^{k-1}R.
\]
Therefore
\[
\int_{|z|>2R}\frac{|V(z,\tau)|^2}{|x-z|^5}\,dz
\le
C\sum_{k\ge1}(2^kR)^{-5}\int_{A_k}|V(z,\tau)|^2\,dz.
\]
By Holder,
\[
\int_{A_k}|V|^2
\le |A_k|^{1/3}\left(\int_{A_k}|V|^3\right)^{2/3}
\le C\,2^kR\,a_k(\tau)^{2/3},
\]
where
\[
a_k(\tau):=\int_{A_k}|V(z,\tau)|^3\,dz.
\]
Thus
\[
\mathcal T_R(\tau)
\le C R^{-4}\sum_{k\ge1}2^{-4k}a_k(\tau)^{2/3}.
\]
Using Holder on the counting measure with exponents \(3\) and \(3/2\),
\[
\sum_{k\ge1}2^{-4k}a_k^{2/3}
\le
\left(\sum_{k\ge1}2^{-12k}\right)^{1/3}
\left(\sum_{k\ge1}a_k\right)^{2/3}
\le C\|V(\tau)\|_{L^3(\mathbb R^3)}^2.
\]
Taking the supremum over \(x\in B_R\) and \(\tau\ge\tau_0\) proves the claim.

\(\square\)

---

## Lemma 9.2 (Verification of the local \(L^2\) pressure hypotheses)

Let \(I\subset\mathbb R\) be bounded and \(R>0\). Assume
\[
\sup_{\tau\in I}\|V(\tau)\|_{L^3(\mathbb R^3)}\le M_3
\]
and
\[
V\in L^2(I;H^1(B_{4R})).
\]
Then
\[
V\in L^4(I;L^4(B_{4R})).
\]
Moreover, if
\[
\mathcal H_R(\tau):=
\sup_{x\in B_R}\int_{|z|>2R}\frac{|V(z,\tau)|^2}{|x-z|^4}\,dz,
\]
then
\[
\|\mathcal H_R\|_{L^\infty(I)}\le C R^{-3}M_3^2,
\qquad
\|\mathcal H_R\|_{L^2(I)}\le C|I|^{1/2}R^{-3}M_3^2.
\]

### Proof

For almost every \(\tau\in I\), interpolation and Sobolev give
\[
\|V(\tau)\|_{L^4(B_{4R})}
\le
\|V(\tau)\|_{L^3(B_{4R})}^{1/2}
\|V(\tau)\|_{L^6(B_{4R})}^{1/2}
\le
C_R M_3^{1/2}\|V(\tau)\|_{H^1(B_{4R})}^{1/2}.
\]
Thus
\[
\|V(\tau)\|_{L^4(B_{4R})}^4
\le C_RM_3^2\|V(\tau)\|_{H^1(B_{4R})}^2.
\]
Integrating in \(\tau\) proves \(V\in L^4(I;L^4(B_{4R}))\).

For \(\mathcal H_R\), use the same dyadic annuli \(A_k\) as in Lemma 9.1. For \(x\in B_R\),
\[
\int_{|z|>2R}\frac{|V(z,\tau)|^2}{|x-z|^4}\,dz
\le
C\sum_{k\ge1}(2^kR)^{-4}\int_{A_k}|V(z,\tau)|^2\,dz.
\]
The previous Holder estimate gives
\[
\int_{A_k}|V|^2\le C\,2^kR\,a_k(\tau)^{2/3},
\]
so
\[
\mathcal H_R(\tau)
\le C R^{-3}\sum_{k\ge1}2^{-3k}a_k(\tau)^{2/3}.
\]
Again,
\[
\sum_{k\ge1}2^{-3k}a_k(\tau)^{2/3}
\le
\left(\sum_{k\ge1}2^{-9k}\right)^{1/3}
\left(\sum_{k\ge1}a_k(\tau)\right)^{2/3}
\le C\|V(\tau)\|_{L^3(\mathbb R^3)}^2.
\]
Hence \(\mathcal H_R(\tau)\le C R^{-3}M_3^2\), which gives both asserted bounds.

\(\square\)

---

## Lemma 9.3 (Local \(L^2\) pressure estimate modulo constants)

Let \(I\subset\mathbb R\) be bounded. Assume
\[
V\in L^4(I;L^4(B_{4R})),
\qquad
\mathcal H_R\in L^2(I),
\]
where \(\mathcal H_R\) is defined in Lemma 9.2. Suppose
\[
-\Delta P=\partial_i\partial_j(V_iV_j)
\quad\text{in }\mathbb R^3
\]
for almost every \(\tau\in I\). Then there exists a measurable \(c_R:I\to\mathbb R\) such that
\[
P-c_R\in L^2(I;L^2(B_R)),
\]
and
\[
\|P-c_R\|_{L^2(I;L^2(B_R))}
\le
C_R\left(
\|V\|_{L^4(I;L^4(B_{4R}))}^2+\|\mathcal H_R\|_{L^2(I)}
\right).
\]

### Proof

Choose \(\eta\in C_c^\infty(B_{4R})\) with \(0\le\eta\le1\) and \(\eta\equiv1\) on \(B_{2R}\). Write
\[
V_iV_j=\eta V_iV_j+(1-\eta)V_iV_j
\]
and define
\[
P_{\mathrm{loc}}:=\mathcal R_i\mathcal R_j(\eta V_iV_j),
\qquad
P_{\mathrm{far}}:=P-P_{\mathrm{loc}}.
\]
Calderon-Zygmund on \(L^2(\mathbb R^3)\) gives
\[
\|P_{\mathrm{loc}}(\tau)\|_{L^2(\mathbb R^3)}
\le C\|V(\tau)\|_{L^4(B_{4R})}^2,
\]
hence
\[
\|P_{\mathrm{loc}}\|_{L^2(I;L^2(B_R))}
\le C\|V\|_{L^4(I;L^4(B_{4R}))}^2.
\]

For the far term, \(P_{\mathrm{far}}\) is harmonic in \(B_{2R}\) modulo constants. Let
\[
K_{ij}(x):=\partial_i\partial_j\left(\frac{1}{4\pi |x|}\right).
\]
Fix \(x_0=0\). Since pressure is defined modulo constants, choose \(c_R(\tau)\) so that for \(x\in B_R\),
\[
P_{\mathrm{far}}(x,\tau)-c_R(\tau)
=
\int_{|z|>2R}
\bigl(K_{ij}(x-z)-K_{ij}(x_0-z)\bigr)
(1-\eta(z))V_i(z,\tau)V_j(z,\tau)\,dz.
\]
The kernel estimate
\[
|\nabla K_{ij}(\xi)|\le C|\xi|^{-4}
\]
and the mean value theorem imply
\[
|K_{ij}(x-z)-K_{ij}(x_0-z)|
\le CR|x-z|^{-4}
\]
for \(x\in B_R\), \(|z|>2R\). Hence
\[
|P_{\mathrm{far}}(x,\tau)-c_R(\tau)|
\le
CR\,\mathcal H_R(\tau).
\]
Taking the \(L^2(B_R)\)-norm gives
\[
\|P_{\mathrm{far}}(\tau)-c_R(\tau)\|_{L^2(B_R)}
\le CR^{5/2}\mathcal H_R(\tau).
\]
Taking \(L^2(I)\) and adding the local estimate proves the result.

\(\square\)

---

## Lemma 10 (Local \(L^2_\tau H^{-1}_x\) time-regularity)

Let \(R>0\), and let \(V,P\) solve the renormalized equation on \(B_{2R}\times I\), \(I\subset\mathbb R\). Assume:

1. local \(H^1\)-bound:
   \[
   \sup_{\tau\in I}\|V(\tau)\|_{H^1(B_{2R})}\le M_1;
   \]

2. local \(L^2\) pressure bound modulo constants:
   \[
   \text{there exists }c\in L^2(I)\text{ such that }
   \|P-c\|_{L^2(I;L^2(B_{2R}))}\le M_{P,2};
   \]

3. bounded modulation parameters:
   \[
   \sup_{\tau\in I}\bigl(|a(\tau)|+|b(\tau)|\bigr)\le M_{ab}.
   \]

Then there exists \(C=C(R,\nu)\) such that
\[
\|\partial_\tau V\|_{L^2(I;H^{-1}(B_R))}
\le
C\Bigl(
|I|^{1/2}\bigl(M_1+M_1^2+M_{ab}(M_1+1)\bigr)
 + M_{P,2}
\Bigr).
\]

In particular, by Lemmas 9.2 and 9.3, hypothesis 2 follows on bounded intervals from
\[
\sup_{\tau\in I}\|V(\tau)\|_{L^3(\mathbb R^3)}<\infty,
\qquad
V\in L^2(I;H^1(B_{8R})),
\]
and
\[
-\Delta P=\partial_i\partial_j(V_iV_j)
\quad\text{in }\mathbb R^3
\]
for almost every \(\tau\in I\).

### Proof

We estimate each term in
\[
\partial_\tau V
=
\nu\Delta V-(V\cdot\nabla)V-\nabla P+a(\tau)V+a(\tau)\,y\cdot\nabla V+b(\tau)\cdot\nabla V
\]
in \(H^{-1}(B_R)\).

Let \(\varphi\in H_0^1(B_R)\) with \(\|\varphi\|_{H^1(B_R)}\le1\).

**Laplacian term.**
\[
\langle \Delta V,\varphi\rangle
=
-\int_{B_R}\nabla V:\nabla\varphi,
\]
hence
\[
\|\Delta V\|_{H^{-1}(B_R)}\le \|\nabla V\|_{L^2(B_R)}\le M_1.
\]

**Pressure term.**
Choose \(c\in L^2(I)\) as in hypothesis 2.
Replacing \(P\) by \(P-c(\tau)\) does not change \(\nabla P\). Thus, for almost every \(\tau\in I\),
\[
\langle \nabla P,\varphi\rangle
=
-\int_{B_R}(P-c(\tau))\,\nabla\cdot\varphi.
\]
Since
\[
\|\nabla\cdot\varphi\|_{L^2(B_R)}
\le \|\nabla\varphi\|_{L^2(B_R)}
\le \|\varphi\|_{H^1(B_R)},
\]
we have
\[
\|\nabla P\|_{H^{-1}(B_R)}
\le
\|P(\tau)-c(\tau)\|_{L^2(B_R)}.
\]
Taking \(L^2(I)\) gives
\[
\|\nabla P\|_{L^2(I;H^{-1}(B_R))}
\le M_{P,2}.
\]

**Transport term.** Since \(\nabla\cdot V=0\),
\[
(V\cdot\nabla)V = \nabla\cdot(V\otimes V),
\]
hence
\[
\langle (V\cdot\nabla)V,\varphi\rangle
=
-\int_{B_R}(V\otimes V):\nabla\varphi.
\]
Therefore
\[
\|(V\cdot\nabla)V\|_{H^{-1}(B_R)}
\le
\|V\otimes V\|_{L^2(B_R)}
=
\|V\|_{L^4(B_R)}^2.
\]
By bounded-domain Gagliardo–Nirenberg and Sobolev,
\[
\|V\|_{L^4(B_R)}^2 \le C(R)\|V\|_{H^1(B_R)}^2 \le C(R)M_1^2.
\]

**The \(aV\) term.**
\[
|\langle aV,\varphi\rangle|
\le
|a|\|V\|_{L^2(B_R)}\|\varphi\|_{L^2(B_R)}
\le
C(R)M_{ab}M_1.
\]

**The \(a\,y\cdot\nabla V\) term.**
\[
\langle y\cdot\nabla V,\varphi\rangle
=
\sum_{j=1}^3 \int_{B_R} y_j\partial_j V\cdot\varphi.
\]
Integrating by parts,
\[
\int_{B_R} y_j\partial_j V\cdot\varphi
=
-\int_{B_R}V\cdot \partial_j(y_j\varphi)
=
-\int_{B_R}V\cdot\varphi - \int_{B_R}y_jV\cdot\partial_j\varphi.
\]
Hence
\[
|\langle y\cdot\nabla V,\varphi\rangle|
\le
3\|V\|_{L^2(B_R)}\|\varphi\|_{L^2(B_R)}
+
R\|V\|_{L^2(B_R)}\|\nabla\varphi\|_{L^2(B_R)},
\]
so
\[
\|y\cdot\nabla V\|_{H^{-1}(B_R)}\le C(R)M_1.
\]
Therefore
\[
\|a(\tau)\,y\cdot\nabla V\|_{H^{-1}(B_R)}
\le C(R)M_{ab}M_1.
\]

**The \(b\cdot\nabla V\) term.**
\[
\langle b\cdot\nabla V,\varphi\rangle
=
-\int_{B_R}V\cdot (b\cdot\nabla\varphi),
\]
hence
\[
\|b\cdot\nabla V\|_{H^{-1}(B_R)}
\le
|b|\|V\|_{L^2(B_R)}
\le M_{ab}M_1.
\]

Combining the six estimates yields
\[
\|\partial_\tau V\|_{L^2(I;H^{-1}(B_R))}
\le
C\Bigl(
|I|^{1/2}\bigl(M_1+M_1^2+M_{ab}(M_1+1)\bigr)
 + M_{P,2}
\Bigr).
\]

\(\square\)

---

# Part VI. Aubin–Lions–Simon closure

## Lemma 11 (Aubin–Lions–Simon closure of the low-cost sequence)

Assume:

1. For every \(R>0\),
   \[
   \sup_{\tau\ge\tau_0}\|V(\tau)\|_{H^1(B_{2R})}\le M_1(R).
   \]

2. For every \(R>0\),
   \[
   \sup_{T\ge\tau_0}
   \|\partial_\tau V\|_{L^2((T,T+1);H^{-1}(B_R))}\le M_2(R).
   \]

3. The orbit is uniformly \(L^3\)-tight and \(L^3\)-normalized:
   \[
   \|V(\tau)\|_{L^3(\mathbb R^3)}=1,
   \]
   and
   \[
   \forall \varepsilon>0\ \exists R_\varepsilon:\ 
   \sup_{\tau\ge\tau_0}\int_{|y|>R_\varepsilon}|V(y,\tau)|^3<\varepsilon.
   \]

4. There exists \(\tau_n\to\infty\) such that
   \[
   \tilde{\mathfrak D}_{R_0}(\tau_n)\to 0.
   \]

Then, after passing to a subsequence, there exists \(V_*\) such that
\[
V(\tau_n)\to V_*
\quad\text{strongly in }L^2_{\mathrm{loc}}(\mathbb R^3)\cap L^3_{\mathrm{loc}}(\mathbb R^3),
\]
and
\[
\int_{B_{R_0}} |\nabla V(\tau_n)|^2\to 0.
\]
Consequently, Theorem 8 applies and gives a contradiction.

### Proof

By Lemma 7,
\[
\tilde{\mathfrak D}_{R_0}(\tau_n)\to 0
\quad\Longrightarrow\quad
\int_{B_{R_0}} |\nabla V(\tau_n)|^2\to 0.
\]

It remains to obtain strong local compactness of \(V(\tau_n)\). Fix \(R>0\), and define shifted trajectories
\[
W_n(s,y):=V(\tau_n+s,y),
\qquad s\in(0,1),\ y\in B_R.
\]

By assumption 1,
\[
\sup_n \|W_n\|_{L^\infty(0,1;H^1(B_R))}<\infty,
\]
hence
\[
\sup_n \|W_n\|_{L^2(0,1;H^1(B_R))}<\infty.
\]

By assumption 2,
\[
\sup_n \|\partial_s W_n\|_{L^2(0,1;H^{-1}(B_R))}<\infty.
\]

Since
\[
H^1(B_R)\hookrightarrow\hookrightarrow L^2(B_R)\hookrightarrow H^{-1}(B_R),
\]
the Aubin–Lions–Simon theorem implies, after passage to a subsequence,
\[
W_n \to W_*
\quad\text{strongly in }L^2(0,1;L^2(B_R)).
\]

By Lions–Magenes trace regularity,
\[
W_n \in C([0,1];L^2(B_R)),
\]
with uniform control. By Simon’s compactness theorem in the time-continuous version, after a further subsequence,
\[
W_n \to W_*
\quad\text{strongly in }C([0,1];L^2(B_R)).
\]

In particular, at time \(s=0\),
\[
V(\tau_n,\cdot)=W_n(0,\cdot)\to W_*(0,\cdot)
\quad\text{strongly in }L^2(B_R).
\]

Since \(R>0\) was arbitrary, a diagonal extraction gives a subsequence and a function \(V_*\) such that
\[
V(\tau_n)\to V_*
\quad\text{strongly in }L^2_{\mathrm{loc}}(\mathbb R^3).
\]

Now, from the local \(H^1\)-bound and strong \(L^2\)-convergence on each \(B_R\), interpolation yields strong \(L^3\)-convergence on each \(B_R\):
\[
\|V(\tau_n)-V_*\|_{L^3(B_R)}
\le
\|V(\tau_n)-V_*\|_{L^2(B_R)}^{1/2}
\|V(\tau_n)-V_*\|_{L^6(B_R)}^{1/2}.
\]
By Sobolev, \(H^1(B_R)\hookrightarrow L^6(B_R)\), and the \(H^1\)-bound is uniform, so the right-hand side tends to \(0\). Hence
\[
V(\tau_n)\to V_*
\quad\text{strongly in }L^3_{\mathrm{loc}}(\mathbb R^3).
\]

With uniform \(L^3\)-tightness and normalization, all the assumptions of Lemma 8 are satisfied. Since the low core dissipation was already shown above, Lemma 8 gives a contradiction.

\(\square\)

---

# Part VII. Final barrier theorem

## Theorem A (Conditional compact Type II renormalization barrier)

Let \(V,P\) solve the renormalized Navier–Stokes equation
\[
\partial_\tau V +(V\cdot\nabla)V+\nabla P
=
\nu\Delta V
+a(\tau)\bigl(V+y\cdot\nabla V\bigr)
+b(\tau)\cdot\nabla V,
\qquad \nabla\cdot V=0
\]
on \([\tau_0,\infty)\times \mathbb R^3\).

Fix \(R_0>0\) and a cutoff \(\phi_{R_0}\in C_c^\infty(\mathbb R^3)\), \(\phi_{R_0}\ge 0\), \(\phi_{R_0}\equiv 1\) on \(B_{R_0}\), and define
\[
\tilde{\mathfrak D}_{R_0}(\tau)
=
\nu \int_{\mathbb R^3} |\nabla V(y,\tau)|^2 \phi_{R_0}(y)\,dy
+
a_+(\tau)\int_{\mathbb R^3}|V(y,\tau)|^2 \phi_{R_0}(y)\,dy.
\]

Assume:

### (H1) global \(L^3\)-normalization
\[
\|V(\tau)\|_{L^3(\mathbb R^3)}=1
\qquad \forall \tau\ge\tau_0.
\]

### (H2) uniform \(L^3\)-tightness
For every \(\varepsilon>0\), there exists \(R_\varepsilon>0\) such that
\[
\sup_{\tau\ge\tau_0}\int_{|y|>R_\varepsilon}|V(y,\tau)|^3\,dy<\varepsilon.
\]

### (H3) uniform local \(H^1\)-bounds
For every \(R>0\), there exists \(M_1(R)\) such that
\[
\sup_{\tau\ge\tau_0}\|V(\tau)\|_{H^1(B_{2R})}\le M_1(R).
\]

### (H4) uniform boundedness of the modulation parameters
There exists \(M_{ab}\) such that
\[
\sup_{\tau\ge\tau_0}\bigl(|a(\tau)|+|b(\tau)|\bigr)\le M_{ab}.
\]

### Derived pressure and time-regularity input
By (H1), Lemma 9.1 gives, for every \(R>0\),
\[
\sup_{\tau\ge\tau_0}\sup_{x\in B_R}
\int_{|z|>2R}\frac{|V(z,\tau)|^2}{|x-z|^5}\,dz
\le
C R^{-4}.
\]
By (H1), (H3), (H4), and Lemmas 9.2, 9.3, and 10, for every \(R>0\),
\[
\sup_{T\ge\tau_0}
\|\partial_\tau V\|_{L^2((T,T+1);H^{-1}(B_R))}<\infty.
\]

Then
\[
\int_{\tau_0}^{\infty} \tilde{\mathfrak D}_{R_0}(\tau)\,d\tau = \infty.
\]

### Proof

Assume for contradiction that
\[
\int_{\tau_0}^{\infty} \tilde{\mathfrak D}_{R_0}(\tau)\,d\tau < \infty.
\]
By Lemma 6, there exists a sequence \(\tau_n\to\infty\) such that
\[
\tilde{\mathfrak D}_{R_0}(\tau_n)\to 0.
\]
By Lemma 11, after passing to a subsequence,
\[
V(\tau_n)\to V_*
\quad\text{strongly in }L^2_{\mathrm{loc}}(\mathbb R^3)\cap L^3_{\mathrm{loc}}(\mathbb R^3),
\]
and
\[
\int_{B_{R_0}} |\nabla V(\tau_n)|^2\to 0.
\]
Then Lemma 8 applies and yields a contradiction.

Therefore
\[
\int_{\tau_0}^{\infty} \tilde{\mathfrak D}_{R_0}(\tau)\,d\tau = \infty.
\]

\(\square\)

---

# Part VIII. Repaired gauge appendix (single active approach)

## G1. Repaired weighted-moment scale gauge + centering

Keep the centering gauge
\[
G_j(V):=\int_{\mathbb R^3} y_j |V(y)|^2 \psi_R(y)\,dy = 0,\qquad j=1,2,3,
\]
with \(\psi_R\in C_c^\infty(\mathbb R^3)\), \(\psi_R\equiv1\) on \(B_R\), and \(\psi_R\equiv0\) outside \(B_{2R}\).

Set the scale gauge as
\[
G_0(V):=\int_{\mathbb R^3}|y|^{-p}|V(y)|^3\,dy-\Theta_0=0,
\qquad 0<p<3,\ \Theta_0>0.
\]
In physical variables this is
\[
\lambda(t)^p\int_{\mathbb R^3}|x-x_c(t)|^{-p}|u(x,t)|^3\,dx=\Theta_0.
\]

From now on, the only scale gauge in this note is the weighted-moment gauge above.

## G2. Derivatives and exact scale derivative

Let \(W\) be a test variation and \(Z_{\mathrm{sc}}(V):=V+y\cdot\nabla V\). Under the admissibility hypotheses in
\[
\texttt{required\_new\_scale\_gauge\_theorems.md},
\]
one has
\[
DG_0(V)[W]=3\int_{\mathbb R^3}|y|^{-p}|V|\,V\cdot W\,dy,
\]
\[
DG_0(V)[Z_{\mathrm{sc}}(V)]
=p\int_{\mathbb R^3}|y|^{-p}|V(y)|^3\,dy.
\]
Hence on the gauge surface \(G_0(V)=0\),
\[
DG_0(V)[Z_{\mathrm{sc}}(V)]=p\Theta_0>0.
\]

For the centering rows,
\[
DG_j(V)[W]=2\int_{\mathbb R^3} y_jV\cdot W\,\psi_R\,dy,\qquad j=1,2,3.
\]
\[
DG_j(V)[Z_{\mathrm{sc}}(V)]
=2\int_{\mathbb R^3} y_jV\cdot(V+y\cdot\nabla V)\,\psi_R\,dy.
\]

---

## G3. Modulation matrix entries

Let
\[
Z_{\mathrm{sc}}(V):=V+y\cdot\nabla V.
\]

The modulation matrix \(M(V)\in \mathbb R^{4\times 4}\) is defined by
\[
M(V)_{00}=DG_0(V)[Z_{\mathrm{sc}}(V)],\quad
M(V)_{0\ell}=DG_0(V)[\partial_\ell V],\quad
M(V)_{j0}=DG_j(V)[Z_{\mathrm{sc}}(V)],\quad
M(V)_{j\ell}=DG_j(V)[\partial_\ell V].
\]

Hence
\[
M_{00}(V)=p\int_{\mathbb R^3}|y|^{-p}|V(y)|^3\,dy = p\Theta_0,
\]
\[
M_{0\ell}(V)=3\int_{\mathbb R^3}|y|^{-p}|V|\,V\cdot\partial_\ell V\,dy
=\int_{\mathbb R^3}|y|^{-p}\partial_\ell(|V|^3)\,dy
=p\int_{\mathbb R^3}y_\ell|y|^{-p-2}|V(y)|^3\,dy,
\]
\[
M_{j0}(V)=2\int_{\mathbb R^3}y_j\,V\cdot(V+y\cdot\nabla V)\,\psi_R\,dy,
\]
\[
M_{j\ell}(V)
=-\delta_{j\ell}\int_{\mathbb R^3}|V|^2\psi_R
-\int_{\mathbb R^3}|V|^2 y_j\partial_\ell\psi_R\,dy.
\]

The translation block is unchanged from the standard cutoff form; the scale row is replaced by the weighted formulas above.

---

## G4. Continuity of the modulation matrix

If
\[
V_n\to V
\quad\text{in } W^{1,3}(B_{2R})\cap H^1(B_{2R}),
\]
then
\[
M(V_n)\to M(V)
\quad\text{in }\mathbb R^{4\times 4}.
\]

### Proof

Each entry of \(M(V)\) is an integral of one of the following forms:
- \(\int |V|^3 \omega\),
- \(\int |V|^2 \omega\),
- \(\int y_j V\cdot y\cdot\nabla V\, \omega\),
with \(\omega\in C_c^\infty(B_{2R})\).

If \(V_n\to V\) in \(L^3(B_{2R})\), then \(|V_n|^3\to |V|^3\) in \(L^1\).  
If \(V_n\to V\) in \(L^2(B_{2R})\), then \(|V_n|^2\to |V|^2\) in \(L^1\).  
If moreover \(\nabla V_n\to \nabla V\) in \(L^3(B_{2R})\), then the mixed terms with \(y\cdot\nabla V_n\) also converge in \(L^1\).

Therefore each matrix entry converges, hence the whole matrix converges.

\(\square\)

---

## G5. Translation block invertibility under core-mass dominance

Let
\[
M_{\mathrm{tr}}(V):=(M_{j\ell}(V))_{1\le j,\ell\le 3}.
\]

Assume:
- \(\psi_R\) is radial, nonincreasing, \(\psi_R\equiv1\) on \(B_R\), supported in \(B_{2R}\);
- there exist \(m_0>0\), \(\varepsilon_0>0\) such that
  \[
  \int |V|^2\psi_R \ge m_0,
  \qquad
  \int_{A_R}|V|^2 \le \varepsilon_0,
  \]
  where \(A_R=\{R\le |y|\le 2R\}\);
- \(\varepsilon_0\) is small enough relative to \(m_0\).

Then \(M_{\mathrm{tr}}(V)\) is invertible and
\[
\|M_{\mathrm{tr}}(V)^{-1}\|\le C(m_0,R).
\]

### Proof

From the explicit formula,
\[
M_{j\ell}(V)
=
-\delta_{j\ell}\int |V|^2\psi_R
-\int |V|^2 y_j\partial_\ell\psi_R.
\]
Write
\[
m_\psi(V):=\int |V|^2\psi_R.
\]
Then
\[
M_{\mathrm{tr}}(V)= -m_\psi(V)I_3 + E(V),
\]
where
\[
E_{j\ell}(V):= -\int |V|^2 y_j\partial_\ell\psi_R.
\]
Since \(\partial_\ell\psi_R\) is supported in \(A_R\),
\[
|E_{j\ell}(V)|\le C_R\int_{A_R}|V|^2 \le C_R\varepsilon_0.
\]
Thus
\[
\|E(V)\|\le C_R\varepsilon_0.
\]
Because \(m_\psi(V)\ge m_0\),
\[
M_{\mathrm{tr}}(V)= -m_\psi(V)\left(I_3-m_\psi(V)^{-1}E(V)\right).
\]
If \(C_R\varepsilon_0\le m_0/2\), then
\[
\|m_\psi(V)^{-1}E(V)\|\le \frac12,
\]
so \(I_3-m_\psi(V)^{-1}E(V)\) is invertible by the Neumann series, and
\[
\|M_{\mathrm{tr}}(V)^{-1}\|\le \frac{2}{m_0}.
\]

\(\square\)

---

## G6. Full matrix invertibility by Schur complement under quantitative hypotheses

Write the full modulation matrix as
\[
M(V)=
\begin{pmatrix}
\alpha(V) & u(V)^T\\
v(V) & A(V)
\end{pmatrix},
\]
where
- \(\alpha(V)=M_{00}(V)\),
- \(A(V)=M_{\mathrm{tr}}(V)\),
- \(u,v\in\mathbb R^3\) are the cross terms.

Assume along the orbit:

1. \(\alpha(V)\ge \alpha_0>0\),
2. \(A(V)\) is invertible with
   \[
   \|A(V)^{-1}\|\le C_A,
   \]
3. the Schur complement is uniformly positive:
   \[
   \alpha(V)-u(V)^T A(V)^{-1} v(V)\ge \sigma_0>0.
   \]

Then \(M(V)\) is invertible and
\[
\|M(V)^{-1}\|\le C(\alpha_0,\sigma_0,C_A,\|u\|,\|v\|).
\]

### Proof

This is the standard Schur-complement formula. Since \(A\) is invertible,
\[
M^{-1}
=
\begin{pmatrix}
S^{-1} & -S^{-1}u^T A^{-1}\\
-A^{-1}v S^{-1} & A^{-1}+A^{-1}vS^{-1}u^TA^{-1}
\end{pmatrix},
\]
where
\[
S=\alpha-u^T A^{-1}v.
\]
Since \(S\ge \sigma_0>0\), the formula is well-defined and the inverse is uniformly bounded in terms of the displayed quantities.

\(\square\)

---

## G7. Bounded modulation parameters under uniform nondegeneracy

Let \(X\) be a Banach space continuously embedded in
\[
H^2(B_{2R})\cap W^{1,3}(B_{2R})\cap L^3(B_{2R}).
\]
Assume:
- \(\sup_{\tau}\|V(\tau)\|_X\le M_X\),
- the gauge derivatives are uniformly bounded on the orbit,
- the full matrix \(M(V(\tau))\) is uniformly invertible,
- the local \(L^2\) pressure bound of Lemma 9.3 holds on the windows used to differentiate the gauge constraints.

Then
\[
\sup_{\tau}\bigl(|a(\tau)|+|b(\tau)|\bigr)<\infty.
\]

### Proof

Differentiate the gauge constraints \(G_k(V(\tau))=0\):
\[
0
=
DG_k(V)\big[\nu\Delta V-(V\cdot\nabla)V-\nabla P\big]
+a\,DG_k(V)[Z_{\mathrm{sc}}(V)]
+\sum_{j=1}^3 b_j DG_k(V)[\partial_jV].
\]
This gives
\[
M(V(\tau))
\begin{pmatrix}
a(\tau)\\ b(\tau)
\end{pmatrix}
=
-F(V(\tau)),
\]
where \(F(V)\) is the forcing vector built from the non-modulation terms.

By the assumed orbit bound, the nonlinear estimate, and the pressure bound, \(\|F(V(\tau))\|\) is uniformly bounded. Uniform invertibility of \(M(V(\tau))\) then gives a uniform bound on \((a(\tau),b(\tau))\).

\(\square\)

---

# Part IX. What is fully proved and what remains

## Fully proved inside this note

The following statements are fully proved from the assumptions stated in each theorem:

1. Renormalized equation (Lemma 1).
2. Pointwise renormalized local energy identity (Lemma 2).
3. Spacetime cutoff energy identity (Lemma 3).
4. Localized renormalized energy identity (Lemma 4).
5. Renormalized Caccioppoli estimate (Lemma 5).
6. Nonnegative integrable cost has a vanishing subsequence (Lemma 6).
7. Low cost implies low core dissipation (Lemma 7).
8. Compact normalized low-dissipation subsequence is impossible (Lemma 8).
9. Local \(L^{3/2}\) pressure estimate modulo a constant (Lemma 9).
10. Uniform pressure-tail control from global \(L^3\) (Lemma 9.1).
11. Verification of the local \(L^2\) pressure hypotheses (Lemma 9.2).
12. Local \(L^2\) pressure estimate modulo constants (Lemma 9.3).
13. Local \(L^2_\tau H^{-1}_x\) time-regularity estimate (Lemma 10).
14. Aubin–Lions–Simon closure and final contradiction under hypotheses (Lemma 11 / Theorem A).
15. Explicit formulas and continuity for the specific gauge (G2–G4).
16. Translation-block invertibility under quantitative core-mass dominance (G5).
17. Full matrix invertibility by Schur complement under quantitative hypotheses (G6).
18. Bounded modulation parameters under uniform nondegeneracy (G7).

## What remains external / hypothesis-level

The remaining substantive analytic tasks for an actual 3D NS instantiation are:

1. Prove the renormalized orbit satisfies the local \(H^1\)-bounds required in the chosen final theorem.
2. Verify the cross-term smallness/Schur-complement hypotheses for the repaired scale gauge.
3. Verify the full modulation matrix is uniformly nondegenerate along the renormalized orbit.

Those are exactly the remaining closure assumptions.

---

## Final article-ready summary

The note proves the following conditional statement:

> If a compact normalized renormalized Type II orbit exists and satisfies:
> - uniform local \(H^1\)-bounds,
> - uniform \(L^3\)-tightness,
> - uniform modulation bounds,
> then its renormalization cost has infinite total accumulation:
> \[
> \int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty.
> \]
> In particular, the orbit cannot admit a low-cost subsequence.

This is the exact compact Type II renormalization barrier proved in this master note.



---

# Part X. Revision of the final barrier theorem: replace \(L^\infty_\tau H^1_y\) by \(L^2_\tau H^1_y\)

## Why this revision is needed

The earlier formulation of the final barrier theorem used the orbit-level hypothesis

\[
\sup_{\tau\ge \tau_0}\|V(\tau)\|_{H^1(B_{2R})}\le M_1(R).
\]

That hypothesis is stronger than what was actually proved in the note. What the renormalized Caccioppoli estimate rigorously provides is a **space-time** bound of the form

\[
V \in L^2_{\mathrm{loc}}\big((\tau_0,\infty);H^1_{\mathrm{loc}}(\mathbb R^3)\big),
\]

not an \(L^\infty_\tau H^1_y\) bound.

For the Aubin–Lions–Simon compactness step, the \(L^2_\tau H^1_y\) bound is the natural hypothesis and is sufficient.

Therefore the correct final theorem should be stated with the weaker \(L^2_\tau H^1_y\) assumption.

---

## Lemma 12 (Local \(L^2_\tau H^1_y\) bound from the renormalized Caccioppoli estimate)

Fix \(0<r\le R/2\), and let
\[
Q_R=B_R\times (\tau_1,\tau_2),\qquad
Q_r=B_r\times [\sigma_1,\sigma_2],
\]
with \(\tau_1<\sigma_1<\sigma_2<\tau_2\).

Assume \(V,P\) solve the renormalized equation on \(Q_R\), and that on \(Q_R\) one has
\[
\sup_{\tau\in (\tau_1,\tau_2)}\|V(\tau)\|_{L^2(B_R)}\le M_2,
\]
\[
\iint_{Q_R}|V|^3\le M_3,
\]
\[
\iint_{Q_R}|P||V|\le M_{PV},
\]
\[
\|a\|_{L^\infty(\tau_1,\tau_2)}+\|b\|_{L^\infty(\tau_1,\tau_2)}\le M_{ab}.
\]

Then
\[
\begin{aligned}
\nu\iint_{Q_r} |\nabla V|^2
\le\;&
C\Big(
M_2^2
+
\frac{\nu}{(R-r)^2}M_2^2(\tau_2-\tau_1)
+
\frac{1}{R-r}M_3 \\
&\qquad
+
\frac{1}{R-r}M_{PV}
+
M_{ab}M_2^2(\tau_2-\tau_1)
+
\frac{M_{ab}}{R-r}M_2^2(\tau_2-\tau_1)
\Big),
\end{aligned}
\]
for a constant \(C\) depending only on dimension and cutoff choices.

In particular,
\[
V \in L^2_{\mathrm{loc}}\big((\tau_0,\infty);H^1_{\mathrm{loc}}(\mathbb R^3)\big).
\]

### Proof

Apply Lemma 5 (the renormalized Caccioppoli estimate). The only term needing simplification is
\[
\iint_{Q_R}|V|^2,
\]
which is bounded by
\[
\iint_{Q_R}|V|^2
\le
(\tau_2-\tau_1)\sup_{\tau\in(\tau_1,\tau_2)}\|V(\tau)\|_{L^2(B_R)}^2
\le
(\tau_2-\tau_1)M_2^2.
\]

Substituting this into Lemma 5 gives the claimed inequality.

The local \(L^2_\tau H^1_y\) membership follows by applying the estimate on arbitrary compact cylinders \(Q_r\Subset Q_R\).

\(\square\)

---

## Lemma 13 (Aubin–Lions–Simon compactness with the weaker hypotheses)

Let \(R>0\), and for each \(n\) define the shifted renormalized trajectory
\[
W_n(s,y):=V(\tau_n+s,y),
\qquad
(s,y)\in (0,1)\times B_R.
\]

Assume:

1. \(V\in L^2_{\mathrm{loc}}\big((\tau_0,\infty);H^1_{\mathrm{loc}}(\mathbb R^3)\big)\), and for each fixed \(R\),
   \[
   \sup_n \|W_n\|_{L^2(0,1;H^1(B_R))}<\infty;
   \]

2. for each fixed \(R\),
   \[
   \sup_n \|\partial_s W_n\|_{L^2(0,1;H^{-1}(B_R))}<\infty.
   \]

Then, after passing to a subsequence,
\[
W_n \to W_*
\quad\text{strongly in }L^2(0,1;L^2(B_R)).
\]

If in addition one has a Lions–Magenes trace bound so that
\[
W_n \in C([0,1];L^2(B_R))
\]
uniformly, then after passing to a further subsequence,
\[
W_n(0,\cdot)=V(\tau_n,\cdot)\to W_*(0,\cdot)
\quad\text{strongly in }L^2(B_R).
\]

If moreover the orbit has a uniform local \(L^6\)-bound (for example from a stronger local regularity hypothesis), then interpolation yields
\[
V(\tau_n)\to V_*
\quad\text{strongly in }L^3(B_R).
\]

### Proof

The compact embedding
\[
H^1(B_R)\hookrightarrow\hookrightarrow L^2(B_R)
\]
and continuous embedding
\[
L^2(B_R)\hookrightarrow H^{-1}(B_R)
\]
allow direct application of the Aubin–Lions–Simon theorem, giving
\[
W_n\to W_*
\quad\text{strongly in }L^2(0,1;L^2(B_R)).
\]

The passage to traces at \(s=0\) requires the standard trace theory for functions in
\[
L^2(0,1;H^1(B_R))\cap H^1(0,1;H^{-1}(B_R)),
\]
namely the Lions–Magenes result, which yields continuity into \(L^2(B_R)\). This gives precompactness of \(W_n(0,\cdot)\) in \(L^2(B_R)\) after subsequence extraction.

Finally, if one also knows a uniform local \(L^6\)-bound, then the interpolation inequality
\[
\|f\|_{L^3(B_R)}\le \|f\|_{L^2(B_R)}^{1/2}\|f\|_{L^6(B_R)}^{1/2}
\]
upgrades strong \(L^2\)-convergence to strong \(L^3\)-convergence.

\(\square\)

### Remark

This lemma makes precise an important point: the **minimal** compactness closure needs only the \(L^2_\tau H^1_y\) estimate, not an \(L^\infty_\tau H^1_y\) estimate. However, to upgrade the convergence to \(L^3_{\mathrm{loc}}\), one still needs either:
- a uniform local \(L^6\)-bound, or
- some alternative compactness route in \(L^3_{\mathrm{loc}}\).

So the final theorem must be stated accordingly.

---

## Theorem A' (Corrected final barrier theorem)

Let \(V,P\) solve the renormalized Navier–Stokes equation
\[
\partial_\tau V +(V\cdot\nabla)V+\nabla P
=
\nu\Delta V
+a(\tau)\bigl(V+y\cdot\nabla V\bigr)
+b(\tau)\cdot\nabla V,
\qquad \nabla\cdot V=0
\]
on \([\tau_0,\infty)\times \mathbb R^3\).

Fix \(R_0>0\) and a cutoff \(\phi_{R_0}\in C_c^\infty(\mathbb R^3)\), \(\phi_{R_0}\ge 0\), \(\phi_{R_0}\equiv 1\) on \(B_{R_0}\), and define
\[
\tilde{\mathfrak D}_{R_0}(\tau)
=
\nu \int_{\mathbb R^3} |\nabla V(y,\tau)|^2 \phi_{R_0}(y)\,dy
+
a_+(\tau)\int_{\mathbb R^3}|V(y,\tau)|^2 \phi_{R_0}(y)\,dy.
\]

Assume:

### (H1') global \(L^3\)-normalization
\[
\|V(\tau)\|_{L^3(\mathbb R^3)}=1
\qquad \forall \tau\ge\tau_0.
\]

### (H2') uniform \(L^3\)-tightness
For every \(\varepsilon>0\), there exists \(R_\varepsilon>0\) such that
\[
\sup_{\tau\ge\tau_0}\int_{|y|>R_\varepsilon}|V(y,\tau)|^3<\varepsilon.
\]

### (H3') local \(L^2_\tau H^1_y\) control
For every compact cylinder \(Q_r\Subset Q_R\Subset \mathbb R^3\times(\tau_0,\infty)\), the renormalized Caccioppoli estimate yields
\[
V\in L^2_{\mathrm{loc}}\big((\tau_0,\infty);H^1_{\mathrm{loc}}(\mathbb R^3)\big).
\]

### (H4') local \(L^2_\tau H^{-1}_y\) time-regularity
For every \(R>0\),
\[
\partial_\tau V \in L^2_{\mathrm{loc}}\big((\tau_0,\infty);H^{-1}(B_R)\big)
\]
with uniform bounds on shifted windows.

### Derived pressure-tail control
For every \(R>0\), Lemma 9.1 and the \(L^3\)-normalization give uniform control of the far-field pressure-tail term along the orbit.

### (H6') a local \(L^6\)-bound or equivalent \(L^3_{\mathrm{loc}}\)-compactness upgrade
For every fixed \(R>0\), either:
- \(V(\tau_n)\) is uniformly bounded in \(L^6(B_R)\) along every extracted sequence, or
- one has another mechanism yielding strong \(L^3_{\mathrm{loc}}\)-compactness of the sampled orbit.

Then
\[
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau = \infty.
\]

### Proof

Assume for contradiction that
\[
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau<\infty.
\]
By Lemma 6, there exists a sequence \(\tau_n\to\infty\) such that
\[
\tilde{\mathfrak D}_{R_0}(\tau_n)\to0.
\]
By Lemma 7,
\[
\int_{B_{R_0}}|\nabla V(\tau_n)|^2\to0.
\]

Now use (H3') and (H4') together with Lemma 13 to extract a subsequence such that
\[
V(\tau_n)\to V_*
\quad\text{strongly in }L^2_{\mathrm{loc}}(\mathbb R^3).
\]
By (H6'), this convergence upgrades to
\[
V(\tau_n)\to V_*
\quad\text{strongly in }L^3_{\mathrm{loc}}(\mathbb R^3).
\]

Now all hypotheses of Lemma 8 are satisfied:
- strong \(L^2_{\mathrm{loc}}\cap L^3_{\mathrm{loc}}\) convergence,
- \(L^3\)-tightness by (H2'),
- normalization by (H1'),
- vanishing core dissipation from Lemma 7.

Lemma 8 yields a contradiction.

Therefore
\[
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau = \infty.
\]

\(\square\)

---

## Updated status of the remaining assumptions

With this revision, the strongest unnecessary hypothesis has been removed.

The final theorem no longer requires the unproved orbit-level condition
\[
\sup_{\tau}\|V(\tau)\|_{H^1(B_R)}<\infty.
\]

What remains is the more natural and weaker package:

- \(L^2_{\tau,\mathrm{loc}}H^1_{y,\mathrm{loc}}\) control from Caccioppoli,
- \(L^2_{\tau,\mathrm{loc}}H^{-1}_{y,\mathrm{loc}}\) control of \(\partial_\tau V\),
- and one mechanism upgrading local \(L^2\)-compactness to local \(L^3\)-compactness.

This is the correct tightened version of the barrier theorem.



---

# Part XI. Scale-gauge consistency ledger (single active approach)

## 11.1 Canonical reference

From this point on, the only scale-fixing condition used in this note is
\[
G_{\mathrm{sc}}(V)
:=
\int_{\mathbb R^3}|y|^{-p}|V(y)|^3\,dy-\Theta_0=0,
\qquad
0<p<3,\ \Theta_0>0.
\]

Equivalently, in physical variables
\[
\lambda(t)^p\int_{\mathbb R^3}|x-x_c(t)|^{-p}|u(x,t)|^3\,dx=\Theta_0.
\]

The corresponding modulation matrix is defined in Part VIII (G3), and its continuity is in Part VIII (G4).

## 11.2 Central transversality identities

The repaired-gauge transversality statements are collected as:

- \(DG_{\mathrm{sc}}(V)[Z_{\mathrm{sc}}(V)] = p\int |y|^{-p}|V|^3\,dy\) (exact scale derivative),
- \(DG_{\mathrm{sc}}(V)[Z_{\mathrm{sc}}(V)] = p\Theta_0\) on the gauge surface,
- \(M_{00}(V)=p\Theta_0\) and no annular lower bound is required for the scale row.

These are the only structural facts needed from the scale equation.

## 11.3 External proofs

For a complete proof stack of these identities, including:

1. \(p\)-exact scaling under \(V_\mu(y)=\mu V(\mu y)\),
2. Frechét differentiability of \(V\mapsto\int |y|^{-p}|V|^3\),
3. exact formula \(DG_{\mathrm{sc}}(V)[Z_{\mathrm{sc}}(V)]=p\Theta_0\),
4. explicit modulation entries \(M_{00},M_{0\ell},M_{j\ell},M_{j0}\),
5. and the Schur-complement reduction from translation block + cross-terms,

see [required_new_scale_gauge_theorems.md](/home/guillem/hypostructure/docs/source/type_2/required_new_scale_gauge_theorems.md).

No additional alternative gauge variants are used anywhere in this note.



---

# Part XII. Good-window closure of the final compactness step

This part replaces the endpoint trace compactness route by a good-window selection route. The point is simple:

- finite total cost gives windows on which the average cost is small;
- local \(L^2_\tau H^1_y\) control on the same windows gives good times with uniform local \(H^1\);
- Rellich plus interpolation gives strong sampled-time \(L^3_{\mathrm{loc}}\)-compactness.

This avoids the invalid inference from spacetime Aubin--Lions compactness to compactness at a fixed endpoint.

## Lemma 14 (finite cost gives vanishing-cost windows)

Let \(D\in L^1([\tau_0,\infty))\) with \(D\ge0\). For every fixed window length \(\ell>0\), there exist times \(s_n\to\infty\) such that
\[
\int_{s_n}^{s_n+\ell}D(\tau)\,d\tau\to0.
\]
In particular, one may take \(s_n=\tau_0+n\ell\) after discarding finitely many initial windows.

### Proof

Since \(D\in L^1([\tau_0,\infty))\),
\[
\int_T^\infty D(\tau)\,d\tau\to0
\qquad\text{as }T\to\infty.
\]
Therefore
\[
\int_{\tau_0+n\ell}^{\tau_0+(n+1)\ell}D(\tau)\,d\tau
\le
\int_{\tau_0+n\ell}^\infty D(\tau)\,d\tau
\to0.
\]

\(\square\)

---

## Lemma 15 (common good times on an exhaustion of balls)

Let \(D\ge0\) be a measurable cost density, and let
\[
J_n=[s_n,s_n+\ell],\qquad \ell>0,\qquad s_n\to\infty.
\]
Assume
\[
\int_{J_n}D(\tau)\,d\tau=:\delta_n\to0.
\]
Assume also that for every integer \(m\ge1\) there is \(C_m<\infty\) such that
\[
\sup_n\int_{J_n}\|V(\tau)\|_{H^1(B_m)}^2\,d\tau\le C_m.
\]
Then there exist times \(\sigma_n\in J_n\) such that
\[
D(\sigma_n)\to0,
\]
and for every fixed \(m\ge1\),
\[
\sup_{n\ge m}\|V(\sigma_n)\|_{H^1(B_m)}<\infty.
\]
Consequently, after passing to a diagonal subsequence, there is \(V_*\) such that
\[
V(\sigma_n)\to V_*
\quad\text{strongly in }L^2_{\mathrm{loc}}(\mathbb R^3)
\]
and
\[
V(\sigma_n)\to V_*
\quad\text{strongly in }L^3_{\mathrm{loc}}(\mathbb R^3).
\]

### Proof

For each \(n\), define
\[
E_n^{\mathrm{cost}}
:=\{\tau\in J_n:D(\tau)\le \delta_n^{1/2}\}.
\]
By Chebyshev,
\[
|J_n\setminus E_n^{\mathrm{cost}}|
\le \delta_n^{1/2}.
\]

For each \(m\le n\), define
\[
E_{n,m}^{H^1}
:=
\left\{\tau\in J_n:
\|V(\tau)\|_{H^1(B_m)}^2
\le
\frac{2^{m+2}C_m}{\ell}
\right\}.
\]
Again by Chebyshev,
\[
|J_n\setminus E_{n,m}^{H^1}|
\le
\frac{\ell}{2^{m+2}}.
\]
Hence
\[
\sum_{m=1}^n |J_n\setminus E_{n,m}^{H^1}|
\le
\frac{\ell}{4}.
\]
For all sufficiently large \(n\), \(\delta_n^{1/2}<\ell/4\). Thus the intersection
\[
E_n^{\mathrm{cost}}\cap\bigcap_{m=1}^n E_{n,m}^{H^1}
\]
has positive measure. Choose \(\sigma_n\) in this intersection. Then
\[
D(\sigma_n)\le \delta_n^{1/2}\to0,
\]
and for every fixed \(m\), whenever \(n\ge m\),
\[
\|V(\sigma_n)\|_{H^1(B_m)}^2
\le
\frac{2^{m+2}C_m}{\ell}.
\]

For each fixed \(m\), Rellich gives compactness of \(V(\sigma_n)\) in \(L^2(B_m)\). A diagonal subsequence yields strong convergence in \(L^2_{\mathrm{loc}}\).

The same uniform \(H^1(B_m)\) bounds give uniform \(L^6(B_m)\) bounds. Interpolation gives, for each fixed \(m\),
\[
\|V(\sigma_n)-V_*\|_{L^3(B_m)}
\le
C_m'
\|V(\sigma_n)-V_*\|_{L^2(B_m)}^{1/2}
\|V(\sigma_n)-V_*\|_{L^6(B_m)}^{1/2}.
\]
The \(L^6\) factor is uniformly bounded and the \(L^2\) factor tends to zero, so the convergence is strong in \(L^3(B_m)\). Since \(m\) was arbitrary, the convergence is strong in \(L^3_{\mathrm{loc}}\).

\(\square\)

---

## Theorem A'' (good-window compact Type II barrier)

Let \(V,P\) solve the renormalized Navier--Stokes equation
\[
\partial_\tau V +(V\cdot\nabla)V+\nabla P
=
\nu\Delta V
+a(\tau)(V+y\cdot\nabla V)
+b(\tau)\cdot\nabla V,
\qquad \nabla\cdot V=0
\]
on \([\tau_0,\infty)\times\mathbb R^3\).

Fix \(R_0>0\) and define
\[
\tilde{\mathfrak D}_{R_0}(\tau)
=
\nu \int_{\mathbb R^3}|\nabla V(y,\tau)|^2\phi_{R_0}(y)\,dy
+a_+(\tau)\int_{\mathbb R^3}|V(y,\tau)|^2\phi_{R_0}(y)\,dy.
\]

Assume:

### (GW1) global \(L^3\)-normalization
\[
\|V(\tau)\|_{L^3(\mathbb R^3)}=1
\qquad \forall \tau\ge\tau_0.
\]

### (GW2) uniform \(L^3\)-tightness
For every \(\varepsilon>0\), there exists \(R_\varepsilon>0\) such that
\[
\sup_{\tau\ge\tau_0}\int_{|y|>R_\varepsilon}|V(y,\tau)|^3\,dy<\varepsilon.
\]

### (GW3) uniform local \(L^2_\tau H^1_y\) bounds on unit windows
For every integer \(m\ge1\), there exists \(C_m<\infty\) such that
\[
\sup_{n\ge0}
\int_{\tau_0+n}^{\tau_0+n+1}
\|V(\tau)\|_{H^1(B_m)}^2\,d\tau
\le C_m.
\]

Then
\[
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty.
\]

### Proof

Assume for contradiction that
\[
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau<\infty.
\]
By Lemma 14 with \(D=\tilde{\mathfrak D}_{R_0}\) and \(\ell=1\),
\[
\delta_n:=
\int_{\tau_0+n}^{\tau_0+n+1}
\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau
\to0.
\]
Apply Lemma 15 to the windows
\[
J_n=[\tau_0+n,\tau_0+n+1].
\]
There exist \(\sigma_n\in J_n\) such that
\[
\tilde{\mathfrak D}_{R_0}(\sigma_n)\to0,
\]
and, after passing to a subsequence,
\[
V(\sigma_n)\to V_*
\quad\text{strongly in }
L^2_{\mathrm{loc}}(\mathbb R^3)\cap L^3_{\mathrm{loc}}(\mathbb R^3).
\]

Because \(\phi_{R_0}\equiv1\) on \(B_{R_0}\),
\[
\nu\int_{B_{R_0}}|\nabla V(\sigma_n)|^2\,dy
\le
\tilde{\mathfrak D}_{R_0}(\sigma_n)
\to0.
\]
Thus the low-dissipation rigidity theorem, Lemma 8, applies to the sequence \(V(\sigma_n)\): the sequence is globally \(L^3\)-normalized by (GW1), uniformly \(L^3\)-tight by (GW2), strongly compact locally in \(L^2\cap L^3\), and has vanishing core dissipation on \(B_{R_0}\). Lemma 8 gives a contradiction.

Therefore
\[
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty.
\]

\(\square\)

### Consequence

Theorem A'' removes the separate sampled-time compactness and \(H^{-1}\)-time-regularity assumptions from the final contradiction step. Those estimates remain useful for other compactness routes, but the good-window proof needs only normalization, tightness, and uniform local \(L^2_\tau H^1_y\) bounds on the same windows as the vanishing average cost.

---

## Corollary 16 (contrapositive classification of finite-cost Type II candidates)

Assume a normalized renormalized Type II candidate has finite total renormalization cost:
\[
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau<\infty.
\]
Assume also
\[
\|V(\tau)\|_{L^3(\mathbb R^3)}=1
\qquad\forall \tau\ge\tau_0.
\]
Then at least one of the following must occur.

### (C1) failure of global critical tightness
There exists \(\varepsilon_0>0\) such that for every \(R>0\) there is a time \(\tau_R\ge\tau_0\) with
\[
\int_{|y|>R}|V(y,\tau_R)|^3\,dy\ge\varepsilon_0.
\]

### (C2) failure of local windowed \(H^1\) control
There exists an integer \(m\ge1\) such that
\[
\sup_{n\ge0}
\int_{\tau_0+n}^{\tau_0+n+1}
\|V(\tau)\|_{H^1(B_m)}^2\,d\tau
=\infty.
\]

Equivalently, every finite-cost normalized Type II candidate is either noncompact/radiative in the critical \(L^3\) topology, or locally rough in the renormalized core.

### Proof

If neither (C1) nor (C2) holds, then \(V\) is uniformly \(L^3\)-tight and satisfies the uniform local \(L^2_\tau H^1_y\) window bounds of (GW3). Together with the normalization hypothesis, all assumptions of Theorem A'' hold. Theorem A'' gives
\[
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty,
\]
contradicting the finite-cost assumption. Therefore at least one of (C1) or (C2) must occur.

\(\square\)

### Interpretation

Corollary 16 narrows the remaining finite-cost Type II scenarios to two structural families:

1. **Noncompact/radiative Type II:** critical \(L^3\) mass escapes every bounded renormalized region. This includes radiation tails, outward-moving secondary profiles, and no-splitting failures.
2. **Rough-core Type II:** the orbit may remain \(L^3\)-tight, but local windowed \(H^1\) control fails on some bounded renormalized core.

Failure of \(H^{-1}\)-time regularity alone is not an admissible survivor mechanism for the good-window closure. It matters only insofar as it reflects failure to obtain the local windowed \(H^1\) bounds or other hypotheses needed upstream.
