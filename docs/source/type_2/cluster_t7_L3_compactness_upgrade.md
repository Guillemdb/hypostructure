# Cluster T7: strong local \(L^3\)-compactness upgrade

This document implements the next cluster after the immediate high-yield one: the sampled-sequence upgrade
from local compactness in \(L^2\) to compactness in \(L^3\).

Assumptions and notation are aligned with
[compact_typeII_master_note_repaired_gauge.md](compact_typeII_master_note_repaired_gauge.md).

---

Let \(R>0\), \(I=(0,1)\), and for each index \(n\) define shifted trajectories
\[
W_n(s,y):=V(\tau_n+s,y),
\qquad (s,y)\in I\times B_{2R},
\]
where \(\tau_n\to\infty\).

## T7.1 (spacetime compactness in \(L^2_tL^2\))

Assume:

1. \(\sup_n \|W_n\|_{L^2(I;H^1(B_{2R}))} < \infty\).
2. \(\sup_n \|\partial_s W_n\|_{L^2(I;H^{-1}(B_R))}<\infty\).

Then there exists a subsequence (not relabelled) and a limit \(W\in L^2(I;L^2(B_R))\) such that
\[
W_n \to W \quad \text{strongly in }L^2(I;L^2(B_R)).
\]

### Proof
Compactness of embeddings
\(H^1(B_R)\hookrightarrow\hookrightarrow L^2(B_R)\hookrightarrow H^{-1}(B_R)\)
with bounds in (1)–(2) gives this from Aubin–Lions–Simon exactly as in Lemma 11 and Lemma 13 in the master-note stack.

---

## T7.2 (spacetime upgrade to \(L^2_tL^3\) from local \(L^6\)-type control)

Assume T7.1 hypotheses and additionally

3. \(\sup_n \|W_n\|_{L^2(I;L^6(B_{2R}))}<\infty\).

Then, passing to the same subsequence,
\[
W_n \to W \quad \text{strongly in }L^2(I;L^3(B_R)).
\]

### Proof
By assumption 3 and reflexivity, after passing to a further subsequence
\[
W_n\rightharpoonup \widetilde W
\quad\text{weakly in }L^2(I;L^6(B_R)).
\]
The strong \(L^2(I;L^2(B_R))\)-limit from T7.1 identifies \(\widetilde W=W\). Hence
\[
W\in L^2(I;L^6(B_R))
\]
and \(\|W_n-W\|_{L^2(I;L^6(B_R))}\) is uniformly bounded.

By Gagliardo–Nirenberg on bounded sets,
\[
\|W_n-W\|_{L^3(B_R)}
\le
C(R)\|W_n-W\|_{L^2(B_R)}^{1/2}\|W_n-W\|_{L^6(B_R)}^{1/2}.
\]
Square and integrate in \(s\):
\[
\int_0^1\|W_n-W\|_{L^3}^2 ds
\le
C\int_0^1\|W_n-W\|_{L^2}\|W_n-W\|_{L^6}
\,ds.
\]
Apply Hölder in time to obtain
\[
\|W_n-W\|_{L^2(I;L^3(B_R))}^2
\le
C\|W_n-W\|_{L^2(I;L^2(B_R))}\,\|W_n-W\|_{L^2(I;L^6(B_R))}.
\]
The first factor tends to zero by T7.1 and the second is uniformly bounded, so the left side tends to zero.

---

## T7.3 (sampled-time \(L^3\) upgrade from sampled-time \(L^2\) compactness)

Assume that, after passing to a subsequence, the sampled states satisfy

4. \(W_n(0,\cdot)\to f\) strongly in \(L^2(B_R)\).
5. \(\sup_n\|W_n(0,\cdot)\|_{L^6(B_R)}<\infty\).

Then
\[
W_n(0,\cdot)\to f\quad \text{strongly in }L^3(B_R).
\]

### Proof
By assumption 5 and weak compactness, the strong \(L^2\)-limit \(f\) belongs to
\(L^6(B_R)\), and
\[
\sup_n\|W_n(0,\cdot)-f\|_{L^6(B_R)}<\infty.
\]
Interpolating at \(s=0\),
\[
\|W_n(0)-f\|_{L^3(B_R)}
\le
C(R)\|W_n(0)-f\|_{L^2(B_R)}^{1/2}\|W_n(0)-f\|_{L^6(B_R)}^{1/2}.
\]
The second factor is bounded and the first tends to zero by assumption 4, so the right-hand side tends to zero.

### Important limitation

T7.1 and T7.2 are spacetime compactness statements. They do not, by themselves, imply convergence at the fixed time \(s=0\). A separate sampled-time \(L^2\)-compactness mechanism is required for T7.3.

A sufficient mechanism is Simon compactness in the form:

6. \(\sup_n\|W_n\|_{L^\infty(I;H^1(B_R))}<\infty\).
7. \(\sup_n\|\partial_sW_n\|_{L^q(I;H^{-1}(B_R))}<\infty\) for some \(q>1\).

Then \((W_n)\) is precompact in \(C([0,1];L^2(B_R))\), so assumption 4 follows after subsequence extraction. Without such an additional input, fixed-time convergence at \(s=0\) is not a consequence of Aubin--Lions in \(L^2(I;L^2)\).

---

## T7.4 (good-time replacement for sampled-time compactness)

Let \(J_n=[s_n,s_n+\ell]\) be time windows with fixed length \(\ell>0\), and let \(\delta_n\downarrow0\). Assume for a fixed radius \(R\):

6. The localized cost is small on average:
\[
\int_{J_n}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau\le \delta_n.
\]
7. The local \(H^1\) energy is uniformly integrable on the same windows:
\[
\int_{J_n}\|V(\tau)\|_{H^1(B_R)}^2\,d\tau\le C_R.
\]

Then there are times \(\sigma_n\in J_n\) such that
\[
\tilde{\mathfrak D}_{R_0}(\sigma_n)\to0,
\qquad
\|V(\sigma_n)\|_{H^1(B_R)}\le \left(\frac{2C_R}{\ell}\right)^{1/2}.
\]
Consequently, after passing to a subsequence,
\[
V(\sigma_n)\to V_*
\quad\text{strongly in }L^2(B_R)
\]
and
\[
V(\sigma_n)\to V_*
\quad\text{strongly in }L^3(B_R).
\]

### Proof

Define
\[
E_n:=\left\{\tau\in J_n:\tilde{\mathfrak D}_{R_0}(\tau)\le \delta_n^{1/2}\right\}.
\]
By Chebyshev,
\[
|J_n\setminus E_n|\le\delta_n^{1/2}.
\]
For all sufficiently large \(n\), \(|E_n|\ge \ell/2\).

Since
\[
\int_{E_n}\|V(\tau)\|_{H^1(B_R)}^2\,d\tau\le C_R,
\]
there exists \(\sigma_n\in E_n\) such that
\[
\|V(\sigma_n)\|_{H^1(B_R)}^2\le \frac{2C_R}{\ell}.
\]
Because \(\sigma_n\in E_n\),
\[
\tilde{\mathfrak D}_{R_0}(\sigma_n)\le\delta_n^{1/2}\to0.
\]
The uniform \(H^1(B_R)\) bound gives, by Rellich, a subsequence converging strongly in \(L^2(B_R)\).

The same \(H^1(B_R)\) bound gives a uniform \(L^6(B_R)\) bound. Interpolation gives
\[
\|V(\sigma_n)-V_*\|_{L^3(B_R)}
\le
C_R\|V(\sigma_n)-V_*\|_{L^2(B_R)}^{1/2}
\|V(\sigma_n)-V_*\|_{L^6(B_R)}^{1/2}.
\]
The \(L^6\) factor is uniformly bounded and the \(L^2\) factor tends to zero, proving strong \(L^3(B_R)\) convergence.

### Role in the barrier proof

T7.4 avoids endpoint trace compactness. Instead of first fixing times \(\tau_n\) and then trying to prove compactness of \(V(\tau_n)\), one selects low-cost times inside windows where the same windows also carry local \(H^1\) control. This discharges the abstract sampled-time \(L^2\)-compactness hypothesis in T7.3 whenever the proof can arrange common good windows.

---

## Corollary for renormalized times

Assume that for each fixed local radius \(R\), the shifted sequence satisfies T7.1 and the sampled states satisfy the two assumptions of T7.3. Then, after passing to a subsequence,
\[
V(\tau_n) \to V_* \quad\text{strongly in }L^3_{\mathrm{loc}}(\mathbb R^3)
\]
where \(V_*\) is the local \(L^2\)-limit of the sampled states.

For clarity, the chain is:

1. Caccioppoli-type input gives (1).
2. Time-regularity input (T6) gives (2).
3. A local \(L^6\)-type upgrade gives (3) and sampled-time \(L^6\) boundedness.
4. T7.1 and T7.2 give spacetime \(L^3\)-compactness.
5. An additional sampled-time \(L^2\)-compactness mechanism gives T7.3.
6. T7.3 gives sampled-time strong \(L^3\)-compactness needed for the low-dissipation contradiction theorem.

This is the missing local compactness piece after the implemented immediate cluster.

## Corollary using good times

If there is a single sequence of selected times \(\sigma_n\) such that, for every fixed local radius \(R\),
\[
\sup_n\|V(\sigma_n)\|_{H^1(B_R)}<\infty
\]
and
\[
\tilde{\mathfrak D}_{R_0}(\sigma_n)\to0,
\]
then
\[
V(\sigma_n)\to V_*
\quad\text{strongly in }L^3_{\mathrm{loc}}(\mathbb R^3)
\]
after diagonal subsequence extraction. T7.4 supplies such a sequence for any fixed radius; to get full \(L^3_{\mathrm{loc}}\) compactness, the good-time selection must be made on common windows with local \(H^1\) control on an exhaustion of balls. In that case, T7.4 replaces the separate sampled-time compactness input in T7.3.
