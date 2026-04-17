# Structural cluster T8: globally compact branch implies tightness and barrier closure

This document implements the conditional structural form of T8. The input is
global \(L^3(\mathbb R^3)\)-precompactness of a single gauge-fixed
renormalized branch. The output is the exact local CKN compact-cylinder retention and
sampled compactness input used by the compact Type II barrier.

The statements are purely functional-analytic. They do not prove that
membership in an abstract Type II branch implies local CKN-precompactness.
That bridge belongs to the structural no-radiation/no-splitting program. The
result proved here is that, once the branch is globally compact in the critical
\(L^3\) topology after gauge fixing, the noncompact/radiative escape route is
closed.

Throughout, \(\mathcal O\subset L^3(\mathbb R^3;\mathbb R^3)\) denotes the
renormalized orbit
\[
\mathcal O:=\{V(\tau):\tau\ge \tau_0\}.
\]

---

## T8.1 (Compact subsets of \(L^3\) are uniformly tight)

Let \(K\subset L^3(\mathbb R^3;\mathbb R^3)\) be compact. Then \(K\) is
uniformly \(L^3\)-tight: for every \(\varepsilon>0\), there exists \(R>0\)
such that
\[
\sup_{f\in K}\int_{|y|>R}|f(y)|^3\,dy<\varepsilon.
\]

### Proof

Fix \(\varepsilon>0\). Set
\[
\delta:=\left(\frac{\varepsilon}{2^{4}}\right)^{1/3}.
\]
Since \(K\) is compact in \(L^3\), there exist \(f_1,\dots,f_N\in K\) such
that
\[
K\subset \bigcup_{j=1}^N B_{L^3}(f_j,\delta),
\]
where the balls are taken in \(L^3(\mathbb R^3)\).

For each \(j\), absolute continuity of the Lebesgue integral gives a radius
\(R_j>0\) such that
\[
\int_{|y|>R_j}|f_j(y)|^3\,dy<\frac{\varepsilon}{2^{4}}.
\]
Set
\[
R:=\max_{1\le j\le N}R_j.
\]
Then
\[
\int_{|y|>R}|f_j(y)|^3\,dy<\frac{\varepsilon}{2^{4}}
\qquad\text{for every }1\le j\le N.
\]

Let \(f\in K\). Pick \(j\) with
\[
\|f-f_j\|_{L^3(\mathbb R^3)}<\delta.
\]
Using
\[
|a+b|^3\le 4(|a|^3+|b|^3),
\]
we get
\[
\int_{|y|>R}|f(y)|^3\,dy
\le
4\int_{|y|>R}|f-f_j|^3\,dy
+4\int_{|y|>R}|f_j|^3\,dy.
\]
The first term is bounded by \(4\delta^3=\varepsilon/4\), and the second is
bounded by \(4\varepsilon/2^4=\varepsilon/4\). Hence
\[
\int_{|y|>R}|f(y)|^3\,dy<\varepsilon.
\]
Taking the supremum over \(f\in K\) proves the claim.

\(\square\)

---

## T8.2 (Compact branch gives sampled \(L^3\)-precompactness)

Assume the renormalized orbit \(\mathcal O\) has compact closure in
\(L^3(\mathbb R^3)\). Then every sequence \(\tau_n\to\infty\) has a
subsequence \(\tau_{n_k}\) and a profile \(V_*\in L^3(\mathbb R^3)\) such that
\[
V(\tau_{n_k})\to V_*
\quad\text{strongly in }L^3(\mathbb R^3).
\]
In particular,
\[
V(\tau_{n_k})\to V_*
\quad\text{strongly in }L^3_{\mathrm{loc}}(\mathbb R^3).
\]

### Proof

Let
\[
K:=\overline{\mathcal O}^{\,L^3(\mathbb R^3)}.
\]
By hypothesis \(K\) is compact. The sequence \(V(\tau_n)\) lies in \(K\).
Sequential compactness of compact metric spaces gives a subsequence
\(V(\tau_{n_k})\) and \(V_*\in K\) such that
\[
\|V(\tau_{n_k})-V_*\|_{L^3(\mathbb R^3)}\to0.
\]
Strong convergence on \(\mathbb R^3\) implies strong convergence on every
bounded ball.

\(\square\)

---

## T8.3 (Compact normalized branch supplies the rigidity compactness inputs)

Assume:

1. compact critical branch:
   \[
   K:=\overline{\mathcal O}^{\,L^3(\mathbb R^3)}
   \quad\text{is compact in }L^3(\mathbb R^3);
   \]
2. normalization:
   \[
   \|V(\tau)\|_{L^3(\mathbb R^3)}=1
   \qquad\text{for all }\tau\ge\tau_0;
   \]
3. local \(L^2\)-boundedness on sampled subsequences:
   for every \(R>0\) and every sequence \(\tau_n\to\infty\),
   \[
   \sup_n\|V(\tau_n)\|_{L^2(B_R)}<\infty.
   \]

Then every sequence \(\tau_n\to\infty\) admits a subsequence, not relabelled,
and a profile \(V_*\) such that:
\[
V(\tau_n)\to V_*
\quad\text{strongly in }L^3_{\mathrm{loc}}(\mathbb R^3),
\]
\[
V(\tau_n)\rightharpoonup V_*
\quad\text{weakly in }L^2(B_R)
\quad\text{for every }R>0,
\]
and the sequence is uniformly \(L^3\)-tight:
\[
\forall \varepsilon>0\ \exists R_\varepsilon>0:
\sup_n\int_{|y|>R_\varepsilon}|V(\tau_n,y)|^3\,dy<\varepsilon.
\]

If, in addition,
\[
V(\tau_n)\to V_*
\quad\text{strongly in }L^2_{\mathrm{loc}}(\mathbb R^3),
\]
then the normalization in assumption 2, the tightness above, and the following
strong compactness statement give the normalization, tightness, and compactness
inputs of the low-dissipation rigidity lemma:
\[
V(\tau_n)\to V_*
\quad\text{strongly in }
L^2_{\mathrm{loc}}(\mathbb R^3)\cap L^3_{\mathrm{loc}}(\mathbb R^3).
\]

### Proof

By T8.2, after passing to a subsequence,
\[
V(\tau_n)\to V_*
\quad\text{strongly in }L^3(\mathbb R^3),
\]
hence strongly in \(L^3_{\mathrm{loc}}\).

By T8.1 applied to the compact set \(K\), the full orbit \(\mathcal O\), and
therefore the sampled subsequence, is uniformly \(L^3\)-tight.

Fix \(R>0\). Assumption 3 gives boundedness of \(V(\tau_n)\) in \(L^2(B_R)\).
Since \(L^2(B_R)\) is reflexive, a further subsequence converges weakly in
\(L^2(B_R)\) to some \(W_R\). The strong \(L^3(B_R)\)-limit is \(V_*\).
Testing against \(C_c^\infty(B_R)\), which lies in both \(L^2(B_R)\) and
\(L^{3/2}(B_R)\), identifies \(W_R=V_*\). A diagonal extraction over
\(R=1,2,\dots\) gives weak convergence in \(L^2(B_m)\) for every integer
\(m\ge1\). For an arbitrary radius \(R>0\), choose an integer \(m>R\). Weak
convergence in \(L^2(B_m)\) implies weak convergence after restriction to
\(L^2(B_R)\).

The final assertion is immediate after adding strong
\(L^2_{\mathrm{loc}}\)-convergence.

\(\square\)

---

## T8.4 (Globally \(L^3\)-compact branch closes the noncompactness alternative)

Assume:

1. \(\overline{\mathcal O}^{\,L^3(\mathbb R^3)}\) is compact;
2. \(\|V(\tau)\|_{L^3(\mathbb R^3)}=1\) for all \(\tau\ge\tau_0\);
3. for every \(R>0\),
   \[
   \sup_{T\ge\tau_0}
   \int_T^{T+1}\|V(\tau)\|_{H^1(B_R)}^2\,d\tau<\infty.
   \]
4. \(V\) is an admissible renormalized orbit to which Theorem A'' in
   the master note applies.

Then the noncompact/radiative Type II alternative is excluded:
\[
\forall\varepsilon>0\ \exists R_\varepsilon>0:
\sup_{\tau\ge\tau_0}\int_{|y|>R_\varepsilon}|V(y,\tau)|^3\,dy<\varepsilon.
\]
Moreover, if the total localized renormalization cost is finite, then the
good-window compactness mechanism of Theorem A'' applies and yields the
low-dissipation rigidity contradiction. Hence
\[
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty
\]
for every fixed core radius \(R_0\) for which the localized cost is defined.

### Proof

The tightness conclusion is exactly T8.1 applied to
\[
K=\overline{\mathcal O}^{\,L^3(\mathbb R^3)}.
\]
Assumptions 2 and 3 are precisely the normalization and windowed local
\(H^1\)-control hypotheses in Theorem A''. Therefore, under finite total cost,
Theorem A'' produces sampled times with strong local compactness and vanishing
core dissipation. The low-dissipation rigidity lemma then contradicts
normalization and tightness. Thus finite total localized renormalization cost
is impossible.

\(\square\)

---

## Consequence for the structural program

T8 reduces the globally \(L^3\)-compact branch case to the compact Type II
barrier. A finite-cost Type II candidate with a globally compact \(L^3\)-branch
must therefore fail at least one hypothesis needed to invoke the compact
barrier theorem, such as:

- uniform local \(L^2_\tau H^1_y\) control on unit windows;
- admissibility of the renormalized formulation used by Theorem A'';
- representation by one globally compact normalized branch in \(L^3(\mathbb R^3)\).

The remaining structural tasks T9--T11 are therefore focused on the mechanisms
that prevent compact branch reduction: radiation, splitting, and failure of
single-profile irreducibility.

The branch-membership implication
\[
\text{Type II branch membership}
\quad\Longrightarrow\quad
\overline{\mathcal O}^{\,L^3(\mathbb R^3)}\text{ compact}
\]
is not proved in this file. It is the global structural bridge targeted by the
no-radiation, no-splitting, and irreducibility steps.

---

# T9. No-radiation criteria

T9 isolates the precise analytic content of the no-radiation step. The
statements below do not assume a particular scattering theory. They give
checkable conditions under which a compact core plus a remainder has no
nontrivial critical \(L^3\) radiation and therefore reduces to the globally
compact branch case handled by T8.

## Definition T9.0 (Radiative critical mass)

Let \(\mathcal O=\{V(\tau):\tau\ge\tau_0\}\subset L^3(\mathbb R^3)\).
The orbit has a nontrivial radiative critical tail if there exist
\(\varepsilon_0>0\), radii \(R_n\to\infty\), and times
\(\tau_n\to\infty\) such that
\[
\int_{|y|>R_n}|V(y,\tau_n)|^3\,dy\ge \varepsilon_0
\qquad\text{for every }n.
\]
The orbit satisfies the no-radiation condition on the renormalized tail if no
such triple exists.

By T9.1 below, this condition is equivalent to
\[
\lim_{R\to\infty}
\limsup_{\tau\to\infty}
\int_{|y|>R}|V(y,\tau)|^3\,dy=0.
\]

## T9.1 (No-radiation is equivalent to tail-local compact-cylinder mass retention)

The no-radiation condition in Definition T9.0 is equivalent to uniform
\(L^3\)-tightness on sufficiently late renormalized tails:
\[
\forall\varepsilon>0\ \exists R_\varepsilon>0\ \exists T_\varepsilon\ge\tau_0:
\sup_{\tau\ge T_\varepsilon}
\int_{|y|>R_\varepsilon}|V(y,\tau)|^3\,dy<\varepsilon.
\]

### Proof

If the displayed uniform tightness condition holds, then no nontrivial
radiative critical tail can exist, because for the corresponding
\(\varepsilon_0\) one may choose \(R_\varepsilon\) and \(T_\varepsilon\) with
tail mass \(<\varepsilon_0\) for every \(\tau\ge T_\varepsilon\). For all
large \(n\), both \(R_n\ge R_\varepsilon\) and
\(\tau_n\ge T_\varepsilon\), so
\[
\int_{|y|>R_n}|V(y,\tau_n)|^3\,dy
\le
\int_{|y|>R_\varepsilon}|V(y,\tau_n)|^3\,dy
<\varepsilon_0,
\]
contradicting the definition of a radiative tail.

Conversely, suppose uniform tightness fails. Then there exists
\(\varepsilon_0>0\) such that for every \(R>0\) and every \(T\ge\tau_0\) there
is a time \(\tau_{R,T}\ge T\) with
\[
\int_{|y|>R}|V(y,\tau_{R,T})|^3\,dy\ge\frac{\varepsilon_0}{2}.
\]
Taking \(R=n\) and \(T=n\), set \(\tau_n:=\tau_{n,n}\). Then \(R_n=n\to\infty\),
\(\tau_n\ge n\to\infty\), and
\[
\int_{|y|>R_n}|V(y,\tau_n)|^3\,dy\ge\frac{\varepsilon_0}{2}.
\]
This is a nontrivial radiative critical tail. Hence absence of such tails is
equivalent to tail-local compact-cylinder mass retention.

\(\square\)

---

## T9.2 (Compact core plus vanishing \(L^3\) remainder gives no radiation)

Assume that for all \(\tau\ge\tau_0\),
\[
V(\tau)=Q(\tau)+W(\tau)
\quad\text{in }L^3(\mathbb R^3),
\]
where:

1. the core family
   \[
   \mathcal Q:=\{Q(\tau):\tau\ge\tau_0\}
   \]
   has compact closure in \(L^3(\mathbb R^3)\);
2. the remainder is uniformly small in the critical norm:
   \[
   \lim_{T\to\infty}\sup_{\tau\ge T}
   \|W(\tau)\|_{L^3(\mathbb R^3)}=0.
   \]

Then the orbit satisfies the tail no-radiation condition. More precisely, for
every \(\varepsilon>0\) there exist \(T_\varepsilon\ge\tau_0\) and
\(R_\varepsilon>0\) such that
\[
\sup_{\tau\ge T_\varepsilon}
\int_{|y|>R_\varepsilon}|V(y,\tau)|^3\,dy<\varepsilon.
\]

### Proof

Fix \(\varepsilon>0\). By T8.1 applied to
\(\overline{\mathcal Q}^{\,L^3}\), choose \(R_\varepsilon>0\) such that
\[
\sup_{\tau\ge\tau_0}
\int_{|y|>R_\varepsilon}|Q(y,\tau)|^3\,dy
<\frac{\varepsilon}{2^{4}}.
\]
By assumption 2, choose \(T_\varepsilon\ge\tau_0\) such that
\[
\sup_{\tau\ge T_\varepsilon}
\|W(\tau)\|_{L^3(\mathbb R^3)}^3
<\frac{\varepsilon}{2^{4}}.
\]
For \(\tau\ge T_\varepsilon\), the inequality
\[
|Q+W|^3\le4(|Q|^3+|W|^3)
\]
gives
\[
\int_{|y|>R_\varepsilon}|V(y,\tau)|^3\,dy
\le
4\int_{|y|>R_\varepsilon}|Q(y,\tau)|^3\,dy
+4\int_{|y|>R_\varepsilon}|W(y,\tau)|^3\,dy
<\frac{\varepsilon}{4}+\frac{\varepsilon}{4}
<\varepsilon.
\]
Taking the supremum over \(\tau\ge T_\varepsilon\) proves the result.

\(\square\)

---

## T9.3 (Compact core plus vanishing \(L^3\) remainder gives asymptotic compactness)

Under the assumptions of T9.2, the orbit is asymptotically sequentially
compact in \(L^3(\mathbb R^3)\): for every sequence \(\tau_n\to\infty\),
there exists a subsequence \(\tau_{n_k}\) and \(V_*\in L^3(\mathbb R^3)\)
such that
\[
V(\tau_{n_k})\to V_*
\quad\text{strongly in }L^3(\mathbb R^3).
\]

### Proof

Let \(\tau_n\to\infty\). Since \(\overline{\mathcal Q}^{\,L^3}\) is compact,
there exists a subsequence, not relabelled, and \(Q_*\in L^3(\mathbb R^3)\)
such that
\[
Q(\tau_n)\to Q_*
\quad\text{strongly in }L^3(\mathbb R^3).
\]
Assumption 2 gives
\[
\|W(\tau_n)\|_{L^3(\mathbb R^3)}\to0.
\]
Therefore
\[
\|V(\tau_n)-Q_*\|_{L^3}
\le
\|Q(\tau_n)-Q_*\|_{L^3}
+\|W(\tau_n)\|_{L^3}
\to0.
\]
Thus \(V_*:=Q_*\) is the desired strong \(L^3\)-limit.

\(\square\)

---

## T9.4 (No-radiation certificate from asymptotic \(L^3\)-decoupling)

Assume that for every sequence \(\tau_n\to\infty\) there is a subsequence, not
relabelled, and a decomposition
\[
V(\tau_n)=Q_n+W_n
\quad\text{in }L^3(\mathbb R^3)
\]
such that:

1. the core sequence \((Q_n)\) is precompact in \(L^3(\mathbb R^3)\);
2. asymptotic \(L^3\)-decoupling holds:
   \[
   \|V(\tau_n)\|_{L^3}^3
   =
   \|Q_n\|_{L^3}^3+\|W_n\|_{L^3}^3+o(1);
   \]
3. the core carries all limiting critical mass:
   \[
   \|V(\tau_n)\|_{L^3}^3-\|Q_n\|_{L^3}^3\to0.
   \]

Then
\[
\|W_n\|_{L^3(\mathbb R^3)}\to0
\]
along the subsequence. Consequently, every sequence \(\tau_n\to\infty\)
admits a strongly \(L^3\)-convergent subsequence of \(V(\tau_n)\).

### Proof

Subtract assumption 3 from assumption 2:
\[
\|W_n\|_{L^3}^3
=
\|V(\tau_n)\|_{L^3}^3-\|Q_n\|_{L^3}^3+o(1)
\to0.
\]
Thus \(\|W_n\|_{L^3}\to0\). Since \((Q_n)\) is precompact in \(L^3\), pass to
a further subsequence with \(Q_n\to Q_*\) strongly in \(L^3\). Then
\[
\|V(\tau_n)-Q_*\|_{L^3}
\le
\|Q_n-Q_*\|_{L^3}+\|W_n\|_{L^3}
\to0.
\]

\(\square\)

---

## T9.5 (No-radiation closure of the compact Type II barrier)

Assume \(V\) is an admissible renormalized orbit to which Theorem A'' applies
and that:

1. \(\|V(\tau)\|_{L^3(\mathbb R^3)}=1\) for all sufficiently large \(\tau\);
2. for every \(R>0\),
   \[
   \sup_{T\ge\tau_0}
   \int_T^{T+1}\|V(\tau)\|_{H^1(B_R)}^2\,d\tau<\infty;
   \]
3. the fixed-tail no-radiation condition holds: there exists
   \(T_*\ge\tau_0\) such that
   \[
   \lim_{R\to\infty}
   \sup_{\tau\ge T_*}
   \int_{|y|>R}|V(y,\tau)|^3\,dy=0
   \]

Then finite total localized renormalization cost is impossible:
\[
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty.
\]

### Proof

After increasing \(\tau_0\) to the tail time \(T_*\), assumptions 1 and 3 give
the normalization and local compact-cylinder mass retention hypotheses of Theorem A''.
Assumption 2 gives the required local windowed \(H^1\)-control. Therefore all
hypotheses of Theorem A'' hold on the tail. Theorem A'' gives
\[
\int_{T_*}^\infty \tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty.
\]
Since the localized cost is nonnegative, this implies
\[
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty.
\]

\(\square\)

---

## T9 status

T9 proves the no-radiation reduction in every setting where either:

- a compact core plus a critical remainder vanishing in \(L^3\) is available,
  which gives tail no-radiation and asymptotic sequential \(L^3\)-compactness;
- asymptotic \(L^3\)-decoupling shows that the core carries all critical mass,
  which gives asymptotic sequential \(L^3\)-compactness;
- fixed-tail local compact-cylinder mass retention is supplied directly as a certificate,
  which gives the tightness input needed by Theorem A''.

What remains structural is to prove one of these no-radiation inputs from the
Navier-Stokes Type II branch data. That task is the radiative exclusion bridge
feeding T10 and T11.

---

# T10. No-splitting and single-profile saturation criteria

T10 isolates the exact functional-analytic content of the no-splitting step.
It does not prove that Navier-Stokes dynamics forbids multiple profiles. It
proves that, once a profile decomposition satisfies \(L^3\)-decoupling and a
single profile carries all limiting critical mass, all secondary profiles and
remainders vanish in \(L^3\). Thus the candidate reduces to the compact
single-core branch handled by T8 and T9.

## Definition T10.0 (Asymptotic \(L^3\)-profile decomposition)

Let \(\tau_n\to\infty\). An asymptotic \(L^3\)-profile decomposition of
\(V(\tau_n)\) with \(J\in\mathbb N\) profiles is a representation
\[
V(\tau_n)=\sum_{j=1}^J Q_n^{(j)}+W_n
\quad\text{in }L^3(\mathbb R^3),
\]
where \(Q_n^{(j)},W_n\in L^3(\mathbb R^3)\), together with the decoupling
identity
\[
\|V(\tau_n)\|_{L^3}^3
=
\sum_{j=1}^J\|Q_n^{(j)}\|_{L^3}^3
+\|W_n\|_{L^3}^3+o(1).
\]
The profile masses are the limits, when they exist,
\[
m_j:=\lim_{n\to\infty}\|Q_n^{(j)}\|_{L^3}^3,
\qquad
m_{\mathrm{rem}}:=\lim_{n\to\infty}\|W_n\|_{L^3}^3.
\]

The decomposition is compact-core admissible if each sequence
\((Q_n^{(j)})_{n\ge1}\) is precompact in \(L^3(\mathbb R^3)\).

---

## T10.1 (Decoupling gives nonnegative mass accounting)

Assume \(V(\tau_n)\) admits an asymptotic \(L^3\)-profile decomposition and
\[
\|V(\tau_n)\|_{L^3}^3\to M\in[0,\infty).
\]
After passing to a subsequence, all profile and remainder masses exist and
satisfy
\[
M=\sum_{j=1}^J m_j+m_{\mathrm{rem}}.
\]
In particular, if \(M=1\), then
\[
0\le m_j\le1,\qquad 0\le m_{\mathrm{rem}}\le1.
\]

### Proof

The decoupling identity gives
\[
\sum_{j=1}^J\|Q_n^{(j)}\|_{L^3}^3+\|W_n\|_{L^3}^3
=
\|V(\tau_n)\|_{L^3}^3+o(1).
\]
The right-hand side is bounded because \(\|V(\tau_n)\|_{L^3}^3\to M\). Since
the left-hand side is a finite sum of nonnegative terms, each sequence
\(\|Q_n^{(j)}\|_{L^3}^3\) and \(\|W_n\|_{L^3}^3\) is bounded. Passing to a
subsequence, all finitely many nonnegative sequences converge. Taking the
limit in the decoupling identity gives
\[
M=\sum_{j=1}^J m_j+m_{\mathrm{rem}}.
\]
The normalized case \(M=1\) follows immediately.

\(\square\)

---

## T10.2 (Single-profile saturation kills all secondary profiles)

Assume \(V(\tau_n)\) admits an asymptotic \(L^3\)-profile decomposition with
\[
\|V(\tau_n)\|_{L^3}^3\to M
\]
and, after relabelling the profiles,
\[
\|V(\tau_n)\|_{L^3}^3-\|Q_n^{(1)}\|_{L^3}^3\to0.
\]
Then
\[
\sum_{j=2}^J\|Q_n^{(j)}\|_{L^3}^3+\|W_n\|_{L^3}^3\to0.
\]
Consequently,
\[
\|Q_n^{(j)}\|_{L^3}\to0
\quad (2\le j\le J),
\qquad
\|W_n\|_{L^3}\to0.
\]

### Proof

Subtract the saturation identity from the decoupling identity:
\[
\sum_{j=2}^J\|Q_n^{(j)}\|_{L^3}^3+\|W_n\|_{L^3}^3
=
\|V(\tau_n)\|_{L^3}^3-\|Q_n^{(1)}\|_{L^3}^3+o(1).
\]
The right-hand side tends to \(0\). The left-hand side is a sum of
nonnegative terms, so every term in the finite sum tends to \(0\).

\(\square\)

---

## T10.3 (Mass-gap no-splitting criterion)

Assume \(V(\tau_n)\) admits an asymptotic \(L^3\)-profile decomposition,
\[
\|V(\tau_n)\|_{L^3}^3\to1,
\]
and all profile and remainder masses exist. Suppose there is a number
\(\eta\in(0,1)\) such that every nonzero secondary component has mass at least
\(\eta\):
\[
m_j>0\Rightarrow m_j\ge\eta
\quad (j\ge2),
\qquad
m_{\mathrm{rem}}>0\Rightarrow m_{\mathrm{rem}}\ge\eta.
\]
If the leading profile satisfies
\[
m_1>1-\eta,
\]
then
\[
m_j=0\quad (j\ge2),
\qquad
m_{\mathrm{rem}}=0.
\]

### Proof

By T10.1,
\[
1=m_1+\sum_{j=2}^Jm_j+m_{\mathrm{rem}}.
\]
Thus
\[
\sum_{j=2}^Jm_j+m_{\mathrm{rem}}=1-m_1<\eta.
\]
If any secondary profile mass or remainder mass were nonzero, the mass-gap
hypothesis would make the left-hand side at least \(\eta\), a contradiction.
Hence all secondary masses and the remainder mass vanish.

\(\square\)

---

## T10.4 (No-splitting plus compact cores gives asymptotic \(L^3\)-compactness)

Assume that for every sequence \(\tau_n\to\infty\) there is a subsequence, not
relabelled, with a compact-core admissible asymptotic \(L^3\)-profile
decomposition
\[
V(\tau_n)=\sum_{j=1}^J Q_n^{(j)}+W_n.
\]
Assume also that the no-splitting condition holds on every such subsequence:
after relabelling,
\[
\|V(\tau_n)\|_{L^3}^3-\|Q_n^{(1)}\|_{L^3}^3\to0.
\]
Then every sequence \(\tau_n\to\infty\) admits a further subsequence and
\(V_*\in L^3(\mathbb R^3)\) such that
\[
V(\tau_n)\to V_*
\quad\text{strongly in }L^3(\mathbb R^3).
\]

### Proof

By T10.2,
\[
\sum_{j=2}^J\|Q_n^{(j)}\|_{L^3}+\|W_n\|_{L^3}\to0.
\]
Here the finite number of secondary profiles is essential: T10.2 gives
\(\|Q_n^{(j)}\|_{L^3}\to0\) for each \(2\le j\le J\), and therefore their
finite sum tends to zero.
Since \((Q_n^{(1)})\) is precompact in \(L^3\), pass to a further subsequence
such that
\[
Q_n^{(1)}\to V_*
\quad\text{strongly in }L^3(\mathbb R^3).
\]
Then
\[
\|V(\tau_n)-V_*\|_{L^3}
\le
\|Q_n^{(1)}-V_*\|_{L^3}
+\sum_{j=2}^J\|Q_n^{(j)}\|_{L^3}
+\|W_n\|_{L^3}
\to0.
\]

\(\square\)

---

## T10.5 (Asymptotic compactness plus finite-window tightness gives fixed-tail tightness)

Assume:

1. asymptotic sequential \(L^3\)-compactness:
   every sequence \(\tau_n\to\infty\) has a subsequence \(\tau_{n_k}\) and
   \(V_*\in L^3(\mathbb R^3)\) such that
   \[
   V(\tau_{n_k})\to V_*
   \quad\text{strongly in }L^3(\mathbb R^3);
   \]
2. finite-window \(L^3\)-tightness:
   for every finite interval \([A,B]\subset[\tau_0,\infty)\),
   \[
   \lim_{R\to\infty}
   \sup_{\tau\in[A,B]}\int_{|y|>R}|V(y,\tau)|^3\,dy=0.
   \]

Then fixed-tail local compact-cylinder mass retention holds on every tail. More precisely,
for every \(T_*\ge\tau_0\),
\[
\lim_{R\to\infty}
\sup_{\tau\ge T_*}\int_{|y|>R}|V(y,\tau)|^3\,dy=0.
\]

### Proof

Fix \(T_*\ge\tau_0\). Suppose the conclusion fails. Then there exists
\(\varepsilon_0>0\) such that for every \(R>0\),
\[
\sup_{\tau\ge T_*}\int_{|y|>R}|V(y,\tau)|^3\,dy\ge\varepsilon_0.
\]
For each integer \(n\ge1\), choose \(\tau_n\ge T_*\) such that
\[
\int_{|y|>n}|V(y,\tau_n)|^3\,dy\ge\frac{\varepsilon_0}{2}.
\]

If \((\tau_n)\) is bounded, then after passing to a subsequence there exists
a finite interval \([A,B]\subset[\tau_0,\infty)\) containing all \(\tau_n\).
Finite-window tightness gives a radius \(R_0\) such that
\[
\sup_{\tau\in[A,B]}\int_{|y|>R_0}|V(y,\tau)|^3\,dy
<\frac{\varepsilon_0}{2}.
\]
For \(n\ge R_0\), monotonicity of tails gives
\[
\int_{|y|>n}|V(y,\tau_n)|^3\,dy
\le
\int_{|y|>R_0}|V(y,\tau_n)|^3\,dy
<\frac{\varepsilon_0}{2},
\]
a contradiction.

Thus \((\tau_n)\) is unbounded. Passing to a subsequence, assume
\(\tau_n\to\infty\). By asymptotic sequential compactness, after passing to a
further subsequence,
\[
V(\tau_n)\to V_*
\quad\text{strongly in }L^3(\mathbb R^3).
\]
The compact set
\[
\{V_*\}\cup\{V(\tau_n):n\ge1\}
\]
is compact in \(L^3(\mathbb R^3)\), hence uniformly \(L^3\)-tight by T8.1.
Choose \(R_1\) such that
\[
\sup_n\int_{|y|>R_1}|V(y,\tau_n)|^3\,dy
<\frac{\varepsilon_0}{2}.
\]
For \(n\ge R_1\), monotonicity of tails again gives
\[
\int_{|y|>n}|V(y,\tau_n)|^3\,dy
\le
\int_{|y|>R_1}|V(y,\tau_n)|^3\,dy
<\frac{\varepsilon_0}{2},
\]
contradicting the choice of \(\tau_n\). Therefore fixed-tail uniform
\(L^3\)-tightness holds.

\(\square\)

---

## T10.6 (No-splitting closure of the compact Type II barrier)

Assume \(V\) is an admissible renormalized orbit to which Theorem A'' applies
and that:

1. \(\|V(\tau)\|_{L^3(\mathbb R^3)}=1\) for all sufficiently large \(\tau\);
2. for every \(R>0\),
   \[
   \sup_{T\ge\tau_0}
   \int_T^{T+1}\|V(\tau)\|_{H^1(B_R)}^2\,d\tau<\infty;
   \]
3. every sequence \(\tau_n\to\infty\) admits a compact-core admissible
   asymptotic \(L^3\)-profile decomposition satisfying the no-splitting
   condition of T10.4;
4. finite-window \(L^3\)-tightness holds:
   for every finite interval \([A,B]\subset[\tau_0,\infty)\),
   \[
   \lim_{R\to\infty}
   \sup_{\tau\in[A,B]}
   \int_{|y|>R}|V(y,\tau)|^3\,dy=0.
   \]

Then
\[
\int_{\tau_0}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty.
\]

### Proof

By T10.4 and assumption 3, every sequence \(\tau_n\to\infty\) has a strongly
\(L^3(\mathbb R^3)\)-convergent subsequence. By T10.5 and assumption 4,
fixed-tail local compact-cylinder mass retention holds on every tail. Choose \(T_*\) so
large that
\[
\|V(\tau)\|_{L^3(\mathbb R^3)}=1
\qquad\text{for every }\tau\ge T_*.
\]
Then tightness holds on \([T_*,\infty)\), and assumption 2 gives local
windowed \(H^1\)-control on the same tail. Hence all hypotheses of Theorem A''
hold after shifting the initial time to \(T_*\). Theorem A'' therefore
gives
\[
\int_{T_*}^{\infty}\tilde{\mathfrak D}_{R_0}(\tau)\,d\tau=\infty.
\]
Since the cost is nonnegative, the integral from \(\tau_0\) is also infinite.

\(\square\)

---

## T10 status

T10 proves the conditional no-splitting reduction:

- asymptotic \(L^3\)-decoupling gives exact critical mass accounting;
- single-profile saturation eliminates all secondary profiles and remainders;
- a mass gap plus dominance of one profile is a sufficient no-splitting
  criterion;
- no-splitting plus compact-core admissibility gives asymptotic sequential
  \(L^3\)-compactness;
- asymptotic compactness plus finite-window tightness gives fixed-tail
  \(L^3\)-tightness, which is the tightness input used by Theorem A''.

What remains structural is to derive the required asymptotic profile
decomposition, mass decoupling, and single-profile saturation from the
Navier-Stokes Type II branch data. That irreducibility problem is the content
feeding T11.
