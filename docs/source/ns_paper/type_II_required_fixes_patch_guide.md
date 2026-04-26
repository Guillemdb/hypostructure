# Required fixes for `type_II_regularity.tex`

This document gives a concrete patch guide for the revised manuscript.  It focuses only on the remaining required fixes from the mathematical review.  The goal is to preserve the existing local-stratification strategy and the intended unconditional proof status.  None of the edits below introduces a global profile decomposition, a global `L^3(\mathbb R^3)` hypothesis, or a new external assumption.  Every new condition below is either a local test inside the already ordered branch-selection mechanism, or a theorem proving that the branch is routed to an already named row if the local test fails.

The guiding rule is:

```tex
Routing may be by definition; closure must be by an estimate, compactness lemma,
regularity criterion, or already-proved branch closure.
```

The fixes below are organized by target label.  For each one, I give:

1. the exact problem;
2. the mathematical reason it matters;
3. the exact LaTeX block to insert or replace;
4. downstream references that must be updated.

---

# Fix 1 — Separate the strict selected-window branch from the physical Type II germ, and close the bounded-window row locally

## Target locations

Replace or edit the following blocks:

```tex
\begin{definition}[Local Type II sequence]\label{paper1:def:typeII}
...
\end{definition}
```

near line 219, and

```tex
\begin{proposition}[Passage to local Type II sequences]
\label{paper1:prop:paper0-typeII-reduction}
...
\end{proposition}
```

near line 313, and

```tex
\begin{lemma}[Bounded selected-window limits exit Type II]
\label{paper3:lem:scale-rigid-bounded-limit-exit}
...
\end{lemma}
```

near line 3352.

## Problem being fixed

The current text uses the phrase “local Type II sequence” for a represented selected-window sequence satisfying:

```tex
positive local CKN concentration + no nonzero bounded selected-window limit.
```

That is a useful strict branch condition, but it is not identical to the physical Type II definition near the end of the paper:

```tex
physical singular point + failure of the local Type I scale-invariant bound.
```

A bounded selected-window limit cannot simply be declared “not Type II” in the physical sense.  It can only be declared not to belong to the strict selected-window subbranch.  To keep the proof unconditional, the bounded-window case must be explicitly closed or routed.

The right local closure is:

* if the selected window has a nonzero bounded strong `L^3` velocity limit and the pressure atlas is valid, then sufficiently small compact subcylinders are CKN-small, so the bounded part is locally regular/removable;
* if pressure compactness/reconstruction fails, the branch routes to the pressure/gauge row;
* if CKN concentration survives outside the bounded compact part, the ordered selection reselects it as an inner active core, exterior core, multibubble, cascade, or another already named row.

This preserves the local strategy.  It does **not** prove or require a global Type I bound.

## 1A. Replace `paper1:def:typeII`

Replace the entire current definition with this block.  Keep the same label so existing references do not break.

```tex
\begin{definition}[Strict selected-window Type II branch]
\label{paper1:def:typeII}
A represented sequence is said to lie in the strict selected-window Type II
branch if it has positive local Caffarelli--Kohn--Nirenberg concentration on
some fixed compact cylinder and no selected-window subsequence has a nonzero
bounded selected-window limit on any compact cylinder in the sense of
\cref{paper1:def:bounded-window-limit}.
\end{definition}

\begin{remark}[Strict selected-window Type II versus physical Type II]
\label{paper1:rem:strict-selected-window-vs-physical-typeII}
\Cref{paper1:def:typeII} is a local branch condition inside the selected-window
state-space decomposition.  It is not the definition of a physical Type II
singular germ.  A physical Type II germ is defined later by failure of the
local Type I scale-invariant bound at a singular point.  Therefore a bounded
selected-window limit is not discarded as ``not physically Type II''; it is
routed through the bounded-window local closure row.  Only after that row is
closed, or reselected into one of the named local alternatives, may the proof
continue in the strict selected-window branch.
\end{remark}
```

## 1B. Replace `paper1:prop:paper0-typeII-reduction`

Replace the whole proposition and proof with this version.

```tex
\begin{proposition}[Passage to selected-window branches]
\label{paper1:prop:paper0-typeII-reduction}
Let \((x_0,T)\) be a concentration point in the local Type II alternative of
\cref{paper1:thm:paper0-dichotomy}.  After choosing the positive concentration
scales given there and applying the local scale-translation selection used in
the present section to a nondegenerate compact single core, the represented
selected windows form a sequence with positive local Caffarelli--Kohn--Nirenberg
concentration on a fixed compact cylinder.  After passing to a subsequence,
exactly one of the following alternatives occurs:
\begin{enumerate}[label=\textup{(\roman*)}]
\item the selected-window sequence has a nonzero bounded selected-window limit
      on some compact cylinder and is routed to the bounded-window row;
\item no selected-window subsequence has a nonzero bounded selected-window
      limit on any compact cylinder, and the represented selected windows lie
      in the strict selected-window Type II branch of \cref{paper1:def:typeII}.
\end{enumerate}
\end{proposition}

\begin{proof}
The local concentration dichotomy supplies the positive CKN concentration
sequence and the exclusion of the local Type I alternative at the chosen
physical point.  The scale-translation selection rewrites those shrinking
physical cylinders as fixed selected windows.  On the resulting represented
sequence, run the bounded-window test: either some selected-window subsequence
converges strongly in \(L^3\) on a compact cylinder to a nonzero bounded limit,
in which case item \textup{(i)} occurs, or no such subsequence exists.  The
second alternative is precisely the strict selected-window branch condition of
\cref{paper1:def:typeII}.  This is an exhaustive dichotomy on the selected
windows and uses no global compactness or global critical norm.
\end{proof}
```

## 1C. Insert a small-CKN lemma for bounded selected-window limits

Insert this lemma immediately before the old bounded-window exit lemma, or replace that old lemma with the theorem in 1D below and put this lemma immediately before it.

```tex
\begin{lemma}[Bounded selected-window limits are CKN-small away from pressure failure]
\label{paper3:lem:bounded-window-ckn-smallness}
Let \((V_n,P_n)\) be represented suitable windows on a compact cylinder
\(Q_R(z_0)\).  Suppose that, after passing to a subsequence,
\[
   V_n\to V_* \quad\text{strongly in } L^3(Q_R(z_0)),
   \qquad V_*\in L^\infty(Q_R(z_0)).
\]
Assume also that the local pressure atlas/reconstruction row has not failed on
\(Q_R(z_0)\).  Then for every compact subcylinder
\(Q\Subset Q_R(z_0)\) and every \(\varepsilon>0\), there exists
\(0<\rho_Q<R\), depending only on \(Q\), \(\varepsilon\), the local pressure
atlas, and \(\|V_*\|_{L^\infty(Q_R)}\), such that all sufficiently large \(n\)
satisfy
\[
   C_{V_n}(z,\rho_Q)+D_{P_n}(z,\rho_Q)<\varepsilon
\]
for every admissible parabolic cylinder \(Q_{\rho_Q}(z)\Subset Q\).  If this
conclusion fails, then the first failed local test is the pressure atlas,
pressure gauge, or pressure compactness row.
\end{lemma}

\begin{proof}
Fix \(Q\Subset Q_R(z_0)\).  Because \(V_*\in L^\infty(Q_R)\), for every
\(z\in Q\),
\[
   C_{V_*}(z,\rho)
   =\rho^{-2}\int_{Q_\rho(z)}|V_*|^3
   \le C\|V_*\|_{L^\infty(Q_R)}^3\rho^3
   \to0
\]
as \(\rho\downarrow0\), uniformly for \(Q_\rho(z)\Subset Q\).  Hence choose
\(\rho_Q\) so that this velocity contribution is below \(\varepsilon/8\).
Strong convergence in \(L^3(Q_R)\) then gives, after possibly decreasing
\(\rho_Q\) only finitely many times through a finite cover of \(Q\),
\[
   C_{V_n-V_*}(z,\rho_Q)<\varepsilon/8
\]
for all sufficiently large \(n\) and all admissible centers in the cover.
Thus the velocity part is small.

It remains to control the pressure oscillation.  On the retained pressure-atlas
branch, decompose locally on nested cylinders as
\[
   P_n=P_n^{\rm loc}+H_n+c_n(s),
\]
where \(P_n^{\rm loc}\) is the Calder\'on--Zygmund pressure generated by the
localized quadratic stress and \(H_n\) is harmonic in space on the inner
cylinder.  Since \(V_n\to V_*\) strongly in \(L^3\),
\[
   V_n\otimes V_n\to V_*\otimes V_*\quad\text{strongly in }L^{3/2},
\]
and local Calder\'on--Zygmund boundedness gives strong convergence of the local
pressure parts modulo functions of time in \(L^{3/2}\).  The limit generated by
\(V_*\otimes V_*\) is locally bounded in the scaled CKN sense on sufficiently
small cylinders, because \(V_*\) is bounded.

For the harmonic part, either the pressure atlas/gauge compactness row fails,
which is one of the named pressure rows, or the harmonic functions are compact
in smooth norms on smaller cylinders after subtracting spatial means.  In the
compact case their oscillation on \(Q_\rho(z)\) is \(o(\rho^2)\) in the scaled
\(D\)-quantity as \(\rho\downarrow0\), uniformly on \(Q\).  Therefore the
pressure contribution is below \(\varepsilon/2\) for the same sufficiently small
\(\rho_Q\), after increasing \(n\) if necessary.

Combining the velocity and pressure estimates gives the displayed CKN-smallness.
If any pressure step above is unavailable, the first unavailable item is exactly
the pressure atlas, pressure gauge, or pressure compactness row, so the bounded
window is not treated as a retained terminal branch.
\end{proof}
```

## 1D. Replace `paper3:lem:scale-rigid-bounded-limit-exit`

Replace the old lemma with this theorem.  Keep the same label to avoid broken references.  The title changes, but the label remains.

```tex
\begin{theorem}[Bounded selected-window row is locally removable or reselected]
\label{paper3:lem:scale-rigid-bounded-limit-exit}
Let \((V_n,P_n,a_n,b_n)\) be a represented sequence on a compact selected
cylinder \(Q_R\), and assume that, after passing to a selected-window
subsequence, there is \(V_*\in L^\infty(Q_R)\), \(V_*\not\equiv0\), such that
\[
   V_n\to V_*\qquad\text{strongly in }L^3(Q_R).
\]
Then this sequence cannot be a terminal branch of the strict selected-window
Type II row.  More precisely, after running the ordered local tests, exactly
one of the following occurs:
\begin{enumerate}[label=\textup{(\roman*)}]
\item the local pressure atlas, pressure gauge, or pressure compactness row
      fails;
\item the bounded selected window is CKN-small on a smaller compact core and is
      locally regular/removable by the CKN criterion;
\item the retained concentration is not in the bounded compact core and is
      reselected as a smaller active core, exterior core, multibubble/cascade,
      gauge-degenerate, or other named local row in the state-space
      decomposition.
\end{enumerate}
Consequently a bounded selected-window limit is a closed local routing row, not
an independent retained terminal Type II state.
\end{theorem}

\begin{proof}
First, the existence of a nonzero bounded selected-window limit is exactly the
negation of the strict selected-window condition in \cref{paper1:def:typeII}.
Thus such a sequence is not in the strict selected-window Type II branch.
This observation alone is only branch routing, not physical closure.

To close the bounded row locally, apply
\cref{paper3:lem:bounded-window-ckn-smallness}.  If the pressure-atlas
hypotheses needed there fail, item \textup{(i)} occurs.  Otherwise, on each
compact subcylinder of the bounded window, all sufficiently large selected
windows are CKN-small after shrinking once.  By the Caffarelli--Kohn--Nirenberg
regularity criterion, the bounded compact core is regular/removable, giving
item \textup{(ii)}.

If the original positive CKN concentration of the physical Type II germ still
survives after this regular compact core is removed, then it cannot be carried
by the bounded compact core.  The ordered local selection is then rerun on the
remaining concentration.  Since the state-space tests are ordered before
terminal retention, the surviving concentration is recorded as an inner active
core, exterior core, multibubble/cascade, gauge-degenerate, pressure, or other
named local row.  This is item \textup{(iii)}.  No global Type I bound, global
\(L^3\) bound, or global profile decomposition is used.
\end{proof}
```

## 1E. Downstream edits required by Fix 1

Wherever the paper currently says that a bounded selected-window limit “exits Type II,” change it to one of the following phrases:

```tex
exits the strict selected-window Type II branch and is closed by
\cref{paper3:lem:scale-rigid-bounded-limit-exit}
```

or

```tex
is routed to the bounded-window local row, which is regular/removable or
reselected by \cref{paper3:lem:scale-rigid-bounded-limit-exit}
```

Do **not** say:

```tex
bounded selected-window limit implies the physical germ is not Type II
```

unless you have separately proved a full local Type I scale-invariant bound at the physical point.  The patch above deliberately avoids needing that stronger global-scale statement.

---

# Fix 2 — Break the selected-window estimates/core-retention cycle and replace the a.e.-time `L^2` floor by a spacetime retained-mass floor

## Target locations

Replace or edit:

```tex
\begin{proposition}[Selected-window suitable estimates]
\label{paper5:prop:selected-window-suitable-estimates}
...
\end{proposition}
```

near line 5482,

```tex
\begin{proposition}[Derivation of the selected-window estimates from the preceding sections]
\label{paper5:prop:selected-window-derived}
...
\end{proposition}
```

near line 5526,

```tex
\begin{theorem}[Windowwise retained-core alternative]
\label{paper5:thm:core-retention}
...
\end{theorem}
```

near line 6195, and Step 2 of

```tex
\begin{theorem}[Autonomous reduced-limit theorem]
\label{paper5:thm:autonomous-limit}
```

## Problem being fixed

The current proof has a circular dependency:

```tex
selected-window suitable estimates
  -> selected-window derived estimates
  -> retained-core theorem
  -> selected-window suitable estimates
```

Also, the retained-core theorem currently gives a very strong lower bound:

```tex
\int |V_n(y,s)|^2\chi(y)\,dy \ge \eta
\quad\text{for a.e. }s\in(0,T).
```

The selection mechanism naturally gives a spacetime retained active-window mass floor, for example

```tex
\int_J\int_{B_R}|V_n|^3 \ge \eta,
```

not an a.e.-in-time `L^2` floor unless a separate persistence lemma is proved.  The autonomous-limit nontriviality argument only needs the spacetime `L^3` floor, because the local compactness gives strong `L^3` convergence on compact cylinders by interpolation.

## 2A. Replace `paper5:prop:selected-window-suitable-estimates`

Replace the whole proposition and proof with this acyclic upper-estimate proposition.  Keep the label.

```tex
\begin{proposition}[Selected-window compact upper estimates]
\label{paper5:prop:selected-window-suitable-estimates}
Let \((\tau_n)\) be a thick autonomous-window sequence and set
\[
  V_n(y,s):=V(y,\tau_n+s),\qquad
  P_n(y,s):=P(y,\tau_n+s),
\]
\[
  \alpha_n(s):=a(\tau_n+s),\qquad
  \beta_n(s):=b(\tau_n+s),
  \qquad s\in(0,T).
\]
Assume that the represented equation, pressure reconstruction, compact-cylinder
CKN upper control, local Caccioppoli estimate, and bounded modulation package
are available on these selected windows.  Then, after passing to a subsequence,
for every bounded Lipschitz domain \(K\Subset\mathbb R^3\), there is a constant
\(C_K<\infty\), independent of \(n\), such that
\[
  \|V_n\|_{L^\infty(0,T;L^2(K))}
  +\|V_n\|_{L^2(0,T;H^1(K))}
  +\|P_n-(P_n)_K\|_{L^{3/2}((0,T)\times K)}
  \le C_K,
\]
where
\[
  (P_n)_K(s):=\frac1{|K|}\int_KP_n(y,s)\,dy.
\]
Each \(V_n\) is divergence free.
\end{proposition}

\begin{proof}
The renormalized representation theorem supplies the shifted equation and the
divergence-free condition.  The pressure reconstruction gives, after
subtracting spatial means,
\[
  \sup_n\|P_n-(P_n)_K\|_{L^{3/2}((0,T)\times K)}<\infty
\]
on every compact \(K\).  The local Caccioppoli estimate applied on a slightly
larger compact cylinder gives
\[
  \sup_n\|V_n\|_{L^\infty(0,T;L^2(K))}
  +\sup_n\|V_n\|_{L^2(0,T;H^1(K))}<\infty.
\]
These estimates are upper compactness estimates only.  They do not use, and do
not imply, any retained lower mass floor.  Retained nontriviality is supplied
separately by \cref{paper5:lem:retained-selected-window-mass-floor} below.
\end{proof}
```

## 2B. Replace `paper5:prop:selected-window-derived`

The old proposition becomes redundant after 2A.  Replace it with a short compatibility corollary so references to the label remain meaningful.

```tex
\begin{proposition}[Derivation of the selected-window upper estimates from the preceding sections]
\label{paper5:prop:selected-window-derived}
Let a repaired-gauge branch satisfy the renormalized representation and pressure
reconstruction of the compact-window representation theorem on the windows
\([\tau_n,\tau_n+T]\).  Assume that on each compact cylinder \((0,T)\times K\)
the local compact-cylinder Caffarelli--Kohn--Nirenberg upper quantities are
finite uniformly in \(n\), and that the local Caccioppoli estimate applies on a
slightly larger compact cylinder.  Then
\cref{paper5:prop:selected-window-suitable-estimates} holds.
\end{proposition}

\begin{proof}
This is exactly the proof of
\cref{paper5:prop:selected-window-suitable-estimates}: pressure reconstruction
controls \(P_n-(P_n)_K\) in \(L^{3/2}\), the local Caccioppoli estimate controls
\(V_n\) in \(L^\infty_tL^2_x\cap L^2_tH^1_x\) on compact sets, and the
represented equation preserves \(\nabla\cdot V_n=0\).  No retained lower bound
is used in this derivation.
\end{proof}
```

## 2C. Insert the retained spacetime mass-floor lemma

Insert this after `paper5:prop:selected-window-derived` and before the time-regularity/local-compactness lemmas.

```tex
\begin{lemma}[Retained selected-window spacetime mass floor]
\label{paper5:lem:retained-selected-window-mass-floor}
Let \((\tau_n)\) be selected fixed-length terminal windows for a represented
branch produced by the selected-core construction, and write
\[
  V_n(y,s):=V(y,\tau_n+s),\qquad s\in(0,T).
\]
If the loss-of-retained-active-core row does not occur on these windows, then,
after passing to a subsequence, there exist a compact ball \(B_R\), a nonempty
compact time interval \(J\Subset(0,T)\), and \(\eta>0\) such that
\[
   \int_J\int_{B_R}|V_n(y,s)|^3\,dy\,ds\ge \eta
\]
for all sufficiently large \(n\).
\end{lemma}

\begin{proof}
The selected-core construction marks, on each admissible selected window, an
inner compact renormalized core carrying positive localized CKN velocity mass.
If no subsequence admitted one fixed compact ball, one fixed compact time
subinterval, and one positive lower threshold as displayed above, then after
shrinking once inside every candidate core the retained active mass would be
lost along all further selected-window subsequences.  That is precisely the
loss-of-retained-active-core row in the ordered local first-failure ledger.
Therefore, once that row is excluded, a subsequence exists on which a single
compact spacetime window carries the displayed positive \(L^3\) mass.
\end{proof}
```

## 2D. Replace `paper5:thm:core-retention`

Replace the theorem with this version.  It now gives the correct spacetime lower bound, not an a.e.-time `L^2` floor.

```tex
\begin{theorem}[Windowwise retained-core alternative]
\label{paper5:thm:core-retention}
Let \((\tau_n)\) be a selected fixed-length terminal window sequence of length
\(T>0\) for a represented Type II branch produced by the selected-core
construction of the preceding sections, and write
\[
  V_n(y,s):=V(y,\tau_n+s),\qquad s\in(0,T).
\]
Then, after passing to a subsequence, exactly one of the following alternatives
holds:
\begin{enumerate}[label=\textup{(\roman*)}]
\item the loss-of-retained-active-core row occurs on these selected windows;
\item there exist \(R>0\), a compact interval \(J\Subset(0,T)\), and
      \(\eta>0\) such that
      \[
        \int_J\int_{B_R}|V_n(y,s)|^3\,dy\,ds\ge\eta
      \]
      for all sufficiently large \(n\).
\end{enumerate}
In particular, whenever the loss-of-retained-active-core row is excluded on the
selected windows, the translated sequence satisfies the retained spacetime
nontriviality condition of
\cref{paper5:lem:retained-selected-window-mass-floor}.
\end{theorem}

\begin{proof}
This is the dichotomy proved in
\cref{paper5:lem:retained-selected-window-mass-floor}.  Either no fixed compact
spacetime subwindow carries a positive amount of the selected active mass, in
which case the ordered first-failure ledger records loss of the retained active
core, or such a compact spacetime subwindow exists after passing to a
subsequence.  The theorem asserts only this retained spacetime lower bound.  It
does not assert an a.e.-in-time \(L^2\) lower floor.
\end{proof}
```

## 2E. Replace Step 2 in `paper5:thm:autonomous-limit`

In `paper5:thm:autonomous-limit`, replace the current Step 2 beginning with:

```tex
\emph{Step 2: nontriviality of the limit profile.}
Define for \(s\in(0,T)\)
...
```

and ending with:

```tex
Hence \(U\not\equiv0\), so \(U\not\equiv0\).
```

or the corresponding end of the current nontriviality step, with this block:

```tex
\emph{Step 2: nontriviality of the limit profile.}
By \cref{paper5:lem:retained-selected-window-mass-floor}, after passing to the
same subsequence there exist \(R>0\), \(J\Subset(0,T)\), and \(\eta>0\) such
that
\[
   \int_J\int_{B_R}|V_n|^3\,dy\,ds\ge\eta
\]
for all sufficiently large \(n\).

From \cref{paper5:lem:autonomous-window-compactness},
\[
   V_n\to U\quad\text{strongly in }L^2(J;L^2(B_R)),
\]
and the upper estimates give \(V_n\) bounded in
\(L^\infty(J;L^2(B_R))\cap L^2(J;H^1(B_R))\).  By the local Sobolev
interpolation estimate in three dimensions,
\[
   \sup_n\|V_n\|_{L^{10/3}(J\times B_R)}<\infty.
\]
Interpolating the strong \(L^2\) convergence with the uniform
\(L^{10/3}\) bound gives
\[
   V_n\to U\quad\text{strongly in }L^3(J\times B_R).
\]
Therefore
\[
   \int_J\int_{B_R}|U|^3\,dy\,ds
   =\lim_{n\to\infty}\int_J\int_{B_R}|V_n|^3\,dy\,ds
   \ge\eta>0.
\]
Hence \(U\not\equiv0\).
```

## 2F. Downstream reference edits for Fix 2

Search for phrases saying the selected-window estimates contain a lower bound.  Replace them as follows.

Replace:

```tex
localized nontriviality condition in Proposition~\ref{paper5:prop:selected-window-suitable-estimates}
```

with:

```tex
retained spacetime nontriviality condition in
\cref{paper5:lem:retained-selected-window-mass-floor}
```

Replace:

```tex
The retained selected-core lower bound is the last condition in that proposition.
```

with:

```tex
The retained selected-core lower bound is supplied separately by
\cref{paper5:lem:retained-selected-window-mass-floor}.
```

The local compactness and time-regularity lemmas may continue to cite
`paper5:prop:selected-window-suitable-estimates`, because they only need upper estimates.

---

# Fix 3 — Correct the weak proof of the renormalized local energy inequality

## Target location

Replace the weak-solution paragraph in:

```tex
\begin{lemma}[Renormalized local energy inequality]
\label{paper6a:lem:renormalized-lei}
```

near lines 17210–17225.

## Problem being fixed

The displayed inequality has the right scaling structure, but the weak proof currently tests the physical local energy inequality with

```tex
\psi(x,t)=\phi((x-x_c(t))/\lambda(t),\tau(t)).
```

That is missing the critical prefactor `\lambda(t)^{-1}`.  The correct physical test is

```tex
\psi(x,t)=\lambda(t)^{-1}\phi((x-x_c(t))/\lambda(t),\tau(t)).
```

Without the prefactor, the endpoint and dissipation terms do not scale to the displayed renormalized inequality.

The manuscript already proves the correct absolutely continuous pullback lemma earlier, with the correct prefactor.  The cleanest fix is to cite that lemma.

## Replacement block

Inside the proof of `paper6a:lem:renormalized-lei`, replace the paragraph beginning with:

```tex
For suitable weak solutions, apply the physical local energy inequality with
```

through the end of the proof with this:

```tex
For suitable weak solutions, apply
\cref{paper6a:lem:ac-pullback-local-energy} with
\[
   \Lambda(s)=\lambda(t(s)),\qquad X(s)=x_c(t(s)),
   \qquad s=\tau(t),
\]
where \(dt=\Lambda(s)^2\,ds\).  Equivalently, the physical test function is
\[
  \psi(x,t)=\lambda(t)^{-1}
  \phi\left(\frac{x-x_c(t)}{\lambda(t)},\tau(t)\right).
\]
This is exactly the critical test-function scaling needed to preserve the
endpoint and dissipation terms.  Since the present lemma assumes \(C^1\) charts
on compact windows, it is a special case of the absolutely continuous pullback
lemma.  The displayed renormalized local energy inequality follows directly.
\end{proof}
```

If you prefer not to cite the previous lemma, then keep the derivation but replace the test function by the prefactored one and recompute every term.  The citation version is safer because the previous lemma already contains the correct calculation.

---

# Fix 4 — Make the exterior pressure-only routing acyclic

## Target locations

Replace or edit the four lemmas:

```tex
\label{paper8:lem:exterior-ckn-carrier-or-pressure-route}
\label{paper8:lem:fixed-annular-stress-ckn-dichotomy}
\label{paper8:lem:far-field-harmonic-tail-routes-to-dyadic-carrier}
\label{paper8:lem:exterior-pressure-only-not-terminal}
```

near lines 14175–14844.

## Problem being fixed

The current dependency has a cycle:

```tex
exterior CKN carrier or pressure route
  -> pressure-only not terminal
  -> far-field harmonic tail
  -> fixed annular stress dichotomy
  -> exterior CKN carrier or pressure route
```

The fix is to make the first lemma a pure split only.  Then the order is acyclic:

```tex
pure CKN split
  -> fixed-annulus stress test
  -> far-field harmonic tail decomposition
  -> pressure-only not terminal
```

## 4A. Replace `paper8:lem:exterior-ckn-carrier-or-pressure-route`

```tex
\begin{lemma}[Exterior CKN mass has a velocity or pressure carrier]
\label{paper8:lem:exterior-ckn-carrier-or-pressure-route}
Let \(x_n\in A_{r,R}\) and \(\rho_n\downarrow0\) form an exterior
concentration branch on \(A_{r,R}\), meaning that
\[
  \rho_n^{-2}\int_{T^*-\rho_n^2}^{T^*}\int_{B_{\rho_n}(x_n)}
  \left(|u|^3+|p-(p)_{B_{\rho_n}(x_n)}(t)|^{3/2}\right)
  \,dx\,dt
  \ge \varepsilon_{\rm ext}
\]
for all \(n\).  After passing to a subsequence, one of the following
alternatives holds:
\begin{enumerate}[label=\textup{(\roman*)}]
\item \emph{velocity carrier:}
\[
  \rho_n^{-2}
  \int_{T^*-\rho_n^2}^{T^*}\int_{B_{\rho_n}(x_n)}
  |u|^3\,dx\,dt
  \ge \frac{\varepsilon_{\rm ext}}2;
\]
\item \emph{pressure carrier:}
\[
  \rho_n^{-2}
  \int_{T^*-\rho_n^2}^{T^*}\int_{B_{\rho_n}(x_n)}
  |p-(p)_{B_{\rho_n}(x_n)}(t)|^{3/2}\,dx\,dt
  \ge \frac{\varepsilon_{\rm ext}}2.
\]
\end{enumerate}
This lemma is only the velocity-pressure split of exterior CKN mass.  It does
not close the pressure-only case; that closure is supplied later by
\cref{paper8:lem:exterior-pressure-only-not-terminal}.
\end{lemma}

\begin{proof}
For each \(n\), the exterior CKN lower bound is the sum of a velocity part and
a pressure oscillation part.  If both parts were strictly smaller than
\(\varepsilon_{\rm ext}/2\) along a subsequence, their sum would be strictly
smaller than \(\varepsilon_{\rm ext}\), contradicting the definition of the
exterior concentration branch.  Passing to a subsequence gives one of the two
alternatives.  No pressure-only closure is used in this proof.
\end{proof}
```

## 4B. Replace `paper8:lem:fixed-annular-stress-ckn-dichotomy`

This version may cite the pure split above, because that split no longer depends on pressure-only closure.

```tex
\begin{lemma}[Fixed annular stress is regular or has a local carrier]
\label{paper8:lem:fixed-annular-stress-ckn-dichotomy}
Let \(\mathcal A\Subset\mathbb R^3\setminus B_2\) be a fixed annulus in a
local frame and let \(I\Subset(-\infty,0]\) be a compact terminal time
interval.  Suppose a subsequence has a positive annular stress level
\[
  \limsup_{n\to\infty}
  \|U_n\otimes U_n\|_{L^q_\tau(I;L^{3/2}_y(\mathcal A))}>0
\]
for some \(1<q<\infty\).  Then the ordered local CKN test on
\(\mathcal A\times I\) gives one of the following alternatives:
\begin{enumerate}[label=\textup{(\roman*)}]
\item a velocity-pressure CKN carrier occurs on a subcylinder contained in a
      slightly larger annulus.  The pure split
      \cref{paper8:lem:exterior-ckn-carrier-or-pressure-route} then records
      it as a velocity carrier or a pressure carrier;
\item all sufficiently small admissible subcylinders in the annulus are below
      \(\varepsilon_{\rm CKN}\).  Then the annular stress is a regular
      compact-window annular source.  It is absorbed into the retained smooth
      annular part of the local pressure atlas, unless the harmonic-tail test
      sends its pressure effect to the ordered pressure row.
\end{enumerate}
Thus a positive fixed-annulus stress level is not promoted directly to a
singular carrier; it is tested locally first.
\end{lemma}

\begin{proof}
Apply the terminal CKN test to all admissible subcylinders whose doubled
cylinders remain in a fixed slightly larger annulus containing \(\mathcal A\).
If one of these cylinders has CKN quantity at least \(\varepsilon_{\rm ext}\),
then the pure velocity-pressure split
\cref{paper8:lem:exterior-ckn-carrier-or-pressure-route} gives item
\textup{(i)}.

If no such cylinder exists, every admissible subcylinder is below
\(\varepsilon_{\rm ext}<\varepsilon_{\rm CKN}\).  The CKN regularity theorem and
a finite-overlap cover of compact subannuli give local smoothness on
\(\mathcal A\times I\).  The stress lower bound is therefore a regular annular
source rather than a singular exterior carrier.  Since \(\mathcal A\) is at
bounded normalized distance from the core in the same local frame, this regular
source belongs to the compact-window pressure atlas.  If it is not absorbed
there, the first failed atlas condition is the ordered pressure row.  No
pressure-only closure theorem is used in this proof.
\end{proof}
```

## 4C. Replace `paper8:lem:far-field-harmonic-tail-routes-to-dyadic-carrier`

```tex
\begin{lemma}[Far-field harmonic tails route to dyadic carriers]
\label{paper8:lem:far-field-harmonic-tail-routes-to-dyadic-carrier}
Let \(B_1\Subset B_2\) be nested balls in a local frame and let
\[
  H_n(\tau)=H_n^{\rm far}(\tau)
\]
be the mean-zero harmonic pressure on \(B_2\) generated, in the local pressure
atlas, by the quadratic stress \(F_n=U_n\otimes U_n\) outside \(B_4\).  Let
\(\{\theta_k\}_{k\ge2}\) be a smooth dyadic partition of unity on
\(\mathbb R^3\setminus B_4\), with
\[
  \operatorname{supp}\theta_k
  \subset B_{2^{k+1}}\setminus B_{2^{k-1}} .
\]
Assume the pressure reconstruction and gauge rows have not failed.  If, for
some \(1<q<\infty\),
\[
  \limsup_{n\to\infty}
  \|H_n\|_{L^q_\tau L^{3/2}_y(B_1)}>0,
\]
then, after passing to a subsequence, one of the following alternatives occurs:
\begin{enumerate}[label=\textup{(\roman*)}]
\item a dyadic exterior velocity-stress occupation state is present:
\[
  \limsup_{n\to\infty}
  \sum_{k\ge K_n}2^{-2k}
  \|U_n\otimes U_n\|_{L^q_\tau L^{3/2}_y(\mathcal A_k)}
  >0
\]
for some \(K_n\to\infty\);
\item a fixed dyadic annulus carries a positive annular stress lower bound,
      which is then tested by
      \cref{paper8:lem:fixed-annular-stress-ckn-dichotomy}; it is either a
      regular annular source or a local CKN carrier.
\end{enumerate}
In item \textup{(i)}, the dyadic occupation state is tested by
\cref{paper8:lem:dyadic-exterior-occupation-selection}: it either produces a
dyadic CKN carrier or is regular on the far dyadic annuli.  If the dyadic tail
is regular but its harmonic pressure contribution remains nonzero on the
interior cylinder, the branch is assigned to the ordered pressure row.  Thus a
nonzero far-field harmonic pressure tail is a dyadic exterior carrier, a
regular dyadic tail with vanishing harmonic effect, or the ordered pressure row.
\end{lemma}

\begin{proof}
Write
\[
  H_n=\sum_{k\ge2}H_{n,k}+h_n,
  \qquad
  H_{n,k}
  =
  \mathcal R_i\mathcal R_j(\theta_kF_{n,ij})
  -\fint_{B_2}\mathcal R_i\mathcal R_j(\theta_kF_{n,ij})\,dy,
\]
where \(h_n\) is the pressure-gauge/reconstruction remainder.  Since the gauge
and reconstruction rows have not failed, \(h_n\) is absent modulo functions of
time in the retained pressure atlas.

For \(y\in B_2\) and \(z\in\operatorname{supp}\theta_k\), the second derivative
of the Newtonian kernel is \(O(2^{-3k})\).  Therefore
\[
  \|H_{n,k}(\tau)\|_{L^{3/2}(B_1)}
  \le
  C2^{-3k}\|F_n(\tau)\|_{L^1(\mathcal A_k)}
  \le
  C2^{-2k}\|F_n(\tau)\|_{L^{3/2}(\mathcal A_k)},
\]
because \(|\mathcal A_k|^{1/3}\simeq2^k\).  Summing in \(k\) and taking the
\(L^q_\tau\) norm gives
\[
  \|H_n\|_{L^q_\tau L^{3/2}(B_1)}
  \le
  C
  \sum_{k\ge2}2^{-2k}
  \|F_n\|_{L^q_\tau L^{3/2}(\mathcal A_k)} .
\]
If the left-hand side has positive limsup, the weighted dyadic stress sum has
positive limsup.  Either a fixed finite set of dyadic annuli carries a positive
portion of that limsup, or the positive portion escapes to \(k\to\infty\).  The
fixed case gives item \textup{(ii)} and is routed by
\cref{paper8:lem:fixed-annular-stress-ckn-dichotomy}.  The escaping case gives
item \textup{(i)}.  Applying
\cref{paper8:lem:dyadic-exterior-occupation-selection} to the escaping dyadic
occupation state gives a dyadic CKN carrier or a regular dyadic tail.  In the
regular-tail case, either the corresponding harmonic contribution vanishes
modulo functions of time on \(B_1\), or the nonzero harmonic contribution is
the ordered pressure row.  No exterior pressure-only closure theorem is used in
this proof.
\end{proof}
```

## 4D. Replace `paper8:lem:exterior-pressure-only-not-terminal`

```tex
\begin{lemma}[Exterior pressure-only concentration is not terminal]
\label{paper8:lem:exterior-pressure-only-not-terminal}
Let an exterior concentration branch on \(A_{r,R}\) satisfy the pressure-carrier
alternative of the pure split
\cref{paper8:lem:exterior-ckn-carrier-or-pressure-route}, and suppose that no
velocity carrier occurs on any smaller exterior or separated-core cylinder
after passing to subsequences.  Then the pressure carrier is not an independent
terminal exterior row.  More precisely, after local pressure decomposition on
nested cylinders, one of the following occurs:
\begin{enumerate}[label=\textup{(\roman*)}]
\item the local Calder\'on--Zygmund pressure part forces a velocity-stress
      carrier on a slightly larger exterior cylinder;
\item the far-field harmonic pressure tail produces a dyadic exterior
      occupation state, which is regular or gives a dyadic CKN carrier by
      \cref{paper8:lem:dyadic-exterior-occupation-selection};
\item the pressure reconstruction or pressure gauge fails, which is the ordered
      non-exterior pressure row;
\item all pressure contributions vanish modulo functions of time on the smaller
      cylinder, contradicting the pressure-carrier lower bound.
\end{enumerate}
\end{lemma}

\begin{proof}
Work on a nested exterior cylinder and decompose the pressure into the local
Calder\'on--Zygmund part generated by \(\zeta u\otimes u\), the harmonic part,
and a function of time.  If the local Calder\'on--Zygmund part carries the
pressure lower bound, Calder\'on--Zygmund boundedness and the quantitative
converse in \cref{paper8:lem:nested-ball-harmonic-tail} force a nonzero
velocity-stress carrier on a slightly larger exterior cylinder.  This is item
\textup{(i)}.

Thus any persistent pressure lower bound must lie in the harmonic remainder.
Apply \cref{paper8:lem:harmonic-cross-pressure-routing} on nested cylinders.  A
nonvanishing harmonic remainder is either pressure-reconstruction/gauge failure,
which is item \textup{(iii)}, or it is generated by a far-field tail.  For the
far-field tail, use
\cref{paper8:lem:far-field-harmonic-tail-routes-to-dyadic-carrier}.  A fixed
annular stress level is first tested by
\cref{paper8:lem:fixed-annular-stress-ckn-dichotomy}; a selected local CKN
carrier gives item \textup{(i)}, while a regular annular source is removable
and cannot be an independent pressure-only exterior exit.  An escaping dyadic
stress occupation state is tested by
\cref{paper8:lem:dyadic-exterior-occupation-selection}; it is either regular on
far dyadic annuli or gives a dyadic CKN carrier, which is item \textup{(ii)}.
If the dyadic tail is regular and no pressure row occurs, then the harmonic
tail vanishes modulo functions of time on the smaller cylinder.  Together with
vanishing of the local Calder\'on--Zygmund part this contradicts the original
pressure-carrier lower bound, giving item \textup{(iv)}.  Hence pressure-only
exterior concentration has been routed to a velocity/dyadic exterior carrier or
to the ordered non-exterior pressure row; it is not an independent terminal
exterior exit.
\end{proof}
```

## 4E. Dependency check after Fix 4

After the replacement, the dependencies should be:

```tex
paper8:lem:exterior-ckn-carrier-or-pressure-route
  uses only: definition of exterior CKN concentration.

paper8:lem:fixed-annular-stress-ckn-dichotomy
  uses: pure split + CKN regularity + finite cover.

paper8:lem:far-field-harmonic-tail-routes-to-dyadic-carrier
  uses: dyadic kernel estimate + fixed-annular-stress dichotomy.

paper8:lem:exterior-pressure-only-not-terminal
  uses: local pressure decomposition + harmonic routing + far-field theorem.
```

There should be no citation from any of the first three lemmas back to `paper8:lem:exterior-pressure-only-not-terminal`.

---

# Fix 5 — Add an ordered local state-space exhaustion algorithm before the state decomposition theorem

## Target location

Insert a new theorem immediately before:

```tex
\begin{theorem}[Local Type II decomposition into local alternatives]
\label{paper6:thm:state-decomposition}
```

near line 19803, then replace the statement/proof of that theorem.

## Problem being fixed

The current state-decomposition theorem says the cases exhaust the local alternatives, but the proof does not explicitly run the ordered finite test algorithm.  For the final theorem, this is the exact point where an arbitrary physical Type II concentration sequence must be forced into one of the named local rows.

The proof must explicitly say:

* what tests are run;
* in what order;
* what happens at the first failure;
* why the retained case is one of the listed alternatives;
* why the bounded-window row is not silently discarded.

## 5A. Insert this theorem before `paper6:thm:state-decomposition`

```tex
\begin{theorem}[Ordered local state-space exhaustion algorithm]
\label{paper6:thm:ordered-local-state-space-exhaustion}
Starting from any positive physical local Type II concentration sequence, run
the following finite ordered local tests after passing to subsequences whenever
a compactness alternative is selected:
\begin{enumerate}[label=\textup{(T\arabic*)}]
\item C1 analytic admissibility and physical suitable data;
\item positive CKN concentration selection and scale-translation normalization;
\item bounded selected-window test;
\item representation, pressure reconstruction, and repaired-gauge admissibility;
\item single-core versus multibubble/cascade/gauge-degenerate selection;
\item velocity carrier versus pressure-only carrier;
\item compact upper estimates and retained spacetime active-core mass;
\item local windowed \(H^1\) and rough-core test;
\item scale-collapse, scale-rigid, exterior, and separated-core tests;
\item canonical cost-validation and retained terminal-state test.
\end{enumerate}
Then exactly one of the following occurs:
\begin{enumerate}[label=\textup{(\roman*)}]
\item a first failed test occurs, and the branch is assigned to the named local
      row associated to that first failure;
\item no first failure occurs, and the branch reaches one of the retained
      alternatives listed in \cref{paper6:thm:state-decomposition}.
\end{enumerate}
Every first failed row in item \textup{(i)} is one of the named adjacent or
non-retained local rows treated by the closure ledger.  In particular, a
bounded selected-window limit is routed through
\cref{paper3:lem:scale-rigid-bounded-limit-exit} and is not discarded by
redefining physical Type II.
\end{theorem}

\begin{proof}
The procedure is finite and ordered.  At each test, either the local hypotheses
needed for the next test hold after passing to a subsequence, or the first
failed hypothesis is recorded as the corresponding named row.

Test \textup{(T1)} is the C1 admissibility/suitability test.  Its failures are
class-incompatible or named analytic-data exits.  Test \textup{(T2)} selects a
positive CKN concentration sequence and normalizes it to fixed local windows.
Test \textup{(T3)} applies the bounded-window dichotomy of
\cref{paper1:prop:paper0-typeII-reduction}.  If a bounded selected-window limit
appears, it is closed or reselected by
\cref{paper3:lem:scale-rigid-bounded-limit-exit}; otherwise the branch enters
the strict selected-window row.

Test \textup{(T4)} applies the representation, pressure reconstruction, and
repaired-gauge tests.  Failure of any of these is by definition the corresponding
representation, pressure, or gauge-degenerate row, all of which are ordered
before terminal retention.  Test \textup{(T5)} decides whether a single retained
compact core remains or whether mass splits into multibubble, cascade, or
noncompact/gauge-degenerate alternatives.  Test \textup{(T6)} applies the local
velocity-pressure carrier split; the pressure-only case is routed through the
acyclic pressure atlas and exterior-pressure closures.

Test \textup{(T7)} separates upper compactness from retained nontriviality:
upper estimates are supplied by
\cref{paper5:prop:selected-window-suitable-estimates}, while the retained
spacetime mass floor is supplied by
\cref{paper5:lem:retained-selected-window-mass-floor}.  Test \textup{(T8)} is
the rough-core/local windowed \(H^1\) test.  Test \textup{(T9)} is the local
scale and exterior/separated-core stratification.  Test \textup{(T10)} runs the
canonical cost-validation layer; noncanonical diagnostics route to the canonical
channel and are not independent terminal exits.

Because the list of tests is finite, either a first failed test occurs or no
failure occurs.  If a first failure occurs, the branch is assigned to the named
row attached to that test.  If no failure occurs, all retained hypotheses needed
for the local state-space alternatives hold, and the branch reaches one of the
retained alternatives listed in \cref{paper6:thm:state-decomposition}.  The
proof uses only local compactness, local pressure decomposition, finite
subsequence extraction, and ordered first-failure routing.
\end{proof}
```

## 5B. Replace `paper6:thm:state-decomposition`

Replace the theorem and proof with this version.  Keep the label.

```tex
\begin{theorem}[Local Type II decomposition into local alternatives]
\label{paper6:thm:state-decomposition}
Every positive local Type II concentration sequence in
\cref{paper6:def:local-typeII-sequence}, after passing to subsequences and
selected windows, either exits through a named non-retained local row recorded
by \cref{paper6:thm:ordered-local-state-space-exhaustion}, or enters one of the
following retained alternatives:
\begin{enumerate}[label=\textup{(\roman*)}]
\item a represented compact-core branch carrying a single retained active core,
which by \cref{paper6:lem:compact-core-dissipation-dichotomy} either has
vanishing-dissipation selected windows, exits through a named adjacent local row
before retention, or enters the retained local cost-divergence branch;
\item a multibubble, cascade, or gauge-degenerate branch;
\item loss of local windowed \(H^1\) control on a retained compact cylinder;
\item a finite-cost scale-collapse branch with nonvanishing selected core;
\item a retained compact scale-collapse or scale-rigid terminal state.
\end{enumerate}
\end{theorem}

\begin{proof}
Apply \cref{paper6:thm:ordered-local-state-space-exhaustion}.  If a first
failed test occurs, the branch exits through the named local row associated to
that first failure.  This includes the bounded-window row, pressure-only row,
representation/gauge rows, rough-core row, exterior/separated-core rows, and
cost-validation rows.

If no first failure occurs, then the representation theorem gives the
nondegenerate single-core chart whenever a unique retained core is selected.
Failure of that selection, splitting of positive CKN mass, compactness failure
inside the local concentration selection, or scale cascade is the
multibubble/cascade/gauge-degenerate side.  On a represented retained core,
rough-core loss is equivalent by \cref{paper6:thm:rough-core} to failure of
compact CKN control on an enclosing cylinder, hence returns to the named
rough-core or multibubble/cascade row.  Genuine scale collapse is divided by
\cref{paper6:thm:scale-stratification} into finite-cost scale collapse or one
of the retained compact scale-collapse/scale-rigid terminal states.  These are
exactly the alternatives listed above.
\end{proof}
```

---

# Fix 6 — Make noncanonical carrier/cost validation a routing theorem plus canonical-availability lemma

## Target location

Insert a lemma immediately before:

```tex
\begin{theorem}[Noncanonical carrier and cost validation are closed adjacent rows]
\label{paper6:thm:noncanonical-cost-routing-closed}
```

near line 20278, then replace that theorem.

## Problem being fixed

The current theorem says noncanonical carrier/cost validation rows are closed because the final proof uses only the canonical identity cost or comparable costs.  That is correct routing, but it must be backed by a theorem saying the canonical identity-cost channel is always available on retained suitable branches.  Otherwise the proof can sound like:

```tex
This row is closed because we do not use that diagnostic.
```

The fix is to insert a canonical-availability lemma.

## 6A. Insert this lemma before `paper6:thm:noncanonical-cost-routing-closed`

```tex
\begin{lemma}[Canonical identity-cost validation is available on retained suitable branches]
\label{paper6:lem:canonical-identity-cost-available}
Let a represented branch reach the cost-validation layer after the ordered
local state-space tests, and assume it has not exited through a representation,
pressure-reconstruction, repaired-gauge, suitable-energy, modulation, exterior,
rough-core, or active-core-loss row.  Then the canonical identity-cost channel
associated with the suitable local energy inequality is available on every
retained compact terminal window.  If this canonical channel is not available,
then the first failed test is one of the named rows ordered before cost
validation.
\end{lemma}

\begin{proof}
Reaching the retained cost-validation layer means that the branch has already
passed the representation and repaired-gauge tests, has a valid local pressure
atlas, satisfies the suitable local energy inequality in repaired variables, and
has the compact upper estimates and retained active-core mass needed on the
selected window.  The renormalized local energy inequality supplies the
identity-cost energy channel with the standard kinetic energy and dissipation
terms, and the carrier subidentities are supplied by the suitable-energy carrier
package used in the canonical cost construction.

If any ingredient needed to form this canonical identity-cost channel is absent,
then the branch could not have reached the retained cost-validation layer: the
missing ingredient is exactly a first failure of representation, pressure,
repaired gauge, suitable energy, modulation, exterior leakage, rough-core
control, or active-core retention.  These rows are ordered before cost
validation in \cref{paper6:thm:ordered-local-state-space-exhaustion}.  Hence
canonical identity-cost validation is always available on the retained branch;
otherwise an earlier named row occurs.
\end{proof}
```

## 6B. Replace `paper6:thm:noncanonical-cost-routing-closed`

Replace the theorem and proof with this version.  Keep the label.

```tex
\begin{theorem}[Noncanonical carrier and cost validation route to canonical validation]
\label{paper6:thm:noncanonical-cost-routing-closed}
Let \(\omega\) be a renormalized Type II solution with active supercritical
scaling on a terminal tail.  Let a candidate continuation lie in a compact
cost-divergence regime through a noncanonical carrier or cost-validation
channel different from the canonical retained identity-cost channel.  Then the
noncanonical channel is not an independent terminal exit.  More precisely,
after running the ordered local validation tests of
\cref{paper7:lem:no-unrecorded-cost-escape}, exactly one of the following
occurs:
\begin{enumerate}[label=\textup{(\alph*)}]
\item the noncanonical cost is comparable to the canonical identity cost on the
      retained stratum, and the retained cost-divergence exclusion applies;
\item the noncanonical diagnostic is not comparable, in which case it is
      discarded as a diagnostic and the canonical identity-cost channel is run;
\item the canonical channel is unavailable, in which case
      \cref{paper6:lem:canonical-identity-cost-available} routes the branch to
      an earlier named row;
\item a first failed canonical validation test occurs, and that failure is one
      of the named cost-side adjacent rows listed in
      \cref{paper7:lem:no-unrecorded-cost-escape} and discharged by the ambient
      adjacent wrapper.
\end{enumerate}
Consequently the row ``carrier or cost-validation failure outside the canonical
retained identity-cost channel'' is a routing row, not an independent remaining
terminal state.
\end{theorem}

\begin{proof}
Run the ordered validation tests of
\cref{paper7:lem:no-unrecorded-cost-escape}.  If all tests pass and the chosen
noncanonical cost is comparable to the canonical identity cost in the sense of
\cref{def:c4-cost-comparison}, then
\cref{paper7:lem:comparable-cost-compatibility} identifies the retained
noncanonical compact cost with the canonical identity-cost channel up to the
allowed one-sided comparison.  The retained cost-divergence exclusion
\cref{paper7:cor:retained-stratum-cost-exclusion} applies.  This is item
\textup{(a)}.

If the noncanonical cost is not comparable, it is not used to retain a terminal
Type II branch.  Instead the proof reruns validation with the canonical
identity-cost channel.  By
\cref{paper6:lem:canonical-identity-cost-available}, this channel is available
on every retained suitable branch.  If it is not available, the missing
ingredient is an earlier named row ordered before cost validation, giving item
\textup{(c)}.

If the canonical channel is available but fails one of its ordered validation
tests, the first failure is one of the named cost-side adjacent rows listed in
\cref{paper7:lem:no-unrecorded-cost-escape}.  Those rows are discharged by
\cref{paper7:cor:ambient-adjacent-wrapper-discharged} and the local closure
results cited there.  This gives item \textup{(d)}.

Thus every noncanonical cost diagnostic is either comparable and therefore
covered by the retained cost-divergence exclusion, or incomparable and therefore
rerouted to the canonical channel.  The proof uses only the suitable local
energy package, compact-window cost comparison, and finite ordered validation
tests.  It introduces no noncanonical schedule, global \(L^3\) bound, global
profile decomposition, or hidden external hypothesis.
\end{proof}
```

---

# Fix 7 — Repair gauge differentiability with the singular weight

## Target location

The current gauge class and differentiability proposition are near lines 1261–1455:

```tex
\mathcal A_{p,R}:=...
```

and

```tex
\begin{proposition}[Differentiability of \(\mathcal F\)]
\label{paper2:prop:gauge-C1}
...
\end{proposition}
```

## Problem being fixed

The scale component of the gauge map is

```tex
\mathcal F_0(\lambda,c;V)
=\int \chi_R(Y)|Y|^{-p}|V(c+\lambda Y)|^3\,dY-\Theta_0.
```

Differentiating in the translation parameter `c` against the singular weight `|Y|^{-p}` is not justified by ordinary dominated convergence from the currently stated assumptions.  The singularity is fixed at `Y=0`, while the profile is translated.  Therefore a retained repaired-gauge branch must either have the local translated weighted integrability needed to differentiate, or it must be routed to the gauge-degenerate row.

This is not a new external assumption.  It is a local gauge-admissibility test inside the already ordered repaired-gauge branch selection.

## 7A. Insert a translation-admissible gauge class after the existing `\mathcal A_{p,R}` definition

Insert this immediately after the existing definition of `\mathcal A_{p,R}`.

```tex
For the actual repaired-gauge implicit-function step we use the following local
translation-admissible subclass.  A profile failing this subclass is not kept in
the retained repaired-gauge row; it is routed to the gauge-degenerate row.
For some \(\delta_0>0\), define
\[
\mathcal A^{\rm tr}_{p,R}:=
\left\{V\in\mathcal A_{p,R}:
\begin{array}{l}
\displaystyle
\sup_{|c|+|\lambda-1|<\delta_0}
\int_{B_{2R}} |Y|^{-p}|V(c+\lambda Y)|^3\,dY<\infty,\\[2mm]
\displaystyle
\sup_{|c|+|\lambda-1|<\delta_0}
\int_{B_{2R}} |Y|^{-p-1}|V(c+\lambda Y)|^3\,dY<\infty,\\[2mm]
\displaystyle
\sup_{|c|+|\lambda-1|<\delta_0}
\int_{B_{2R}} |Y|^{-p}|V(c+\lambda Y)|\,
        |Y\cdot\nabla V(c+\lambda Y)|\,dY<\infty
\end{array}\right\}.
\]
The number \(\delta_0\) is part of the local retained gauge chart.  The larger
buffer ball in the definition of \(\mathcal A_{p,R}\) is chosen so that
\(c+\lambda B_{2R}\subset B_{4R}\) whenever
\(|c|+|\lambda-1|<\delta_0\).
```

## 7B. Insert a gauge-entry lemma after the new class definition

```tex
\begin{lemma}[Gauge differentiability entry test]
\label{paper2:lem:gauge-differentiability-entry}
On every selected retained repaired-gauge window, either the local profiles
belong to \(\mathcal A^{\rm tr}_{p,R}\) on the relevant compact chart, or the
branch exits through the repaired-gauge degeneracy row.  Thus all uses of the
Jacobian and implicit-function theorem in the retained repaired-gauge branch
may assume \(V\in\mathcal A^{\rm tr}_{p,R}\) without adding an external
hypothesis.
\end{lemma}

\begin{proof}
The repaired-gauge construction is an ordered local test.  To remain in the
retained repaired-gauge row, the gauge map must be differentiable in the local
scale and translation parameters and its localized Jacobian must be defined.
The translated weighted bounds in \(\mathcal A^{\rm tr}_{p,R}\) are exactly the
local integrability conditions needed for those derivatives.  If one of these
bounds fails on the selected chart, then the finite-dimensional gauge map is not
available as a retained \(C^1\) chart.  By the ordered gauge convention this is
recorded before terminal retention as repaired-gauge degeneracy.  Therefore the
retained branch may use \(\mathcal A^{\rm tr}_{p,R}\), while the failure is an
already named local row, not a new assumption.
\end{proof}
```

## 7C. Insert the weighted translation differentiability lemma before `paper2:prop:gauge-C1`

```tex
\begin{lemma}[Weighted translation differentiability for the scale gauge]
\label{paper2:lem:weighted-translation-differentiability}
Let \(0<p<3\), let \(\chi_R\in C_c^\infty(B_{2R})\), and let
\(V\in\mathcal A^{\rm tr}_{p,R}\).  Define
\[
   I(c,\lambda)
   :=
   \int_{\mathbb R^3}\chi_R(Y)|Y|^{-p}|V(c+\lambda Y)|^3\,dY
\]
for \(|c|+|\lambda-1|<\delta_0\).  Then \(I\) is continuously differentiable
in \((c,\lambda)\) on this neighborhood.  In particular, at \((c,\lambda)=(0,1)\),
\[
\partial_{c_k} I(0,1)
=
 p\int \chi_R(Y)Y_k|Y|^{-p-2}|V(Y)|^3\,dY
-
 \int (\partial_k\chi_R)(Y)|Y|^{-p}|V(Y)|^3\,dY,
\]
and
\[
\partial_\lambda I(0,1)
=
3\int \chi_R(Y)|Y|^{-p}|V(Y)|V(Y)\cdot(Y\cdot\nabla V(Y))\,dY.
\]
\end{lemma}

\begin{proof}
The scale derivative follows directly from the definition of
\(\mathcal A^{\rm tr}_{p,R}\): for \(|\lambda-1|+|c|<\delta_0\), the difference
quotients are controlled by the translated weighted bound involving
\(|Y|^{-p}|V(c+\lambda Y)|\,|Y\cdot\nabla V(c+\lambda Y)|\).

For the translation derivative, first assume \(V\) is smooth.  Then
\[
\partial_{c_k} I(0,1)
=
3\int \chi_R|Y|^{-p}|V|V\cdot\partial_kV\,dY.
\]
Since \(\partial_k(|V|^3)=3|V|V\cdot\partial_kV\), integration by parts against
the singular weight gives
\[
3\int \chi_R|Y|^{-p}|V|V\cdot\partial_kV
=
-\int \partial_k(\chi_R|Y|^{-p})|V|^3.
\]
Because
\[
\partial_k(\chi_R|Y|^{-p})
=(\partial_k\chi_R)|Y|^{-p}-p\chi_RY_k|Y|^{-p-2},
\]
this is exactly the displayed formula.

For nonsmooth \(V\in\mathcal A^{\rm tr}_{p,R}\), regularize \(V\) inside the
buffer ball \(B_{4R}\).  The smooth identity holds for the regularizations.
The terms involving \((\partial_k\chi_R)|Y|^{-p}|V|^3\) are controlled by local
\(L^3\) away from the singularity and by the weighted \(|Y|^{-p}\) bound near
the singularity.  The terms involving \(Y_k|Y|^{-p-2}|V|^3\) are controlled by
\(|Y|^{-p-1}|V|^3\), which is part of the translated gauge-admissible bound.
Therefore the regularized identities pass to the limit.

Continuity of the derivatives in \((c,\lambda)\) follows from the same argument
with \(V(c+\lambda Y)\), using the uniform translated bounds in
\(\mathcal A^{\rm tr}_{p,R}\) and the dominated convergence theorem after
regularization.  Hence \(I\in C^1\) on the retained gauge chart.
\end{proof}
```

## 7D. Replace `paper2:prop:gauge-C1`

Replace the current proposition and proof with this version.  The key changes are that it uses `\mathcal A^{\rm tr}_{p,R}` and cites the new lemma.

```tex
\begin{proposition}[Differentiability of \(\mathcal F\)]
\label{paper2:prop:gauge-C1}
For each fixed \(V\in\mathcal A^{\rm tr}_{p,R}\), the map
\((\lambda,c)\mapsto \mathcal F(\lambda,c;V)\) is \(C^1\) in a neighborhood of
\((1,0)\).  In each component \(\mathcal F_\alpha\), all derivatives under the
integral sign are legitimate on the retained gauge chart.
\end{proposition}

\begin{proof}
Fix \((\lambda,c)\) near \((1,0)\), and abbreviate
\[
V_{\lambda,c}(Y):=\lambda V(c+\lambda Y).
\]
The neighborhood is chosen so that \(c+\lambda B_{2R}\subset B_{4R}\).

For the scale component,
\[
\mathcal F_0(\lambda,c;V)
=
\lambda^3
\int \chi_R(Y)|Y|^{-p}|V(c+\lambda Y)|^3\,dY-
\Theta_0.
\]
The differentiability of the inner weighted integral in \((c,\lambda)\) is
\cref{paper2:lem:weighted-translation-differentiability}.  Multiplication by
\(\lambda^3\) preserves \(C^1\).  At \((1,0)\), the translation derivative is
therefore
\[
\partial_{c_k}\mathcal F_0(1,0;V)
=
 p\int \chi_RY_k|Y|^{-p-2}|V|^3
-
 \int (\partial_k\chi_R)|Y|^{-p}|V|^3,
\]
and the scale derivative is
\[
\partial_\lambda\mathcal F_0(1,0;V)
=
3\int \chi_R|Y|^{-p}|V|^3
+3\int \chi_R|Y|^{-p}|V|V\cdot(Y\cdot\nabla V).
\]
These quantities are finite by the definition of \(\mathcal A^{\rm tr}_{p,R}\).

For the centering components,
\[
\mathcal F_j(\lambda,c;V)
=
\lambda^2\int Y_j\chi_R(Y)|V(c+\lambda Y)|^2\,dY,
\qquad j=1,2,3.
\]
The weight is smooth and compactly supported away from no singular derivative.
Since \(V\in W^{1,3}_{\rm loc}(B_{4R})\), local Sobolev regularity gives the
required local differentiability of the translation and scale action in the
integrability class used by the compactly supported centering integrals.  The
formal derivatives are
\[
\partial_\lambda V_{\lambda,c}
=\lambda^{-1}V_{\lambda,c}+\lambda^{-1}Y\cdot\nabla V_{\lambda,c},
\qquad
\partial_{c_k}V_{\lambda,c}=\lambda^{-1}\partial_kV_{\lambda,c},
\]
and hence
\[
\partial_\lambda \mathcal F_j
=\int 2Y_j\chi_R\,V_{\lambda,c}\cdot\partial_\lambda V_{\lambda,c}\,dY,
\]
\[
\partial_{c_k}\mathcal F_j
=\int 2Y_j\chi_R\,V_{\lambda,c}\cdot\partial_{c_k}V_{\lambda,c}\,dY.
\]
The local \(W^{1,3}\) and compact support bounds give continuity of these
centering derivatives in \((\lambda,c)\).  Combining the scale and centering
components proves \((\lambda,c)\mapsto\mathcal F(\lambda,c;V)\in C^1\) near
\((1,0)\).
\end{proof}
```

## 7E. Downstream edit for Fix 7

Whenever a theorem invokes the implicit function theorem for the repaired gauge, add the sentence:

```tex
By \cref{paper2:lem:gauge-differentiability-entry}, the retained repaired-gauge
branch either lies in \(\mathcal A^{\rm tr}_{p,R}\), where
\cref{paper2:prop:gauge-C1} applies, or has already exited through the
repaired-gauge degeneracy row.
```

This prevents the strengthened differentiability class from becoming an unstated assumption.

---

# Final consistency checklist after applying all fixes

After implementing the seven fixes, run the following consistency audit.

## A. Search for forbidden wording

Search the `.tex` file for:

```text
bounded selected-window limits exit Type II
not a local Type II sequence
by definition, not Type II
```

Make sure every occurrence has been replaced by wording that says the branch exits the **strict selected-window branch**, or is routed through the bounded-window row.

## B. Search for the old lower-bound form

Search for:

```tex
\int_{\mathbb R^3}|V_n(y,s)|^2\chi(y)\,dy\ge\eta
```

If it still appears as a retained-core consequence, replace it by the spacetime mass floor:

```tex
\int_J\int_{B_R}|V_n|^3\ge\eta.
```

Only keep an a.e.-time `L^2` floor if you add and prove a separate persistence theorem.

## C. Check the dependency order

The following dependencies should be acyclic:

```tex
bounded-window CKN smallness
  -> bounded-window removable/reselectable
  -> state-space exhaustion
  -> state decomposition
  -> state elimination
  -> final physical Type II exclusion
```

```tex
selected-window upper estimates
  -> time regularity
  -> local compactness
  -> autonomous limit

retained selected-window mass floor
  -> nontriviality of autonomous limit
```

```tex
exterior CKN pure split
  -> fixed-annular stress test
  -> far-field harmonic tail routing
  -> exterior pressure-only not terminal
```

```tex
canonical identity-cost availability
  -> noncanonical cost routes to canonical validation
```

```tex
gauge differentiability entry test
  -> weighted translation differentiability
  -> gauge C1 proposition
  -> repaired gauge implicit function theorem
```

## D. Check that no new global assumption has been introduced

The patched text should repeatedly say that failures of local requirements route to named rows.  In particular:

* `\mathcal A^{\rm tr}_{p,R}` is not an external assumption; failure is repaired-gauge degeneracy.
* bounded selected-window convergence is not used to prove global Type I; it is locally regular/removable or reselected.
* retained nontriviality uses a spacetime local `L^3` floor, not global `L^3`.
* noncanonical costs are not excluded by ignoring them; they route to the canonical channel, whose availability is proved locally.

## E. Compile and reference check

After editing, compile at least twice and check for:

```text
undefined references
multiply-defined labels
citation warnings
```

The labels intentionally preserved in this guide are:

```tex
paper1:def:typeII
paper1:prop:paper0-typeII-reduction
paper3:lem:scale-rigid-bounded-limit-exit
paper5:prop:selected-window-suitable-estimates
paper5:prop:selected-window-derived
paper5:thm:core-retention
paper6a:lem:renormalized-lei
paper8:lem:exterior-ckn-carrier-or-pressure-route
paper8:lem:fixed-annular-stress-ckn-dichotomy
paper8:lem:far-field-harmonic-tail-routes-to-dyadic-carrier
paper8:lem:exterior-pressure-only-not-terminal
paper6:thm:state-decomposition
paper6:thm:noncanonical-cost-routing-closed
paper2:prop:gauge-C1
```

The new labels introduced are:

```tex
paper1:rem:strict-selected-window-vs-physical-typeII
paper3:lem:bounded-window-ckn-smallness
paper5:lem:retained-selected-window-mass-floor
paper6:thm:ordered-local-state-space-exhaustion
paper6:lem:canonical-identity-cost-available
paper2:lem:gauge-differentiability-entry
paper2:lem:weighted-translation-differentiability
```

If any of the new labels conflict with labels already present in your edited manuscript, rename the new label by adding `-v2` or another suffix and update the corresponding references.

---

# Summary of what the patch accomplishes

These edits do not change the proof strategy.  They make the local-stratification proof legally tighter at the exact points where the current text was vulnerable:

1. bounded selected-window limits are no longer confused with physical Type II exclusion;
2. selected-window upper estimates no longer depend on retained-core lower bounds;
3. retained nontriviality is proved from a spacetime local `L^3` mass floor;
4. the renormalized local energy inequality uses the correct critical test-function scaling;
5. exterior pressure-only routing is acyclic;
6. the state-space decomposition is backed by a finite ordered exhaustion algorithm;
7. noncanonical cost diagnostics route through a proved canonical channel;
8. singular-weight gauge differentiability is justified on the retained gauge chart, with failure routed to gauge degeneracy.

After these changes, the manuscript still follows the same local branch-routing architecture, but the main closure steps are supported by explicit local estimates and acyclic dependencies rather than definitional exits.
