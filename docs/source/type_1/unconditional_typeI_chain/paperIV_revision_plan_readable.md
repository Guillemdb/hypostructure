
# Paper IV referee-proofing revision plan

**Manuscript reviewed:** `paperIV_residual_branch.tex`  
**Working title:** *Local Exclusion of Residual Type I Navier--Stokes Seregin Limits*  
**Purpose of this memo:** give a detailed revision plan that keeps the existing strategy intact, introduces no new analytic hypotheses, and makes the residual-closure proof easier for a PDE referee to check line by line.

This memo is not a correctness certificate. It identifies the places where the present draft should be made more explicit, more acyclic, and more precise so that the intended proof is auditable. The main recommendation is to keep the two-paper architecture: the setup paper should reduce Type I exclusion to residual closure, and Paper IV should prove the residual closure theorem as a self-contained residual-branch closure theorem.

---

## 1. Executive verdict

The split into a setup/reduction paper and a residual-closure paper is the right architecture. Paper IV is the right place to close the branch

```tex
\mathcal R(\mathcal S;I,J)=\varnothing.
```

The current Paper IV draft is much closer to a complete residual closure than the setup draft, but it still has several referee-risk points. Most are not strategic flaws; they are proof-presentation and quotient/gauge precision issues. The central revisions should do four things:

1. **Make the dependency graph completely acyclic.** A referee must be able to see what is imported, what is proved earlier in Paper IV, and what is only assembled later. Avoid theorem statements whose proof says “this follows from the theorem proved below” unless they are explicitly labeled as forward summaries.

2. **Separate genuine exclusion from quotient removal.** The affine/parasitic class is not the same kind of object as the axisymmetric, rotational, stationary-hull, Young, log-diffuse, or coherent-tail classes. It is partly a normalization/quotient direction. Do not state `\Caff=\varnothing` without qualifying it as the retained normalized affine stratum.

3. **Unify affine-normalized oscillation.** The manuscript currently uses two inequivalent minimizations: one over constant vectors `c\in\mathbb R^3`, another over time-dependent gauges `c(s)`. This must be made consistent because the affine/parasitic stratum explicitly contains time-dependent homogeneous modes.

4. **Strengthen all pressure and compactness passages.** The strategy relies on local pressure-gauge compactness, whole-space pressure representatives, affine pressure quotients, and first-bad pressure compactness. These are the places where PDE referees will look hardest.

After the changes below, the paper can still prove the same theorem by the same route:

```tex
local Type I setup
  -> normalized Seregin residual class
  -> ordered residual decomposition
  -> lower-stratum closures
  -> first-bad hidden-scale routing
  -> terminal local concentration package
  -> sequence-L^3 + Albritton--Barker
  -> residual class empty.
```

No new hypothesis is needed. The revisions are about definitions, lemmas, proof details, theorem ordering, and presentation.

---

## 2. Immediate arXiv/journal gate fixes

These are the edits I would make before any arXiv upload, even if the mathematical proof is otherwise unchanged.

### Must-fix items

**Line 307 — TeX source.** The command `\texttt{proof_setup.tex}` contains an unescaped underscore and can stop strict LaTeX compilation. Replace it with `\texttt{proof\_setup.tex}`.

**Lines 169--181 — duplicated introduction prose.** The introduction contains a complete sentence followed by a dangling duplicate beginning “regularity and Liouville theorems. All rotational-flux estimates...”. Delete the duplicate or rewrite the two sentences as one clean paragraph.

**Lines 334--364 — ordered decomposition.** The prose says that after the first nine alternatives are removed, the remainder is `\Cgen`, but the displayed list places `\Cgen` fifth. Move `\Cgen` to the end of the ordered display.

**Lines 449--461 and final assembly — affine/parasitic branch.** The theorem says `\Caff=\varnothing`, but `\Caff` is also a quotient/normalization stratum for homogeneous modes. Replace literal emptiness with a retained-normalized statement such as `\Caff^{ret}=\varnothing` or “no normalized retained residual representative remains in the affine/parasitic stratum.”

**Lines 3372--3384 vs. 5400--5419 — oscillation convention.** `\operatorname{osc}_3` minimizes over constant vectors, but multiscale affine oscillation minimizes over time-dependent gauges. Introduce two named functionals or redefine one consistently; Section 4 gives a concrete fix.

**Lines 5781--5844 — pressure representation.** The normalized pressure representation is too compressed for a central pressure/gauge claim. Split it into pressure reconstruction, harmonic-removal, and affine-quotient lemmas with explicit kernel estimates.

### Should-fix items

**Theorem labels.** Some theorem environments have labels beginning with `lem:` or `hyp:`. For example, local active-locus extraction is a theorem but labeled `lem:maximal-active-family`; pressure representation is a proposition labeled `lem:normalized-pressure-representation`. Rename labels for readability and future cross-reference sanity.

**Abstract.** The abstract says the residual class is divided into four analytic classes, but the refined decomposition has ten fine strata. Say “four analytic blocks, with the generic block refined into affine/log-diffuse/Young/coherent/generic alternatives,” or list the fine strata.

**Repeated stationary-hull closures.** There are early statements `thm:R3-no-attainable-degenerate-family`, `thm:R3-exclusion` and later `thm:R3S-no-active-stationary-carrier`, `thm:R3S-R3-empty`. Either move the terminal package before the closure or mark early theorems as “forward summary, proved in Sections ...”.

**Bibliography/imports.** Imported Paper 0/I/II/III results are sometimes described in prose instead of cited with exact theorem numbers. Add an import ledger with exact theorem/definition numbers from each companion paper.

I tested a temporary version with the underscore fixed. It compiled after repeated LaTeX runs and produced a 91-page PDF; the final run reported only one underfull vbox. That is a good sign for TeX mechanics, but it does not remove the mathematical clarity issues above.

---

## 3. Recommended global theorem architecture

### 3.1 State Paper IV as a residual theorem, not as a second setup paper

Paper IV should not spend rhetorical energy re-proving the entire local Type I exclusion pipeline. It should start from the normalized residual input produced by Paper I and then close it. The local Type I bridge can remain, but it should be framed as either:

- an appendix-style verification that the setup hypotheses apply to local pointwise Type I, or
- a short “entry theorem” used only to connect the pair of papers.

The main Paper IV theorem should look like this.

```tex
\begin{theorem}[Residual closure theorem]
Let \mathcal S be a normalized Seregin collection produced by the setup theorem,
with local suitability, local pressure-gauge compactness, bounded centered
profiles, and retained compact velocity L^3 mass. Let \mathcal R^\#(\mathcal S;I,J)
denote the retained residual class after the ordered setup removals and after
canonical affine/parasitic normalization. Then
\[
   \mathcal R^\#(\mathcal S;I,J)=\varnothing.
\]
\end{theorem}
```

This keeps the strategy unchanged but avoids a misleading statement that all affine/parasitic mathematical objects are literally nonexistent.

### 3.2 Use two residual classes: raw and normalized

Introduce this distinction early.

```tex
\mathcal R_{raw}(\mathcal S;I,J)
```

is the complement of the lower setup alternatives before affine normalization.

```tex
\mathcal R^\#(\mathcal S;I,J)
```

is the retained normalized residual class after canonical affine/parasitic quotient removal.

Then the decomposition becomes

```tex
\mathcal R^\#(\mathcal S;I,J)
\subset
\Cax\cup\Crot\cup\Cstat
\cup\Clogdiff\cup\Cyoung\cup\Chomcrit\cup\Clogper\cup\Capcrit\cup\Cgen,
```

with a separate sentence:

```tex
Any descendant routed to \Caff is discharged from \mathcal R^\# by the canonical
affine/parasitic normalization; no retained normalized representative remains
there.
```

This preserves every argument currently using `\Caff`, but it stops the theorem from saying something stronger than the quotient construction proves.

### 3.3 Add a one-page dependency ledger

Place this immediately after the main theorem. PDE referees appreciate knowing exactly where a long proof is going. Use a table like this.

**Base Seregin extraction.** Status: imported from Paper I. Depends on local concentration and terminal no-escape. It must not depend on Paper IV residual closure.

**Small, stationary, tight, and closed structured alternatives.** Status: imported from the setup/Papers II--III. They depend on their companion-paper proofs, not on first-bad routing.

**Axisymmetric closure.** Status: proved here. Depends on the centered circulation equation and the axis-absorption Liouville theorem. It should not depend on the terminal concentration theorem.

**Rotational closure.** Status: proved here. Depends on the local Coriolis-flux identity and local concentration routing. It should not invoke a global Coriolis-Leray Liouville theorem.

**Stationary-hull closure.** Status: proved here or assembled here. Depends on scale reselection and the terminal concentration theorem. It should not depend on generic residual closure.

**Affine/parasitic discharge.** Status: normalization/quotient proved here. Depends on affine symmetry and homogeneous gauge compactness. It should not depend on Albritton--Barker.

**Log-diffuse discharge.** Status: proved here. Depends on log-window heredity and first-bad routing. It should not depend on generic residual closure.

**Young discharge.** Status: proved here. Depends on hidden-scale variance and first-bad routing. It should not use a global critical norm.

**Coherent-tail closure.** Status: proved here. Depends on bounded-origin realization. It should not depend on the finite separated-family theorem.

**First-bad routing.** Status: proved here. Depends on pressure compactness and the variance Liouville theorem. It should not depend on final generic closure.

**Terminal concentration theorem.** Status: proved here. Depends on active-locus extraction, diffuse-defect trichotomy, and active recurrence. It should not assume `\Cgen=\varnothing`.

**Generic residual closure.** Status: final assembly. Depends on all previous closures, finite separated-family exclusion, sequence-`L^3`, and Albritton--Barker. Nothing later should be needed.

The important point is to make the graph acyclic: no closure theorem should depend on itself through “terminal concentration,” “critical-tail discharge,” or “generic closure.”

---

## 4. Affine/parasitic quotient: the largest presentation risk

### 4.1 The current issue

The draft defines `\Caff` around lines 3523--3549 as a lower stratum of spatially homogeneous or parasitic modes. That is a quotient/normalization object, not a usual PDE class like `\Cax` or `\Crot`. Later, the main closure theorem and final assembly state `\Caff=\varnothing`.

This creates a logical ambiguity:

- If `\Caff` is the class of raw homogeneous modes, it is not literally empty; homogeneous quotient profiles exist as formal modes.
- If `\Caff` is the class of retained normalized residual representatives after canonical parasitic-free normalization, then it can be empty, but the notation must say so.

### 4.2 Recommended wording

Replace statements of the form

```tex
\Caff=\varnothing.
```

by one of these more precise alternatives:

```tex
\Caff^{ret}=\varnothing.
```

or

```tex
No retained normalized residual representative lies in the affine/parasitic
quotient stratum.
```

or

```tex
Every branch that reaches \Caff is discharged by the canonical affine/parasitic
normalization and therefore does not remain in \mathcal R^\#(\mathcal S;I,J).
```

Then the final theorem can state:

```tex
The genuine retained residual alternatives
\Cax, \Crot, \Cstat, \Clogdiff, \Cyoung, \Chomcrit, \Clogper, \Capcrit, \Cgen
are empty. The affine/parasitic branch is not retained after canonical normalization.
Consequently \mathcal R^\#(\mathcal S;I,J)=\varnothing.
```

This is not a new hypothesis. It is just the normalization already described in `def:affine-constant-lower-stratum`.

### 4.3 Separate two oscillation functionals

Current draft conflict:

- `def:observer-oscillation` uses

```tex
\inf_{c\in\mathbb R^3}\iint_Q |U-c|^3.
```

- `def:multiscale-affine-oscillation` uses

```tex
\inf_{c\in L^\infty((-1,1);\mathbb R^3)}
\int_{-1}^1\int_{B_r}|U-c(s)|^3.
```

Both are useful, but they detect different things. A time-dependent homogeneous mode `b(s)` has zero spatial oscillation after minimization over `c(s)`, but it may have positive oscillation after minimization over a constant vector.

Introduce two names.

```tex
\operatorname{osc}_{3,sp}(U;Q)
:=
\inf_{c\in L^3(I_Q;\mathbb R^3)}\iint_Q |U(y,\tau)-c(\tau)|^3\,dy\,d\tau.
```

Use this for:

- affine/parasitic quotient routing,
- hidden-scale variance,
- first-bad mesoscopic scales,
- spatially homogeneous gauge compactness,
- Young critical-tail detection after homogeneous gauge removal.

Then define

```tex
\operatorname{act}_{3}(U;Q)
:=
\inf_{c\in\mathbb R^3}\iint_Q |U(y,\tau)-c|^3\,dy\,d\tau.
```

Use this for:

- global non-affine activity functional `\Phi`,
- compact activity sets,
- detecting actual constant profiles after mildness/parasitic-free normalization.

This single change removes the main notational ambiguity in the first-bad and affine sections.

### 4.4 Rewrite the affine dichotomy accordingly

The revised theorem should read conceptually as follows.

```tex
\begin{theorem}[Affine normalization dichotomy]
Let (U,\Pi) be a normalized state. Exactly one of the following occurs.
\begin{enumerate}
\item Some constant-in-time activity functional \operatorname{act}_3, or equivalently
      \Phi, is positive on the parasitic-free representative.
\item All spatial oscillations \operatorname{osc}_{3,sp} vanish; the branch is
      a spatially homogeneous quotient mode and is discharged into \Caff.
\end{enumerate}
In the normalized retained residual space only the first alternative remains.
\end{theorem}
```

Do not make the theorem depend on a hidden smoothness of the time gauge. The present draft correctly says that measurable gauges are used only quotient-wise; preserve that, but make the functional notation consistent.

---

## 5. Pressure and gauge compactness: make this referee-proof

The pressure handling is the most likely place for a PDE referee to pause. The current strategy is good: use whole-space Riesz representatives where available, subtract time gauges locally, and treat affine pressure terms only as the pressure paired with spatially homogeneous parasitic velocity modes. But the proof should be split into smaller lemmas.

### 5.1 Split `prop:normalized-pressure-representation`

Current proposition around lines 5781--5844 tries to prove too much at once. Split it into four statements.

#### Lemma A: whole-space representative survives exact operations

State:

```tex
At the finite-energy Leray level, after the standard time gauge,
p_n=\mathcal R_i\mathcal R_j(u_{n,i}u_{n,j}).
This identity is preserved under physical Type I scaling, centered variables,
centered time translations, spatial translations, and covariant translations.
```

Proof details to add:

- Write how the Riesz kernel scales.
- Specify exactly which gauges are time-only and therefore invisible to gradients.
- Clarify that local pressure representatives are obtained by restriction from the whole-space representative plus local time gauges.

#### Lemma B: local harmonic remainders from pressure gauges are affine only

A local pressure limit can differ from the Riesz representative by a harmonic function. You need to explicitly show that no non-affine harmonic part survives the normalized construction.

Suggested statement:

```tex
Let p_n=\mathcal R_i\mathcal R_j(f_{n,ij}) with \|f_n\|_{L^\infty}\le M.
If p_n-a_n(t) converges in L^{3/2}_{loc} after admissible gauges along one of
the normalized state-space operations, then the difference between the pressure
limit and the Riesz representative of the velocity limit is a harmonic polynomial
of degree at most one. The degree-one part is exactly the affine parasitic
pressure; outside \Caff it is zero.
```

What the proof must include:

- Use test functions with zero spatial mean and zero first moments to kill constants and affine functions.
- Use elliptic regularity for harmonic remainders.
- Prove that any nonzero Hessian of a harmonic remainder would contradict convergence of whole-space pressure representatives against second-derivative test functions.
- Make clear that this is a local statement on compact cylinders and does not require a global pressure norm.

#### Lemma C: scaled pressure compactness at first-bad scales

The current `lem:scaled-pressure-compactness-affine` is close. Strengthen the estimate into a standalone kernel calculation.

Suggested statement:

```tex
If \|U_n\|_{L^\infty}\le M, r_n\to\infty, and
Q_n(x,s)=r_n^{-1}\Pi_n(r_nx,s), then on every compact B_R\times I
there exist affine functions L_n(x,s)=b_n(s)\cdot x+a_n(s) such that
\|Q_n-L_n\|_{L^{3/2}(B_R\times I)}\to0.
```

Add the computation:

- The Riesz representative is BMO with norm `O(M^2)`.
- On `B_{Rr_n}`, subtract the spatial average.
- After multiplying by `r_n^{-1}`, the `L^{3/2}(B_R)` norm is `O(r_n^{-1})` after scaling.
- The remaining affine part is exactly the possible parasitic pressure.

This makes the sentence “in fact one may take `Q=0` in the non-affine quotient” safe.

#### Lemma D: product convergence `Q_n W_n -> Q W`

Do not hide this inside pressure compactness. State it explicitly:

```tex
If Q_n\to Q strongly in L^{3/2}_{loc} and W_n\rightharpoonup^*W in L^\infty_{loc},
then Q_n W_n\to QW in distributions.
```

The proof is one paragraph, but it reassures the reader before the variance argument.

### 5.2 Be precise about pressure gauges in first-bad quotient equations

The current draft wisely says no pointwise derivative of `c_n(s)` is required. Make this a visible rule:

> All PDE identities at first-bad scale are written for the ungauged field `V_n`. The gauged field `W_n=V_n-c_n(s)` is used only for oscillation and Young-measure variance. Any occurrence of an affine pressure paired with `c_n` is a quotient notation, not an asserted local `L^{3/2}` pressure unless compactness of `c_n` has already been proved.

Put this rule in a boxed remark before `lem:first-bad-blowdown-equation`.

### 5.3 Local Type I pressure estimate

In `prop:local-typeI-terminal-compactness`, the nonlocal harmonic pressure estimate is important. It should be expanded slightly.

Add:

- the explicit pressure kernel difference estimate used for `h_n(x,t)-h_n(0,t)`;
- why the inner annuli remain in `B_rho(x_*)`;
- why the outer annuli are controlled by global finite energy after scaling;
- the time integration step in `L^{3/2}`;
- the fact that the chosen pressure representative is the whole-space Leray pressure representative.

This avoids the common referee objection: “local Type I control does not control nonlocal pressure.” Your proof has the right answer; make every step visible.

---

## 6. Local Type I bridge: make it optional and exact

The local Type I bridge is useful, but Paper IV should not look like it depends on a circular re-entry into the setup paper. I recommend the following presentation.

### 6.1 Move the bridge after the residual theorem or into an appendix

The first main theorem of Paper IV should be residual closure. Then a short corollary can say:

```tex
Combining this residual closure theorem with the setup paper gives local
pointwise Type I exclusion.
```

The detailed bridge can be an appendix titled:

> Appendix A. Local pointwise Type I implies the admissible terminal no-escape package.

This keeps the main residual proof focused.

### 6.2 AB theorem statement

The Albritton--Barker weak Serrin package around lines 995--1042 should be quoted with exact theorem/lemma number and exact endpoint statement. If the theorem in the cited paper is not literally stated in your notation, say:

```tex
The following is the precise form of the Albritton--Barker estimate used here.
It follows from [AB, Lemma 2.5] after scaling and replacing their notation by
A,C,D,E.
```

Then include a one-paragraph derivation of the endpoint `p=∞,q=2` from the local pointwise Type I envelope:

```tex
\|u(t)\|_{L^\infty(B_\rho)}\le M(T-t)^{-1/2}
\Rightarrow
\|u\|_{L^{2,\infty}_tL^\infty_x(Q_\rho)}\le C M.
```

The current proof has this, but it should be tied more tightly to the exact AB statement.

### 6.3 Terminal no-escape inclusion geometry

In `thm:local-typeI-terminal-energy-package`, explicitly verify the inclusion

```tex
B_{L_R r}(x_*)\times(T-L_R^2r^2,T)\subset Q_{\theta\rho}(z_*).
```

State the chosen condition on `r_R`, e.g.

```tex
L_R r_R < \theta \rho,
\qquad
L_R^2 r_R^2 < \theta^2 \rho^2.
```

This is minor, but it removes a geometric ambiguity near the terminal slice.

### 6.4 Avoid overclaiming “unconditional” inside Paper IV

The corollary `cor:local-typeI-unconditional` is rhetorically dangerous in a companion-paper architecture. It is fine mathematically if Paper IV is complete, but for readability I would rename it:

```tex
Corollary [Local Type I exclusion after setup plus residual closure]
```

or move it to the final assembly. This makes clear that the conclusion is the combination of two papers, not a hidden assumption inside Paper IV.

---

## 7. Axisymmetric class: make the absorption Liouville theorem fully standalone

The axisymmetric section is conceptually strong. The centered Ornstein--Uhlenbeck drift gives a scalar Liouville theorem for the angular circulation. The proof should be made more classical and less compressed.

### 7.1 Boundary at the axis

The operator contains the singular term `-rho^{-1} partial_rho`. Comparison on `(0,R)` should be justified by regularization.

Add a short paragraph:

```tex
We first prove the comparison on [\varepsilon,R]\times[-Z_1,Z_1] with smooth
Dirichlet data, then let \varepsilon\downarrow0. The locally uniform axis
condition G(\rho,Z,\tau)\to0 and boundedness of G allow passage to the limit.
```

Also explicitly state why smooth bounded axisymmetric velocity implies

```tex
V_\theta(\rho,Z,\tau)=O(\rho)
```

near `rho=0`, locally uniformly. This can be one lemma or a cited standard fact.

### 7.2 Killed-strip decay

The killed-strip decay estimate is the heart of `lem:axis-absorption-liouville`. A referee may ask why the killed process necessarily hits `rho=0` or `rho=R` from a compact substrip uniformly as `s -> -∞`, especially with unbounded `Z`.

Make the proof modular:

1. Prove decay on a finite rectangle `(0,R)×(-Z_1,Z_1)` with Dirichlet boundary on all sides.
2. Use a uniform positive absorption probability in a fixed time from compact subsets of `[δ,R-δ]×[-Z_K,Z_K]`.
3. Let `Z_1 -> ∞` using monotone convergence of killed semigroups.
4. Then send `s -> -∞`.

Do not rely only on the phrase “boundary Harnack and strong maximum principle.” State the actual contraction:

```tex
\sup_K q(s+T_R) \le 1-\vartheta_R.
```

Then iterate it.

### 7.3 Barrier comparison on unbounded `Z`

The barrier `h_R(rho)` is independent of `Z`, so it does not control the artificial `Z=±Z_1` boundaries in a truncated comparison. Use the killed-kernel formulation or add a temporary barrier in `Z`, then send its coefficient to zero.

Recommended rewrite:

- Define `W_R^±`.
- Represent `(W_R^±)_+` by the killed semigroup in the strip rather than comparing directly on unbounded `Z`.
- Use the killed-strip decay to remove the initial-time contribution.
- Avoid claiming boundary control at `Z=±∞`.

The conclusion remains unchanged.

---

## 8. Rotational class: clarify that it is routed, not globally classified

The rotational section is good if the reader understands that the proof is a local routing argument, not a global Liouville theorem for all bounded rotating Leray profiles.

### 8.1 Change theorem wording

Current theorem title:

```tex
Stratified exclusion of rotational relative equilibria
```

This is acceptable, but the proof should explicitly say:

> We do not prove that every bounded co-rotating stationary profile is zero. We prove that any such profile occurring as a retained residual Seregin state either generates a retained local concentration branch already excluded by the terminal package, or remains in the compact local equation and is discharged by the ordered residual closure.

This prevents a referee from expecting a standalone global Coriolis-Leray Liouville theorem.

### 8.2 Local Coriolis flux identity

For the Bernoulli/flux identity, add:

- a line justifying integrations by parts using compact support of the cutoff;
- a statement that the pressure gauge does not affect the flux because only gradients or cutoff-integrated divergence terms appear;
- a definition of the swept tube and the finite covering number `N_{χ,T}`;
- the exact relationship between superthreshold flux and a CKN-active recentered cylinder.

### 8.3 Fixed rotation parameter

The definition uses a rotation parameter. If the proof only treats a fixed profile with fixed `Ω`, say so. If sequences of rotational profiles with `Ω_n` could occur, add a compactness reduction:

- `Ω_n` bounded: pass to a subsequence;
- `Ω_n` unbounded: show it routes to a lower fast-rotation/oscillation alternative or is impossible by the retained compactness assumptions.

Do not leave this implicit if the class is defined for a family of rotation rates.

---

## 9. Degenerate stationary-hull class: remove forward-reference circularity

The stationary-hull section currently states early closure theorems and then later proves a terminal concentration package that those closures use. This is a presentation problem, not necessarily a strategy problem.

### 9.1 Combine duplicate closures

Current early statements around lines 2722--2758 and later statements around lines 2864--3114 overlap. Consolidate them.

Recommended order:

1. Define stationary phase space and retained concentration.
2. Prove scale reselection for centered hull limits.
3. Prove stationary-family constrained trajectories are stationary.
4. State a proposition:

```tex
Assuming the terminal local concentration theorem, no degenerate stationary-hull
profile with retained local concentration occurs.
```

5. Later, after the terminal theorem is proved, assemble the unconditional stationary-hull closure.

This makes the dependency explicit and acyclic.

### 9.2 Scale reselection details

In `lem:R3-hull-is-seregin`, write the actual scale/time calculation. If a centered time translation by `s` corresponds to physical time scaling by `e^{-s}`, then the physical blow-up scale changes by `e^{-s/2}`. Include:

```tex
\tau=-\log(-t),\quad \tau\mapsto\tau+s
\quad\Longleftrightarrow\quad
-t\mapsto e^{-s}(-t),\quad r\mapsto e^{-s/2}r.
```

Then show:

- pressure gauges transform correctly;
- retained local concentration is preserved by lower semicontinuity;
- local suitability survives the reselection.

### 9.3 Abstract Banach phase space

The stationary family uses an abstract Banach space `\mathfrak B`. Make clear that this is not a new hypothesis on all residual profiles. It is only part of the definition of the stationary-hull stratum.

Suggested sentence:

> The choice of `\mathfrak B` is not an additional assumption in the residual theorem; it is the witness by which a candidate is assigned to the stationary-hull alternative. If no such witness exists, the candidate is not in `\Cstat` and is handled by another branch.

---

## 10. Tail state space and recurrent core rigidity

This is one of the most sophisticated parts of the paper. It needs a very clean functional-analytic presentation.

### 10.1 Compact normalized state space

In `def:compact-normalized-state-space`, clarify how the bound `M` behaves under affine normalization. The current text says affine normalization is included whenever the affine parameter is bounded, and `S_c` maps `X_M` to `X_{M+|c|}`. That is correct, but the compactness statement for `X_M^#` should not accidentally include states whose bounds have grown past `M`.

Suggested wording:

```tex
For fixed M, X_M^# contains only normalized descendants whose canonical
representative has velocity bound at most M. Affine normalizations with bounded
parameter are included after enlarging M once; throughout an argument M is chosen
large enough for the finite list of normalized representatives under consideration.
```

or define `X_{\le M}^#` after choosing `M` large enough at the start.

### 10.2 Existence of invariant measure on a recurrent core

In `thm:recurrent-tail-core-rigidity`, the proof says the group is amenable and chooses an invariant Borel probability measure. Add the standard compact-flow argument:

- the group generated by covariant translations and time translations is amenable, or at least the relevant action admits Følner averages;
- start with any probability measure on the compact invariant core;
- average along a Følner net;
- take a weak-* limit;
- restrict to a minimal component to get full support.

This is not a new hypothesis; it is a standard compact amenable-action fact, but it should be stated because the proof uses full support at the end.

### 10.3 Observables and infinitesimal generators

The recurrent-core pressure lemma evaluates `W_i(0,0)`, `∂_iP_W(0,0)`, and derivatives via generators `D_i`. Point evaluations of pressure gradients are fine for smooth compact cores, but spectral arguments in `L^2(\mathcal M,\mu)` are safer with mollified observables.

Add a mollified-observable paragraph:

1. Fix a smooth compactly supported kernel `φ_ε`.
2. Define

```tex
u_i^ε(W)=\iint W_i(y,\tau)φ_ε(y,\tau)\,dy\,d\tau.
```

3. Prove the spectral identities for `u_i^ε`, `p_i^ε`, `b_{ij}^ε`.
4. Let `ε -> 0` using local smooth compactness.

This avoids domain objections for the generators `D_i`.

### 10.4 Pressure spectral reconstruction

The proof currently says there is no additional harmonic component after using the curl-free and divergence equations. This is plausible but should be elevated to a named lemma.

Suggested lemma:

```tex
\begin{lemma}[Translation-spectrum pressure reconstruction]
On the nonzero spatial spectral subspace of a compact invariant core, the
pressure gradient is uniquely determined by
D_i p_i=-D_iD_j(u_i u_j),\qquad D_kp_i=D_ip_k.
Consequently the nonzero spectral pressure gradient is the Riesz transform of
u_i u_j in the translation representation, and any remaining pressure gradient
lies in the spatial zero mode.
\end{lemma}
```

Then prove by the joint spectral theorem for the commuting unitary representation.

### 10.5 Energy average rigor

Instead of directly evaluating the equation at `(0,0)` and averaging, add a preliminary mollified identity and then pass to the point evaluation. This makes the integration-by-parts under invariant measure rigorous.

The energy identity should be presented as:

```tex
<|\nabla W|^2> + 1/2<|W-u^0|^2> = 0.
```

or equivalently the current formula. State explicitly that both terms are nonnegative, so each vanishes. Then use full support to conclude `∇W=0` for every profile in the core.

---

## 11. Terminal active-locus extraction and diffuse defects

The terminal concentration machinery is essential for closing the generic branch. It should be made maximally transparent.

### 11.1 Rename and isolate the active-locus theorem

`lem:maximal-active-family` is a theorem environment labeled as a lemma. Rename to something like:

```tex
\label{thm:maximal-active-family}
```

and update references. The active-locus extraction is a structural theorem, not a small lemma.

### 11.2 Closed active set vs. finite profile family

The active locus is a closed set of observer centers and may be infinite. Later finite-separated-family arguments use finite families. Add a short selection lemma:

```tex
If a compact active locus contains more than one asymptotically separated
retained concentration component, then for some finite subcollection one obtains
a finite separated retained concentration profile family.
```

And conversely:

```tex
If no finite separated family exists, then every retained active locus collapses
to a single terminal profile modulo the active successor relation.
```

This prevents a referee from thinking the argument silently replaces an infinite active set by a finite set.

### 11.3 Local compactness failure discharge

`lem:rough-stratum-discharge` should be carefully limited to ancestor-realized retained recenterings. The statement “Paper I compactness excludes retained local compactness failure” is too broad unless the recentering remains inside the admissible local Type I window or has been routed to a terminal exterior/diffuse alternative.

Suggested wording:

> For every ancestor-realized retained recentering that remains in the local terminal window of the setup extraction, the Paper I compactness package excludes loss of local suitability, dissipation compactness, or pressure-gauge compactness. Recenterings leaving that window are, by definition, exterior/diffuse/noncompact terminal alternatives and are treated by the terminal decomposition.

This uses the current strategy, but avoids overclaiming.

### 11.4 Diffuse-defect compactification

The draft refers to an observer compactification. Define enough of it to be checkable:

- what space is compactified;
- what topology is used;
- what functions/observables are continuous on the compactification;
- how covariant maps extend, at least for the fixed maps used;
- why probability measures are tight in the compactification.

If the exact compactification is not important, say:

> Choose a compactification generated by the countable algebra of bounded continuous observables used in the proof. All actions appearing in the proof are required only through their induced continuous maps on this compactification.

This is not a new analytic assumption; it is the standard Gelfand compactification of the observable algebra.

### 11.5 Renormalized defect windows

For normalized defect states obtained by restricting to windows with positive mass and dividing by the mass, explicitly distinguish:

- the original unnormalized defect measure;
- the normalized probability state;
- the support point/window being selected;
- the lower-stratum discharge implication for the ancestor measure.

This makes `thm:ancestor-realization-discharge` more convincing.

---

## 12. Log-annular virial and log-diffuse branch

The log-annular virial section is a good mechanism, but the current presentation sometimes states “assume virials legitimate” implicitly. Convert those assumptions into local compactness lemmas derived from existing bounds.

### 12.1 Add a “realized log-window compactness” lemma

Before `lem:recurrent-virial-equality-routing`, insert:

```tex
\begin{lemma}[Realized log-window virial compactness]
For every ancestor-realized log-window descendant produced by the diffuse-defect
compactification, the annular virial observables, pressure flux terms, and
cutoff error terms appearing in the log-annular identity are compact along the
selected subsequence after the admissible pressure gauges.
\end{lemma}
```

Proof should use only:

- boundedness of velocities;
- local pressure-gauge compactness;
- the annular cutoff support away from the origin;
- first-bad/hidden-scale routing if noncompactness appears.

Then `lem:recurrent-virial-equality-routing` becomes an actual theorem with no hidden assumption.

### 12.2 Check signs in the log-virial identity

The identity around `lem:log-annular-oscillation-virial` has the expected drift contribution, but make it easy to verify by adding one calculation block:

```tex
x=e^\rho\omega,\qquad R^{-1}=e^{-\rho},
\qquad \frac12 x\cdot\nabla(R^{-1}E)
=\frac12\partial_\rho(e^{-\rho}E).
```

Then show exactly how it moves to the left-hand side. This prevents sign-check interruptions.

### 12.3 Log-window descendant selection

When a log-diffuse measure has positive mass in some window, the mass of the chosen window may be small. That is fine because you normalize to a probability state. Say explicitly:

> The normalized window descendant is a support-state of the original defect measure; it is not asserted to preserve the original absolute threshold. The discharge implication is via support/heredity, not by retaining the same mass number.

The current draft says something like this in the ancestor principle. Repeat it in the log-window section.

---

## 13. Hidden-scale Young variance and first-bad routing

This is probably the mathematical heart of Paper IV. It should be written as the cleanest part of the paper.

### 13.1 Keep the quotient rule visible

Before the first-bad definitions, insert a “rule of use” paragraph:

> `V_n` solves the scaled PDE and satisfies the local energy inequality. `W_n=V_n-c_n(s)` measures oscillation after removal of homogeneous gauges. All variance statements for `W_n` are obtained by first proving identities for `V_n` and then passing through deterministic translation of Young measures, unless the gauge sequence itself produces `\Caff`.

This is already in the proof, but it should be visible before the reader enters the technical section.

### 13.2 First-bad scale definition

Make the selected residual exterior window explicit in `def:first-bad-mesoscopic-scale`:

- What is the ambient macroscopic scale `R_n`?
- What subset of observer centers is the supremum over?
- What has been removed before selecting the first bad scale?
- Is the dyadic minimality global in the window or local around the selected center?

A more referee-friendly statement:

```tex
A first bad scale is selected inside a fixed ancestor-realized residual exterior
window E_n after removal of all unit active observers and affine/parasitic
homogeneous gauges. The supremum over z is always over observer centers whose
r-cylinders remain inside E_n with the prescribed finite-overlap enlargement.
```

### 13.3 Translation-defect trichotomy

The lemma “translation defect produces activity, affine collapse, or a bad radius” is crucial. Add a detailed proof of the case `ell_n -> ∞`:

- Cover the macroscopic segment joining `(x,s)` and `(x+h_n,s+θ_n)` by `O(1)` balls at scale `|h_n|+|θ_n|^{1/2}` in macroscopic variables.
- Pull back to centered variables: radius becomes `ell_n`.
- If all these pulled-back cylinders have small affine-normalized spatial oscillation, finite-overlap transfer forces the translated difference to be homogeneous.
- If the translated difference is not homogeneous, one pulled-back cylinder has oscillation at least `ε_0`.

Explicitly define how `ε_0(η_0,K)` is chosen from finite-overlap constants.

### 13.4 Barycenter and variance inequality

`lem:first-bad-young-variance` is strong. Make the derivation completely test-function based.

Recommended structure:

1. **Compactness of gauges.** By homogeneous gauge compactness, either `\Caff` occurs or `c_n -> c` in `L^3_loc`.
2. **Young measures.** Define `ν` for `V_n`, `\tildeν` for `W_n`, and show their variances agree.
3. **Barycenter equation.** Test the ungauged equation against divergence-free vector fields. Viscosity and nonlinearity vanish because of `r_n^{-2}` and `r_n^{-1}`. Pressure converges by the pressure compactness lemma.
4. **Energy inequality.** Use the local energy inequality for `V_n`, not a formal dot product. Test with nonnegative scalar `φ`. Show the diffusion and nonlinear flux vanish, the dissipation is nonnegative and may be dropped, and the pressure flux converges.
5. **Subtract barycenter energy.** Derive

```tex
\partial_s \mathcal V + \frac12 x\cdot\nabla\mathcal V + \mathcal V\le0.
```

This avoids formal differentiability issues and makes clear that no derivative of `c_n` is used.

### 13.5 Bounded variance Liouville

The current characteristic proof is good. Make the distributional-to-characteristic step more explicit:

- mollify in space-time;
- solve the adjoint transport equation with compactly supported terminal test function;
- pass to the limit;
- then use the pointwise characteristic inequality at Lebesgue points.

A one-page proof is enough.

### 13.6 Strong compactness from zero variance

After `\mathcal V=0`, write:

```tex
\tilde\nu_{x,s}=\delta_{\bar W(x,s)}\quad a.e.
```

Then:

- convergence in measure follows from fundamental Young-measure convergence;
- uniform `L^∞` gives uniform integrability of `|W_n-\bar W|^3`;
- Vitali gives strong `L^3_loc` convergence.

This is already present in compressed form; expand it slightly.

### 13.7 Routing theorem conclusion

In `thm:first-bad-mesoscopic-routing`, make explicit why the strong limit is a coherent critical tail rather than just a solution of the linear transport equation. The key is:

- first-bad lower bound survives strong convergence;
- the limit is non-affine in the spatial-gauge quotient;
- it is obtained by a strong annular blow-down;
- hence it belongs to the coherent critical-tail state space by definition.

Add a reference to the exact definition of `\Ccohcrit`.

---

## 14. Coherent critical tails and bounded-origin realization

This is one of the strongest parts of the paper. It should be moved or highlighted as a flagship mechanism because it gives a concrete PDE reason coherent critical tails vanish.

### 14.1 State the scaling contradiction explicitly

For homogeneous `(-1)` tails, write:

```tex
W(r\omega,s)=r^{-1}A(\omega,s).
```

If this tail is realized by bounded profiles on cylinders meeting the macroscopic origin, then for each fixed small `r`, boundedness gives

```tex
|r^{-1}A(\omega,s)|\le M,
```

so

```tex
|A(\omega,s)|\le Mr.
```

Let `r -> 0`, hence `A=0`.

For log-periodic tails, say:

- bounded-origin realization forces the log profile to vanish along `rho -> -∞`;
- log-periodicity transports that zero to all log radii;
- hence the tail is zero.

For aperiodic minimal tails:

- zero lies in the compact hull by bounded-origin vanishing;
- minimality forces the hull to be `{0}`;
- this contradicts nonzero normalized tail mass.

This argument is intuitive and powerful. Put it in the theorem statement or immediately after it.

### 14.2 Define “realized” precisely

The coherent-tail rigidity depends on the tail being realized from the original bounded residual sequence, not merely being an abstract solution of the linear transport equation. Add a definition:

```tex
A coherent critical tail is bounded-origin realized if its realizing annular
cylinders can be chosen so that, after undoing the blow-down, each fixed
macroscopic neighborhood of the tail origin corresponds to physical/centered
points inside uniformly bounded profiles.
```

Then all coherent-tail classes use this definition.

### 14.3 Connect first-bad routing to bounded-origin realization

After first-bad routing produces `\Ccohcrit`, explicitly state why that coherent tail is bounded-origin realized. The proof should trace the ancestor-realized labels:

```tex
first-bad blow-down
  <- hidden-scale defect window
  <- critical-tail compactification
  <- ancestor residual Seregin sequence with uniform L^∞ bound.
```

This turns bounded-origin realization from a plausible statement into a verified inheritance property.

---

## 15. Separated concentration profiles, active recurrence, and sequence-`L^3`

The final generic closure depends on the terminal concentration theorem, finite separated-family exclusion, and indecomposable sequence-`L^3`. This is a long chain; make every arrow explicit.

### 15.1 Time-slice to CKN activity

`lem:timeslice-to-ckn-activity` uses local regularity to thicken a time-slice `L^3` lower bound into parabolic CKN activity. Add details:

- The centered drift coefficient is unbounded globally, but on `B_2` it is bounded.
- Interior estimates depend only on `M` and the local pressure gauge bounds.
- The pressure part of `\mathcal E_1` is nonnegative after gauge, so velocity lower bound suffices.

This lemma is small but important for converting sequence-`L^3`/exterior mass into parabolic concentration.

### 15.2 Terminal indecomposability definition

The definition correctly distinguishes retained tail limit, diffuse exterior concentration, and noncompact recentering. Add a short remark:

> The definition is parabolic. Diffuseness is tested by unit CKN cylinders in spacetime, not by spatial time-slice balls alone.

The current text says this; keep it and maybe bold it in prose because it protects the sequence-`L^3` contrapositive.

### 15.3 Sequence-`L^3` completeness

`lem:atomic-sequence-L3` is central. Make the contrapositive proof more quantitative.

Suggested additions:

- If `A(τ_n)=∞`, explain how to choose `R_n` with exterior mass at least one.
- If `A(τ_n)<∞`, choose `R_n` so slowly that the interior bounded contribution is at most half the total.
- Define `m_n^{par}` and state explicitly that it is the supremum of pressure-normalized CKN activity over exterior parabolic windows.
- If `m_n^{par} >= eps_*`, compactness gives a retained concentrating tail limit unless local compactness fails; both contradict indecomposability.
- If `m_n^{par}<eps_*`, then either a recentering loses compactness or the exterior mass is diffuse by definition; both contradict indecomposability.

The current proof is close; the revised proof should read like a dichotomy lemma.

### 15.4 Active successor relation

For `thm:active-path-space-recurrence`, define the active successor relation as a closed relation on a compact set. Then prove recurrence in a standard topological way:

1. Compact active set `K`.
2. Closed successor relation `R subset K×K`.
3. Every active point has at least one successor unless an already-excluded branch occurs.
4. Infinite path space is compact by Tychonoff/diagonal compactness.
5. A recurrent point or compact recurrent component exists.
6. The recurrent core is excluded by the compact active descendant theorem.

If the proof uses more than compactness, state the extra ingredient as a lemma already proved, not as intuition.

### 15.5 No infinite active descendant chain

`thm:compact-active-descendant` should contain the exact contradiction mechanism. A referee will ask: compactness alone allows infinite recurrent chains. What prohibits them?

From the draft, the intended answer seems to be:

- active recurrence gives a compact retained recurrent core;
- recurrent-core rigidity makes it affine/constant;
- affine/constant core is discharged by `\Caff`;
- retained activity then contradicts non-affine activity threshold.

State this in the theorem proof in exactly that order.

### 15.6 Finite separated-family exclusion

The finite graph/cycle argument is plausible, but it needs to avoid assuming a finite mass budget. Emphasize that the contradiction is not “too many bubbles consume too much mass.” It is:

```tex
finite family -> either indecomposable profile or infinite/recurrent active chain
indecomposable profile -> AB contradiction
infinite/recurrent active chain -> compact active descendant contradiction
```

This is already the strategy; make it prominent.

### 15.7 AB endpoint Liouville use

The AB theorem statement `thm:AB-main` should include the exact assumptions: mild ancient, bounded if required, sequence `L^3`, and the domain `R^3×(-∞,0)`. If AB’s theorem requires boundedness or mildness in a particular sense, quote that sense exactly.

Then in `lem:no-atomic-active`, when shifting by `T<0`, spell out:

```tex
s_k=t_k-T\to-\infty,
\|u^T(\cdot,s_k)\|_{L^3}=\|u(\cdot,t_k)\|_{L^3}=\|U(\cdot,\tau_k)\|_{L^3}.
```

The draft has this; keep it and ensure no sign ambiguity in `t_k=-e^{-τ_k}`.

---

## 16. Final assembly: make it a certificate, not a second proof

The final assembly should be a short, formal verification that the dependency graph has been exhausted.

### 16.1 Replace “empty arrows” for `\Caff`

Current final chain includes:

```tex
\Caff \xrightarrow{oscillatory entry} \varnothing.
```

Use:

```tex
\Caff \xrightarrow{canonical affine/parasitic normalization}
\hbox{not retained in }\mathcal R^\#.
```

This matches the quotient nature of the branch.

### 16.2 Use a final theorem with hypotheses and conclusion only

The final proof can be almost mechanical:

1. Let `V in R^#`.
2. By ordered decomposition, `V` is in one of the fine alternatives.
3. Each fine alternative is discharged by a cited theorem.
4. Therefore no `V` exists.

Avoid re-explaining the local Type I entry in this proof. Put that in a corollary after the residual theorem.

### 16.3 Add a “no hidden global theorem” paragraph

PDE readers will worry that the proof secretly uses a global critical-norm estimate or a global bounded ancient Liouville theorem. Add a final remark:

> The only external Liouville input used after residual closure is the Albritton--Barker endpoint theorem applied to parasitic-free bounded mild physical pullbacks with a backward `L^3` sequence. The proof does not assume a global `L^3` bound, a global CKN budget, or a Liouville theorem for all bounded ancient centered solutions.

You already say this in several places; collect it in one explicit final remark.

---

## 17. Suggested revised main theorem block

Here is a concrete replacement for the current theorem around lines 449--461.

```tex
\begin{theorem}[Refined residual closure for the complete ordered decomposition]
\label{thm:refined-residual-closure-complete-decomposition}
Let \mathcal S be a normalized Seregin collection satisfying the setup inputs
of \Cref{hyp:base-seregin-package,hyp:previous-nonresidual}. Let
\mathcal R^\#(\mathcal S;I,J) be the retained residual class after the ordered
setup removals and after canonical affine/parasitic normalization. Then
\[
   \mathcal R^\#(\mathcal S;I,J)=\varnothing.
\]
More precisely, the genuine retained alternatives
\[
\Cax,\ \Crot,\ \Cstat,\ \Clogdiff,\ \Cyoung,\
\Chomcrit,\ \Clogper,\ \Capcrit,\ \Cgen
\]
are empty as retained normalized residual classes. Every branch routed to
\Caff is discharged by the canonical affine/parasitic quotient and is not a
retained element of \mathcal R^\#.
\end{theorem}
```

Then the proof becomes:

```tex
\begin{proof}
By the ordered decomposition, every element of \mathcal R^\# lies in one of the
listed fine alternatives, unless it is routed to \Caff. The affine/parasitic
case is not retained by the normalization theorem. The axisymmetric, rotational,
stationary-hull, log-diffuse, Young, coherent-tail, and generic cases are
excluded respectively by ... . Hence \mathcal R^\# is empty.
\end{proof}
```

This is clearer and avoids overstating the affine branch.

---

## 18. Suggested revised decomposition definition

Replace the current display in `def:refined-decomposition` by:

```tex
\[
\begin{gathered}
   \Cax,\qquad \Crot,\qquad \Cstat,\qquad \Caff,\\
   \Clogdiff,\qquad \Cyoung,\qquad
   \Chomcrit,\qquad \Clogper,\qquad \Capcrit,\\
   \Cgen .
\end{gathered}
\]
```

Then add:

```tex
The word ``ordered'' means that \Cgen is not tested until all preceding
alternatives have been removed. Thus \Cgen is the complement of the first nine
fine alternatives inside the coarse residual class, after affine/parasitic
normalization.
```

If you keep `\Caff` in the display, immediately say:

```tex
The affine/parasitic entry is a routing/discharge branch rather than a retained
PDE class in the final normalized residual space.
```

---

## 19. Suggested revised abstract paragraph

Current abstract says “four analytic classes” but later uses ten fine strata. A safer replacement:

> After the previously treated small, stationary `L^3`, uniformly `L^3`-tight, and closed structured/decay alternatives have been removed, the remaining residual class is decomposed into four analytic blocks. The first three are the axisymmetric bounded-circulation block, the rotational relative-equilibrium block, and the degenerate stationary-hull block. The fourth, the generic terminal block, is refined into affine/parasitic quotient, log-diffuse, Young, coherent homogeneous, coherent log-periodic, coherent aperiodic, and terminal generic alternatives. The affine branch is discharged by canonical parasitic-free normalization; the log-diffuse and Young branches are reduced to first-bad mesoscopic-scale routing; the coherent critical-tail branches are excluded by bounded-origin realization; and the remaining generic branch is closed by the terminal local concentration theorem, finite separated-family exclusion, sequence-`L^3` completeness, parasitic-free mildness inheritance, and the Albritton--Barker endpoint Liouville theorem.

This accurately matches the paper’s structure.

---

## 20. Wording and presentation improvements for PDE referees

### 20.1 Use theorem status tags

At the start of theorem statements or immediately before them, use tags such as:

- **Imported**
- **Definition/stratification**
- **Local estimate**
- **Compactness**
- **Routing theorem**
- **Closure theorem**
- **Assembly**

Example:

```tex
\begin{theorem}[Imported setup theorem: admissible Type I Seregin extraction]
```

This makes the 90-page manuscript much easier to referee.

### 20.2 Add a notation table

Include a table near the start:

Use short bullet entries instead of a wide table:

- `V,P`: centered Seregin profile; first used in the introduction.
- `U,Q`: physical ancient representative or terminal profile, depending on section; make this section-dependent use explicit.
- `W,Q`: blown-down, co-rotating, or tail profile; define locally in each section.
- `\mathfrak X_M`: terminal state space; first defined at `def:terminal-state-space`.
- `\mathfrak X_M^#`: normalized descendant state space; first defined at `def:compact-normalized-state-space`.
- `\Caff`: affine/parasitic routing stratum; first defined at `def:affine-constant-lower-stratum`.
- `\operatorname{osc}_{3,sp}`: proposed spatial oscillation modulo time-dependent homogeneous gauges.
- `\operatorname{act}_3` and `\Phi`: constant-mode/non-affine retained activity.
- `\calE_1`: pressure-normalized unit CKN activity; define in the terminal section.

This will save referees time.

### 20.3 Add a threshold table

The paper uses several thresholds: `η_0`, `η_v`, `η_1`, `ε_*`, `ε_0`, etc. Add a table:

Use compact bullet entries:

- `η_v`: chosen for the local concentration sequence; depends on setup concentration; used for initial velocity mass.
- `η_1`: chosen after terminal no-escape; depends on `η_v`; used for Seregin extraction away from the terminal layer.
- `ε_*`: retained CKN threshold; chosen in the setup/terminal package; used for active recenterings.
- `ε_0`: first-bad oscillation threshold; depends on the hidden-scale defect lower bound; used for first-bad routing.
- `η_K`: compact activity lower bound; depends on the compact activity set; used for active recurrence.

State the ordering only if needed. Avoid hidden “after lowering thresholds” phrases without saying which threshold is being lowered and whether it affects previous arguments.

### 20.4 Add pressure-gauge table

Use compact bullet entries:

- `a_{n,K}(t)`: local pressure time gauge; measurable in time; used only because pressure is defined modulo time functions.
- `c_n(s)`: homogeneous velocity gauge at first-bad scale; bounded, then either compact or routed to `\Caff`; no derivative is used unless compactness/quotient interpretation has already been established.
- `β(s)·y+α(s)`: affine parasitic pressure; distributional in time; only paired with a homogeneous mode.
- `L_n(x,s)`: scaled affine pressure at first-bad scale; affine in space and measurable in time; discarded in the quotient or routed to `\Caff`.

This table directly addresses likely pressure/gauge referee questions.

### 20.5 Remove rhetorical overclaiming

Avoid phrases like:

- “this is exactly the result proved below” inside a theorem proof;
- “the theorem follows by the residual analysis” without listing dependencies;
- “empty” for quotient branches;
- “formal consequence” where compactness or heredity needs a proof;
- “no global estimate is used” unless you immediately state which local estimates replace it.

Use more precise phrases:

- “This is an ordered routing branch.”
- “This branch is discharged by normalization.”
- “The following proposition is a forward summary; the proof is completed in Sections ... .”
- “The compactness used here is local on fixed terminal cylinders.”

### 20.6 Figure and decision tree

The rendered decision tree is useful. Consider moving it after the theorem map and making the caption more explicit:

- setup/imported nodes in one style;
- Paper IV proved closure nodes in another style;
- quotient/discharge nodes in a third style.

Do not let the figure imply that `\Caff` is a usual class proved empty in the same way as `\Cax`.

---

## 21. Section-by-section revision checklist

### Introduction and main theorem

- Fix duplicate paragraph and underscore.
- Define raw residual vs normalized residual.
- Reorder the fine decomposition.
- Replace literal `\Caff=\varnothing` by retained-normalized discharge.
- Add dependency ledger and theorem map.
- Move long local Type I bridge to appendix or clearly mark it as optional entry verification.

### Local Type I bridge

- Quote AB with exact theorem/lemma reference and exact endpoint assumptions.
- Expand pressure tail estimate.
- Verify inclusion geometry near terminal cylinder.
- Avoid calling the result “unconditional” inside Paper IV unless final residual closure has already been proved in the same theorem chain.

### Axisymmetric class

- Regularize the axis boundary in comparison.
- Make killed-strip decay a standalone lemma.
- Handle unbounded `Z` truncation explicitly.
- State smooth axisymmetric axis behavior.

### Rotational class

- Emphasize local routing, not global Liouville.
- Specify finite tube cover and flux threshold.
- Clarify fixed or varying rotation parameter.
- State pressure gauge invariance of the flux identity.

### Stationary-hull class

- Remove duplicate closure theorems or label them as forward summaries.
- Add scale/time reselection formula.
- Clarify `\mathfrak B` is a witness for the stratum, not a global assumption.
- Ensure terminal theorem dependency is explicit.

### Tail geometry and recurrent core

- Clarify `\mathfrak X_M^#` bound tracking.
- Add invariant measure existence proof.
- Use mollified observables for generator/spectral arguments.
- Add translation-spectrum pressure reconstruction lemma.
- Make energy averaging rigorous before point evaluation.

### Terminal concentration and diffuse defects

- Rename active-locus theorem label.
- Define compactification by observable algebra or specify actual compactification.
- Add finite-selection lemma from closed active loci.
- Restrict local compactness discharge to ancestor-realized retained recenterings.
- Separate normalized defect states from ancestor defect measures.

### Log-annular/log-diffuse branch

- Add realized log-window virial compactness lemma.
- Check and display log-virial signs.
- Explain normalized log-window descendants as support states.
- Make hidden-scale reduction theorem depend explicitly on first-bad routing.

### First-bad/Young branch

- Separate spatial-gauge oscillation from constant activity.
- State the quotient rule before definitions.
- Expand translation-defect trichotomy.
- Split pressure representation into small lemmas.
- Derive variance inequality using test functions and local energy inequality.
- Prove bounded variance Liouville in distributional form.
- Show strong compactness via Dirac Young measure + Vitali.

### Coherent critical tails

- Define bounded-origin realization.
- Highlight homogeneous/log-periodic/aperiodic vanishing mechanisms.
- Show first-bad coherent tails inherit bounded-origin realization.

### Separated profiles and sequence-`L^3`

- Quantify time-slice-to-CKN thickening.
- State active successor relation as closed relation.
- Make recurrence/compact core contradiction explicit.
- Emphasize finite separated-family exclusion is topological/dynamical, not a mass-budget argument.
- Quote AB endpoint theorem exactly.

### Final assembly

- Make final proof mechanical.
- Replace `\Caff -> \varnothing` arrow.
- Add final “no hidden global theorem” remark.
- Ensure all imported results have exact citations.

---

## 22. Suggested revision order

### Pass 1: mechanical and theorem-statement fixes

1. Fix the TeX underscore.
2. Remove duplicated introduction prose.
3. Reorder the decomposition display.
4. Introduce raw vs normalized residual notation.
5. Replace all unqualified `\Caff=\varnothing` statements.
6. Add dependency ledger.
7. Add notation/threshold/gauge tables.

### Pass 2: quotient and pressure coherence

1. Introduce `osc_{3,sp}` and `act_3` or equivalent names.
2. Update `def:observer-oscillation`, `def:multiscale-affine-oscillation`, `thm:closed-nonaffine-activity`, and first-bad lemmas.
3. Split pressure representation proposition.
4. Strengthen first-bad pressure compactness.
5. Add the first-bad quotient rule as a boxed remark.

### Pass 3: compactness and recurrent-core rigor

1. Clarify `X_M^#` compactness with bound tracking.
2. Add invariant measure construction.
3. Replace point-evaluation spectral proof with mollified observables or justify point evaluations by a domain lemma.
4. Add translation-spectrum pressure reconstruction.
5. Make recurrent-core energy identity fully rigorous.

### Pass 4: branch-specific proof expansions

1. Axisymmetric killed-strip/axis regularization.
2. Rotational flux localization details.
3. Stationary-hull dependency cleanup.
4. Log-window virial compactness.
5. First-bad variance derivation expansion.
6. Bounded-origin realization inheritance.
7. Active recurrence and finite separated-family graph proof clarity.

### Pass 5: final readability polish

1. Normalize theorem labels.
2. Move appendix-like entry material.
3. Shorten theorem proofs that are now pure assembly.
4. Run LaTeX until no unresolved references.
5. Check PDF pages containing long tables and TikZ figures.
6. Add complete bibliographic data and exact companion-paper theorem references.

---

## 23. Likely referee questions and the exact insertions that answer them

### Question 1: “Is the residual closure theorem conditional on another unproved closure?”

Answer by adding the dependency ledger and making `thm:first-bad-mesoscopic-routing`, `thm:terminal-local-stratification-package`, and `thm:R4-final-closure` visibly earlier than the final assembly.

### Question 2: “What does it mean that the affine class is empty?”

Answer by distinguishing raw affine quotient modes from retained normalized residual representatives. Replace `\Caff=\varnothing` with “branches reaching `\Caff` are discharged by canonical affine/parasitic normalization.”

### Question 3: “How can pressure limits have only affine harmonic remainders?”

Answer by splitting the pressure proposition and proving the harmonic-removal lemma using whole-space Riesz representatives, test functions killing constants/affine terms, and elliptic regularity.

### Question 4: “Does the first-bad variance equation differentiate the time-dependent gauge?”

Answer by keeping PDE identities for `V_n`, using `W_n` only for oscillation/Young measures, and proving variance invariance under strongly convergent homogeneous shifts.

### Question 5: “Why does no hidden-scale variance imply strong compactness?”

Answer by stating the exact chain: zero variance -> Dirac Young measure -> convergence in measure -> Vitali -> strong `L^3_loc`.

### Question 6: “Why are coherent critical tails realized near the bounded origin?”

Answer by defining bounded-origin realization and tracing it through ancestor-realized first-bad blow-downs.

### Question 7: “Why is the finite separated-family exclusion not using a hidden finite energy/critical mass budget?”

Answer by explicitly presenting the graph/recurrence dichotomy and sequence-`L^3`/AB closure.

### Question 8: “Where exactly is Albritton--Barker used?”

Answer by isolating it in two places only:

1. local weak-Serrin Type I package for terminal no-escape, if you keep the entry bridge in Paper IV;
2. endpoint ancient Liouville theorem for parasitic-free bounded mild physical pullbacks with a backward `L^3` sequence.

No other global Liouville theorem is used.

---

## 24. Final pre-submission QA checklist

Before uploading Paper IV to arXiv or sending it to a journal, verify:

- [ ] The TeX source compiles from a clean directory without manual intervention.
- [ ] All labels resolve after repeated runs.
- [ ] No theorem proof consists only of “this is proved below” unless the theorem is explicitly a forward summary.
- [ ] The final theorem states residual closure for the normalized retained residual class.
- [ ] `\Caff` is consistently treated as quotient/discharge, not as an ordinary nonempty/empty PDE class.
- [ ] All uses of affine-normalized oscillation specify whether the gauge is constant in time or time-dependent.
- [ ] All pressure representations specify the gauge class.
- [ ] First-bad PDE identities are written for ungauged fields.
- [ ] The variance inequality is derived from the local energy inequality in distributional form.
- [ ] The active path-space recurrence proof has a closed relation and compact path-space argument.
- [ ] The AB theorem is quoted exactly with assumptions.
- [ ] Each imported companion-paper result is cited by exact theorem or definition number.
- [ ] The abstract accurately reflects the quotient nature of the affine branch and the fine residual decomposition.
- [ ] Long proof-assembly theorems include dependency lists.
- [ ] Figures and tables do not imply a circular dependency.

---

## 25. Bottom-line recommendation

Do not change the strategy. The strategy is coherent as a two-paper program: Paper I sets up the normalized Seregin residual class and makes Type I exclusion conditional on residual closure; Paper IV closes the residual class by ordered routing, first-bad hidden-scale compactness, terminal local concentration, and endpoint sequence-`L^3` Liouville.

The main thing to change is the **auditability** of the proof. Make the quotient branches precise, make pressure/gauge operations explicit, make the dependency graph acyclic, and turn assembly statements into clean certificates. If those revisions are made, a PDE expert can referee the manuscript by checking a sequence of local estimates and compactness/routing lemmas rather than trying to infer the logical structure from a long narrative.
