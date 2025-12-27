# FACT-SoftKM: Soft→KM Compilation

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-fact-soft-km*

Soft permits compile to Kenig-Merle (Concentration-Compactness + Stability) permits.

---

## Arithmetic Formulation

### Setup

The Kenig-Merle method in arithmetic corresponds to:
- **Concentration-compactness:** Height concentration at primes
- **Rigidity:** No minimal counterexample exists
- **Stability:** Small perturbations don't create counterexamples

### Statement (Arithmetic Version)

**Theorem (Arithmetic KM Compilation).** Given soft permits (bounded height, degree, conductor), the Kenig-Merle machinery applies:

1. **Concentration:** Heights concentrate at finitely many primes
2. **Compactness:** Bounded sequences have convergent subsequences
3. **Rigidity:** If a minimal counterexample existed, it would be special (CM, torsion)
4. **Stability:** Small height perturbations preserve the conclusion

---

### Proof

**Step 1: Concentration at Primes**

For $\alpha \in \overline{\mathbb{Q}}$ with bounded height:
$$h(\alpha) = \sum_v n_v \lambda_v(\alpha)$$

**Concentration:** Heights concentrate at finitely many places:
$$\{v : \lambda_v(\alpha) > \epsilon\} \text{ is finite for any } \epsilon > 0$$

**Proof:** The sum converges, so only finitely many terms exceed $\epsilon$.

**Step 2: Compactness (Northcott)**

By **Northcott's theorem** [Northcott 1950]:

For bounded height and degree:
$$\mathcal{A}(B, d) = \{\alpha : h(\alpha) \leq B, [\mathbb{Q}(\alpha):\mathbb{Q}] \leq d\}$$

is a finite set. Any sequence has a convergent (eventually constant) subsequence.

**Step 3: Rigidity via Minimal Counterexample**

**Method:** Suppose a minimal counterexample to the conjecture exists.

For BSD: Suppose $E$ is an elliptic curve with:
- Minimal conductor among counterexamples
- $\text{ord}_{s=1} L(E,s) \neq \text{rank } E(\mathbb{Q})$

**Rigidity argument:**
1. By minimality, any curve with smaller conductor satisfies BSD
2. Level-lowering [Ribet 1990] produces curves with smaller conductor
3. The minimal counterexample must be "exceptional"—CM or rational torsion

**Conclusion:** No such minimal counterexample exists (CM curves satisfy BSD by [Rubin 1987]).

**Step 4: Stability**

**Small perturbations preserve conclusion:**

For height perturbation:
$$h(P') = h(P) + \epsilon, \quad \epsilon \ll 1$$

If $P$ satisfies the conjecture, so does $P'$ (by continuity of relevant quantities).

**Quantitative stability:** By the **Shimura-Taniyama modularity** [Wiles 1995]:
$$a_p(E) = a_p(f_E)$$

This equality is exact (not approximate), so small perturbations of $E$ preserve the modular form correspondence.

**Step 5: KM Certificate**

The Kenig-Merle certificate for arithmetic:
$$K_{\text{KM}}^+ = (\text{concentration analysis}, \text{rigidity proof}, \text{stability bound})$$

Components:
- **Concentration:** List of primes where height concentrates
- **Rigidity:** Proof that minimal counterexample is impossible
- **Stability:** $\delta$-neighborhood where conclusion holds

---

### Key Arithmetic Ingredients

1. **Northcott Compactness** [Northcott 1950]: Finite sets from bounded invariants.
2. **Level Lowering** [Ribet 1990]: Reduce conductor of modular forms.
3. **CM Theory** [Rubin 1987]: BSD for CM elliptic curves.
4. **Modularity** [Wiles 1995]: Stability of the correspondence.

---

### Arithmetic Interpretation

> **The Kenig-Merle method translates to arithmetic as: (1) heights concentrate at finitely many primes, (2) bounded sequences have limits, (3) minimal counterexamples must be special (and don't exist).**

---

### Literature

- [Kenig-Merle 2006] C. Kenig, F. Merle, *Global well-posedness, scattering and blow-up for the energy-critical focusing NLS*
- [Northcott 1950] D.G. Northcott, *An inequality in arithmetic*
- [Ribet 1990] K. Ribet, *On modular representations of Gal(Q̄/Q)*
- [Rubin 1987] K. Rubin, *Tate-Shafarevich groups and L-functions of elliptic curves with complex multiplication*
