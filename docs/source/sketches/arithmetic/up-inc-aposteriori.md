# UP-IncAposteriori: A-Posteriori Discharge

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-up-inc-aposteriori*

Inconclusive cases are discharged a-posteriori by the completed proof.

---

## Arithmetic Formulation

### Setup

"A-posteriori discharge" in arithmetic means:
- Proof of general theorem retroactively verifies special cases
- Inconclusivity for specific objects is discharged by theorem
- Bootstrap from proven theorem to individual verification

### Statement (Arithmetic Version)

**Theorem (A-Posteriori Arithmetic Discharge).** General theorems discharge special cases:

1. **Modularity:** Once proven for all $E/\mathbb{Q}$, discharges individual cases
2. **Fermat:** Proof of FLT discharges all $(a, b, c, n)$
3. **Sato-Tate:** Proof for all non-CM $E$ discharges individual curves

---

### Proof

**Step 1: Modularity Discharge**

**Before Wiles (1995):**
- Each curve $E$ required individual verification of modularity
- Computational: match $a_p(E)$ with $a_p(f)$ for many $p$
- Inconclusive for curves without known modular form

**After Wiles:**
$$\forall E/\mathbb{Q}: E \text{ is modular}$$

**A-posteriori discharge:**
- Every $E/\mathbb{Q}$ is now known to be modular
- No individual verification needed
- Theorem discharges all cases simultaneously

**Step 2: Fermat Discharge**

**Before Wiles (1995):**
- Individual cases of $x^n + y^n = z^n$ checked by various methods
- $n = 3$ (Euler), $n = 4$ (Fermat), $n = 5$ (Dirichlet), ...
- Cases $n > 10^6$ inconclusive

**After Wiles:**
$$\forall n \geq 3, \forall (a, b, c) \in \mathbb{Z}^3: a^n + b^n \neq c^n \text{ (non-trivially)}$$

**A-posteriori discharge:**
- All cases simultaneously proven
- No need for individual case analysis
- Modularity of Frey curve provides uniform proof

**Step 3: Sato-Tate Discharge**

**Before BLGHT (2011):**
- Individual curves required numerical verification of Sato-Tate
- Check distribution of $a_p/2\sqrt{p}$ for many $p$
- Never conclusive (only statistical evidence)

**After BLGHT:**
$$\forall E/\mathbb{Q} \text{ non-CM}: \{a_p/2\sqrt{p}\} \to \frac{2}{\pi}\sqrt{1-t^2} dt$$

**A-posteriori discharge:**
- Every non-CM curve satisfies Sato-Tate
- Numerical checks become redundant
- Theorem provides complete verification

**Step 4: BSD A-Posteriori (Partial)**

**Current state:**
- BSD proven for analytic rank $\leq 1$ [Gross-Zagier, Kolyvagin]
- Higher rank remains open

**A-posteriori for rank ≤ 1:**
- All curves with $\text{ord}_{s=1} L(E, s) \leq 1$ satisfy BSD
- Individual verification unnecessary for this class

**Future:** Complete BSD proof would discharge all cases.

**Step 5: Discharge Mechanism**

**General pattern:**
1. **Individual cases:** Each object $X_i$ needs verification
2. **General theorem:** $\forall X: P(X)$ is proven
3. **A-posteriori:** $P(X_i)$ holds by instantiation

**Logical structure:**
$$\frac{\forall X: P(X)}{P(X_i)} \text{ (Universal Instantiation)}$$

**Step 6: A-Posteriori Certificate**

The a-posteriori certificate:
$$K_{\text{AP}}^+ = (\text{general theorem}, \text{instance}, \text{instantiation proof})$$

**Components:**
- **Theorem:** Statement $\forall X: P(X)$ and its proof reference
- **Instance:** Specific object $X_0$
- **Instantiation:** Verification that $X_0$ satisfies premises

**Example (Modularity of $E$):**
- Theorem: All elliptic curves over $\mathbb{Q}$ are modular [Wiles 1995, BCDT 2001]
- Instance: $E: y^2 = x^3 - x$
- Instantiation: $E$ is an elliptic curve over $\mathbb{Q}$ ✓

---

### Key Arithmetic Ingredients

1. **Wiles' Theorem** [Wiles 1995]: Modularity of semistable curves.
2. **BCDT** [BCDT 2001]: Modularity of all elliptic curves.
3. **BLGHT** [BLGHT 2011]: Sato-Tate for non-CM curves.
4. **Gross-Zagier-Kolyvagin** [1986-1988]: BSD for rank ≤ 1.

---

### Arithmetic Interpretation

> **A-posteriori discharge uses proven general theorems to verify individual cases. Once modularity is proven for all elliptic curves, each curve's modularity follows by instantiation. This retroactive verification is the most powerful form of discharge—proving the general case eliminates all special case inconclusivity.**

---

### Literature

- [Wiles 1995] A. Wiles, *Modular elliptic curves and Fermat's Last Theorem*
- [BCDT 2001] C. Breuil, B. Conrad, F. Diamond, R. Taylor, *On the modularity of elliptic curves over Q*
- [BLGHT 2011] T. Barnet-Lamb, D. Geraghty, M. Harris, R. Taylor, *Potential automorphy*
- [Kolyvagin 1988] V. Kolyvagin, *Finiteness of E(Q) and Ш(E,Q)*
