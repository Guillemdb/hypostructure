# UP-IncComplete: Inconclusive Discharge

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-up-inc-complete*

Inconclusive cases can be discharged through alternative verification paths.

---

## Arithmetic Formulation

### Setup

"Inconclusive discharge" in arithmetic means:
- Primary method fails to verify a conjecture
- Alternative path provides verification
- Inconclusivity is resolved through different techniques

### Statement (Arithmetic Version)

**Theorem (Arithmetic Inconclusivity Discharge).** When primary verification is inconclusive:

1. **Alternative path:** Different techniques may succeed
2. **Modularity discharge:** Analytic failure → algebraic approach
3. **Descent discharge:** Cohomological approach when direct fails

---

### Proof

**Step 1: BSD Inconclusivity**

**Primary approach:** Compute $\text{ord}_{s=1} L(E, s)$ and compare to rank.

**Inconclusive case:** Numerical computation uncertain:
- $L(E, 1) \approx 0$ (is it exactly 0?)
- High rank makes descent difficult

**Discharge via Heegner points [Gross-Zagier]:**
1. If $w(E) = -1$, construct Heegner point $P_K$
2. If $\hat{h}(P_K) \neq 0$, then rank $\geq 1$
3. Combined with $L'(E, 1) \neq 0$ proves rank = 1

**Step 2: Rank Inconclusivity**

**Primary:** 2-descent to bound rank.

**Inconclusive case:** 2-Selmer gives upper bound but doesn't determine exact rank.

**Discharge options:**
1. **Higher descent:** 4-descent, 8-descent to tighten bounds
2. **Visualisation:** Search for points of bounded height
3. **Heegner points:** Construct explicit non-torsion point

**Step 3: Sha Inconclusivity**

**Primary:** Compute 2-Selmer and subtract known rank.

**Inconclusive case:** Upper bound on $|\text{Ш}[2]|$ but not exact value.

**Discharge via BSD formula:**
$$|\text{Ш}| = \frac{L^{(r)}(E, 1) \cdot |E(\mathbb{Q})_{\text{tors}}|^2}{r! \cdot \Omega \cdot \text{Reg} \cdot \prod c_p}$$

If other quantities are known, $|\text{Ш}|$ is determined.

**Step 4: RH Inconclusivity**

**Primary:** Numerically verify zeros on critical line.

**Inconclusive case:** Zero too close to call (near $\Re(s) = 1/2$).

**Discharge options:**
1. **Higher precision:** Compute to more decimal places
2. **Explicit formula:** Use connection to primes
3. **Odlyzko-Schönhage:** Rigorous zero verification algorithm

**Step 5: Hodge Inconclusivity**

**Primary:** Check if Hodge class is algebraic.

**Inconclusive case:** Class is absolute Hodge but algebraicity uncertain.

**Discharge for abelian varieties [Deligne]:**
- Absolute Hodge on AV → algebraic
- Reduces general case to AV case

**Step 6: Discharge Certificate**

The discharge certificate:
$$K_{\text{Disch}}^+ = (\text{primary failure}, \text{alternative path}, \text{resolution})$$

**Components:**
- **Primary:** Why initial approach was inconclusive
- **Alternative:** Which technique resolved it
- **Resolution:** Final verification

**Example (BSD for $E$):**
- Primary: L-value computation inconclusive at $s = 1$
- Alternative: Heegner point construction
- Resolution: $P_K$ has height $\neq 0$, proving rank ≥ 1

---

### Key Arithmetic Ingredients

1. **Gross-Zagier** [Gross-Zagier 1986]: Heegner points for inconclusive cases.
2. **Higher Descent** [Cassels 1962]: Sharper Selmer bounds.
3. **Odlyzko-Schönhage** [OS 1988]: Rigorous zero verification.
4. **Deligne's Theorem** [Deligne 1982]: Absolute Hodge → algebraic for AVs.

---

### Arithmetic Interpretation

> **Arithmetic inconclusivity is discharged through alternative techniques. When L-value computation is uncertain, Heegner points provide algebraic verification. When descent is inconclusive, higher descent or point search resolves. Multiple verification paths ensure no case remains truly inconclusive.**

---

### Literature

- [Gross-Zagier 1986] B. Gross, D. Zagier, *Heegner points and derivatives of L-series*
- [Cassels 1962] J.W.S. Cassels, *Diophantine equations with special reference to elliptic curves*
- [Odlyzko-Schönhage 1988] A.M. Odlyzko, A. Schönhage, *Fast algorithms for multiple evaluations of the Riemann zeta function*
- [Deligne 1982] P. Deligne, *Hodge cycles on abelian varieties*
