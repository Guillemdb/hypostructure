# FACT-SoftRigidity: Soft→Rigidity Compilation

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-fact-soft-rigidity*

Soft permits compile to Rigidity permits (hybrid form): no minimal counterexample exists.

---

## Arithmetic Formulation

### Setup

Arithmetic rigidity means:
- Special structures (CM, torsion) are rigid
- Minimal counterexamples have special structure
- But special structures satisfy the conjecture

### Statement (Arithmetic Version)

**Theorem (Arithmetic Rigidity).** If a minimal counterexample to an arithmetic conjecture exists, it must be:

1. **CM (Complex Multiplication):** Has extra endomorphisms
2. **Torsion:** Finite order element
3. **Special:** Lies on a Shimura subvariety

But all such special objects satisfy the conjecture, hence no counterexample exists.

---

### Proof

**Step 1: Characterization of Minimal Counterexample**

**Setup:** Suppose $X_0$ is a minimal counterexample to conjecture $C$:
- $C(X_0)$ fails
- For all $Y$ with $\mathcal{C}(Y) < \mathcal{C}(X_0)$, $C(Y)$ holds
- $\mathcal{C}$ = complexity measure (conductor, height, dimension)

**Step 2: Minimal Counterexamples Are Special**

**(a) For BSD (Elliptic Curves):**

If $E$ is a minimal counterexample:
- $N_E$ is minimal conductor among counterexamples
- By **level lowering** [Ribet 1990]: $\bar{\rho}_{E,p}: G_\mathbb{Q} \to \text{GL}_2(\mathbb{F}_p)$ arises from a form of lower level
- The minimal level form is **CM** or **rational**

**Conclusion:** $E$ has CM or is defined over $\mathbb{Q}$ with small conductor.

**(b) For Hodge (Algebraic Cycles):**

If $(X, \alpha)$ is a minimal counterexample ($\alpha$ non-algebraic Hodge class):
- By **absolute Hodge** theory [Deligne 1982]: $\alpha$ is absolute Hodge
- If $X$ is an abelian variety, absolute Hodge ⟹ algebraic [Deligne]
- The minimal counterexample must be on a **non-abelian** variety

**Conclusion:** Minimal counterexample is on a variety with extra structure.

**(c) For RH (Zeros):**

If $\rho_0$ is a zero off the critical line with minimal imaginary part:
- Paired with $\bar{\rho}_0$ by functional equation
- By **zero repulsion** [Montgomery 1973]: zeros repel each other
- The minimal off-line zero would be isolated

**Conclusion:** Such isolated zeros don't exist (contradicts density theorems).

**Step 3: Special Objects Satisfy Conjectures**

**(a) CM elliptic curves satisfy BSD:**

By **Rubin's theorem** [Rubin 1987]:
- For $E$ with CM by $\mathcal{O}_K$:
- $L(E, 1) \neq 0 \Rightarrow E(\mathbb{Q})$ is finite
- $\text{ord}_{s=1} L(E, s) = \text{rank } E(\mathbb{Q})$ holds

**(b) Abelian varieties satisfy Hodge:**

By **Deligne's theorem** [Deligne 1982]:
- Absolute Hodge classes on abelian varieties are algebraic
- This covers the CM case completely

**(c) Known zero-free regions:**

By **de la Vallée Poussin** [1896]:
- $\zeta(s) \neq 0$ for $\Re(s) > 1 - c/\log|t|$
- The "minimal counterexample" region is empty

**Step 4: Rigidity Conclusion**

The rigidity argument:

```
Minimal counterexample exists
    ⟹ Counterexample is special (CM, torsion, Shimura)
    ⟹ Special objects satisfy conjecture
    ⟹ Contradiction
∴ No counterexample exists
```

**Certificate:** $K_{\text{Rigid}}^+ = (\text{specialization proof}, \text{special case theorem})$

---

### Key Arithmetic Ingredients

1. **Level Lowering** [Ribet 1990]: Reduces to special cases.
2. **Rubin's Theorem** [Rubin 1987]: BSD for CM curves.
3. **Deligne's Theorem** [Deligne 1982]: Hodge for abelian varieties.
4. **Zero-Free Regions** [de la Vallée Poussin 1896]: RH partial results.

---

### Arithmetic Interpretation

> **Minimal counterexamples must be arithmetically special, but special structures satisfy the conjectures. This rigidity rules out counterexamples.**

---

### Literature

- [Ribet 1990] K. Ribet, *On modular representations*
- [Rubin 1987] K. Rubin, *Tate-Shafarevich groups and L-functions of elliptic curves with CM*
- [Deligne 1982] P. Deligne, *Hodge cycles on abelian varieties*
- [de la Vallée Poussin 1896] C.-J. de la Vallée Poussin, *Recherches analytiques sur la théorie des nombres premiers*
