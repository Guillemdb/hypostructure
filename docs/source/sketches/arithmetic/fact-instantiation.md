# FACT-Instantiation: Instantiation Metatheorem

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-fact-instantiation*

Abstract permits instantiate to concrete arithmetic certificates.

---

## Arithmetic Formulation

### Setup

"Instantiation" in arithmetic means:
- Abstract theorem schema becomes concrete statement
- Parameters are filled with specific arithmetic objects
- Proof adapts to the specific case

### Statement (Arithmetic Version)

**Theorem (Arithmetic Instantiation).** Given abstract permit schema $\Pi$ and arithmetic object $X$:

1. **Instantiation:** $\Pi[X]$ is a concrete arithmetic statement
2. **Certificate:** Proof of $\Pi$ instantiates to proof of $\Pi[X]$
3. **Automation:** Instantiation is algorithmically computable

---

### Proof

**Step 1: Abstract Permit Schemas**

**BSD schema:**
$$\Pi_{\text{BSD}}[E] = \left(\text{ord}_{s=1} L(E, s) = \text{rank } E(\mathbb{Q})\right)$$

**RH schema:**
$$\Pi_{\text{RH}}[L] = \left(\forall \rho: L(\rho) = 0 \Rightarrow \Re(\rho) = \frac{1}{2}\right)$$

**Hodge schema:**
$$\Pi_{\text{Hodge}}[X, \alpha] = \left(\alpha \in H^{p,p}(X) \cap H^{2p}(X, \mathbb{Q}) \Rightarrow \alpha \text{ is algebraic}\right)$$

**Step 2: Parameter Filling**

**For BSD with $E = X_0(11)$:**
$$\Pi_{\text{BSD}}[X_0(11)] = \left(\text{ord}_{s=1} L(X_0(11), s) = \text{rank } X_0(11)(\mathbb{Q})\right)$$

**Concrete values:**
- $L(X_0(11), 1) \neq 0$ (computed numerically)
- $X_0(11)(\mathbb{Q}) = \{O\}$ (only identity)
- Both sides = 0 ✓

**Step 3: Proof Instantiation**

**Abstract proof of BSD (rank 0 case):**
1. If $L(E, 1) \neq 0$, then by [Kolyvagin]: $E(\mathbb{Q})$ is finite
2. Hence $\text{rank } E(\mathbb{Q}) = 0$
3. And $\text{ord}_{s=1} L(E, s) = 0$

**Instantiated proof for $X_0(11)$:**
1. Compute $L(X_0(11), 1) = 0.2538...$ (Cremona)
2. $L(X_0(11), 1) \neq 0$, so by Kolyvagin: $X_0(11)(\mathbb{Q})$ is finite
3. Compute: $X_0(11)(\mathbb{Q}) = \{O\}$
4. Both sides = 0 ✓

**Step 4: Instantiation Algorithm**

```
INSTANTIATE(schema, object):
  # Parse schema
  (statement, proof_template) = parse_schema(schema)

  # Fill parameters
  concrete_statement = substitute(statement, object)

  # Compute required values
  values = compute_values(object, required_by(proof_template))

  # Instantiate proof
  concrete_proof = []
  FOR step in proof_template:
    concrete_step = substitute(step, object, values)
    IF step.requires_computation:
      result = compute(concrete_step)
      concrete_proof.append(result)
    ELSE:
      concrete_proof.append(cite(step.theorem, object))

  RETURN (concrete_statement, concrete_proof)
```

**Step 5: Certificate Generation**

**Instantiated certificate:**
$$K^+[E] = (\text{computed invariants}, \text{instantiated proof steps}, \text{references})$$

**For BSD certificate of $E$:**
```
{
  object: E,
  invariants: {
    conductor: N_E,
    rank: r,
    L_value: L(E, 1),
    regulator: Reg_E,
    Sha_order: |Sha(E)|
  },
  proof: [
    "L(E, 1) computed to be non-zero",
    "By Kolyvagin [1988], E(Q) is finite",
    "Rank = 0 verified by descent",
    "BSD equality verified numerically"
  ],
  status: VERIFIED
}
```

**Step 6: Meta-Instantiation**

**Claim:** Instantiation is itself an instance of a metatheorem.

The schema:
$$\Pi_{\text{Inst}}[\Pi, X] = \left(\Pi[X] \text{ is a valid arithmetic statement with proof}\right)$$

is instantiated to specific $(\Pi, X)$ pairs.

---

### Key Arithmetic Ingredients

1. **Kolyvagin's Theorem** [Kolyvagin 1988]: BSD for analytic rank ≤ 1.
2. **Cremona's Tables** [Cremona 1997]: Computed L-values and ranks.
3. **Gross-Zagier** [Gross-Zagier 1986]: Heegner point method for rank 1.
4. **Modularity** [Wiles 1995]: Enables L-function computation.

---

### Arithmetic Interpretation

> **Abstract arithmetic theorems (BSD, RH, Hodge schemas) instantiate to concrete statements about specific objects. The instantiation is algorithmic: fill parameters, compute values, and assemble the certificate. This automates proof generation for verified objects.**

---

### Literature

- [Kolyvagin 1988] V. Kolyvagin, *Finiteness of E(Q) and Ш(E,Q)*
- [Cremona 1997] J. Cremona, *Algorithms for Modular Elliptic Curves*
- [Gross-Zagier 1986] B. Gross, D. Zagier, *Heegner points and derivatives of L-series*
- [Wiles 1995] A. Wiles, *Modular elliptic curves and Fermat's Last Theorem*
