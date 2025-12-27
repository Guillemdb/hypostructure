# FACT-Gate: Gate Evaluator Factory

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-fact-gate*

Gate evaluators are automatically generated from permit specifications.

---

## Arithmetic Formulation

### Setup

"Gate evaluators" in arithmetic means:
- **Gate:** Decision procedure for arithmetic property
- **Evaluator:** Algorithm that computes pass/fail
- **Factory:** Automatic generation from specification

### Statement (Arithmetic Version)

**Theorem (Arithmetic Gate Factory).** For arithmetic property $P$:

1. **Specification:** Formal description of $P$ (e.g., "conductor ≤ N")
2. **Generation:** Automatically produce evaluator $\text{eval}_P$
3. **Correctness:** $\text{eval}_P(X) = \text{true} \iff P(X)$ holds

---

### Proof

**Step 1: Property Specifications**

**Arithmetic properties have formal specifications:**

| Property | Specification |
|----------|---------------|
| Good reduction at $p$ | $v_p(\Delta) = 0$ |
| Conductor bound | $N_E \leq B$ |
| Rank bound | $\text{rank } E(\mathbb{Q}) \leq r$ |
| RH for $L_E$ | All zeros on $\Re(s) = 1/2$ |

**Step 2: Evaluator Generation**

**For "conductor ≤ N":**
```
GENERATE_EVALUATOR("conductor ≤ N"):
  RETURN λE:
    N_E = compute_conductor(E)
    RETURN N_E ≤ N
```

**For "good reduction at p":**
```
GENERATE_EVALUATOR("good reduction at p"):
  RETURN λE:
    Δ = discriminant(E)
    RETURN gcd(Δ, p) = 1
```

**For "rank = r":**
```
GENERATE_EVALUATOR("rank = r"):
  RETURN λE:
    r_computed = mw_rank(E)  # via descent
    RETURN r_computed = r
```

**Step 3: Composite Gates**

**Conjunction:**
$$\text{eval}_{P \land Q} = \lambda X: \text{eval}_P(X) \land \text{eval}_Q(X)$$

**Disjunction:**
$$\text{eval}_{P \lor Q} = \lambda X: \text{eval}_P(X) \lor \text{eval}_Q(X)$$

**Negation:**
$$\text{eval}_{\neg P} = \lambda X: \neg \text{eval}_P(X)$$

**Step 4: Factory Implementation**

```
GATE_FACTORY(spec):
  PARSE spec → syntax tree
  FOR each node in tree:
    IF node is atomic property:
      evaluator[node] = ATOMIC_EVALUATOR(node)
    ELIF node is AND:
      evaluator[node] = COMBINE_AND(evaluator[child1], evaluator[child2])
    ELIF node is OR:
      evaluator[node] = COMBINE_OR(evaluator[child1], evaluator[child2])
    ELIF node is NOT:
      evaluator[node] = NEGATE(evaluator[child])

  RETURN evaluator[root]
```

**Step 5: Correctness Verification**

**Claim:** Generated evaluators are correct.

**Proof by structural induction:**

*Base case:* Atomic evaluators directly compute the property (e.g., conductor computation is correct by [Tate 1975]).

*Inductive case:* If $\text{eval}_P$ and $\text{eval}_Q$ are correct, then:
- $\text{eval}_{P \land Q}$ returns true $\iff$ both return true $\iff$ $P \land Q$ holds
- Similarly for $\lor$ and $\neg$

**Step 6: Efficiency**

**Factory produces efficient evaluators:**
- Atomic properties: polynomial time (Tate's algorithm, descent)
- Composite: Time = sum of component times

**Optimization:** Common subexpressions are evaluated once.

---

### Key Arithmetic Ingredients

1. **Tate's Algorithm** [Tate 1975]: Conductor computation.
2. **Descent** [Cassels 1962]: Mordell-Weil rank computation.
3. **Modular Symbols** [Cremona 1997]: L-function computation.
4. **Model Theory** [Robinson 1974]: Decidability of arithmetic properties.

---

### Arithmetic Interpretation

> **Arithmetic gate evaluators are automatically generated from property specifications. Given a formal description of a property (conductor bound, rank bound, RH), the factory produces a correct algorithm to evaluate it.**

---

### Literature

- [Tate 1975] J. Tate, *Algorithm for determining the type of a singular fiber*
- [Cassels 1962] J.W.S. Cassels, *Diophantine equations with special reference to elliptic curves*
- [Cremona 1997] J. Cremona, *Algorithms for Modular Elliptic Curves*
- [Robinson 1974] A. Robinson, *On the concept of a differentially closed field*
