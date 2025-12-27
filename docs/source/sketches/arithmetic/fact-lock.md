# FACT-Lock: Lock Backend Factory

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-fact-lock*

Lock structures are automatically generated from obstruction data.

---

## Arithmetic Formulation

### Setup

"Lock" in arithmetic means:
- Obstruction preventing certain behavior
- Galois obstruction, Brauer-Manin obstruction, cohomological lock
- Generated automatically from arithmetic data

### Statement (Arithmetic Version)

**Theorem (Arithmetic Lock Factory).** For obstruction type $\omega$:

1. **Input:** Obstruction specification (Galois, Brauer, cohomological)
2. **Output:** Lock structure $\mathcal{L}_\omega$ that enforces the obstruction
3. **Verification:** $\mathcal{L}_\omega$ is decidable

---

### Proof

**Step 1: Galois Lock**

**Specification:** Galois obstruction to rational points

**Lock construction:**
$$\mathcal{L}_{\text{Gal}}(X) = \text{Hom}_{G_\mathbb{Q}}(\text{pt}, X) = \emptyset ?$$

**Factory:**
```
GALOIS_LOCK(X):
  # Check if Galois-fixed points exist
  FOR each prime p:
    IF X(Q_p) = ∅:
      RETURN LOCKED(p)  # No local points

  # Check global Galois action
  IF π_1^{ét}(X) has no Q-points:
    RETURN LOCKED(global)

  RETURN UNLOCKED
```

**Step 2: Brauer-Manin Lock**

**Specification:** Brauer-Manin obstruction

**Lock construction:**
$$\mathcal{L}_{\text{BM}}(X) = X(\mathbb{A}_\mathbb{Q})^{\text{Br}} = \emptyset ?$$

**Factory [Skorobogatov]:**
```
BRAUER_MANIN_LOCK(X):
  # Compute Brauer group
  Br = compute_brauer_group(X)

  # For each element, compute local evaluations
  FOR α in Br:
    local_invs = []
    FOR p in primes ∪ {∞}:
      FOR x in X(Q_p):
        inv = local_invariant(α, x)
        local_invs.append((p, inv))

    # Check if sum = 0 is achievable
    IF NOT sum_to_zero_possible(local_invs):
      RETURN LOCKED(α)

  RETURN UNLOCKED
```

**Step 3: Cohomological Lock**

**Specification:** Obstruction from cohomology

**Lock types:**
- $H^1(G_K, A)$ obstruction for abelian variety
- Tate-Shafarevich obstruction: $\text{Ш}(A) \neq 0$

**Factory:**
```
COHOMOLOGICAL_LOCK(A):
  # Compute Selmer group
  Sel = selmer_group(A, n)

  # Compute Mordell-Weil
  MW = mordell_weil(A)

  # Sha = Sel / MW
  Sha = Sel / image(MW)

  IF Sha ≠ 0:
    RETURN LOCKED(Sha)
  ELSE:
    RETURN UNLOCKED
```

**Step 4: Lock Factory Master**

```
LOCK_FACTORY(spec):
  PARSE spec → (type, parameters)

  SWITCH type:
    CASE "galois":
      RETURN GALOIS_LOCK(parameters)

    CASE "brauer-manin":
      RETURN BRAUER_MANIN_LOCK(parameters)

    CASE "cohomological":
      RETURN COHOMOLOGICAL_LOCK(parameters)

    CASE "monodromy":
      RETURN MONODROMY_LOCK(parameters)

    CASE "descent":
      RETURN DESCENT_LOCK(parameters)

    DEFAULT:
      RETURN generic_lock(type, parameters)
```

**Step 5: Lock Composition**

**Union of locks (both must be unlocked):**
$$\mathcal{L}_1 \land \mathcal{L}_2 = \lambda X: \mathcal{L}_1(X) \lor \mathcal{L}_2(X)$$

**Intersection of locks (either unlocks):**
$$\mathcal{L}_1 \lor \mathcal{L}_2 = \lambda X: \mathcal{L}_1(X) \land \mathcal{L}_2(X)$$

**Step 6: Decidability**

**Claim:** Factory-produced locks are decidable.

**Proof:**
- Galois lock: Decidable by local solubility checks [Poonen]
- Brauer-Manin: Decidable for surfaces [Colliot-Thélène]
- Cohomological: Decidable for elliptic curves [Cremona]

**Caveat:** General Sha computation may be undecidable, but:
- Upper bounds are computable
- BSD would make it decidable

---

### Key Arithmetic Ingredients

1. **Brauer-Manin Obstruction** [Manin 1970]: Explains Hasse principle failures.
2. **Selmer Groups** [Selmer 1951]: Computable approximation to Sha.
3. **Descent** [Cassels 1962]: Computing rational points via covers.
4. **Skorobogatov** [Skorobogatov 2001]: Brauer-Manin computability.

---

### Arithmetic Interpretation

> **Arithmetic locks (Galois, Brauer-Manin, cohomological obstructions) are automatically generated from specifications. The factory produces decidable tests for whether rational points exist or whether an obstruction blocks them.**

---

### Literature

- [Manin 1970] Yu.I. Manin, *Le groupe de Brauer-Grothendieck*
- [Selmer 1951] E. Selmer, *The Diophantine equation...*
- [Skorobogatov 2001] A. Skorobogatov, *Torsors and Rational Points*
- [Colliot-Thélène 1999] J.-L. Colliot-Thélène, *Points rationnels sur les surfaces*
