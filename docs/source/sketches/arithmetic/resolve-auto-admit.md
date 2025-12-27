# RESOLVE-AutoAdmit: Automatic Admissibility

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-resolve-auto-admit*

Surgery admissibility can be determined automatically from local data.

---

## Arithmetic Formulation

### Setup

"Automatic admissibility" in arithmetic means:
- Admissibility of surgery is computable from local invariants
- No global computation required for decision
- Local-to-global principle for surgery

### Statement (Arithmetic Version)

**Theorem (Automatic Arithmetic Admissibility).** Surgery admissibility is determined by local invariants:

1. **Local test:** For each prime $p$, test $\sigma$ at $p$
2. **Global conclusion:** $\sigma$ admissible $\iff$ admissible at all $p$
3. **Computability:** Each local test is decidable

---

### Proof

**Step 1: Local Admissibility Criterion**

For surgery $\sigma: X \dashrightarrow X'$:

**Local admissibility at $p$:**
$$\sigma \text{ is } p\text{-admissible} \iff X_{\mathbb{Q}_p} \sim X'_{\mathbb{Q}_p}$$

where $\sim$ denotes preserving:
- Local points: $X(\mathbb{Q}_p) \leftrightarrow X'(\mathbb{Q}_p)$
- Local L-factor: $L_p(X, s) = L_p(X', s)$
- Local Galois: $\rho_X|_{G_{\mathbb{Q}_p}} \cong \rho_{X'}|_{G_{\mathbb{Q}_p}}$

**Step 2: Global from Local**

**Theorem:** $\sigma$ is globally admissible $\iff$ $\sigma$ is $p$-admissible for all $p$.

**Proof:**

**($\Rightarrow$)** Global admissibility implies local admissibility at each $p$ (restriction).

**($\Leftarrow$)** By Hasse principle for admissibility:
- L-function: $L(X, s) = \prod_p L_p(X, s)$ (Euler product)
- If $L_p(X, s) = L_p(X', s)$ for all $p$, then $L(X, s) = L(X', s)$

For rational points, use local-global principle or Brauer-Manin.

**Step 3: Decidability of Local Tests**

**Algorithm for $p$-admissibility:**

**(a) For reduction type:**
```
LOCAL_REDUCTION(E, p):
  Use Tate's algorithm to classify E mod p
  RETURN reduction_type ∈ {good, multiplicative, additive}
```

**(b) For local points:**
```
LOCAL_POINTS(X, p):
  IF X is smooth mod p:
    RETURN |X(F_p)| by point counting
  ELSE:
    Use Hensel lifting
```

**(c) For local Galois:**
```
LOCAL_GALOIS(ρ, p):
  Compute ρ|_{I_p} (inertia action)
  Compute ρ(Frob_p) (Frobenius)
  RETURN (inertia_type, Frobenius_trace)
```

**Step 4: Automatic Admissibility Algorithm**

```
AUTO_ADMIT(σ: X → X'):
  admissible = TRUE
  FOR p in primes up to conductor bound:
    IF NOT p_admissible(σ, p):
      admissible = FALSE
      BREAK

  # Finite check suffices by:
  # - Conductor bound: Only finitely many p with bad reduction
  # - Good reduction: σ automatically p-admissible

  RETURN admissible
```

**Termination:** Finite since $\{p : p | N_X \cdot N_{X'}\}$ is finite.

**Step 5: Examples**

**(a) Quadratic twist $E \mapsto E^{(d)}$:**
- Admissible at $p \nmid 2d$: Same reduction type
- Check $p | 2d$: Local computation with Tate

**(b) Base change $A \mapsto A_{K'}$:**
- Admissible at $p$ unramified in $K'/K$
- Check ramified primes: Finite set

**(c) Level lowering $f \mapsto f'$:**
- Admissible at $p$ with $p \nmid N/N'$
- Check $p | N/N'$: Ribet's criterion

---

### Key Arithmetic Ingredients

1. **Tate's Algorithm** [Tate 1975]: Local reduction classification.
2. **Hensel's Lemma** [Hensel 1908]: Lifting local solutions.
3. **Conductor Formula** [Ogg 1967]: Bounds ramification.
4. **Local-Global** [Hasse 1923]: Global from local data.

---

### Arithmetic Interpretation

> **Surgery admissibility is automatically decidable: check local admissibility at each prime dividing the conductor. The global decision follows from the local-global principle for arithmetic invariants.**

---

### Literature

- [Tate 1975] J. Tate, *Algorithm for determining the type of a singular fiber*
- [Hensel 1908] K. Hensel, *Theorie der algebraischen Zahlen*
- [Ogg 1967] A. Ogg, *Elliptic curves and wild ramification*
- [Hasse 1923] H. Hasse, *Über die Äquivalenz quadratischer Formen*
