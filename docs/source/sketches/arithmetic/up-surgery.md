# UP-Surgery: Surgery Promotion

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-up-surgery*

Surgery operations promote local improvements to global resolution.

---

## Arithmetic Formulation

### Setup

"Surgery promotion" in arithmetic means:
- Local resolution at each prime promotes to global resolution
- Semistable reduction at all primes → global good behavior
- Local-to-global principle for surgery

### Statement (Arithmetic Version)

**Theorem (Arithmetic Surgery Promotion).** Local surgeries combine:

1. **Local-to-global:** If $X_p$ is resolved for all $p$, then $X$ is resolved
2. **Finite surgery:** Only finitely many primes need surgery
3. **Composition:** Surgeries compose to give minimal resolution

---

### Proof

**Step 1: Local Surgery at Each Prime**

For elliptic curve $E/\mathbb{Q}$:

**At each prime $p | N_E$:**
- Identify reduction type (Tate's algorithm)
- Apply local surgery (twist or base change)
- Achieve semistable reduction at $p$

**Step 2: Finiteness of Surgery Locus**

**Claim:** Only finitely many primes need surgery.

**Proof:**
- $E$ has good reduction outside $\{p : p | N_E\}$
- This set is finite (conductor is a positive integer)
- Only these primes require surgery

**Bound:** Number of surgeries $\leq \omega(N_E)$ (number of prime factors).

**Step 3: Surgery Composition**

**For primes $p_1, \ldots, p_k$:**

Let $\sigma_i$ be the surgery at $p_i$.

**Composition:**
$$\sigma = \sigma_k \circ \cdots \circ \sigma_1$$

**Independence:** If $p_i \neq p_j$, surgeries at $p_i$ and $p_j$ commute.

**Proof:** Local surgeries at different primes affect different local factors.

**Step 4: Global Resolution**

**Theorem:** After composing all local surgeries:
$$E' = \sigma(E)$$

has semistable reduction at all primes.

**Proof:** By Grothendieck's semistable reduction theorem:
- Each $\sigma_i$ resolves $p_i$
- Surgeries don't worsen other primes
- Composition achieves global semistable reduction

**Step 5: Minimal Surgery**

**Definition:** Surgery $\sigma$ is minimal if:
- Degree of base change is minimal
- Twist discriminant is minimal
- Conductor of $\sigma(E)$ is minimal among resolutions

**Existence [Grothendieck]:**
- Minimal surgery exists
- Unique up to isomorphism over the base change field

**Algorithm:**
```
MINIMAL_SURGERY(E):
  surgeries = []
  FOR p in prime_divisors(N_E):
    type = kodaira_type(E, p)
    surgery = MINIMAL_LOCAL_SURGERY(E, p, type)
    surgeries.append(surgery)

  RETURN compose(surgeries)
```

**Step 6: Promotion Certificate**

The surgery promotion certificate:
$$K_{\text{SurgProm}}^+ = (\{(p_i, \sigma_i)\}, \text{composition } \sigma, \text{resolution proof})$$

**Components:**
- **Local surgeries:** List of $(p, \sigma_p)$ pairs
- **Global surgery:** Composition $\sigma$
- **Resolution:** Proof that $\sigma(E)$ is semistable everywhere

---

### Key Arithmetic Ingredients

1. **Grothendieck's Theorem** [SGA 7]: Semistable reduction achievable.
2. **Tate's Algorithm** [Tate 1975]: Identify local reduction type.
3. **Local-Global Principle** [Hasse 1923]: Combine local data.
4. **Néron Models** [Néron 1964]: Canonical global models.

---

### Arithmetic Interpretation

> **Local surgeries at each bad prime compose to give global resolution. Since only finitely many primes divide the conductor, and surgeries at different primes commute, the global surgery is the product of local surgeries. This promotes local improvement to global semistable reduction.**

---

### Literature

- [Grothendieck 1972] A. Grothendieck, *SGA 7: Groupes de Monodromie*
- [Tate 1975] J. Tate, *Algorithm for determining the type of a singular fiber*
- [Hasse 1923] H. Hasse, *Über die Äquivalenz quadratischer Formen*
- [Néron 1964] A. Néron, *Modèles minimaux des variétés abéliennes*
