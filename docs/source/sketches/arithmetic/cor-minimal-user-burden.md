# COR-MinimalUserBurden: Minimal User Burden

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: cor-minimal-user-burden*

The user need only specify the target conjecture and object; all verification machinery is automatic.

---

## Arithmetic Formulation

### Setup

"Minimal user burden" in arithmetic means:
- User provides: elliptic curve, variety, or L-function
- System computes: all relevant invariants automatically
- User receives: verification certificate

### Statement (Arithmetic Version)

**Corollary (Minimal Arithmetic Input).** To verify BSD/RH/Hodge for object $X$:

1. **Input:** Defining equations or coefficients of $X$
2. **Automatic:** System computes conductor, L-function, rational points, etc.
3. **Output:** Verification status and certificate

---

### Proof

**Step 1: Elliptic Curve Input**

**User provides:** Weierstrass coefficients $(a_1, a_2, a_3, a_4, a_6)$

**System computes automatically:**
- Discriminant $\Delta$
- j-invariant
- Conductor $N_E$ [via Tate's algorithm]
- Mordell-Weil group $E(\mathbb{Q})$ [via descent]
- L-function $L(E, s)$ [via modular form]

**Proof of automation:** By [Cremona 1997], all these are algorithmically computable.

**Step 2: L-function Input**

**User provides:** Dirichlet coefficients $\{a_n\}$ or Euler product

**System computes automatically:**
- Analytic continuation
- Functional equation parameters $(\Gamma\text{-factors}, \epsilon)$
- Zero locations (numerically)
- Special values

**Proof of automation:** By [Rubinstein 2001], L-function analysis is effective.

**Step 3: Variety Input**

**User provides:** Defining polynomials $\{f_1, \ldots, f_k\}$

**System computes automatically:**
- Smooth locus
- Singularities and resolution
- Cohomology (via étale or de Rham)
- Zeta function

**Proof of automation:** By [Kedlaya 2001], point counting and zeta computation are polynomial time.

**Step 4: Verification Pipeline**

```
VERIFY(X):
  1. PARSE input → identify type (curve, surface, L-function)
  2. COMPUTE invariants automatically:
     - Conductor/discriminant
     - L-function coefficients
     - Rational points/cohomology
  3. RUN verification algorithm:
     - BSD: Compare rank vs ord L-value
     - RH: Check zeros on critical line
     - Hodge: Check algebraicity of Hodge classes
  4. RETURN certificate
```

**User burden:** Only step 1 (input) requires human involvement.

**Step 5: Databases and Lookup**

**For known objects:**
- LMFDB contains precomputed data
- User inputs label → system retrieves all invariants

**Example:**
- Input: `11.a1` (LMFDB label)
- Output: All of $E$'s invariants, L-function, rational points

**Minimal burden:** Single identifier suffices.

**Step 6: Certificate Generation**

The verification certificate includes:
$$K^+ = (\text{invariants}, \text{L-function data}, \text{verification result}, \text{proof references})$$

**User receives:**
- Pass/Fail for the conjecture
- All computed invariants
- References to theorems used

---

### Key Arithmetic Ingredients

1. **Cremona's Algorithms** [Cremona 1997]: Elliptic curve computations.
2. **Rubinstein's lcalc** [Rubinstein 2001]: L-function numerics.
3. **Kedlaya's Algorithm** [Kedlaya 2001]: Fast zeta computation.
4. **LMFDB** [LMFDB]: Database of L-functions and modular forms.

---

### Arithmetic Interpretation

> **The user need only provide defining data—coefficients, polynomials, or database labels. All verification machinery (conductor computation, L-function analysis, point counting) runs automatically, returning a certificate.**

---

### Literature

- [Cremona 1997] J. Cremona, *Algorithms for Modular Elliptic Curves*
- [Rubinstein 2001] M. Rubinstein, *Computational methods and experiments in analytic number theory*
- [Kedlaya 2001] K. Kedlaya, *Counting points on hyperelliptic curves*
- [LMFDB] The LMFDB Collaboration, *The L-functions and Modular Forms Database*, https://www.lmfdb.org
