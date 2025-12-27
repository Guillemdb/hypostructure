# FACT-Transport: Equivalence + Transport Factory

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-fact-transport*

Equivalences are automatically equipped with transport structure.

---

## Arithmetic Formulation

### Setup

"Transport" in arithmetic means:
- Moving arithmetic structures along equivalences
- Isogenies, Galois conjugation, base change
- Automatic lifting of properties

### Statement (Arithmetic Version)

**Theorem (Arithmetic Transport Factory).** For equivalence $\phi: X \to Y$:

1. **Transport exists:** Properties of $X$ transport to $Y$
2. **Automatic:** Transport maps are explicitly computable
3. **Coherence:** Transports compose correctly

---

### Proof

**Step 1: Isogeny Transport**

For isogeny $\phi: E \to E'$ of elliptic curves:

**Transported structures:**
- Points: $\phi: E(\mathbb{Q}) \to E'(\mathbb{Q})$ (homomorphism)
- Torsion: $\phi: E[n] \to E'[n]$ (up to kernel)
- L-function: $L(E, s) = L(E', s)$ (same!)
- Height: $\hat{h}(\phi(P)) = \deg(\phi) \cdot \hat{h}(P)$

**Transport formula [Silverman]:**
$$\phi_*: E(\mathbb{Q})/\ker\phi \xrightarrow{\sim} \text{Im}(\phi) \subset E'(\mathbb{Q})$$

**Step 2: Galois Transport**

For Galois extension $K'/K$ and variety $X/K$:

**Base change transport:**
$$X_{K'} = X \times_K K'$$

**Transported structures:**
- Points: $X(K) \hookrightarrow X(K')$ (inclusion)
- Cohomology: $H^i(X_{\bar{K}}, \mathbb{Q}_\ell) \cong H^i(X_{K'} \times_{K'} \bar{K}, \mathbb{Q}_\ell)$
- L-function: $L(X_{K'}, s) = \prod_{\chi} L(X, s, \chi)$ where $\chi$ runs over characters of $\text{Gal}(K'/K)$

**Step 3: Transport Factory**

```
TRANSPORT_FACTORY(φ: X → Y):
  IDENTIFY type of φ:
    - isogeny
    - base change
    - birational map
    - twist

  SWITCH type:
    CASE "isogeny":
      RETURN {
        points: λP ↦ φ(P),
        height: λh ↦ deg(φ) * h,
        L_function: identity,
        torsion: λT ↦ φ(T) mod ker(φ)
      }

    CASE "base_change":
      RETURN {
        points: λP ↦ P (injection),
        cohomology: identity,
        L_function: product_over_characters
      }

    CASE "birational":
      RETURN {
        points: λP ↦ φ(P) if defined,
        L_function: ratio of local factors,
        Picard: Pic(X) → Pic(Y) + exceptional
      }

    CASE "twist":
      d = twist_parameter
      RETURN {
        points: λP ↦ twist_map(P, d),
        L_function: λL ↦ L(s, χ_d),
        conductor: λN ↦ N * d² / gcd(N, d)²
      }
```

**Step 4: Coherence**

**Composition law:** For $\phi: X \to Y$ and $\psi: Y \to Z$:
$$\text{Transport}_{\psi \circ \phi} = \text{Transport}_\psi \circ \text{Transport}_\phi$$

**Verification:**
- Heights: $\hat{h}((\psi \circ \phi)(P)) = \deg(\psi)\deg(\phi) \cdot \hat{h}(P)$ ✓
- L-functions: Compose correctly ✓
- Points: $(\psi \circ \phi)_* = \psi_* \circ \phi_*$ ✓

**Step 5: Automatic Transport Construction**

**Algorithm:**
```
AUTO_TRANSPORT(equiv):
  # Detect equivalence type
  type = classify_equivalence(equiv)

  # Build transport structure
  transport = TRANSPORT_FACTORY(equiv)

  # Verify coherence
  IF is_composition(equiv):
    (φ, ψ) = decompose(equiv)
    ASSERT transport = compose(AUTO_TRANSPORT(φ), AUTO_TRANSPORT(ψ))

  RETURN transport
```

**Step 6: Examples**

**(a) 2-isogeny $E \to E'$:**
- Points: $E(\mathbb{Q}) \to E'(\mathbb{Q})$ with kernel of order 2
- Height: Doubles
- L-function: Unchanged

**(b) Quadratic base change $E \to E_K$ where $K = \mathbb{Q}(\sqrt{d})$:**
- Points: $E(\mathbb{Q}) \hookrightarrow E(K)$
- L-function: $L(E_K, s) = L(E, s) \cdot L(E^{(d)}, s)$

**(c) Quadratic twist $E \to E^{(d)}$:**
- Points: Twisted by $d$
- L-function: Twisted by $\chi_d$

---

### Key Arithmetic Ingredients

1. **Isogeny Theory** [Silverman 1986]: Transport along isogenies.
2. **Base Change** [Weil 1956]: Extension of scalars.
3. **Twist Theory** [Silverman 1994]: Quadratic twist structure.
4. **Descent** [Cassels 1962]: Recovering rational structure.

---

### Arithmetic Interpretation

> **Arithmetic equivalences (isogenies, base changes, twists) automatically carry transport structure. The factory produces explicit maps for points, heights, L-functions, and cohomology. Transport composes coherently.**

---

### Literature

- [Silverman 1986] J. Silverman, *The Arithmetic of Elliptic Curves*
- [Weil 1956] A. Weil, *The field of definition of a variety*
- [Silverman 1994] J. Silverman, *Advanced Topics in the Arithmetic of Elliptic Curves*
- [Cassels 1962] J.W.S. Cassels, *Arithmetic on curves of genus 1*
