# FACT-Surgery: Surgery Schema Factory

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-fact-surgery*

Surgery schemas are automatically generated from singularity classifications.

---

## Arithmetic Formulation

### Setup

"Surgery schema" in arithmetic means:
- Recipe for resolving a singularity type
- Automatically determined by Kodaira-Néron classification
- Produces explicit resolution map

### Statement (Arithmetic Version)

**Theorem (Arithmetic Surgery Factory).** For singularity type $\tau$:

1. **Input:** Kodaira-Néron type (I_n, II, III, IV, I*_n, II*, III*, IV*)
2. **Output:** Surgery schema $\sigma_\tau$ that resolves the singularity
3. **Correctness:** $\sigma_\tau$ produces semistable or good reduction

---

### Proof

**Step 1: Kodaira-Néron Classification**

For elliptic curve $E/K$ at prime $\mathfrak{p}$:

| Type | Description | Component Group | Surgery |
|------|-------------|-----------------|---------|
| I_0 | Good | 1 | None |
| I_n | Multiplicative | $\mathbb{Z}/n\mathbb{Z}$ | Already semistable |
| II | Cusp | 1 | Quadratic twist |
| III | Two tangent lines | $\mathbb{Z}/2\mathbb{Z}$ | Quadratic twist |
| IV | Three concurrent lines | $\mathbb{Z}/3\mathbb{Z}$ | Cubic base change |
| I*_n | Bad multiplicative | $\mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$ | Quadratic twist |
| II* | Bad cusp | 1 | Sextic base change |
| III* | Bad tangent | $\mathbb{Z}/2\mathbb{Z}$ | Quartic base change |
| IV* | Bad concurrent | $\mathbb{Z}/3\mathbb{Z}$ | Cubic base change |

**Step 2: Surgery Schema Generation**

**For type II (cusp):**
```
SURGERY_II(E, p):
  # Find d such that E^(d) has better reduction
  FOR d in squarefree divisors of p:
    E_d = quadratic_twist(E, d)
    IF reduction_type(E_d, p) in {I_0, I_n}:
      RETURN (E_d, d)
  # Fallback to base change
  RETURN base_change(E, Q(√p))
```

**For type IV (three concurrent):**
```
SURGERY_IV(E, p):
  # Need cube root
  K' = Q(∛p)
  E' = base_change(E, K')
  ASSERT reduction_type(E', p) in {I_0, I_n}
  RETURN (E', K')
```

**For type I*_n (bad multiplicative):**
```
SURGERY_I*n(E, p):
  # Quadratic twist removes additive part
  d = find_twist_discriminant(E, p)
  E_d = quadratic_twist(E, d)
  ASSERT reduction_type(E_d, p) = I_{2n}
  RETURN (E_d, d)
```

**Step 3: Factory Algorithm**

```
SURGERY_FACTORY(type):
  SWITCH type:
    CASE "I_0": RETURN identity_surgery
    CASE "I_n": RETURN identity_surgery  # already semistable
    CASE "II":  RETURN SURGERY_II
    CASE "III": RETURN SURGERY_III
    CASE "IV":  RETURN SURGERY_IV
    CASE "I*_n": RETURN SURGERY_I*n
    CASE "II*": RETURN SURGERY_II*
    CASE "III*": RETURN SURGERY_III*
    CASE "IV*": RETURN SURGERY_IV*
```

**Step 4: Semistable Reduction Theorem**

**Theorem [Grothendieck]:** Every surgery produced by the factory achieves semistable reduction.

**Proof:** For each type:
- Type II, III, I*_n: Quadratic twist suffices [Silverman]
- Type IV, IV*: Cubic base change suffices
- Type II*, III*: Higher degree base change suffices

The factory covers all cases by Kodaira-Néron completeness.

**Step 5: Minimal Surgery**

**Claim:** The factory produces minimal surgeries.

**Minimality criteria:**
- Twist before base change (simpler)
- Use smallest degree extension
- Preserve as much rational structure as possible

**Algorithm:**
```
MINIMAL_SURGERY(E, p):
  type = kodaira_neron_type(E, p)

  IF type in {I_0, I_n}:
    RETURN identity  # no surgery needed

  # Try twist first
  FOR d in [p, -p, -1, ...]:
    E_d = twist(E, d)
    IF type(E_d, p) is semistable:
      RETURN twist_surgery(d)

  # Fall back to base change
  deg = minimal_extension_degree(type)
  RETURN base_change_surgery(deg)
```

**Step 6: Verification**

**Post-surgery verification:**
```
VERIFY_SURGERY(E, surgery):
  E' = apply(surgery, E)
  FOR each prime p with bad reduction:
    type' = kodaira_neron_type(E', p)
    ASSERT type' in {I_0, I_n}  # semistable
  RETURN SUCCESS
```

---

### Key Arithmetic Ingredients

1. **Kodaira-Néron Classification** [Kodaira 1964, Néron 1964]: Complete type list.
2. **Tate's Algorithm** [Tate 1975]: Type determination.
3. **Grothendieck's Theorem** [SGA 7]: Semistable reduction exists.
4. **Twist Theory** [Silverman 1994]: Quadratic twist effects.

---

### Arithmetic Interpretation

> **Surgery schemas are automatically generated from Kodaira-Néron type. Each fiber type has a canonical resolution—quadratic twist for types II/III/I*, base change for types IV/II*/III*/IV*. The factory produces minimal surgeries achieving semistable reduction.**

---

### Literature

- [Kodaira 1964] K. Kodaira, *On compact analytic surfaces II-III*
- [Néron 1964] A. Néron, *Modèles minimaux*
- [Tate 1975] J. Tate, *Algorithm for determining the type of a singular fiber*
- [Grothendieck 1972] A. Grothendieck, *SGA 7*
