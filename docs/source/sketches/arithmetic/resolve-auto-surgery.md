# RESOLVE-AutoSurgery: Automatic Surgery

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-resolve-auto-surgery*

Surgery operations can be selected and applied automatically based on singularity type.

---

## Arithmetic Formulation

### Setup

"Automatic surgery" in arithmetic means:
- Given singularity type, the appropriate resolution is determined algorithmically
- No human choice needed—surgery is canonical
- Algorithm terminates with resolved object

### Statement (Arithmetic Version)

**Theorem (Automatic Arithmetic Surgery).** For singularity type $\tau$:

1. **Detection:** $\tau$ is algorithmically identifiable from local data
2. **Selection:** Canonical surgery $\sigma_\tau$ is determined by $\tau$
3. **Application:** $\sigma_\tau$ is effectively computable

---

### Proof

**Step 1: Singularity Detection**

**For elliptic curves (Tate's algorithm):**

Input: Weierstrass equation $y^2 = x^3 + ax + b$ and prime $p$

```
DETECT_SINGULARITY(E, p):
  Compute v_p(Δ) where Δ = -16(4a³ + 27b²)
  Compute v_p(c₄) where c₄ = -48a

  IF v_p(Δ) = 0:
    RETURN "good reduction"
  ELIF v_p(c₄) = 0:
    RETURN "multiplicative" (type I_n or I*_n)
  ELSE:
    Apply Tate's classification → type II, III, IV, II*, III*, IV*
```

**Output:** Kodaira-Néron type

**Step 2: Surgery Selection**

| Singularity Type | Canonical Surgery |
|------------------|-------------------|
| Good reduction | None needed |
| Multiplicative (I_n) | Already semistable |
| Additive (II-IV) | Quadratic twist or base change |
| Wild (II*-IV*) | Higher degree base change |

**Selection algorithm:**
```
SELECT_SURGERY(type):
  SWITCH type:
    CASE "good": RETURN identity
    CASE "multiplicative": RETURN identity (semistable)
    CASE "additive":
      IF type in {II, IV}:
        RETURN quadratic_twist(d) for appropriate d
      ELSE:
        RETURN base_change(K') for minimal K'
    CASE "wild":
      Compute minimal K' achieving semistable
      RETURN base_change(K')
```

**Step 3: Surgery Application**

**Quadratic twist surgery:**
```
APPLY_TWIST(E, d):
  # E: y² = x³ + ax + b
  # E^(d): y² = x³ + d²ax + d³b
  RETURN E^(d)
```

**Base change surgery:**
```
APPLY_BASE_CHANGE(E, K'):
  # Extend scalars from K to K'
  RETURN E ×_K K'
```

**Blowup surgery:**
```
APPLY_BLOWUP(X, Z):
  # Blowup X along smooth center Z
  Construct Proj of blowup algebra
  RETURN (X̃, π: X̃ → X)
```

**Step 4: Semistable Reduction Algorithm**

**Input:** Abelian variety $A/K$, prime $\mathfrak{p}$

**Algorithm [Grothendieck]:**
```
SEMISTABLE_REDUCTION(A, p):
  1. Compute Néron model A over O_K
  2. Identify component group Φ_p
  3. IF Φ_p is unipotent:
       Compute toric part dimension t
       Find K'/K with [K':K] | |Φ_p|
       RETURN base_change(A, K')
  4. ELSE:
       Apply Raynaud's uniformization
       RETURN explicit semistable model
```

**Termination:** By [Grothendieck], algorithm terminates for finite $K'/K$.

**Step 5: Full Automatic Surgery**

**Unified algorithm:**
```
AUTO_SURGERY(X):
  resolutions = []

  FOR each prime p with bad reduction:
    type = DETECT_SINGULARITY(X, p)
    surgery = SELECT_SURGERY(type)
    X = APPLY_SURGERY(X, surgery)
    resolutions.append((p, surgery))

  VERIFY: X now has good/semistable reduction everywhere
  RETURN (X, resolutions)
```

**Correctness:** By semistable reduction theorem, this always succeeds.

**Step 6: Complexity Analysis**

**Conductor bound:** Surgery needed only at $p | N_X$.

**Degree bound:** Base change degree $\leq$ exponent of conductor at $p$.

**Termination:** Algorithm is polynomial in $\log N_X$.

---

### Key Arithmetic Ingredients

1. **Tate's Algorithm** [Tate 1975]: Classifies reduction types.
2. **Grothendieck's Semistable Reduction** [SGA 7]: Existence of resolution.
3. **Raynaud's Uniformization** [Raynaud 1974]: Explicit semistable models.
4. **Néron Models** [Néron 1964]: Canonical smooth models.

---

### Arithmetic Interpretation

> **Arithmetic surgery is fully automatic: detect singularity type via Tate's algorithm, select canonical resolution (twist, base change, or blowup), and apply. The semistable reduction theorem guarantees termination.**

---

### Literature

- [Tate 1975] J. Tate, *Algorithm for determining the type of a singular fiber*
- [Grothendieck 1972] A. Grothendieck, *SGA 7: Groupes de Monodromie*
- [Raynaud 1974] M. Raynaud, *Schémas en groupes de type (p,...,p)*
- [Néron 1964] A. Néron, *Modèles minimaux des variétés abéliennes*
