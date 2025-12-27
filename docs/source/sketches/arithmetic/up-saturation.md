# UP-Saturation: Saturation Promotion

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-up-saturation*

Saturation of inequalities promotes to equality at the boundary.

---

## Arithmetic Formulation

### Setup

"Saturation" in arithmetic means:
- Inequality becomes equality under special conditions
- Bounds are achieved by extremal objects
- Promotion from "<" to "="

### Statement (Arithmetic Version)

**Theorem (Arithmetic Saturation).** For arithmetic inequality $f(X) \leq B$:

1. **Saturation locus:** $\{X : f(X) = B\}$ is non-empty
2. **Promotion:** Objects in saturation locus are extremal/special
3. **Characterization:** Saturating objects have explicit arithmetic description

---

### Proof

**Step 1: Height Saturation**

**Inequality:** For elliptic curve $E/\mathbb{Q}$ with good reduction at $p$:
$$h(j_E) \leq c \cdot \log N_E$$

**Saturation:** Curves with $h(j_E) = c \cdot \log N_E$ are:
- CM curves with small discriminant
- Curves with conductor = prime power

**Characterization [Silverman 1984]:**
$$h(j_E) = c \cdot \log N_E \iff E \text{ has minimal height for its conductor}$$

**Step 2: Ramanujan Saturation**

**Inequality (Ramanujan bound):**
$$|a_p(E)| \leq 2\sqrt{p}$$

**Saturation:** $|a_p(E)| = 2\sqrt{p}$ never occurs for $p$ of good reduction (strict bound).

**Supersingular case:** $a_p(E) = 0$ is "saturation" of:
$$|a_p(E)| \geq 0$$

CM curves saturate more symmetry constraints.

**Step 3: L-function Saturation**

**Inequality (Lindelöf hypothesis):**
$$|L(1/2 + it, \chi)| \leq C_\epsilon |t|^\epsilon$$

**Saturation:** Achieved along critical moments. By [Soundararajan 2008]:
$$\max_{|t| \leq T} |\zeta(1/2 + it)| \sim \exp\left(\sqrt{\frac{\log T}{2\log\log T}}\right)$$

**Extremal objects:** Zeros that nearly collide, creating large peaks.

**Step 4: Mordell-Weil Saturation**

**Subgroup saturation:** $\Lambda \subset E(\mathbb{Q})$ is saturated at $p$ if:
$$E(\mathbb{Q})/\Lambda \text{ has no } p\text{-torsion}$$

**Saturation promotion [Silverman]:**
- Input: Generators $P_1, \ldots, P_r$ of $\Lambda$
- Test: For each prime $p$, check if any $Q$ satisfies $pQ \in \Lambda$ but $Q \notin \Lambda$
- Output: Saturated generators

**Algorithm:**
```
SATURATE(Λ, p):
  FOR Q in E(Q)_p-division points:
    IF pQ ∈ Λ and Q ∉ Λ:
      Λ = Λ + ZQ  # Enlarge
  RETURN Λ
```

**Step 5: Regulator Saturation**

**Inequality:** For rank $r$ curve:
$$\text{Reg}_E \geq c_r > 0$$

**Saturation:** Curves with minimal regulator for their rank.

**Characterization:** By [Lang's conjecture], there are finitely many curves with $\text{Reg}_E < R$ for fixed $R$.

**Extremal:** CM curves often have small regulators.

**Step 6: Saturation Certificate**

The saturation certificate:
$$K_{\text{Sat}}^+ = (f, B, X_{\text{sat}}, \text{proof of } f(X_{\text{sat}}) = B)$$

**Components:**
- **Inequality:** $f \leq B$
- **Saturating object:** $X_{\text{sat}}$
- **Verification:** Computation showing $f(X_{\text{sat}}) = B$

---

### Key Arithmetic Ingredients

1. **Silverman's Height Bounds** [Silverman 1984]: Height vs conductor.
2. **Hasse-Weil** [Weil 1948]: Ramanujan bound.
3. **Mordell-Weil Saturation** [Cremona 1997]: Subgroup saturation algorithm.
4. **Soundararajan** [Soundararajan 2008]: Extreme values of L-functions.

---

### Arithmetic Interpretation

> **Arithmetic inequalities saturate at extremal objects. Height bounds saturate for CM curves, Mordell-Weil subgroups saturate when complete, L-function bounds saturate at extreme values. Saturation characterizes special arithmetic structures.**

---

### Literature

- [Silverman 1984] J. Silverman, *The arithmetic of elliptic curves*
- [Weil 1948] A. Weil, *Sur les courbes algébriques et les variétés qui s'en déduisent*
- [Cremona 1997] J. Cremona, *Algorithms for Modular Elliptic Curves*
- [Soundararajan 2008] K. Soundararajan, *Extreme values of zeta and L-functions*
