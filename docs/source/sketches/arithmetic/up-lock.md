# UP-Lock: Lock Promotion

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-up-lock*

Lock structures promote local obstructions to global impossibility.

---

## Arithmetic Formulation

### Setup

"Lock promotion" in arithmetic means:
- Local obstruction at one prime → global obstruction
- Galois obstruction at one place → no global points
- Promotion from local-to-global impossibility

### Statement (Arithmetic Version)

**Theorem (Arithmetic Lock Promotion).** Local locks promote:

1. **Local-global obstruction:** $X(\mathbb{Q}_p) = \emptyset \Rightarrow X(\mathbb{Q}) = \emptyset$
2. **Brauer-Manin promotion:** Local Brauer obstruction → global obstruction
3. **Galois promotion:** Local Galois lock → global Galois lock

---

### Proof

**Step 1: Local Solubility Lock**

**Local emptiness locks globally:**

If $X(\mathbb{Q}_p) = \emptyset$ for some $p \leq \infty$:
$$X(\mathbb{Q}) \subset X(\mathbb{Q}_p) = \emptyset$$

**Promotion:** A single locked prime locks globally.

**Example:** $x^2 + y^2 = -1$ over $\mathbb{Q}$
- $X(\mathbb{R}) = \emptyset$ (no real solutions)
- Therefore $X(\mathbb{Q}) = \emptyset$

**Step 2: Brauer-Manin Lock**

**Setup:** $X(\mathbb{Q}_v) \neq \emptyset$ for all $v$, but Brauer obstruction exists.

**Brauer evaluation:**
$$\text{ev}_\alpha: X(\mathbb{A}_\mathbb{Q}) \to \mathbb{Q}/\mathbb{Z}$$
$$\text{ev}_\alpha((x_v)) = \sum_v \text{inv}_v(\alpha(x_v))$$

**Lock condition:** If for all $(x_v) \in \prod_v X(\mathbb{Q}_v)$:
$$\sum_v \text{inv}_v(\alpha(x_v)) \neq 0$$

then $X(\mathbb{Q}) = \emptyset$.

**Promotion:** Local Brauer values don't sum to zero → locked globally.

**Step 3: Galois Lock**

**Galois action on points:**
$$G_\mathbb{Q} \curvearrowright X(\overline{\mathbb{Q}})$$

**Fixed points:** $X(\mathbb{Q}) = X(\overline{\mathbb{Q}})^{G_\mathbb{Q}}$

**Lock mechanism:**
If $G_\mathbb{Q}$-action has no fixed points on any component:
$$X(\mathbb{Q}) = \emptyset$$

**Local-to-global:** By Čebotarev, if Frobenius action at $p$ has no fixed point for density 1 set of primes, then global action has no fixed point.

**Step 4: Descent Lock**

For covering $\pi: Y \to X$:

**Descent obstruction:**
$$X(\mathbb{Q}) = \bigcup_{\xi \in H^1(\mathbb{Q}, \text{Aut}(\pi))} \pi(Y_\xi(\mathbb{Q}))$$

**Lock:** If $Y_\xi(\mathbb{Q}) = \emptyset$ for all twists $\xi$:
$$X(\mathbb{Q}) = \emptyset$$

**Promotion:** Local obstructions on all twists promote to global.

**Step 5: Sha Lock**

**Tate-Shafarevich group:**
$$\text{Ш}(A) = \ker\left(H^1(\mathbb{Q}, A) \to \prod_v H^1(\mathbb{Q}_v, A)\right)$$

**Lock:** Non-trivial Ш means torsors locally trivial but globally non-trivial.

**Promotion:** Local triviality + global non-triviality = locked.

**Step 6: Lock Promotion Certificate**

The lock promotion certificate:
$$K_{\text{LockProm}}^+ = (\text{local locks}, \text{promotion rule}, \text{global lock})$$

**Components:**
- **Local locks:** $(p_1, \text{lock}_1), \ldots$
- **Promotion:** How local combines to global
- **Global:** Proof of global obstruction

**Examples:**
- ($\mathbb{R}$, no real points) → $X(\mathbb{Q}) = \emptyset$
- (Brauer, $\sum \text{inv} \neq 0$) → $X(\mathbb{Q}) = \emptyset$
- (Descent, all twists empty) → $X(\mathbb{Q}) = \emptyset$

---

### Key Arithmetic Ingredients

1. **Hasse Principle** [Hasse 1923]: Local-to-global (when it holds).
2. **Brauer-Manin** [Manin 1970]: Obstruction to Hasse principle.
3. **Descent** [Colliot-Thélène 1987]: Cover-based obstructions.
4. **Tate-Shafarevich** [Tate 1958]: Principal homogeneous spaces.

---

### Arithmetic Interpretation

> **Arithmetic locks promote from local to global. A single empty local point set locks globally. Brauer-Manin obstructions promote local invariant constraints to global impossibility. Descent locks combine twists. This promotion machinery explains failures of the Hasse principle.**

---

### Literature

- [Hasse 1923] H. Hasse, *Über die Äquivalenz quadratischer Formen*
- [Manin 1970] Yu.I. Manin, *Le groupe de Brauer-Grothendieck*
- [Colliot-Thélène 1987] J.-L. Colliot-Thélène, J.-J. Sansuc, *La descente sur les variétés rationnelles*
- [Tate 1958] J. Tate, *WC-groups over p-adic fields*
