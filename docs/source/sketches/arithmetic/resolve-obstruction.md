# RESOLVE-Obstruction: Obstruction Capacity Collapse

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-resolve-obstruction*

Obstructions to resolution have bounded capacity: if obstructions exceed a threshold, resolution becomes impossible; below threshold, obstructions can be overcome.

---

## Arithmetic Formulation

### Setup

"Obstruction capacity" in arithmetic means:
- **Obstruction:** Shafarevich-Tate group, Brauer-Manin obstruction
- **Capacity:** Size/complexity measure of obstruction
- **Collapse:** Obstruction exceeding bounds blocks solutions

### Statement (Arithmetic Version)

**Theorem (Arithmetic Obstruction Capacity).** For variety $X/\mathbb{Q}$:

1. **Obstruction measure:** $\text{Cap}(X) = \#\text{Ш}(X/\mathbb{Q})$ or Brauer group size
2. **Capacity bound:** If $\text{Cap}(X) < \infty$, rational points may exist
3. **Collapse:** If obstruction is "infinite" or non-trivial, Hasse principle fails

---

### Proof

**Step 1: Shafarevich-Tate Group**

**Definition:** For abelian variety $A/K$:
$$\text{Ш}(A/K) = \ker\left(H^1(K, A) \to \prod_v H^1(K_v, A)\right)$$

**Obstruction:** $\text{Ш}(A/K) \neq 0$ obstructs local-global principle for torsors.

**Finiteness conjecture [Tate-Shafarevich]:** $\#\text{Ш}(A/K) < \infty$.

**Step 2: Brauer-Manin Obstruction**

**Brauer group:**
$$\text{Br}(X) = H^2_{\text{ét}}(X, \mathbb{G}_m)$$

**Brauer-Manin set:**
$$X(\mathbb{A}_\mathbb{Q})^{\text{Br}} = \{(P_v) \in X(\mathbb{A}_\mathbb{Q}) : \sum_v \text{inv}_v(A(P_v)) = 0 \ \forall A \in \text{Br}(X)\}$$

**Obstruction [Manin 1970]:**
$$X(\mathbb{Q}) \subset X(\mathbb{A}_\mathbb{Q})^{\text{Br}} \subset X(\mathbb{A}_\mathbb{Q})$$

If $X(\mathbb{A}_\mathbb{Q})^{\text{Br}} = \emptyset$ but $X(\mathbb{A}_\mathbb{Q}) \neq \emptyset$: Brauer-Manin obstruction.

**Step 3: Capacity Measure**

**For Ш:**
$$\text{Cap}_{\text{Ш}}(A) = \log \#\text{Ш}(A/K)$$

(conjecturally finite)

**For Brauer:**
$$\text{Cap}_{\text{Br}}(X) = \#(\text{Br}(X)/\text{Br}_0(X))$$

**Step 4: BSD and Capacity**

**BSD Formula [Birch-Swinnerton-Dyer]:**
$$\lim_{s \to 1} \frac{L(E, s)}{(s-1)^r} = \frac{\#\text{Ш}(E/\mathbb{Q}) \cdot \Omega_E \cdot \text{Reg}(E) \cdot \prod_p c_p}{\#E(\mathbb{Q})_{\text{tors}}^2}$$

**Capacity bound:** $\#\text{Ш}$ appears in BSD; controlling it controls L-value.

**Step 5: Capacity Collapse**

**Theorem:** If obstruction exceeds capacity:
- $\text{Ш}(A/K)[p^\infty] = \infty$ (if BSD fails): no finite capacity
- $X(\mathbb{A}_\mathbb{Q})^{\text{Br}} = \emptyset$: obstruction blocks all rational points

**Collapse criterion:**
$$\text{Cap}(X) > C_{\text{threshold}} \implies X(\mathbb{Q}) = \emptyset$$

**Step 6: Positive Capacity Implies Solvability**

**Theorem [Descent]:** If $\text{Ш}(E/\mathbb{Q})[2] = 0$ and Selmer group is small:
$$\text{rank}(E/\mathbb{Q}) = \text{rank}(\text{Sel}^2(E/\mathbb{Q}))$$

**Effective descent:** Small capacity $\Rightarrow$ explicit point search succeeds.

---

### Key Arithmetic Ingredients

1. **Shafarevich-Tate Group** [Tate 1962]: Principal obstruction.
2. **Brauer-Manin Obstruction** [Manin 1970]: Cohomological obstruction.
3. **BSD Conjecture** [Birch-Swinnerton-Dyer 1965]: L-function formula.
4. **Descent Theory** [Cassels 1962]: Computing Selmer groups.

---

### Arithmetic Interpretation

> **Arithmetic obstructions have measurable capacity. The Shafarevich-Tate group and Brauer-Manin obstruction measure how far local solutions are from globalizing. When capacity is finite and small, descent methods find rational points; when capacity "collapses" (obstruction is essential), no global solutions exist.**

---

### Literature

- [Tate 1962] J. Tate, *Duality theorems in Galois cohomology*
- [Manin 1970] Y. Manin, *Le groupe de Brauer-Grothendieck*
- [Cassels 1962] J.W.S. Cassels, *Arithmetic on curves of genus 1*
- [Poonen-Stoll 1999] B. Poonen, M. Stoll, *The Cassels-Tate pairing*
