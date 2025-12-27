# LOCK-Entropy: Holographic Entropy Lock — GMT Translation

## Original Statement (Hypostructure)

The holographic entropy lock shows that entropy bounds from holography create barriers preventing configurations from exceeding area-based entropy limits.

## GMT Setting

**Holographic Entropy:** Entropy bounded by boundary area

**Bekenstein Bound:** Maximum entropy in region with given energy and size

**Lock:** Configurations violating entropy bound are inaccessible

## GMT Statement

**Theorem (Holographic Entropy Lock).** For geometric configurations:

1. **Bekenstein Bound:** $S \leq 2\pi ER/(\hbar c)$ for region of size $R$ and energy $E$

2. **Holographic Bound:** $S \leq A/(4l_P^2)$ where $A$ is boundary area

3. **Lock:** Configurations with $S > S_{\max}$ are dynamically excluded

## Proof Sketch

### Step 1: Bekenstein Bound

**Theorem (Bekenstein, 1981):** For weakly gravitating system with energy $E$ in region of size $R$:
$$S \leq \frac{2\pi k_B ER}{\hbar c}$$

**Reference:** Bekenstein, J. D. (1981). Universal upper bound on the entropy-to-energy ratio for bounded systems. *Phys. Rev. D*, 23, 287-298.

### Step 2: Black Hole Entropy

**Bekenstein-Hawking Formula:** Black hole entropy:
$$S_{\text{BH}} = \frac{k_B c^3 A}{4 G \hbar} = \frac{A}{4 l_P^2}$$

where $l_P = \sqrt{G\hbar/c^3}$ is Planck length.

**Reference:** Hawking, S. W. (1975). Particle creation by black holes. *Comm. Math. Phys.*, 43, 199-220.

### Step 3: Holographic Principle

**'t Hooft-Susskind:** Maximum entropy in region bounded by area:
$$S \leq \frac{A}{4 l_P^2}$$

**Covariant Bound (Bousso):** Entropy on light-sheet bounded by area.

**Reference:** Bousso, R. (1999). The holographic principle. *Rev. Mod. Phys.*, 74, 825-874.

### Step 4: GMT Entropy

**Entropy of Current:** For $T \in \mathbf{I}_k(M)$, define complexity entropy:
$$S(T) = \log(\text{number of microstates compatible with } T)$$

**Geometric Bound:** $S(T) \leq C \cdot \mathcal{H}^{k-1}(\partial T)$ (area of boundary).

### Step 5: Entropy and Singular Sets

**Singular Entropy:** Singularities carry entropy:
$$S_{\text{sing}}(T) \propto \mathcal{H}^{k-2}(\text{sing}(T))$$

(lower-dimensional, hence smaller than bulk entropy).

**Bound:** Singular set complexity bounded by boundary area.

### Step 6: Thermodynamic Lock

**Second Law:** Entropy cannot decrease:
$$S(T_t) \geq S(T_0)$$

for irreversible flow.

**Lock:** Configurations with entropy above equilibrium maximum are excluded from approach.

### Step 7: AdS/CFT Realization

**Ryu-Takayanagi Formula:** For CFT region $A$:
$$S_A = \frac{\text{Area}(\gamma_A)}{4 G_N}$$

where $\gamma_A$ is minimal surface homologous to $A$ in AdS bulk.

**Reference:** Ryu, S., Takayanagi, T. (2006). Holographic derivation of entanglement entropy from AdS/CFT. *Phys. Rev. Lett.*, 96, 181602.

### Step 8: Minimal Surfaces and Entropy

**GMT Connection:** Ryu-Takayanagi uses area-minimizing surfaces:
- Find $\gamma_A \in \mathbf{I}_{d-1}(\text{AdS})$ with $\partial \gamma_A = \partial A$
- Minimize area
- Area gives entanglement entropy

**Reference:** Headrick, M., Takayanagi, T. (2007). A holographic proof of the strong subadditivity of entanglement entropy. *Phys. Rev. D*, 76, 106013.

### Step 9: Entropy Barriers

**Maximum Entropy Configuration:** Given boundary $\partial T$:
$$S_{\max} = C \cdot \mathcal{H}^{k-1}(\partial T)$$

**Barrier:** No current with boundary $\partial T$ can have entropy $> S_{\max}$.

**Lock:** Dynamics constrained by entropy bounds.

### Step 10: Compilation Theorem

**Theorem (Holographic Entropy Lock):**

1. **Bekenstein:** $S \leq 2\pi ER/\hbar c$

2. **Holographic:** $S \leq A/4l_P^2$

3. **Minimal Surface:** Entropy computed from area-minimizer

4. **Lock:** Entropy bound excludes configurations

**Applications:**
- Information bounds in geometric measure theory
- Holographic complexity
- Constraints on singular sets

## Key GMT Inequalities Used

1. **Bekenstein:**
   $$S \leq 2\pi ER/\hbar c$$

2. **Area Bound:**
   $$S \leq A/4l_P^2$$

3. **Ryu-Takayanagi:**
   $$S_A = \text{Area}(\gamma_A)/4G_N$$

4. **Singular Entropy:**
   $$S_{\text{sing}} \propto \mathcal{H}^{k-2}(\text{sing})$$

## Literature References

- Bekenstein, J. D. (1981). Universal upper bound on entropy. *Phys. Rev. D*, 23.
- Hawking, S. W. (1975). Particle creation by black holes. *Comm. Math. Phys.*, 43.
- Bousso, R. (1999). The holographic principle. *Rev. Mod. Phys.*, 74.
- Ryu, S., Takayanagi, T. (2006). Holographic entanglement entropy. *Phys. Rev. Lett.*, 96.
- Headrick, M., Takayanagi, T. (2007). Strong subadditivity. *Phys. Rev. D*, 76.
