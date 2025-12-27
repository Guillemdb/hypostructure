# LOCK-Reconstruction: Structural Reconstruction Lock — GMT Translation

## Original Statement (Hypostructure)

The structural reconstruction lock shows that certain invariants uniquely determine the structure, locking the configuration against ambiguity through reconstruction theorems.

## GMT Setting

**Reconstruction:** Structure determined by invariant data

**Uniqueness:** Data determines structure uniquely

**Lock:** Multiple structures with same invariants are forbidden

## GMT Statement

**Theorem (Structural Reconstruction Lock).** For rectifiable currents:

1. **Invariant Data:** Mass, boundary, homology class, density function

2. **Reconstruction:** Under appropriate hypotheses, data determines current

3. **Lock:** Uniqueness prevents ambiguity in identification

## Proof Sketch

### Step 1: Uniqueness of Minimizers

**Theorem:** For strictly convex functional $\Phi$ on $\mathbf{I}_k(M)$:
$$\Phi(T) = \Phi(S), \partial T = \partial S \implies T = S$$

**Reference:** Federer, H. (1969). *Geometric Measure Theory*. Springer, §5.1.

### Step 2: Boundary and Mass Determine Current

**Plateau Problem:** Given $\Gamma = \partial T_0$, the area-minimizing current with boundary $\Gamma$ is unique (for generic $\Gamma$).

**Hardt-Simon:** Generically, minimizers are unique.

**Reference:** Hardt, R., Simon, L. (1979). Boundary regularity and embedded solutions for the oriented Plateau problem. *Ann. of Math.*, 110, 439-486.

### Step 3: Density Function

**Density:** For $T \in \mathbf{I}_k(M)$:
$$\Theta^k(T, x) = \lim_{r \to 0} \frac{\mathbf{M}(T \llcorner B_r(x))}{\omega_k r^k}$$

**Reconstruction:** Current $T$ determined by support and density function.

**Reference:** Preiss, D. (1987). Geometry of measures in $\mathbb{R}^n$: distribution, rectifiability, and densities. *Ann. of Math.*, 125, 537-643.

### Step 4: Federer's Structure Theorem

**Theorem:** For $T \in \mathbf{I}_k(M)$ with $\mathbf{M}(T) < \infty$:
$$T = \theta \cdot \llbracket M \rrbracket$$

where $M$ is $k$-rectifiable set and $\theta$ is integer-valued density.

**Reference:** Federer, H. (1969). *Geometric Measure Theory*. Springer, §4.2.

**Reconstruction:** Support $M$ and multiplicity $\theta$ determine $T$.

### Step 5: Flat Metric Determines Currents

**Flat Norm:**
$$\mathbf{F}(T) = \inf\{\mathbf{M}(R) + \mathbf{M}(S) : T = R + \partial S\}$$

**Completeness:** $\mathbf{I}_k(M)$ is complete in flat norm.

**Uniqueness:** Flat limit is unique.

**Reference:** Federer, H., Fleming, W. (1960). Normal and integral currents. *Ann. of Math.*, 72, 458-520.

### Step 6: Radon-Nikodym for Currents

**Decomposition:** Current $T$ decomposes uniquely:
$$T = T_{\text{abs}} + T_{\text{sing}}$$

absolutely continuous + singular parts.

**Reference:** Mattila, P. (1995). *Geometry of Sets and Measures*. Cambridge.

**Reconstruction:** Absolute and singular parts uniquely determined.

### Step 7: Spectral Reconstruction

**Inverse Spectral:** For manifolds, spectrum $\{\lambda_i\}$ determines geometry up to finite ambiguity.

**Reference:** Gordon, C., Webb, D., Wolpert, S. (1992). Isospectral plane domains. *Invent. Math.*, 110.

**Lock:** Spectral data usually determines manifold uniquely.

### Step 8: Hodge-Theoretic Reconstruction

**Hodge Decomposition:**
$$\Omega^k(M) = \mathcal{H}^k \oplus d\Omega^{k-1} \oplus \delta\Omega^{k+1}$$

**Reconstruction:** Harmonic representatives unique in cohomology class.

**Reference:** Hodge, W. V. D. (1941). *The Theory and Applications of Harmonic Integrals*. Cambridge.

### Step 9: Lock by Uniqueness

**Mechanism:** If invariant $I(T)$ uniquely determines $T$:
- $I(T_1) = I(T_2) \implies T_1 = T_2$
- No ambiguity in reconstruction
- Configuration "locked" by invariants

**Examples:**
- Minimal surface with given boundary (generically)
- Harmonic form in cohomology class
- Calibrated current in calibrated class

### Step 10: Compilation Theorem

**Theorem (Structural Reconstruction Lock):**

1. **Data:** Mass, boundary, homology, density function

2. **Uniqueness:** Data determines structure (under hypotheses)

3. **Reconstruction:** Explicit recovery from invariants

4. **Lock:** Uniqueness prevents structural ambiguity

**Applications:**
- Inverse problems in geometry
- Rigidity of minimal surfaces
- Uniqueness in variational problems

## Key GMT Inequalities Used

1. **Density Reconstruction:**
   $$T = \Theta \cdot \llbracket \text{spt}(T) \rrbracket$$

2. **Flat Completeness:**
   $$\mathbf{F}(T_n - T) \to 0 \implies T_n \to T$$

3. **Minimizer Uniqueness:**
   $$\Phi \text{ strictly convex} \implies \text{unique minimizer}$$

4. **Structure:**
   $$T \in \mathbf{I}_k \implies T = \theta \llbracket M \rrbracket$$

## Literature References

- Federer, H. (1969). *Geometric Measure Theory*. Springer.
- Federer, H., Fleming, W. (1960). Normal and integral currents. *Ann. of Math.*, 72.
- Preiss, D. (1987). Geometry of measures. *Ann. of Math.*, 125.
- Hardt, R., Simon, L. (1979). Boundary regularity. *Ann. of Math.*, 110.
- Mattila, P. (1995). *Geometry of Sets and Measures*. Cambridge.
- Hodge, W. V. D. (1941). *Harmonic Integrals*. Cambridge.
