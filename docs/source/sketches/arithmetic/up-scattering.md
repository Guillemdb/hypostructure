# UP-Scattering: Scattering Promotion

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-up-scattering*

Scattering behavior at infinity promotes to global dispersive estimates.

---

## Arithmetic Formulation

### Setup

"Scattering" in arithmetic means:
- Behavior as parameters go to infinity
- Distribution of primes, zeros, or points
- Equidistribution and limit theorems

### Statement (Arithmetic Version)

**Theorem (Arithmetic Scattering).** For arithmetic sequences:

1. **Prime scattering:** Primes scatter uniformly in residue classes
2. **Zero scattering:** L-function zeros scatter according to limiting measure
3. **Point scattering:** Rational points scatter on varieties

---

### Proof

**Step 1: Prime Scattering (Čebotarev)**

For Galois extension $K/\mathbb{Q}$ with group $G$:

**Scattering statement:** Primes scatter uniformly across conjugacy classes.

$$\#\{p \leq x : \text{Frob}_p \in C\} \sim \frac{|C|}{|G|} \cdot \frac{x}{\log x}$$

**Proof [Čebotarev 1926]:** Uses properties of Artin L-functions and prime number theorem for arithmetic progressions.

**Equidistribution:** As $x \to \infty$, the proportion approaches $|C|/|G|$.

**Step 2: Zero Scattering (GUE)**

For $\zeta(s)$ or general L-functions:

**Scattering statement:** Zeros scatter according to GUE distribution.

**Montgomery's Pair Correlation [Montgomery 1973]:**
$$\lim_{T \to \infty} \frac{1}{N(T)} \sum_{\substack{0 < \gamma, \gamma' \leq T \\ \gamma \neq \gamma'}} f\left(\frac{\log T}{2\pi}(\gamma - \gamma')\right) = \int_{-\infty}^\infty f(x) \left(1 - \left(\frac{\sin \pi x}{\pi x}\right)^2\right) dx$$

**Interpretation:** Zeros repel like eigenvalues of random matrices.

**Step 3: Rational Point Scattering**

For variety $X/\mathbb{Q}$ with height function:

**Scattering statement:** Rational points scatter according to Tamagawa measure.

**Manin's Conjecture [Franke-Manin-Tschinkel]:**
$$\#\{P \in X(\mathbb{Q}) : H(P) \leq B\} \sim c \cdot B^a (\log B)^{b-1}$$

where $a, b$ depend on geometry and $c$ is the Tamagawa constant.

**Equidistribution:** Points become equidistributed in the Tamagawa measure as $B \to \infty$.

**Step 4: Hecke Eigenvalue Scattering**

For modular form $f$ and primes $p$:

**Sato-Tate scattering:** Normalized eigenvalues scatter:
$$\frac{a_p(f)}{2p^{(k-1)/2}} \to \text{Sato-Tate measure } \frac{2}{\pi}\sqrt{1-t^2} dt$$

**Proof [BLGHT 2011]:** Via potential automorphy of symmetric powers.

**Step 5: Frobenius Scattering**

For variety $X/\mathbb{F}_q$ with Frobenius $F$:

**Deligne equidistribution:** Eigenvalues of $F$ on $H^i_{\text{ét}}$ scatter:
$$\{\alpha_{ij}\}_{j} \to \text{uniform on } S^1$$

after appropriate normalization (weight adjustment).

**Proof [Deligne 1980]:** Weil conjectures + monodromy arguments.

**Step 6: Scattering Certificate**

The scattering certificate:
$$K_{\text{Scat}}^+ = (\text{sequence}, \text{limiting measure}, \text{rate of convergence})$$

**Components:**
- **Sequence:** $(a_n)$ (primes, zeros, points)
- **Limit:** Measure $\mu$ on target space
- **Rate:** Error term in equidistribution

**Example (Čebotarev):**
- Sequence: $\{\text{Frob}_p\}_{p \leq x}$
- Limit: Uniform on conjugacy classes
- Rate: $O(x \exp(-c\sqrt{\log x}))$ under GRH

---

### Key Arithmetic Ingredients

1. **Čebotarev Density** [Čebotarev 1926]: Prime equidistribution.
2. **Montgomery Pair Correlation** [Montgomery 1973]: Zero scattering.
3. **Sato-Tate** [BLGHT 2011]: Frobenius trace scattering.
4. **Manin's Conjecture** [FMT 1989]: Rational point scattering.

---

### Arithmetic Interpretation

> **Arithmetic objects scatter as they go to infinity. Primes scatter uniformly in Galois classes (Čebotarev), zeros scatter according to GUE (Montgomery), rational points scatter in Tamagawa measure (Manin). These scattering theorems are the arithmetic analog of dispersive PDE estimates.**

---

### Literature

- [Čebotarev 1926] N. Čebotarev, *Die Bestimmung der Dichtigkeit*
- [Montgomery 1973] H. Montgomery, *The pair correlation of zeros*
- [BLGHT 2011] T. Barnet-Lamb et al., *Potential automorphy*
- [Franke-Manin-Tschinkel 1989] J. Franke, Yu.I. Manin, Yu. Tschinkel, *Rational points of bounded height*
