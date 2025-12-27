# KRNL-Trichotomy: Structural Resolution

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-krnl-trichotomy*

Every trajectory with finite breakdown time classifies into exactly one of three outcomes: Global Existence (dispersion), Global Regularity (concentration with permits satisfied), or Genuine Singularity (permits violated).

---

## Arithmetic Formulation

### Setup

Consider an arithmetic dynamical system where the "trajectory" is the behavior of an arithmetic object under a natural operation:

- **State space:** The set of algebraic numbers $\overline{\mathbb{Q}}$, or points on an algebraic variety $X(\overline{\mathbb{Q}})$
- **Evolution:** Galois action, Frobenius iteration, or height-increasing operations
- **Height:** The Weil height $h: \overline{\mathbb{Q}} \to \mathbb{R}_{\geq 0}$

The "breakdown time" corresponds to the point where heights become unbounded or arithmetic structure degenerates.

### Statement (Arithmetic Version)

**Theorem (Arithmetic Trichotomy).** Let $\alpha \in \overline{\mathbb{Q}}$ be an algebraic number with $h(\alpha) < \infty$. Under iteration of Galois conjugation or height operations, exactly one of three outcomes occurs:

| **Outcome** | **Arithmetic Mode** | **Mechanism** |
|-------------|---------------------|---------------|
| **Global Existence** | Dispersion | Heights decay, orbit is finite |
| **Global Regularity** | Concentration | Heights bounded, orbit factors through CM/torsion |
| **Genuine Singularity** | Escape | Heights unbounded, orbit is Zariski-dense |

---

### Proof

**Step 1: Height Dichotomy via Northcott**

By the **Northcott property** [Northcott 1950], for fixed degree $d$ and height bound $B$:
$$\#\{\beta \in \overline{\mathbb{Q}} : [\mathbb{Q}(\beta):\mathbb{Q}] \leq d, \, h(\beta) \leq B\} < \infty$$

Given $\alpha$ with $[\mathbb{Q}(\alpha):\mathbb{Q}] = d$ and $h(\alpha) = B$, the Galois orbit:
$$\mathcal{O}(\alpha) = \{\sigma(\alpha) : \sigma \in \text{Gal}(\overline{\mathbb{Q}}/\mathbb{Q})\}$$

has at most $d$ elements (exactly $d$ if $\mathbb{Q}(\alpha)/\mathbb{Q}$ is Galois).

**Dichotomy:** Either:
- $\mathcal{O}(\alpha)$ is finite and heights are bounded → **Concentration**
- Heights grow without bound under some operation → **Potential singularity**

**Step 2: Profile Extraction via Arakelov Theory**

When heights concentrate (remain bounded), we extract an arithmetic "profile" using **Arakelov geometry** [Arakelov 1974, Faltings 1984].

For an algebraic point $P \in X(\overline{\mathbb{Q}})$ on a variety $X/\mathbb{Q}$, define the **Arakelov height**:
$$\hat{h}(P) = \sum_{v \text{ place}} [k_v : \mathbb{Q}_v] \cdot \log \|P\|_v$$

The height is a **sum of local contributions** (cf. Euler product for L-functions).

**Profile decomposition:** By the **equidistribution theorem** [Szpiro-Ullmo-Zhang 1997]:
$$\frac{1}{\#\mathcal{O}(\alpha)} \sum_{\sigma} \delta_{\sigma(\alpha)} \to \mu_{\text{canonical}}$$

as $[\mathbb{Q}(\alpha):\mathbb{Q}] \to \infty$ with $h(\alpha) \to 0$. The limiting measure $\mu_{\text{canonical}}$ is the canonical height measure on $X$.

**Step 3: Classification of Limit Profiles**

**(a) Mode D.D (Dispersion/Global Existence):**

If $h(\alpha_n) \to 0$ for a sequence $\alpha_n$, then by **Lehmer's conjecture** bounds [Dobrowolski 1979]:
$$h(\alpha) \geq \frac{c}{d} \left(\frac{\log\log d}{\log d}\right)^3$$

for $\alpha$ not a root of unity. The limit $h \to 0$ forces:
- Either $\alpha_n \to$ roots of unity (torsion)
- Or $d \to \infty$ (degree escapes)

In the torsion case, $\alpha^n = 1$ for some $n$, and the orbit is finite. This is **global existence** in arithmetic terms.

**(b) Mode C.D/S.E (Concentration/Global Regularity):**

If heights concentrate at a positive value $h(\alpha) = h_0 > 0$ but orbits remain finite, then by **Zhang's theorem** [Zhang 1998]:

The sequence $\alpha_n$ with $h(\alpha_n) \to h_0 > 0$ equidistributes to the canonical measure. The limiting object is:
- A **CM point** (complex multiplication), or
- A **special subvariety** (Shimura variety)

These satisfy all arithmetic "permits" (bounded height, finite orbits, algebraic cycles).

**(c) Mode C.E (Genuine Singularity):**

If $h(\alpha_n) \to \infty$, we have "arithmetic blow-up":
- **Lehmer's problem:** Points with small height are constrained
- **Bogomolov's conjecture** [Ullmo 1998]: Non-torsion points on subvarieties have positive height

A sequence with $h \to \infty$ has **Zariski-dense orbit** and represents a genuine arithmetic singularity—no finite arithmetic structure captures it.

**Step 4: Trichotomy Verification**

The three cases are mutually exclusive and exhaustive:

1. $\lim h = 0$ → torsion or root of unity → **D.D**
2. $0 < \lim h < \infty$ → CM/special point → **Regularity**
3. $\lim h = \infty$ → Zariski-dense orbit → **Singularity**

By **Faltings' theorem** (Mordell conjecture) [Faltings 1983], curves of genus $\geq 2$ have finitely many rational points. This forces arithmetic trajectories into one of the three modes.

---

### Key Arithmetic Ingredients

1. **Northcott's Theorem** [Northcott 1950]: Finiteness of bounded height algebraic numbers.

2. **Arakelov Geometry** [Arakelov 1974, Faltings 1984]: Height theory on arithmetic varieties.

3. **Equidistribution** [Szpiro-Ullmo-Zhang 1997]: Galois orbits equidistribute to canonical measures.

4. **Bogomolov Conjecture** [Ullmo 1998, Zhang 1998]: Positive height lower bounds on subvarieties.

5. **Faltings' Theorem** [Faltings 1983]: Finiteness of rational points on curves of genus $\geq 2$.

---

### Arithmetic Interpretation

> **Every algebraic number, under Galois dynamics, either disperses to torsion, concentrates at a special (CM) structure, or escapes to arithmetic infinity.**

The trichotomy provides:
- **Complete classification:** No fourth case exists
- **Decidability:** Each case is detectable by height computations
- **Structural insight:** "Regularity" = special arithmetic structure (CM, Shimura)

---

### Literature

- [Northcott 1950] D.G. Northcott, *An inequality in the theory of arithmetic on algebraic varieties*
- [Faltings 1983] G. Faltings, *Endlichkeitssätze für abelsche Varietäten über Zahlkörpern*, Invent. Math.
- [Szpiro-Ullmo-Zhang 1997] L. Szpiro, E. Ullmo, S.-W. Zhang, *Équirépartition des petits points*, Invent. Math.
- [Zhang 1998] S.-W. Zhang, *Equidistribution of small points on abelian varieties*, Ann. Math.
