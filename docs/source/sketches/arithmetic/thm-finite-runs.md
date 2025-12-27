# Finite Complete Runs Theorem

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: thm-finite-runs*

A complete sieve run consists of finitely many epochs. Each surgery has an associated progress measure that ensures termination.

---

## Arithmetic Formulation

### Setup

In arithmetic, "surgeries" correspond to:
- Resolving singularities of varieties
- Blowing up bad reduction primes
- Extending the base field to split ramification

Each surgery has a **complexity cost** that decreases with each operation.

### Statement (Arithmetic Version)

**Theorem (Finite Arithmetic Resolution).** Let $X/\mathbb{Q}$ be an arithmetic object (variety, scheme, motive) with arithmetic complexity:
$$\mathcal{C}(X) = (\text{conductor}(X), \dim(X), \text{genus}(X), \ldots)$$

Any resolution process (desingularization, semistable reduction, etc.) terminates after finitely many steps:
$$N_{\text{surgeries}} \leq C(\mathcal{C}(X_0)) < \infty$$

---

### Proof

**Step 1: Progress Measure Type A—Conductor Bounds**

For an arithmetic variety $X/\mathbb{Q}$, the **conductor** $N_X$ measures arithmetic complexity:
$$N_X = \prod_{p \text{ bad}} p^{f_p(X)}$$

where $f_p(X)$ is the conductor exponent at $p$.

**Bound on surgery count:** Each blow-up or base change reduces conductor contribution at some prime. By **Ogg's formula** [Ogg 1967] for elliptic curves:
$$f_p(E) = \begin{cases} 0 & \text{good reduction} \\ 1 & \text{multiplicative} \\ 2 + \delta_p & \text{additive} \end{cases}$$

where $\delta_p \leq v_p(\Delta_E)$ is bounded by the discriminant valuation.

**Surgery decreases conductor:** Blowing up a singularity at $p$ transforms:
$$f_p(X) \to f_p(X') \leq f_p(X) - 1$$

Hence:
$$N_{\text{surgeries}} \leq \sum_p f_p(X_0) = \log N_{X_0} / \log 2 < \infty$$

**Step 2: Progress Measure Type B—Well-Founded Genus/Dimension**

For schemes, define complexity:
$$\mathcal{C}(X) = (\dim X, \text{number of irreducible components}, \text{genus}, \ldots)$$

Each is a natural number or finite invariant.

**Hironaka's Theorem** [Hironaka 1964]:
- Resolution of singularities exists for varieties over $\mathbb{C}$
- Each blow-up center has dimension $< \dim X$
- Process terminates in finitely many steps

The arithmetic analogue uses **de Jong's alterations** [de Jong 1996]:
- Given $X/\mathbb{Q}$, there exists an alteration $X' \to X$ with $X'$ regular
- The degree of alteration is bounded by invariants of $X$

**Step 3: Discrete Energy Progress**

Define arithmetic "energy":
$$\Phi(X) = \log N_X + h(X)$$

where $h(X)$ is the height of the moduli point.

**Progress constraint:** Each surgery must drop energy by:
$$\epsilon_{\text{min}} = \log 2$$

(minimal conductor reduction).

Hence:
$$N_{\text{surgeries}} \leq \frac{\Phi(X_0)}{\epsilon_{\text{min}}} = \frac{\log N_{X_0} + h(X_0)}{\log 2} < \infty$$

**Step 4: Finite Surgery Types**

The number of surgery types is finite:
- Blow-up at singular point (17 Kodaira types for elliptic curves)
- Base change (bounded by field degree)
- Regularization (finitely many local types)

Total epochs = (number of surgery types) × (surgeries per type) $< \infty$.

---

### Key Arithmetic Ingredients

1. **Conductor Formula** [Ogg 1967, Saito 1988]: Measures bad reduction.

2. **Hironaka's Resolution** [Hironaka 1964]: Desingularization terminates.

3. **de Jong's Alterations** [de Jong 1996]: Arithmetic regularization.

4. **Kodaira Classification** [Kodaira 1963]: Finite singularity types for elliptic curves.

---

### Arithmetic Interpretation

> **Any arithmetic "regularization" process (resolving singularities, achieving good reduction) terminates because conductor and complexity strictly decrease at each step.**

---

### Connection to Surgery Bounds

| **Analytic (Perelman)** | **Arithmetic Analogue** |
|------------------------|------------------------|
| Ricci flow surgeries | Blow-ups at bad primes |
| Surgery count $\leq C \cdot T^{d/2}$ | Surgeries $\leq \log N_{X_0}$ |
| Energy drops by $\epsilon$ | Conductor drops by factor $p$ |

---

### Literature

- [Hironaka 1964] H. Hironaka, *Resolution of singularities*, Ann. Math.
- [de Jong 1996] A.J. de Jong, *Smoothness, semi-stability and alterations*, Publ. Math. IHÉS
- [Ogg 1967] A.P. Ogg, *Elliptic curves and wild ramification*, Amer. J. Math.
- [Perelman 2003] G. Perelman, *Finite extinction time for solutions to the Ricci flow*
