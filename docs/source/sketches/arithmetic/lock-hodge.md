# LOCK-Hodge: Monodromy-Weight Lock

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-lock-hodge*

The monodromy-weight filtration locks Hodge structures: monodromy type constrains possible Hodge numbers.

---

## Arithmetic Formulation

### Setup

"Monodromy-weight lock" in arithmetic means:
- **Monodromy:** Action of $\pi_1$ on cohomology near degeneration
- **Weight filtration:** Filtration from nilpotent monodromy
- **Lock:** Monodromy determines limiting Hodge structure

### Statement (Arithmetic Version)

**Theorem (Arithmetic Monodromy-Weight Lock).** For degeneration of varieties:

1. **Monodromy:** $T: H^n(X_t, \mathbb{Z}) \to H^n(X_t, \mathbb{Z})$ quasi-unipotent
2. **Weight filtration:** $W_\bullet$ determined by $N = \log(T^{\text{unip}})$
3. **Lock:** Monodromy type $\Leftrightarrow$ limiting Hodge data

---

### Proof

**Step 1: Degeneration Setup**

**Family:** $f: \mathcal{X} \to \Delta$ smooth over $\Delta^* = \Delta \setminus \{0\}$.

**Monodromy:** Loop around $0$ gives:
$$T: H^n(X_t, \mathbb{Z}) \to H^n(X_t, \mathbb{Z})$$

**Quasi-unipotence [Landman]:** $(T^k)^{n+1} = \text{Id}$ for some $k$.

**Step 2: Nilpotent Logarithm**

**Unipotent part:** $T^{\text{unip}} = T^k$ is unipotent.

**Logarithm:** $N = \log(T^{\text{unip}}) = \sum_{j=1}^n \frac{(-1)^{j+1}}{j}(T^{\text{unip}} - I)^j$

**Nilpotent:** $N^{n+1} = 0$.

**Step 3: Weight Filtration**

**Definition [Deligne]:** Unique filtration $W_\bullet$ such that:
- $N(W_i) \subset W_{i-2}$
- $N^k: \text{Gr}^W_{n+k} \xrightarrow{\sim} \text{Gr}^W_{n-k}$ for $k > 0$

**Monodromy-weight conjecture:** $W_\bullet$ is the monodromy weight filtration centered at $n$.

**Reference:** [Schmid 1973]

**Step 4: Limiting Mixed Hodge Structure**

**Theorem [Schmid]:** There exists limiting mixed Hodge structure on $H^n_{\text{lim}}$:
- Weight filtration $W_\bullet$ from monodromy
- Hodge filtration $F^\bullet$ from limit of varying filtration

**Nilpotent orbit:**
$$e^{-zN} \cdot F(t) \to F_\infty \quad \text{as } t \to 0$$

**Step 5: Lock Mechanism**

**Observation:** Monodromy type determines:
1. Jordan blocks of $N$ $\Rightarrow$ weight filtration $W_\bullet$
2. Limiting Hodge numbers:
$$h^{p,q}_{\text{lim}} = \dim \text{Gr}^p_F \text{Gr}^{p+q}_W H^n_{\text{lim}}$$

**Lock:** Fixed monodromy $\Rightarrow$ locked limiting Hodge data.

**Step 6: Arithmetic Application**

**Elliptic curves:** For $E_t \to \Delta$ degenerating to nodal curve:
- Monodromy: $T = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$ on $H^1$
- Weight: $W_0 = \langle \text{vanishing cycle} \rangle$, $W_2 = H^1$

**L-function:** Conductor at bad prime reflects monodromy type.

---

### Key Arithmetic Ingredients

1. **Monodromy Theorem** [Landman 1973]: Quasi-unipotence.
2. **Schmid's Theorem** [Schmid 1973]: Limiting MHS.
3. **Deligne's Weight Filtration** [Deligne 1971]: MWC construction.
4. **SL_2-Orbit Theorem** [Cattani-Kaplan-Schmid]: Asymptotic behavior.

---

### Arithmetic Interpretation

> **Monodromy locks Hodge structure. Near a degeneration, the monodromy action determines the weight filtration on limiting cohomology. This "locks" the possible Hodge decompositions: knowing how cycles permute under monodromy fixes the limiting Hodge numbers. In arithmetic, monodromy type at bad primes is visible in L-function conductor.**

---

### Literature

- [Schmid 1973] W. Schmid, *Variation of Hodge structure: the singularities of the period mapping*
- [Deligne 1971] P. Deligne, *Théorie de Hodge II*
- [Landman 1973] A. Landman, *On the Picard-Lefschetz transformation*
- [Cattani-Kaplan-Schmid 1986] E. Cattani, A. Kaplan, W. Schmid, *Degeneration of Hodge structures*
