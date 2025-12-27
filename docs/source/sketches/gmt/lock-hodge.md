# LOCK-Hodge: Monodromy-Weight Lock — GMT Translation

## Original Statement (Hypostructure)

The monodromy-weight lock shows that the monodromy weight filtration on cohomology creates rigidity, locking the Hodge structure against certain degenerations.

## GMT Setting

**Monodromy:** Action of fundamental group on cohomology of fibers

**Weight Filtration:** Filtration on limiting mixed Hodge structure

**Lock:** Monodromy-weight conjecture constraints on degeneration

## GMT Statement

**Theorem (Monodromy-Weight Lock).** For degeneration of varieties:

1. **Monodromy Operator:** $N = \log(T)$ for unipotent monodromy $T$

2. **Weight Filtration:** $W_\bullet$ defined by $N$

3. **Monodromy-Weight:** Filtration centered at weight $k$ (for $k$-forms)

4. **Lock:** Hodge numbers constrained by monodromy type

## Proof Sketch

### Step 1: Variation of Hodge Structure

**Definition:** A variation of Hodge structure over $S$ is:
- Local system $\mathcal{H}_\mathbb{Z}$
- Hodge filtration $F^\bullet$ varying holomorphically
- Griffiths transversality: $\nabla F^p \subset F^{p-1} \otimes \Omega^1_S$

**Reference:** Griffiths, P. (1968). Periods of integrals on algebraic manifolds. *Bull. Amer. Math. Soc.*, 76, 228-296.

### Step 2: Monodromy Around Singularity

**Local System:** For $f: X \to \Delta^*$ smooth family over punctured disk:
$$T: H^k(X_t) \to H^k(X_t)$$

monodromy action.

**Monodromy Theorem (Griffiths-Landman):** $T$ is quasi-unipotent:
$$(T^n - I)^{k+1} = 0$$

for some $n$.

**Reference:** Schmid, W. (1973). Variation of Hodge structure: the singularities of the period mapping. *Invent. Math.*, 22, 211-319.

### Step 3: Limiting Mixed Hodge Structure

**Deligne Extension:** The Hodge bundles extend to $\Delta$.

**Limiting MHS:** On $H^k_{\lim}$, there is mixed Hodge structure:
- Weight filtration $W_\bullet$ from monodromy
- Hodge filtration $F^\bullet$ from limit

**Reference:** Deligne, P. (1971). Théorie de Hodge II. *Publ. Math. IHES*, 40, 5-57.

### Step 4: Monodromy Weight Filtration

**Definition:** For nilpotent $N$, the weight filtration $W_\bullet$ is unique satisfying:
- $N(W_i) \subset W_{i-2}$
- $N^k: \text{Gr}^W_{k+r} \xrightarrow{\sim} \text{Gr}^W_{-k+r}$ for $k > 0$

centered at weight $r$.

**Reference:** Steenbrink, J. (1976). Limits of Hodge structures. *Invent. Math.*, 31, 229-257.

### Step 5: Monodromy-Weight Conjecture

**Statement:** For smooth projective degeneration:
$$W_\bullet = M_\bullet[k]$$

where $M_\bullet$ is monodromy filtration of $N$ on $H^k$.

**Proved:** In characteristic zero by Saito, Sabbah, etc.

**Reference:** Saito, M. (1988). Modules de Hodge polarisables. *Publ. RIMS Kyoto*, 24, 849-995.

### Step 6: Hodge Number Constraints

**Limiting Hodge Numbers:** $h^{p,q}_{\lim}$ satisfy:
$$h^{p,q}_{\lim} = \dim \text{Gr}^p_F \text{Gr}^{p+q}_W H^{p+q}_{\lim}$$

**Constraints:** Monodromy type (Jordan blocks) determines which $h^{p,q}$ vanish.

### Step 7: GMT Connection

**Currents from VHS:** The Hodge bundles define currents:
$$[F^p] \in \mathbf{I}_*(B)$$

over base $B$.

**Singular Set:** At degeneration locus, singular set of current controlled by monodromy.

### Step 8: Rigidity from Monodromy

**Lock Mechanism:** If monodromy is trivial ($N = 0$):
- Weight filtration is trivial
- Hodge structure extends without mixing
- Configuration is "locked" in pure state

**Non-trivial Monodromy:** Implies weight mixing, constrains possible Hodge types.

### Step 9: Nilpotent Orbit Theorem

**Theorem (Schmid):** As $t \to 0$:
$$e^{-zN} \cdot F(t) \to F_\infty$$

where $z = \frac{1}{2\pi i}\log t$.

**Reference:** Schmid, W. (1973). Variation of Hodge structure. *Invent. Math.*, 22.

**Consequence:** Limiting Hodge filtration well-defined.

### Step 10: Compilation Theorem

**Theorem (Monodromy-Weight Lock):**

1. **Monodromy:** $N = \log(T)$ nilpotent operator

2. **Weight Filtration:** $W_\bullet$ uniquely determined by $N$

3. **MWC:** Weight = monodromy filtration shifted

4. **Lock:** Hodge structure constrained by monodromy type

**Applications:**
- Degeneration of algebraic varieties
- Mixed Hodge theory
- Constraints on moduli compactification

## Key GMT Inequalities Used

1. **Quasi-unipotence:**
   $$(T^n - I)^{k+1} = 0$$

2. **Weight Isomorphism:**
   $$N^k: \text{Gr}^W_{k+r} \xrightarrow{\sim} \text{Gr}^W_{-k+r}$$

3. **Hodge-Weight:**
   $$h^{p,q}_{\lim} = \dim \text{Gr}^p_F \text{Gr}^{p+q}_W$$

4. **Nilpotent Orbit:**
   $$e^{-zN} F(t) \to F_\infty$$

## Literature References

- Griffiths, P. (1968). Periods of integrals. *Bull. Amer. Math. Soc.*, 76.
- Schmid, W. (1973). Variation of Hodge structure. *Invent. Math.*, 22.
- Deligne, P. (1971). Théorie de Hodge II. *Publ. Math. IHES*, 40.
- Steenbrink, J. (1976). Limits of Hodge structures. *Invent. Math.*, 31.
- Saito, M. (1988). Modules de Hodge polarisables. *Publ. RIMS Kyoto*, 24.
