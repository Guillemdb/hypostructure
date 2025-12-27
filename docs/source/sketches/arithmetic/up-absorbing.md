# UP-Absorbing: Absorbing Boundary Promotion

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-up-absorbing*

Absorbing boundaries capture all trajectories entering them: once inside, trajectories cannot escape, creating irreversible transitions.

---

## Arithmetic Formulation

### Setup

"Absorbing boundary" in arithmetic means:
- **Boundary:** Reduction modulo prime / specialization
- **Absorbing:** Information lost upon reduction cannot be recovered
- **Irreversibility:** Torsion under reduction is one-way

### Statement (Arithmetic Version)

**Theorem (Arithmetic Absorbing Boundary).** Reduction modulo $\mathfrak{p}$ is absorbing:

1. **Reduction map:** $\tilde{}: E(\mathbb{Q}) \to \tilde{E}(\mathbb{F}_p)$ is a homomorphism
2. **Kernel absorption:** $E_1(\mathbb{Q}_p) := \ker(\tilde{})$ is "absorbed" — finite index in $E(\mathbb{Q}_p)$
3. **Irreversibility:** Multiple points reduce to same $\tilde{P}$; lifting is not unique

---

### Proof

**Step 1: Reduction Homomorphism**

**Good reduction:** For $E/\mathbb{Q}$ with good reduction at $p$:
$$\tilde{}: E(\mathbb{Q}) \hookrightarrow E(\mathbb{Q}_p) \twoheadrightarrow \tilde{E}(\mathbb{F}_p)$$

**Homomorphism:** $\widetilde{P + Q} = \tilde{P} + \tilde{Q}$

**Reference:** [Silverman, Ch. VII]

**Step 2: Formal Group and Kernel**

**Formal group:** $E_1(\mathbb{Q}_p) = \ker(\tilde{})$

**Filtration:**
$$E_0(\mathbb{Q}_p) \supset E_1(\mathbb{Q}_p) \supset E_2(\mathbb{Q}_p) \supset \cdots$$

where $E_n(\mathbb{Q}_p) = \{P : v_p(x(P)) \leq -2n\}$.

**Absorption:** Points deep in $E_n$ are absorbed into higher filtration levels.

**Step 3: Irreversibility**

**Many-to-one:** The reduction map is surjective but not injective on $E(\mathbb{Q}_p)$:
$$\#\ker(\tilde{}) = [E(\mathbb{Q}_p) : E_0(\mathbb{Q}_p)] \cdot \#E_1(\mathbb{Q}_p)$$

**Lifting ambiguity:** Given $\tilde{P} \in \tilde{E}(\mathbb{F}_p)$, lifts form coset of $E_1$.

**Step 4: Torsion Absorption**

**Theorem [reduction of torsion]:** For $P \in E(\mathbb{Q})_{\text{tors}}$:
$$\tilde{}: E(\mathbb{Q})_{\text{tors}} \hookrightarrow \tilde{E}(\mathbb{F}_p)$$

is injective for $p$ of good reduction, $p \nmid \#E(\mathbb{Q})_{\text{tors}}$.

**Absorption boundary:** Bad primes "absorb" torsion structure.

**Step 5: Selmer Absorption**

**Selmer group:** Local conditions absorbed:
$$\text{Sel}^n(E/\mathbb{Q}) = \ker\left(H^1(\mathbb{Q}, E[n]) \to \prod_v \frac{H^1(\mathbb{Q}_v, E[n])}{E(\mathbb{Q}_v)/nE(\mathbb{Q}_v)}\right)$$

**Local absorption:** Local image $E(\mathbb{Q}_v)/nE(\mathbb{Q}_v)$ absorbs part of $H^1$.

**Step 6: Absorbing Certificate**

The absorbing boundary certificate:
$$K_{\text{Absorb}}^+ = (\text{reduction map}, \ker(\tilde{}), \text{formal group filtration})$$

---

### Key Arithmetic Ingredients

1. **Reduction Theory** [Silverman Ch. VII]: Good/bad reduction.
2. **Formal Groups** [Lubin-Tate]: $p$-adic structure of kernel.
3. **Torsion Injection** [Silverman]: Torsion embeds under good reduction.
4. **Selmer Groups** [Cassels]: Local-global descent.

---

### Arithmetic Interpretation

> **Reduction modulo $p$ is an absorbing boundary: information flows from $\mathbb{Q}$ to $\mathbb{F}_p$ but not back. The kernel of reduction (formal group) absorbs points that "fall into" the boundary. This irreversibility is the arithmetic analogue of absorbing boundary conditions in dynamics.**

---

### Literature

- [Silverman 2009] J. Silverman, *The Arithmetic of Elliptic Curves*, GTM 106
- [Lubin-Tate 1965] J. Lubin, J. Tate, *Formal complex multiplication*
- [Cassels 1991] J.W.S. Cassels, *Lectures on Elliptic Curves*, LMS
