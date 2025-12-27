# Closure Termination Theorem

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: thm-closure-termination*

Under the certificate finiteness condition, the promotion closure $\text{Cl}(\Gamma)$ is computable in finite time. The closure is independent of the order in which upgrade rules are applied (confluence).

---

## Arithmetic Formulation

### Setup

In arithmetic, "promotion closure" corresponds to:
- Deriving all consequences of known facts
- Applying all applicable theorems
- Reaching a fixed point of deduction

### Statement (Arithmetic Version)

**Theorem (Arithmetic Deductive Closure).** Let $\Gamma$ be a set of arithmetic certificates (proved facts). The **deductive closure**:
$$\text{Cl}(\Gamma) = \bigcup_{n=0}^\infty \Gamma_n$$

where $\Gamma_{n+1} = \Gamma_n \cup \{K : \Gamma_n \vdash K\}$, satisfies:

1. **Termination:** $\text{Cl}(\Gamma)$ is computable in finite time
2. **Confluence:** The result is independent of derivation order
3. **Completeness:** All derivable facts are included

---

### Proof

**Step 1: Certificate Lattice Structure**

Define the **lattice of arithmetic facts**:
$$(\mathcal{L}, \subseteq) = (\mathcal{P}(\mathcal{K}), \subseteq)$$

where $\mathcal{K}$ is the set of all arithmetic certificates for the given problem.

**Lattice properties:**
- **Partial order:** $\Gamma_1 \subseteq \Gamma_2$ (set inclusion)
- **Meet:** $\Gamma_1 \cap \Gamma_2$
- **Join:** $\Gamma_1 \cup \Gamma_2$
- **Bottom:** $\emptyset$
- **Top:** $\mathcal{K}$

The lattice is **complete** (all subsets have supremum).

**Step 2: Promotion Operator**

Define $F: \mathcal{L} \to \mathcal{L}$ by:
$$F(\Gamma) = \Gamma \cup \{K : \exists \text{ rule } R, \Gamma \vdash_R K\}$$

**Arithmetic derivation rules $R$:**

| **Rule** | **Premise** | **Conclusion** |
|----------|-------------|----------------|
| Height transitivity | $h(\alpha) \leq h(\beta), h(\beta) \leq B$ | $h(\alpha) \leq B$ |
| Galois containment | $G_1 \leq G_2, G_2$ solvable | $G_1$ solvable |
| Local-global | $\forall p: K_p$ (local) | $K$ (global) |
| L-function factorization | $L(M, s) = L(M_1, s) \cdot L(M_2, s)$ | Properties transfer |

**Step 3: Monotonicity Verification**

**Claim:** $F$ is monotonic: $\Gamma_1 \subseteq \Gamma_2 \Rightarrow F(\Gamma_1) \subseteq F(\Gamma_2)$.

**Proof:** If $\Gamma_1 \subseteq \Gamma_2$:
- $\Gamma_1 \subseteq \Gamma_2 \subseteq F(\Gamma_2)$ (inclusion in domain)
- If $\Gamma_1 \vdash_R K$, then $\Gamma_2 \vdash_R K$ (weakening)
- Hence $F(\Gamma_1) \subseteq F(\Gamma_2)$ $\square$

**Step 4: Knaster-Tarski Fixed Point**

By **Knaster-Tarski** [Tarski 1955]:

A monotonic operator $F$ on a complete lattice has a least fixed point:
$$\text{lfp}(F) = \bigcup_{n=0}^\infty F^n(\bot) = \bigcup_{n=0}^\infty F^n(\emptyset)$$

For our operator:
$$\text{Cl}(\Gamma) = \text{lfp}(F_\Gamma) \quad \text{where } F_\Gamma(\Delta) = \Gamma \cup F(\Delta)$$

**Step 5: Finiteness Condition**

The **certificate finiteness condition** ensures $|\mathcal{K}| < \infty$.

**For arithmetic problems:**
- Heights bounded: $h \leq B$ gives $N(B, d) < \infty$ algebraic numbers
- Degrees bounded: $d \leq D$ gives finite extensions
- Galois groups bounded: $|G| \leq d!$ gives finite group conditions

Hence $|\mathcal{K}| \leq N(B, D) \cdot |\text{Subgroups}(S_D)| \cdot (\text{finite local data}) < \infty$.

**Step 6: Termination Proof**

**Kleene iteration** [Kleene 1952]:
$$\Gamma_0 = \Gamma, \quad \Gamma_{n+1} = F(\Gamma_n)$$

Since $\Gamma_n \subseteq \Gamma_{n+1} \subseteq \mathcal{K}$ and $|\mathcal{K}| < \infty$:
$$\exists N: \Gamma_N = \Gamma_{N+1} = \text{Cl}(\Gamma)$$

Termination in at most $|\mathcal{K}|$ steps.

**Step 7: Confluence (Order Independence)**

**Claim:** $\text{Cl}(\Gamma)$ is independent of derivation order.

**Proof:** The closure is characterized as the **least fixed point**, which is unique. Different derivation orders may reach the fixed point in different numbers of steps, but the result is the same.

Formally, by the **Church-Rosser property** for the derivation system: if $\Gamma \vdash^* \Delta_1$ and $\Gamma \vdash^* \Delta_2$ are two derivation sequences, there exists $\Delta_3$ with $\Delta_1 \vdash^* \Delta_3$ and $\Delta_2 \vdash^* \Delta_3$.

Since we iterate to fixed point, both sequences reach $\text{Cl}(\Gamma)$.

---

### Key Arithmetic Ingredients

1. **Knaster-Tarski Theorem** [Tarski 1955]: Fixed points in complete lattices.

2. **Kleene Iteration** [Kleene 1952]: Constructive fixed-point computation.

3. **Church-Rosser Property** [Church-Rosser 1936]: Confluence of derivations.

4. **Finiteness of Arithmetic Data** [Northcott 1950]: Bounded height/degree gives finite sets.

---

### Arithmetic Interpretation

> **All arithmetic consequences of a set of facts can be computed in finite time. The result is independent of the order of deduction—the same conclusions are reached regardless of proof strategy.**

---

### Literature

- [Tarski 1955] A. Tarski, *A lattice-theoretical fixpoint theorem and its applications*, Pacific J. Math.
- [Kleene 1952] S.C. Kleene, *Introduction to Metamathematics*
- [Davey-Priestley 2002] B.A. Davey, H.A. Priestley, *Introduction to Lattices and Order*, 2nd ed.
