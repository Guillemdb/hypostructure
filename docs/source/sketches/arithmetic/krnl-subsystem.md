# KRNL-Subsystem: Subsystem Inheritance

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-krnl-subsystem*

Subsystems inherit regularity from containing systems.

---

## Arithmetic Formulation

### Setup

"Subsystem inheritance" in arithmetic means:
- Subvarieties inherit properties from ambient variety
- Subgroups inherit structure from containing group
- Restriction of Galois representation preserves properties

### Statement (Arithmetic Version)

**Theorem (Arithmetic Subsystem Inheritance).** Regularity passes to subsystems:

1. **Subvariety:** $Y \subset X$ inherits good reduction from $X$
2. **Subgroup:** $B \subset A$ (abelian subvariety) inherits finiteness from $A$
3. **Galois restriction:** $\rho|_H$ inherits properties from $\rho$

---

### Proof

**Step 1: Subvariety Inheritance**

For closed immersion $Y \hookrightarrow X$ over $\mathcal{O}_K$:

**Good reduction inheritance:**
$$X \text{ has good reduction at } \mathfrak{p} \Rightarrow Y \text{ has good reduction at } \mathfrak{p}$$

**Proof:**
- $X$ good at $\mathfrak{p}$ means smooth special fiber $X_\mathfrak{p}$
- $Y_\mathfrak{p} \hookrightarrow X_\mathfrak{p}$ is closed immersion
- Smoothness is inherited by regular embeddings

**Caveat:** $Y$ may have worse reduction than $X$ at some primes.

**Step 2: Abelian Subvariety Inheritance**

For $B \hookrightarrow A$ (abelian subvariety):

**Mordell-Weil inheritance:**
$$A(K) \text{ finite} \Rightarrow B(K) \text{ finite}$$

**Proof:** $B(K) \subset A(K)$ is a subgroup of a finitely generated group.

**Rank inheritance:**
$$\text{rank } B(K) \leq \text{rank } A(K)$$

**Height inheritance:**
$$\hat{h}_A|_B = c \cdot \hat{h}_B$$

for some positive constant $c$ [Faltings height comparison].

**Step 3: Galois Restriction Inheritance**

For $H \subset G_K$ (subgroup, e.g., $G_L$ for $L/K$):

**Representation restriction:** $\rho|_H: H \to \text{GL}(V)$

**Property inheritance:**
- $\rho$ unramified at $\mathfrak{p}$ ⇒ $\rho|_H$ unramified at primes above $\mathfrak{p}$
- $\rho$ crystalline ⇒ $\rho|_H$ crystalline
- $\rho$ semistable ⇒ $\rho|_H$ semistable

**Proof:** Ramification and crystallinity are detected on inertia, which inherits.

**Step 4: L-function Inheritance**

For $Y \subset X$:

**L-function relation:**
$$L(Y, s) \text{ divides } L(X, s) \text{ (in suitable sense)}$$

**Precise statement:** Cohomology inclusion:
$$H^i(Y) \hookrightarrow H^i(X) \oplus H^{i-2}(X)(-1) \oplus \cdots$$

gives L-function factorization.

**Step 5: Conductor Inheritance**

For abelian subvariety $B \subset A$:

**Conductor relation:**
$$N_B | N_A^{\dim B}$$

**Proof:** Bad reduction of $B$ only at primes of bad reduction of $A$.

More precisely: $f_\mathfrak{p}(B) \leq f_\mathfrak{p}(A)$ for conductor exponents.

**Step 6: Inheritance Certificate**

The inheritance certificate:
$$K_{\text{Inh}}^+ = (Y \subset X, \text{property}, \text{inheritance proof})$$

**Components:**
- **Subsystem:** $Y \hookrightarrow X$ or $B \hookrightarrow A$
- **Property:** Good reduction, finiteness, regularity
- **Proof:** How property descends to subsystem

---

### Key Arithmetic Ingredients

1. **Regular Embedding** [EGA IV]: Smooth subvarieties of smooth varieties.
2. **Mordell-Weil** [Weil 1928]: Subgroup of f.g. is f.g.
3. **Galois Restriction** [Serre 1979]: Ramification inheritance.
4. **Faltings Comparison** [Faltings 1983]: Height comparison for subvarieties.

---

### Arithmetic Interpretation

> **Arithmetic regularity descends to subsystems. Subvarieties inherit good reduction, abelian subvarieties inherit finite generation, Galois restrictions inherit crystallinity. The subsystem can only be as bad as the containing system, and often is better.**

---

### Literature

- [Grothendieck 1966] A. Grothendieck, *EGA IV*
- [Weil 1928] A. Weil, *L'arithmétique sur les courbes algébriques*
- [Serre 1979] J.-P. Serre, *Local Fields*
- [Faltings 1983] G. Faltings, *Endlichkeitssätze für abelsche Varietäten*
