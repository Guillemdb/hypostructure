# UP-Lockback: Lock-Back Theorem

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-up-lockback*

Lock structures feed back to strengthen prior permits.

---

## Arithmetic Formulation

### Setup

"Lock-back" in arithmetic means:
- Obstruction information feeds back to improve bounds
- Galois lock provides information about structure
- Non-existence proofs strengthen existence bounds elsewhere

### Statement (Arithmetic Version)

**Theorem (Arithmetic Lock-Back).** Locks provide feedback:

1. **Galois lock-back:** No rational points → refined Galois structure
2. **Brauer lock-back:** Brauer obstruction → control of other varieties
3. **L-function lock-back:** Zero location → arithmetic information

---

### Proof

**Step 1: Galois Information from Non-Existence**

**Setup:** $X$ is a variety with $X(\mathbb{Q}) = \emptyset$.

**Lock-back principle:** The proof of emptiness reveals Galois structure.

**Example (Selmer curves):**
- $X: 3x^3 + 4y^3 + 5z^3 = 0$ has no rational points
- Proof uses 3-descent and Galois cohomology
- Lock-back: $H^1(G_\mathbb{Q}, E[3])$ structure is determined

**Information gained:**
$$X(\mathbb{Q}) = \emptyset \Rightarrow \text{Ш}(E)[3] \neq 0$$

**Step 2: Brauer Lock-Back**

**Setup:** $X$ with Brauer-Manin obstruction.

**Lock-back to other varieties:**
- If $X \to Y$ is a morphism and $X$ has Brauer obstruction
- Then $Y$ has information about its Brauer group

**Fibration lock-back [Colliot-Thélène]:**
For $\pi: X \to C$ (fibration over curve):
- Brauer obstruction on fiber $X_c$ for all $c \in C(\mathbb{Q})$
- Implies: $X(\mathbb{Q}) = \emptyset$ or all fibers have obstruction

**Step 3: L-function Zero Lock-Back**

**Setup:** Zero of $L(E, s)$ at $s = 1$.

**Lock-back to arithmetic:**
$$L(E, 1) = 0 \Rightarrow \text{rank } E(\mathbb{Q}) \geq 1 \text{ (under BSD)}$$

**Refined lock-back [Gross-Zagier]:**
$$L'(E, 1) \neq 0 \land L(E, 1) = 0 \Rightarrow \text{rank } E(\mathbb{Q}) = 1$$

**Information propagation:** The zero location determines arithmetic.

**Step 4: Descent Lock-Back**

**Setup:** $n$-descent on elliptic curve $E$.

**Lock computation:** Computing $\text{Sel}^{(n)}(E)$ reveals:
- Upper bound on rank
- Lower bound on Sha

**Lock-back:**
$$\dim \text{Sel}^{(n)}(E) = r + \dim \text{Ш}[n]$$

If $\text{Sel}^{(n)}$ is too small for rank $\geq r$, then rank $< r$ (locked out).

**Step 5: Non-Vanishing Lock-Back**

**Setup:** $L(E, 1) \neq 0$.

**Lock-back [Kolyvagin]:**
- $E(\mathbb{Q})$ is finite
- $\text{Ш}(E)$ is finite

**Quantitative lock-back:**
$$|L(E, 1)| \geq c(N) > 0 \Rightarrow |E(\mathbb{Q})| \leq B(N)$$

The non-vanishing provides explicit bounds.

**Step 6: Lock-Back Certificate**

The lock-back certificate:
$$K_{\text{LB}}^+ = (\text{lock}, \text{feedback}, \text{improved bound})$$

**Components:**
- **Lock:** What obstruction/non-existence was proven
- **Feedback:** What information this provides
- **Improved bound:** How other invariants are constrained

**Example:**
- Lock: $X(\mathbb{Q}) = \emptyset$ proven via Brauer-Manin
- Feedback: $\text{Br}(X)/\text{Br}(\mathbb{Q}) \neq 0$
- Improved bound: Related varieties have constrained Brauer groups

---

### Key Arithmetic Ingredients

1. **Descent Theory** [Cassels 1962]: Selmer groups lock rank.
2. **Brauer-Manin** [Manin 1970]: Obstruction provides structural info.
3. **Gross-Zagier** [Gross-Zagier 1986]: Zero order determines rank.
4. **Kolyvagin** [Kolyvagin 1988]: Non-vanishing implies finiteness.

---

### Arithmetic Interpretation

> **Arithmetic locks feed back information. Proving no rational points reveals Galois cohomology structure. Brauer obstructions constrain related varieties. L-function zeros determine rank. This lock-back mechanism turns negative results into positive structural information.**

---

### Literature

- [Cassels 1962] J.W.S. Cassels, *Arithmetic on curves of genus 1*
- [Manin 1970] Yu.I. Manin, *Le groupe de Brauer-Grothendieck*
- [Gross-Zagier 1986] B. Gross, D. Zagier, *Heegner points and derivatives of L-series*
- [Kolyvagin 1988] V. Kolyvagin, *Finiteness of E(Q) and Ш(E,Q)*
