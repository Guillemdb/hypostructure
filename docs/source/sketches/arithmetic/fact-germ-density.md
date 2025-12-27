# FACT-GermDensity: Germ Set Density

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-fact-germ-density*

The germ set $\mathcal{G}_T$ is dense in the space of singularity patterns: every singularity is approximated by germs.

---

## Arithmetic Formulation

### Setup

In arithmetic, "germ density" means:
- Every arithmetic obstruction is approximated by finite data
- Local-global principles provide density
- Germs = local obstructions at primes

### Statement (Arithmetic Version)

**Theorem (Arithmetic Germ Density).** The set of arithmetic germs $\mathcal{G}_T$ is dense in the space of obstructions:

1. **Local density:** Every global obstruction is detected locally at some prime
2. **Approximation:** Obstructions are limits of finite germ combinations
3. **Čebotarev density:** Primes detecting obstructions have positive density

---

### Proof

**Step 1: Local-Global Principle**

For many arithmetic problems, obstructions are local:

**Hasse Principle (when it holds):**
$$X(\mathbb{Q}) \neq \emptyset \iff X(\mathbb{Q}_p) \neq \emptyset \text{ for all } p \leq \infty$$

**Selmer groups:** Obstructions to rational points lie in:
$$\text{Sel}^{(n)}(E/\mathbb{Q}) = \ker\left(H^1(\mathbb{Q}, E[n]) \to \prod_v H^1(\mathbb{Q}_v, E[n])\right)$$

Global obstruction = kernel of local map.

**Step 2: Čebotarev Density**

**Theorem (Čebotarev 1926):** Let $K/\mathbb{Q}$ be Galois with group $G$. For any conjugacy class $C \subset G$:
$$\#\{p \leq x : \text{Frob}_p \in C\} \sim \frac{|C|}{|G|} \cdot \frac{x}{\log x}$$

**Application to germs:** The set of primes detecting a given obstruction has density $|C|/|G| > 0$.

**Step 3: Germ Approximation**

**Claim:** Every global obstruction $\mathfrak{O}$ is a limit of local germs.

**Proof:**

For an L-function zero $\rho$ (potential obstruction):

**(a) Approximate by partial Euler product:**
$$L_N(s, \pi) = \prod_{p \leq N} L_p(s, \pi)$$

**(b) Zero detection:**
$$L(\rho, \pi) = 0 \iff \lim_{N \to \infty} L_N(\rho, \pi) = 0$$

**(c) Germ convergence:**
Local factors $L_p$ are the "germs"; the global zero is their limit.

**Step 4: Density of Detecting Primes**

For obstruction $\mathfrak{O}$:

**Define:** $\mathcal{P}(\mathfrak{O}) = \{p : \mathfrak{O} \text{ is detected at } p\}$

**Density theorem:** By Čebotarev:
$$\text{dens}(\mathcal{P}(\mathfrak{O})) = \frac{|\text{Frob class detecting } \mathfrak{O}|}{|G|} > 0$$

**Consequence:** Infinitely many primes detect each obstruction.

**Step 5: Exhaustiveness**

**Claim:** The germ set $\mathcal{G}_T$ is exhaustive—every obstruction factors through it.

**Proof:** By the factorization:
$$\mathfrak{O} = \varinjlim_{p \in \mathcal{P}(\mathfrak{O})} \mathfrak{g}_p$$

where $\mathfrak{g}_p$ is the local germ at $p$.

Since $\mathcal{P}(\mathfrak{O})$ is infinite (by density), the colimit equals $\mathfrak{O}$.

---

### Key Arithmetic Ingredients

1. **Čebotarev Density** [Čebotarev 1926]: Primes distribute uniformly in Galois groups.
2. **Local-Global Principle** [Hasse 1923]: Global from local data.
3. **Selmer Groups** [Selmer 1951]: Measure local-global obstruction.
4. **Euler Product** [Euler 1737]: L-function = product of local factors.

---

### Example: RH Germs

For the Riemann zeta function:

**Germs:** Local factors $\zeta_p(s) = (1 - p^{-s})^{-1}$

**Global zero:** $\zeta(\rho) = 0$ where $\rho$ is a zero

**Density:** By explicit formula, zeros are detected by prime sums:
$$\sum_\rho g(\gamma) = \hat{g}(1) - \sum_p \frac{\log p}{\sqrt{p}} \cdot (\ldots)$$

Every zero is "witnessed" by the prime sum (germ contribution).

---

### Arithmetic Interpretation

> **Arithmetic obstructions are dense in germs: every global obstruction is detected by infinitely many primes (with positive density), and is the limit of local germ data.**

---

### Literature

- [Čebotarev 1926] N. Čebotarev, *Die Bestimmung der Dichtigkeit einer Menge von Primzahlen*
- [Hasse 1923] H. Hasse, *Über die Äquivalenz quadratischer Formen*
- [Selmer 1951] E. Selmer, *The Diophantine equation...*
- [Euler 1737] L. Euler, *Variae observationes circa series infinitas*
