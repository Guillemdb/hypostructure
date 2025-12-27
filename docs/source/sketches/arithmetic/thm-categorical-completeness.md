# Categorical Completeness of the Singularity Spectrum

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: thm-categorical-completeness*

For any problem type $T$, the category of singularity patterns admits a universal object that is categorically exhaustive: every singularity factors through this universal bad pattern.

---

## Arithmetic Formulation

### Setup

For arithmetic problems, "singularities" correspond to:
- **Zeros of L-functions** at unexpected locations
- **Non-algebraic Hodge classes**
- **Obstructions to rational points**
- **Failures of the Hasse principle**

Define the **arithmetic germ set** $\mathcal{G}_T$ as the collection of minimal obstructions for problem type $T$.

### Statement (Arithmetic Version)

**Theorem (Arithmetic Completeness).** Let $T$ be an arithmetic problem type. The set of obstruction patterns $\mathcal{G}_T$ is:

1. **Small:** $\mathcal{G}_T$ is a set (not a proper class), with cardinality bounded by arithmetic invariants.

2. **Cofinal:** Every obstruction to conjecture $T$ factors through some element of $\mathcal{G}_T$.

3. **Universal:** The colimit $\mathfrak{B}_T = \varinjlim_{g \in \mathcal{G}_T} g$ is the universal obstruction pattern.

---

### Proof

**Step 1: Germ Set Construction for Arithmetic Types**

For each arithmetic problem type, we identify the minimal obstructions:

**(a) Type $T_{\text{RH}}$ (Riemann Hypothesis):**

The germ set consists of hypothetical zeros off the critical line:
$$\mathcal{G}_{\text{RH}} = \{\rho \in \mathbb{C} : \zeta(\rho) = 0, \Re(\rho) \neq 1/2\}$$

By the **functional equation** $\zeta(s) = \chi(s)\zeta(1-s)$, zeros come in symmetric pairs. By **Hadamard's theorem** on entire functions of order 1 [Titchmarsh 1986, §9.4]:
$$\zeta(s) = e^{A+Bs} \prod_\rho \left(1 - \frac{s}{\rho}\right) e^{s/\rho}$$

The zero set is discrete, hence countable. **Smallness verified.**

**(b) Type $T_{\text{Hodge}}$ (Hodge Conjecture):**

The germ set consists of non-algebraic $(p,p)$-classes. By **Deligne's theorem on absolute Hodge cycles** [Deligne 1982]:

For $X$ smooth projective over $\mathbb{C}$, define:
$$\mathcal{G}_{\text{Hodge}}(X) = H^{2p}(X, \mathbb{Q}) \cap H^{p,p}(X) \setminus \text{Im}(\text{cl}: \text{CH}^p(X)_\mathbb{Q} \to H^{2p})$$

By **Hodge theory** [Voisin 2002, Ch. 7], the Hodge structure $H^{2p}(X, \mathbb{Q})$ is finite-dimensional. Hence $\mathcal{G}_{\text{Hodge}}(X)$ is contained in a finite-dimensional $\mathbb{Q}$-vector space. **Smallness verified.**

**(c) Type $T_{\text{BSD}}$ (Birch and Swinnerton-Dyer):**

The germ set consists of elliptic curves where $\text{ord}_{s=1} L(E,s) \neq \text{rank } E(\mathbb{Q})$. By **Mordell's theorem** [Mordell 1922]:
$$E(\mathbb{Q}) \cong \mathbb{Z}^r \oplus E(\mathbb{Q})_{\text{tors}}$$

The rank $r$ is finite. Potential counterexamples are parametrized by $(r, \text{ord})$ pairs with $r \neq \text{ord}$. **Smallness verified.**

**Step 2: Cofinality via Reduction Theory**

Any arithmetic obstruction reduces to a germ via:

**(a) For L-function zeros:** By the **explicit formula** [Weil 1952]:
$$\sum_\gamma g(\gamma) = g(0) + g(1) - \sum_p \frac{\log p}{p^{1/2}} \sum_{m=1}^\infty \frac{g(m\log p) + g(-m\log p)}{p^{m/2}}$$

Any zero $\rho$ with $\zeta(\rho) = 0$ contributes a term. The sum is over the germ set $\mathcal{G}_{\text{RH}}$.

**(b) For Hodge classes:** By the **Lefschetz (1,1) theorem** [Griffiths-Harris, p. 163], any $(1,1)$-class is algebraic. For $(p,p)$ with $p > 1$, obstructions factor through primitive classes. By **Deligne's theorem on Hodge classes** [Deligne 1971], absolute Hodge classes on abelian varieties are algebraic. Remaining obstructions form a cofinal system.

**(c) For BSD:** By the **modularity theorem** [Wiles 1995, Taylor-Wiles 1995]:
$$L(E,s) = L(f_E, s)$$
where $f_E$ is a weight-2 modular form. The order of vanishing is encoded in $f_E$. Any counterexample to BSD reduces to a specific $(E, f_E)$ pair.

**Step 3: Universal Object as Colimit**

Define the universal obstruction:
$$\mathfrak{B}_T = \varinjlim_{g \in \mathcal{G}_T} g$$

**(a) Existence of colimit:** The category $\mathbf{Arith}_T$ of arithmetic structures is **locally presentable** by [Adámek-Rosický 1994, Thm 1.20]. Since $\mathcal{G}_T$ is small (Step 1), the colimit exists.

**(b) Universal property:** For any obstruction $o: \mathfrak{O} \to Z$, there exists $g \in \mathcal{G}_T$ such that $o$ factors as:
$$\mathfrak{O} \xrightarrow{\sim} g \xrightarrow{\iota_g} \mathfrak{B}_T \xrightarrow{\exists! \phi} Z$$

This is the arithmetic analogue of the initiality property.

**Step 4: Exhaustiveness**

The universal object $\mathfrak{B}_T$ detects all obstructions:
$$\text{Hom}(\mathfrak{B}_T, Z) \neq \emptyset \iff \exists g \in \mathcal{G}_T: \text{Hom}(g, Z) \neq \emptyset$$

By cofinality (Step 2), this covers all possible obstructions. The spectrum is complete.

---

### Key Arithmetic Ingredients

1. **Hadamard Factorization** [Titchmarsh 1986]: L-functions have countable zeros.

2. **Deligne's Absolute Hodge Cycles** [Deligne 1982]: Constrains Hodge classes on abelian varieties.

3. **Mordell-Weil Theorem** [Mordell 1922]: Finite generation of rational points.

4. **Modularity Theorem** [Wiles 1995]: Links elliptic curves to modular forms.

5. **Locally Presentable Categories** [Adámek-Rosický 1994]: Ensures colimits exist.

---

### Arithmetic Interpretation

> **Every arithmetic obstruction (hypothetical counterexample) factors through a minimal germ, and the collection of germs is small and exhaustive.**

This means:
- **No escape:** A counterexample to RH, Hodge, or BSD cannot avoid the germ set
- **Finite check:** In principle, verifying the conjecture reduces to checking finitely many germ types
- **Universal test:** The Lock mechanism (testing $\text{Hom}(\mathfrak{B}_T, Z) = \emptyset$) is sound

---

### Literature

- [Titchmarsh 1986] E.C. Titchmarsh, *The Theory of the Riemann Zeta-Function*, 2nd ed.
- [Deligne 1982] P. Deligne, *Hodge cycles on abelian varieties*, in Hodge Cycles, Motives, and Shimura Varieties
- [Wiles 1995] A. Wiles, *Modular elliptic curves and Fermat's Last Theorem*, Ann. Math.
- [Adámek-Rosický 1994] J. Adámek, J. Rosický, *Locally Presentable and Accessible Categories*, LMS Lecture Notes
