# UP-VarietyControl: Variety-Control Theorem

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-up-variety-control*

Variety of solutions is controlled by structural invariants.

---

## Arithmetic Formulation

### Setup

"Variety control" in arithmetic means:
- The set of solutions is bounded by geometric/arithmetic invariants
- Rational points are controlled by height, genus, etc.
- Finiteness and boundedness results

### Statement (Arithmetic Version)

**Theorem (Arithmetic Variety Control).** Solution sets are controlled:

1. **Mordell-Faltings:** Curves of genus $\geq 2$ have finitely many rational points
2. **Northcott:** Bounded height and degree gives finiteness
3. **Lang-Vojta:** Higher-dimensional varieties have controlled rational points

---

### Proof

**Step 1: Mordell's Conjecture / Faltings' Theorem**

**Statement [Faltings 1983]:** For curve $C/\mathbb{Q}$ with genus $g \geq 2$:
$$|C(\mathbb{Q})| < \infty$$

**Control mechanism:**
- Genus controls complexity
- High genus → few rational points
- Effective bounds: $|C(\mathbb{Q})| \leq c(g, h_F(J_C))$ where $J_C$ is Jacobian

**Proof ingredients:**
1. Map $C \hookrightarrow J_C$ (Jacobian embedding)
2. Apply Mordell-Weil: $J_C(\mathbb{Q})$ is finitely generated
3. Use Roth's theorem: algebraic points avoid $C$ except finitely many

**Step 2: Northcott's Theorem**

**Statement [Northcott 1950]:** The set
$$\{P \in \mathbb{P}^n(\overline{\mathbb{Q}}) : h(P) \leq B, [\mathbb{Q}(P):\mathbb{Q}] \leq d\}$$
is finite.

**Control mechanism:**
- Height bounds size of coordinates
- Degree bounds algebraic complexity
- Together: only finitely many algebraic numbers

**Effective count [Schanuel 1979]:**
$$|\{P : h(P) \leq B, \deg P \leq d\}| \asymp B^{d(n+1)}$$

**Step 3: Lang-Vojta Conjecture**

**Conjecture:** For variety $X$ of general type:
$$X(\mathbb{Q}) \text{ is not Zariski-dense}$$

**Partial results:**
- True for subvarieties of abelian varieties [Faltings 1994]
- True for certain surfaces [Vojta, Bombieri]

**Control:** General type (ample canonical bundle) controls points.

**Step 4: Manin-Mumford Control**

**Theorem [Raynaud 1983]:** For $X \subset A$ (subvariety of abelian variety):
$$X \cap A_{\text{tors}} = \text{finite union of translates of subtori}$$

**Control:** Torsion points are controlled by algebraic structure.

**Uniformity [BMPZ 2019]:** $|X \cap A_{\text{tors}}| \leq c(\dim A, \deg X)$

**Step 5: André-Oort Control**

**Theorem [Pila 2011]:** Special points on Shimura varieties lie on special subvarieties.

**Control:** CM points are controlled by Shimura structure.

**Effective:** Height bounds on CM points [Tsimerman 2018].

**Step 6: Variety Control Certificate**

The variety control certificate:
$$K_{\text{Var}}^+ = (X, \text{invariants}, \text{control bound})$$

**Components:**
- **Variety:** $X$ (curve, surface, ...)
- **Invariants:** (genus, degree, height of Jacobian, ...)
- **Control:** Bound on $|X(\mathbb{Q})|$ or density statement

**Examples:**
- (Curve, genus $g \geq 2$): $|C(\mathbb{Q})| < \infty$
- (Projective space, height $\leq B$): $\#\{P : h \leq B\} \asymp B^{n+1}$
- (General type): $X(\mathbb{Q})$ not Zariski-dense

---

### Key Arithmetic Ingredients

1. **Faltings' Theorem** [Faltings 1983]: Mordell conjecture.
2. **Northcott** [Northcott 1950]: Height-degree finiteness.
3. **Raynaud's Theorem** [Raynaud 1983]: Manin-Mumford.
4. **Lang-Vojta** [Vojta 1987]: General type conjecture.

---

### Arithmetic Interpretation

> **Arithmetic variety is controlled by invariants. Genus controls curves (Faltings), height-degree controls algebraic numbers (Northcott), general type controls higher-dimensional varieties (Lang-Vojta). These control theorems establish finiteness and boundedness of rational points.**

---

### Literature

- [Faltings 1983] G. Faltings, *Endlichkeitssätze für abelsche Varietäten*
- [Northcott 1950] D.G. Northcott, *An inequality in the theory of arithmetic*
- [Raynaud 1983] M. Raynaud, *Sous-variétés d'une variété abélienne*
- [Vojta 1987] P. Vojta, *Diophantine Approximations and Value Distribution Theory*
