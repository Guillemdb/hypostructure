# ACT-Surgery-2: Structural Surgery Principle (Alternative)

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-act-surgery-2*

Alternative formulation of structural surgery for resolution.

---

## Arithmetic Formulation

### Setup

"Structural surgery (alternative)" focuses on:
- Surgery as modification of arithmetic invariants
- Surgery preserving certain structures while modifying others
- Controlled surgery with explicit bounds

### Statement (Arithmetic Version)

**Theorem (Controlled Arithmetic Surgery).** Surgery operations have controlled effects:

1. **Isogeny surgery:** Preserves L-function, modifies torsion
2. **Twist surgery:** Modifies conductor, preserves rank parity
3. **Base change surgery:** Multiplies invariants predictably

---

### Proof

**Step 1: Isogeny Surgery**

**Operation:** $\phi: E \to E'$ isogeny of degree $n$.

**Preserved:**
- L-function: $L(E, s) = L(E', s)$
- Conductor: $N_E = N_{E'}$

**Modified:**
- Torsion: $E(\mathbb{Q})_{\text{tors}} \neq E'(\mathbb{Q})_{\text{tors}}$ in general
- Regulator: $\text{Reg}_{E'} = n \cdot \text{Reg}_E$
- Sha: $|\text{Ш}(E')| = |\text{Ш}(E)| / n^2 \cdot (\text{correction})$

**Control:** BSD quotient is preserved (Cassels-Tate).

**Step 2: Quadratic Twist Surgery**

**Operation:** $E \mapsto E^{(d)}$ for squarefree $d$.

**Preserved:**
- j-invariant: $j(E^{(d)}) = j(E)$
- Rank parity: $\text{rank } E^{(d)}(\mathbb{Q}) \equiv \text{rank } E(\mathbb{Q}) \pmod{2}$ (not always)

**Modified:**
- Conductor: $N_{E^{(d)}} \approx N_E \cdot d^2 / \gcd(N_E, d)^2$
- L-function: $L(E^{(d)}, s) = L(E, s, \chi_d)$
- Root number: $w(E^{(d)}) = w(E) \cdot \chi_d(-N_E) \cdot \prod_{p|d} \ldots$

**Control:** Twist changes are computable.

**Step 3: Base Change Surgery**

**Operation:** $E \mapsto E_K$ for field extension $K/\mathbb{Q}$.

**Transformation rules:**
- Rank: $\text{rank } E(K) = \text{rank } E(\mathbb{Q}) + \text{(new)}$
- Conductor: $N_{E_K} = N_E^{[K:\mathbb{Q}]} \cdot \text{(ramification)}$
- L-function: $L(E_K, s) = \prod_\chi L(E, s, \chi)$

**Control:** Effects determined by Galois group of $K/\mathbb{Q}$.

**Step 4: Blowup Surgery**

**Operation:** $\pi: \tilde{X} \to X$ blowup at center $Z$.

**Preserved (birational invariants):**
- Kodaira dimension
- Generic properties

**Modified:**
- Picard group: $\text{Pic}(\tilde{X}) = \text{Pic}(X) \oplus \mathbb{Z}^r$
- Cohomology: Extra classes from exceptional divisors

**Control:** Exceptional divisor contribution is explicit.

**Step 5: Surgery Composition**

**Sequential surgery:**
$$E \xrightarrow{\phi} E' \xrightarrow{\text{twist } d} E'^{(d)} \xrightarrow{K/\mathbb{Q}} E'^{(d)}_K$$

**Composite effect:**
- L-function: $L(E, s) \to L(E, s) \to L(E, s, \chi_d) \to \prod_\chi L(E, s, \chi_d \chi)$
- Conductor: Transforms by explicit formula
- Rank: Can only increase or stay same (generically)

**Step 6: Surgery Control Certificate**

The surgery control certificate:
$$K_{\text{SurgCtrl}}^+ = (\text{surgery type}, \text{input}, \text{output}, \text{transformation rules})$$

**Components:**
- **Type:** Isogeny, twist, base change, blowup
- **Input/Output:** Before and after invariants
- **Rules:** Explicit transformation formulas

---

### Key Arithmetic Ingredients

1. **Cassels-Tate** [Cassels 1962]: Isogeny invariance of BSD quotient.
2. **Twist Theory** [Silverman 1994]: Quadratic twist formulas.
3. **Base Change** [Weil 1956]: L-function factorization.
4. **Birational Geometry** [Hartshorne]: Blowup effects.

---

### Arithmetic Interpretation

> **Arithmetic surgery is controlled: each operation has explicit, computable effects on invariants. Isogenies preserve L-functions, twists modify conductors predictably, base changes multiply by Galois data. This controlled surgery enables systematic modification of arithmetic objects.**

---

### Literature

- [Cassels 1962] J.W.S. Cassels, *Arithmetic on curves of genus 1*
- [Silverman 1994] J. Silverman, *Advanced Topics in the Arithmetic of Elliptic Curves*
- [Weil 1956] A. Weil, *The field of definition of a variety*
- [Hartshorne 1977] R. Hartshorne, *Algebraic Geometry*
