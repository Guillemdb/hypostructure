# THM-MetaIdentifiability: Meta-Identifiability

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-thm-meta-identifiability*

The structural framework is identifiable from its invariants: the theory can recognize its own structure.

---

## Arithmetic Formulation

### Setup

"Meta-identifiability" in arithmetic means:
- **Axioms:** Number field axioms, height properties, Galois structure
- **Identification:** Axioms determine the category of arithmetic objects
- **Self-reference:** Arithmetic can characterize its own foundation

### Statement (Arithmetic Version)

**Theorem (Arithmetic Meta-Identifiability).** Arithmetic structures are self-characterizing:

1. **Field axioms:** $\mathbb{Q}$ is the unique prime field of characteristic 0
2. **Height axioms:** Northcott + product formula characterize heights
3. **Galois axioms:** Absolute Galois group determines all extensions

---

### Proof

**Step 1: Characterization of $\mathbb{Q}$**

**Theorem:** $\mathbb{Q}$ is the unique field with:
- Characteristic 0
- Prime (no proper subfields)
- Complete with respect to all absolute values (Ostrowski)

**Reference:** [Artin-Whaples 1945]

**Step 2: Axiomatic Height Theory**

**Weil height machine axioms:**
1. **Functoriality:** $h_\phi = \deg(\phi) \cdot h + O(1)$
2. **Product formula:** $\sum_v \log|\alpha|_v = 0$
3. **Northcott:** Finite sets of bounded height and degree

**Identification [Lang]:** These axioms uniquely determine height on $\bar{\mathbb{Q}}$.

**Step 3: Absolute Galois Group**

**Definition:** $G_\mathbb{Q} = \text{Gal}(\bar{\mathbb{Q}}/\mathbb{Q})$

**Neukirch-Uchida Theorem:** For number fields $K, L$:
$$G_K \cong G_L \iff K \cong L$$

**Meta-identification:** The absolute Galois group identifies the field.

**Reference:** [Neukirch 1969, Uchida 1976]

**Step 4: Grothendieck's Section Conjecture**

**Conjecture:** For hyperbolic curve $X/K$:
$$X(K) \leftrightarrow \text{sections of } \pi_1^{\text{ét}}(X) \to G_K$$

**Meta-identification:** Rational points identified by group-theoretic data.

**Step 5: Self-Reference in Arithmetic**

**Gödel-like:** Arithmetic contains enough structure to:
- Encode finite sequences (Gödel numbering)
- Express properties of arithmetic itself
- Formalize the axioms

**But:** No incompleteness paradox for basic number-theoretic statements (decidability issues remain).

**Step 6: Identification Certificate**

$$K_{\text{Meta}}^+ = (\text{axiom system}, \text{structure characterized}, \text{uniqueness})$$

Examples:
- (Field axioms, $\mathbb{Q}$, unique prime field char 0)
- (Galois group, number field, Neukirch-Uchida)
- (Height axioms, Weil height, Lang's characterization)

---

### Key Arithmetic Ingredients

1. **Artin-Whaples Axioms** [1945]: Product formula characterization.
2. **Neukirch-Uchida** [1969, 1976]: Field from Galois group.
3. **Lang's Height Machine** [1983]: Axiomatic heights.
4. **Section Conjecture** [Grothendieck]: Anabelian geometry.

---

### Arithmetic Interpretation

> **Arithmetic is meta-identifiable. The axioms of height theory, Galois theory, and field theory determine the category of number fields and their arithmetic uniquely. The Neukirch-Uchida theorem shows that $G_K$ identifies $K$ — the symmetry structure knows the field. This self-characterization is the arithmetic analogue of a theory recognizing its own axioms.**

---

### Literature

- [Artin-Whaples 1945] E. Artin, G. Whaples, *Axiomatic characterization of fields*
- [Neukirch 1969] J. Neukirch, *Kennzeichnung der p-adischen Zahlkörper*
- [Uchida 1976] K. Uchida, *Isomorphisms of Galois groups*
- [Lang 1983] S. Lang, *Fundamentals of Diophantine Geometry*
