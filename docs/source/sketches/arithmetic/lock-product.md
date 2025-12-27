# LOCK-Product: Product-Regularity

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: mt-lock-product*

Regularity of products is determined by regularity of factors.

---

## Arithmetic Formulation

### Setup

"Product regularity" in arithmetic means:
- $A \times B$ regular iff both $A$ and $B$ regular
- L-function of product = product of L-functions
- Galois representation of product = tensor product

### Statement (Arithmetic Version)

**Theorem (Arithmetic Product Regularity).** Products inherit regularity:

1. **Good reduction:** $A \times B$ has good reduction at $p$ iff both do
2. **L-function:** $L(A \times B, s) = L(A, s) \cdot L(B, s)$ (for appropriate products)
3. **Rank:** $\text{rank}(A \times B)(\mathbb{Q}) = \text{rank } A(\mathbb{Q}) + \text{rank } B(\mathbb{Q})$

---

### Proof

**Step 1: Good Reduction Product**

For abelian varieties $A, B$ over $K$:

**Product:** $(A \times B)/K$

**Reduction:** $(A \times B)_p = A_p \times B_p$

**Good reduction equivalence:**
$$A \times B \text{ has good reduction at } p \iff A, B \text{ both have good reduction at } p$$

**Proof:** Néron model satisfies $(A \times B)^0 = A^0 \times B^0$.

**Step 2: L-function Product**

For motives $M, N$:

**Product L-function:**
$$L(M \otimes N, s) \neq L(M, s) \cdot L(N, s) \text{ in general}$$

**But for direct sum:**
$$L(M \oplus N, s) = L(M, s) \cdot L(N, s)$$

**Abelian variety case:**
$$L(A \times B, s) = L(A, s) \cdot L(B, s)$$

since $H^1(A \times B) = H^1(A) \oplus H^1(B)$.

**Step 3: Rank Additivity**

For elliptic curves $E, E'$ over $\mathbb{Q}$:

**Product Mordell-Weil:**
$$(E \times E')(\mathbb{Q}) \cong E(\mathbb{Q}) \times E'(\mathbb{Q})$$

**Rank:**
$$\text{rank}(E \times E')(\mathbb{Q}) = \text{rank } E(\mathbb{Q}) + \text{rank } E'(\mathbb{Q})$$

**Proof:** The isomorphism is canonical.

**Step 4: Conductor Product**

For abelian varieties:

**Conductor of product:**
$$N_{A \times B} = N_A^{\dim B} \cdot N_B^{\dim A} \cdot (\text{correction})$$

**Simplified (elliptic curves):**
$$N_{E \times E'} = N_E \cdot N_{E'}$$

(when $E, E'$ have coprime conductors)

**Step 5: Galois Representation Product**

**Direct sum representation:**
$$\rho_{A \times B} = \rho_A \oplus \rho_B$$

**Tensor product:**
$$\rho_A \otimes \rho_B: G_K \to \text{GL}(V_A \otimes V_B)$$

**Regularity transfer:**
- $\rho_{A \times B}$ crystalline $\iff$ $\rho_A, \rho_B$ crystalline
- Product of geometric = geometric

**Step 6: Product Regularity Certificate**

The product regularity certificate:
$$K_{\text{Prod}}^+ = ((A, B), \text{product type}, \text{regularity equivalence})$$

**Components:**
- **Factors:** $A, B$ with their regularity data
- **Product:** $A \times B$ or $A \otimes B$
- **Equivalence:** Proof that product regularity $\Leftrightarrow$ factor regularity

---

### Key Arithmetic Ingredients

1. **Néron Models** [Néron 1964]: Product of Néron models.
2. **Künneth Formula** [Grothendieck]: Cohomology of products.
3. **Mordell-Weil** [Weil 1928]: Group law on products.
4. **Conductor Formula** [Serre 1970]: Conductor of products.

---

### Arithmetic Interpretation

> **Arithmetic regularity respects products. Good reduction, L-functions, and ranks behave multiplicatively/additively for products. A product is regular iff its factors are. This product principle reduces analysis of composite objects to their factors.**

---

### Literature

- [Néron 1964] A. Néron, *Modèles minimaux des variétés abéliennes*
- [Grothendieck 1967] A. Grothendieck, *Cohomologie ℓ-adique et fonctions L*
- [Weil 1928] A. Weil, *L'arithmétique sur les courbes algébriques*
- [Serre 1970] J.-P. Serre, *Facteurs locaux des fonctions zêta*
