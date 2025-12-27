# THM-ExpansionAdjunction: Expansion Adjunction — GMT Translation

## Original Statement (Hypostructure)

There is an adjunction between thin (local) data and full (global) data: local certificates expand to global ones, and global data restricts to local data, with these operations forming an adjoint pair.

## GMT Setting

**Thin Data:** Local information $\text{loc}_x(T)$ at points $x \in M$

**Full Data:** Global current $T \in \mathbf{I}_k(M)$

**Expansion:** $\text{Exp}: \text{Local} \to \text{Global}$

**Restriction:** $\text{Res}: \text{Global} \to \text{Local}$

## GMT Statement

**Theorem (Expansion Adjunction).** There is an adjunction:
$$\text{Exp} \dashv \text{Res}: \mathbf{I}_k(M) \rightleftarrows \prod_{x \in M} \mathcal{G}_x$$

where $\mathcal{G}_x$ is the germ space at $x$. Explicitly:

1. **Adjunction Formula:**
$$\text{Hom}_{\text{global}}(\text{Exp}(L), T) \cong \text{Hom}_{\text{local}}(L, \text{Res}(T))$$

2. **Unit:** $\eta_L: L \to \text{Res}(\text{Exp}(L))$ (local data embeds in restriction of expansion)

3. **Counit:** $\varepsilon_T: \text{Exp}(\text{Res}(T)) \to T$ (expansion of restrictions maps to original)

## Proof Sketch

### Step 1: Local Data Category

**Germ Space:** At $x \in M$:
$$\mathcal{G}_x := \{\text{germ}_x(T) : T \in \mathbf{I}_k(M)\}$$

**Product of Germs:** The local data category is:
$$\text{Local} := \prod_{x \in M} \mathcal{G}_x$$

with morphisms preserving germ structure.

**Reference:** Bredon, G. E. (1997). *Sheaf Theory*. Springer. [Chapter I]

### Step 2: Restriction Functor

**Definition:** For $T \in \mathbf{I}_k(M)$:
$$\text{Res}(T) := (\text{germ}_x(T))_{x \in M} \in \prod_x \mathcal{G}_x$$

**Functoriality:** For morphism $f: T \to S$:
$$\text{Res}(f)_x := \text{germ}_x(f): \text{germ}_x(T) \to \text{germ}_{f(x)}(S)$$

**Properties:**
- Preserves identity: $\text{Res}(\text{id}_T) = \text{id}_{\text{Res}(T)}$
- Preserves composition: $\text{Res}(g \circ f) = \text{Res}(g) \circ \text{Res}(f)$

### Step 3: Expansion Functor

**Definition:** Given local data $L = (\gamma_x)_{x \in D}$ for dense $D \subset M$:
$$\text{Exp}(L) := \text{unique } T \text{ with } \text{germ}_x(T) = \gamma_x \text{ for } x \in D$$

**Existence (Federer, 1969):** Integral currents are determined by their restrictions to dense subsets.

**Reference:** Federer, H. (1969). *Geometric Measure Theory*. Springer. [Section 4.1]

**Uniqueness:** By the constancy theorem, if germs agree on a dense set, currents are equal.

### Step 4: Adjunction Verification

**Claim:** $\text{Exp} \dashv \text{Res}$.

**Proof:** We verify the adjunction isomorphism:
$$\text{Hom}(\text{Exp}(L), T) \cong \text{Hom}(L, \text{Res}(T))$$

**Forward Map:** Given $f: \text{Exp}(L) \to T$, define $\tilde{f}: L \to \text{Res}(T)$ by:
$$\tilde{f}_x := \text{germ}_x(f)$$

**Backward Map:** Given $g: L \to \text{Res}(T)$, define $\hat{g}: \text{Exp}(L) \to T$ by gluing local maps.

**Inverse Verification:** These maps are mutually inverse.

### Step 5: Unit and Counit

**Unit $\eta$:** For local data $L$:
$$\eta_L: L \to \text{Res}(\text{Exp}(L))$$

defined by $(\eta_L)_x = \text{germ}_x(\text{Exp}(L)) = \gamma_x$ (identity on compatible germs).

**Counit $\varepsilon$:** For global $T$:
$$\varepsilon_T: \text{Exp}(\text{Res}(T)) \to T$$

defined by $\varepsilon_T = \text{id}_T$ (expansion of restrictions equals original).

**Triangle Identities:**
1. $\varepsilon_{\text{Exp}(L)} \circ \text{Exp}(\eta_L) = \text{id}_{\text{Exp}(L)}$
2. $\text{Res}(\varepsilon_T) \circ \eta_{\text{Res}(T)} = \text{id}_{\text{Res}(T)}$

### Step 6: Sheaf-Theoretic Interpretation

**Sheaf of Currents:** Define presheaf $\mathcal{I}_k$ on $M$:
$$\mathcal{I}_k(U) := \mathbf{I}_k(U)$$

**Sheaf Condition:** $\mathcal{I}_k$ is a sheaf:
- Compatible local sections glue to global sections
- Sections are determined by restrictions

**Adjunction as Sheafification:** The expansion-restriction adjunction is the sheafification adjunction:
$$\text{sheafify} \dashv \text{forget}: \text{Sh}(M) \rightleftarrows \text{PreSh}(M)$$

**Reference:** Mac Lane, S., Moerdijk, I. (1994). *Sheaves in Geometry and Logic*. Springer.

### Step 7: Tangent Cone Expansion

**Local-to-Global for Singularities:** Given tangent cones $C_x$ at each $x \in \text{sing}(T)$:
$$\text{Exp}(\{C_x\}_{x \in \text{sing}}) \to T$$

reconstructs $T$ near its singular set.

**Theorem (Simon, 1983):** Currents are determined by their tangent cones at singular points plus regular part.

**Reference:** Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU.

### Step 8: Profile Expansion

**Profile Decomposition to Current:**
$$\text{Exp}: \{V^l, (x_j^l, \lambda_j^l)\} \mapsto T = \sum_l V^l_j + w_j$$

reconstructs current from profile data.

**Adjoint:** Restriction extracts profiles:
$$\text{Res}: T \mapsto \{V^l\}_l$$

### Step 9: Application to Permits

**Local Permits:** $\Pi_x^{\text{loc}}$ — permit data at $x$

**Global Permits:** $\Pi$ — global permit configuration

**Adjunction:**
$$\text{Exp}(\{\Pi_x^{\text{loc}}\}) \to \Pi \to \text{Res}(\Pi) = \{\Pi_x^{\text{loc}}\}$$

**Consistency:** Local permits are compatible iff they expand to a global permit.

### Step 10: Compilation Theorem

**Theorem (Expansion Adjunction):**

1. **Adjunction:** $\text{Exp} \dashv \text{Res}$

2. **Unit/Counit:** Satisfy triangle identities

3. **Sheaf Interpretation:** Currents form a sheaf, adjunction is sheafification

4. **Applications:**
   - Local-to-global reconstruction of currents
   - Profile-to-current expansion
   - Permit consistency checking

**Constructive Content:**
- Given compatible local data, construct global current
- Given global current, extract local data
- Verify compatibility conditions algorithmically

## Key GMT Inequalities Used

1. **Local Determination:**
   $$T|_U = S|_U \text{ for dense } U \implies T = S$$

2. **Sheaf Gluing:**
   $$\{T_\alpha\} \text{ compatible on overlaps} \implies \exists ! T: T|_{U_\alpha} = T_\alpha$$

3. **Germ Equality:**
   $$\text{germ}_x(T) = \text{germ}_x(S) \implies T = S \text{ near } x$$

4. **Profile Reconstruction:**
   $$T = \sum_l V^l_j + w_j$$

## Literature References

- Federer, H. (1969). *Geometric Measure Theory*. Springer.
- Bredon, G. E. (1997). *Sheaf Theory*. Springer.
- Mac Lane, S., Moerdijk, I. (1994). *Sheaves in Geometry and Logic*. Springer.
- Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU.
- Awodey, S. (2010). *Category Theory*. Oxford.
- Johnstone, P. T. (2002). *Sketches of an Elephant: A Topos Theory Compendium*. Oxford.
