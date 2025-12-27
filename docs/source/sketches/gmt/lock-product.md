# LOCK-Product: Product-Regularity Lock — GMT Translation

## Original Statement (Hypostructure)

The product-regularity lock shows that product structures impose rigidity: if a current has local product form, regularity in factor directions implies global regularity.

## GMT Setting

**Product Structure:** Current $T$ locally has form $T = S \times \sigma$ for current $S$ and simplex $\sigma$

**Factor Regularity:** Regularity in each factor direction

**Global Regularity:** Product regularity implies full regularity

## GMT Statement

**Theorem (Product-Regularity Lock).** For $T \in \mathbf{I}_k(M \times N)$ with product structure:

1. **Local Product:** $T = S \times \llbracket\sigma\rrbracket$ locally, where $S \in \mathbf{I}_{k-j}(M)$, $\sigma$ is $j$-simplex in $N$

2. **Factor Regularity:** $\text{sing}(S) = \emptyset \implies \text{sing}(T) = \emptyset$ locally

3. **Lock:** Singularities in product must arise from factor singularities

## Proof Sketch

### Step 1: Product Currents

**Definition:** For $S \in \mathbf{I}_k(M)$, $R \in \mathbf{I}_j(N)$:
$$S \times R \in \mathbf{I}_{k+j}(M \times N)$$

defined by:
$$\langle S \times R, \omega \otimes \eta \rangle = \langle S, \omega \rangle \cdot \langle R, \eta \rangle$$

**Reference:** Federer, H. (1969). *Geometric Measure Theory*. Springer, §4.1.8.

### Step 2: Product of Rectifiable Sets

**Fubini for Rectifiable Sets:** If $E \subset M$ is $k$-rectifiable and $F \subset N$ is $j$-rectifiable:
$$E \times F \subset M \times N \text{ is } (k+j)\text{-rectifiable}$$

**Hausdorff Measure:**
$$\mathcal{H}^{k+j}(E \times F) = \mathcal{H}^k(E) \cdot \mathcal{H}^j(F)$$

**Reference:** Mattila, P. (1995). *Geometry of Sets and Measures*. Cambridge, §8.10.

### Step 3: Tangent Cones of Products

**Product Tangent Cone:** At $(x, y) \in M \times N$:
$$\text{Tan}(S \times R, (x,y)) = \text{Tan}(S, x) \times \text{Tan}(R, y)$$

**Regular Point:** If $x$ regular for $S$ and $y$ regular for $R$:
- $\text{Tan}(S, x) = $ plane
- $\text{Tan}(R, y) = $ plane
- $\text{Tan}(S \times R, (x,y)) = $ plane (product of planes)

### Step 4: Singular Set of Products

**Lemma:** For $T = S \times R$:
$$\text{sing}(T) \subset (\text{sing}(S) \times N) \cup (M \times \text{sing}(R))$$

*Proof:*
- If $(x, y) \notin (\text{sing}(S) \times N) \cup (M \times \text{sing}(R))$
- Then $x \in \text{reg}(S)$ and $y \in \text{reg}(R)$
- Tangent cone at $(x,y)$ is product of planes = plane
- Therefore $(x, y) \in \text{reg}(T)$

### Step 5: Slicing and Product Structure

**Slicing Theorem:** For $T \in \mathbf{I}_k(M \times N)$ and projection $\pi: M \times N \to N$:
$$T = \int_N \langle T, \pi, y \rangle \, dy$$

where $\langle T, \pi, y \rangle \in \mathbf{I}_{k-\dim N}(M \times \{y\})$.

**Reference:** Federer, H. (1969). *Geometric Measure Theory*. Springer, §4.3.

**Product Recognition:** $T$ is product if slices are independent of $y$.

### Step 6: Regularity Propagation

**Theorem:** If $T = S \times \llbracket\sigma\rrbracket$ with $S$ regular:

Then $T$ is regular throughout $M \times \text{int}(\sigma)$.

*Proof:*
1. $S$ regular means $\text{sing}(S) = \emptyset$
2. $\llbracket\sigma\rrbracket$ is regular on interior of simplex
3. Product has singularities only at $\text{sing}(S) \times \sigma \cup M \times \partial\sigma$
4. With $\text{sing}(S) = \emptyset$, only boundary singularities remain

### Step 7: Künneth Formula for Currents

**Homological Product:** At homology level:
$$H_*(M \times N) \cong H_*(M) \otimes H_*(N)$$

**Reference:** Hatcher, A. (2002). *Algebraic Topology*. Cambridge, §3.B.

**Current Analogue:** Homology class of $S \times R$ is $[S] \otimes [R]$.

### Step 8: Monotonicity for Products

**Product Monotonicity:** For $T = S \times R$:
$$\Theta^{k+j}(T, (x,y), r) = \Theta^k(S, x, r) \cdot \Theta^j(R, y, r)$$

at small scales where product structure holds.

**Density Bound:** Product density bounded by product of factor densities.

### Step 9: Splitting Theorems

**Cheeger-Colding Splitting:** If metric space has a line, it splits:
$$X \cong Y \times \mathbb{R}$$

**Reference:** Cheeger, J., Colding, T. H. (1996). Lower bounds on Ricci curvature and the almost rigidity of warped products. *Ann. of Math.*, 144, 189-237.

**Application:** Product structure in tangent cone forces splitting.

### Step 10: Compilation Theorem

**Theorem (Product-Regularity Lock):**

1. **Product Form:** $T = S \times R$ locally

2. **Factor Regularity:** $\text{sing}(T) \subset (\text{sing}(S) \times N) \cup (M \times \text{sing}(R))$

3. **Lock:** Singularities determined by factors

4. **Propagation:** Factor regularity implies product regularity

**Applications:**
- Dimension reduction for regularity
- Product structure in minimal varieties
- Splitting of singular sets

## Key GMT Inequalities Used

1. **Product Mass:**
   $$\mathbf{M}(S \times R) = \mathbf{M}(S) \cdot \mathbf{M}(R)$$

2. **Singular Set Containment:**
   $$\text{sing}(S \times R) \subset (\text{sing}(S) \times N) \cup (M \times \text{sing}(R))$$

3. **Product Density:**
   $$\Theta(S \times R, (x,y)) = \Theta(S, x) \cdot \Theta(R, y)$$

4. **Hausdorff Product:**
   $$\mathcal{H}^{k+j}(E \times F) = \mathcal{H}^k(E) \cdot \mathcal{H}^j(F)$$

## Literature References

- Federer, H. (1969). *Geometric Measure Theory*. Springer.
- Mattila, P. (1995). *Geometry of Sets and Measures*. Cambridge.
- Hatcher, A. (2002). *Algebraic Topology*. Cambridge.
- Cheeger, J., Colding, T. H. (1996). Almost rigidity of warped products. *Ann. of Math.*, 144.
- Simon, L. (1983). *Lectures on Geometric Measure Theory*. Australian National University.
