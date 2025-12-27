# KRNL-StiffPairing: Stiff Pairing / No Null Directions — GMT Translation

## Original Statement (Hypostructure)

If stiffness, topological background, and gradient consistency certificates hold, then the bilinear pairing on the state space has no null directions — every degree of freedom is either free or obstructed.

## GMT Setting

**Ambient Space:** $(M^n, g)$ — Riemannian manifold

**Current Space:** $\mathbf{I}_k(M)$ — integral $k$-currents

**Bilinear Pairing:** $\langle \cdot, \cdot \rangle: \mathbf{I}_k(M) \times \mathbf{I}_{n-k}(M) \to \mathbb{Z}$ — intersection pairing

**Free/Obstruction Decomposition:** $\mathbf{I}_k(M) \cong \mathbf{I}_k^{\text{free}}(M) \oplus \mathbf{I}_k^{\text{obs}}(M) \oplus \mathbf{I}_k^{\text{null}}(M)$

**Stiffness:** Non-degeneracy of the Hessian at critical points

## GMT Statement

**Theorem (No Null Directions for Stiff Pairings).** Let $(M^n, g)$ be a compact oriented Riemannian manifold. Consider the intersection pairing:
$$\langle \cdot, \cdot \rangle: H_k(M; \mathbb{Z}) \times H_{n-k}(M; \mathbb{Z}) \to \mathbb{Z}$$

Assume:

1. **(Stiffness)** The Morse function $\Phi: M \to \mathbb{R}$ satisfies the Morse-Smale condition: stable and unstable manifolds intersect transversally.

2. **(Topological Bound)** The Betti numbers satisfy $b_k(M) < \infty$ for all $k$.

3. **(Gradient Consistency)** The flow $-\nabla \Phi$ is Kupka-Smale: all periodic orbits are hyperbolic.

Then the radical of the intersection pairing is trivial:
$$\text{rad}(\langle \cdot, \cdot \rangle) := \{[T] \in H_k(M) : \langle [T], [S] \rangle = 0 \, \forall [S] \in H_{n-k}(M)\} = \{0\}$$

Equivalently: $H_k(M) = H_k^{\text{free}}(M) \oplus H_k^{\text{torsion}}(M)$ with no additional null directions.

## Proof Sketch

### Step 1: Intersection Pairing via Currents

**Current Representatives:** For $[T] \in H_k(M; \mathbb{Z})$, choose a representative integral current $T \in \mathbf{I}_k(M)$ with $\partial T = 0$.

**Intersection Number:** For transversely intersecting currents $T \in \mathbf{I}_k(M)$ and $S \in \mathbf{I}_{n-k}(M)$:
$$\langle T, S \rangle = \sum_{p \in \text{spt}(T) \cap \text{spt}(S)} \epsilon_p \cdot \Theta^0(T, p) \cdot \Theta^0(S, p)$$

where $\epsilon_p = \pm 1$ is the orientation sign and $\Theta^0$ is the multiplicity.

**Homological Pairing:** The intersection pairing descends to homology:
$$\langle [T], [S] \rangle = \langle T, S' \rangle$$

for any $S'$ homologous to $S$ and transverse to $T$.

### Step 2: Poincaré Duality and Non-Degeneracy

**Poincaré Duality:** For compact oriented $M^n$:
$$\text{PD}: H_k(M; \mathbb{Z}) \xrightarrow{\cong} H^{n-k}(M; \mathbb{Z})$$

**Cap Product Formulation:** The intersection pairing factors through the cap product:
$$\langle [T], [S] \rangle = \langle \text{PD}([T]) \cup \text{PD}([S]), [M] \rangle$$

**Non-Degeneracy:** By Poincaré duality, if $\langle [T], [S] \rangle = 0$ for all $[S]$, then $\text{PD}([T]) = 0$ in cohomology, hence $[T] = 0$ in homology (modulo torsion).

### Step 3: Morse-Theoretic Perspective

**Morse Complex:** Given a Morse function $\Phi: M \to \mathbb{R}$, the Morse complex is:
$$CM_k(\Phi) = \bigoplus_{p \in \text{Crit}_k(\Phi)} \mathbb{Z} \cdot \langle p \rangle$$

where $\text{Crit}_k(\Phi)$ is the set of critical points of index $k$.

**Boundary Operator:** $\partial: CM_k \to CM_{k-1}$ counts gradient flow lines:
$$\partial \langle p \rangle = \sum_{q \in \text{Crit}_{k-1}} n(p, q) \langle q \rangle$$

where $n(p, q) = \#(W^u(p) \cap W^s(q)) \mod 2$ is the (signed) count of flow lines from $p$ to $q$.

### Step 4: Stiffness and Transversality

**Morse-Smale Condition:** The gradient flow $-\nabla \Phi$ is Morse-Smale if:
$$W^u(p) \pitchfork W^s(q) \quad \text{for all } p, q \in \text{Crit}(\Phi)$$

**Consequence:** The intersection $W^u(p) \cap W^s(q)$ is a smooth manifold of dimension:
$$\dim(W^u(p) \cap W^s(q)) = \text{ind}(p) - \text{ind}(q)$$

**Stiffness = Transversality:** The stiffness condition ($|\partial \Phi| \geq C |\Phi - \Phi_c|^{1-\theta}$) ensures exponential convergence to critical points, which is a quantitative form of hyperbolicity guaranteeing Morse-Smale transversality.

### Step 5: Pairing via Stable/Unstable Manifolds

**Dual Morse Complex:** For the function $-\Phi$ (reversed flow), the unstable manifolds become stable and vice versa.

**Pairing on Morse Chains:** The intersection pairing on Morse chains is:
$$\langle \langle p \rangle, \langle q \rangle \rangle = \delta_{\text{ind}(p) + \text{ind}(q), n} \cdot \#(W^u(p) \cap W^u_-(q))$$

where $W^u_-(q)$ is the unstable manifold for $-\Phi$.

**Non-Degeneracy at Chain Level:** By Morse-Smale transversality, if $\text{ind}(p) + \text{ind}(q) = n$, the intersection is transverse and 0-dimensional (finite set of points).

**Theorem (Morse-Theoretic Poincaré Duality):** The pairing $\langle \cdot, \cdot \rangle$ on Morse homology is non-degenerate.

### Step 6: Removing Null Directions

**Null Direction:** A class $[T] \in H_k(M)$ is null if $\langle [T], [S] \rangle = 0$ for all $[S]$.

**Obstruction to Null:** By Morse-Smale transversality, the stable manifold $W^s(p)$ of a critical point $p$ of index $k$ is a cycle representing a non-zero class in $H_k(M)$.

The dual unstable manifold $W^u(p)$ (for $-\Phi$) represents a class in $H_{n-k}(M)$ with:
$$\langle [W^s(p)], [W^u(p)] \rangle = \pm 1$$

(a single transverse intersection point at $p$).

**No Null Directions:** Every non-zero $[T] \in H_k(M)$ can be written as $[T] = \sum_p a_p [W^s(p)]$. Choosing $[S] = [W^u(p_0)]$ for any $p_0$ with $a_{p_0} \neq 0$:
$$\langle [T], [S] \rangle = a_{p_0} \neq 0$$

Hence $[T]$ is not null.

## Key GMT Inequalities Used

1. **Poincaré Duality Isomorphism:**
   $$H_k(M; \mathbb{Z}) \cong H^{n-k}(M; \mathbb{Z})$$

2. **Transverse Intersection Count:**
   $$\dim(A \cap B) = \dim(A) + \dim(B) - n \text{ if } A \pitchfork B$$

3. **Morse Inequality:**
   $$\#\text{Crit}_k(\Phi) \geq b_k(M)$$

4. **Łojasiewicz Convergence Rate:**
   $$d(S_t x, p) \leq C e^{-\lambda t} \text{ for } S_t x \to p \in \text{Crit}(\Phi)$$

## Literature References

- Milnor, J. (1963). *Morse Theory*. Princeton University Press.
- Schwarz, M. (1993). *Morse Homology*. Birkhäuser.
- Banyaga, A., Hurtubise, D. (2004). *Lectures on Morse Homology*. Kluwer.
- Latour, F. (1994). Existence de 1-formes fermées non singulières. *Annales de l'Institut Fourier*, 44(4), 1037-1060.
- Hutchings, M. (2002). Lecture notes on Morse homology. *Available online*.
- Nicolaescu, L. (2011). *An Invitation to Morse Theory*. 2nd ed. Springer.
