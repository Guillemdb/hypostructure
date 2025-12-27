# RESOLVE-Conservation: Conservation of Flow — GMT Translation

## Original Statement (Hypostructure)

Surgery does not create or destroy net mass/energy. The total conserved quantity before surgery equals the total after surgery plus the excised contribution.

## GMT Setting

**Pre-Surgery Current:** $T \in \mathbf{I}_k(M)$ — integral current before surgery

**Surgery Region:** $\Sigma \subset M$ — singular set being excised

**Post-Surgery Current:** $T' \in \mathbf{I}_k(M \setminus \Sigma)$ — current after surgery

**Conserved Quantity:** $\mathbf{M}(T)$ — mass functional

## GMT Statement

**Theorem (Conservation of Flow).** Let $T \in \mathbf{I}_k(M)$ undergo surgery at $\Sigma$ producing $T'$. Then:

$$\mathbf{M}(T) = \mathbf{M}(T') + \mathbf{M}(T \llcorner \Sigma) + \mathcal{E}_{\text{bdry}}$$

where:
- $T \llcorner \Sigma$ is the restriction of $T$ to $\Sigma$
- $\mathcal{E}_{\text{bdry}} = O(\mathcal{H}^{k-1}(\partial \Sigma))$ is boundary contribution

Moreover, the boundary operator is preserved:
$$\partial T' = \partial T - \partial(T \llcorner \Sigma) + \text{gluing terms}$$

## Proof Sketch

### Step 1: Mass Decomposition

**Localization of Mass (Federer, 1969):** For $T \in \mathbf{I}_k(M)$ and measurable $A \subset M$:
$$\mathbf{M}(T) = \mathbf{M}(T \llcorner A) + \mathbf{M}(T \llcorner (M \setminus A))$$

when $A$ and $M \setminus A$ are separated.

**Reference:** Federer, H. (1969). *Geometric Measure Theory*. Springer. [Section 4.1.7]

**Surgery Decomposition:** With $A = B_\varepsilon(\Sigma)$ (tubular neighborhood):
$$T = T \llcorner B_\varepsilon(\Sigma) + T \llcorner (M \setminus B_\varepsilon(\Sigma))$$

### Step 2: Boundary Behavior Under Restriction

**Restriction and Boundary (Federer-Fleming, 1960):** For $T \in \mathbf{I}_k(M)$:
$$\partial(T \llcorner A) = (\partial T) \llcorner A + T \llcorner \partial A$$

where $T \llcorner \partial A$ is the slice of $T$ by $\partial A$.

**Reference:** Federer, H., Fleming, W. (1960). Normal and integral currents. *Ann. of Math.*, 72, 458-520.

**Slicing Formula:** If $\partial A = f^{-1}(0)$ for Lipschitz $f$:
$$\langle T, f, 0 \rangle = T \llcorner \{f = 0\}$$

### Step 3: Gluing Construction

**Gluing Currents (White, 1989):** Given $T_1 \in \mathbf{I}_k(U_1)$ and $T_2 \in \mathbf{I}_k(U_2)$ with compatible boundaries on $U_1 \cap U_2$:
$$T_1 \cup T_2 \in \mathbf{I}_k(U_1 \cup U_2)$$

satisfies $\mathbf{M}(T_1 \cup T_2) = \mathbf{M}(T_1) + \mathbf{M}(T_2) - \mathbf{M}(T_1 \cap T_2)$.

**Reference:** White, B. (1989). A new proof of the compactness theorem for integral currents. *Comment. Math. Helv.*, 64, 207-220.

**Surgery Gluing:** Replace $T \llcorner B_\varepsilon(\Sigma)$ with standard profile $V$:
$$T' = (T \llcorner (M \setminus B_\varepsilon(\Sigma))) \cup V$$

### Step 4: Mass Conservation Identity

**Conservation Formula:** By additivity:
$$\mathbf{M}(T) = \mathbf{M}(T \llcorner (M \setminus B_\varepsilon(\Sigma))) + \mathbf{M}(T \llcorner B_\varepsilon(\Sigma))$$

After surgery:
$$\mathbf{M}(T') = \mathbf{M}(T \llcorner (M \setminus B_\varepsilon(\Sigma))) + \mathbf{M}(V)$$

**Mass Change:**
$$\mathbf{M}(T) - \mathbf{M}(T') = \mathbf{M}(T \llcorner B_\varepsilon(\Sigma)) - \mathbf{M}(V)$$

### Step 5: Energy Conservation for Gradient Flows

**Energy-Dissipation Equality (Ambrosio-Gigli-Savaré, 2008):** For gradient flows:
$$\Phi(T(0)) - \Phi(T(t)) = \int_0^t |\partial \Phi|^2(T(s)) \, ds$$

**Reference:** Ambrosio, L., Gigli, N., Savaré, G. (2008). *Gradient Flows in Metric Spaces and in the Space of Probability Measures*. Birkhäuser.

**Surgery Conservation:** At surgery time $t_*$:
$$\Phi(T(t_*^-)) = \Phi(T(t_*^+)) + \Delta\Phi_{\text{surg}}$$

where $\Delta\Phi_{\text{surg}} \geq \epsilon_T > 0$ is the energy drop.

### Step 6: Boundary Conservation

**Boundary Compatibility:** Surgery preserves the global boundary:
$$\partial T' = \partial T$$

on $M \setminus \overline{B_\varepsilon(\Sigma)}$.

**Proof:** The gluing region $\partial B_\varepsilon(\Sigma)$ has matching orientation:
$$\partial(T \llcorner (M \setminus B_\varepsilon)) = (\partial T) \llcorner (M \setminus B_\varepsilon) - \langle T, d_\Sigma, \varepsilon \rangle$$
$$\partial V = \langle T, d_\Sigma, \varepsilon \rangle$$

where $d_\Sigma$ is the distance to $\Sigma$.

### Step 7: Noether's Theorem for Currents

**Symmetry and Conservation (Almgren, 1966):** If the variational problem has symmetry group $G$, then:
$$J_\xi(T) := \int_T \iota_\xi \omega$$

is conserved along the flow, where $\xi$ is the infinitesimal generator.

**Reference:** Almgren, F. J. (1966). Some interior regularity theorems for minimal surfaces and an extension of Bernstein's theorem. *Ann. of Math.*, 84, 277-292.

**Conservation Under Surgery:** If surgery respects $G$-symmetry:
$$J_\xi(T) = J_\xi(T')$$

### Step 8: Bookkeeping Identity

**Full Conservation Identity:**
$$\underbrace{\mathbf{M}(T)}_{\text{before}} = \underbrace{\mathbf{M}(T')}_{\text{after}} + \underbrace{\mathbf{M}(T \llcorner \Sigma)}_{\text{excised}} + \underbrace{\mathbf{M}(V) - \mathbf{M}(T \llcorner B_\varepsilon(\Sigma) \setminus \Sigma)}_{\text{replacement error}}$$

**Controlled Error:** By capacity bounds:
$$|\mathbf{M}(V) - \mathbf{M}(T \llcorner B_\varepsilon(\Sigma))| \leq C \cdot \varepsilon^{k-\dim(\Sigma)}$$

## Key GMT Inequalities Used

1. **Mass Additivity:**
   $$\mathbf{M}(T) = \mathbf{M}(T \llcorner A) + \mathbf{M}(T \llcorner A^c)$$

2. **Slicing Bound:**
   $$\int_0^R \mathbf{M}(\langle T, f, t \rangle) \, dt \leq \text{Lip}(f) \cdot \mathbf{M}(T)$$

3. **Energy-Dissipation:**
   $$\Phi(T_0) - \Phi(T_t) = \int_0^t \mathfrak{D}(T_s) \, ds$$

4. **Boundary Operator:**
   $$\partial(T \llcorner A) = (\partial T) \llcorner A + \langle T, \partial A \rangle$$

## Literature References

- Federer, H. (1969). *Geometric Measure Theory*. Springer.
- Federer, H., Fleming, W. (1960). Normal and integral currents. *Ann. of Math.*, 72.
- White, B. (1989). Compactness theorem for integral currents. *Comment. Math. Helv.*, 64.
- Ambrosio, L., Gigli, N., Savaré, G. (2008). *Gradient Flows*. Birkhäuser.
- Almgren, F. J. (1966). Interior regularity theorems for minimal surfaces. *Ann. of Math.*, 84.
- Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU.
