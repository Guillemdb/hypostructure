# KRNL-WeakStrong: Weak-Strong Uniqueness — GMT Translation

## Original Statement (Hypostructure)

If a "strong" solution with stiffness exists, any "weak" solution constructed via compactness must coincide with it. Regularity implies uniqueness.

## GMT Setting

**Ambient Space:** $(\Omega, |\cdot|, \mathcal{L}^n)$ — bounded domain in $\mathbb{R}^n$

**Weak Solutions:** $\mathbf{I}_k^{\text{weak}}(\Omega)$ — integral currents satisfying equations in distributional sense

**Strong Solutions:** $\mathbf{I}_k^{\text{strong}}(\Omega)$ — currents represented by smooth submanifolds

**Energy Space:** $W^{1,p}(\Omega)$ — Sobolev space with $p > n$ (supercritical regularity)

## GMT Statement

**Theorem (Weak-Strong Uniqueness for Currents).** Let $T_w, T_s \in \mathbf{I}_k(\Omega \times [0, T])$ be two integral currents representing solutions to a variational problem with:

1. **(Weak Solution)** $T_w$ satisfies the Euler-Lagrange equation in the sense of currents:
   $$\partial T_w + H_{T_w} \llcorner \|T_w\| = 0$$
   with bounded mass: $\mathbf{M}(T_w) \leq \Lambda$

2. **(Strong Solution)** $T_s$ is represented by a smooth $k$-dimensional submanifold $\Sigma_s \subset \Omega$ and satisfies the equation classically with Serrin-type integrability:
   $$\int_0^T \|H_{T_s}\|_{L^r(\Sigma_s)}^p \, dt < \infty$$
   where $\frac{k}{p} + \frac{n-k}{r} < 1$ (Serrin condition)

3. **(Same Initial Data)** $T_w \llcorner \{t = 0\} = T_s \llcorner \{t = 0\}$ as currents

Then $T_w = T_s$ as currents on $\Omega \times [0, T]$.

## Proof Sketch

### Step 1: Energy Inequality for Weak Solutions

**Weak Solution Energy:** By the definition of weak solution via concentration-compactness (Lions, 1984), $T_w$ satisfies the energy inequality:
$$\mathbf{M}(T_w \llcorner \{t = t_1\}) + \int_0^{t_1} \int |H_{T_w}|^2 \, d\|T_w\| \, dt \leq \mathbf{M}(T_w \llcorner \{t = 0\})$$

**Reference:** Lions, P.-L. (1984). The concentration-compactness principle in the calculus of variations. *Ann. Inst. H. Poincaré Anal. Non Linéaire*, 1, 109-145.

### Step 2: Strong Solution Regularity

**Serrin Condition:** The strong solution $T_s$ satisfies the regularity estimate:
$$\|T_s\|_{L^\infty([0,T]; C^{1,\alpha})} \leq C(\|T_s\|_{L^p_t L^r_x}, \|T_s(0)\|_{C^{1,\alpha}})$$

This is the parabolic/elliptic analogue of the Prodi-Serrin regularity criterion.

**Reference:**
- Serrin, J. (1962). On the interior regularity of weak solutions of the Navier-Stokes equations. *Arch. Rational Mech. Anal.*, 9, 187-195.
- Prodi, G. (1959). Un teorema di unicità per le equazioni di Navier-Stokes. *Ann. Mat. Pura Appl.*, 48, 173-182.

### Step 3: Difference Current and Energy Estimate

**Difference:** Define $S := T_w - T_s$ as a current. Note that $S$ is not necessarily an integral current (may have cancellations).

**Flat Norm Control:** The flat norm satisfies:
$$\mathbb{F}(S) \leq \mathbf{M}(T_w - T_s) = \int_{\Sigma_w \triangle \Sigma_s} d\mathcal{H}^k$$

where $\Sigma_w, \Sigma_s$ are the supports (with multiplicities).

**Energy Difference:** Define:
$$E(t) := \int_{\text{spt}(T_w) \cap \text{spt}(T_s)} \text{dist}(T_w, T_s)^2 \, d\mathcal{H}^k$$

This measures the "squared distance" between the two currents.

### Step 4: Gronwall Estimate

**Evolution of Energy Difference:** By the variational structure:
$$\frac{d}{dt} E(t) \leq C \|H_{T_s}\|_{L^r}^p \cdot E(t)$$

where $C$ depends on the geometry and the Serrin exponents.

**Derivation:** The key is the bilinear structure of the mean curvature equation:
$$\partial_t T - \Delta_T T = \text{nonlinear terms}$$

Testing the difference equation against $S = T_w - T_s$:
$$\frac{1}{2}\frac{d}{dt}\|S\|^2 + \|\nabla S\|^2 \leq |\langle \text{nonlinear}(T_w) - \text{nonlinear}(T_s), S \rangle|$$

The RHS is controlled by:
$$\leq C\|H_{T_s}\|_{L^r}^{p-1} \|S\| \cdot \|S\| \leq C\|H_{T_s}\|_{L^r}^p \|S\|^2$$

using Hölder and interpolation.

### Step 5: Gronwall Conclusion

**Gronwall's Lemma:** With $E(0) = 0$ (same initial data):
$$E(t) \leq E(0) \cdot \exp\left( C \int_0^t \|H_{T_s}\|_{L^r}^p \, ds \right) = 0$$

for all $t \in [0, T]$.

**Reference:** Gronwall, T. H. (1919). Note on the derivatives with respect to a parameter of the solutions of a system of differential equations. *Ann. of Math.*, 20, 292-296.

### Step 6: Current Equality

**Zero Energy $\Rightarrow$ Equality:** $E(t) = 0$ implies:
$$\text{dist}(T_w, T_s) = 0 \quad \mathcal{H}^k\text{-a.e.}$$

on the common support. By the structure of integral currents:
$$T_w = T_s \quad \text{as currents}$$

**Multiplicity Agreement:** Since both are integral currents with the same boundary and same support, their multiplicities agree by the constancy theorem.

**Reference:** Federer, H. (1969). *Geometric Measure Theory*. Springer. [Section 4.1.7: Constancy Theorem]

### Step 7: Serrin Class Characterization

**Prodi-Serrin Class for Currents:** The weak-strong uniqueness holds when the strong solution belongs to:
$$T_s \in L^p([0, T]; W^{1,r}(\Omega; \mathbf{I}_k))$$

with $\frac{k}{p} + \frac{n-k}{r} \leq 1$, $r > n - k$, $p \geq 2$.

**Critical Case:** At the endpoint $\frac{k}{p} + \frac{n-k}{r} = 1$, additional regularity (BMO or Besov) is needed.

**Reference:**
- Escauriaza, L., Seregin, G., Šverák, V. (2003). $L_{3,\infty}$-solutions of Navier-Stokes equations and backward uniqueness. *Russian Math. Surveys*, 58, 211-250.
- Seregin, G. (2012). *Lecture Notes on Regularity Theory for the Navier-Stokes Equations*. World Scientific.

## Key GMT Inequalities Used

1. **Energy Inequality for Weak Solutions:**
   $$\mathbf{M}(T(t)) + \int_0^t \int |H|^2 \, d\|T\| \leq \mathbf{M}(T(0))$$

2. **Serrin Integrability:**
   $$\int_0^T \|H_{T_s}\|_{L^r}^p \, dt < \infty, \quad \frac{k}{p} + \frac{n-k}{r} < 1$$

3. **Gronwall's Inequality:**
   $$E'(t) \leq f(t) E(t) \implies E(t) \leq E(0) e^{\int_0^t f(s) ds}$$

4. **Interpolation (Gagliardo-Nirenberg):**
   $$\|u\|_{L^r} \leq C\|u\|_{L^2}^{1-\theta}\|\nabla u\|_{L^2}^\theta$$

## Literature References

- Lions, J.-L. (1969). *Quelques Méthodes de Résolution des Problèmes aux Limites Non Linéaires*. Dunod.
- Serrin, J. (1962). On the interior regularity of weak solutions of the Navier-Stokes equations. *Arch. Rational Mech. Anal.*, 9, 187-195.
- Prodi, G. (1959). Un teorema di unicità per le equazioni di Navier-Stokes. *Ann. Mat. Pura Appl.*, 48, 173-182.
- Leray, J. (1934). Sur le mouvement d'un liquide visqueux emplissant l'espace. *Acta Math.*, 63, 193-248.
- Federer, H. (1969). *Geometric Measure Theory*. Springer.
- Simon, L. (1983). *Lectures on Geometric Measure Theory*. ANU.
