# FACT-SoftKM: Soft→KM Compilation — GMT Translation

## Original Statement (Hypostructure)

The soft permits compile to a Kenig-Merle type rigidity theorem: at critical energy, solutions are classified into a finite list of special solutions.

## GMT Setting

**Critical Energy:** $E_c$ — threshold energy for global behavior change

**Ground State:** $W \in \mathbf{I}_k(M)$ — optimizer for Sobolev inequality

**Rigidity:** At $E_c$, solutions are exactly $\{0, W, \text{solitons}\}$

## GMT Statement

**Theorem (Soft→KM Compilation).** Under soft permits, the critical element $T^* \in \mathbf{I}_k(M)$ with $\Phi(T^*) = \Phi_c$ belongs to:

$$T^* \in \{0\} \cup \{g \cdot W : g \in G\} \cup \{\text{traveling waves}\}$$

where:
- $G$ is the symmetry group (translations, rotations, scaling)
- $W$ is the unique (up to $G$) minimizer of $\Phi$ among non-zero elements
- Traveling waves satisfy $T(t, x) = W(x - vt)$ for some velocity $v$

## Proof Sketch

### Step 1: Threshold Dichotomy

**Kenig-Merle Framework (2006):** Define:
$$E_c := \sup\{E : \Phi(T_0) < E \implies T_t \text{ exists globally and scatters}\}$$

**Dichotomy:** For $\Phi(T_0) < E_c$: global existence + scattering. For $\Phi(T_0) > E_c$: may blow up.

**Reference:** Kenig, C., Merle, F. (2006). Global well-posedness, scattering and blow-up for the energy-critical focusing non-linear wave equation. *Acta Math.*, 201, 147-212.

### Step 2: Critical Element Existence

**Use of $K_{C_\mu}^+$:** At energy $E_c$, compactness provides a critical element.

**Theorem (Critical Element):** There exists $T^* \in \mathbf{I}_k(M)$ with $\Phi(T^*) = E_c$ such that:
- $T^*$ does not scatter forward in time, or
- $T^*$ does not scatter backward in time

**Proof:** Take sequence $T_j$ with $\Phi(T_j) \to E_c^-$ that fails to scatter. By profile decomposition (FACT-SoftProfDec), extract limiting profile $T^*$.

### Step 3: Compactness of Critical Orbit

**Use of $K_{D_E}^+ \land K_{C_\mu}^+$:** The orbit $\{T^*(t) : t \in \mathbb{R}\}$ is precompact.

**Proof:** If not precompact, profile decomposition gives multiple profiles at energy $< E_c$, which scatter. But then $T^*$ scatters, contradiction.

**Reference:** Killip, R., Visan, M. (2010). The focusing energy-critical nonlinear Schrödinger equation in dimensions five and higher. *Amer. J. Math.*, 132, 361-424.

### Step 4: Variational Characterization

**Ground State:** $W$ minimizes:
$$E_c = \inf\{\Phi(T) : T \neq 0, \, \nabla \Phi(T) = 0\}$$

**Pohozaev Identity:** Critical points satisfy:
$$\Phi(T) = \frac{1}{n} \int |\nabla T|^2 = \frac{1}{2} \int |T|^{2^*}$$

**Reference:** Pohozaev, S. I. (1965). Eigenfunctions of the equation $\Delta u + \lambda f(u) = 0$. *Soviet Math. Dokl.*, 6, 1408-1411.

**Optimal Constants (Talenti, 1976):** The optimizer $W$ for Sobolev:
$$\|u\|_{L^{2^*}} \leq C_S \|\nabla u\|_{L^2}$$

is explicit: $W(x) = (1 + |x|^2)^{-(n-2)/2}$ (up to scaling).

**Reference:** Talenti, G. (1976). Best constant in Sobolev inequality. *Ann. Mat. Pura Appl.*, 110, 353-372.

### Step 5: Rigidity via Concentration

**Use of $K_{\text{SC}_\lambda}^+$:** Scale coherence forces the critical element to be scale-invariant.

**Duyckaerts-Merle Rigidity (2008):** If $T^*$ has compact orbit at critical energy:
$$T^* = W \quad \text{or} \quad T^* = W^- \text{ (threshold solution)}$$

**Reference:** Duyckaerts, T., Merle, F. (2008). Dynamic of threshold solutions for energy-critical NLS. *GAFA*, 18, 1787-1840.

**GMT Version:** The critical current $T^*$ satisfies:
$$(\eta_{0,\lambda})_\# T^* = T^* \quad \text{for some } \lambda > 0$$

This scaling symmetry forces $T^*$ to be a cone, hence $T^* \in \mathcal{L} = \{W, \text{solitons}\}$.

### Step 6: Classification of Solitons

**Traveling Wave Ansatz:** $T(t,x) = W(x - vt)$ satisfies the flow iff:
$$-v \cdot \nabla W = -\nabla \Phi(W)$$

**Soliton ODE:** Reduces to:
$$\Delta W + W^{2^*-1} = v \cdot \nabla W$$

**Classification (Berestycki-Lions, 1983):** Finite-energy solitons are:
- Static: $v = 0$ gives $W$ (ground state)
- Moving: $v \neq 0$ gives Lorentz-boosted $W$

**Reference:** Berestycki, H., Lions, P.-L. (1983). Nonlinear scalar field equations I & II. *Arch. Rational Mech. Anal.*, 82, 313-375.

### Step 7: Uniqueness Up to Symmetry

**Symmetry Group:** $G = \text{Transl} \times \text{Rot} \times \text{Scale} \times \text{Phase}$

**Theorem:** Any critical element $T^*$ with $\Phi(T^*) = E_c$ satisfies:
$$T^* \in G \cdot W := \{g \cdot W : g \in G\}$$

**Proof:** By Steps 3-6:
1. $T^*$ has compact orbit (Step 3)
2. $T^*$ is critical point of $\Phi$ (Step 4)
3. $T^*$ is scale-invariant (Step 5)
4. Scale-invariant critical points are $G \cdot W$ (Step 6)

### Step 8: Compilation Theorem

**Theorem (Soft→KM):** The compilation:
$$(K_{D_E}^+, K_{C_\mu}^+, K_{\text{SC}_\lambda}^+, K_{\text{LS}_\sigma}^+) \to \text{KM-Rigidity}$$

produces:
- Critical energy $E_c$
- Ground state $W$
- Classification $T^* \in \{0, G \cdot W, \text{solitons}\}$
- Dichotomy: $\Phi < E_c \Rightarrow$ scatter; $\Phi = E_c \Rightarrow$ rigid; $\Phi > E_c \Rightarrow$ may blow up

**Reference:** Killip, R., Visan, M. (2013). Nonlinear Schrödinger equations at critical regularity. *Clay Math. Proc.*, 17.

### Step 9: GMT Examples

**Example 1: Area-Minimizing Cones**
- Critical energy: $E_c = $ area of Simons cone
- Ground state: $W = $ Simons cone $\{x_1^2 + \cdots + x_4^2 = x_5^2 + \cdots + x_8^2\}$
- Rigidity: Minimal cones with this energy are exactly Simons cone

**Reference:** Bombieri, E., De Giorgi, E., Giusti, E. (1969). Minimal cones and the Bernstein problem. *Invent. Math.*, 7, 243-268.

**Example 2: Harmonic Maps**
- Critical energy: $E_c = $ energy of harmonic sphere
- Ground state: $W = $ conformal map $S^2 \to S^2$
- Rigidity: Bubbles are classified

**Reference:** Sacks, J., Uhlenbeck, K. (1981). The existence of minimal immersions of 2-spheres. *Ann. of Math.*, 113, 1-24.

## Key GMT Inequalities Used

1. **Sobolev Embedding:**
   $$\|T\|_{L^{2^*}} \leq C_S \|\nabla T\|_{L^2}$$

2. **Pohozaev Identity:**
   $$\Phi(T) = \frac{1}{n}\|\nabla T\|_2^2 = \frac{1}{2}\|T\|_{2^*}^{2^*}$$

3. **Concentration-Compactness:**
   $$\Phi(T^*) = E_c \implies T^* \text{ has compact orbit}$$

4. **Rigidity:**
   $$T^* \in \{0, G \cdot W, \text{solitons}\}$$

## Literature References

- Kenig, C., Merle, F. (2006). Energy-critical focusing NLW. *Acta Math.*, 201.
- Duyckaerts, T., Merle, F. (2008). Threshold solutions. *GAFA*, 18.
- Talenti, G. (1976). Best constant in Sobolev inequality. *Ann. Mat. Pura Appl.*, 110.
- Berestycki, H., Lions, P.-L. (1983). Nonlinear scalar field equations. *Arch. Rational Mech. Anal.*, 82.
- Killip, R., Visan, M. (2010). Energy-critical NLS. *Amer. J. Math.*, 132.
- Bombieri, E., De Giorgi, E., Giusti, E. (1969). Minimal cones. *Invent. Math.*, 7.
