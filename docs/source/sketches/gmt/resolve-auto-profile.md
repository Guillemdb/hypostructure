# RESOLVE-AutoProfile: Automatic Profile Classification — GMT Translation

## Original Statement (Hypostructure)

Profile classification is automatically computed via a multi-mechanism dispatcher (CC+Rigidity, Attractor+Morse, Tame+LS, Lock/Exclusion). The mechanism is transparent — only the certificate matters.

## GMT Setting

**Input:** Blow-up sequence $\{T_j\}$ with soft certificates $K_{D_E}^+, K_{C_\mu}^+, K_{\text{SC}_\lambda}^+, K_{\text{LS}_\sigma}^+$

**Output:** Profile certificate $K_{\text{prof}}^+ = (T_\infty, \text{class}, \text{mechanism tag})$

**Mechanisms:**
- A: Concentration-Compactness + Rigidity (Lions-Kenig-Merle)
- B: Global Attractor + Morse Decomposition (Temam-Raugel)
- C: O-Minimal + Łojasiewicz-Simon (van den Dries-Simon)
- D: Lock/Hom-Exclusion (categorical)

## GMT Statement

**Theorem (Automatic Profile Classification).** Given soft certificates, the profile trichotomy is automatically computed by the following dispatch:

```
IF MechA applies THEN output K_prof^+ (CC-Rig)
ELSE IF MechB applies THEN output K_prof^+ (Attr-Morse)
ELSE IF MechC applies THEN output K_prof^+ (Tame-LS)
ELSE IF MechD applies THEN output K_prof^+ (Lock-Excl)
ELSE output K_prof^- (mechanism exhausted)
```

Each mechanism is **sound**: if it produces $K_{\text{prof}}^+$, the classification is correct.

## Proof Sketch

### Step 1: Mechanism A — Concentration-Compactness + Rigidity

**Prerequisites:** $K_{D_E}^+ \land K_{C_\mu}^+ \land K_{\text{SC}_\lambda}^+ \land K_{\text{Mon}_\phi}^+ \land K_{\text{Rep}_K}^+$

**Lions' Profile Decomposition (1984):** For bounded sequences $\{u_n\} \subset \dot{H}^s(\mathbb{R}^n)$:
$$u_n = \sum_{j=1}^J V^j_{n} + w_n^J$$

where:
- $V^j_n(x) = \frac{1}{\lambda_n^{(n-2s)/2}} V^j\left(\frac{x - x_n^j}{\lambda_n^j}\right)$ — profiles
- $\|w_n^J\|_{L^p} \to 0$ as $J \to \infty$ then $n \to \infty$ — remainder

**Reference:** Lions, P.-L. (1984). Concentration-compactness. *Ann. Inst. H. Poincaré*, 1.

**Kenig-Merle Rigidity (2006):** If the critical element $u^*$ satisfies $E(u^*) = E_c$ (critical energy), then:
$$u^* \in \{0, W, \text{traveling waves}\}$$

where $W$ is the ground state.

**Reference:** Kenig, C., Merle, F. (2006). Global well-posedness for energy-critical NLS. *Acta Math.*, 201, 147-212.

**Classification Output:**
- If profiles finite: Case 1 (Library)
- If profiles form continuous family with rigidity: Case 2 (Tame)

### Step 2: Mechanism B — Global Attractor + Morse

**Prerequisites:** $K_{D_E}^+ \land K_{C_\mu}^+ \land K_{\text{TB}_\pi}^+$

**Global Attractor (Temam, 1988):** For dissipative systems:
$$\mathcal{A} = \omega(B_0) = \bigcap_{t \geq 0} \overline{\bigcup_{s \geq t} S_s(B_0)}$$

where $B_0$ is the absorbing set.

**Reference:** Temam, R. (1988). *Infinite-Dimensional Dynamical Systems in Mechanics and Physics*. Springer.

**Morse Decomposition (Conley, 1978):** The attractor admits:
$$\mathcal{A} = M_1 \cup \cdots \cup M_k \cup C(\{M_i\})$$

where $M_i$ are Morse sets (maximal invariant sets) and $C(\cdot)$ are connecting orbits.

**Reference:** Conley, C. (1978). *Isolated Invariant Sets and the Morse Index*. CBMS 38, AMS.

**Classification Output:**
- Morse sets = profiles in library
- Connecting orbits = tame family
- Chaotic attractors = wild (rare under dissipation)

### Step 3: Mechanism C — O-Minimal + Łojasiewicz-Simon

**Prerequisites:** $K_{D_E}^+ \land K_{\text{LS}_\sigma}^+ \land K_{\text{TB}_O}^+$

**O-Minimal Definability:** The energy functional $\Phi$ is definable in an o-minimal structure (e.g., $\mathbb{R}_{\text{an}}$, $\mathbb{R}_{\text{exp}}$).

**Kurdyka-Łojasiewicz Inequality (1998):** For definable functions:
$$|\nabla(\psi \circ \Phi)|(x) \geq 1$$

on $\{a < \Phi < b\} \setminus \text{Crit}(\Phi)$, where $\psi$ is a desingularizing function.

**Reference:** Kurdyka, K. (1998). On gradients of functions definable in o-minimal structures. *Ann. Inst. Fourier*, 48, 769-783.

**Classification Output:**
- Definable critical set = finite union of smooth strata
- Each stratum is a tame family
- No wild profiles possible in o-minimal setting

### Step 4: Mechanism D — Lock/Hom-Exclusion

**Prerequisites:** $K_{\text{Cat}_{\text{Hom}}}^{\text{blk}}$ (Lock blocked)

**Categorical Obstruction:** If $\text{Hom}(\mathbb{H}_{\text{bad}}, \mathbb{H}(T)) = \emptyset$, then no singular profile can exist in $T$.

**Forced Library Membership:** The absence of bad morphisms forces:
$$T_\infty \in \mathcal{L} \cup \{0\}$$

Any limiting profile must be either zero (dispersion) or belong to the canonical library (regular limit).

**Classification Output:**
- Lock blocked $\Rightarrow$ only library profiles
- Wild profiles excluded categorically

### Step 5: Dispatcher Logic and Termination

**Dispatcher Algorithm:**
```python
def classify_profile(soft_certs):
    if MechA.preconditions_met(soft_certs):
        result = MechA.run()
        if result.success:
            return K_prof_plus(result.profile, "CC-Rig")

    if MechB.preconditions_met(soft_certs):
        result = MechB.run()
        if result.success:
            return K_prof_plus(result.profile, "Attr-Morse")

    if MechC.preconditions_met(soft_certs):
        result = MechC.run()
        if result.success:
            return K_prof_plus(result.profile, "Tame-LS")

    if MechD.preconditions_met(soft_certs):
        result = MechD.run()
        if result.success:
            return K_prof_plus(result.profile, "Lock-Excl")

    return K_prof_minus(mechanism_failures=[A,B,C,D])
```

**Termination:** Each mechanism has finite runtime:
- MechA: Finite iteration of profile extraction (energy bound)
- MechB: Finite Morse decomposition (Conley index computation)
- MechC: Finite cell decomposition (o-minimal)
- MechD: Finite tactic enumeration (E1-E10)

### Step 6: Soundness of Multi-Mechanism OR

**Soundness Theorem:** If ANY mechanism produces $K_{\text{prof}}^+$, the classification is valid.

*Proof:* Each mechanism's soundness is established independently:
- MechA: Lions (1984), Kenig-Merle (2006)
- MechB: Temam (1988), Conley (1978)
- MechC: van den Dries (1998), Kurdyka (1998)
- MechD: Federer (1969), categorical completeness

The disjunction inherits soundness from components.

### Step 7: Downstream Independence

**Certificate Interface:** All downstream theorems depend only on $K_{\text{prof}}^+$:
```
K_prof^+ = (T_∞, classification ∈ {Library, Tame, Wild}, mechanism_tag)
```

**Tag Opacity:** The mechanism tag is metadata only. No downstream proof examines which mechanism fired.

**Consequence:** The multi-mechanism approach is **modular** — adding new mechanisms (MechE, etc.) does not affect existing proofs.

## Key GMT Inequalities Used

1. **Lions Profile Decomposition:**
   $$u_n = \sum_j V^j_n + w_n, \quad \|w_n\|_{L^p} \to 0$$

2. **Kenig-Merle Rigidity:**
   $$E(u^*) = E_c \implies u^* \in \{0, W, \text{solitons}\}$$

3. **Kurdyka-Łojasiewicz:**
   $$|\nabla(\psi \circ \Phi)| \geq 1 \text{ on } \{a < \Phi < b\} \setminus \text{Crit}$$

4. **Conley Index:**
   $$h(S) \in \{\text{homotopy types}\} \text{ classifies isolated invariant sets}$$

## Literature References

- Lions, P.-L. (1984). Concentration-compactness I & II. *Ann. Inst. H. Poincaré*, 1.
- Kenig, C., Merle, F. (2006). Global well-posedness for energy-critical NLS. *Acta Math.*, 201.
- Temam, R. (1988). *Infinite-Dimensional Dynamical Systems*. Springer.
- Conley, C. (1978). *Isolated Invariant Sets and the Morse Index*. AMS.
- van den Dries, L. (1998). *Tame Topology and O-minimal Structures*. Cambridge.
- Kurdyka, K. (1998). On gradients of functions definable in o-minimal structures. *Ann. Inst. Fourier*.
- Bahouri, H., Gérard, P. (1999). High frequency approximation. *Amer. J. Math.*, 121.
