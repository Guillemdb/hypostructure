# LOCK-Virtual: Virtual Cycle Correspondence Lock — GMT Translation

## Original Statement (Hypostructure)

The virtual cycle correspondence shows that virtual fundamental classes provide well-defined invariants even for singular or obstructed moduli spaces, locking the count against deformation.

## GMT Setting

**Virtual Cycle:** Correction to fundamental class accounting for obstructions

**Correspondence:** Virtual class ↔ expected count

**Lock:** Virtual invariants unchanged under deformation

## GMT Statement

**Theorem (Virtual Cycle Correspondence Lock).** For moduli space $\mathcal{M}$ with obstruction theory:

1. **Virtual Class:** $[\mathcal{M}]^{\text{vir}} \in H_*(\mathcal{M})$ of expected dimension

2. **Correspondence:** Integrals over $[\mathcal{M}]^{\text{vir}}$ give deformation-invariant counts

3. **Lock:** Virtual invariants constant in families

## Proof Sketch

### Step 1: Perfect Obstruction Theory

**Definition:** A perfect obstruction theory on $\mathcal{M}$ is morphism:
$$\phi: E^\bullet \to \mathbb{L}_{\mathcal{M}}$$

where $E^\bullet$ is 2-term complex of vector bundles and $\mathbb{L}_{\mathcal{M}}$ is cotangent complex.

**Reference:** Behrend, K., Fantechi, B. (1997). The intrinsic normal cone. *Invent. Math.*, 128, 45-88.

### Step 2: Virtual Fundamental Class

**Construction:** Given obstruction theory, construct:
$$[\mathcal{M}]^{\text{vir}} \in A_{\text{vd}}(\mathcal{M})$$

where $\text{vd} = \text{rk}(E^0) - \text{rk}(E^1)$ is virtual dimension.

**Reference:** Li, J., Tian, G. (1998). Virtual moduli cycles and Gromov-Witten invariants of algebraic varieties. *J. Amer. Math. Soc.*, 11, 119-174.

### Step 3: Intrinsic Normal Cone

**Normal Cone:** For embedding $\mathcal{M} \subset \mathcal{N}$:
$$C_{\mathcal{M}/\mathcal{N}} = \text{Spec}_{\mathcal{M}}(\bigoplus_k I^k/I^{k+1})$$

**Intrinsic:** $\mathfrak{C}_{\mathcal{M}} \subset h^1/h^0(E^\bullet)$ is independent of embedding.

**Virtual Class:** $[\mathcal{M}]^{\text{vir}} = 0^![\mathfrak{C}_{\mathcal{M}}]$ where $0^!$ is refined Gysin.

### Step 4: Deformation Invariance

**Theorem:** If $\mathcal{M}_t$ varies in family with compatible obstruction theory:
$$[\mathcal{M}_t]^{\text{vir}} \text{ is constant in homology}$$

**Reference:** Behrend, K. (1997). Gromov-Witten invariants in algebraic geometry. *Invent. Math.*, 127, 601-617.

### Step 5: GMT Interpretation

**Moduli as Current:** Moduli space $\mathcal{M}$ defines current:
$$[\mathcal{M}] \in \mathbf{I}_{\dim \mathcal{M}}(\text{ambient})$$

**Virtual Correction:** When $\mathcal{M}$ is singular/obstructed, virtual class corrects dimension:
$$\dim[\mathcal{M}]^{\text{vir}} = \text{expected dimension}$$

### Step 6: Integration and Invariants

**Virtual Integral:** For class $\alpha \in H^*(\mathcal{M})$:
$$\int_{[\mathcal{M}]^{\text{vir}}} \alpha$$

is virtual invariant.

**Gromov-Witten:** GW invariants defined as virtual integrals over $\overline{\mathcal{M}}_{g,n}(X, \beta)$.

**Reference:** Kontsevich, M., Manin, Yu. (1994). Gromov-Witten classes, quantum cohomology, and enumerative geometry. *Comm. Math. Phys.*, 164, 525-562.

### Step 7: Donaldson-Thomas Theory

**DT Invariants:** Virtual counts of sheaves on Calabi-Yau 3-folds:
$$DT_\beta(X) = \int_{[\mathcal{M}_\beta(X)]^{\text{vir}}} 1$$

**Reference:** Thomas, R. P. (2000). A holomorphic Casson invariant for Calabi-Yau 3-folds. *J. Differential Geom.*, 54, 367-438.

**Lock:** DT invariants independent of complex structure moduli.

### Step 8: Comparison with Euler Characteristic

**Behrend Function:** For scheme $\mathcal{M}$:
$$\nu_{\mathcal{M}}: \mathcal{M} \to \mathbb{Z}$$

constructible function.

**Weighted Count:**
$$\chi(\mathcal{M}, \nu_{\mathcal{M}}) = \int_{[\mathcal{M}]^{\text{vir}}} 1$$

**Reference:** Behrend, K. (2009). Donaldson-Thomas type invariants via microlocal geometry. *Ann. of Math.*, 170, 1307-1338.

### Step 9: Virtual Localization

**Theorem (Graber-Pandharipande):** For $\mathbb{C}^*$-action on $\mathcal{M}$:
$$[\mathcal{M}]^{\text{vir}} = \sum_F \frac{[\mathcal{M}^F]^{\text{vir}}}{e(N^{\text{vir}}_F)}$$

**Reference:** Graber, T., Pandharipande, R. (1999). Localization of virtual classes. *Invent. Math.*, 135, 487-518.

**Computation:** Virtual class computable via fixed loci.

### Step 10: Compilation Theorem

**Theorem (Virtual Cycle Correspondence Lock):**

1. **Virtual Class:** $[\mathcal{M}]^{\text{vir}}$ from perfect obstruction theory

2. **Expected Dimension:** Virtual dimension matches expected

3. **Invariance:** Virtual integrals constant in families

4. **Lock:** Virtual counts protected against deformation

**Applications:**
- Gromov-Witten invariants
- Donaldson-Thomas invariants
- Enumerative geometry

## Key GMT Inequalities Used

1. **Virtual Dimension:**
   $$\text{vd} = \text{rk}(E^0) - \text{rk}(E^1)$$

2. **Deformation Invariance:**
   $$\int_{[\mathcal{M}_t]^{\text{vir}}} \alpha = \text{const}$$

3. **Localization:**
   $$[\mathcal{M}]^{\text{vir}} = \sum_F [\mathcal{M}^F]^{\text{vir}}/e(N^{\text{vir}})$$

4. **Behrend:**
   $$\chi(\mathcal{M}, \nu) = \deg[\mathcal{M}]^{\text{vir}}$$

## Literature References

- Behrend, K., Fantechi, B. (1997). Intrinsic normal cone. *Invent. Math.*, 128.
- Li, J., Tian, G. (1998). Virtual moduli cycles. *J. Amer. Math. Soc.*, 11.
- Behrend, K. (1997). Gromov-Witten invariants. *Invent. Math.*, 127.
- Kontsevich, M., Manin, Yu. (1994). Gromov-Witten classes. *Comm. Math. Phys.*, 164.
- Thomas, R. P. (2000). Holomorphic Casson invariant. *J. Differential Geom.*, 54.
- Behrend, K. (2009). DT via microlocal geometry. *Ann. of Math.*, 170.
- Graber, T., Pandharipande, R. (1999). Localization of virtual classes. *Invent. Math.*, 135.
