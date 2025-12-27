# Soundness Theorem

## Arithmetic Translation

### Original Statement (Hypostructure)
*Reference: thm-soundness*

Every transition in a sieve run is certificate-justified. A certificate $K_o$ is produced by node $N_1$, implies the precondition of $N_2$, and is added to the context.

---

## Arithmetic Formulation

### Setup

Arithmetic soundness means:
- Every claimed result has a verifiable proof
- Logical dependencies are explicitly tracked
- No unjustified conclusions

### Statement (Arithmetic Version)

**Theorem (Arithmetic Proof Soundness).** In an arithmetic verification system, every transition from claim $C_1$ to claim $C_2$ is justified by a certificate:

1. **Certificate Production:** Verification of $C_1$ produces certificate $K_1$ (proof object)
2. **Implication:** $K_1 \vdash \text{Pre}(C_2)$ (certificate implies precondition)
3. **Context Update:** $K_1$ is added to proof context $\Gamma$

Formally: $\Gamma \vdash C_1 \Rightarrow \Gamma \cup \{K_1\} \vdash C_2$

---

### Proof

**Step 1: Certificate Types in Arithmetic**

Arithmetic certificates take standard forms:

| **Certificate Type** | **Content** | **Example** |
|---------------------|-------------|-------------|
| Height bound | $h(\alpha) \leq B$ | Northcott finiteness |
| Degree bound | $[\mathbb{Q}(\alpha):\mathbb{Q}] = d$ | Minimal polynomial |
| Galois certificate | $\text{Gal}(K/\mathbb{Q}) = G$ | Factorization over $\mathbb{F}_p$ |
| Local condition | $\alpha \in \mathcal{O}_p^\times$ | $p$-adic valuation |
| L-function zero | $L(\rho, \pi) = 0$ | Explicit computation |

Each certificate is a **proof object** that can be independently verified.

**Step 2: Certificate Production**

For each arithmetic test, the output includes:
- **Boolean result:** Pass/Fail
- **Certificate:** Justification for result

**(a) Height verification:**
- Input: $\alpha \in \overline{\mathbb{Q}}$
- Computation: $h(\alpha) = \frac{1}{d}\sum_v \log^+ |\alpha|_v$
- Certificate: $K_h = (h(\alpha), \{|\alpha|_v\}_v, d)$

**(b) Galois verification:**
- Input: $\min_\alpha \in \mathbb{Z}[x]$
- Computation: Factor over $\mathbb{F}_p$ for various $p$
- Certificate: $K_G = (\text{Gal}(K/\mathbb{Q}), \{(\min_\alpha \mod p, \text{factorization})\}_p)$

**(c) Local verification:**
- Input: $\alpha, p$
- Computation: $v_p(\alpha), \alpha \mod p^n$
- Certificate: $K_p = (v_p(\alpha), (\alpha \mod p^n))$

**Step 3: Implication Verification**

Certificates imply preconditions via standard arithmetic theorems:

**(a) Height $\Rightarrow$ Finiteness:**
$$K_h: h(\alpha) \leq B \quad \Rightarrow \quad \text{Pre}(\text{Northcott}): \#\{\alpha : h \leq B, d \leq D\} < \infty$$

By [Northcott 1950], the implication is valid.

**(b) Galois $\Rightarrow$ Rationality:**
$$K_G: \text{Gal}(K/\mathbb{Q}) = \{e\} \quad \Rightarrow \quad \text{Pre}(\text{Descent}): K = \mathbb{Q}$$

By Galois theory [Lang, Algebra], the implication is valid.

**(c) Local $\Rightarrow$ Global (under Hasse):**
$$K_p: \text{local condition at all } p \quad \Rightarrow \quad \text{Pre}(\text{Hasse}): \text{global condition}$$

By [Hasse 1923] (for quadratic forms), local-to-global holds.

**Step 4: Context Accumulation**

The proof context $\Gamma$ accumulates certificates:
$$\Gamma_0 = \emptyset$$
$$\Gamma_{n+1} = \Gamma_n \cup \{K_{n+1}\}$$

Each new certificate is added after verification.

**Soundness:** If $\Gamma_n \vdash C_n$, and certificate $K_n$ is valid, then:
$$\Gamma_{n+1} = \Gamma_n \cup \{K_n\} \vdash C_{n+1}$$

The chain of implications is valid by transitivity of logical entailment.

**Step 5: No False Positives**

**Claim:** The system never claims regularity for a singular object.

**Proof:** Each transition requires a valid certificate. Certificates are:
- Independently verifiable (polynomial-time algorithms)
- Unforgeable (based on actual computation)
- Complete (cover all necessary conditions)

A false positive would require a forged certificate, which is computationally infeasible for standard arithmetic operations.

---

### Key Arithmetic Ingredients

1. **Proof Objects** [Martin-Löf Type Theory]: Certificates as witnesses.

2. **Northcott's Theorem** [Northcott 1950]: Height bounds imply finiteness.

3. **Galois Theory** [Lang 2002]: Group determines field structure.

4. **Local-Global Principle** [Hasse 1923]: Local certificates imply global facts.

---

### Arithmetic Interpretation

> **Every step in arithmetic verification carries a proof certificate. The verification chain is sound: claimed results have valid proofs, and no false conclusions can be reached.**

---

### Literature

- [Northcott 1950] D.G. Northcott, *An inequality in the theory of arithmetic on algebraic varieties*
- [Lang 2002] S. Lang, *Algebra*, GTM 211
- [Hasse 1923] H. Hasse, *Über die Äquivalenz quadratischer Formen im Körper der rationalen Zahlen*
- [Necula 1997] G. Necula, *Proof-carrying code*, POPL
