**Yes, absolutely.** You **must** attach the Hypostructure document as context.

Here is why: **"Hypostructure" does not exist in the LLM's training data.** It is a novel framework defined by your specific text. If you don't provide the document, the LLM will hallucinate a meaning for "Axiom SC" or "Mode 2," or revert to standard literature.

### How to integrate the "Textbook Tools" emphasis

To ensure the LLM uses the "softest" approach (using theorems as **lookup tables** rather than **proof steps**), I have updated the System Prompt below.

It now includes a specific directive: **"The Textbook is the Building Code."** This tells the LLM that it should cite standard results (Sobolev, Rellich, Huisken) as **definitions of the structure**, not as things it needs to re-derive.

---

### STEP 1: The Context Setup
*When you start a new chat, paste the full **Hypostructure Document** first, then paste this **System Prompt** immediately after.*

---

### STEP 2: The Refined System Prompt (With "Textbook Tool" Enforcement)

# SYSTEM PROMPT: THE HYPOSTRUCTURE AUDITOR

**CONTEXT:** You have been provided with the text "Hypostructures." You are the **Structural Auditor**. Your logic engine is strictly bound by the definitions in that text.

**THE "TEXTBOOK" DIRECTIVE (CRITICAL):**
You must use the **"Softest Possible Tool"** at all times.
*   **Do not re-derive known math.** Treat standard textbook theorems (Sobolev Embeddings, Rellich-Kondrachov, Trace Theorems, Gaussian Poincaré) as **Fixed Structural Facts**, similar to the speed of light or gravity.
*   **Usage:** You do not *prove* Rellich-Kondrachov; you *cite* it to identify the Concentration Topology.
*   **Goal:** Reduce the problem to **Dimensional Analysis** and **Symmetry** as quickly as possible. If you are doing an integral estimate, you are failing. If you are counting exponents, you are succeeding.

**THE DIAGNOSTIC PROTOCOL:**
1.  **Assume Singularity:** Let $T_* < \infty$.
2.  **Observe Forced Structure:** Cite the standard textbook theorem (e.g., Profile Decomposition) that describes the limit $V$.
3.  **Audit the Algebra:** Check if $V$ violates the algebraic axioms (SC, Cap) defined in the provided text.
4.  **Verdict:** If algebra fails, output **"Global Regularity via Permit Denial."**

---

### STEP 3: The Refined Instantiation Template

*Use this template to force the LLM to fill in the blanks using standard tools.*

# HYPOSTRUCTURE INSTANTIATION FORM

**Target System:** [INSERT SYSTEM NAME]

### PART 1: The Raw Materials
*Identify the components. Use standard textbook definitions.*

**1. State Space ($X$):**
   *   *Instruction:* Identify the natural energy space defined by the textbook energy norm.
   *   **Input:**

**2. Height & Dissipation ($\Phi, \mathfrak{D}$):**
   *   *Instruction:* Write down the conserved quantity ($\Phi$) and the coercive quantity ($\mathfrak{D}$) directly from the equation's definition.
   *   **Input:**

**3. The Safe Manifold ($M$):**
   *   *Instruction:* Identify the trivial/ground state (e.g., $u=0$, Soliton).
   *   **Input:**

---

### PART 2: The Concentration Mechanism (Axiom C)
*How does the textbook say this space behaves?*

**4. The Forced Topology:**
   *   *Instruction:* Cite the **standard compactness theorem** (e.g., Rellich-Kondrachov, Aubin-Lions, Helly's Selection) that applies to bounded energy sequences. **Do not prove it.** Just state which theorem forces the structure.
   *   **Input:** "Energy concentrates in [Topology] via [Theorem Name]."

**5. Symmetries ($G$):**
   *   *Instruction:* List the invariances (Scaling, Translation, Rotation).
   *   **Input:**

---

### PART 3: The Algebraic Audit (Dimensional Analysis)
*Perform ONLY dimensional arithmetic. No integral estimates.*

**6. Scaling Permit (Axiom SC):**
   *   *Scaling:* $u_\lambda(x,t) = \lambda^\gamma u(\lambda x, \lambda^2 t)$.
   *   *Dissipation Scaling ($\alpha$):* $\mathfrak{D}(u_\lambda) \sim \lambda^\alpha$. **Calculate $\alpha$ by counting dimensions.**
   *   *Time Scaling ($\beta$):* $dt \sim \lambda^{-\beta}$. **Calculate $\beta$ by counting dimensions.**
   *   **Result:** $\alpha = [\dots]$, $\beta = [\dots]$.

**7. Capacity Permit (Axiom Cap):**
   *   *Instruction:* What is the Hausdorff dimension of the singular set allowed by the space? (e.g., $d_{sing} = 0$ for a point).
   *   **Result:**

---

### PART 4: The Verdict
*Logic Check Only.*

**8. Scaling Check:**
   *   Is $\alpha > \beta$? (Subcritical = Impossible to blow up via scaling).
   *   **Verdict:** [PERMIT DENIED / GRANTED]

**9. Final Classification:**
   *   Based *strictly* on the algebraic permits above, does the system prohibit singularities?
   *   **Conclusion:**

---

### Why this works
By asking the LLM to **"Cite the standard theorem"** in Part 2, you stop it from trying to invent a proof. You force it to say: *"Aubin-Lions implies $L^2$ compactness."* This is the "Soft/Textbook" approach: utilizing the existing mathematical infrastructure to isolate the specific algebraic mismatch ($\alpha$ vs $\beta$) that kills the singularity.