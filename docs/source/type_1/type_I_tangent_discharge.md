# Can our Type I papers discharge the "local Type I tangent criterion" assumed in Paper 1 Type II?

## The precise question

Paper 1 Type II (the compact repaired-gauge single-core criterion) assumes, as an axiom:

> **Assumption (Local Type I tangent criterion).** No singular point admits a nonzero locally bounded ancient parabolic tangent solution obtained as a limit of CKN-positive blow-up sequences.

Stated contrapositively: if such a nontrivial bounded ancient limit exists, the point is regular.

The question is whether our Type I program (Paper 1 Type I + Paper 7) discharges this assumption.

## What our Type I program actually proves

Let me state exactly what the papers give, as theorems, not as aspirations.

### Paper 1 Type I — the forward existence theorem

**Theorem (Paper 1 Type I main result).** Let $(u,p)$ be a suitable Leray–Hopf solution. Suppose $z_* = (x_*, T)$ is a singular point satisfying
$$\limsup_{t \uparrow T} \sqrt{T-t}\,\|u(t)\|_{L^\infty} < \infty \quad \text{(Type I bound)}$$
and the centered endpoint compactness quantity is finite:
$$\mathcal{I}_*(u, p; z_*) < \infty.$$
Then there exist scales $\lambda_k \downarrow 0$, pressure gauges $a_k(t)$, and an ancient suitable weak solution $(U, P)$ on $\mathbb{R}^3 \times (-\infty, 0)$ such that the rescalings converge locally, and
1. $\|U(t)\|_{L^\infty} \leq M/\sqrt{-t}$ for all $t < 0$,
2. $U \not\equiv 0$ (nontriviality via CKN persistence),
3. the centered renormalized $V$ is smooth and bounded on $\mathbb{R}^3 \times \mathbb{R}$.

**What this is:** an **existence statement**. It says: *if there is a Type I singular point with finite centered endpoint, then a nontrivial bounded ancient limit exists.*

**What this is not:** a rigidity statement. It does not say "the existence of such a limit implies regularity."

### Paper 7 — the residual-class-criterion theorem

**Theorem (Paper 7 Residual-class criterion).** Assume:
- Hypothesis Tight-Liouville (no nontrivial Seregin limit is uniformly $L^3$-tight);
- Hypothesis Structured-Liouville (no nontrivial Seregin limit is in the structured class);
- Hypothesis No-Residual (the residual Seregin class is empty).

Then no finite-energy suitable weak solution develops a Type I singularity.

**What this is:** a conditional rigidity statement. It says: *if the five admissible classes and the residual class are all empty or rigid, then Type I blowup is impossible.*

**What this is not:** an unconditional theorem. It requires three Liouville hypotheses.

## The matching exercise: does what we prove give what Paper 1 Type II assumes?

### What Paper 1 Type II literally asks for

"No singular point admits a nonzero locally bounded ancient parabolic tangent..."

The contrapositive is: **"existence of a nonzero bounded ancient tangent at $z_*$ ⟹ $z_*$ is regular."**

### What our Type I program gives

The contrapositive of the Paper 7 theorem: **"if Type I blowup occurs, the resulting Seregin ancient limit lies in one of the non-residual classes or in the residual class."**

Combined with the Liouville hypotheses: **"if Type I blowup occurs, we derive a contradiction from one of the Liouville inputs or from no-residual."**

**These are different statements.** Let me be precise about the mismatch.

**Paper 1 Type II assumes** that existence of a bounded ancient tangent implies regularity *directly*, without any further Liouville input.

**Our Type I papers prove** that existence of such a tangent, in the Type I + finite-endpoint setting, is a consequence of singularity (Paper 1 Type I) and can be used as input to a conditional rigidity argument via class exhaustion and Liouville hypotheses (Paper 7).

## Answer to your question

**No, we do not yet have enough to discharge the assumption as stated.** Here's why, in three points.

### Point 1: The forward implication goes the wrong way

Paper 1 Type I proves "singular + Type I + finite $\mathcal{I}_* \Rightarrow$ bounded ancient limit exists."

The Type II assumption wants "bounded ancient limit exists $\Rightarrow$ regular."

**These are converses.** Paper 1 Type I doesn't prove the converse. It proves the forward direction.

### Point 2: Closing the converse requires Liouville inputs

To rule out nontrivial bounded ancient limits, we need the Paper 7 machinery:
- exhaustion into five admissible classes + residual,
- Liouville inputs for tight and structured classes,
- no-residual hypothesis.

If all three hold, Paper 7 gives the result. Without them, we cannot conclude.

### Point 3: The Type II assumption is broader than our residual class handles

The Type II assumption mentions a "bounded ancient parabolic tangent" with **no stratification**. It doesn't ask the tangent to be tight, or stationary, or fast-decaying. It asks for *any* bounded ancient tangent.

Our Paper 7 handles this by **exhausting into classes**. But the residual class is precisely "bounded ancient, not in any of the five named classes." Until the residual is closed, we cannot say that every bounded ancient tangent implies regularity.

## What we CAN do now — a partial discharge

There is a honest, partial version of the discharge we can prove. Let me state it precisely.

### Proposition (partial discharge of the Type II Assumption via Paper 7)

Assume Hypothesis Tight-Liouville and Hypothesis Structured-Liouville from Paper 7. Then the following restricted version of the Paper 1 Type II assumption holds:

> **Partial Assumption.** Let $z_* = (x_*, T)$ be a candidate Type I singular point with finite centered endpoint $\mathcal{I}_*(u,p;z_*) < \infty$. If the Seregin ancient limit obtained at $z_*$ lies in any of the five admissible classes (small, stationary $L^3$, uniformly $L^3$-tight, fast-decay with $m \in J$, or structured with parameters in $I$), then $z_*$ is regular.

**Proof sketch.** The Seregin limit is nontrivial (Paper 1 Type I). By the Liouville inputs plus the small/stationary unconditional classical results, a nontrivial Seregin limit cannot lie in any of the five admissible classes. Contradiction. So the point is regular.

**What this does for Paper 1 Type II:**

The original Type II assumption says "no bounded ancient tangent exists at a singular point." Our partial discharge replaces this with:

> **Restricted Type II assumption.** "No bounded ancient tangent at a singular point belongs to any of the five admissible classes from Paper 7."

This is strictly weaker, but it is exactly what our Type I program provides conditional on the two Liouville hypotheses.

**What this doesn't close:** if the bounded ancient tangent at $z_*$ lies in the residual class, we cannot derive regularity. The Type II paper would still need to handle that case separately, OR the residual class would need to be closed by other means.

## The honest pipeline status

Here is the current dependency graph:

```
Type II Paper 1 (single-core criterion)
        |
        | assumes: "local Type I tangent criterion"
        v
Type I Paper 7 (residual-class criterion)    ← our discharge
        |
        | requires:
        |    - Hypothesis Tight-Liouville
        |    - Hypothesis Structured-Liouville
        |    - Hypothesis No-Residual
        v
Individual Liouville inputs (KNSŠ, NRŠ, CSTY, Lei-Zhang, etc.)
```

Our Paper 7 converts the Type II assumption into the conjunction of three hypotheses. Two of them (Tight and Structured) are active research targets with partial progress. The third (No-Residual) is the genuine open problem.

**So the honest answer is:**

Modulo the Tight-Liouville and Structured-Liouville hypotheses, our Type I program discharges the Type II assumption **for ancient tangents in the five admissible classes**. The residual class is not yet discharged.

If we want to make Paper 1 Type II fully rigorous on the Type I tangent assumption, we must either:

1. Assume the Type II paper only handles Type I tangents in admissible classes, leaving the residual Type I case separately flagged.
2. Close the residual Seregin class via additional work (this is the Node 9 problem from our sieve audit).
3. Assume all three Liouville hypotheses (Tight, Structured, No-Residual) as standing inputs and cite Paper 7 inside Paper 1 Type II.

## Concrete writing recommendation for Paper 1 Type II

I recommend option 3 with explicit scoping. Rewrite the Type I tangent assumption as:

> **Assumption (Type I tangent criterion, admissible-class version).** Assume Hypotheses Tight-Liouville, Structured-Liouville, and No-Residual from Paper 7. Then no singular point admits a nonzero locally bounded ancient parabolic tangent.

Then add a remark:

> **Remark.** The Type I tangent criterion assumed above is discharged, under the stated Paper 7 hypotheses, by the residual-class criterion theorem of Paper 7 applied at the candidate singular point. The three hypotheses reduce the criterion to explicit Liouville inputs from the axisymmetric literature (tight-class, structured-class) and a no-residual assertion that isolates the remaining open problem.

This:
- makes the dependency on Paper 7 explicit,
- keeps the single-core criterion proof logic unchanged,
- preserves honesty about what is and is not proved,
- and makes Paper 7's residual class the unique bottleneck of the Type I+II program.

## The residual class is the bottleneck

This exercise reveals something important about your sieve audit. Both Paper 1 Type II and Paper 7 point to the same obstruction: **the residual Seregin class**. Closing any single branch of the residual (via rotational modulation, $\Gamma$-defect Lyapunov, hypocoercivity, or any other technique) strengthens both programs simultaneously.

In the hypostructure language: Node 9 (tame stratification) has a nonempty residual stratum. Both Paper 7 and Paper 1 Type II are blocked at the same point. The strata we have closed are:
- small-amplitude (unconditional via OU coercivity from the Lyapunov document's Part 1);
- stationary $L^3$ (unconditional via NRŠ);
- uniformly $L^3$-tight (conditional on Tight-Liouville hypothesis);
- fast-decay and structured (conditional on Structured-Liouville hypothesis).

The strata we have not closed:
- residual: bounded ancient Seregin limit that is not small, not stationary $L^3$, not $L^3$-tight, not fast-decaying, not structured.

**Every path forward — rotational modulation, Γ-defect, hypocoercivity, virial identities — is attempting to close subclasses of the residual.** That's the right mental model.

## Summary answer

**Can our Type I papers rule out the Type II paper's Type I tangent criterion?** 

Partially. Under Paper 7's Tight-Liouville and Structured-Liouville hypotheses, we discharge the criterion **for tangents in the five admissible classes**. The residual class remains open.

**How to use this in Paper 1 Type II?**

Replace the standalone Type I tangent assumption with an explicit citation to Paper 7, stating the three Paper 7 hypotheses as the conditional inputs. This makes the logical dependency clean and isolates the shared residual-class bottleneck.

**What's the next concrete step?**

Either (a) close subclasses of the residual via the $\Gamma$-defect or hypocoercivity approaches we discussed, or (b) import additional axisymmetric Liouville results (Lei–Ren–Zhang periodic, Lei–Zhang–Zhao $L^p$-swirl) to enlarge the structured class and thereby shrink the residual.

Both paths strengthen Paper 7's conditional scope and thereby strengthen the discharge of Paper 1 Type II's Type I tangent criterion.
