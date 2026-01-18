# **0. Introduction and Motivation**

We begin with the core insight:

> **If a decoder observes enough transitions of the form**
> [
> y = A,x,
> ]
> **then it can solve for the unknown matrix** (A).

Every observed pair ((x,y)) gives one linear constraint on (A).
With enough **linearly independent** constraints, the decoder simply solves a linear system.

This suggests a communication scheme:

1. **Encode payload as a matrix** (A).
2. **Transmit transitions** ((x,,y=A x)) through the channel.
3. **Decoder reconstructs** (A).

But this carries at most (n^2) bits for an (n\times n) matrix — far too small.

To achieve kilobit-scale payloads with tiny packet sizes (e.g., (n=7)), we inject a **feature map**

[
y = A,F(x),
]

where:

* (x \in {0,1}^n),
* (F(x) \in {0,1}^m),
* (A \in \mathbb{F}_2^{n\times m}),
* (m) may be as large as (2^n) (the number of ANF monomials).

This expands the payload size to:

[
\log_2 D = m \log_2(2^n - 2),
]

which for (n=7) gives thousands of bits.

But now several problems arise:

* What properties must (F(x)) satisfy?
* Which matrices (A) can be safely used?
* How do we ensure the decoder sees independent feature vectors?
* How do we map a payload index into a *good* matrix (A)?
* How do we guarantee recovery even if some transitions are erased?

This document explains the full design.

---

# **1. The Core Algebra: Decoding Means Reconstructing a Linear Map**

We transmit transitions obeying:

[
y = A,F(x).
]

Collect transitions:

[
(x_1,y_1), (x_2,y_2),\ldots
]

Build matrices:

[
X_{\text{obs}} = [F(x_1)\ \dots\ F(x_k)],
\qquad
Y_{\text{obs}} = [y_1\ \dots\ y_k].
]

If the decoder finds an (m\times m) invertible submatrix:

[
X^{(m)}_{\text{obs}} \in \mathbb{F}_2^{m\times m},
]

then it can compute:

[
A = Y^{(m)}*{\text{obs}} \left( X^{(m)}*{\text{obs}} \right)^{-1}.
]

Everything reduces to **linear algebra over GF(2)**.

From this, several constraints follow **strictly and inevitably**.

---

# **2. Why We Need a Feature Map**

If (A\in\mathbb{F}_2^{n\times m}), the payload is (nm) bits.
For (n=7) and (m=1000), this is only 7000 bits — decent, but we want much more.

Therefore we choose (m) as large as possible, up to (2^n), and define (F:{{0,1}^n}\to{0,1}^m) such that:

1. (\operatorname{rank}{F(x)} = m),
2. There exist (m) vectors (x_i) giving an invertible matrix (X=[F(x_1)\dots F(x_m)]),
3. These (m) feature vectors must appear in the generator dynamics.

The feature map must therefore be **full-rank, explicit, cheap to compute**, and **deterministic**.

---

# **3. Critical Requirements on (F(x))**

From the decoding equation:

[
A = Y X^{-1},
]

we require:

### 1. **Rank condition**

[
\operatorname{rank}{F(x): x\in{0,1}^n} = m.
]

### 2. **Existence of an invertible basis**

There must exist states (x_1,\dots,x_m) such that:

[
X = [F(x_1)\dots F(x_m)]
]

is invertible.

### 3. **Discoverability**

The generator (state traversal) must reach those (x_i).

### 4. **Linear structure**

Feature coordinates must be monomials.
This keeps the algebra linear in the coefficients of (A).

### 5. **Efficient computability**

ANF monomials satisfy all constraints.

---

# **4. Our Choice: ANF Monomials With Constant Term**

Feature coordinates are monomials:

[
x_{i_1} x_{i_2} \dots x_{i_k},
]

sorted by degree and lex order.

### Why monomials?

* They form a basis of all Boolean functions.
* Full rank.
* The evaluation matrix for all inputs is invertible (Walsh–Hadamard-like structure).
* Easy to compute.

### Why include the constant term?

* Guarantees (F(0)\neq 0).
* Stabilizes the basis construction.
* Ensures every input has a nonzero feature vector.

Including the constant term allows:

[
m \le 2^n.
]

---

# **5. Formal Constraints on (F)**

Given the structure (A = Y X^{-1}), we need:

1. Full rank: (\operatorname{rank}(F)=m),
2. Basis existence: (X) invertible,
3. Reachability: generator reaches chosen (x_i),
4. Diversity: generator doesn’t collapse onto a low-dimensional manifold,
5. Computational simplicity.

ANF monomials satisfy all of the above.

---

# **6. Other Feature Families Exist — We Focus on Monomials**

Other families:

* Walsh–Hadamard functions,
* Parity-check embeddings,
* Polynomial Boolean networks,
* Learned neural embeddings.

But ANF monomials strike the right balance:

* guaranteed rank,
* simple structure,
* easy to invert,
* reproducible.

(A future improvement: encode part of the payload in the *choice* of monomials.)

---

# **7. Channel Constraint: No Identical Consecutive Packets**

To transmit transitions ((x,y)), we must obey:

> **No two consecutive outputs may be equal.**

The efficient solution is:

[
x_{t+1} = y_t,
]

so we transmit:

[
x_1,, y_1,, y_1,, y_2,,y_3,\dots
]

in practice:

[
x_1,\ y_1,\ x_2=y_1,\ y_2,\ \dots
]

This avoids repetition except in degenerate cases, which we solve by separators (zero vector) and controlled restarts.

Thus the *same algebraic recurrence* used for decoding is also the **generator rule**.

---

# **8. Generator Logic: Ensuring Independent Feature Vectors**

To recover (A), the decoder needs **m** independent feature vectors.

The generator must:

* explore the graph of all states,
* detect revisits,
* avoid fixed points (A F(x)=x),
* avoid absorbing 0 states (A F(x)=0),
* use random jumps,
* restart on loop detection.

This is analogous to building a **nonlinear Krylov sequence**:

[
x,;A F(x),;A F(A F(x)),\dots
]

which is empirically rich.

---

# **9. Why Not All Matrices (A) Are Safe**

The channel censors the following transitions:

* ((0,0)),
* ((x,0)),
* ((0,x)),
* ((x,x)).

Some matrices necessarily produce such transitions, making recovery impossible:
the decoder cannot observe the columns of (Y) corresponding to the basis.

Therefore we restrict (A) to a **safe family** that avoids these transitions *entirely*.

---

# **10. Constructing a Safe Family of Matrices**

Let (x_1,\dots,x_m) be the chosen basis states and:

[
X=[F(x_1)\dots F(x_m)].
]

Pick output columns (y_i\neq 0) such that:

* (y_i \ne 0)   → avoids ((x_i,0)),
* (y_i \ne x_i) → avoids ((x_i,x_i)),
* (x_i \neq 0)  → avoids ((0,x_i)),
* (F(0)\neq 0) and zero transitions are used only as separators.

Thus:

[
A = Y X^{-1}
]

is safe.

For each column:

* forbidden = {0, x_i},
* allowed = {1,…,2^n−1} \ {x_i}.

This gives:

[
2^n - 2
]

choices per column.

Total allowed matrices:

[
D = (2^n - 2)^m.
]

---

# **11. Mapping Message Index ↔ Matrix (A)**

Given:

[
0 \le \text{index} < (2^n - 2)^m,
]

decode it in base (2^n-2):

[
(d_1,\dots,d_m),\quad d_i \in [0, 2^n-3].
]

Let:

[
y_i = \text{allowed}[d_i]
]

i.e., the (d_i)-th element of the list
({1,\dots,2^n-1}\setminus{x_i}).

Build:

[
Y=[y_1\dots y_m],
]

and compute:

[
A = Y X^{-1}.
]

This mapping is a **bijection** between indices and safe matrices.

---

# **12. Allowed Range of (m)**

With constant term:

[
1 \le m \le 2^n.
]

Without constant term (not used):

[
1 \le m \le 2^n - 1.
]

---

# **13. Full Recovery Algorithm**

Decoder collects transitions ((x,y)), ignoring:

* separators,
* repeats,
* degenerate pairs.

It keeps transitions until it finds (m) linearly independent vectors (F(x_i)).

Then:

[
X_{\text{obs}}^{(m)}=[F(x_{i_1})\dots F(x_{i_m})],
\qquad
Y_{\text{obs}}^{(m)}=[y_{i_1}\dots y_{i_m}].
]

Compute:

[
A = Y_{\text{obs}}^{(m)} (X_{\text{obs}}^{(m)})^{-1}.
]

Then:

[
Y_{\text{rec}} = A X.
]

Decode payload index from (Y_{\text{rec}}).
Recovery succeeds because:

* safe matrices always produce non-erased transitions for basis states,
* generator will eventually reach the required basis,
* ANF monomials guarantee rank.

---

# **14. Why Safe Matrices Guarantee Recovery**

The safe family ensures:

* no fixed points,
* no zero outputs,
* no collapsed outputs,
* no rank-deficient (Y),
* generator never stabilizes prematurely,
* sufficient coverage of state space,
* eventual discovery of the feature basis.

Thus recovery is **guaranteed**.

---

# **15. Information-Theoretic Interpretation**

Message bits:

[
B = m \log_2(2^n - 2) \approx m n.
]

As long as the traversal covers enough transitions:

[
R \approx \frac{1}{\alpha}
]

with very small redundancy factor (\alpha).

This is extremely efficient for tiny packet sizes.

---

# **16. Rare Failure Modes**

Failures occur only when:

* (m) is extremely large (near (2^n)),
* traversal accidentally enters a very short cycle,
* unlucky dynamics,
* insufficient samples.

These are probabilistic, not structural.

---

# **17. Future Directions**

* Encode payload also in the *choice* of monomials.
* Explore nonlinear Krylov theory.
* Add soft redundancy.
* Combine several feature families.

---

# **18. Summary**

This system implements a nonlinear, fountain-like coding scheme based on:

* full-rank ANF feature maps with constant term,
* a safe family of matrices (A),
* a bijection between payload indices and safe matrices,
* a generator that explores the state graph while avoiding degeneracy,
* a decoder that reconstructs (A) via linear algebra.

Everything hinges on the identity:

[
A = Y X^{-1},
]

which fixes the necessary constraints on:

* feature map,
* matrix family,
* traversal,
* index mapping,
* decoding.