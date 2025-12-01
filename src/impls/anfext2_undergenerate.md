# **0. Introduction and Motivation**

We begin from a very simple, powerful insight:

> **If a channel allows us to observe sufficiently many transitions of the form**
> [
> y = A x,
> ]
> **then we can recover the unknown matrix** (A).

This is the classical basis of *linear probing*: if the decoder observes pairs ((x,y)), then:

* Each pair gives one linear equation in the unknowns of (A).
* With enough independent pairs, the decoder solves a linear system and finds (A).

This naive idea immediately suggests a communication scheme:

* **Encode payload as a matrix (A)**.
* **Transmit transitions ((x,y=A x))** over a channel.
* **Decoder reconstructs (A)** from enough observations.

However:

* If (A) is (n \times n), the payload is only (n^2) bits.
* We want **much more**.

So we introduce a nonlinear feature map (F), such that:

[
y = A,F(x)
]

where

* (x\in{0,1}^n),
* (F(x)\in {0,1}^m),
* (A\in \mathbb{F}_2^{n\times m}),
* (m) can be **huge** — up to (2^n) with ANF monomials.

This expands the payload to:

[
\text{payload bits} = \log_2 D
]

where (D) is the number of matrices we allow.
Even for small (n=7), you can hold tens of thousands of bits.

But this introduces nontrivial problems:

* What constraints must (F) satisfy for recovery to work?
* Which matrices (A) can be used?
* How do we ensure the generator produces recoverable transitions?
* How do we design the traversal so the decoder sees enough independent feature vectors?
* How do we map an index into a nondegenerate matrix?
* How do we guarantee recovery under channel restrictions?

This README answers *all* these questions.

---

# **1. The Core Algebra: Decoding Requires Reconstructing a Linear Map**

The fundamental identity of our scheme is:

[
y = A F(x).
]

If we collect transitions:

[
(x_1,y_1), (x_2,y_2), \dots
]

then we can build:

[
X_{\text{obs}} = [F(x_1);F(x_2);\dots;F(x_k)]
]
[
Y_{\text{obs}} = [y_1;y_2;\dots;y_k]
]

**If** the decoder finds a subset of columns giving:

[
X_{\text{obs}}^{(m)} \in \mathbb{F}_2^{m\times m}
\quad\text{invertible},
]

then:

[
A = Y_{\text{obs}}^{(m)} (X_{\text{obs}}^{(m)})^{-1}.
]

Thus decoding is **pure linear algebra**.

From this, several constraints fall out **inevitably**.

---

# **2. Why We Need a Feature Map (F)**

The number of bits carried by a matrix (A\in \mathbb{F}_2^{n\times m}) is:

[
n\cdot m.
]

If (m=n), this is only (n^2) bits.

To encode multi-kilobit payloads using tiny packets (e.g. 7 bits), we need **huge (m)**.

The feature map (F) must satisfy:

* It must generate a space whose rank is exactly **(m)**.
* The set ({F(x): x\in {0,1}^n}) must span (\mathbb{F}_2^m).
* There must exist (m) states (x_i) such that their feature columns form an invertible (m\times m) matrix.

Thus **(F) must be a full-rank embedding**.

---

# **3. Requirements on (F(x))** (ABSOLUTELY CRITICAL)

From

[
A = Y X^{-1}
]

we require:

### 1. **The feature vectors (F(x)) must span an (m)-dimensional vector space.**

Otherwise (X^{-1}) doesn’t exist.

### 2. **The decoder must be able to observe enough transitions to extract an invertible subset of columns.**

Thus (F(x)) must:

* Be efficiently sampleable via generator dynamics,
* Produce many distinct values,
* Not collapse under the dynamics induced by (A).

### 3. **Feature vectors must be linear combinations of fixed monomials to maintain linear algebra structure.**

This prevents nonlinear degeneracies.

### 4. **The mapping (x\mapsto F(x)) must be deterministic and known to both encoder and decoder.**

### 5. **The total number of monomials used = m must satisfy**

[
1 \le m \le 2^n.
]

Using ANF monomials we get perfect rank.

---

# **4. Choosing A Concrete Family: ANF Monomials With Constant Term**

Among the huge set of possible (F), we pick a specific one to study:

* Every coordinate of (F) is a monomial of the form
  [
  \prod_{i\in S} x_i,
  ]
* Sorted in degree-then-lex order.
* Include the constant term (mask 0).

### Why monomials?

Because:

* They form a linearly independent basis of the algebra of Boolean functions.
* The matrix of all monomials evaluated on all (2^n) inputs is invertible (Walsh–Hadamard related structure).
* Easy to compute.
* Beautiful algebraic properties.
* Full rank guaranteed.
* The number of monomials is huge, reaching (2^n).

### Why include the constant term?

Because:

* It gives (F(0)=1\cdot e_0\neq 0).
* Needed for building a basis including (x=0).
* Strongly stabilizes the system.
* Helps to avoid degeneracies in (X).

---

# **5. Formalizing the Constraints on F**

Given that (A = Y X^{-1}), the feature matrix must satisfy:

### 1. **Rank condition**

[
\text{rank}({F(x): x\in{0,1}^n})=m.
]

### 2. **Existence of invertible basis**

There must exist states (x_1,\dots,x_m) such that:

[
X=[F(x_1)\dots F(x_m)]
\quad\text{is invertible.}
]

### 3. **Basis must be discoverable**

These states must appear in the generator traversal.

### 4. **Feature values must produce enough variety**

The generator must never collapse into a subspace where (F(x)) loses rank.

### 5. **Feature maps must be computationally cheap**

ANF monomials satisfy all 5 perfectly.

---

# **6. There Are Many Valid Families — We Study Monomials Only**

The constraints above allow many possible families:

* Spectral kernels (Walsh–Hadamard basis)
* Polynomial Boolean networks
* Nonlinear coordinate transforms
* Parity-check inspired embeddings
* Neural hash embeddings
* Algebraic normal form bases (monomials)

We choose **ANF monomials with constant term** because:

* Structured,
* Full rank,
* Easy to enumerate,
* Easy to compute,
* Easy to invert basis.

### Future idea:

> Encode *part* of the payload into the *choice* of the monomial subset itself.

This would allow **super-payload** approaches, combining:

* Matrix-based payload,
* Feature-set choice payload.

---

# **7. How Do We Send ((x,y)) Pairs Over a Channel That Forbids Identical Consecutive Packets?**

We need to transmit transitions ((x,y)), but the channel forbids:

* Two consecutive equal packets.

Naively, one might send:

[
y,; x,; 0,; y,; x,; 0,;\dots
]

But this is extremely inefficient.

We improved the idea:

> **Use the previous output as the next input.**

If you transmit a sequence:

[
x_1, y_1, x_2, y_2, x_3, y_3,\dots
]

where:

[
x_{t+1} = y_t,
]

then:

* We naturally transmit a chain of transitions.
* Every two consecutive packets are distinct (rare degeneracy handled by resets).
* Rate nearly doubles.

This suggests adopting:

[
x_{t+1} = A F(x_t)
]

not only as a decoding equation, but also as the **generator rule**.

---

# **8. The Generator: Ensuring Fast Discovery of Independent Feature Vectors**

To recover (A), the decoder needs to observe **m linearly independent** feature vectors (F(x)).

We emperically tried several generators and came up with this restrictions being optimal:

* Explore the state space,
* Avoid long cycles,
* Avoid fixed points (AF(x)=x) (channel limitation),
* Avoid zeros (AF(x)=0) (because of its use as delimiter for jumps between sequences),
* Randomize restarts,
* Jump to “tail states” with no incoming edges,
* Emit separators (zero vectors) on revisits.

This protocol was designed *empirically* and informed by analogies with Krylov subspace generation.

### Connection to Krylov sequences under nonlinear maps

Consider the iterative process:

[
x_{t+1}=A F(x_t).
]

Although nonlinear, it behaves like a nonlinear generalization of:

[
x_{t+1} = A x_t.
]

The Krylov subspace in the linear case:

[
{x,; A x,; A^2 x,;\dots}
]

has rich theoretical properties.

Our generator behaves similarly:

* The set of states visited reflects the repeated application of a map (A\circ F).
* Empirically, this produces high feature diversity.
* Failure mode = entering a short cycle → therefore must avoid all fixed points and small cycles.

This explains the restrictions on (A).

---

# **9. Why Arbitrary A Is Unsafe**

Under the discussed generation scheme. Transitions (x, x), (x, 0), (0, x) become unabservable. And there exists degenerate matrices that are impossible to recover without this transitions.

Thus **we must restrict (A)** to a safe subset.

---

# **10. Constructing a Safe Family of Matrices A**

Let:

* (X=[F(x_1)\dots F(x_m)]) be the invertible feature basis.
* (Y=[y_1\dots y_m]) be the corresponding output columns.

We set:

[
A = Y X^{-1}.
]

Restrictions on columns (y_i):

1. **(y_i \ne 0)** — to avoid trapping in zero state.
2. **(y_i \ne x_i)** — avoid fixed points.
3. **(y_i \ne y_j)** — avoid collapsed outputs.
4. **All valid** (y_i\in{2,3,4,\dots,(2^n-1)}).

Thus:

* Forbidden values: 0,1
* Allowed values: 2..(2^n−1)
* Total allowed per column:
  [
  2^n - 2.
  ]

Thus:

[
D = (2^n - 2)^m
]

is the number of allowed matrices.

---

# **11. The Payload Index ↔ Matrix A Mapping**

Given an index:

[
\text{index} \in [0,; (2^n-2)^m-1]
]

we decode it in base (2^n-2):

[
(d_1,\dots,d_m), \quad d_i \in [0,(2^n-3)].
]

Then define:

[
y_i = d_i + 2.
]

Convert each (y_i) into an n-bit vector → build column i of Y.

Then:

[
A = Y X^{-1}.
]

This is a **bijection** onto the safe matrix family.

---

# **12. Allowed Ranges of m**

With constant term:

[
1 \le m \le 2^n.
]

(There are (2^n) monomials including constant term.)

Without constant term (not used here):

[
1 \le m \le 2^n - 1.
]

---

# **13. Complete Recovery Method**

Decoder observes transitions:

[
(x,y)= (x,;A F(x)).
]

It ignores:

* transitions across zero separators,
* repeated transitions,
* re-entrant dynamics.

It accumulates a set (T) of transitions:

[
T = {(x_i, y_i) }.
]

For each (x_i):

1. Compute (F(x_i)).
2. Check independence using bitmask RREF.
3. Collect until m independent feature vectors found.

Then:

[
X_{\text{obs}}^{(m)} = [F(x_{i_1})\dots F(x_{i_m})],
]
[
Y_{\text{obs}}^{(m)} = [y_{i_1}\dots y_{i_m}].
]

Solve:

[
A = Y_{\text{obs}}^{(m)} (X_{\text{obs}}^{(m)})^{-1}.
]

Compute:

[
Y_{\text{rec}} = A X.
]

Then decode payload index from (Y_{\text{rec}}).

Recovery succeeds because:

* All safe matrices avoid degenerate trajectories.
* Generator explores sufficiently.
* ANF monomials guarantee richness and independence.

---

# **14. Why Recovery Always Works for Safe A**

The safe family ensures:

* No cycles of length 1 or 2,
* No collapsing outputs,
* No zero outputs,
* No rank-deficiency in (Y),
* Generator never gets stuck,
* Enough transitions,
* Feature diversity,
* Eventually decoder sees full-rank (X_{\text{obs}}).

Thus safe A is **guaranteed recoverable**.

---

# **15. Information-Theoretic View**

Ideal rate:

[
R \approx \frac{1-\varepsilon}{\alpha},
]

where:

* (\varepsilon) = erasure probability
* (\alpha \approx 1.00 - 1.20) = small rank-overhead

Thus the scheme approaches channel capacity for moderately small (\alpha).

Payload bits:

[
B = m\log_2(2^n-2) \approx m n.
]

Efficiency extremely high given tiny packet size n.

---

# **16. Known Failure Modes (Rare)**

Failures occur only if:

* (m) extremely large (close to (2^n)),
* Some safe matrices generate long small cycles (rare empirical events),
* Generator traversal is unlucky,
* Need longer transmission to see enough independent transitions.

These are not structural failures; they reflect:

* finite sample size,
* deterministic dynamics,
* unlucky seeds.

---

# **17. Future Directions**

* Encode payload also into *choice of monomial subset*.
* Study nonlinear Krylov-like sequences deeper.
* Add soft redundancy on transitions.
* Hybrid schemes mixing several feature families.

---

# **18. Summary**

This system is a **complete nonlinear fountain-like coding scheme** built on:

* Full-rank feature embeddings (F) using ANF monomials with constant term,
* A safe set of matrices (A) constructed via constraints on output columns,
* A bijective index → matrix mapping with payload domain ((2^n-2)^m),
* A generator that explores state space and avoids degeneracy,
* A decoder that reconstructs (A) via linear algebra over GF(2).

Everything stems from the core principle:

[
A = Y X^{-1},
]

which drives all constraints on:

* feature map (F),
* matrix family (A),
* generator traversal,
* index mapping,
* decoding procedure.