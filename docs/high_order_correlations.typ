#set document(
  title: "High-Order Correlation Analysis in Detector Error Models",
  author: "correlation-analysis",
)
#set page(margin: (x: 1.0in, y: 1.0in))
#set text(lang: "en", size: 10.5pt)
#set par(justify: true)
#show math.equation: set block(breakable: true)

#title[High-Order Correlation Analysis in Detector Error Models]

This note explains the algorithm implemented in `correlation.cal_high_order_correlations`. The presentation follows the structure of the code in `src/correlation/numeric.py`: build clusters, estimate expectations from samples, solve one local inverse problem per cluster, and then reconcile overlapping clusters into a single global set of hyperedge probabilities.

= Problem Setup

Consider a cluster with root detector set $R$. For every non-empty subset $S subset.eq R$, define an independent Bernoulli event with probability $p_S$. If that event occurs, it flips every detector in $S$.

For each detector $i in R$, the observed detector bit is the XOR of all events that touch $i$:

$ X_i = xor.big_(S subset.eq R, S != emptyset, i in S) E_S $

The sampled detector data gives the joint moments

$ x_T = op("E")[product_(i in T) X_i], quad T subset.eq R, T != emptyset $

for every non-empty subset $T subset.eq R$.

The inverse problem is: recover every $p_S$ from the measured moments $x_T$.

= Cluster Construction

The full detector error model can contain many hyperedges that do not interact. The code therefore divides the requested hyperedges into clusters. Each cluster is defined by its root hyperedge and contains every non-empty subset of that root.

This reduces the global inverse problem into many small local problems. If a root has size $k$, the associated local system has only $2^k - 1$ unknowns.

= Expectations From Samples

The code uses exact sample means instead of fitting a parametric model.

For weight-1 and weight-2 subsets, expectations are taken directly from the two-point expectation matrix:

$ x_({i}) = op("E")[X_i] $

$ x_({i, j}) = op("E")[X_i X_j] $

For higher-order subsets, the implementation batches hyperedges of the same weight and computes

$ x_T = op("E")[product_(i in T) X_i] $

with vectorized boolean reductions over the sampled detector matrix.

= Nonlinear Form Of The Local Model

For any non-empty subset $T subset.eq R$, the old numerical solver used the exact nonlinear equation

$ x_T = sum_(C subset.eq H_T, T subset.eq Delta C) product_(S in C) p_S product_(S in H_T - C) (1 - p_S) $

where $H_T$ is the set of all non-empty subsets of $R$ that intersect $T$, and $Delta C$ denotes the symmetric difference of the sets selected in $C$.

This equation is correct, but expensive to solve directly because a generic root finder has to evaluate it many times.

= Parity-Moment Transform

The optimized implementation changes variables instead of iterating on the nonlinear system.

For each non-empty $T subset.eq R$, define the parity moment

$ m_T = op("E")[product_(i in T) (1 - 2 X_i)] $

Because $X_i in {0, 1}$, we have $1 - 2 X_i = (-1)^(X_i)$. Expanding the product gives a linear transform from ordinary moments to parity moments:

$ m_T = 1 + sum_(U subset.eq T, U != emptyset) (-2)^(|U|) x_U $

This is the first precomputed matrix in the implementation.

= Exact Factorization

Each event $E_S$ flips the parity over $T$ if and only if $|S inter T|$ is odd. Since the events are independent,

$ m_T = product_(S subset.eq R, S != emptyset, |S inter T| " odd") (1 - 2 p_S) $

Now define

$ q_S = log(1 - 2 p_S) $

$ b_T = log(m_T) $

Then the multiplicative model becomes linear:

$ b_T = sum_(S subset.eq R, S != emptyset) A_(T, S) q_S $

with

$ A_(T, S) = 1 quad "if" quad |T inter S| " is odd" $

$ A_(T, S) = 0 quad "otherwise" $

In matrix form,

$ b = A q $

so the local solution is

$ q = A^(-1) b $

and therefore

$ p_S = (1 - exp(q_S)) / 2 $

This is an exact inversion of the cluster model, not an approximation.

= Numerical Fallback

The direct solve requires $m_T > 0$ so that $log(m_T)$ is defined. Finite-shot sampling can occasionally produce invalid empirical moments. When that happens, the implementation falls back to the legacy `scipy.optimize.root` solve for that cluster only.

This keeps the fast path exact and cheap while preserving robustness on noisy data.

= Reconciling Overlapping Clusters

Different clusters can contain the same low-weight hyperedge. After each cluster is solved locally, the code processes hyperedges from high weight to low weight and removes contributions that have already been assigned to larger solved hyperedges.

If independent events with probabilities $a$ and $b$ are XOR-combined, the resulting probability is

$ a xor b = a + b - 2 a b $

Applying this repeatedly gives the total probability mass contributed by already-solved supersets of a hyperedge. If a local cluster solve gives probability $p_"local"$ and the supersets contribute $p_"sum"$, the corrected probability is

$ p_"corrected" = (p_"local" - p_"sum") / (1 - 2 p_"sum") $

The final estimate for a shared hyperedge is the mean of the corrected values contributed by all relevant clusters.

= Complexity And Practical Effect

For a cluster of size $k$, the direct local solve operates on vectors and matrices of size $2^k - 1$. In practice, detector-error-model clusters are small, so the direct solve is dominated by sample expectation extraction rather than by inversion.

The current implementation therefore spends most of its time in two places:

1. Building sample expectations for all requested higher-order subsets.
2. Reconciling overlapping clusters into final probabilities.

The closed-form local solve eliminates the old repeated Python-level residual evaluation and turns the former solver bottleneck into a tiny dense linear-algebra step.

= Validation Strategy

The repository validates the implementation in three complementary ways:

1. The direct cluster solve is checked against the legacy root solver on synthetic clusters.
2. Pairwise-only high-order runs are checked against the analytic second-order solver.
3. Full high-order runs are checked against decomposed detector error models from `stim`.

This combination tests the algebraic inversion itself, the low-order limit, and the full end-to-end numerical pipeline.
