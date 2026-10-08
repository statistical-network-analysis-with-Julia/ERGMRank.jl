# Rank Terms

All terms are the statistics of R `ergm.rank` (Krivitsky & Butts 2017) and
are golden-master tested against `ergm.rank` 4.1.2. Every term implements
two methods: `compute(term, rnet)`, the full statistic (a sum over the
triples or quadruples of its definition, written from the definitions
below and checked against `ergm.rank`'s `summary()` output), and
[`swap_change`](@ref)`(term, rnet, ego, j, k)`, the
change in the statistic when `ego` swaps its ranks of `j` and `k` —
evaluated from the comparisons the swap touches (O(n) per term, O(n²) for
the nonconformity variants), never by recomputing either full statistic,
and asserted equal to `compute(swapped) − compute(observed)` exactly in
the test suite. The sampler, the swap-MPLE design and `gof` all run on it.

Throughout, ``y_{ij}`` is the rank ego ``i`` assigns alter ``j``
(greater = higher standing).

## RankDeference

`rank.deference` — deference (aversion): the number of ordered triples
``(i, l, j)`` with ``y_{l j} > y_{l i}`` and ``y_{i l} > y_{i j}`` — actor
``l`` ranks ``j`` over ``i``, while ``i`` ranks ``l`` over ``j``.

```julia
using ERGMRank, Random

RankDeference()
```

A negative coefficient indicates aversion to deference.

## RankNonconformity

`rank.nonconformity` — disagreement between actors' rankings.

```julia
RankNonconformity(:all)       # global nonconformity (default)
RankNonconformity(:localAND)  # local nonconformity
```

- `:all`: over unordered actor pairs ``\{i, j\}`` and ordered alter pairs
  ``(k, l)``, counts comparisons on which ``i`` and ``j`` disagree:
  ``(y_{ik} > y_{il}) \ne (y_{jk} > y_{jl})``.
- `:localAND`: counts disagreements of ego ``i`` with actors ``l`` whom
  ``i`` ranks over both ``j`` and ``k``, where ``l`` ranks ``j`` over
  ``k`` but ``i`` ranks ``k`` at least as high as ``j``.

A negative coefficient captures conformity pressure.

!!! warning "Not implemented: `rank.nonconformity`'s other variants"
    `ergm.rank`'s `rank.nonconformity(to = "local1")`, `"local2"`,
    `"geometric"` and `"thresholds"` have **no counterpart** in ERGMRank.jl:
    only `:all` and `:localAND` exist, and the constructor refuses anything
    else at once — `RankNonconformity(:local1)` throws

    ```
    ArgumentError: RankNonconformity: variant must be :all or :localAND (got :local1);
    ergm.rank's local1/local2/geometric/thresholds variants are not implemented in ERGMRank.jl
    ```

    There is no silent fallback to `:all`. Models that need those variants
    must stay in R for now (each needs its own change statistic and an
    `ergm.rank` fixture; tracked as a known limitation in the CHANGELOG).

## RankNodeICov

`rank.nodeicov` — attractiveness/popularity covariate: for every ego and
ordered alter pair ``(j, k)`` with ``j`` ranked over ``k``, adds
``x_j - x_k``.

```julia
wealth = [10.0, 36.0, 55.0, 44.0, 20.0]   # one value per actor

RankNodeICov(wealth; label = "wealth")
```

A positive coefficient means high-covariate actors tend to be ranked
higher.

## RankInconsistency

`rank.inconsistency` — the (weighted) number of ego–alter-pair comparisons
on which the network disagrees with a fixed reference ranking ``r``:

```math
\sum_{i} \sum_{j \ne k} w_{ijk}\,
\mathbb{1}\bigl[(y_{ij} > y_{ik}) \ne (r_{ij} > r_{ik})\bigr],
```

with every weight 1 unless `weights` is given.

```julia
reference = RankNetwork(5)                          # e.g. the previous wave
RankInconsistency(reference)                        # a RankNetwork or a rank matrix
# ergm.rank's weights / wtname / wtcenter: an n×n×n array, or a function of
# (ego, alter j, alter k); here comparisons made by the first two egos count double
w = RankInconsistency(reference; weights = (i, j, k) -> i <= 2 ? 2.0 : 1.0,
                      wtname = "early", wtcenter = false)
name(w)                                             # "rank.inconsistency:early"
```

`wtcenter=true` subtracts the mean weight first (as in R: the mean over the
triples of distinct actors for a function; over every non-`NaN` entry for an
array, so mark the cells with a repeated index `NaN` to leave them out), and
the name gains a trailing `c`. The weighted statistic is checked against
`ergm.rank`'s `summary()` by a provenanced fixture, for a function and an
array of weights, centred and not.

Useful for measuring drift from a prior wave or an exogenous ordering.

## RankEdgeCov

`rank.edgecov` — dyadic covariate: for every ego and ordered alter pair
``(j, k)`` with ``j`` ranked over ``k``, adds ``c_{ij} - c_{ik}``.

```julia
cov_matrix = rand(Xoshiro(2), 5, 5)   # dyadic covariate, e.g. distance

RankEdgeCov(cov_matrix; label = "distance")
```

With `cov[i, j] = x[j]` this reduces exactly to `RankNodeICov(x)`.
