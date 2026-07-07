# Rank Terms

All terms follow the statistics of R `ergm.rank` (implemented from its
`wtchangestats_rank.c`) and are golden-master tested against
`ergm.rank` 4.1.2.

Throughout, ``y_{ij}`` is the rank ego ``i`` assigns alter ``j``
(greater = higher standing).

## RankDeference

`rank.deference` — deference (aversion): the number of ordered triples
``(i, l, j)`` with ``y_{l j} > y_{l i}`` and ``y_{i l} > y_{i j}`` — actor
``l`` ranks ``j`` over ``i``, while ``i`` ranks ``l`` over ``j``.

```julia
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

## RankNodeICov

`rank.nodeicov` — attractiveness/popularity covariate: for every ego and
ordered alter pair ``(j, k)`` with ``j`` ranked over ``k``, adds
``x_j - x_k``.

```julia
RankNodeICov(wealth; label = "wealth")
```

A positive coefficient means high-covariate actors tend to be ranked
higher.

## RankInconsistency

`rank.inconsistency` — the number of ego–alter-pair comparisons on which
the network disagrees with a fixed reference ranking:
``(y_{ij} > y_{ik}) \ne (r_{ij} > r_{ik})``.

```julia
RankInconsistency(reference_matrix)
RankInconsistency(reference_rank_network)
```

Useful for measuring drift from a prior wave or an exogenous ordering.

## RankEdgeCov

`rank.edgecov` — dyadic covariate: for every ego and ordered alter pair
``(j, k)`` with ``j`` ranked over ``k``, adds ``c_{ij} - c_{ik}``.

```julia
RankEdgeCov(cov_matrix; label = "distance")
```

With `cov[i, j] = x[j]` this reduces exactly to `RankNodeICov(x)`.
