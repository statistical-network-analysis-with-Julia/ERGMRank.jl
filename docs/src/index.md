# ERGMRank.jl

ERGMs for rank-order relational data: exponential-family random graph
models for networks whose edge values are complete rankings, following
Krivitsky & Butts (2017) and the R `ergm.rank` package.

## The Model

Each ego rank-orders all alters. The observation for ego ``i`` is a
permutation of the ``n-1`` alters, encoded as rank values with **greater
values indicating higher standing**. The model is

```math
P(\mathbf{Y} = \mathbf{y}) \propto \exp\left(\theta' g(\mathbf{y})\right)
```

on the space of complete orderings, under the discrete-uniform
[`CompleteOrderReference`](@ref) — every valid rank configuration has equal
baseline weight, so the structure comes from the sample-space constraint
and the statistics ``g``.

## Highlights

- [`RankNetwork`](@ref) enforces the complete-ranking invariant (each
  ego's ranks are a permutation of `1:(n-1)`), preserved by
  [`swap_ranks!`](@ref) (the AlterSwap move)
- Terms match `ergm.rank` exactly and are golden-master tested against
  R `ergm.rank` 4.1.2: [`RankDeference`](@ref),
  [`RankNonconformity`](@ref), [`RankNodeICov`](@ref),
  [`RankInconsistency`](@ref), [`RankEdgeCov`](@ref)
- [`ergm_rank`](@ref) fits by swap-based maximum pseudo-likelihood
- [`simulate_rank_ergm`](@ref) samples with AlterSwap Metropolis moves

## Contents

```@contents
Pages = [
    "getting_started.md",
    "guide/rank_networks.md",
    "guide/terms.md",
    "guide/estimation.md",
    "api/types.md",
    "api/terms.md",
    "api/estimation.md",
]
Depth = 2
```

## References

1. Krivitsky, P.N. & Butts, C.T. (2017). Exponential-family random graph
   models for rank-order relational data. *Sociological Methodology*, 47(1), 68-112.
2. Krivitsky, P.N. (2012). Exponential-family random graph models for
   valued networks. *Electronic Journal of Statistics*, 6, 1100-1128.
