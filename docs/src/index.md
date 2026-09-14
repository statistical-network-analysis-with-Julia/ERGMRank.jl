# ERGMRank.jl

Model complete rankings of other actors, such as each person’s ordering of everyone else in a group. ERGMRank.jl preserves the ranking sample space, evaluates rank statistics, and fits by swap-based pseudo-likelihood or MCMC maximum likelihood.

| First analysis | Learn the model or data | Reference and detail |
|:--|:--|:--|
| [Build and fit rankings](getting_started.md) | [Understand complete rankings](guide/rank_networks.md) | [Compare MPLE and MCMC-MLE](guide/estimation.md) |

!!! note "Supported scope"

    Every actor must rank all other actors without ties: each row’s off-diagonal values are a permutation of `1:(n-1)`, with larger values indicating higher standing. Partial rankings, tied rankings, and missing dyads are refused. Swap-MPLE differs from R ergm.rank’s MCMC-MLE objective; select the estimator explicitly.

## Installation

```@raw html
<p>Use Julia <strong>1.12 or newer</strong> and the <a href="/getting-started/">shared workspace installation guide</a>. These development packages are not yet registered; the guide prepares the required sibling checkouts and a Julia environment for the examples.</p>
```

## Quick Start

Fit a quick swap-MPLE to the bundled first week of Newcomb’s sociometric rankings:

```julia
using ERGMRank

rankings = newcomb_week1()
fit = fit_ergm_rank(rankings, [RankDeference(), RankNonconformity()]; method=:mple)
display(fit)
```

The two terms describe deference and conformity in rankings. This first fit demonstrates the conditional swap objective. Follow the [estimation guide](guide/estimation.md) for `method=:mcmle`, seeded simulation, convergence checks, and the distinction between Julia’s absolute and R’s null-relative log-likelihood reporting.

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
  [`swap_ranks!`](@ref) (the AlterSwap move); [`as_rank_network`](@ref)
  builds one from a rank matrix or — honouring the ecosystem conversion
  contract (reject a masked or undirected network, report every dropped
  attribute) — from a directed `Networks.Network` with a rank edge attribute
- The implemented terms have deterministic fixtures generated with
  R `ergm.rank` 4.1.2: [`RankDeference`](@ref),
  [`RankNonconformity`](@ref) (`:all`/`:localAND`; the other `ergm.rank`
  variants are refused, not silently approximated), [`RankNodeICov`](@ref),
  [`RankInconsistency`](@ref), [`RankEdgeCov`](@ref) — each with a
  per-swap change statistic [`swap_change`](@ref) ported from
  `wtchangestats_rank.c`
- [`ergm_rank`](@ref) fits by swap-based maximum pseudo-likelihood
  (`method=:mple`, fast, with R's `drop` semantics for boundary statistics
  and a loud "MPLE does not exist" for separated models) or by **MCMC
  maximum likelihood** (`method=:mcmle`, the estimator of `ergm.rank`,
  asserted against it on the Newcomb fixture), with the full StatsAPI
  surface and the shared result-metadata protocol on the result
- [`simulate_rank_ergm`](@ref) samples with AlterSwap Metropolis moves on
  the ecosystem's one `ERGM.mh_toggle!` kernel — allocation-free steps,
  swap-scaled burn-in defaults, every draw reproducible from `rng`

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

## Citation

If you use ERGMRank.jl in your work, please cite it using the entry in
[`CITATION.bib`](https://github.com/statistical-network-analysis-with-Julia/ERGMRank.jl/blob/main/CITATION.bib):

```biblatex
@misc{SNWJERGMRankJL,
  author = {{Statistical Network Analysis with Julia}},
  title = {ERGMRank.jl: Exponential Random Graph Models for Rank-Order Relational Data in Julia},
  year = {2026},
  url = {https://github.com/statistical-network-analysis-with-Julia/ERGMRank.jl},
  note = {Homepage: https://statistical-network-analysis-with-Julia.github.io/ERGMRank.jl; GitHub: https://github.com/statistical-network-analysis-with-Julia}
}
```
