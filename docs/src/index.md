# ERGMRank.jl

Model complete rankings of other actors, such as each person’s ordering of everyone else in a group. ERGMRank.jl preserves the ranking sample space, evaluates rank statistics, and fits by MCMC maximum likelihood (the default, R ergm.rank’s estimator) or by a fast swap-based pseudo-likelihood.

| First analysis | Learn the model or data | Reference and detail |
|:--|:--|:--|
| [Build and fit rankings](getting_started.md) | [Understand complete rankings](guide/rank_networks.md) | [Estimate, check and simulate](guide/estimation.md) |

!!! note "Supported scope"

    Every actor must rank all other actors without ties: each row’s off-diagonal values are a permutation of `1:(n-1)`, with larger values indicating higher standing. Partial rankings, tied rankings, and missing dyads are refused. The default estimator is the MCMC MLE, as in R ergm.rank; `method=:mple` selects the swap pseudo-likelihood, a different and faster estimator whose default fit reports no p-values. The [README](https://github.com/statistical-network-analysis-with-Julia/ERGMRank.jl#not-implemented) lists what is not implemented.

## Installation

```@raw html
<p>Use Julia <strong>1.12 or newer</strong> and the <a href="/getting-started/">shared workspace installation guide</a>. These development packages are not yet registered; the guide prepares the required sibling checkouts and a Julia environment for the examples.</p>
```

## Quick Start

Fit the bundled first week of Newcomb’s sociometric rankings by MCMC maximum likelihood (10–20 s; the bridge estimate of the log-likelihood is skipped here to keep it short):

```julia
using ERGMRank, Random

rankings = newcomb_week1()
fit = fit_ergm_rank(rankings, [RankDeference(), RankNonconformity()];
                    bridge_rungs=0, rng=Xoshiro(1))
display(fit)
```

The two terms describe deference and conformity in rankings; the estimates reproduce R `ergm.rank`’s (−0.153 and −0.0066) up to Monte-Carlo error. Follow the [estimation guide](guide/estimation.md) for the MCMC budget and convergence checks, goodness of fit, the fast swap pseudo-likelihood (`method=:mple`), seeded simulation, and the distinction between Julia’s absolute and R’s null-relative log-likelihood reporting.

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
  attribute) — from a directed `NetworkCore.Network` with a rank edge attribute
- The implemented terms have deterministic fixtures generated with
  R `ergm.rank` 4.1.2: [`RankDeference`](@ref),
  [`RankNonconformity`](@ref) (`:all`/`:localAND`; the other `ergm.rank`
  variants are refused, not silently approximated), [`RankNodeICov`](@ref),
  [`RankInconsistency`](@ref) (with `ergm.rank`'s `weights`, `wtname` and
  `wtcenter`), [`RankEdgeCov`](@ref) — each with a per-swap change
  statistic [`swap_change`](@ref)
- [`ergm_rank`](@ref) fits by **MCMC maximum likelihood** by default (the
  estimator of `ergm.rank`, asserted against it on two Newcomb fixtures) or
  by swap-based maximum pseudo-likelihood (`method=:mple`: fast, with R's
  `drop` semantics for boundary statistics, a loud "MPLE does not exist"
  for separated models, and no p-values unless bootstrapped or opted into),
  with the full StatsAPI surface and the shared result-metadata protocol on
  the result
- [`gof`](@ref) compares the observed ranking with simulated ones on the
  model statistics and on auxiliary statistics the model does not contain
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
[`CITATION.bib`](https://github.com/statistical-network-analysis-with-Julia/ERGMRank.jl/blob/main/CITATION.bib).
Please also cite the R package it follows, `ergm.rank`, and the paper that
introduced the models (Krivitsky & Butts 2017, reference 1 above); the
ecosystem's [How to cite](https://statistical-network-analysis-with-julia.github.io/citing/)
page lists the full references.

```biblatex
@misc{SNWJERGMRankJL,
  author = {Santoni, Simone},
  title = {ERGMRank.jl: Exponential Random Graph Models for Rank-Order Relational Data in Julia},
  year = {2026},
  url = {https://github.com/statistical-network-analysis-with-Julia/ERGMRank.jl},
  note = {Homepage: https://statistical-network-analysis-with-Julia.github.io/ERGMRank.jl; GitHub: https://github.com/statistical-network-analysis-with-Julia}
}
```
