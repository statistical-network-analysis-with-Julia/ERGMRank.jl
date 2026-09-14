# Getting Started

Encode complete rankings, inspect the statistics, and fit a swap-MPLE before simulating ranked networks. The R migration section explains which estimator and statistic comparisons are meaningful and which inputs are deliberately rejected.

!!! note "Before you begin"

    Every actor must rank all other actors without ties: each row’s off-diagonal values are a permutation of `1:(n-1)`, with larger values indicating higher standing. Partial rankings, tied rankings, and missing dyads are refused. Swap-MPLE differs from R ergm.rank’s MCMC-MLE objective; select the estimator explicitly.

## Installation

```@raw html
<p>Use Julia <strong>1.12 or newer</strong> and the <a href="/getting-started/">shared workspace installation guide</a>. These development packages are not yet registered; the guide prepares the required sibling checkouts and a Julia environment for the examples.</p>
```

Run the blocks below in order in that environment. They build on variables from earlier steps; stochastic examples use seeded random number generators where shown.

## Building a rank network

A rank network is a square matrix of rank values: row ``i`` holds the
ranks ego ``i`` assigns to the alters, with greater values indicating
higher standing, and each row's off-diagonal entries a permutation of
`1:(n-1)`.

```julia
using ERGMRank      # exports the shared `compute`/`name` verbs and the StatsAPI surface

m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
rnet = as_rank_network(m)

get_rank(rnet, 1, 2)   # 3 — ego 1 ranks actor 2 highest
is_valid_ranking(rnet) # true
```

Passing a matrix that is not a valid set of complete rankings (duplicate
ranks within an ego, gaps, nonzero diagonal) throws an `ArgumentError`.

A directed `Networks.Network` whose arcs carry the ranks as an edge
attribute — the shape `ergm.rank`'s `newcomb` data has in R — converts with
the same function; see [From a `Network`](guide/rank_networks.md#From-a-Network)
for what the adapter preserves, rejects and reports:

```julia
using Networks
net = network(4)                          # directed
for i in 1:4, j in 1:4
    i == j && continue
    add_edge!(net, i, j)
    set_edge_attribute!(net, :rank, i, j, m[i, j])
end
rank_matrix(as_rank_network(net)) == m    # true
```

## Computing statistics

```julia
compute(RankDeference(), rnet)                    # 6.0
compute(RankNonconformity(:all), rnet)            # 10.0
compute(RankNodeICov([10, 20, 30, 40]), rnet)     # -40.0
```

These values are verified against R `ergm.rank` 4.1.2 in the test suite.

## Fitting a model

```julia
result = ergm_rank(rnet, [RankDeference(), RankNodeICov([10, 20, 30, 40])])
display(result)           # the R-style block the REPL prints (print/string give one line)
result.converged          # true
coeftable(result)         # the table `display(result)` printed, as an object
coeftable(result)["rank.deference"].estimate == coef(result)[1]   # true
confint(result)           # Wald limits, one row per coefficient
aic(result), bic(result)  # pseudo-AIC / pseudo-BIC of the swap pseudo-likelihood
approximations(result)    # what the estimator did NOT do (from Networks.jl)
```

By default `ergm_rank` is the swap-based maximum pseudo-likelihood
estimator, not `ergm.rank`'s MCMC MLE, and its default standard errors are
expected to be anticonservative — the printed output says so,
`approximations(result)` says so, and `se=:bootstrap` gives a
parametric-bootstrap covariance instead. `method=:mcmle` fits the MCMC MLE
itself — the estimator of `ergm.rank`, reproduced on the Newcomb fixture
within the two implementations' Monte-Carlo spread:

```julia
using Random
mle = ergm_rank(rnet, [RankDeference(), RankNodeICov([10, 20, 30, 40])];
                method=:mcmle, n_samples=512, rng=Xoshiro(1))
mle.method                # :mcmle
mle.converged             # true: t-ratios and Hotelling test passed
approximations(mle)       # Monte-Carlo error (in the SEs), bridge log-likelihood
```

See the [estimation guide](guide/estimation.md) for the MCMLE's budget and
diagnostics, for the log-likelihood convention (`loglikelihood(mle)` is
absolute; R's `logLik` is relative to θ = 0, a constant `n·log((n−1)!)`
apart), and for what happens when the swap-MPLE does not exist (a
statistic at the boundary of its attainable range, or a separated model) and
when a fit does not converge: every such case warns at fit time, is recorded
on the result, and is printed under `Converged: false`.

## Coming from `ergm.rank`

An `ergm.rank` session maps call by call. The one thing to know: `ergm()`
always fits the MCMC MLE, while `ergm_rank`'s default `method=:mple` is the
swap pseudo-likelihood — pass `method=:mcmle` for R's estimator.

| R `ergm.rank` | ERGMRank.jl |
|---|---|
| `ergm(nw ~ rank.deference + rank.nonconformity("all"), response="rank", reference=~CompleteOrder, control=control.ergm(MCMC.samplesize=2048, MCMC.burnin=8192, MCMC.interval=512))` | `ergm_rank(as_rank_network(net), [RankDeference(), RankNonconformity(:all)]; method=:mcmle, n_samples=2048, burnin=8192, interval=512)` |
| `response="rank"` | `as_rank_network(net; attr=:rank)` — the `RankNetwork` is the rank response |
| `reference=~CompleteOrder` | implicit: the only reference measure (`CompleteOrderReference`) |
| `summary(fit)` | `display(fit)` / `coeftable(fit)`; `logLik`→`loglikelihood` (absolute; R's is relative to θ = 0), `AIC`/`BIC`→`aic`/`bic`, `nobs` |
| `simulate(fit, nsim=100)` | `simulate_rank_ergm(fit; n_sim=100)` |
| `gof(fit)` | `gof(fit)` |

`newcomb[[1]]` is `newcomb_week1()`, `summary(nw ~ rank.deference, response="rank")`
is `compute(RankDeference(), rnet)`, and `control.ergm(seed=)` is `rng=`.

## Common mistakes

The mistakes an R user is most likely to make are refused with an error
that names the fix:

```julia
using Test
# An unobserved rank has no face value: a RankNetwork holds COMPLETE orderings
m_missing = Matrix{Union{Missing, Int}}(m); m_missing[1, 2] = missing
@test_throws ArgumentError as_rank_network(m_missing)

# Fewer than 3 actors: no ego has two alters to compare
@test_throws ArgumentError ergm_rank(RankNetwork(2), [RankDeference()])

# An unsupported standard-error method (the shared `Networks.check_se` message)
@test_throws ArgumentError ergm_rank(rnet, [RankDeference()]; se=:sandwich)

# No draws requested
@test_throws ArgumentError simulate_rank_ergm(rnet, [RankDeference()], [0.5]; n_sim=0)

# A Network with a masked (unobserved) dyad: a rank has no face value, so there
# is no `missing=:face` — the adapter refuses (see the Rank Networks guide)
set_missing_dyad!(net, 1, 2)
@test_throws ArgumentError as_rank_network(net)

# ergm.rank's rank.nonconformity("local1"/"local2"/"geometric"/"thresholds")
# are not implemented: only :all and :localAND exist
@test_throws ArgumentError RankNonconformity(:local1)

# A binary ERGM.jl term has no rank statistic; a Network needs as_rank_network
# first; a covariate is passed as values, not as an attribute name
using ERGM: Edges
@test_throws ArgumentError ergm_rank(rnet, [RankDeference(), Edges()])
@test_throws ArgumentError ergm_rank(net, [RankDeference()])
@test_throws ArgumentError RankNodeICov(:age)
```

The README's "Common mistakes" section lists the exact error text of each.

## Simulating

```julia
draws = simulate_rank_ergm(rnet, [RankDeference()], [0.5];
                           n_sim = 100, burnin = 1000, interval = 20)
all(is_valid_ranking, draws)   # true
```

A positive coefficient on a statistic makes configurations with more of it
more probable; every draw remains a valid complete ranking because the
AlterSwap move only ever exchanges two of an ego's ranks.
