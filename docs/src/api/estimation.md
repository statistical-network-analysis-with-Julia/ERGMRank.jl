# Estimation and Simulation

## Fitting

`fit_ergm_rank` (and its aliases) takes `method=:mple` (the swap
pseudo-likelihood, the default) or `method=:mcmle` (the MCMC maximum
likelihood estimate of R `ergm.rank`). The MCMLE keywords — `n_samples`,
`burnin`, `interval`, `n_chains`, `init`, `gamma0`, `max_step_norm`,
`conv_threshold`, `hotelling_alpha`, `bridge_rungs`, `bridge_samples`,
`verbose` — are `ERGM.mcmle`'s vocabulary, spelled the same, and `maxiter`
caps the MCMLE iterations (default 20) or the swap-MPLE's Newton iterations
(default 100). `se` resolves per method: `:hessian`/`:bootstrap` under
`:mple`, `:fisher` under `:mcmle`. The convergence report of an MCMLE fit is
an `ERGM.MCMLEConvergence` in `fit.mcmc_convergence`. A single term or a
tuple of terms is accepted in place of the vector; a `Network` or a rank
matrix in place of the `RankNetwork`, and a binary ERGM.jl term in the term
list, are refused with an `ArgumentError` naming the fix. `fit_rank_ergm` is
a **deprecated binding** (the first access in a session warns under
`--depwarn=yes`; Julia warns once per deprecated binding) removed in 0.3.

```@docs
fit_ergm_rank
ergm_rank
fit_rank_ergm
```

## Data

```@docs
newcomb_week1
```

## Result metadata

The shared result-metadata protocol of Networks.jl (`fit_metadata`,
`approximations`, `estimand`, `objective`, `is_exact`, `se_method`,
`missing_method`) is exported by ERGMRank, so `approximations(fit)` works
with just `using ERGMRank`. `estimand(fit)` is `:rank_ergm`,
`missing_method(fit)` is `:none` (a `RankNetwork` has no dyad mask) and
`approximations(fit)` lists, per method, what the estimator did not do
exactly (the swap pseudo-likelihood and its anticonservative pseudo-Hessian
errors, or the Monte-Carlo approximation of the likelihood and the bridge),
plus any non-convergence, dropped boundary statistic, undefined standard
error or excluded bootstrap replicate. The three below carry rank-specific
answers.

```@docs
objective(::RankERGMResult)
is_exact(::RankERGMResult)
se_method(::RankERGMResult)
```

## StatsAPI

The full ecosystem StatsAPI surface (`coef`, `stderror`, `vcov`, `confint`,
`loglikelihood`, `nobs`, `dof`, `aic`, `bic`, `coeftable`) is defined on
[`RankERGMResult`](@ref) and pinned by `Networks.check_statsapi` in the test
suite. `coef`, `stderror`, `vcov`, `loglikelihood` read the result's fields;
`nobs` is the number of (ego, unordered alter pair) swap comparisons of a
`:mple` fit and the `n(n−1)` ordered dyads of a `:mcmle` fit (R's
`nobs.ergm`); `dof` counts the finite coefficients. `loglikelihood` is the
maximized swap pseudo-log-likelihood of a `:mple` fit (`NaN` when no
comparison is left after a boundary drop) and the bridge estimate of the
**absolute** log-likelihood of a `:mcmle` fit (`NaN` under
`bridge_rungs=0`) — R's `logLik` is relative to θ = 0, `n·log((n−1)!)`
higher. The five below carry rank-specific caveats.

```@docs
nobs(::RankERGMResult)
aic(::RankERGMResult)
bic(::RankERGMResult)
confint(::RankERGMResult)
coeftable(::RankERGMResult)
```

## Simulation

```@docs
simulate_rank_ergm
```

## Goodness of fit

```@docs
gof(::RankERGMResult)
```
