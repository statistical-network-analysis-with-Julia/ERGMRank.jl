# Estimation and Simulation

## Fitting

`fit_ergm_rank` (= `ergm_rank`) takes `method=:mcmle` (the MCMC maximum
likelihood estimate of R `ergm.rank`, the default) or `method=:mple` (the
swap pseudo-likelihood). The MCMLE keywords — `n_samples`,
`burnin`, `interval`, `n_chains`, `init`, `gamma0`, `max_step_norm`,
`termination`, `conv_precision`, `conv_confidence`, `max_n_samples`,
`conv_threshold`, `hotelling_alpha`, `bridge_rungs`, `bridge_samples`,
`verbose` — are `ERGM.mcmle`'s vocabulary, spelled the same, and `maxiter`
caps the MCMLE iterations (default 60) or the swap-MPLE's Newton iterations
(default 100). `se` resolves per method: `:fisher` under `:mcmle`; under
`:mple`, not given (pseudo-Hessian standard errors, with z, p and intervals
withheld), `:hessian` (the naive Wald table, opted into) or `:bootstrap`. The convergence report of an MCMLE fit is
an `ERGM.MCMLEConvergence` in `fit.mcmc_convergence`. A single term or a
tuple of terms is accepted in place of the vector; a `Network` or a rank
matrix in place of the `RankNetwork`, and a binary ERGM.jl term in the term
list, are refused with an `ArgumentError` naming the fix.

```@docs
fit_ergm_rank
ergm_rank
```

## Data

```@docs
newcomb_week1
```

## Result metadata

The shared result-metadata protocol of NetworkCore.jl (`fit_metadata`,
`approximations`, `estimand`, `objective`, `is_exact`, `se_method`,
`missing_method`) is exported by ERGMRank, so `approximations(fit)` works
with just `using ERGMRank`. `estimand(fit)` is `:rank_ergm`,
`missing_method(fit)` is `:none` (a `RankNetwork` has no dyad mask) and
`approximations(fit)` lists, per method, what the estimator did not do
exactly (the Monte-Carlo approximation of the likelihood and the bridge, or
the swap pseudo-likelihood, its anticonservative pseudo-Hessian errors and
the inference withheld from them),
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
[`RankERGMResult`](@ref) and pinned by `NetworkCore.check_statsapi` in the test
suite. `coef`, `stderror`, `vcov`, `loglikelihood` read the result's fields;
`nobs` is the number of (ego, unordered alter pair) swap comparisons of a
`:mple` fit and the `n(n−1)` ordered dyads of a `:mcmle` fit (R's
`nobs.ergm`); `dof` counts the finite coefficients. `loglikelihood` is the
maximized swap pseudo-log-likelihood of a `:mple` fit (`NaN` when no
comparison is left after a boundary drop) and the bridge estimate of the
**absolute** log-likelihood of a `:mcmle` fit (`NaN` under
`bridge_rungs=0`) — R's `logLik` is relative to θ = 0, `n·log((n−1)!)`
higher. `coefnames` returns the term names, the labels `coeftable` prints.
The five below carry rank-specific caveats; `coefnames` is listed with them.

```@docs
nobs(::RankERGMResult)
aic(::RankERGMResult)
bic(::RankERGMResult)
confint(::RankERGMResult)
coeftable(::RankERGMResult)
coefnames(::RankERGMResult)
```

## Simulation

```@docs
simulate_rank_ergm
```

## Goodness of fit

```@docs
gof(::RankERGMResult)
```
