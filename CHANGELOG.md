# Changelog

All notable changes to ERGMRank.jl are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

First public release of ERGMRank.jl: exponential-family random graph models
for rank-order relational data (Krivitsky & Butts 2017), following R's
`ergm.rank`. Term values, the MCMC MLE, its standard errors and its
log-likelihood are checked against `ergm.rank` 4.1.2 (R 4.6.1, ergm 4.12.0)
by provenanced fixtures. Version 0.1.0 was a development version, never
released.

**Dependency renamed:** the foundation package is now `NetworkCore` (developed as `Networks`); write `using NetworkCore` where code said `using Networks`. Types and functions keep their names.

### Highlights

- **`RankNetwork`**: complete rankings (each ego ranks every alter, greater
  value = higher standing, as in `ergm.rank`), validated at construction and
  preserved by `swap_ranks!`. `as_rank_network` converts a rank matrix or a
  directed `NetworkCore.Network` with a rank edge attribute, rejecting what a
  ranking cannot hold (an undirected or two-mode network, a masked dyad, an
  incomplete ranking) and reporting every attribute it drops.
- **The `ergm.rank` terms**: `RankDeference`, `RankNonconformity(:all)` and
  `(:localAND)`, `RankNodeICov`, `RankEdgeCov`, and `RankInconsistency` with
  R's `weights`, `wtname` and `wtcenter` arguments. Each has an exact,
  allocation-free change statistic for the AlterSwap move (`swap_change`).
  The summary statistics are an independent implementation of the published
  term definitions (Krivitsky & Butts 2017), checked against `ergm.rank`'s
  output by fixtures; `RankNonconformity(:all)` is computed in O(n³).
- **MCMC maximum likelihood by default**: `ergm_rank` (= `fit_ergm_rank`)
  fits `ergm.rank`'s estimator — AlterSwap Metropolis sampling, Hummel-stepped
  Newton iterations with a step at every iteration, standard errors that
  include the Monte-Carlo component (R's "MCMC %"), and a path-sampling
  log-likelihood. On Newcomb's fraternity (week 1, two terms; week 2, three
  terms) it reproduces R's coefficients within `ergm.rank`'s own
  seed-to-seed spread and R's standard errors within 15 %.
- **Swap pseudo-likelihood** (`method=:mple`): a fast, deterministic
  estimator of this package, with R's `drop` treatment of a statistic at the
  boundary of its attainable range and a loud "MPLE does not exist" for a
  separated model (decided exactly by NetworkCore's shared separation test;
  the fit names the separated terms in `fit.separated_terms`, reports `NaN`
  z values, p-values and `confint`, and refuses `se=:bootstrap`). It runs on
  ERGM.jl's one pseudo-likelihood fitter: a count observed at 0 is fixed at
  -Inf even when no swap changes it, `drop=false` (R's `control.ergm(drop=)`)
  refuses a boundary statistic instead, and a statistic no swap changes, or
  a linear combination of the others, is `NaN` (R's `NA`) with the rest
  fitted without it — it used to stall every coefficient at 0. It is a
  different estimator from the MLE, and its
  pseudo-Hessian standard errors are too small, so by default it reports no
  z values, p-values or confidence intervals; `se=:bootstrap` gives a
  parametric-bootstrap covariance, and `se=:hessian` opts in to the naive
  Wald table.
- **Simulation and goodness of fit**: `simulate_rank_ergm` (AlterSwap
  Metropolis on the ecosystem's shared kernel, swap-scaled burn-in and
  interval, reproducible from `rng`) and `gof`, which compares the observed
  ranking with simulated ones on the model statistics and on auxiliary
  statistics the model does not contain (omitted structural statistics, the
  received-rank profile, dyadic rank differences, Kendall's τ between egos).
  The draws behind `gof` and the bootstrap are thinned by their effective
  sample size.
- **Shared conventions**: the full StatsAPI surface (`coef`, `coefnames`,
  `stderror`, `vcov`, `confint`, `loglikelihood`, `nobs`, `dof`, `aic`,
  `bic`, `coeftable`; `coefnames(fit)` returns the coefficient labels), the result-metadata protocol (`approximations(fit)` says what
  an estimator did not do exactly), one keyword vocabulary (`maxiter`,
  `n_sim`, `n_boot`, `burnin`, `interval`, `rng`), and results that do not
  depend on the thread count.
- **Nothing fails silently**: non-convergence, a boundary statistic,
  separation, undefined standard errors and excluded bootstrap replicates
  (which bias the bootstrap standard errors downward, as the fit states)
  warn at fit time, are listed by `approximations(fit)` and printed by
  `show`; a fit with a non-finite coefficient cannot be simulated from or
  passed to `gof`.
- `newcomb_week1()`: week 1 of Newcomb's fraternity rankings, the data the
  estimator claims rest on.
- **Built on ERGM.jl's extension API** (`ERGM.Extension`): the rank terms
  declare their attainable ranges as methods of
  `ERGM.Extension.attainable_range` on a `RankNetwork` (no range table of
  their own), and the swap MPLE reads the separation verdict
  `mple_fit_design` returns, so the separation check runs once per fit.

### Differences from R `ergm.rank`

- `loglikelihood`, `aic` and `bic` of an MCMLE fit are absolute; R's `logLik`
  is relative to the uniform-ordering model θ = 0. They differ by the
  constant `n·log((n−1)!)` (and `−2n·log((n−1)!)` for AIC/BIC); differences
  between models on the same ranking agree. `nobs` is R's: `n(n−1)`.
- Covariates and reference rankings are passed as values (a vector, a
  matrix, a `RankNetwork`), not as attribute names.
- Term names carry a `rank.` prefix (`rank.deference`, `rank.nonconformity`,
  `rank.nonconformity.localAND`, `rank.nodeicov.<label>`,
  `rank.inconsistency[:<wtname>[c]]`, `rank.edgecov.<label>`).
- A statistic at an end of its attainable range is refused by the MCMLE
  with an explanation; one extreme only among the single swaps of the
  observed ranking, an unidentified one or a separated starting value is
  refused the same way (or run from a user-supplied `init=` with a warning).
- The swap change statistics are derived independently from the term
  definitions as a swap move, which has no counterpart in `ergm.rank`'s
  single-cell change functions.
- `method=:mple`, `se=:bootstrap` and the auxiliary panels of `gof` have no
  counterpart in `ergm.rank`.

### Known limitations

- **`rank.nonconformity`'s `local1`, `local2`, `geometric` and `thresholds`
  variants** are not implemented; `RankNonconformity` accepts `:all` and
  `:localAND` and refuses anything else with an `ArgumentError`.
- **Attribute-name arguments** (`rank.nodeicov("age")`, `rank.edgecov(x,
  "attr")`, `rank.inconsistency(x, "attr")`) are not implemented: pass the
  values. An attribute name is refused with an `ArgumentError`.
- **Effective-size-adaptive MCMC sampling** is not implemented: the MCMLE
  stops by R ergm 4's confidence rule and boosts its sample as R does, but
  each sample is `n_samples` draws at a fixed `interval`.
- **Partial, tied or partly unobserved rankings** cannot be represented or
  fitted; the `Network` adapter refuses a masked dyad.
- **`offset()` terms, `constraints=` and MCMC diagnostics plots** are not
  implemented.
- **Inference from the swap pseudo-likelihood's Hessian** is not calibrated:
  a `method=:mple` fit reports no z, p or confidence interval by default, and
  `se=:bootstrap` describes the swap-MPLE, not the MLE.
- **R's drop under the MCMC MLE** is not implemented: the rankings sharing
  an extreme statistic are not connected by single swaps, so the MCMLE
  refuses a count observed at 0 with an `ArgumentError` and points at
  `method=:mple`, which applies the drop.
