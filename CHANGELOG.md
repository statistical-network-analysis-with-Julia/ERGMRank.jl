# Changelog

All notable changes to ERGMRank.jl are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

Release driven by the 2026-07 expert-panel review: the package is rebuilt
around R `ergm.rank`'s complete-ordering model (Krivitsky & Butts), with a
real swap-based pseudo-likelihood estimator, the ergm.rank term set, and the
ecosystem-wide StatsAPI/GOF conventions.

### Breaking

- **`RankNetwork` redesigned.** Previously a parametric wrapper around a
  `Network{T}` with a sparse `ranks::Dict` and `max_rank`, forwarding Graphs
  methods; now a non-generic struct holding `n` and a dense
  `ranks::Matrix{Int}` with a per-ego complete-ordering invariant (ranks
  `1:(n-1)` per ego). `RankNetwork(n)` returns a fully populated index-order
  ranking (was empty). *Migration:* construct from a full rank matrix
  (`RankNetwork(mat)` / `as_rank_network(mat)`); Graphs methods (`nv`, `ne`,
  `edges`) are gone.
- **Rank orientation flipped to match R ergm.rank:** greater value now means
  higher standing (`y[i,j] > y[i,k]` means i ranks j over k); previously a
  lower rank number meant higher preference. *Migration:* invert comparisons
  and covariate signs built against the old convention.
- **No more `nothing`/`missing` ranks:** `get_rank` returns `Int` (0 on the
  diagonal, was `Union{Int,Nothing}`); `rank_matrix` returns `Matrix{Int}`;
  `as_rank_network` takes a complete `Matrix{Int}` (no `max_rank` keyword,
  no `missing` support). *Migration:* stop pattern-matching
  `nothing`/`missing`; supply complete orderings.
- **Term set replaced by the ergm.rank terms.** Removed with no drop-in
  replacement: `RankEdges`, `RankMutual`, `RankTransitivity`,
  `RankNonconsensus`, `RankLocaltriangle`, `RankNodecov`, `RankAbsdiff`, and
  the `PlackettLuce`/`ThurstoneMosteller` reference structs. `RankDeference`
  keeps its name but is redefined to the ergm.rank triple count (its
  `cutoff` field is gone). *Migration:* re-express models with the new term
  set (`RankNonconformity`, `RankNodeICov`, `RankInconsistency`,
  `RankEdgeCov`, `RankDeference`).
- **`simulate_rank_ergm` signature and return changed** to
  `simulate_rank_ergm(rnet, terms, θ; n_sim, burnin, interval, rng) ->
  Vector{RankNetwork}` (AlterSwap Metropolis), plus a convenience method on
  `RankERGMResult`; previously `(n, terms, coef; max_rank, burnin)` returned
  a single random-toggle placeholder network. *Migration:* pass a starting
  `RankNetwork` and expect a vector.
- **`method=` is back with two values, and the swap-MPLE keywords moved
  behind it.** `fit_ergm_rank(rnet, terms; method=:mple)` (the default) is
  the swap pseudo-likelihood; `method=:mcmle` is the new MCMC MLE (see
  Added). `maxiter` now defaults per method (`nothing` → 100 Newton
  iterations under `:mple`, 20 MCMLE iterations under `:mcmle`), `se`
  defaults per method (`nothing` → `:hessian` / `:fisher`) and is validated
  against that method's vocabulary — `se=:hessian` or `se=:bootstrap` with
  `method=:mcmle`, and `se=:fisher` with `method=:mple`, are
  `ArgumentError`s from `Networks.check_se` naming the allowed tuple. An
  unknown `method` is an `ArgumentError` naming `(:mple, :mcmle)`.
  *Migration:* none for code that passed no `method`; the 0.1 spelling
  `method=:mple` works again.
- **`RankERGMResult` gained five more fields** for the MCMLE: `method::Symbol`
  (`:mple`/`:mcmle`), `mcmc_convergence::Union{Nothing,ERGM.MCMLEConvergence}`,
  `mcmc_samples::Union{Nothing,Matrix{Float64}}` (the statistics sampled at
  the returned coefficients), `vcov_fisher::Union{Nothing,Matrix{Float64}}`
  (the Fisher part of `vcov`) and `bridge_rungs::Int`; the 6-, 7- and
  9-argument positional constructors still build a `:mple` result, and the
  inner constructor refuses a `method` other than the two. `Networks.objective`
  now answers `:likelihood` for a `:mcmle` result (`:pseudolikelihood` for
  `:mple`, as before) and `se_method` can answer `:fisher`. *Migration:* code
  that constructed the struct positionally with all fields must pass the five
  new ones.
- **A separated or boundary model no longer "converges" silently.** A
  statistic at the boundary of its attainable range (no single swap can lower
  — or raise — it; `RankInconsistency(rnet)` fitted to `rnet` itself) used to
  return a large finite coefficient with a meaningless standard error and
  `converged == true`; it now gets R `ergm`'s `drop=TRUE` treatment
  (coefficient `∓Inf`, standard error 0, the rest estimated on the untouched
  comparisons, a warning quoting R's "at their smallest attainable values").
  A perfectly separated model — a *combination* of statistics no swap lowers;
  the 4-actor README network under `RankDeference() + RankNonconformity()` is
  one, and reported θ ≈ (9.4, 10.1) with standard errors of 7,066 — now
  returns `converged == false` with R's "The MPLE does not exist!" warning.
  *Migration:* a fit that used to look fine with absurd standard errors will
  now warn and report `converged == false`; drop the offending term or use
  more actors. `dof(fit)` counts only finite coefficients.
- **`RankERGMResult` gained two fields**, `boot_replicates::Union{Nothing,
  Matrix{Float64}}` (the `n_boot × p` refits of `se=:bootstrap`, NaN rows for
  excluded replicates; `nothing` otherwise) and `n_kept::Int` (the swap
  comparisons the finite coefficients were estimated on). The 6- and
  7-argument positional constructors still work. *Migration:* code that
  constructed the struct positionally with all fields must pass the two new
  ones.
- **`as_rank_network` refuses `missing`/`nothing` and non-integer entries**
  with an `ArgumentError` (previously a raw `MethodError`/`InexactError` from
  `Matrix{Int}`). `fit_ergm_rank` refuses an empty term list and a network with
  fewer than 3 actors (previously an empty design reached the optimizer);
  `simulate_rank_ergm` and `gof` refuse `n_sim < 1`. The `se=` validator is the
  shared `Networks.check_se`, so the message is now
  `fit_ergm_rank: se must be one of (:hessian, :bootstrap) (got :x)`.
- **Combinatorics removed from the dependencies** (declared, never used; panel
  item 29). *Migration:* none for users; an environment that relied on
  ERGMRank to pull it in must add it itself.
- **Sampler burn-in and thinning defaults now scale with the ranking size**
  (panel item 24e). `simulate_rank_ergm`, `gof` and `fit_ergm_rank`'s
  `boot_burnin`/`boot_interval` default to `nothing`, resolved by the
  ecosystem's one dyad-scaled rule `ERGM._mcmc_defaults` applied to the
  number of swaps `n(n−1)(n−2)/2` — the size of the AlterSwap proposal space,
  the rank analogue of the dyad count: `burnin = 20 n_swaps`, `interval =
  max(100, n_swaps ÷ 10)` (`ERGMRank._mcmc_defaults(rnet)`). Was a fixed
  `500`/`50`, which for a 17-actor ranking (2,040 swaps) burned in for a
  quarter of one sweep. Now 240/100 for 4 actors, 3,360/100 for 8, 40,800/204
  for 17. *Migration:* an explicit integer is honoured exactly as before;
  a default run draws from a longer (better-mixed) chain, so a seeded
  simulation, `gof` or `se=:bootstrap` result obtained with the old defaults
  is reproduced by passing `burnin=500, interval=50` (`boot_burnin=500,
  boot_interval=50`) explicitly.
- **Display follows Networks.jl's two-method convention.** Two-argument
  `show` on a `RankNetwork` or a `RankERGMResult` (what `print`, `string`
  and containers use) is now the ONE-LINE form — `RankNetwork(4 actors)`,
  `RankERGMResult(swap-MPLE, 2 terms, converged)` — and the R-style block
  (rank matrix; estimator, log-likelihood, `Converged:`, the coefficient
  table, caveats) moved to the `MIME"text/plain"` method the REPL calls.
  *Migration:* `println(fit)` now prints one line; use `display(fit)` or
  `show(io, MIME("text/plain"), fit)` for the block (the test suite's
  `sprint(show, fit)` assertions were changed accordingly). A
  `Vector{RankNetwork}` of simulated draws no longer prints one block per
  element.
- **`fit_rank_ergm` is a deprecated binding that warns on access**
  (`Base.@deprecate_binding` semantics: still `=== fit_ergm_rank`, and every
  access prints `WARNING: Use of ERGMRank.fit_rank_ergm is deprecated, use
  fit_ergm_rank instead.` under `--depwarn=yes`, which `Pkg.test` sets; off
  by default in an interactive session). Removed in 0.3. It was documented as
  deprecated but was a silent `const` alias, so a user had no nudge before
  the removal. *Migration:* call `fit_ergm_rank` or `ergm_rank`.
- **`nobs` of a `method=:mcmle` fit is R's dyad count `n(n−1)`**, not the
  swap-comparison count `n(n−1)(n−2)/2` (which `nobs` of a `:mple` fit still
  returns — the pseudo-likelihood's observations). R `ergm.rank`'s
  `nobs.ergm` is `network.dyadcount`, so `bic` of an MCMLE fit now uses
  `log(272)` on Newcomb's 17 actors where it used `log(2040)`; `fit.n_kept`
  of a `:mcmle` result is `n(n−1)` too. *Migration:* a `bic` value of a
  `:mcmle` fit changes by `dof·(log(n(n−1)) − log(n(n−1)(n−2)/2))`; `:mple`
  values are unchanged.
- **The pseudo-log-likelihood of a fit with no comparison left is `NaN`, not
  `0.0`.** When every swap comparison changes a statistic dropped at the
  boundary of its attainable range (`n_kept == 0` — the other coefficients
  are then NaN, "not identified", or every term was dropped), `loglikelihood`,
  `aic` and `bic` are `NaN` and `show` prints `Pseudo-log-likelihood: not
  defined (…)` / `Pseudo-AIC: not defined, pseudo-BIC: not defined`. They
  used to be the `0.0` of an empty product, which printed as
  "Pseudo-AIC: 0.0" — a perfect fit — beside a NaN coefficient and
  `Converged: false`. *Migration:* test `isnan(loglikelihood(fit))` (or
  `fit.n_kept == 0`) instead of `== 0.0`. When every column is dropped but
  some comparisons change no statistic (`n_kept > 0`), the value is the
  exact limit `n_kept·log(½)` (each such row has probability σ(0) = ½) —
  it used to be the `0.0` of an empty product there too, printing
  "Pseudo-log-likelihood: 0.0 / Pseudo-AIC: 0.0" for e.g. `RankDeference()`
  alone on a ranking where the statistic is at its minimum while six swaps
  leave it unchanged.
- **Error texts that used to omit their context now name it**: a
  wrongly sized covariate says which term (`RankNodeICov("x"): the
  covariate has 3 values but the network has 4 actors; pass one value per
  actor, in actor order`; `RankEdgeCov(...)`/`RankInconsistency: ... is
  3×3 but the network has 4 actors; pass a 4×4 matrix ...`) instead of
  `covariate length 3 ≠ number of actors 4`; an invalid rank matrix says
  `RankNetwork: not a valid complete ranking — ego 1's ranks [1, 3, 3] are
  not a permutation of 1:3; ...`; and `RankNonconformity(:local1)` says
  `RankNonconformity: variant must be :all or :localAND (got :local1);
  ergm.rank's local1/local2/geometric/thresholds variants are not
  implemented in ERGMRank.jl` (was `variant must be :all or :localAND`).
  *Migration:* code matching those strings must match the new ones.
- **Minimum Julia raised to 1.12**; package UUID regenerated. *Migration:*
  upgrade Julia and re-resolve environments pinning the old UUID.

### Added

- **Provenanced golden fixture for every term statistic**:
  `test/fixtures/rank_terms.toml`, generated by `test/fixtures/r/rank_terms.R`
  (a `summary()` call, seconds), freezes ergm.rank 4.1.2's `rank.deference`,
  `rank.nonconformity("all")`, `rank.nonconformity("localAND")`,
  `rank.nodeicov`, `rank.inconsistency` and `rank.edgecov` on the 4-actor
  README network (`6, 10, 4, −40, 6, −40`) and on a seeded random 8-actor
  ranking with integer covariates and a reference ranking (every input frozen
  in `[values]`), asserted with `check_golden` at 1e-9. Four of the six
  values used to be pinned by bare literals with no `[provenance]`; the
  README's "validated against R ergm.rank 4.1.2 on golden-master fixtures"
  is now true of all six.
- **`newcomb_week1()`** (exported): Newcomb's fraternity, week 1 — R
  `ergm.rank`'s `newcomb[[1]]` as a 17-actor `RankNetwork`, verbatim the
  ranking the golden fixture froze (the test suite asserts the two are
  identical), so every Newcomb number the README and the estimation guide
  quote (swap-MPLE `[-0.1409, -0.00585]`, MCMLE `[-0.1531, -0.00659]`,
  statistics `(844, 12748)`) is reproducible by a reader; the guide gained a
  "Newcomb week 1" section fitting both estimators on it. (A
  `Networks.load_dataset(:newcomb)` has been requested cross-repo; this is the
  loader until then.)
- **A "Coming from `ergm.rank`" table** in the README and the Getting
  Started page: the R fit call with `response=`/`reference=`/`control.ergm`
  against the `ergm_rank(...; method=:mcmle, n_samples, burnin, interval)`
  call, `summary`/`simulate`/`gof`, `newcomb[[1]]`, `control.ergm(seed=)` →
  `rng=`, and the two non-one-to-one points: R's default estimator is
  `method=:mcmle` here, and the log-likelihood convention (below).
- **The `logLik` convention is stated, and asserted against R.** R
  `ergm.rank`'s `logLik(fit)` is *relative* to the uniform-ordering model
  θ = 0 (whose likelihood `ergm` defines as 0 for a valued ERGM: "Null model
  likelihood calculation is not implemented for valued ERGMs"), while
  `loglikelihood(fit)` of a `:mcmle` fit is the *absolute* bridge estimate
  `θ̂'g(y) − log Z(θ̂)` with `log Z(0) = n·log((n−1)!)` exact: `R's logLik =
  loglikelihood(fit) + n·log((n−1)!)`, AIC/BIC shift by `−2n·log((n−1)!)`
  (≈ 1042 on Newcomb), differences between models on the same ranking are
  unaffected. The `aic`/`bic` docstring, the estimation guide, the README
  and the migration table say so (the docstring and guide used to claim
  the numbers "estimate the quantities R's logLik/AIC estimate", which was
  false by that constant). `test/fixtures/r/newcomb_rank.R` now records
  `mle_loglik` (R's `logLik`), `mle_loglik_mc_se` (its "MC Std. Err."),
  `mle_df`, `mle_nobs` (272 = 17·16), `mle_aic`, `mle_bic`, and the fixture
  testset runs one MCMLE with `bridge_rungs=16` (`bridge_samples=512`) and
  asserts `loglikelihood(mle) + 17·log(16!) ≈ mle_loglik` within
  `4 × (mle_loglik_mc_se + julia_bridge_sd)` — `julia_bridge_sd` (1.9e-2)
  being ERGMRank's bridge over five Julia seeds, frozen in `[tolerance]`
  with the five values — plus `nobs`, `dof` and the AIC/BIC shift.
- **`nobs(::RankERGMResult)` has its own docstring** stating the two
  conventions (comparisons under `:mple`, R's dyads under `:mcmle`).
- **Actionable errors for four more `ergm.rank`-user mistakes** (each quoted
  in the README's "Common mistakes"): a binary ERGM.jl term in a rank model
  (`fit_ergm_rank: Edges is not a rank term; a rank model takes
  RankDeference, RankNonconformity, RankNodeICov, RankInconsistency,
  RankEdgeCov …` — from `fit_ergm_rank` and `simulate_rank_ergm`, checked
  up front by `_check_rank_terms` through the term contract `compute` +
  `swap_change`, instead of a `MethodError` from inside the design builder);
  a `Network` or a rank matrix where a `RankNetwork` is expected (`the first
  argument is a Network{…}, not a RankNetwork; call as_rank_network(...)
  first …`); a covariate given as an attribute name, ERGM.jl's
  `NodeCov(:age)` spelling (`RankNodeICov(:age): a RankNetwork carries no
  vertex attributes, so pass the covariate VALUES themselves … e.g.
  RankNodeICov(vertex_attribute_vector(net, :age, Float64); label="age")`;
  `RankEdgeCov(:x)` likewise). A single term or a tuple of terms in place
  of the vector is now accepted by `fit_ergm_rank`/`ergm_rank` and
  `simulate_rank_ergm` (`ergm_rank(rnet, RankDeference())`).
- **`simulate_rank_ergm` accepts any real `θ`** (`AbstractVector{<:Real}`:
  `[0]`, a range) — it always converted internally, but the signature
  refused an integer vector with a raw `MethodError`.
- **The Newton-decrement convergence criterion lives in `Networks.newton_fit`**
  (reconciliation round): the shared kernel declares convergence on
  ½∇ℓ'(−H)⁻¹∇ℓ < tol beside the gradient norm, so ERGMRank's
  `_stalled_at_optimum` shim and its direct testset are deleted; the testset
  "newton_fit converges on the swap design without a local shim" pins the
  kernel's verdict on the Newcomb design (converged, decrement < tol,
  ‖∇ℓ‖ < 1e-3) and that the shim no longer exists. The Newcomb swap-MPLE
  literal pinned at 1e-12 moved by 5e-12 (`-0.14090974899347272,
  -0.005853837701704666`): the kernel now ends at the polished maximum
  instead of the iterate one step short that the shim declared converged. The boundary sentence
  now says "swap comparisons" for the rows the remaining coefficients are
  fitted on and that ergm.rank has no drop of its own (`_warn_boundary`'s
  `noun=`/`note=`); the six ERGM helpers ERGMRank calls are `public` and the
  reach-in testset asserts it instead of holding them `@test_broken`.
- **Docstring examples are asserted, not just executed**: a top-level line
  ending in `# true` / `# false` (with or without a trailing explanation)
  is now checked against its claim in the "Every exported docstring carries
  a runnable example" testset (40+ such claims), and every example must
  leave stderr silent (the deprecated `fit_rank_ergm`'s example, which
  demonstrates the warning, is the one exception). The `fit_ergm_rank` and
  `coeftable` examples used to run `RankDeference() + RankNonconformity()`
  on the 4-actor network — the separated design the package correctly
  reports with `converged == false` — under a `fit.converged # true`
  comment; both now fit the well-posed `[RankDeference(),
  RankNodeICov([10, 20, 30, 40])]`.
- **Provenanced golden fixture against a real `ergm.rank` fit** (issue #8).
  `test/fixtures/newcomb_rank.toml` freezes an ergm.rank 4.1.2 MCMLE of
  `newcomb[[1]] ~ rank.deference + rank.nonconformity("all")` under the
  CompleteOrder reference, regenerable with `Rscript
  test/fixtures/r/newcomb_rank.R > test/fixtures/newcomb_rank.toml` (**slow**:
  ~17 min, six MCMLE fits).

  **The fixture asserts the MCMC MLE and characterises the swap-MPLE.**
  `method=:mcmle` (below) is `ergm.rank`'s estimator: run at `ergm.rank`'s own
  MCMC budget (read from the fixture's `mcmc_control`: 2048 draws, burn-in
  8192, interval 512) on two Julia seeds, its coefficients are **asserted**
  against `mle_coefficients` within `4 × (mle_seed_sd + julia_seed_sd)` — R's
  seed-to-seed sd (7.4e-4 / 5.6e-5) plus Julia's (5.1e-4 / 7.2e-5, measured
  over five seeds and frozen in the `[tolerance]` block with the five
  estimates and the justification) — and its standard errors within 15 % of
  `mle_std_errors` (observed: under 4 %; both include the MCMC-error
  component). The Julia MCMLE's mean sits 0.8 and 0.7 R seed-sd from R's
  estimate; the largest deviation over five seeds was a quarter of the band.

  `method=:mple` fits a swap pseudo-likelihood, whose overlapping (ego,
  alter-pair) comparisons are multiplied as if independent — a different
  estimator, so its coefficients are not tolerated at a convenient atol, and
  what the testset *asserts* is the character of the gap:

  - The **observed sufficient statistics match exactly** (844, 12748) — asserted
    at 1e-9. The term formulas are right.
  - The gap is **systematic, not Monte-Carlo**: swap-MPLE `[−0.14091, −0.0058538]`
    against ergm.rank's `[−0.15310, −0.0065927]`, which is **16x and 13x**
    ergm.rank's own seed-to-seed sd (asserted > 5). No MCMC budget closes it —
    the MCMLE at the same budget sits inside 1 R sd.
  - But it is **small where it counts**: 0.30 and 0.43 of an ergm.rank standard
    error (asserted < 0.6) — same sign, same order, same substantive story. A
    different estimator, not a broken one.
  - The inverse-pseudo-Hessian standard errors are **anticonservative by a
    measured factor**: 3.9x (deference) and 2.0x (nonconformity) narrower than
    the MLE's. `se=:bootstrap` recovers most of it (0.0280 against R's 0.0404),
    and the testset asserts that it does.

- **MCMC maximum likelihood estimation: `fit_ergm_rank(rnet, terms;
  method=:mcmle)`** (finding N4, closed) — the estimator of R `ergm.rank`,
  on `ERGM.mcmle`'s iteration with the AlterSwap chain as the sampler and
  the swap-MPLE as the start: Hummel step length (`gamma0`, at most doubling
  per iteration, 1 once the observed statistics lie inside the 95 %
  Mahalanobis radius of the sampled cloud), `max_step_norm` cap,
  singular-covariance stop, convergence by `ERGM.mcmc_convergence`
  (per-statistic t-ratios < `conv_threshold`, Hotelling T² with the Geyer
  effective sample size > `hotelling_alpha`, only at full step length), the
  report recorded as `fit.mcmc_convergence::ERGM.MCMLEConvergence` and the
  final sample as `fit.mcmc_samples`. Keywords are `ERGM.mcmle`'s, spelled
  the same and pinned against its `kwarg_decl`: `n_samples=1024`,
  `burnin`/`interval` (`nothing` → the swap-scaled rule), `maxiter=20`,
  `n_chains=1`, `init` (default: the swap-MPLE), `gamma0`, `max_step_norm`,
  `conv_threshold`, `hotelling_alpha`, `bridge_rungs=16`, `bridge_samples`,
  `verbose`, `rng`.
  - **Standard errors (`se=:fisher`)**: `vcov = V + V·Σ_mc·V` with `V = Σ̂⁻¹`
    the inverse covariance of the final sample and `Σ_mc` the Geyer
    initial-sequence covariance of its mean — the Monte-Carlo component of
    the estimating equation (Hunter & Handcock 2006 §3.3), through
    `ERGM._mcmle_covariance` (the same numbers `ERGM.mcmle` reports).
    `fit.vcov_fisher` is `V` alone and `show` prints R's `summary.ergm`
    "MCMC %" column.
  - **Log-likelihood by path sampling**: `loglik = θ̂'g_obs − [log Z(θ̂) −
    log Z(0)] − n·log((n−1)!)`, the ratio integrated along `θ_u = u·θ̂` by
    the trapezoid rule over `bridge_rungs` segments with one seeded chain per
    grid point (`ERGM._bridge_logZ`'s estimator with the uniform ordering
    model — exact normalizer — as the reference), so `loglikelihood`, `aic`
    and `bic` of a `:mcmle` fit are the absolute log-likelihood and its
    information criteria; R's `logLik`/`AIC` differ by the constant
    `n·log((n−1)!)` — see the logLik-convention entry above. It runs after
    the final sample (coefficients and SEs are bit-identical
    with and without it); **`bridge_rungs=0` skips it** and the three are
    `NaN`, `show` prints "not estimated", `approximations` records it. At
    n = 4 the test suite checks the bridge against exact enumeration of all
    (3!)⁴ = 1296 orderings (< 0.05 at 32 rungs), the MCMLE against the exact
    MLE (< 0.25 SE) and `vcov_fisher` against the exact Fisher information.
  - **Reproducible and thread-count independent**: one chain runs on the
    caller's `rng`; `n_chains > 1` draws one seed per chain from `rng`, runs
    each on its own task with its own `Xoshiro`, and concatenates in chain
    order — a fit depends on `rng` and `n_chains` only, pinned bit for bit
    (and chain by chain against the drawn seeds).
  - **Loud non-convergence** in the ERGM pattern: a warning after the loop
    quoting the last max t-ratio, Hotelling p and step length; the same
    numbers in `approximations(fit)` and under `Converged: false` in `show`;
    continue with `init=coef(fit)`. A start that already passes both tests
    is returned unchanged, as `ERGM.mcmle` returns its MPLE start.
  - **Refusals**: a boundary statistic (the swap-MPLE start would be ±Inf)
    is an `ArgumentError` with R's sentence unless `init=` is supplied (then
    a warning: the MLE may not exist, the tests decide); a separated
    swap-MPLE start is refused pointing at `init=`; `n_samples < 2`,
    `n_chains < 1`, `maxiter < 1`, `bridge_rungs < 0`, an `init` of the wrong
    length or non-finite, are `ArgumentError`s.
  - **Result metadata**: `objective(fit) == :likelihood`, `se_method(fit) ==
    :fisher`, `is_exact(fit) == false` (a Monte-Carlo estimate: the docstring
    gives both reasons), `approximations(fit)` lists the MCMC approximation
    of the likelihood (Monte-Carlo error included in the SEs), the bridge (or
    its absence) and any non-convergence. `show` prints `Estimation: MCMC MLE
    (AlterSwap Metropolis, K bridge rungs)`, `Log-likelihood: … (path-sampling
    bridge estimate)`, `AIC:`/`BIC:` (not "Pseudo-"), the Fisher + Monte-Carlo
    standard-error line and the MCMC % column, and no pseudo-likelihood
    caveat.
  - **Cost**: (iterations + 1 + bridge_rungs + 1) chains of `burnin +
    n_samples × interval` steps, each step O(n)–O(n²) per term (the sampler
    keeps the running statistics current from the accepted swaps' change
    statistics — no `compute` per draw). The 17-actor Newcomb fit at R's
    budget converges in 2–3 iterations, ~25 s without the bridge.

- **Robust standard errors: `fit_ergm_rank(rnet, terms; se=:bootstrap)`** (also
  via `ergm_rank`/`fit_rank_ergm`), with the same keywords and semantics as
  `ERGM.mple`'s: `n_boot=100`, `boot_burnin`, `boot_interval`, `rng`. Simulate
  `n_boot` rank networks at θ̂ with the AlterSwap Metropolis sampler
  (`simulate_rank_ergm`), refit the swap MPLE on each, and report the empirical
  covariance — on the ONE shared `Networks.bootstrap_cov` loop. **The point
  estimates are unchanged; only the covariance is replaced.** This matters more
  here than anywhere: the swap pseudo-likelihood's comparisons are *explicitly*
  not independent (each ranking enters n − 2 of them), so the inverse-Hessian SEs
  are anticonservative for every rank fit, with no exact special case to exempt
  — and they were printed with significance stars (issue #9, ERGMRank#1). On the
  test fixture the bootstrap SEs are **more than 2× larger** than the Hessian
  ones on every coefficient.
- `se_method(fit)` now reports what was actually used (`:hessian`/`:bootstrap`),
  read off the new `RankERGMResult.se_type` field, and `approximations(fit)` and
  `show` drop the anticonservatism caveat when a bootstrap was used. The
  *point-estimate* caveat (swap pseudo-likelihood, no consistency claimed) stays
  in both, because `is_exact` is unconditionally false for a rank fit.

- ergm.rank-faithful terms ported from `wtchangestats_rank.c`:
  `RankNonconformity(:all | :localAND)`, `RankNodeICov`,
  `RankInconsistency`, `RankEdgeCov`; `CompleteOrderReference` reference
  measure (discrete-uniform over complete orderings).
- `fit_ergm_rank` as the canonical entry point; `ergm_rank` (R-faithful) and
  `fit_rank_ergm` (legacy) kept as `const` aliases.
- Rank-manipulation API: `get_rank`, `set_rank!`, `swap_ranks!` (the
  AlterSwap elementary move), `is_valid_ranking`.
- `gof(::RankERGMResult)` extending the ecosystem-wide `Networks.gof`,
  simulating via AlterSwap and returning a `Networks.GOFResult`.
- **The full StatsAPI surface** (panel item 15): `coef`, `stderror`, `vcov`,
  `confint` (Wald, `level=`), `loglikelihood`, `nobs` (ego ×
  unordered-alter-pair comparisons), `dof` (finite coefficients), `aic`/`bic`
  (the **pseudo**-AIC/BIC of the swap pseudo-likelihood, `bic` on the kept
  comparisons — not comparable to R's bridge-sampled `logLik`) and `coeftable`
  (a `Networks.CoefficientTable`, the very table `show` prints, with p-values
  from `Networks.z_pvalues`; a fixed coefficient shows z = ∓Inf, p = 0 as R
  does). Pinned by `Networks.check_statsapi(fit; strict=true)`.
  `RankERGMResult` gains a `vcov` field.
- **R's `drop` semantics for boundary statistics** (see Breaking):
  `ERGM._boundary_columns_iterated` on the swap design, `∓Inf`/SE 0, the rest
  fitted on the untouched rows, `n_kept` on the result, `approximations(fit)`
  and `show` name the fixed coefficient; `se=:bootstrap` refused with an
  `ArgumentError` when a coefficient is fixed. If every comparison changes a
  dropped statistic, the remaining coefficients are NaN, `converged == false`,
  with a "not identified" warning — never a Newton "solution" on an empty
  design.
- **Bootstrap replicates without a finite refit are excluded** (a simulated
  ranking that puts a statistic at the boundary of its attainable range, a
  separated replicate): NaN rows in `fit.boot_replicates`, excluded from the
  covariance, one warning naming the count, an `approximations` entry; fewer
  than 2 finite refits is an `ArgumentError`. Previously one such replicate
  made every standard error NaN. The refits run with `warn=false` (no
  boundary sentence per simulated replicate).
- **Actionable errors** for the common mistakes: a rank matrix with
  `missing`/`nothing` (a `RankNetwork` holds complete orderings and a rank has
  no face value — the message points at the Networks.jl missing-data guide),
  non-integer ranks, fewer than 3 actors (same sentence from the fitter and the
  sampler), an empty term list, `n_sim < 1`, a θ/terms length mismatch, and
  the invalid-ranking message now names the ego at fault.
- Keyword-vocabulary tests: `fit_ergm_rank`, `simulate_rank_ergm` and `gof`
  expose `maxiter`/`n_sim`/`rng` (never `max_iter`, `n_sims`, `seed`),
  checked through `Base.kwarg_decl`; `fit_rank_ergm === fit_ergm_rank ===
  ergm_rank` pinned.
- **The `Network` → `RankNetwork` adapter** (panel items 4 and 13; the
  invariant table's footnote used to say "there is no `Network`→`RankNetwork`
  adapter today"): `as_rank_network(net::Network; attr=:rank,
  missing=:error, report=false)` — the Julia counterpart of R's
  `as.matrix(nw, attrname = "rank")` on `ergm.rank`'s `newcomb` — honours
  the ecosystem conversion contract. **Preserved**: the actor set and the
  ranks (each arc `i → j`'s edge attribute `attr`). **Rejected** with an
  `ArgumentError` that says why: an undirected network (rankings are
  ego-specific), a two-mode network, a masked (unobserved) dyad —
  `require_observed(net, missing; context="as_rank_network")` with
  `supports_missing(as_rank_network) == true` and
  `missing_policies(as_rank_network) == (:error,)`: **no `:face` is
  offered**, because an unobserved rank has no face value to condition on,
  and asking for `missing=:face` is refused with that explanation — a dyad
  without an arc or without the attribute, a non-integer rank, and an ego
  whose ranks are not a permutation of `1:(n-1)` (the message names the
  ego and the attribute). **Dropped and reported**: with `report=true` the
  call returns `(rnet, ConversionReport)` with one `record_drop!` entry per
  other edge attribute, vertex attribute, network attribute and self-loop
  arc; `is_lossless` is `true` exactly when `attr` was the only attribute.
  Tested on a directed 5-actor `Network` (round trip through `rank_matrix`,
  every refusal, `dropped_fields`/`is_lossless`); the testset is what the
  new `Network`→`RankNetwork` column of Networks.jl's invariant table cites.
- **`using ERGMRank` suffices for the documented workflow.** The statistic
  protocol `compute`/`name` and the result-metadata protocol
  (`fit_metadata`, `approximations`, `estimand`, `objective`, `is_exact`,
  `se_method`, `missing_method`) are exported — the same Networks.jl bindings
  every fitting package exports (`ERGMRank.compute === Networks.compute`
  pinned; `using ERGM: compute` is gone from the README and the docs).
  Co-loading is pinned in a fresh process: `using ERGM, ERGMRank` leaves
  `compute`, `name`, `gof`, `coeftable`, `fit_ergm_rank` defined and
  single-owned (and, when the sibling checkouts and the root development
  environment are present, `ERGMCount`/`ERGMMulti`/`TERGM` are co-loaded
  too — a sibling that does not load on its own is left out with an
  `@info`, since that is not a co-loading defect).
- **Every export has a docstring with a runnable `# Example`** — added to
  `RankNetwork`, `RankNetwork(n)`, `get_rank`, `set_rank!`, `swap_ranks!`,
  `is_valid_ranking`, `rank_matrix`, `CompleteOrderReference`,
  `RankERGMModel`, `RankERGMResult`, the five terms, `ergm_rank`,
  `fit_rank_ergm`, `objective`, `is_exact` and `se_method` — and a
  "Every exported docstring carries a runnable example" testset (ERGM.jl's)
  evaluates every ```julia block of every exported docstring in a fresh
  module that has done nothing but `using ERGMRank`.
- **The reach-in test scans the source**: every `ERGM._x`/`Networks._x`
  mentioned in `src/ERGMRank.jl` must be defined and `public`; the six
  requested in ERGM.jl (`_boundary_columns_iterated`, `_warn_boundary`,
  `_separated`, `_mcmle_covariance`, `_warn_degenerate_stats`,
  `_bridge_logZ`) are `@test_broken` until they land, and any new private
  reach-in fails outright.
- README "Common mistakes" (the exact error text of each: undirected
  network, incomplete ranking, masked dyad, `se=:sandwich`, unconverged
  fit, wrong covariate length, too few actors, an unported nonconformity
  variant) and "Known limitations" sections; a "From a `Network`" section
  in the rank-networks guide and the README; the API pages list the new
  exports, the display convention and the result-metadata methods.
- **`swap_change(term, rnet, ego, j, k)`** (exported): the change in a term's
  statistic when `ego` swaps its ranks of `j` and `k`, `compute(swapped) −
  compute(observed)`, ported from `ergm.rank`'s `wtchangestats_rank.c`
  change functions for all five terms (`RankDeference`, both
  `RankNonconformity` variants, `RankNodeICov`, `RankInconsistency`,
  `RankEdgeCov`). It evaluates only the comparisons the swap touches — the
  ordered alter pairs of ego's row containing `j` or `k` — O(n) per term and
  O(n²) for the nonconformity variants, allocation-free, and it is the term
  contract: every term implements `compute` and `swap_change`; the
  brute-force difference (`_swap_delta_bruteforce`) survives only as the
  test oracle, which every method is asserted `==` to on 200 random
  (network, ego, j, k) cases at each of n = 4, 5, 8 and on every swap of the
  R-validated 4-actor network. Bad swaps (`ego ∈ {j, k}`, `j == k`, out of
  range) and wrongly sized covariates are `ArgumentError`s.
- **Benchmark harness** (panel item 7 pattern): `benchmark/Project.toml`
  (sources ERGMRank, ERGM and Networks by path, so it instantiates from a
  fresh clone of the layout), `benchmark/benchmarks.jl` (BenchmarkTools suite
  printing `BENCHJL` lines: `swap_change` per term at 25 and 50 actors with
  an order-of-growth assertion — a recomputation regression shows as ≥ 8×
  — the derived Metropolis step at n = 50, `_rank_design` and the
  swap-MPLE fit at n = 17) and `benchmark/regression_tests.jl` (0-byte pins
  for every `swap_change`, `_swap_delta!` at p = 2/6/8 and the kernel step;
  the `_rank_design` allocation bound), picked up by the site repository's
  `tools/run_benchmarks.jl ERGMRank`.
- A thread-count-independence test: the `se=:bootstrap` standard errors
  equal those of `Networks.bootstrap_cov(...; threaded=false)` with the same
  `rng` bit for bit (all the randomness lives in the one simulation chain;
  CI runs a 4-thread cell). Five draws of the AlterSwap chain recorded from
  the pre-change loop are pinned as literals ("mh_toggle! adoption is
  bit-identical to the 0.2 loop"), alongside the testset that re-runs the
  old brute-force loop verbatim.

### Performance

- **The swap-MPLE derivative loop no longer allocates (review finding 15).**
  `_rank_mple_fit` carried its own logistic loop with a per-comparison
  `(pr*(1-pr)) .* (d * d')` inside it — a fresh `p×p` matrix on every one of the
  `n(n−1)(n−2)/2` (ego, alter-pair) comparisons of every Newton evaluation,
  **229 KB per evaluation** on a 17-actor ranking. The swap pseudo-likelihood
  *is* a logistic likelihood on the swap-difference rows with the response
  identically `true` (the observed order is always the "success"), so it now runs
  on the shared `ERGM.logistic_derivatives` — the same builder ERGMMulti and
  TERGM use: **192 bytes** per evaluation, independent of the number of
  comparisons, and **4.3x faster** (0.201 ms -> 0.046 ms). The swap design is
  held as one dense `(comparisons × p)` matrix (`_rank_design`) rather than a
  vector-of-vectors. Pinned by an `@allocated` regression test. The summation
  order moves from row-wise accumulation to BLAS, so the arithmetic is not
  bit-identical — but the fitted coefficients are: measured against the old
  loop on the same design, **max|Δθ| = 1.1e-16** (one ulp). Newton's last step
  is quadratically convergent, so a last-ulp difference in the gradient and
  Hessian does not move the fixed point.
- **The AlterSwap sampler runs on the shared `ERGM.mh_toggle!` kernel** (panel
  item 28) with the swap `(ego, j, k)` as its move: `change!` fills the
  kernel's `delta` workspace in place through the new `_swap_delta!`
  (written in place by the `@generated` `_swap_delta!` over the term tuple —
  see below; no dynamic dispatch, no boxed `Float64` per term) and returns
  `false` (a swap has no removal direction), `apply!` is
  `swap_ranks!`, `on_sample` pushes a copy. A Metropolis step allocates **0
  bytes** (was one fresh `Vector{Float64}` plus two statistic vectors per
  step), and the sampled sequence is **bit-identical** to the hand-written loop
  it replaced — the testset re-runs that loop verbatim and compares every
  draw. `_rank_design` fills its matrix through the same in-place kernel.
- **Per-step and per-row cost is O(n)–O(n²), not O(n³)–O(n⁴)** (WP2). The
  change statistics of a swap were two full `compute` evaluations per term
  (swap, recompute, restore): O(n³) for deference and the covariate terms,
  O(n⁴) for nonconformity, on every Metropolis step and every row of the
  swap-MPLE design. They are now the per-term `swap_change` (above), written
  into the kernel's workspace by a `_swap_delta!` `@generated` over the term
  tuple (the `ERGM._change_stat_tuple` pattern, so there is no `map`
  fallback cliff at 32 terms), **0 bytes** per step and per row (pinned at
  p = 2 and p = 8). Measured on the 17-actor Newcomb ranking under
  `rank.deference + rank.nonconformity`: the 2,040-row design in **15 ms
  (was 1.0 s)**, the whole swap-MPLE fit in **14 ms (was 1.9 s)**, the
  coefficients bit-for-bit identical (`[-0.14090974898889186,
  -0.005853837701451518]`, pinned as a regression literal of our own
  estimator — not an R value); at n = 50 a `swap_change` costs ≈ 1 μs for
  deference/nodeicov/inconsistency/edgecov and ≈ 60–170 μs for the two
  nonconformity variants. `_rank_design` allocates the matrix plus a fixed
  workspace (≤ `8·m·p + 4096` bytes, pinned) behind a function barrier on
  the term tuple; a derived Metropolis step at n = 50 under those two terms
  costs ≈ 56 μs and 0 bytes. **The non-fixture part of the test suite runs
  in ~35 s (was 1 min 54 s)** with ~1,100 more assertions (1,400+ vs 269);
  the golden-fixture testset adds ~2 min for two MCMLE fits at R's budget,
  one of them with the 16-rung bridge (see Changed) — the price of asserting
  against a real MCMLE, not to be cut by lowering the budget below R's.

### Changed

- **The newcomb golden fixture is now an assertion, not a `@test_broken`.**
  `test/fixtures/r/newcomb_rank.R` gained the `[tolerance]` keys
  `julia_seed_sd`, `coefficient_sd_multiple = 4` and `std_errors_rtol = 0.15`
  with their provenance (see Added), its prose says which estimator is
  asserted and which is characterised, and the TOML was regenerated with
  `Rscript` (the R values are seed-fixed and came back identical; only the
  `date` and the prose changed). The fixture testset now runs two MCMLE fits
  at R's budget on top of the swap-MPLE characterisation — one of them with
  the 16-rung bridge (`bridge_samples=512`, ~30 s more) to assert the
  log-likelihood against R's `logLik` (see Added) — about 2 min in all.
  The TOML was regenerated a second time with `Rscript` after the
  `mle_loglik`/`mle_loglik_mc_se`/`mle_df`/`mle_nobs`/`mle_aic`/`mle_bic`
  keys and the `julia_bridge_sd`/`loglik_sd_multiple` tolerances were added
  to the script; every previously frozen R value came back identical.
- **The README's Installation section no longer points at a root workspace
  project** (`julia --project=.` in the clone root), which exists only in
  the private development tree: it now says to clone the repositories side
  by side and `Pkg.develop` the three paths.
- **The one stray warning in the test log is asserted**: the continuation
  fit of the "MCMC MLE non-convergence is loud" testset is wrapped in
  `@test_logs`.
- `show(::RankERGMResult)` prints an `Estimation:` line naming the estimator
  (`swap pseudo-likelihood (swap-MPLE)` or `MCMC MLE (AlterSwap Metropolis, K
  bridge rungs)`); the log-likelihood, AIC/BIC and standard-error lines, the
  non-convergence caveat, the undefined-SE note and the closing caveat are
  per method. `Networks.is_exact`'s docstring gives the two reasons no rank
  fit is exact. The AlterSwap proposal is one function, `_propose_swap`,
  shared by `simulate_rank_ergm`, the MCMLE chains and the bridge (same draw
  order as before: the bit-identity pins still hold).
- Estimation is a real swap-based pseudo-likelihood maximized by the shared
  `Networks.newton_fit` on the `Networks.logistic_derivatives` kernel (was a
  placeholder gradient loop on a logistic approximation); both are imported
  from Networks.jl by name (`ERGM.newton_fit` is the same binding, pinned),
  and SEs/vcov come from the inverse negative Hessian.
- **Non-convergence is loud.** A fit that exhausts `maxiter` warns at fit time
  (quoting the gradient norm), lists the caveat in `approximations(fit)`, and
  `show` prints it directly under `Converged: false` — the three come from the
  same fields. Previously `converged == false` was a silent field.
- **A stalled Newton iterate at the optimum counts as converged.**
  `Networks.newton_fit` tests an absolute gradient norm (< √tol), which on a
  2,040-row swap design with Hessian entries in the thousands sits below the
  objective's rounding floor: Newton stalls *at* the maximum with ‖∇ℓ‖ ≈ 5e-4
  and a step of 1e-8, reports `converged == false`, and BLAS rounding decides
  which bootstrap replicates trip it (1 or 3 of 60 on the Newcomb fixture,
  depending on thread count). `_stalled_at_optimum` rescued an iterate whose
  Newton decrement ½∇ℓ'(−H)⁻¹∇ℓ (the objective gain a further step could
  deliver) is below `tol`; the criterion now lives in `Networks.newton_fit`
  and the shim is deleted (see Changed).
- `show(::RankERGMResult)` prints the pseudo-AIC/BIC line, the non-convergence
  / fixed-coefficient / undefined-SE / bootstrap-exclusion notes under the
  verdict, and the coefficient table through `coeftable(fit)`.
- NaN standard errors (a pseudo-Hessian that is not negative definite —
  `Networks.newton_fit` warns) are named in `approximations(fit)` and `show`
  ("standard errors are undefined") instead of printing bare `NaN`s.
- **CI and Documentation workflows derive their sibling clone lists from
  `[sources]`** (panel item 29) — a POSIX `sed` over `Project.toml`
  (`Networks`, `ERGM`) and over `docs/Project.toml` for the docs build (the
  `path = ".."` self-entry never matches), so the list cannot drift again
  (it used to hand-type SNA, which the package does not use; SNA is also
  gone from `docs/Project.toml`). CI runs the benchmark environment's
  `regression_tests.jl` as an allocation-regression gate on the Linux cell
  (instantiated from the committed `benchmark/Project.toml`, panel item 7),
  and runs that cell with `JULIA_NUM_THREADS=4` so the threaded bootstrap
  refits are exercised on more than one thread. The layout the workflow
  reconstructs (this package plus only the derived siblings, no Manifests)
  was rebuilt in a scratch directory and `Pkg.instantiate()`,
  `Pkg.test()`, the benchmark gates and the docs build all pass in it; the
  site's `tools/check_clean_depot.jl` installs and loads ERGMRank from the
  committed metadata.
- Combinatorics dropped from `[deps]`/`[compat]`; it was never used (see
  Breaking).
- `show(::RankERGMResult)` prints through the shared
  `Networks.print_coeftable` and labels the reference measure and
  pseudo-log-likelihood explicitly.
- **The "consistent approximation" claim for swap-MPLE has been withdrawn.**
  The docstring, README, and estimation guide previously called the
  estimator a "fast, consistent approximation" to `ergm.rank`'s MCMC MLE
  without naming an asymptotic regime or any assumptions under which that
  would hold. The docs now describe what the estimator does — maximize a
  pseudo-likelihood built from pairwise swap comparisons, which are not
  independent — state plainly that no consistency result is established
  here, and warn that the standard errors, being the inverse observed
  pseudo-Hessian, are expected to be anticonservative under dependence. No
  numerical behaviour changed.

### Known limitations

- **`rank.nonconformity`'s `local1`, `local2`, `geometric` and `thresholds`
  variants are not implemented.** `RankNonconformity` accepts `:all` and
  `:localAND` only; any other variant is refused at construction with
  `ArgumentError: RankNonconformity: variant must be :all or :localAND (got
  :local1); ergm.rank's local1/local2/geometric/thresholds variants are not
  implemented in ERGMRank.jl` — there is no silent fallback to `:all`. Each
  variant needs its own change statistic and an `ergm.rank` fixture. Stated
  in the README, the terms guide and the `RankNonconformity` docstring.
- The `Network`→`RankNetwork` adapter refuses a masked dyad and offers no
  `missing=:face` (a rank has no face value), so a partially observed
  ranking cannot be fitted at all: drop the unobserved actors first.
- `method=:mple` does not reproduce `ergm.rank`'s coefficients (see Added:
  the gap is characterised, `method=:mcmle` is the estimator that matches).

### Fixed

- CI collects coverage in the single-threaded Julia 1.12 Linux job and runs
  the full four-thread suite without coverage instrumentation, avoiding
  excessive overhead in parallel MCMC and bridge loops. BLAS uses one thread
  per calling task; coverage processing and upload follow the instrumented job.
- CI uses a fixed ranking for the boundary-likelihood regression and checks
  bootstrap thread independence with the documented exclusion of non-finite
  refits, including matching NaN rows across serial and threaded runs.
- **`gof(fit)` and `simulate_rank_ergm(fit)` refuse a fit with a non-finite
  coefficient** (one fixed at `±Inf` by a statistic at the boundary of its
  attainable range, or `NaN` — not identified, `converged == false`), as
  does `simulate_rank_ergm(rnet, terms, θ)` for a non-finite `θ`, with an
  `ArgumentError` naming the term(s), pointing at `fit.converged` /
  `approximations(fit)` and at removing the term (R's `drop=TRUE`). They
  used to run silently: θ'Δ is NaN (or ∓Inf) at every proposal, so the
  sampler rejected every swap and returned copies of the observed ranking,
  which `gof` printed as a perfect fit (every simulated statistic equal to
  the observed one, MC p-value 1.0000) — a number reported from an
  unidentified fit with nothing loud, and inconsistent with the
  `se=:bootstrap` refusal of the same fits.
- **The swap-MPLE documentation no longer claims the pseudo-likelihood
  "coincides with the likelihood" for dyad-independent-analogue models.**
  It never does — the comparisons within one ego's row must form a total
  order, so there is no rank analogue of dyad independence — and the claim
  was numerically false for the very term it fit best: on the 4-actor
  example network with `RankNodeICov([1, 2, 3, 4])`, exact enumeration of
  the 1296 orderings gives the MLE −0.0767 while the swap-MPLE is −0.0880;
  `method=:mcmle` run to a tight convergence threshold lands at −0.0772
  (at the default threshold and 1024 draws it satisfies its convergence
  test at the swap-MPLE start and takes no step). The `fit_ergm_rank` warning
  block, the `objective` docstring and the estimation guide now say so, and
  describe each factor as the two-state conditional `P(y | {y, y_swapped})`
  rather than "the relative order of j and k given the rest of the
  rankings" (a non-adjacent swap also reorders j and k relative to the
  intermediate alters). Pinned by a testset (|swap-MPLE − exact| > 0.01,
  |MCMLE − exact| < 0.25 SE and < a tenth of the swap-MPLE's gap).
- **`RankERGMModel` has `show` methods**: `string(fit.model)` is
  `RankERGMModel(1 term, 4 actors, CompleteOrder)` and the REPL block lists
  the reference, the network and the term names, following the
  one-line/block convention of `RankNetwork` and `RankERGMResult` (it used
  to print the raw struct with the full rank matrix).
- Error-message grammar: a wrongly sized covariate or reference matrix says
  "pass a 4×4 matrix" (was "an 4×4"); a θ/terms length mismatch in
  `simulate_rank_ergm` and `_swap_delta!` says "the model has 1 term" (was
  "1 terms").
- The docs said the deprecated `fit_rank_ergm` binding warns on "every
  access"; Julia warns once per deprecated binding per session, and the
  README, guide, API page, CLAUDE.md and docstring now say "the first
  access in a session".
- The estimation guide's "Scaled defaults" snippet computed the burn-in and
  interval from the documented formula instead of calling the private
  `ERGMRank._mcmc_defaults`; the `simulate_rank_ergm`/`gof`/`fit_ergm_rank`
  docstrings likewise give the formula rather than a private name.
- Two-sided p-values come from the ONE shared `Networks.z_pvalues` (panel
  item 13): floored at `floatmin(Float64)`, so |z| beyond ~38 prints `<1e-16`
  rather than `0.0`, and a NaN standard error gives a NaN p-value. The
  package's own unfloored `_z_pvalues` copy (which underflowed to exactly
  `0.0` for |z| beyond ~38, and to a subnormal that printed as 0 well before)
  is deleted; `@test !isdefined(ERGMRank, :_z_pvalues)`.

## [0.1.0] - 2026-02-09

Initial release: prototype rank-order network type, rank ERGM terms, and
placeholder estimation.
