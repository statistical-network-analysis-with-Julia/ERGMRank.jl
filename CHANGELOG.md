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
- **`ergm_rank` dropped the `method=:mple` keyword** (only
  `maxiter`/`tol`). *Migration:* remove `method=`.
- **Minimum Julia raised to 1.12**; package UUID regenerated. *Migration:*
  upgrade Julia and re-resolve environments pinning the old UUID.

### Added

- **Provenanced golden fixture against a real `ergm.rank` fit** (issue #8).
  `test/fixtures/newcomb_rank.toml` freezes an ergm.rank 4.1.2 MCMLE of
  `newcomb[[1]] ~ rank.deference + rank.nonconformity("all")` under the
  CompleteOrder reference, regenerable with `Rscript
  test/fixtures/r/newcomb_rank.R > test/fixtures/newcomb_rank.toml` (**slow**:
  ~17 min, six MCMLE fits).

  **The two packages do not fit the same estimator, and the fixture says so.**
  `ergm.rank` fits the MCMC MLE; `ergm_rank` fits a swap pseudo-likelihood, whose
  overlapping (ego, alter-pair) comparisons are multiplied as if independent. The
  coefficient comparison is therefore **`@test_broken`**, not tolerated at a
  convenient atol, and what the testset *asserts* is the character of the gap:

  - The **observed sufficient statistics match exactly** (844, 12748) — asserted
    at 1e-9. The term formulas are right.
  - The gap is **systematic, not Monte-Carlo**: swap-MPLE `[−0.14091, −0.0058538]`
    against ergm.rank's `[−0.15310, −0.0065927]`, which is **16x and 13x**
    ergm.rank's own seed-to-seed sd (7.4e-4 / 5.6e-5). No MCMC budget closes it.
  - But it is **small where it counts**: 0.30 and 0.43 of an ergm.rank standard
    error — same sign, same order, same substantive story. A different estimator,
    not a broken one.
  - The inverse-pseudo-Hessian standard errors are **anticonservative by a
    measured factor**: 3.9x (deference) and 2.0x (nonconformity) narrower than
    the MLE's. `se=:bootstrap` recovers most of it (0.0280 against R's 0.0404),
    and the testset asserts that it does.

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
- StatsAPI accessors: `coef`, `stderror`, `vcov`, `loglikelihood`, `nobs`,
  `dof` (nobs = ego × unordered-alter-pair comparisons); `RankERGMResult`
  gains a `vcov` field.

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

### Changed

- Estimation is a real swap-based pseudo-likelihood maximized by the shared
  `ERGM.newton_fit` (was a placeholder gradient loop on a logistic
  approximation); SEs/vcov come from the inverse negative Hessian.
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

### Fixed

- Two-sided p-values computed via `ccdf(Normal(), |z|)` no longer underflow
  to exactly `0.0` for |z| beyond ~8.3.

## [0.1.0] - 2026-02-09

Initial release: prototype rank-order network type, rank ERGM terms, and
placeholder estimation.
