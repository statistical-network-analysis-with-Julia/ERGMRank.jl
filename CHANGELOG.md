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

- ergm.rank-faithful terms ported from `wtchangestats_rank.c`:
  `RankNonconformity(:all | :localAND)`, `RankNodeICov`,
  `RankInconsistency`, `RankEdgeCov`; `CompleteOrderReference` reference
  measure (discrete-uniform over complete orderings).
- `fit_ergm_rank` as the canonical entry point; `ergm_rank` (R-faithful) and
  `fit_rank_ergm` (legacy) kept as `const` aliases.
- Rank-manipulation API: `get_rank`, `set_rank!`, `swap_ranks!` (the
  AlterSwap elementary move), `is_valid_ranking`.
- `gof(::RankERGMResult)` extending the ecosystem-wide `Network.gof`,
  simulating via AlterSwap and returning a `Network.GOFResult`.
- StatsAPI accessors: `coef`, `stderror`, `vcov`, `loglikelihood`, `nobs`,
  `dof` (nobs = ego × unordered-alter-pair comparisons); `RankERGMResult`
  gains a `vcov` field.

### Changed

- Estimation is a real swap-based pseudo-likelihood maximized by the shared
  `ERGM.newton_fit` (was a placeholder gradient loop on a logistic
  approximation); SEs/vcov come from the inverse negative Hessian.
- `show(::RankERGMResult)` prints through the shared
  `Network.print_coeftable` and labels the reference measure and
  pseudo-log-likelihood explicitly.

### Fixed

- Two-sided p-values computed via `ccdf(Normal(), |z|)` no longer underflow
  to exactly `0.0` for |z| beyond ~8.3.

## [0.1.0] - 2026-02-09

Initial release: prototype rank-order network type, rank ERGM terms, and
placeholder estimation.
