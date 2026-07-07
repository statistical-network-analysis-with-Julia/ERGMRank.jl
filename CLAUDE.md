# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ERGMRank.jl is a Julia port of the R `ergm.rank` package (from the StatNet collection) for fitting Exponential Random Graph Models to rank-order networks, where each actor assigns ordinal ranks to a subset of other actors.

## Development Commands

- **Run tests:** `julia --project=. -e 'using Pkg; Pkg.test()'`
- **Build docs:** `julia --project=docs docs/make.jl`
- **Load package in REPL:** `julia --project=.` then `using ERGMRank`

## Architecture

The entire package lives in a single source file: `src/ERGMRank.jl`. It is organized into these sections:

1. **RankNetwork** -- dense `Matrix{Int}` of rank values; row i = ego i's ranking of the alters, GREATER value = HIGHER standing (ergm.rank convention). The complete-ranking invariant (each row's off-diagonal is a permutation of `1:(n-1)`) is validated at construction, preserved by `swap_ranks!` (the AlterSwap move), and checkable with `is_valid_ranking`.
2. **Reference measure** -- `CompleteOrderReference`: discrete-uniform over complete orderings (constant; cancels from likelihood ratios; the constraint is what matters).
3. **Terms** -- `RankDeference`, `RankNonconformity(:all/:localAND)`, `RankNodeICov`, `RankInconsistency`, `RankEdgeCov`, implemented from ergm.rank's `wtchangestats_rank.c` summary functions and golden-master tested against R ergm.rank 4.1.2 (values in test/runtests.jl).
4. **Estimation** -- `ergm_rank` (alias `fit_rank_ergm`): swap-based maximum pseudo-likelihood — for each ego and unordered alter pair, the observed relative order is logistic in θ'[g(y) − g(y_swapped)]; Newton-Raphson with step-halving.
5. **Simulation** -- `simulate_rank_ergm`: Metropolis with the symmetric AlterSwap proposal; every state is a valid complete ranking.

## Key Dependencies

- **ERGM.jl** -- Provides the `AbstractERGMTerm` base type and the `name`/`compute` generics (extended via `import ERGM: name, compute`)

Requires Julia 1.12+.

## Conventions

- All term structs subtype `AbstractERGMTerm` and implement `name()` (returning a dotted string like `"rank.mutual.3"`) and `compute()` (returning `Float64`).
- `RankNetwork` stores a dense `Matrix{Int}`; it does not wrap a `Network`.
- Rankings use values `1:(n-1)` per ego where GREATER value = HIGHER standing (matches R ergm.rank; note this is the opposite of "rank 1 = best").
- Functions mutating a `RankNetwork` use the `!` suffix convention (`set_rank!`).
- The module exports all public API symbols explicitly at the top of the module definition.
