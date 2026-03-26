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

1. **RankNetwork data structure** -- `RankNetwork{T}` wraps a `Network{T}` with a `Dict{Tuple{T,T}, Int}` for rank storage. Provides `set_rank!`, `get_rank`, `get_rankings_by`, `get_rankings_of`, `rank_matrix`, and `as_rank_network`.
2. **Reference measures** -- `PlackettLuce` and `ThurstoneMosteller` structs (currently stubs).
3. **Rank-specific ERGM terms** -- Eight term types, all subtypes of `AbstractERGMTerm` (from ERGM.jl): `RankEdges`, `RankMutual`, `RankTransitivity`, `RankNonconsensus`, `RankDeference`, `RankLocaltriangle`, `RankNodecov`, `RankAbsdiff`. Each implements `name()` and `compute()`.
4. **Model and estimation** -- `RankERGMModel`, `RankERGMResult`, and `ergm_rank()` (aliased as `fit_rank_ergm`). Currently supports `:mple` method via `rank_mple()`.
5. **Simulation** -- `simulate_rank_ergm()` generates random rank networks via MCMC.

## Key Dependencies

- **ERGM.jl** -- Provides `AbstractERGMTerm` base type
- **Network.jl** -- Underlying network data structure (`Network{T}`)
- **Graphs.jl** -- Graph interface (`nv`, `ne`, `has_edge`, `add_edge!`, `rem_edge!`)
- **Optim.jl**, **StatsBase.jl** -- Optimization and statistical utilities

Requires Julia 1.9+.

## Conventions

- All term structs subtype `AbstractERGMTerm` and implement `name()` (returning a dotted string like `"rank.mutual.3"`) and `compute()` (returning `Float64`).
- `RankNetwork` is parameterized by node ID type `T` (defaults to `Int`).
- Rankings use 1-based ordinal values where lower rank = higher preference.
- Functions mutating a `RankNetwork` use the `!` suffix convention (`set_rank!`).
- The module exports all public API symbols explicitly at the top of the module definition.
