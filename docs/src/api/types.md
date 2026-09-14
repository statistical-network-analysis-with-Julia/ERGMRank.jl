# Types

## Module

```@docs
ERGMRank
```

## Core types

```@docs
RankNetwork
CompleteOrderReference
RankERGMModel
RankERGMResult
```

## Rank access and conversion

`as_rank_network` has two methods: from an integer rank matrix, and — the
`Network`→`RankNetwork` adapter of the ecosystem conversion contract — from
a directed `Networks.Network` with a rank edge attribute (`attr=:rank`,
`missing=:error` only, `report=true` for a `Networks.ConversionReport`).
`Networks.supports_missing(as_rank_network)` is `true` and
`Networks.missing_policies(as_rank_network) == (:error,)`.

```@docs
get_rank
set_rank!
swap_ranks!
is_valid_ranking
as_rank_network
rank_matrix
```

## Display

`RankNetwork`, [`RankERGMModel`](@ref) and [`RankERGMResult`](@ref) follow
Networks.jl's display convention: two-argument `show` (used by `print`,
`string` and inside containers such as the `Vector{RankNetwork}` of simulated
draws) is a one-line form — `RankNetwork(4 actors)`, `RankERGMModel(1 term, 4
actors, CompleteOrder)`, `RankERGMResult(swap-MPLE, 2 terms, converged)` — and
the `MIME"text/plain"` method the REPL calls prints the block (the rank matrix
for up to 12 actors; the reference, network and term names for a model; the
R-style coefficient table, convergence verdict and caveats for a fit).
`display(x)` prints the block from a script.
