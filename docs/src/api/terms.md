# Terms

Every term implements the ecosystem's statistic protocol — `compute(term,
rnet)` and `name(term)` are the one `Networks.compute`/`Networks.name`
generic each (reached through ERGM.jl and **exported by ERGMRank**, so
`using ERGMRank` suffices) — plus the rank-specific [`swap_change`](@ref).
`RankNonconformity` offers `:all` and `:localAND` only; `ergm.rank`'s
`local1`/`local2`/`geometric`/`thresholds` variants are refused with an
`ArgumentError` that says they are not implemented (see the
[terms guide](../guide/terms.md)).

```@docs
RankDeference
RankNonconformity
RankNodeICov
RankInconsistency
RankEdgeCov
```

## Swap change statistics

```@docs
swap_change
```
