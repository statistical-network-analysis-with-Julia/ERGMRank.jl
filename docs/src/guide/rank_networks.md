# Rank Networks

## The data structure

[`RankNetwork`](@ref) stores an ``n \times n`` matrix of rank values.
Row ``i`` is ego ``i``'s ranking of the alters:

- the diagonal is 0 (no self-ranks),
- off-diagonal entries in each row are a permutation of `1:(n-1)`,
- **greater values indicate higher standing** (`y[i,j] > y[i,k]` means
  ego `i` ranks `j` over `k`), matching R `ergm.rank`.

The permutation invariant is the defining constraint of complete
rank-order data. It is validated at construction and can be re-checked
with [`is_valid_ranking`](@ref) at any time.

```julia
using ERGMRank

rnet = RankNetwork(5)          # index-order rankings
m = rank_matrix(rnet)          # to a matrix (copy)
rnet = as_rank_network(m)      # from a rank matrix (validated)
```

## Modifying rankings

The elementary move of the rank sample space is the **AlterSwap**:
exchange the ranks an ego assigns to two alters. It always preserves the
invariant:

```julia
swap_ranks!(rnet, 1, 2, 3)   # ego 1 swaps the ranks of alters 2 and 3
```

`set_rank!` writes a single rank value and can transiently break the
invariant — use it only when rebuilding a full ranking, and validate
afterwards:

```julia
old = get_rank(rnet, 1, 2)
set_rank!(rnet, 1, 2, 4)         # may transiently duplicate a rank in row 1
if !is_valid_ranking(rnet)
    set_rank!(rnet, 1, 2, old)   # restore the permutation
end
is_valid_ranking(rnet)           # true again
```

`ergm_rank` and `simulate_rank_ergm` reject invalid rankings.

## The reference measure

[`CompleteOrderReference`](@ref) is the discrete-uniform distribution over
the possible complete orderings of the alters by each ego — the reference
measure of `ergm.rank`. Because it is constant over the sample space it
cancels from all likelihood ratios; what remains is the constraint itself,
which the AlterSwap proposal maintains by construction.

## From a `Network`

In R, `ergm.rank`'s data (`newcomb`) are `network` objects whose edge
attribute `"rank"` carries ego `i`'s rank of `j` on the arc `i → j`, read
with `as.matrix(nw, attrname = "rank")`. The same shape in Julia is a
directed `Networks.Network` with a rank edge attribute, and
[`as_rank_network`](@ref) converts it — honouring the ecosystem's
[conversion contract](https://statistical-network-analysis-with-Julia.github.io/Networks.jl/dev/guide/conversion_invariants/):
preserve what a `RankNetwork` can hold, reject what it cannot, report the
rest.

```julia
using ERGMRank, Networks

net = network(4)                       # directed; a ranking is ego-specific
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
for i in 1:4, j in 1:4
    i == j && continue
    add_edge!(net, i, j)
    set_edge_attribute!(net, :rank, i, j, m[i, j])
end
rnet = as_rank_network(net)            # attr=:rank is the default
rank_matrix(rnet) == m                 # true: the ranks are preserved exactly
```

**Preserved**: the actor set and the ranks. **Rejected** with an
`ArgumentError` that says why: an undirected network (ego `i`'s rank of `j`
and ego `j`'s rank of `i` must be two different arcs), a two-mode network
(every ego must rank every alter), a dyad with no arc or no rank value, a
non-integer rank, and an ego whose ranks are not a permutation of `1:(n-1)`
— the message names the ego and the attribute. **Dropped and reported**:
everything a `RankNetwork` has no place for — every other edge attribute,
every vertex attribute, every network attribute (and self-loop arcs). Ask
for the report with `report=true`:

```julia
set_vertex_attribute!(net, :sex, ["m", "f", "m", "f"])
set_network_attribute!(net, :title, "week 1")
rnet, rep = as_rank_network(net; report=true)
dropped_fields(rep)                    # [:sex, :title]
is_lossless(rep)                       # false — it would be true with only :rank present
rep                                    # ConversionReport: Network → RankNetwork, each drop with its reason
```

**A masked (unobserved) dyad is refused, and there is no `:face` policy.**
`as_rank_network` calls `require_observed(net, missing; context =
"as_rank_network")`, declares `supports_missing(as_rank_network) == true`
and `missing_policies(as_rank_network) == (:error,)`: a `RankNetwork` holds
*complete* orderings, and an unobserved rank has no face value to condition
on — unlike a binary tie, an absent rank cannot be read as "0" — so the
`missing=:face` opt-in the rest of the ecosystem offers would be a lie
here. Drop the actors whose rankings were not observed, or
`clear_missing_dyads!` once every dyad really has been observed.

```julia
using Test
set_missing_dyad!(net, 1, 2)           # ego 1's rank of 2 was not observed
@test_throws ArgumentError as_rank_network(net)
@test_throws ArgumentError as_rank_network(net; missing=:face)   # not offered, and says so
missing_policies(as_rank_network)      # (:error,)
```

The vertex attribute dropped above is not lost to the *model*: keep it
beside the `RankNetwork` and pass it to a covariate term, e.g.
`RankNodeICov(vertex_attribute_vector(net, :age, Float64))` — see the
[terms guide](terms.md).
