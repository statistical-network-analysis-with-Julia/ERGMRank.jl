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
rnet = RankNetwork(5)          # index-order rankings
rnet = as_rank_network(m)      # from a rank matrix (validated)
rank_matrix(rnet)              # back to a matrix (copy)
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
set_rank!(rnet, 1, 2, 4)
is_valid_ranking(rnet) || error("fix the ranking before modeling")
```

`ergm_rank` and `simulate_rank_ergm` reject invalid rankings.

## The reference measure

[`CompleteOrderReference`](@ref) is the discrete-uniform distribution over
the possible complete orderings of the alters by each ego — the reference
measure of `ergm.rank`. Because it is constant over the sample space it
cancels from all likelihood ratios; what remains is the constraint itself,
which the AlterSwap proposal maintains by construction.
