# Rank Networks

This guide covers how to work with rank-order network data in ERGMRank.jl, including creating, querying, and manipulating RankNetwork objects.

## RankNetwork Structure

A `RankNetwork` stores directed ranking relationships between actors. Internally, it wraps a `Network{T}` for graph operations and maintains a separate `Dict` mapping `(ranker, rankee)` pairs to integer rank values.

```julia
struct RankNetwork{T}
    network::Network{T}     # Underlying graph structure
    ranks::Dict{Tuple{T,T}, Int}  # (i,j) -> rank i gives to j
    max_rank::Int            # Maximum rank value
end
```

### Properties

- Rankings are always directed: actor $i$ ranks actor $j$ (asymmetric by nature)
- Self-rankings are not allowed ($i$ cannot rank itself)
- Rank values are positive integers from 1 to `max_rank`
- An actor may rank zero, some, or all other actors
- Each actor's rankings are independent (no constraint that all actors rank the same number of others)

## Creating Rank Networks

### From Scratch

```julia
using ERGMRank

# Create a rank network with 10 actors
# Each actor can assign ranks 1 through max_rank
rnet = RankNetwork(10; max_rank=5)

# Set individual rankings
set_rank!(rnet, 1, 2, 1)   # Actor 1 ranks Actor 2 as 1st
set_rank!(rnet, 1, 3, 2)   # Actor 1 ranks Actor 3 as 2nd
set_rank!(rnet, 1, 5, 3)   # Actor 1 ranks Actor 5 as 3rd

# Actor 2's rankings
set_rank!(rnet, 2, 1, 1)   # Actor 2 ranks Actor 1 as 1st
set_rank!(rnet, 2, 4, 2)   # Actor 2 ranks Actor 4 as 2nd
```

### With Type Parameter

```julia
# Integer-indexed actors (default)
rnet = RankNetwork{Int}(10; max_rank=5)

# Equivalent shorthand
rnet = RankNetwork(10; max_rank=5)
```

### Default Max Rank

If `max_rank` is not specified, it defaults to $n - 1$ (every actor can rank all others):

```julia
# max_rank = 9 by default
rnet = RankNetwork(10)
```

### From a Matrix

Convert an existing rank matrix to a RankNetwork:

```julia
# Matrix where entry (i,j) = rank that i gives to j
# Use `missing` for unranked alters
mat = Union{Int, Missing}[
    missing  1        3        2        missing
    2        missing  1        missing  3
    1        missing  missing  3        2
    missing  2        missing  missing  1
    3        1        2        missing  missing
]

rnet = as_rank_network(mat)

# With explicit max_rank
rnet = as_rank_network(mat; max_rank=5)
```

### From Survey Data

Rank data commonly comes from surveys where respondents name or rank others:

```julia
# Suppose you have survey responses as a list of tuples
# (ranker, rankee, rank)
responses = [
    (1, 3, 1), (1, 5, 2), (1, 2, 3),  # Student 1's rankings
    (2, 1, 1), (2, 4, 2),              # Student 2's rankings
    (3, 5, 1), (3, 1, 2), (3, 2, 3),  # Student 3's rankings
    (4, 2, 1), (4, 3, 2),              # Student 4's rankings
    (5, 3, 1), (5, 1, 2), (5, 4, 3),  # Student 5's rankings
]

rnet = RankNetwork(5; max_rank=3)
for (i, j, r) in responses
    set_rank!(rnet, i, j, r)
end
```

## Querying Rankings

### Individual Rankings

```julia
# Get the rank that actor i gives to actor j
rank = get_rank(rnet, 1, 3)    # Returns Int or nothing

if !isnothing(rank)
    println("Actor 1 ranks Actor 3 as #$rank")
else
    println("Actor 1 does not rank Actor 3")
end
```

### Rankings by an Actor

Get all rankings made by a specific actor, sorted by rank:

```julia
rankings = get_rankings_by(rnet, 1)
# Returns Vector{Tuple{T, Int}}: [(alter, rank), ...]
# Sorted by rank (lowest first)

for (alter, rank) in rankings
    println("Actor 1 ranks Actor $alter as #$rank")
end
# Actor 1 ranks Actor 3 as #1
# Actor 1 ranks Actor 5 as #2
# Actor 1 ranks Actor 2 as #3
```

### Rankings of an Actor

Get all rankings received by a specific actor:

```julia
rankings = get_rankings_of(rnet, 3)
# Returns Vector{Tuple{T, Int}}: [(ranker, rank), ...]

for (ranker, rank) in rankings
    println("Actor $ranker ranks Actor 3 as #$rank")
end
```

This is useful for understanding an actor's popularity or reputation.

### Graph Interface

RankNetwork forwards standard graph operations to its underlying Network:

```julia
nv(rnet)           # Number of vertices (actors)
ne(rnet)           # Number of ranking edges
vertices(rnet)     # Iterate over vertices
edges(rnet)        # Iterate over edges
is_directed(rnet)  # Always true
```

## Matrix Conversion

### To Matrix

Convert a RankNetwork to a matrix of ranks:

```julia
mat = rank_matrix(rnet)
# Matrix{Union{Int, Missing}} of size n x n
# mat[i,j] = rank that i gives to j, or missing if not ranked
```

### From Matrix

Convert a matrix back to a RankNetwork:

```julia
rnet = as_rank_network(mat)
```

The function:
- Infers `max_rank` from the maximum non-missing value (or use the `max_rank` keyword)
- Skips `missing` and `nothing` entries
- Validates that ranks are positive integers

## Data Validation

### Self-Ranking

Attempting to rank oneself throws an error:

```julia
set_rank!(rnet, 1, 1, 1)  # ArgumentError: Cannot rank self
```

### Invalid Rank Values

Ranks must be in the range $[1, \text{max\_rank}]$:

```julia
set_rank!(rnet, 1, 2, 0)   # ArgumentError: Rank must be in 1:5
set_rank!(rnet, 1, 2, 10)  # ArgumentError: Rank must be in 1:5
```

### Duplicate Rankings

Setting a rank for a dyad that already has a rank overwrites the previous value:

```julia
set_rank!(rnet, 1, 2, 1)   # Actor 1 ranks Actor 2 as #1
set_rank!(rnet, 1, 2, 3)   # Now Actor 1 ranks Actor 2 as #3 (overwritten)
```

## Network Statistics

### Basic Counts

```julia
n = nv(rnet)                    # Number of actors
n_rankings = ne(rnet)           # Total number of rankings
avg_rankings = ne(rnet) / nv(rnet)  # Average rankings per actor
```

### Popularity Analysis

```julia
# Who is most frequently ranked?
popularity = Dict{Int, Int}()
for v in vertices(rnet)
    popularity[v] = length(get_rankings_of(rnet, v))
end

# Sort by popularity
sorted = sort(collect(popularity), by=x -> x[2], rev=true)
for (actor, count) in sorted
    println("Actor $actor: ranked by $count others")
end
```

### Top Choice Analysis

```julia
# Who is most frequently ranked #1?
top_choices = Dict{Int, Int}()
for v in vertices(rnet)
    for (ranker, rank) in get_rankings_of(rnet, v)
        if rank == 1
            top_choices[v] = get(top_choices, v, 0) + 1
        end
    end
end
```

### Reciprocity Analysis

```julia
# Check mutual rankings
mutual_count = 0
for (i, j) in keys(rnet.ranks)
    if !isnothing(get_rank(rnet, j, i))
        mutual_count += 1
    end
end
# Each mutual pair is counted twice
println("Mutual ranking pairs: ", mutual_count / 2)
```

## Working with Subsets

### Actor Rankings Within Top k

```julia
# Get actors that i ranks in top k
function top_k_alters(rnet, i, k)
    rankings = get_rankings_by(rnet, i)
    return [alter for (alter, rank) in rankings if rank <= k]
end

# Actor 1's top 3 choices
top3 = top_k_alters(rnet, 1, 3)
```

### Filtering by Rank Threshold

```julia
# Get all dyads ranked in top k
function top_k_dyads(rnet, k)
    dyads = Tuple{Int, Int}[]
    for ((i, j), rank) in rnet.ranks
        if rank <= k
            push!(dyads, (i, j))
        end
    end
    return dyads
end
```

## Common Patterns in Rank Data

### Complete vs. Partial Rankings

- **Complete rankings**: Every actor ranks all $n-1$ others (set `max_rank = n-1`)
- **Partial rankings**: Each actor ranks only their top $k$ choices (set `max_rank = k`)

Most real-world rank data is partial, where actors nominate and rank only a subset of others.

### Tied Rankings

ERGMRank.jl does not currently support tied rankings. Each rank value must be a distinct positive integer. If your data has ties, consider:

- Breaking ties randomly
- Using the midrank (average of tied positions)
- Treating the data as count-valued and using ERGMCount.jl instead

### Missing vs. Not Ranked

A `nothing` return from `get_rank` means the actor did not rank that alter. This is different from a low ranking -- it means the alter was not included in the actor's ranking at all. In many applications, "not ranked" is informative (the alter is outside the actor's consideration set).

## Comparison with Other Network Types

| Property | Binary Network | Count Network | Rank Network |
|----------|---------------|---------------|--------------|
| Edge values | 0/1 | Non-negative integers | Ordinal ranks |
| Meaning | Presence/absence | Intensity | Preference ordering |
| Symmetry | Can be undirected | Can be undirected | Always directed |
| Per-actor constraint | None | None | Ranks form a partial permutation |
| Missing edges | No tie | Zero interaction | Not ranked (outside consideration) |
