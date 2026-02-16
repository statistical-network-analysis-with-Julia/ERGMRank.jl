# Getting Started

This tutorial walks through common use cases for ERGMRank.jl, from creating rank networks to fitting and interpreting rank-order ERGM models.

## Installation

Install ERGMRank.jl from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Statistical-network-analysis-with-Julia/ERGMRank.jl")
```

ERGMRank.jl depends on Network.jl and ERGM.jl, which will be installed automatically.

## Basic Workflow

The typical ERGMRank.jl workflow consists of four steps:

1. **Create or load a rank network** -- Construct a RankNetwork with ranking data
2. **Define rank terms** -- Specify which rank-specific statistics to include
3. **Fit the model** -- Estimate coefficients via MPLE
4. **Interpret results** -- Analyze the fitted model

## Step 1: Create a Rank Network

A `RankNetwork` stores ranking data where each actor assigns ordinal ranks to other actors:

```julia
using ERGMRank

# Create a rank network: 10 actors, each can rank up to 5 others
rnet = RankNetwork(10; max_rank=5)

# Actor 1 ranks three others
set_rank!(rnet, 1, 2, 1)   # Actor 1 ranks Actor 2 first
set_rank!(rnet, 1, 5, 2)   # Actor 1 ranks Actor 5 second
set_rank!(rnet, 1, 3, 3)   # Actor 1 ranks Actor 3 third

# Actor 2 ranks two others
set_rank!(rnet, 2, 1, 1)   # Actor 2 ranks Actor 1 first (mutual!)
set_rank!(rnet, 2, 4, 2)   # Actor 2 ranks Actor 4 second

# Actor 3 ranks two others
set_rank!(rnet, 3, 1, 1)   # Actor 3 ranks Actor 1 first
set_rank!(rnet, 3, 2, 2)   # Actor 3 ranks Actor 2 second
```

### Querying Rankings

```julia
# What rank does Actor 1 give to Actor 2?
rank = get_rank(rnet, 1, 2)    # 1

# What rank does Actor 1 give to Actor 4?
rank = get_rank(rnet, 1, 4)    # nothing (not ranked)

# All rankings made by Actor 1 (sorted by rank)
rankings = get_rankings_by(rnet, 1)
# [(2, 1), (5, 2), (3, 3)]

# All rankings received by Actor 1
rankings = get_rankings_of(rnet, 1)
# [(2, 1), (3, 1)]  -- Actors 2 and 3 both rank Actor 1 first
```

### Creating from a Matrix

For existing data in matrix form:

```julia
# Rank matrix: entry (i,j) = rank that i gives to j
# Use `missing` for unranked alters
mat = Union{Int, Missing}[
    missing  1        2        missing
    1        missing  missing  2
    2        1        missing  missing
    missing  missing  1        missing
]

rnet = as_rank_network(mat; max_rank=3)
```

### Converting to a Matrix

```julia
mat = rank_matrix(rnet)
# Matrix{Union{Int, Missing}} with ranks and missing values
```

### Inspecting the Network

```julia
println("Nodes: ", nv(rnet))       # 10
println("Rankings: ", ne(rnet))     # Number of rankings made
println("Directed: ", is_directed(rnet))  # true (always)
```

## Step 2: Define Rank Terms

Rank terms capture structural patterns in the ranking data:

```julia
# Basic model terms
terms = [
    RankEdges(),          # Overall ranking density
    RankMutual(1),        # Mutual top-1 rankings (best friend reciprocity)
    RankTransitivity(3),  # Transitive top-3 rankings
]
```

### Exploring Available Terms

ERGMRank.jl provides terms organized by type:

| Category | Terms | Description |
|----------|-------|-------------|
| **Basic** | `RankEdges` | Ranking density |
| **Reciprocity** | `RankMutual`, `RankDeference` | Mutual and deferential rankings |
| **Transitivity** | `RankTransitivity` | Transitive ranking patterns |
| **Consensus** | `RankNonconsensus` | Disagreement in rankings |
| **Local** | `RankLocaltriangle` | Triangles within ego's rankings |
| **Attribute** | `RankNodecov`, `RankAbsdiff` | Node covariate effects on rankings |

### Example: Comprehensive Model

```julia
terms = [
    # Density
    RankEdges(),

    # Reciprocity at different levels
    RankMutual(1),          # Best-friend reciprocity
    RankMutual(3),          # Top-3 reciprocity

    # Structural effects
    RankTransitivity(3),    # Transitive preferences
    RankNonconsensus(),     # Disagreement

    # Local clustering
    RankLocaltriangle(),    # Triangles in ego rankings
]
```

## Step 3: Fit the Model

Use `ergm_rank` to estimate model parameters:

```julia
result = ergm_rank(rnet, terms)
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `method` | Estimation method | `:mple` |
| `maxiter` | Maximum iterations | `100` |

### Viewing Results

```julia
# Print formatted summary
println(result)

# Output:
# Rank ERGM Results
# =================
# Log-likelihood: -34.5678
# Converged: true
#
# Coefficients:
#   rank.edges                 -0.5432 (SE: 0.1234)
#   rank.mutual.1               1.2345 (SE: 0.3456)
#   rank.transitivity.3         0.6789 (SE: 0.2345)
```

### Accessing Results Programmatically

```julia
# Coefficient vector
result.coefficients

# Standard errors
result.std_errors

# Check convergence
result.converged
```

## Step 4: Interpret Results

Coefficients in rank ERGMs describe the log-odds of observing ranking configurations with higher values of the corresponding statistic.

| Coefficient | Interpretation |
|-------------|----------------|
| `RankEdges` < 0 | Sparse rankings (actors rank fewer others) |
| `RankMutual(1)` > 0 | Best-friend reciprocity (mutual top choices) |
| `RankTransitivity(3)` > 0 | Transitive preferences (friends of friends ranked highly) |
| `RankNonconsensus` < 0 | Consensus (actors agree on who to rank highly) |
| `RankLocaltriangle` > 0 | Local clustering (ranked alters also rank each other) |

**Example interpretations:**

- `RankMutual(1) = 1.5`: The odds of mutually ranking each other first are $\exp(1.5) \approx 4.5$ times higher than expected
- `RankNonconsensus = -0.3`: A mild tendency toward consensus; actors tend to agree about rankings
- `RankTransitivity(3) = 0.8`: If $i$ ranks $j$ in top 3 and $j$ ranks $k$ in top 3, then $i$ is more likely to rank $k$ in top 3

## Complete Example

```julia
using ERGMRank
using Random

Random.seed!(42)

# Create a friendship ranking network
# 15 students each rank their top 3 friends
n = 15
rnet = RankNetwork(n; max_rank=3)

# Simulate some structured rankings
for i in 1:n
    others = setdiff(1:n, i)
    # Each student ranks 3 random others
    ranked = others[randperm(length(others))[1:3]]
    for (r, j) in enumerate(ranked)
        set_rank!(rnet, i, j, r)
    end
end

println("Network: $(nv(rnet)) students, $(ne(rnet)) rankings")

# Define model terms
terms = [
    RankEdges(),           # Ranking density
    RankMutual(1),         # Best friend reciprocity
    RankMutual(3),         # Top-3 reciprocity
    RankTransitivity(3),   # Transitive friendships
    RankNonconsensus(),    # Consensus in rankings
]

# Fit model
result = ergm_rank(rnet, terms)

# Display results
println(result)

# Check convergence
if result.converged
    println("\nModel converged successfully")
else
    println("\nWarning: Model did not converge")
end
```

## Simulating Rank Networks

Generate rank networks from specified parameters:

```julia
# Define terms and coefficients
terms = [RankEdges(), RankMutual(1), RankTransitivity(3)]
coef = [-0.5, 1.5, 0.8]

# Simulate a rank network
rnet_sim = simulate_rank_ergm(15, terms, coef;
    max_rank=5,
    burnin=2000
)

# Inspect the simulated network
println("Simulated rankings: ", ne(rnet_sim))
```

## Comparing Models

```julia
# Model 1: Basic density only
terms1 = [RankEdges()]

# Model 2: Add reciprocity
terms2 = [RankEdges(), RankMutual(1)]

# Model 3: Full model
terms3 = [RankEdges(), RankMutual(1), RankTransitivity(3), RankNonconsensus()]

result1 = ergm_rank(rnet, terms1)
result2 = ergm_rank(rnet, terms2)
result3 = ergm_rank(rnet, terms3)

for (i, r) in enumerate([result1, result2, result3])
    println("Model $i: $(length(r.coefficients)) terms, converged=$(r.converged)")
end
```

## Working with Node Attributes

Include actor-level attributes in the model:

```julia
# Set node attributes on the underlying network
set_vertex_attribute!(rnet.network, 1, :grade, 10)
set_vertex_attribute!(rnet.network, 2, :grade, 11)
# ... set for all actors ...

# Add attribute terms
terms = [
    RankEdges(),
    RankMutual(1),
    RankNodecov(:grade),     # Grade effect on being ranked
    RankAbsdiff(:grade),     # Grade difference effect
]

result = ergm_rank(rnet, terms)
```

## Best Practices

1. **Start with RankEdges**: Always include the density term as a baseline
2. **Check convergence**: Always verify `result.converged == true`
3. **Choose cutoffs carefully**: The cutoff parameter in `RankMutual` and `RankTransitivity` should reflect meaningful thresholds
4. **Start simple**: Begin with RankEdges and one structural term, then add complexity
5. **Consider sample size**: More actors provide more stable estimates
6. **Match max_rank to data**: Set `max_rank` to match the actual maximum number of rankings per actor
7. **Interpret ordinal**: Remember that ranks are ordinal, not cardinal

## Next Steps

- Learn about [Rank Networks](guide/rank_networks.md) in detail
- Explore all [Rank Terms](guide/terms.md) available
- Understand the [Estimation](guide/estimation.md) procedure
