# ERGMRank.jl


[![Network Analysis](https://img.shields.io/badge/Network-Analysis-orange.svg)](https://github.com/statistical-network-analysis-with-Julia/ERGMRank.jl)
[![Build Status](https://github.com/statistical-network-analysis-with-Julia/ERGMRank.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/statistical-network-analysis-with-Julia/ERGMRank.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://statistical-network-analysis-with-Julia.github.io/ERGMRank.jl/stable/)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://statistical-network-analysis-with-Julia.github.io/ERGMRank.jl/dev/)
[![Julia](https://img.shields.io/badge/Julia-1.9+-purple.svg)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="docs/src/assets/logo.svg" alt="ERGMRank.jl icon" width="160">
</p>

ERGMs for Rank-Order Networks in Julia.

## Overview

ERGMRank.jl extends ERGM to handle networks where each actor ranks others. In rank networks, actor i assigns ranks 1, 2, ..., k to a subset of alters, where lower rank indicates higher preference or importance.

This package is a Julia port of the R `ergm.rank` package from the StatNet collection.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/ERGMRank.jl")
```

## Features

- **RankNetwork type**: Specialized data structure for rank data
- **Rank-specific terms**: Mutual ranking, transitivity, consensus
- **Estimation**: MPLE for rank ERGMs
- **Simulation**: Generate random rank networks

## Quick Start

```julia
using ERGMRank

# Create rank network
rnet = RankNetwork(10; max_rank=5)

# Actor 1 ranks actors 2, 3, 4
set_rank!(rnet, 1, 2, 1)  # 1 ranks 2 first
set_rank!(rnet, 1, 3, 2)  # 1 ranks 3 second
set_rank!(rnet, 1, 4, 3)  # 1 ranks 4 third

# Define terms
terms = [
    RankEdges(),
    RankMutual(1),      # Mutual top-1 rankings
    RankTransitivity(3) # Transitive top-3 rankings
]

# Fit model
result = ergm_rank(rnet, terms)
```

## RankNetwork Data Structure

```julia
# Create rank network
rnet = RankNetwork(n; max_rank=n-1)

# Set rankings
set_rank!(rnet, ranker, rankee, rank)

# Query rankings
rank = get_rank(rnet, i, j)  # Rank i gives to j (or nothing)

# Get all rankings by an actor
rankings = get_rankings_by(rnet, i)  # Vector of (alter, rank) tuples

# Get all rankings of an actor
rankings = get_rankings_of(rnet, j)  # Vector of (ranker, rank) tuples

# Convert to matrix
mat = rank_matrix(rnet)  # Matrix with missing for non-rankings

# Create from matrix
rnet = as_rank_network(mat)
```

## Rank-Specific Terms

### Basic Terms
```julia
RankEdges()         # Number of rankings made (density)
```

### Reciprocity
```julia
RankMutual(k)       # Both rank each other in top k
RankDeference(k)    # Rank those who rank you highly
```

### Transitivity
```julia
RankTransitivity(k) # If i→j and j→k high, then i→k high
```

### Consensus
```julia
RankNonconsensus()  # Disagreement in rankings
```

### Local Structure
```julia
RankLocaltriangle() # Triangles within ego's rankings
```

### Attribute Effects
```julia
RankNodecov(:attr)  # Rank-weighted attribute effect
RankAbsdiff(:attr)  # Attribute difference effect on ranking
```

## Model Fitting

```julia
# Fit rank ERGM
result = ergm_rank(rnet, terms; method=:mple)

# View results
println(result)
```

## Simulation

```julia
# Simulate rank network
rnet = simulate_rank_ergm(n, terms, coef; max_rank=5)
```

## Example: Friendship Rankings

```julia
# Students rank their top 3 friends
rnet = RankNetwork(30; max_rank=3)
# ... populate with survey data ...

terms = [
    RankEdges(),           # Overall ranking activity
    RankMutual(1),         # Best friend reciprocity
    RankMutual(3),         # Top-3 reciprocity
    RankTransitivity(3),   # Transitive friendships
    RankNodecov(:grade),   # Grade effect on being ranked
]

result = ergm_rank(rnet, terms)

# Positive RankMutual → tendency to reciprocate rankings
# Positive RankTransitivity → friends of friends ranked highly
```

## Example: Academic Rankings

```julia
# Professors rank PhD programs
# Negative RankNonconsensus → consensus in rankings
# Positive RankNodecov(:publications) → productive programs ranked higher
```

## Mathematical Background

Rank ERGMs model the probability of observing a particular ranking configuration:

```
P(R = r) ∝ exp(θ'g(r))
```

Where g(r) are sufficient statistics computed from the rank structure.

## Documentation

For more detailed documentation, see:

- [Stable Documentation](https://statistical-network-analysis-with-Julia.github.io/ERGMRank.jl/stable/)
- [Development Documentation](https://statistical-network-analysis-with-Julia.github.io/ERGMRank.jl/dev/)

## References

1. Krivitsky, P.N., Butts, C.T. (2017). Exponential-family random graph models for rank-order relational data. *Sociological Methodology*, 47(1), 68-112.

2. Hunter, D.R., Handcock, M.S., Butts, C.T., Goodreau, S.M., Morris, M. (2008). ergm: A package to fit, simulate and diagnose exponential-family models for networks. *Journal of Statistical Software*, 24(3), 1-29.

## License

MIT License - see [LICENSE](LICENSE) for details.
