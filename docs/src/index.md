# ERGMRank.jl

*Exponential Random Graph Models for Rank-Order Networks in Julia*

A Julia package for statistical modeling of networks where actors rank other actors, using rank-order ERGMs.

## Overview

Rank-order networks represent relational data where each actor assigns ordinal ranks to a subset of other actors. Instead of binary ties (present/absent) or valued ties (intensity), rank networks capture preference orderings: actor $i$ ranks actor $j$ first, actor $k$ second, and so on.

ERGMRank.jl is a port of the R [ergm.rank](https://cran.r-project.org/package=ergm.rank) package from the [StatNet](https://statnet.org/) collection, providing tools for modeling the structure of rank-order relational data.

### What is a Rank-Order Network?

In a rank network, each actor assigns ranks to some or all other actors:

```text
Actor A ranks: B=1st, C=2nd, D=3rd
Actor B ranks: A=1st, D=2nd
Actor C ranks: B=1st, A=2nd, D=3rd
```

Lower rank numbers indicate higher preference or importance.

Examples include:

- Students ranking their best friends
- Faculty ranking PhD programs
- Countries ranking trade partners by preference
- Employees ranking colleagues for collaboration
- Voters ranking political candidates (ranked-choice voting)

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Rank Network** | A network where edges carry ordinal rank values |
| **Ranker** | The actor who assigns ranks (analogous to sender) |
| **Rankee** | The actor who receives a rank (analogous to receiver) |
| **Max Rank** | The maximum rank value an actor can assign |
| **Mutual Ranking** | Both actors rank each other (potentially highly) |
| **Consensus** | Agreement across rankers about who should be ranked highly |

### Applications

Rank ERGMs are widely used in:

- **Social network analysis**: Modeling friendship nomination and ranking data
- **Education**: Analyzing student peer evaluations and preferences
- **Organizational studies**: Studying hierarchical preference structures
- **Political science**: Modeling ranked-choice voting patterns
- **Market research**: Analyzing consumer brand preference rankings

## Features

- **RankNetwork data structure**: Specialized type for rank-order network data with efficient query operations
- **Rank-specific terms**: Mutual ranking, transitivity, consensus, deference, local triangles, and attribute effects
- **Reference models**: Plackett-Luce and Thurstone-Mosteller ranking models
- **MPLE estimation**: Maximum Pseudo-Likelihood Estimation for rank ERGMs
- **Simulation**: Generate random rank networks from fitted models via MCMC
- **Conversion utilities**: Convert between rank matrices and RankNetwork objects

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Statistical-network-analysis-with-Julia/ERGMRank.jl")
```

Or for development:

```julia
using Pkg
Pkg.develop(path="/path/to/ERGMRank.jl")
```

## Quick Start

```julia
using ERGMRank

# Create a rank network with 10 actors, max rank 5
rnet = RankNetwork(10; max_rank=5)

# Actor 1 ranks actors 2, 3, 4
set_rank!(rnet, 1, 2, 1)   # 1 ranks 2 first
set_rank!(rnet, 1, 3, 2)   # 1 ranks 3 second
set_rank!(rnet, 1, 4, 3)   # 1 ranks 4 third

# Actor 2 ranks actors 1, 5
set_rank!(rnet, 2, 1, 1)   # 2 ranks 1 first
set_rank!(rnet, 2, 5, 2)   # 2 ranks 5 second

# Define rank-specific terms
terms = [
    RankEdges(),          # Overall ranking density
    RankMutual(1),        # Mutual top-1 rankings
    RankTransitivity(3),  # Transitive top-3 rankings
]

# Fit model
result = ergm_rank(rnet, terms)

# View results
println(result)
```

## Choosing Terms

| Use Case | Recommended Terms |
|----------|-------------------|
| Basic density | [`RankEdges`](@ref) |
| Best-friend reciprocity | [`RankMutual`](@ref)`(1)` |
| Top-k reciprocity | [`RankMutual`](@ref)`(k)` |
| Transitive preferences | [`RankTransitivity`](@ref)`(k)` |
| Ranking agreement | [`RankNonconsensus`](@ref) |
| Deference effects | [`RankDeference`](@ref)`(k)` |
| Local clustering | [`RankLocaltriangle`](@ref) |
| Attribute effects | [`RankNodecov`](@ref)`, ` [`RankAbsdiff`](@ref) |

## Documentation

```@contents
Pages = [
    "getting_started.md",
    "guide/rank_networks.md",
    "guide/terms.md",
    "guide/estimation.md",
    "api/types.md",
    "api/terms.md",
    "api/estimation.md",
]
Depth = 2
```

## Theoretical Background

### The Rank ERGM

Rank ERGMs model the probability of observing a particular ranking configuration:

$$P(R = r) \propto \exp\left(\theta^\top g(r)\right)$$

Where:

- $r$ is the observed ranking configuration (a matrix of ranks)
- $g(r)$ is a vector of sufficient statistics computed from the rank structure
- $\theta$ is the parameter vector to be estimated

Unlike valued ERGMs, rank ERGMs do not require a reference measure because the ranking constraint (each actor's ranks are a permutation) implicitly defines the support.

### Connection to Paired Comparison Models

Rank ERGMs can be viewed as extensions of paired comparison models (Bradley-Terry, Thurstone-Mosteller) that account for higher-order dependencies in the ranking structure. While paired comparison models treat each pairwise preference as independent, rank ERGMs model the joint distribution of all rankings simultaneously.

### Ordinal vs. Cardinal

A key distinction of rank data is that only the ordering matters, not the spacing. Rankings are ordinal: the difference between rank 1 and rank 2 is not necessarily the same as between rank 2 and rank 3. Rank ERGM terms respect this by using rank-based comparisons (e.g., "is $j$ in $i$'s top $k$?") rather than arithmetic operations on rank values.

## References

1. Krivitsky, P.N., Butts, C.T. (2017). Exponential-family random graph models for rank-order relational data. *Sociological Methodology*, 47(1), 68-112.

2. Krivitsky, P.N. (2012). Exponential-family random graph models for valued networks. *Electronic Journal of Statistics*, 6, 1100-1128.

3. Hunter, D.R., Handcock, M.S., Butts, C.T., Goodreau, S.M., Morris, M. (2008). ergm: A package to fit, simulate and diagnose exponential-family models for networks. *Journal of Statistical Software*, 24(3).

4. Luce, R.D. (1959). *Individual Choice Behavior: A Theoretical Analysis*. Wiley.
