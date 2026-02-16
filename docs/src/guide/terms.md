# Rank Terms

Rank terms are sufficient statistics computed from the ranking structure. They capture different mechanisms that drive how actors rank others, from simple reciprocity to complex structural patterns.

## Terms Interface

All rank terms are subtypes of `AbstractERGMTerm` and implement:

```julia
compute(term, rnet::RankNetwork) -> Float64  # Full network statistic
name(term) -> String                          # Human-readable name
```

Some terms also implement the change statistic for MCMC simulation:

```julia
change_stat(term, rnet, i, j) -> Float64  # Change when toggling ranking
```

## Density Term

### RankEdges

The most basic term -- the total number of rankings made across all actors.

```julia
RankEdges()
```

**Statistic**:

$$g(r) = |\{(i,j) : r_{ij} \text{ exists}\}|$$

**Interpretation**:
- Analogous to the edges term in binary ERGMs
- Controls the overall density of rankings
- Negative coefficient: actors rank fewer others on average
- Positive coefficient: actors rank more others on average

**Example**:

```julia
using ERGMRank

rnet = RankNetwork(5; max_rank=3)
set_rank!(rnet, 1, 2, 1)
set_rank!(rnet, 1, 3, 2)
set_rank!(rnet, 2, 1, 1)

term = RankEdges()
println(compute(term, rnet))  # 3.0 (three rankings made)
println(name(term))           # "rank.edges"
```

## Reciprocity Terms

### RankMutual

Counts pairs of actors who mutually rank each other in their top $k$ positions.

```julia
RankMutual(k::Int=1)
```

**Statistic**:

$$g(r) = |\{(i,j) : i < j, \, r_{ij} \leq k, \, r_{ji} \leq k\}|$$

**Parameters**:
- `k`: Cutoff for "top" positions (default: 1, meaning only first-choice mutual)

**Interpretation**:
- `RankMutual(1)` > 0: Tendency for best-friend reciprocity
- `RankMutual(3)` > 0: Tendency for top-3 reciprocity
- Higher cutoffs capture weaker forms of reciprocity

**Example**:

```julia
rnet = RankNetwork(5; max_rank=3)
set_rank!(rnet, 1, 2, 1)   # 1's best friend is 2
set_rank!(rnet, 2, 1, 1)   # 2's best friend is 1 (mutual!)
set_rank!(rnet, 1, 3, 2)   # 1 also ranks 3
set_rank!(rnet, 3, 1, 2)   # 3 also ranks 1 (mutual in top 2)

println(compute(RankMutual(1), rnet))  # 1.0 (pair 1-2)
println(compute(RankMutual(2), rnet))  # 2.0 (pairs 1-2 and 1-3)
```

**Use cases**:
- `RankMutual(1)`: Best friend reciprocity in friendship networks
- `RankMutual(3)`: Close-circle reciprocity
- Multiple `RankMutual` terms with different cutoffs reveal how reciprocity varies by rank depth

### RankDeference

Deference: tendency to rank those who rank you highly also highly.

```julia
RankDeference(k::Int=3)
```

**Statistic**:

$$g(r) = \frac{1}{2}|\{(i,j) : r_{ij} \leq k, \, r_{ji} \leq k, \, i \neq j\}|$$

**Parameters**:
- `k`: Cutoff for "high" ranking

**Interpretation**:
- Similar to `RankMutual` but captures the deference aspect specifically
- Positive coefficient: actors defer to those who value them
- Common in hierarchical social structures

**Example**:

```julia
term = RankDeference(3)
println(name(term))  # "rank.deference.3"
```

## Transitivity

### RankTransitivity

Transitive ranking: if $i$ ranks $j$ highly and $j$ ranks $k$ highly, then $i$ tends to rank $k$ highly.

```julia
RankTransitivity(k::Int=3)
```

**Statistic**:

$$g(r) = |\{(i,j,k) : i \neq j \neq k, \, r_{ij} \leq k, \, r_{jk} \leq k, \, r_{ik} \leq k\}|$$

**Parameters**:
- `k`: Cutoff for "high" ranking

**Interpretation**:
- Captures the tendency for rankings to be transitive
- "Friends of my friends are also my friends"
- Positive coefficient: ranking preferences propagate through the network
- Analogous to triangle closure in binary ERGMs but using rank thresholds

**Visual representation**:

```text
  j
 / \
i   k    (all three rank each other in top k)
 \ /
  (transitive triangle)
```

**Example**:

```julia
rnet = RankNetwork(5; max_rank=3)

# Transitive triple: 1->2, 2->3, 1->3 all in top 3
set_rank!(rnet, 1, 2, 1)
set_rank!(rnet, 2, 3, 1)
set_rank!(rnet, 1, 3, 2)

println(compute(RankTransitivity(3), rnet))  # Counts transitive triples
```

## Consensus

### RankNonconsensus

Measures disagreement in rankings: cases where one actor ranks $j$ above $k$, but another actor ranks $k$ above $j$.

```julia
RankNonconsensus()
```

**Statistic**:

$$g(r) = |\{(i, l, j, k) : r_{ij} < r_{ik}, \, r_{lk} < r_{lj}, \, i \neq l\}|$$

**Interpretation**:
- Counts pairwise disagreements across rankers
- Negative coefficient: tendency toward consensus (actors agree on who is ranked highly)
- Positive coefficient: tendency toward dissensus (actors disagree)
- Related to Kendall's tau distance between rankings

**Example**:

```julia
rnet = RankNetwork(5; max_rank=3)

# Actor 1 ranks 2 above 3
set_rank!(rnet, 1, 2, 1)
set_rank!(rnet, 1, 3, 2)

# Actor 4 ranks 3 above 2 (disagreement!)
set_rank!(rnet, 4, 3, 1)
set_rank!(rnet, 4, 2, 2)

term = RankNonconsensus()
println(compute(term, rnet))  # Counts disagreements
```

**Use cases**:
- Testing whether rankings reflect a shared hierarchy
- Comparing structured vs. random ranking patterns
- Studying opinion polarization

## Local Structure

### RankLocaltriangle

Counts local triangles: cases where two actors ranked by the same ego also rank each other.

```julia
RankLocaltriangle()
```

**Statistic**:

$$g(r) = \sum_i \sum_{j < k \in \text{ranked\_by}(i)} \mathbb{I}(r_{jk} \text{ exists} \wedge r_{kj} \text{ exists})$$

**Interpretation**:
- Captures clustering within ego's ranked alters
- Positive coefficient: ego's top choices tend to also rank each other
- Reflects embedded social circles or cliques
- Analogous to clustering coefficient but for rank data

**Example**:

```julia
rnet = RankNetwork(5; max_rank=3)

# Actor 1 ranks Actors 2 and 3
set_rank!(rnet, 1, 2, 1)
set_rank!(rnet, 1, 3, 2)

# Actors 2 and 3 also rank each other (local triangle)
set_rank!(rnet, 2, 3, 1)
set_rank!(rnet, 3, 2, 1)

term = RankLocaltriangle()
println(compute(term, rnet))  # Counts such local triangles
```

## Attribute Terms

### RankNodecov

Rank-weighted node covariate effect: tests whether actors with higher attribute values tend to be ranked more highly (lower rank numbers).

```julia
RankNodecov(attr::Symbol)
```

**Statistic**:

$$g(r) = \sum_{(i,j)} \frac{1}{r_{ij}} \cdot (x_i + x_j)$$

Each ranking is weighted by the inverse of the rank value, so higher-ranked dyads contribute more.

**Parameters**:
- `attr`: Symbol naming the vertex attribute

**Interpretation**:
- Positive coefficient: actors with higher attribute values appear in higher rankings
- Negative coefficient: actors with lower attribute values are preferred
- The inverse-rank weighting ensures that top positions matter more

**Example**:

```julia
# Set vertex attributes
set_vertex_attribute!(rnet.network, 1, :grade, 10)
set_vertex_attribute!(rnet.network, 2, :grade, 11)
set_vertex_attribute!(rnet.network, 3, :grade, 12)

term = RankNodecov(:grade)
println(name(term))  # "rank.nodecov.grade"
```

### RankAbsdiff

Effect of absolute difference in a node attribute on rankings.

```julia
RankAbsdiff(attr::Symbol)
```

**Statistic**:

$$g(r) = \sum_{(i,j)} \frac{|x_i - x_j|}{r_{ij}}$$

**Parameters**:
- `attr`: Symbol naming the vertex attribute

**Interpretation**:
- Negative coefficient: homophily -- actors prefer to rank those similar to themselves
- Positive coefficient: heterophily -- actors prefer dissimilar others
- Weighted by inverse rank, so the effect is stronger for top-ranked alters

**Example**:

```julia
term = RankAbsdiff(:grade)
println(name(term))  # "rank.absdiff.grade"
```

## Choosing Terms

### By Research Question

| Question | Recommended Terms |
|----------|-------------------|
| How many rankings do actors make? | `RankEdges` |
| Is there best-friend reciprocity? | `RankMutual(1)` |
| Do close friends reciprocate? | `RankMutual(3)` |
| Are preferences transitive? | `RankTransitivity(k)` |
| Is there consensus? | `RankNonconsensus` |
| Do ranked alters know each other? | `RankLocaltriangle` |
| Does attribute X affect rankings? | `RankNodecov(:X)` |
| Does similarity drive rankings? | `RankAbsdiff(:X)` |

### Model Building Strategy

1. **Always start with RankEdges**: Controls baseline ranking density
2. **Add reciprocity**: `RankMutual(1)` is almost always relevant
3. **Test transitivity**: `RankTransitivity` captures structural clustering
4. **Test consensus**: `RankNonconsensus` reveals agreement patterns
5. **Add attributes**: Include relevant actor covariates last

### Example: Progressive Model Building

```julia
# Model 1: Density only
terms1 = [RankEdges()]

# Model 2: Add reciprocity
terms2 = [RankEdges(), RankMutual(1)]

# Model 3: Add transitivity and consensus
terms3 = [RankEdges(), RankMutual(1), RankTransitivity(3), RankNonconsensus()]

# Model 4: Full model with attributes
terms4 = [
    RankEdges(),
    RankMutual(1),
    RankMutual(3),
    RankTransitivity(3),
    RankNonconsensus(),
    RankNodecov(:grade),
    RankAbsdiff(:grade),
]
```

## Computing Terms Manually

You can compute any term on a RankNetwork without fitting a model:

```julia
rnet = RankNetwork(10; max_rank=5)
# ... populate with data ...

for term in [RankEdges(), RankMutual(1), RankNonconsensus()]
    val = compute(term, rnet)
    println("$(name(term)) = $val")
end
```

## Comparison with Binary ERGM Terms

| Binary ERGM Term | Rank ERGM Analogue | Key Difference |
|------------------|--------------------|----------------|
| Edges | RankEdges | Counts rankings vs. ties |
| Mutual | RankMutual(k) | Top-k mutual rankings vs. mutual ties |
| Triangle | RankTransitivity(k) | Transitive top-k vs. triangle closure |
| (none) | RankNonconsensus | Unique to rank data |
| (none) | RankDeference | Unique to rank data |
| NodeCov | RankNodecov | Rank-weighted vs. unweighted |
| AbsDiff | RankAbsdiff | Rank-weighted vs. unweighted |
