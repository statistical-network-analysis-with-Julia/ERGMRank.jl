# Estimation

ERGMRank.jl estimates rank ERGM parameters using Maximum Pseudo-Likelihood Estimation (MPLE). This page covers the estimation procedure, configuration, diagnostics, and best practices.

## Overview

The estimation process follows these steps:

1. **Compute observed statistics**: Calculate all term values on the observed rank network
2. **Construct pseudo-likelihood**: Approximate the joint likelihood using conditional likelihoods
3. **Optimize**: Find parameters that maximize the pseudo-likelihood via gradient-based optimization

## Maximum Pseudo-Likelihood Estimation

### Why MPLE?

The full likelihood of a rank ERGM requires computing a normalizing constant that sums over all possible ranking configurations. For $n$ actors each ranking up to $k$ others, the number of possible configurations is astronomically large. MPLE avoids this intractable computation.

### How It Works

MPLE approximates the full joint likelihood by conditioning on the rest of the network. For each actor's ranking, the conditional distribution given all other actors' rankings is computed, and the product of these conditional likelihoods is maximized.

The optimization proceeds by gradient descent, iteratively updating the coefficient vector to improve the pseudo-likelihood.

## Fitting a Model

### Basic Usage

```julia
result = ergm_rank(rnet, terms)
```

### Full Options

```julia
result = ergm_rank(rnet, terms;
    method = :mple,    # Estimation method
    maxiter = 100      # Maximum iterations
)
```

### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `method` | `Symbol` | Estimation method (currently `:mple`) | `:mple` |
| `maxiter` | `Int` | Maximum optimization iterations | `100` |

### Alternative Syntax

The `fit_rank_ergm` function is an alias for `ergm_rank`:

```julia
# These are equivalent
result = ergm_rank(rnet, terms)
result = fit_rank_ergm(rnet, terms)
```

## Understanding Results

The `RankERGMResult` object contains:

| Field | Type | Description |
|-------|------|-------------|
| `model` | `RankERGMModel` | The fitted model specification |
| `coefficients` | `Vector{Float64}` | Estimated coefficients |
| `std_errors` | `Vector{Float64}` | Standard errors |
| `loglik` | `Float64` | Log-pseudo-likelihood at convergence |
| `converged` | `Bool` | Whether optimization converged |

### Displaying Results

```julia
println(result)
```

Output:

```text
Rank ERGM Results
=================
Log-likelihood: -89.1234
Converged: true

Coefficients:
  rank.edges                -0.5432 (SE: 0.1234)
  rank.mutual.1              1.2345 (SE: 0.3456)
  rank.transitivity.3        0.6789 (SE: 0.2345)
```

### Accessing Results

```julia
# Coefficient vector
result.coefficients

# Standard errors
result.std_errors

# Model specification
result.model.terms      # Vector of terms
result.model.network    # Original RankNetwork
```

## Interpreting Coefficients

### General Interpretation

Coefficients describe the log-odds effect of each unit change in the corresponding statistic on the probability of the observed ranking configuration:

| Coefficient | Meaning |
|-------------|---------|
| $\theta > 0$ | Configurations with higher values of this statistic are more likely |
| $\theta < 0$ | Configurations with lower values of this statistic are more likely |
| $\theta = 0$ | The statistic does not influence the ranking distribution |

### Specific Interpretations

| Term | Coefficient | Interpretation |
|------|-------------|----------------|
| RankEdges | -0.5 | Sparse rankings; actors rank fewer others |
| RankMutual(1) | 1.5 | Strong best-friend reciprocity; odds ratio $\exp(1.5) \approx 4.5$ |
| RankMutual(3) | 0.8 | Moderate top-3 reciprocity |
| RankTransitivity(3) | 0.6 | Transitive preferences; friends of friends ranked highly |
| RankNonconsensus | -0.2 | Mild consensus; actors tend to agree on rankings |
| RankLocaltriangle | 0.4 | Ranked alters tend to know each other |
| RankNodecov(:X) | 0.3 | Actors with higher X tend to be ranked higher |
| RankAbsdiff(:X) | -0.5 | Homophily; actors rank those similar in X more highly |

### Comparing Cutoff Effects

When including multiple `RankMutual` or `RankTransitivity` terms with different cutoffs, the relative magnitudes reveal how the effect varies by rank depth:

```julia
terms = [
    RankEdges(),
    RankMutual(1),     # Best-friend reciprocity
    RankMutual(3),     # Top-3 reciprocity
    RankMutual(5),     # Top-5 reciprocity
]

result = ergm_rank(rnet, terms)

# If coefficients decrease with cutoff:
# reciprocity is strongest for top choices
```

## Convergence

### Checking Convergence

```julia
if result.converged
    println("Model converged")
else
    println("WARNING: Model did not converge")
end
```

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Non-convergence | `converged = false` | Increase `maxiter`, simplify model |
| Large coefficients | $|\theta| > 10$ | Possible near-separation; remove term |
| Large standard errors | SE >> coef | Multicollinearity or sparse data |

### Handling Non-Convergence

```julia
# Increase iterations
result = ergm_rank(rnet, terms; maxiter=500)

# Simplify model
terms_simple = [RankEdges(), RankMutual(1)]
result = ergm_rank(rnet, terms_simple)
```

## Model Comparison

### Progressive Model Building

```julia
# Model 1: Density only
result1 = ergm_rank(rnet, [RankEdges()])

# Model 2: Add reciprocity
result2 = ergm_rank(rnet, [RankEdges(), RankMutual(1)])

# Model 3: Add transitivity
result3 = ergm_rank(rnet, [RankEdges(), RankMutual(1), RankTransitivity(3)])

for (i, r) in enumerate([result1, result2, result3])
    println("Model $i: $(length(r.coefficients)) terms, converged=$(r.converged)")
end
```

### Simulation-Based Validation

After fitting, validate by simulating rank networks and comparing to observed data:

```julia
# Simulate from fitted parameters
terms = result.model.terms
coef = result.coefficients

rnet_sim = simulate_rank_ergm(nv(rnet), terms, coef;
    max_rank=rnet.max_rank,
    burnin=2000
)

# Compare observed vs simulated
for term in terms
    obs = compute(term, rnet)
    sim = compute(term, rnet_sim)
    println("$(name(term)): obs=$(round(obs, digits=2)), sim=$(round(sim, digits=2))")
end
```

## Simulation

### Generating Rank Networks

Simulate rank networks from specified parameters:

```julia
terms = [RankEdges(), RankMutual(1), RankTransitivity(3)]
coef = [-0.5, 1.5, 0.8]

rnet_sim = simulate_rank_ergm(20, terms, coef;
    max_rank=5,
    burnin=2000
)
```

### Simulation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_rank` | Maximum rank value | `n-1` |
| `burnin` | MCMC burn-in iterations | `1000` |

The simulation uses Metropolis-Hastings MCMC, proposing random changes to rankings (add, remove, or modify) and accepting/rejecting based on the change in the model's objective function.

## Best Practices

1. **Always include RankEdges**: Controls baseline density, analogous to the edges term
2. **Check convergence**: Verify `result.converged == true` before interpreting
3. **Start simple**: Begin with RankEdges plus one structural term
4. **Choose cutoffs thoughtfully**: The $k$ parameter in RankMutual and RankTransitivity should be meaningful (e.g., "top 3 friends")
5. **Validate with simulation**: Compare simulated and observed statistics
6. **Consider sample size**: Rank data from few actors may not support complex models
7. **Watch for separation**: Very large coefficients suggest near-perfect prediction by that term
8. **Test multiple cutoffs**: Compare effects at different rank thresholds to understand depth-dependent patterns
