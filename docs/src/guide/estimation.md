# Estimation and Simulation

## Swap-based maximum pseudo-likelihood

[`ergm_rank`](@ref) fits by maximizing the swap-based pseudo-likelihood:
for each ego ``i`` and each unordered alter pair ``\{j, k\}``, the
conditional probability of the observed relative order of ``j`` and ``k``
given every other comparison is

```math
P(\text{observed order} \mid \text{rest}) =
\operatorname{logistic}\bigl(\theta' [g(y) - g(y^{(i:j\leftrightarrow k)})]\bigr)
```

where ``y^{(i:j\leftrightarrow k)}`` is the network with ego ``i``'s ranks
of ``j`` and ``k`` swapped. The product of these conditionals is maximized
by Newton-Raphson with step-halving; standard errors come from the inverse
observed information of the pseudo-likelihood.

This is the rank analogue of dyadwise MPLE — the AlterSwap move plays the
role of the edge toggle. It is fast and consistent, but like every
pseudo-likelihood it understates uncertainty when dependence is strong;
treat the standard errors as approximate.

```julia
result = ergm_rank(rnet, [RankDeference(), RankNonconformity(:localAND)])
result.coefficients
result.loglik        # maximized pseudo-log-likelihood
result.converged
```

## AlterSwap Metropolis simulation

[`simulate_rank_ergm`](@ref) samples from
``P(y) \propto \exp(\theta' g(y))`` on the complete-ordering space with
Metropolis steps: pick a random ego and two random alters, propose
swapping their ranks, accept with probability
``\min(1, \exp(\theta' \Delta g))``. The proposal is symmetric, so the
uniform reference cancels, and every visited state is a valid complete
ranking.

```julia
draws = simulate_rank_ergm(rnet, terms, θ;
                           n_sim = 200, burnin = 2000, interval = 50)
```

## Model checking

Compare observed statistics to their simulated distribution at the fitted
coefficients:

```julia
result = ergm_rank(rnet, terms)
draws = simulate_rank_ergm(result; n_sim = 200)
sim_stats = [compute(terms[1], d) for d in draws]
(observed = compute(terms[1], rnet), simulated_mean = mean(sim_stats))
```
