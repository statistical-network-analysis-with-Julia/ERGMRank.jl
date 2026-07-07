# Getting Started

## Building a rank network

A rank network is a square matrix of rank values: row ``i`` holds the
ranks ego ``i`` assigns to the alters, with greater values indicating
higher standing, and each row's off-diagonal entries a permutation of
`1:(n-1)`.

```julia
using ERGMRank

m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
rnet = as_rank_network(m)

get_rank(rnet, 1, 2)   # 3 — ego 1 ranks actor 2 highest
is_valid_ranking(rnet) # true
```

Passing a matrix that is not a valid set of complete rankings (duplicate
ranks within an ego, gaps, nonzero diagonal) throws an `ArgumentError`.

## Computing statistics

```julia
compute(RankDeference(), rnet)                    # 6.0
compute(RankNonconformity(:all), rnet)            # 10.0
compute(RankNodeICov([10, 20, 30, 40]), rnet)     # -40.0
```

These values are verified against R `ergm.rank` 4.1.2 in the test suite.

## Fitting a model

```julia
result = ergm_rank(rnet, [RankDeference(), RankNodeICov([10, 20, 30, 40])])
println(result)
result.coefficients
result.std_errors
```

## Simulating

```julia
draws = simulate_rank_ergm(rnet, [RankDeference()], [0.5];
                           n_sim = 100, burnin = 1000, interval = 20)
all(is_valid_ranking, draws)   # true
```

A positive coefficient on a statistic makes configurations with more of it
more probable; every draw remains a valid complete ranking because the
AlterSwap move only ever exchanges two of an ego's ranks.
