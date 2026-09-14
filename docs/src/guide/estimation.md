# Estimation and Simulation

[`ergm_rank`](@ref) (= [`fit_ergm_rank`](@ref)) has two estimators,
selected by `method=`: the **swap pseudo-likelihood** (`:mple`, the
default — fast, deterministic, a different estimator from the MLE) and the
**MCMC maximum likelihood estimate** (`:mcmle`, the estimator of R
`ergm.rank`). Both return a [`RankERGMResult`](@ref) whose `method` field
says which one you have.

## Swap-based maximum pseudo-likelihood (`method=:mple`)

[`ergm_rank`](@ref) fits by default by maximizing the swap-based
pseudo-likelihood:
for each ego ``i`` and each unordered alter pair ``\{j, k\}``, the
conditional probability of the observed relative order of ``j`` and ``k``
given every other comparison is

```math
P(\text{observed order} \mid \text{rest}) =
\operatorname{logistic}\bigl(\theta' [g(y) - g(y^{(i:j\leftrightarrow k)})]\bigr)
```

where ``y^{(i:j\leftrightarrow k)}`` is the network with ego ``i``'s ranks
of ``j`` and ``k`` swapped. The product of these conditionals is a logistic
likelihood on the swap-difference rows with the response identically
`true` (the observed order is always the "success"), so it is maximized
with the ecosystem's shared `Networks.logistic_derivatives` kernel and
`Networks.newton_fit` optimizer (Newton-Raphson with step-halving; the
same bindings `ERGM.mple` runs on); standard errors come from the inverse
observed information of the pseudo-likelihood.

This is the rank analogue of dyadwise MPLE — the AlterSwap move plays the
role of the edge toggle. It is fast, and it is what the estimator actually
does: it maximizes a product of pairwise-swap conditionals.

!!! warning "Consistency is not claimed; standard errors are anticonservative"
    Swap-MPLE is **not** the MCMC MLE that R's `ergm.rank` computes, and no
    consistency result for it is established in this package — the
    comparisons entering the product are not independent, and no asymptotic
    regime or set of assumptions under which the estimator is consistent has
    been identified. Each factor of the product is the conditional
    probability of the observed ranking ``y`` against the single alternative
    ``y_{\text{swapped}}`` (ego's ranks of j and k exchanged) — not "the
    order of j and k given the rest of the rankings": a non-adjacent swap
    also reorders j and k relative to the alters ranked between them. **The
    swap pseudo-likelihood never equals the likelihood**, not even for a term
    that decomposes over egos such as `RankNodeICov`: the comparisons within
    one ego's row must form a total order, so there is no rank analogue of
    dyad independence under which the product of two-state conditionals is
    the likelihood. On the 4-actor example network with
    `RankNodeICov([1, 2, 3, 4])`, exact enumeration of all 1296 orderings
    gives the MLE ``θ = -0.0767`` (log-likelihood ``-7.015``) while the
    swap-MPLE is ``-0.0880`` (pseudo-log-likelihood ``-8.057``);
    `method=:mcmle` run to a tight convergence threshold
    (`n_samples=16384, conv_threshold=0.01`) lands at ``-0.0772``, 0.0005
    from the exact MLE, where the swap-MPLE never moves (the test suite pins
    both). Its large-sample behaviour here is
    uncharacterized, and MPLE for dependent ERGMs is known to be biased in
    finite samples.

    On Newcomb's fraternity ranks the swap-MPLE of
    `rank.deference + rank.nonconformity` is `[-0.1409, -0.00585]` against
    `ergm.rank`'s MCMC MLE `[-0.1531, -0.00659]` — a gap of 16× and 13×
    R's own seed-to-seed spread (systematic, not Monte-Carlo), though only
    0.30 and 0.43 of an R standard error. The test suite characterises this
    gap with a provenanced fixture (larger than 5 R seed-sd, smaller than
    0.6 of an R standard error), never tolerates it: **the swap-MPLE does
    not reproduce the coefficients of `ergm.rank`; `method=:mcmle` (below)
    does.**

    Standard errors are the **inverse observed pseudo-Hessian**. That
    curvature treats the overlapping swap comparisons as independent, so the
    standard errors are expected to be **anticonservative** — too small,
    yielding over-narrow confidence intervals and over-liberal tests — under
    dependence. On the Newcomb fixture they are 3.9× and 2.0× narrower than
    the MLE's. Use them as a rough guide only, or refit with
    `se=:bootstrap` (below).

```julia
using ERGMRank, Random

rng = Xoshiro(1)
rnet = RankNetwork(8)              # index-order rankings
for _ in 1:60                      # randomize with AlterSwap moves
    ego, j, k = rand(rng, 1:8), rand(rng, 1:8), rand(rng, 1:8)
    (j == ego || k == ego || j == k) && continue
    swap_ranks!(rnet, ego, j, k)
end

result = ergm_rank(rnet, [RankDeference(), RankNonconformity(:localAND)])
result.coefficients
result.loglik        # maximized pseudo-log-likelihood
result.converged
```

## The StatsAPI surface

Every fitted [`RankERGMResult`](@ref) answers the ecosystem's ten StatsAPI
verbs, and `coeftable(result)` is exactly the table `show(result)` prints:

```julia
coef(result), stderror(result), vcov(result)
confint(result)              # Wald limits from the fit's own standard errors
loglikelihood(result)        # the maximized swap pseudo-log-likelihood
nobs(result)                 # (ego, unordered alter pair) comparisons: 8·7·6/2
dof(result)                  # finite coefficients
aic(result), bic(result)     # pseudo-AIC / pseudo-BIC (see below)
tbl = coeftable(result)
tbl["rank.deference"].p_value == tbl[1].p_value   # true
```

`aic` and `bic` are information criteria of the **pseudo**-likelihood: they
rank models fit by `ergm_rank` against each other on the same network and
nothing else, and they are not comparable to R `ergm.rank`'s `logLik`/`AIC`
(a bridge-sampled estimate of the true likelihood). `nobs(result)` is the
number of swap comparisons for this estimator — its observations — and
`n(n−1)` ordered dyads (R's `nobs.ergm`) for a `:mcmle` fit; see
[`nobs`](@ref). When no comparison is left to estimate on (every one
changes a statistic dropped at the boundary, below), the
pseudo-log-likelihood is undefined and `loglikelihood`/`aic`/`bic` are
`NaN`, printed as "not defined". The p-values come from the shared
`Networks.z_pvalues` (floored at `floatmin(Float64)`, never `0.0`).

## Robust standard errors: `se=:bootstrap`

`se=:bootstrap` simulates `n_boot` rank networks at ``\hat\theta`` with the
AlterSwap sampler, refits the swap MPLE on each, and reports the empirical
covariance of the refits — the same option, keywords (`n_boot`,
`boot_burnin`, `boot_interval`, `rng`) and semantics as `ERGM.mple`'s, on the
one shared `Networks.bootstrap_cov` loop. The point estimates are unchanged;
only the covariance is replaced, and `se_method(result)` reports which one
you have.

```julia
boot = ergm_rank(rnet, [RankDeference(), RankNonconformity(:localAND)];
                 se=:bootstrap, n_boot=40, rng=Xoshiro(2))
coef(boot) == coef(result)                    # true: the point estimate is the same
se_method(boot)                               # :bootstrap
size(boot.boot_replicates)                    # (40, 2): every refit, one row each
```

A replicate on which the swap-MPLE does not exist — the *simulated* ranking
puts a statistic at the boundary of its attainable range, so its refit is
``\mp\infty`` — is **excluded** from the covariance: it stays a `NaN` row of
`boot_replicates`, the exclusion is warned about once and recorded in
`approximations(boot)`, and fewer than two finite refits is an
`ArgumentError`. This is a statement about the simulated replicates, not
about the observed ranking. `se=:bootstrap` is refused outright when a
coefficient of the fit itself is fixed at ``\pm\infty`` (a ranking cannot be
simulated at an infinite coefficient).

## When the swap-MPLE does not exist

The estimator can fail to have a finite maximum in two ways, and both are
loud: a warning at fit time, an entry in `approximations(result)`, and a
line printed directly under `Converged:` in `show(result)`.

**A statistic at the boundary of its attainable range** (R's `drop=TRUE`
semantics). If no single swap of the observed ranking can lower a statistic
(every swap-difference row has one sign), the pseudo-log-likelihood
increases monotonically as its coefficient goes to ``-\infty``; the mirror
case goes to ``+\infty``. `RankInconsistency(rnet)` fitted to `rnet` itself
is the textbook example — the observed ranking is *at* zero inconsistency.
As R `ergm` does, the coefficient is fixed at ``\mp\infty`` with standard
error 0 (z = ``\mp\infty``, p = 0 in the table), a warning quotes R's
sentence ("observed statistic(s) … are at their smallest attainable
values"), and the remaining coefficients are estimated on the swap
comparisons the dropped statistics do not change — the exact limit of the
pseudo-likelihood. `dof(result)` counts only the finite coefficients, and
`bic` uses the number of comparisons that were kept (`result.n_kept`).

```julia
using Test
self = ergm_rank(rnet, [RankInconsistency(rnet)])   # warns: smallest attainable value
coef(self)                                          # [-Inf]
stderror(self)                                      # [0.0]
dof(self)                                           # 0
@test_throws ArgumentError ergm_rank(rnet, [RankInconsistency(rnet)]; se=:bootstrap)
```

If *every* comparison changes a dropped statistic, nothing is left to
estimate the other coefficients on: they are returned as `NaN` with
`converged == false` and a warning that says they are not identified.

**Perfect (quasi-complete) separation** that the column test cannot see: a
*combination* of statistics that no single swap lowers. The 4-actor example
network of the README separates under `RankDeference() +
RankNonconformity()` (the direction ``(1, 1)`` gives ``D\beta \ge 0`` on
every row), and used to "converge" silently to ``\theta \approx (9.4, 10.1)``
with standard errors of 7,066. The fit now returns `converged == false`,
warns with R's sentence ("The MPLE does not exist!"), and says so in
`approximations`: the coefficients are the point at which Newton stopped
on its flat asymptote and mean nothing. Remove a term or collect more
actors.

## Non-convergence

A fit that exhausts `maxiter` is returned with `converged == false`, a
warning at fit time quoting the gradient norm, the caveat printed under
`Converged: false`, and an entry in `approximations(result)`. The
coefficients are the last Newton iterate; increase `maxiter`.

```julia
unconv = ergm_rank(rnet, [RankDeference(), RankNonconformity(:localAND)]; maxiter=1)
unconv.converged                                            # false
any(occursin("not converge", a) for a in approximations(unconv))   # true
```

## MCMC MLE (`method=:mcmle`)

`method=:mcmle` fits the **MCMC maximum likelihood estimate** — the
estimator R's `ergm.rank` computes — on the same iteration as `ERGM.mcmle`,
with the AlterSwap chain (below) as the sampler. Starting from the
swap-MPLE (or `init=`), each iteration draws `n_samples` statistics vectors
``g(Y)`` along the chain at the current ``\theta`` (the running statistics
are kept current from the accepted swaps' change statistics, never
recomputed per draw) and takes the Hummel partial Newton step

```math
\theta \leftarrow \theta + \gamma\,\hat\Sigma^{-1}\bigl(g(y_{\text{obs}}) - \bar g\bigr),
```

where ``\hat\Sigma`` is the sampled covariance and the step length
``\gamma`` starts at `gamma0`, grows (at most doubling per iteration) while
the observed statistics lie outside the sampled cloud (the 95 % Mahalanobis
radius), and is 1 once the cloud covers them; each step is capped at norm
`max_step_norm`. Convergence is declared only at ``\gamma = 1`` and when
both tests of `ERGM.mcmc_convergence` pass on the current sample: every
per-statistic t-ratio ``|g_{\text{obs}} - \bar g| / \operatorname{sd}(g)``
below `conv_threshold` (0.1) and a Hotelling ``T^2`` test of the mean
difference — with a Geyer, autocorrelation-adjusted effective sample size —
non-significant at `hotelling_alpha` (0.05). The final sample at
``\hat\theta`` gives the standard errors and the recorded report
`result.mcmc_convergence` (an `ERGM.MCMLEConvergence`: iterations, step
length, t-ratios, Hotelling p, effective sample size).

```julia
mle = ergm_rank(rnet, [RankDeference(), RankNonconformity(:localAND)];
                method=:mcmle, n_samples=512, rng=Xoshiro(7))
mle.method                                  # :mcmle
mle.converged                               # true
mle.mcmc_convergence.step_length            # 1.0
maximum(mle.mcmc_convergence.t_ratios) < 0.1   # true
se_method(mle)                              # :fisher
fit_metadata(mle).objective                 # :likelihood (the shared protocol, exported)
size(mle.mcmc_samples)                      # (512, 2): the final sample
```

**Standard errors** (`se=:fisher`, the only option under `:mcmle`) are the
inverse Fisher information of the final sample, ``V = \hat\Sigma^{-1}``,
**plus the Monte-Carlo component** ``V \Sigma_{\text{mc}} V`` of the
estimating equation (Hunter & Handcock 2006, §3.3; ``\Sigma_{\text{mc}}``
is the Geyer initial-sequence covariance of the sampled mean) — the same
covariance `ERGM.mcmle` reports. `result.vcov_fisher` is ``V`` alone, and
`show` prints R's `summary.ergm` "MCMC %" column, the share of each
standard error the Monte-Carlo term adds (0 at the default budget on a
well-mixing chain).

**Log-likelihood.** The uniform ordering model ``\theta = 0`` has the exact
normalizer ``\log Z(0) = n \log((n-1)!)``, and path sampling along
``\theta_u = u\hat\theta`` gives

```math
\log Z(\hat\theta) - \log Z(0) = \int_0^1 \mathbb{E}_{\theta_u}[g(Y)]'\hat\theta \, du,
```

integrated by the trapezoid rule over `bridge_rungs` segments (16), each
grid point one seeded chain of `bridge_samples` draws (default
`n_samples`). `loglikelihood(mle)` is that estimate of the **absolute**
log-likelihood ``\hat\theta' g(y) - \log Z(\hat\theta)``, `aic` and `bic`
follow from it, all with their own Monte-Carlo error; at ``n = 4`` the test
suite checks it against exact enumeration of all ``(3!)^4`` orderings. The
bridge runs after the final sample, so coefficients and standard errors
are bit-identical with and without it; **`bridge_rungs=0` skips it** and
`loglik`/`aic`/`bic` are `NaN`, recorded in `approximations(result)` and
printed as "not estimated".

!!! note "R's `logLik` is relative to θ = 0; ERGMRank's is absolute"
    R `ergm.rank`'s `logLik(fit)` is **not** the absolute log-likelihood:
    for a valued ERGM, `ergm` defines the null (θ = 0) model's likelihood
    as 0 — it prints "Null model likelihood calculation is not implemented
    for valued ERGMs at this time" — so `logLik(fit)` is the bridge-sampled
    ratio ``\log L(\hat\theta) - \log L(0)``. ERGMRank reports the absolute
    value, because ``\log Z(0) = n \log((n-1)!)`` is exact for the
    complete-ordering model. The two differ by that constant of the
    ranking size alone:

    ```math
    \texttt{logLik}_R = \texttt{loglikelihood(fit)} + n \log((n-1)!),
    ```

    and R's `AIC`/`BIC` are `aic(fit)`/`bic(fit)` shifted by
    ``-2n\log((n-1)!)``: ``\log 1296 \approx 7.17`` on 4 actors, ``17 \log
    16! \approx 521`` on Newcomb's 17 (so AIC/BIC differ by ≈ 1042 there).
    Differences between models fitted to the same ranking — the only
    comparison either convention licenses, as R's note says — are
    unaffected. `nobs(mle)` is R's for the likelihood: the ``n(n-1)``
    ordered dyads (`nobs.ergm` is `network.dyadcount`; 272 on Newcomb),
    which `bic` uses, not the ``n(n-1)(n-2)/2`` swap comparisons of the
    pseudo-likelihood. The golden fixture asserts the identity against
    `ergm.rank`'s own `logLik` on Newcomb within both bridges' Monte-Carlo
    error (R's "MC Std. Err." and Julia's seed-to-seed sd, both frozen in
    `test/fixtures/newcomb_rank.toml`).

```julia
quick = ergm_rank(rnet, [RankDeference(), RankNonconformity(:localAND)];
                  method=:mcmle, n_samples=512, bridge_rungs=0, rng=Xoshiro(7))
coef(quick) == coef(mle)                    # true: the bridge does not touch θ̂
isnan(loglikelihood(quick))                 # true
```

**The budget rule.** `burnin` and `interval` default to the swap-scaled
rule of the sampler (``20\,n_{\text{swaps}}`` and ``\max(100,
n_{\text{swaps}} \div 10)``), `n_samples=1024` draws per iteration and
`maxiter=20` iterations. `n_chains` splits the draws over independent
chains, each burned in from the observed ranking and seeded from `rng` in
order, then concatenated — a fit depends only on `rng` and `n_chains`,
never on the thread count, and two fits with the same `rng` are identical
bit for bit. Time scales as (iterations + 1 + `bridge_rungs` + 1) ×
(`burnin` + `n_samples` × `interval`) steps, each step O(n)–O(n²) per
term: the 17-actor Newcomb fit at `ergm.rank`'s own budget
(`n_samples=2048, burnin=8192, interval=512`) converges in 2–3 iterations
and takes about 25 s without the bridge.

**Non-convergence is loud.** A fit that exhausts `maxiter` warns with the
last max t-ratio, Hotelling p-value and step length, lists the caveat with
those numbers in `approximations(result)`, and prints it under
`Converged: false`. Continue it from where it stopped with
`init=coef(result)`:

```julia
short = ergm_rank(rnet, [RankDeference(), RankNonconformity(:localAND)];
                  method=:mcmle, maxiter=1, n_samples=64, init=[1.5, 0.2],
                  bridge_rungs=0, rng=Xoshiro(1))          # warns: did not converge
short.converged                                             # false
any(occursin("did not converge", a) for a in approximations(short))   # true
```

**What is refused.** A statistic at the boundary of its attainable range
under single swaps has no swap-MPLE to start from: an `ArgumentError`
unless `init=` is supplied, in which case a warning says the MLE may not
exist and the convergence tests decide. A separated swap-MPLE start (R:
"The MPLE does not exist!") is refused the same way, pointing at `init=`.
`se=:hessian`/`se=:bootstrap` are refused under `:mcmle` by the shared
`Networks.check_se` (the vocabulary is `(:fisher,)`), and an unknown
`method` names both estimators.

**Against R.** The golden fixture (`test/fixtures/newcomb_rank.toml`,
generated by `test/fixtures/r/newcomb_rank.R`) asserts the MCMLE of
`rank.deference + rank.nonconformity("all")` on Newcomb week 1 against
`ergm.rank` 4.1.2 at R's own MCMC budget: the coefficients within
``4 \times (\text{sd}_R + \text{sd}_{\text{Julia}})`` — the two
implementations' seed-to-seed spreads, R's 7.4e-4/5.6e-5 and Julia's
5.1e-4/7.2e-5, both frozen in the fixture — on two Julia seeds, the
standard errors within 15 % (observed: under 4 %), and the bridge
log-likelihood (under R's relative convention, above) within
``4 \times (\text{MC se}_R + \text{sd}_{\text{Julia}})``. The mean Julia
estimate sits 0.8 and 0.7 R seed-sd from R's; the swap-MPLE sat 16 and 13
away.

## Newcomb week 1: the two estimators on the real data

The ranking every number above rests on is bundled: [`newcomb_week1`](@ref)
returns R `ergm.rank`'s `newcomb[[1]]` — 17 fraternity men, each ranking
the other 16 (greater = higher standing) — verbatim the ranking the golden
fixture froze. The headline comparison is then two calls. The swap-MPLE
takes milliseconds; the MCMLE below runs at a reduced budget (a few
seconds) so the page stays executable — at `ergm.rank`'s budget
(`n_samples=2048, burnin=8192, interval=512`) it takes about 25 s and lands
inside R's seed-to-seed spread of `[-0.1531, -0.00659]`:

```julia
using ERGMRank, Random
newcomb = newcomb_week1()
newcomb.n                                              # 17
compute(RankDeference(), newcomb)                      # 844.0  (ergm.rank: summary())
compute(RankNonconformity(:all), newcomb)              # 12748.0

pl = ergm_rank(newcomb, [RankDeference(), RankNonconformity(:all)])
pl.converged                                           # true
round.(coef(pl); digits=4) == [-0.1409, -0.0059]       # true: the swap-MPLE

ml = ergm_rank(newcomb, [RankDeference(), RankNonconformity(:all)];
               method=:mcmle, n_samples=256, burnin=4000, interval=100,
               bridge_rungs=0, rng=Xoshiro(1))
ml.method                                              # :mcmle
nobs(ml) == 17 * 16                                    # true: R's dyad count; nobs(pl) is 2040 comparisons
abs(coef(ml)[1] - (-0.1531)) < 3 * stderror(ml)[1]    # true: within the MLE's uncertainty
```

The fixture testset runs the same MCMLE at R's budget on two seeds and
asserts the coefficients, standard errors and (with `bridge_rungs=16`) the
log-likelihood against the R values.

## AlterSwap Metropolis simulation

[`simulate_rank_ergm`](@ref) samples from
``P(y) \propto \exp(\theta' g(y))`` on the complete-ordering space with
Metropolis steps: pick a random ego and two random alters, propose
swapping their ranks, accept with probability
``\min(1, \exp(\theta' \Delta g))``. The proposal is symmetric, so the
uniform reference cancels, and every visited state is a valid complete
ranking.

**Change statistics.** ``\Delta g`` is the vector of per-term
[`swap_change`](@ref)`(term, rnet, ego, j, k)` values — the port of
`ergm.rank`'s `wtchangestats_rank.c` change functions. A swap of ego's
ranks of ``j`` and ``k`` changes ego's relative order of exactly the alter
pairs that involve ``j`` or ``k`` (and only those whose other member is
ranked between the two), so every term's change is a sum over those O(n)
pairs of whatever the term multiplies the comparison by; the two
nonconformity variants also loop over the other actor whose comparison is
matched, O(n²). Nothing recomputes a full statistic (O(n³)–O(n⁴)), nothing
allocates, and the result equals `compute(term, y_swapped) − compute(term,
y)` exactly — the test suite asserts `==` against that brute-force oracle
on hundreds of random swaps and on every swap of the R-validated 4-actor
network.

**The shared kernel.** The loop is the ecosystem's one Metropolis kernel
`ERGM.mh_toggle!` (accept/reject arithmetic, burn-in, thinning), fed the
swap `(ego, j, k)` as its move, `swap_change` writing into the kernel's
workspace as its change statistics, and `swap_ranks!` as its state
mutation. A swap has no "removal" direction, so the kernel never negates
the log-ratio. A step allocates 0 bytes (pinned), the sampled sequence is
bit-identical to the hand-written loop the kernel replaced (pinned as
literals), and all randomness flows through `rng`.

**Scaled defaults.** `burnin` and `interval` default to `nothing`, resolved
by the ecosystem's one dyad-scaled rule (ERGM.jl's `burnin = 20 n`,
`interval = max(100, n ÷ 10)`) applied to the size of the swap proposal space ``n_{\text{swaps}} = n(n-1)(n-2)/2``:
``\text{burnin} = 20\,n_{\text{swaps}}`` steps and
``\text{interval} = \max(100, n_{\text{swaps}} \div 10)`` steps between
recorded draws — for 8 actors (168 swaps), 3,360 and 100; for 17 actors
(2,040 swaps), 40,800 and 204. A chain that proposes one swap per step
needs a number of steps proportional to the number of swaps to move every
comparison a bounded number of times, so the old fixed `500`/`50` was too
small for any ranking of more than a handful of actors. An explicit
integer is honoured as given; `gof` and the `se=:bootstrap` refits resolve
their `burnin`/`interval` (`boot_burnin`/`boot_interval`) by the same rule.

```julia
terms = [RankDeference(), RankNonconformity(:localAND)]
θ = result.coefficients

n = rnet.n
n_swaps = n * (n - 1) * (n - 2) ÷ 2               # 168 for 8 actors
(burnin = 20n_swaps, interval = max(100, n_swaps ÷ 10))   # (burnin = 3360, interval = 100)
draws = simulate_rank_ergm(rnet, terms, θ; n_sim = 20, rng = Xoshiro(4))   # scaled defaults
explicit = simulate_rank_ergm(rnet, terms, θ;
                              n_sim = 200, burnin = 2000, interval = 50)     # honoured as given
swap_change(terms[1], rnet, 1, 2, 3)              # Δ rank.deference if ego 1 swaps 2 and 3
```

## Model checking

Compare observed statistics to their simulated distribution at the fitted
coefficients — by hand, or through the shared `gof`:

```julia
using Statistics: mean

result = ergm_rank(rnet, terms)
draws = simulate_rank_ergm(result; n_sim = 200)
sim_stats = [compute(terms[1], d) for d in draws]
(observed = compute(terms[1], rnet), simulated_mean = mean(sim_stats))

g = gof(result; n_sim = 100, burnin = 500, interval = 20, rng = Xoshiro(3))
g.statistics[1].labels        # ["rank.deference", "rank.nonconformity.localAND"]
```
