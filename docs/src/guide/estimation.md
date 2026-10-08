# Estimation and Simulation

[`ergm_rank`](@ref) (= [`fit_ergm_rank`](@ref)) has two estimators,
selected by `method=`: the **MCMC maximum likelihood estimate** (`:mcmle`,
the default — the estimator of R `ergm.rank`) and the **swap
pseudo-likelihood** (`:mple` — fast, deterministic, an estimator of this
package that differs from the MLE). Both return a [`RankERGMResult`](@ref)
whose `method` field says which one you have.

The examples on this page use one 8-actor ranking:

```julia
using ERGMRank, Random

rng = Xoshiro(1)
rnet = RankNetwork(8)              # index-order rankings
for _ in 1:60                      # randomize with AlterSwap moves
    ego, j, k = rand(rng, 1:8), rand(rng, 1:8), rand(rng, 1:8)
    (j == ego || k == ego || j == k) && continue
    swap_ranks!(rnet, ego, j, k)
end
terms = [RankDeference(), RankNonconformity(:localAND)]
```

## MCMC MLE (`method=:mcmle`, the default)

By default [`ergm_rank`](@ref) fits the **MCMC maximum likelihood
estimate** — the estimator R's `ergm.rank` computes — on the iteration of
`ERGM.mcmle`,
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
`max_step_norm`. Convergence is declared only at ``\gamma = 1`` and by
**R ergm 4's stopping rule** (`termination=:confidence`, the default): the
estimating equations at the *updated* coefficients — the current sample
importance-reweighted to them — must lie, with `conv_confidence` (99 %)
confidence, inside the tolerance ellipsoid
``x'(\texttt{conv\_precision}\cdot\hat\Sigma)^{-1}x \le 1``
(`conv_precision = 0.1`). When the test fails near the solution the next
sample is enlarged, by the factor the test asks for (at most 2 per
iteration, up to `max_n_samples`, default `16·n_samples`), as R does.
`termination=:hotelling` selects the older rule at a fixed sample size: every
per-statistic t-ratio ``|g_{\text{obs}} - \bar g| / \operatorname{sd}(g)``
below `conv_threshold` (0.1) and a Hotelling ``T^2`` test of the mean
difference — with a Geyer, autocorrelation-adjusted effective sample size —
non-significant at `hotelling_alpha` (0.05). The whole iteration — step,
stopping rule, boost, final sample, covariance — is ERGM.jl's
`ERGM.Extension.mcmle_solve`, the one `ERGM.mcmle` runs; this package supplies the
AlterSwap sampler.

**A step is taken at every iteration, the first included** (statnet's order,
and `ERGM.mcmle`'s): a start that already passes the tests is still refined
by one Monte-Carlo Newton step from its own sample, so the swap-MPLE start
is never returned as "the MCMLE". The returned ``\hat\theta`` is the update
from the sample that passed, and that sample is the final sample: the
standard errors, `result.mcmc_samples` and the report
`result.mcmc_convergence` (an `ERGM.MCMLEConvergence`: iterations, step
length, t-ratios, Hotelling p, effective sample size) come from it, and
`result.termination` records the rule, its p-value and the size of that
sample (`show` prints it under `Converged:`). A fit that ends unconverged
draws a fresh sample at the returned coefficients instead.

```julia
mle = ergm_rank(rnet, terms; n_samples=512, rng=Xoshiro(7))
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

integrated by composite Simpson's rule over `bridge_rungs` segments (16), each
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
    `ergm.rank`'s own `logLik` on Newcomb within 4 standard deviations of
    the difference of two bridges as noisy as R's (``4\sqrt{2}`` times R's
    "MC Std. Err.", frozen in `test/fixtures/newcomb_rank.toml`).

```julia
quick = ergm_rank(rnet, terms; n_samples=512, bridge_rungs=0, rng=Xoshiro(7))
coef(quick) == coef(mle)                    # true: the bridge does not touch θ̂
isnan(loglikelihood(quick))                 # true
```

**The budget rule.** `burnin` and `interval` default to the swap-scaled
rule of the sampler (``20\,n_{\text{swaps}}`` and ``\max(100,
n_{\text{swaps}} \div 10)``), `n_samples=1024` draws per iteration and
at most `maxiter=60` iterations (R ergm's `MCMLE.maxit`). `n_chains` splits the draws over independent
chains, each burned in from the observed ranking and seeded from `rng` in
order, then concatenated — a fit depends only on `rng` and `n_chains`,
never on the thread count, and two fits with the same `rng` are identical
bit for bit. Time scales as (iterations + 1 + `bridge_rungs` + 1) ×
(`burnin` + `n_samples` × `interval`) steps, each step O(n)–O(n²) per
term: the 17-actor Newcomb fit at `ergm.rank`'s own budget
(`n_samples=2048, burnin=8192, interval=512`) converges in 2–4 iterations
and takes about 25 s without the bridge; at the default budget, 10–20 s.

**Non-convergence is loud.** A fit that exhausts `maxiter` warns with the
last max t-ratio, Hotelling p-value and step length, lists the caveat with
those numbers in `approximations(result)`, and prints it under
`Converged: false`. Continue it from where it stopped with
`init=coef(result)`:

```julia
short = ergm_rank(rnet, terms; maxiter=1, n_samples=64, init=[1.5, 0.2],
                  bridge_rungs=0, rng=Xoshiro(1))          # warns: did not converge
short.converged                                             # false
any(occursin("did not converge", a) for a in approximations(short))   # true
```

**What is refused.** A statistic observed at an end of its attainable
range — a count (nonconformity, the unweighted inconsistency, deference)
observed at 0 — has no finite MLE. R `ergm`'s `drop=TRUE` would fix its
coefficient at ``\mp\infty`` and estimate the rest by MCMC on the rankings
that share the observed value, but those rankings are not connected by
single swaps (on 4 actors the 138 rankings with local nonconformity 0 fall
into 22 swap classes; the 24 with nonconformity 0 are isolated), so the
AlterSwap chain cannot be held there: the MCMLE refuses such a model with an
`ArgumentError` that points at `method=:mple`, which applies the drop.
(`ergm.rank`'s terms declare no bound, so R does not drop either; its
sampler stops moving.) A statistic extreme only among the single swaps of
the observed ranking has no swap-MPLE to start from, but that does not show
that no MLE exists: an `ArgumentError` unless `init=` is supplied, in which
case a warning says so and the convergence tests decide. A separated
swap-MPLE start (R: "The MPLE does not exist!") and a coefficient the
swap-MPLE cannot identify are refused the same way, pointing at `init=`.
`drop=false` (R's `control.ergm(drop=)`) refuses a boundary statistic under
either estimator.
`se=:hessian`/`se=:bootstrap` are refused under `:mcmle` (the vocabulary is
`(:fisher,)`; the message says to pass `method=:mple` with them), and an
unknown `method` names both estimators.

**Against R.** Two golden fixtures, generated by the checked-in
`test/fixtures/r/*.R` scripts with `ergm.rank` 4.1.2, assert the MCMLE at
R's own MCMC budget (`n_samples=2048, burnin=8192, interval=512`):

- `newcomb_rank.toml` — `rank.deference + rank.nonconformity("all")` on
  Newcomb week 1: the coefficients within ``4\sqrt{2}\,\text{sd}_R`` of R's
  fit (``\text{sd}_R`` is `ergm.rank`'s own seed-to-seed spread, frozen in
  the fixture: for the same estimator a Julia estimate is one more draw from
  it) on two Julia seeds, the standard errors within 15 % (observed: under
  5 %), and the bridge log-likelihood (under R's relative convention, above)
  within ``4\sqrt{2}`` times R's Monte-Carlo standard error. No tolerance
  uses a number measured on ERGMRank.jl.
- `newcomb2_rank.toml` — `rank.deference + rank.nonconformity("localAND") +
  rank.inconsistency(week 1)` on Newcomb week 2: coefficients within
  ``4\sqrt{1 + 1/5}\,\text{sd}_R`` of the mean of R's five replication
  fits, standard errors within 15 %.

## Swap-based maximum pseudo-likelihood (`method=:mple`)

`method=:mple` maximizes the swap-based pseudo-likelihood, an estimator of
this package (`ergm.rank` has no pseudo-likelihood): for each ego ``i`` and each unordered alter pair ``\{j, k\}``, the
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
with the ecosystem's shared `NetworkCore.logistic_derivatives` kernel and
`NetworkCore.newton_fit` optimizer (Newton-Raphson with step-halving; the
same bindings `ERGM.mple` runs on).

This is the rank analogue of dyadwise MPLE — the AlterSwap move plays the
role of the edge toggle. It takes milliseconds, is deterministic, and is the
MCMLE's starting point; it is useful for screening models before paying for
the MCMC fit.

!!! warning "A different estimator from the MLE; no consistency is claimed"
    Swap-MPLE is **not** the MCMC MLE that R's `ergm.rank` computes, and no
    consistency result for it is established in this package — the
    comparisons entering the product are not independent. Each factor of the
    product is the conditional probability of the observed ranking ``y``
    against the single alternative ``y_{\text{swapped}}`` (ego's ranks of j
    and k exchanged) — not "the order of j and k given the rest of the
    rankings": a non-adjacent swap also reorders j and k relative to the
    alters ranked between them. **The swap pseudo-likelihood never equals the
    likelihood**, not even for a term that decomposes over egos such as
    `RankNodeICov`: the comparisons within one ego's row must form a total
    order, so there is no rank analogue of dyad independence. On the 4-actor
    example network with `RankNodeICov([1, 2, 3, 4])`, exact enumeration of
    all 1296 orderings gives the MLE ``θ = -0.0767`` (log-likelihood
    ``-7.015``) while the swap-MPLE is ``-0.0880`` (pseudo-log-likelihood
    ``-8.057``); the MCMLE recovers the exact MLE up to Monte-Carlo error
    (the test suite pins both).

**How far the swap-MPLE is from the MLE depends on the model.** Two fitted
`ergm.rank` fixtures on Newcomb's fraternity measure it:

| Model | `ergm.rank` MCMLE | swap-MPLE | gap, in MLE standard errors | pseudo-Hessian SE vs MLE SE |
|:--|:--|:--|:--|:--|
| week 1: `rank.deference + rank.nonconformity("all")` | −0.1531, −0.00659 | −0.1409, −0.00585 | 0.30, 0.43 | 3.9×, 2.0× too small |
| week 2: `rank.deference + rank.nonconformity("localAND") + rank.inconsistency(week 1)` | −0.2053, −0.0044, −0.1377 | −0.1636, −0.0100, −0.1263 | 0.89, 0.83, 0.76 | 3.3×, 2.2×, 2.0× too small |

On week 1 the gap is small against the standard error but 16× and 13× R's
own seed-to-seed spread, i.e. systematic; on week 2 the nonconformity
coefficient is 2.3× the MLE's. The test suite characterises both gaps with
the provenanced fixtures and never tolerates them: **the swap-MPLE does not
reproduce the coefficients of `ergm.rank`; the default `method=:mcmle`
does.**

### Inference from a swap-MPLE fit

The inverse observed pseudo-Hessian treats the overlapping swap comparisons
as independent, so the standard errors it gives are too small. A simulation
(10 actors, 300 rankings drawn from a known model, 95 % Wald intervals for
the true coefficients; `deference + nodeicov` and `deference +
nonconformity`) measured:

| Standard errors | Coverage of the true coefficients |
|:--|:--|
| swap-MPLE, pseudo-Hessian | 0.59, 0.77 and 0.48, 0.64 |
| swap-MPLE, `se=:bootstrap` | 0.95, 0.97 and 0.98, 0.99 |
| MCMC MLE (the default) | 0.96, 0.95 and 0.96, 0.98 |

So a swap-MPLE fit reports its inference as `ERGM.mple` does under dyadic
dependence:

- **by default** — the estimates and the pseudo-Hessian standard errors, but
  no inference built on them: the z and p columns of `coeftable`/`show` are
  `NaN` with a note saying why, [`confint`](@ref) refuses with an
  `ArgumentError`, and `approximations(result)` records it
  (`result.inference_withheld`);
- **`se=:hessian`, passed explicitly** — the written opt-in to the naive
  Wald table, printed with the anticonservatism warning;
- **`se=:bootstrap`** — a parametric-bootstrap covariance with z, p and
  intervals (below).

```julia
result = ergm_rank(rnet, terms; method=:mple)
result.coefficients
result.loglik               # maximized pseudo-log-likelihood
result.converged
result.inference_withheld   # true
all(isnan, coeftable(result).p_values)        # true
naive = ergm_rank(rnet, terms; method=:mple, se=:hessian)
coef(naive) == coef(result)                   # true: only the presentation differs
size(confint(naive))                          # (2, 2): the naive Wald intervals, opted into
```

## The StatsAPI surface

Every fitted [`RankERGMResult`](@ref) answers the ecosystem's ten StatsAPI
verbs, and `coeftable(result)` is exactly the table `show(result)` prints:

```julia
coef(naive), stderror(naive), vcov(naive)
confint(naive)               # Wald limits from the fit's own standard errors
loglikelihood(naive)         # the maximized swap pseudo-log-likelihood
nobs(naive)                  # (ego, unordered alter pair) comparisons: 8·7·6/2
dof(naive)                   # finite coefficients
aic(naive), bic(naive)       # pseudo-AIC / pseudo-BIC (see below)
tbl = coeftable(naive)
tbl["rank.deference"].p_value == tbl[1].p_value   # true
```

For a swap-MPLE fit `aic` and `bic` are information criteria of the
**pseudo**-likelihood: they rank swap-MPLE fits against each other on the
same network and nothing else, and they are not comparable to R `ergm.rank`'s `logLik`/`AIC`
(a bridge-sampled estimate of the true likelihood). `nobs(result)` is the
number of swap comparisons for this estimator — its observations — and
`n(n−1)` ordered dyads (R's `nobs.ergm`) for a `:mcmle` fit; see
[`nobs`](@ref). When no comparison is left to estimate on (every one
changes a statistic dropped at the boundary, below), the
pseudo-log-likelihood is undefined and `loglikelihood`/`aic`/`bic` are
`NaN`, printed as "not defined". The p-values come from the shared
`NetworkCore.z_pvalues` (floored at `floatmin(Float64)`, never `0.0`).

## Bootstrap standard errors: `se=:bootstrap`

`se=:bootstrap` (under `method=:mple`) simulates `n_boot` rank networks at ``\hat\theta`` with the
AlterSwap sampler, refits the swap MPLE on each, and reports the empirical
covariance of the refits — the same option, keywords (`n_boot`,
`boot_burnin`, `boot_interval`, `rng`) and semantics as `ERGM.mple`'s, on the
one shared `NetworkCore.bootstrap_cov` loop. The point estimates are unchanged;
only the covariance is replaced, and `se_method(result)` reports which one
you have. It measures the sampling variability of the swap-MPLE (coverage in
the table above); it does not move the swap-MPLE toward the MLE.

The replicates are successive draws of one chain, and the covariance treats
them as independent. With the default `boot_interval` the thinning is
therefore **ESS-aware**: when the effective sample size of the model
statistics over the `n_boot` draws (at least 20) is below `n_boot/2`, the
draws are taken again, once, at an interval scaled up by `n_boot/ESS` (at
most 8×), and an effective sample size still below `n_boot/4` is warned
about. (On Newcomb's 17 actors the statistics of successive draws at the
default interval have lag-1 autocorrelations of 0.2 and 0.5; the adapted
interval is 4× longer.) An explicit `boot_interval` is honoured as given.

```julia
boot = ergm_rank(rnet, terms; method=:mple, se=:bootstrap, n_boot=40, rng=Xoshiro(2))
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
line printed directly under `Converged:` in `show(result)`. (The MCMLE,
which starts from the swap-MPLE, refuses both cases with an
`ArgumentError`; see above.)

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
self = ergm_rank(rnet, [RankInconsistency(rnet)]; method=:mple)   # warns: smallest attainable value
coef(self)                                          # [-Inf]
stderror(self)                                      # [0.0]
dof(self)                                           # 0
@test_throws ArgumentError ergm_rank(rnet, [RankInconsistency(rnet)]; method=:mple, se=:bootstrap)
@test_throws ArgumentError ergm_rank(rnet, [RankInconsistency(rnet)])   # the MCMLE refuses it
```

A count observed at 0 is dropped even when no swap changes it (its
attainable range, not the design, shows the bound), and `drop=false` refuses
a boundary statistic instead of fixing it. If *every* comparison changes a
dropped statistic, nothing is left to estimate the other coefficients on:
they are returned as `NaN` with `converged == false` and a warning that says
they are not identified.

**A statistic the swap comparisons cannot identify** — one that no swap
changes (a constant covariate), or a linear combination of the preceding
statistics — is reported as `NaN` (R's `NA`) with a warning and a note in
`show` and `approximations`; the other coefficients are exactly the fit
without it, and the fit stays converged.

```julia
flat = ergm_rank(rnet, [RankDeference(), RankNodeICov(ones(rnet.n))]; method=:mple)  # warns: not identified
isnan(coef(flat)[2])                                                          # true
coef(flat)[1] == coef(ergm_rank(rnet, [RankDeference()]; method=:mple))[1]   # true
```

**Separation** that the column test cannot see: a *combination* of
statistics that no single swap lowers. The 4-actor example network of the
README separates under `RankDeference() + RankNonconformity()` (the
direction ``(1, 1)`` gives ``D\beta \ge 0`` on every row), and used to
"converge" silently to ``\theta \approx (9.4, 10.1)`` with standard errors
of 7,066. Separation is decided exactly, on the swap design itself, by the
separation test the whole ecosystem shares (NetworkCore's
`logistic_separation`), and handled by the shared policy: the fit warns
(quoting R's "The MPLE does not exist!"), returns `converged == false`,
names the separated terms in `fit.separated_terms`, reports `NaN` z values,
p-values and `confint` even under `se=:hessian`, refuses `se=:bootstrap`,
and says so in `approximations`. The coefficients are the point at which
Newton stopped on its flat asymptote and mean nothing. Remove a term or
collect more actors.

## Non-convergence of the swap-MPLE

A swap-MPLE fit that exhausts `maxiter` is returned with `converged == false`, a
warning at fit time quoting the gradient norm, the caveat printed under
`Converged: false`, and an entry in `approximations(result)`. The
coefficients are the last Newton iterate; increase `maxiter`.

```julia
unconv = ergm_rank(rnet, terms; method=:mple, maxiter=1)
unconv.converged                                            # false
any(occursin("not converge", a) for a in approximations(unconv))   # true
```

## Newcomb week 1: the two estimators on the real data

The ranking every number above rests on is bundled: [`newcomb_week1`](@ref)
returns R `ergm.rank`'s `newcomb[[1]]` — 17 fraternity men, each ranking
the other 16 (greater = higher standing) — verbatim the ranking the golden
fixture froze. The headline comparison is then two calls. The swap-MPLE
takes milliseconds; the MCMLE below runs at a reduced budget (a few
seconds) so the page stays quick to execute — at the default budget it
takes 10–20 s, at `ergm.rank`'s (`n_samples=2048, burnin=8192,
interval=512`) about 25 s, and lands inside R's seed-to-seed spread of
`[-0.1531, -0.00659]`:

```julia
using ERGMRank, Random
newcomb = newcomb_week1()
newcomb.n                                              # 17
compute(RankDeference(), newcomb)                      # 844.0  (ergm.rank: summary())
compute(RankNonconformity(:all), newcomb)              # 12748.0

pl = ergm_rank(newcomb, [RankDeference(), RankNonconformity(:all)]; method=:mple)
pl.converged                                           # true
round.(coef(pl); digits=4) == [-0.1409, -0.0059]       # true: the swap-MPLE

ml = ergm_rank(newcomb, [RankDeference(), RankNonconformity(:all)];
               n_samples=256, burnin=4000, interval=100,
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
[`swap_change`](@ref)`(term, rnet, ego, j, k)` values. They are derived
independently from the term definitions as a swap move, which has no
counterpart in `ergm.rank`'s single-cell change functions. A swap of ego's
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
their `burnin`/`interval` (`boot_burnin`/`boot_interval`) by the same rule,
and lengthen a default interval when the draws are autocorrelated (below).

```julia
θ = coef(mle)

n = rnet.n
n_swaps = n * (n - 1) * (n - 2) ÷ 2               # 168 for 8 actors
(burnin = 20n_swaps, interval = max(100, n_swaps ÷ 10))   # (burnin = 3360, interval = 100)
draws = simulate_rank_ergm(rnet, terms, θ; n_sim = 20, rng = Xoshiro(4))   # scaled defaults
explicit = simulate_rank_ergm(rnet, terms, θ;
                              n_sim = 200, burnin = 2000, interval = 50)     # honoured as given
swap_change(terms[1], rnet, 1, 2, 3)              # Δ rank.deference if ego 1 swaps 2 and 3
```

## Model checking

[`gof`](@ref) simulates rankings from the fitted model and compares the
observed ranking with them: for each statistic the observed value, the
simulated mean and 95 % envelope, and a two-sided Monte-Carlo p-value (the
shared `NetworkCore.GOFResult`).

The first panel holds the model's own statistics. **For an MCMLE fit these
are matched in expectation by construction** — the MLE solves
``\mathbb{E}_\theta[g(Y)] = g(y)`` — so their p-values check that the fit
converged and nothing else. The fit of the model is judged on the auxiliary
panels, none of which is a model statistic:

| Panel | What it measures |
|:--|:--|
| `structural statistics (not in the model)` | those of `rank.deference`, `rank.nonconformity`, `rank.nonconformity.localAND` the model omits |
| `mean received rank (sorted)` | each actor's mean received rank, sorted: the popularity profile, the rank analogue of a degree distribution |
| `dyadic rank difference` | the number of dyads with ``\lvert y_{ij} - y_{ji} \rvert = d``: reciprocity of standing |
| `ego agreement (Kendall tau)` | the number of ego pairs by Kendall's τ between their rankings of the alters they share, in five bins |

The test suite checks the panels' power: rankings generated with a strong
popularity gradient and fitted without it pass on the model statistic and
are rejected on the received-rank profile.

The envelopes and p-values treat the simulated rankings as independent
draws. With the default `interval` the thinning is ESS-aware, exactly as for
the bootstrap above; an explicit `interval` is honoured, and draws that
remain autocorrelated are warned about.

```julia
using Statistics: mean

g = gof(mle; n_sim = 100, rng = Xoshiro(3))
[s.name for s in g.statistics]   # the model statistics and the auxiliary panels
g.statistics[1].labels           # ["rank.deference", "rank.nonconformity.localAND"]

# By hand: any statistic of the simulated rankings
draws = simulate_rank_ergm(mle; n_sim = 200, rng = Xoshiro(5))
sim_stats = [compute(RankNonconformity(:all), d) for d in draws]
(observed = compute(RankNonconformity(:all), rnet), simulated_mean = mean(sim_stats))
```
