"""
    ERGMRank.jl - ERGMs for Rank-Order Relational Data

Exponential-family random graph models for networks whose edge values are
ranks: each ego rank-orders all alters (Krivitsky & Butts 2017).

The sample space is the set of complete orderings of the alters by each
ego, with the discrete-uniform `CompleteOrderReference` over orderings.
Statistics follow the R `ergm.rank` package: higher rank values indicate
higher standing (`y[i,j] > y[i,k]` means ego `i` ranks `j` over `k`).

Port of the R ergm.rank package from the StatNet collection.
"""
module ERGMRank

using Distributions
using ERGM
using LinearAlgebra
using Random
using Statistics

import ERGM: name, compute
# Shared presentation infrastructure (Networks.jl): the ONE `gof` generic all
# model packages extend, plus the common coefficient-table printer and
# GOF containers
import Networks: gof, print_coeftable, GOFStatistic, GOFResult

# The ONE shared bootstrap loop (Networks.jl `src/bootstrap.jl`): simulate,
# refit, empirical covariance. `se=:bootstrap` supplies the two callbacks; the
# loop, the threading and the rng discipline are not reimplemented here.
import Networks: bootstrap_cov

# The shared result-metadata protocol (Networks.jl `src/results.jl`): the
# generic accessors that say what a fit actually did. Imported by name because
# ERGMRank adds methods for `RankERGMResult`; `fit_metadata(fit)` collects them.
import Networks: estimand, objective, is_exact, se_method, missing_method,
                 approximations
import StatsAPI
import StatsAPI: coef, stderror, vcov, loglikelihood, nobs, dof

# Core types
export RankNetwork, RankERGMModel, RankERGMResult
export CompleteOrderReference

# Rank access and manipulation
export get_rank, set_rank!, swap_ranks!, is_valid_ranking
export as_rank_network, rank_matrix

# Terms (matching R ergm.rank)
export RankDeference, RankNonconformity, RankNodeICov
export RankInconsistency, RankEdgeCov

# Estimation and simulation
export fit_ergm_rank, ergm_rank, fit_rank_ergm
export simulate_rank_ergm

# Goodness of fit (method of the shared Networks.jl `gof` generic)
export gof

# StatsAPI methods (re-exported so `coef(fit)` etc. work with just
# `using ERGMRank`)
export coef, stderror, vcov, loglikelihood, nobs, dof

# =============================================================================
# RankNetwork
# =============================================================================

"""
    RankNetwork

A network of complete rankings: each ego assigns every alter a distinct
rank in `1:(n-1)`, with **greater values indicating higher standing**
(`get_rank(rnet, i, j) > get_rank(rnet, i, k)` means ego `i` ranks `j`
over `k`), following the R `ergm.rank` convention.

# Fields
- `n::Int`: Number of actors
- `ranks::Matrix{Int}`: `ranks[i, j]` is the rank ego `i` assigns alter
  `j`; the diagonal is 0
"""
struct RankNetwork
    n::Int
    ranks::Matrix{Int}

    function RankNetwork(ranks::Matrix{Int}; validate::Bool=true)
        n = size(ranks, 1)
        size(ranks, 2) == n || throw(ArgumentError("rank matrix must be square"))
        if validate
            msg = _ranking_violation(ranks)
            isnothing(msg) || throw(ArgumentError(msg))
        end
        new(n, copy(ranks))
    end
end

"""
    RankNetwork(n::Int)

Create a rank network with `n` actors in which every ego ranks the alters
in index order.
"""
function RankNetwork(n::Int)
    ranks = zeros(Int, n, n)
    for i in 1:n
        r = 0
        for j in 1:n
            i == j && continue
            r += 1
            ranks[i, j] = r
        end
    end
    return RankNetwork(ranks; validate=false)
end

# Returns nothing when valid, or a description of the first violation
function _ranking_violation(ranks::Matrix{Int})
    n = size(ranks, 1)
    for i in 1:n
        ranks[i, i] == 0 ||
            return "ego $i has a nonzero self-rank; the diagonal must be 0"
        row = [ranks[i, j] for j in 1:n if j != i]
        sort(row) == collect(1:(n-1)) ||
            return "ego $i's ranks $(sort(row)) are not a permutation of 1:$(n-1); " *
                   "each ego must assign each alter a distinct rank"
    end
    return nothing
end

"""
    is_valid_ranking(rnet::RankNetwork) -> Bool

Check that every ego's ranks form a permutation of `1:(n-1)` — the
structural invariant of complete rank-order data.
"""
is_valid_ranking(rnet::RankNetwork) = isnothing(_ranking_violation(rnet.ranks))

Base.copy(rnet::RankNetwork) = RankNetwork(copy(rnet.ranks); validate=false)

nactors(rnet::RankNetwork) = rnet.n

function Base.show(io::IO, rnet::RankNetwork)
    print(io, "RankNetwork with $(rnet.n) actors ",
          "(complete orderings of $(rnet.n - 1) alters per ego)")
end

"""
    get_rank(rnet::RankNetwork, i, j) -> Int

The rank ego `i` assigns alter `j` (greater = higher standing); 0 for
`i == j`.
"""
get_rank(rnet::RankNetwork, i::Int, j::Int) = rnet.ranks[i, j]

"""
    set_rank!(rnet::RankNetwork, i, j, rank)

Set the rank ego `i` assigns alter `j`. This can transiently break the
per-ego permutation invariant; prefer [`swap_ranks!`](@ref), which
preserves it. Validate afterwards with [`is_valid_ranking`](@ref).
"""
function set_rank!(rnet::RankNetwork, i::Int, j::Int, rank::Int)
    i == j && throw(ArgumentError("cannot set a self-rank"))
    1 <= rank <= rnet.n - 1 ||
        throw(ArgumentError("rank must be in 1:$(rnet.n - 1)"))
    rnet.ranks[i, j] = rank
    return rnet
end

"""
    swap_ranks!(rnet::RankNetwork, ego, j, k)

Swap the ranks ego assigns alters `j` and `k` (the AlterSwap move). This
is the elementary move of the rank sample space: it always preserves the
complete-ordering invariant.
"""
function swap_ranks!(rnet::RankNetwork, ego::Int, j::Int, k::Int)
    (ego == j || ego == k) && throw(ArgumentError("ego cannot swap its own rank"))
    rnet.ranks[ego, j], rnet.ranks[ego, k] = rnet.ranks[ego, k], rnet.ranks[ego, j]
    return rnet
end

"""
    as_rank_network(mat::AbstractMatrix) -> RankNetwork

Build a `RankNetwork` from a matrix of rank values (row `i` holds ego
`i`'s ranks of the alters, greater = higher standing).
"""
as_rank_network(mat::AbstractMatrix) = RankNetwork(Matrix{Int}(mat))

"""
    rank_matrix(rnet::RankNetwork) -> Matrix{Int}

The matrix of rank values (a copy).
"""
rank_matrix(rnet::RankNetwork) = copy(rnet.ranks)

# =============================================================================
# Reference measure
# =============================================================================

"""
    CompleteOrderReference

The discrete-uniform reference measure over the complete orderings of the
alters by each ego — the reference measure of `ergm.rank`. Under this
reference every valid rank configuration has equal baseline weight, so it
cancels out of all likelihood ratios; it is the *sample-space constraint*
(each ego's ranks form a permutation) that carries the structure,
maintained by the AlterSwap move.
"""
struct CompleteOrderReference end

# =============================================================================
# Terms (statistics follow ergm.rank's wtchangestats_rank.c)
# =============================================================================

"""
    RankDeference <: AbstractERGMTerm

Deference (aversion), `rank.deference` in ergm.rank: the number of ordered
triples (ego `i`, deferred-to `l`, other `j`) such that `l` ranks `j` over
`i` while `i` ranks `l` over `j`.
"""
struct RankDeference <: AbstractERGMTerm end

name(::RankDeference) = "rank.deference"

function compute(::RankDeference, rnet::RankNetwork)
    n = rnet.n
    y = rnet.ranks
    total = 0.0
    for v1 in 1:n, v3 in 1:n
        v3 == v1 && continue
        for v2 in 1:n
            (v2 == v1 || v2 == v3) && continue
            if y[v3, v2] > y[v3, v1] && y[v1, v3] > y[v1, v2]
                total += 1.0
            end
        end
    end
    return total
end

"""
    RankNonconformity(variant=:all) <: AbstractERGMTerm

Nonconformity, `rank.nonconformity` in ergm.rank.

- `:all` — global nonconformity: over unordered actor pairs {i, j} and
  ordered alter pairs (k, l), count comparisons on which i and j disagree
  (`(y_ik > y_il) ≠ (y_jk > y_jl)`).
- `:localAND` — local nonconformity (Krivitsky & Butts): ego i disagrees
  with an actor l that i ranks over both j and k, counting cases where l
  ranks j over k while i ranks k at least as high as j.
"""
struct RankNonconformity <: AbstractERGMTerm
    variant::Symbol

    function RankNonconformity(variant::Symbol=:all)
        variant in (:all, :localAND) ||
            throw(ArgumentError("variant must be :all or :localAND"))
        new(variant)
    end
end

name(t::RankNonconformity) =
    t.variant == :all ? "rank.nonconformity" : "rank.nonconformity.localAND"

function compute(t::RankNonconformity, rnet::RankNetwork)
    n = rnet.n
    y = rnet.ranks
    total = 0.0

    if t.variant == :all
        for v1 in 1:n, v2 in 1:(v1-1)
            for v3 in 1:n
                (v3 == v1 || v3 == v2) && continue
                for v4 in 1:n
                    (v4 == v1 || v4 == v2 || v4 == v3) && continue
                    if (y[v1, v3] > y[v1, v4]) != (y[v2, v3] > y[v2, v4])
                        total += 1.0
                    end
                end
            end
        end
    else  # :localAND (v1=i, v2=j, v3=l, v4=k in Krivitsky & Butts)
        for v1 in 1:n, v2 in 1:n
            v2 == v1 && continue
            for v3 in 1:n
                (v3 == v1 || v3 == v2) && continue
                y[v1, v3] > y[v1, v2] || continue
                for v4 in 1:n
                    (v4 == v1 || v4 == v2 || v4 == v3) && continue
                    y[v1, v3] > y[v1, v4] || continue
                    if y[v3, v2] > y[v3, v4] && y[v1, v2] <= y[v1, v4]
                        total += 1.0
                    end
                end
            end
        end
    end

    return total
end

"""
    RankNodeICov(x) <: AbstractERGMTerm

Attractiveness/popularity covariate, `rank.nodeicov` in ergm.rank: for
each ego and each ordered alter pair (j, k) with j ranked over k, add
`x[j] − x[k]`. A positive coefficient means high-covariate actors tend to
be ranked higher.

# Fields
- `x::Vector{Float64}`: Actor-level covariate
- `label::String`: Name for output (default "x")
"""
struct RankNodeICov <: AbstractERGMTerm
    x::Vector{Float64}
    label::String

    RankNodeICov(x::AbstractVector{<:Real}; label::String="x") =
        new(Float64.(x), label)
end

name(t::RankNodeICov) = "rank.nodeicov.$(t.label)"

function compute(t::RankNodeICov, rnet::RankNetwork)
    n = rnet.n
    length(t.x) == n ||
        throw(ArgumentError("covariate length $(length(t.x)) ≠ number of actors $n"))
    y = rnet.ranks
    total = 0.0
    for v1 in 1:n, v2 in 1:n
        v2 == v1 && continue
        for v3 in 1:n
            (v3 == v1 || v3 == v2) && continue
            if y[v1, v2] > y[v1, v3]
                total += t.x[v2] - t.x[v3]
            end
        end
    end
    return total
end

"""
    RankInconsistency(ref) <: AbstractERGMTerm

Inconsistency, `rank.inconsistency` in ergm.rank: the number of ego–alter
pair comparisons on which the network disagrees with a reference ranking
(`(y_ij > y_ik) ≠ (ref_ij > ref_ik)`).

# Fields
- `ref::Matrix{Int}`: Reference rank matrix (same convention)
"""
struct RankInconsistency <: AbstractERGMTerm
    ref::Matrix{Int}
end

RankInconsistency(ref_net::RankNetwork) = RankInconsistency(rank_matrix(ref_net))

name(::RankInconsistency) = "rank.inconsistency"

function compute(t::RankInconsistency, rnet::RankNetwork)
    n = rnet.n
    size(t.ref) == (n, n) ||
        throw(ArgumentError("reference matrix size $(size(t.ref)) ≠ ($n, $n)"))
    y = rnet.ranks
    r = t.ref
    total = 0.0
    for v1 in 1:n, v2 in 1:n
        v2 == v1 && continue
        for v3 in 1:n
            (v3 == v1 || v3 == v2) && continue
            if (y[v1, v2] > y[v1, v3]) != (r[v1, v2] > r[v1, v3])
                total += 1.0
            end
        end
    end
    return total
end

"""
    RankEdgeCov(cov) <: AbstractERGMTerm

Dyadic covariate, `rank.edgecov` in ergm.rank: for each ego and ordered
alter pair (j, k) with j ranked over k, add `cov[i, j] − cov[i, k]`.

# Fields
- `cov::Matrix{Float64}`: Dyadic covariate matrix
- `label::String`: Name for output
"""
struct RankEdgeCov <: AbstractERGMTerm
    cov::Matrix{Float64}
    label::String

    RankEdgeCov(cov::AbstractMatrix{<:Real}; label::String="cov") =
        new(Float64.(cov), label)
end

name(t::RankEdgeCov) = "rank.edgecov.$(t.label)"

function compute(t::RankEdgeCov, rnet::RankNetwork)
    n = rnet.n
    size(t.cov) == (n, n) ||
        throw(ArgumentError("covariate matrix size $(size(t.cov)) ≠ ($n, $n)"))
    y = rnet.ranks
    total = 0.0
    for v1 in 1:n, v2 in 1:n
        v2 == v1 && continue
        for v3 in 1:n
            (v3 == v1 || v3 == v2) && continue
            if y[v1, v2] > y[v1, v3]
                total += t.cov[v1, v2] - t.cov[v1, v3]
            end
        end
    end
    return total
end

# All-terms statistic vector
compute_all_stats(terms, rnet::RankNetwork) =
    [compute(term, rnet) for term in terms]

# Change in the statistic vector from swapping ego's ranks of j and k
# (generic brute-force evaluation; swaps in place and restores)
function _swap_delta(terms, rnet::RankNetwork, ego::Int, j::Int, k::Int)
    before = compute_all_stats(terms, rnet)
    swap_ranks!(rnet, ego, j, k)
    after = compute_all_stats(terms, rnet)
    swap_ranks!(rnet, ego, j, k)
    return after .- before
end

# =============================================================================
# Model and estimation
# =============================================================================

"""
    RankERGMModel

Rank-order ERGM specification: terms plus the observed `RankNetwork`
under the `CompleteOrderReference`.
"""
struct RankERGMModel
    terms::Vector{AbstractERGMTerm}
    network::RankNetwork
    reference::CompleteOrderReference
end

"""
    RankERGMResult

Results from fitting a rank-order ERGM. `loglik` is the maximized
swap-based pseudo-log-likelihood (see [`ergm_rank`](@ref)).

`se_type` records how `std_errors`/`vcov` were ACTUALLY obtained:

- `:hessian` — the inverse negative Hessian of the swap pseudo-likelihood. The
  swap comparisons overlap, so these are expected anticonservative.
- `:bootstrap` — the parametric bootstrap of `fit_ergm_rank(...; se=:bootstrap)`
  (simulate rank networks at θ̂ with the AlterSwap sampler, refit, empirical
  covariance).

It is what `Networks.se_method(fit)` reports, and what `show` reads before
deciding whether the anticonservatism caveat is still true of this fit.
"""
struct RankERGMResult
    model::RankERGMModel
    coefficients::Vector{Float64}
    std_errors::Vector{Float64}
    vcov::Matrix{Float64}
    loglik::Float64
    converged::Bool
    se_type::Symbol
end

# Backwards-compatible constructor: a result built without an `se_type` reports
# the inverse-Hessian standard errors it in fact had.
RankERGMResult(model, coefficients, std_errors, vcov, loglik, converged) =
    RankERGMResult(model, coefficients, std_errors, vcov, loglik, converged,
                   :hessian)

# Two-sided normal p-values via the complementary CDF (the naive
# 2(1 − cdf) form underflows to exactly 0 beyond |z| ≈ 8.3); NaN standard
# errors give NaN p-values, which the shared printer renders as "NaN"
_z_pvalues(z::AbstractVector{Float64}) = 2 .* ccdf.(Normal(), abs.(z))

function Base.show(io::IO, result::RankERGMResult)
    println(io, "Rank-Order ERGM Results")
    println(io, "=======================")
    println(io, "Reference: CompleteOrder")
    println(io, "Pseudo-log-likelihood: $(round(result.loglik, digits=4))")
    println(io, "Converged: $(result.converged)")
    println(io, "Std. errors: ", result.se_type === :bootstrap ?
                "parametric bootstrap" : "inverse pseudo-Hessian")
    println(io)
    z = result.coefficients ./ result.std_errors
    print_coeftable(io, [name(term) for term in result.model.terms],
                    result.coefficients, result.std_errors, _z_pvalues(z);
                    z_values=z)

    # Honest-uncertainty caveat (mirroring ERGM.jl's show), and the prose twin
    # of what `approximations(result)` reports. The POINT ESTIMATE is a swap
    # pseudo-likelihood estimate either way — no rank fit is exact, which is why
    # `is_exact(::RankERGMResult)` is unconditionally false — but the standard
    # errors are only anticonservative when they are the inverse pseudo-Hessian.
    # A bootstrap covariance does not treat the overlapping comparisons as
    # independent, so saying it is anticonservative would be a lie.
    println(io)
    if result.se_type === :bootstrap
        println(io, "Note: this model was fit by swap-based maximum pseudolikelihood, so the")
        println(io, "point estimates are those of a pseudo-likelihood (not the MCMC MLE). The")
        println(io, "standard errors are a parametric bootstrap: they do not treat the")
        println(io, "overlapping swap comparisons as independent.")
    else
        println(io, "Warning: this model was fit by swap-based maximum pseudolikelihood.")
        println(io, "The pairwise-swap comparisons overlap, so the standard errors (inverse")
        println(io, "pseudo-Hessian) ignore that dependence and are expected to be")
        println(io, "anticonservative; the p-values should be treated as a rough guide.")
        println(io, "Refit with `se=:bootstrap` for a parametric-bootstrap covariance.")
    end
end

# ============================================================================
# The shared result-metadata protocol (Networks.jl `src/results.jl`)
# ============================================================================
#
# `fit_metadata(fit)` collects these accessors, so the caveats that
# `fit_ergm_rank`'s docstring spells out in prose are what a machine reads too.

estimand(::RankERGMResult) = :rank_ergm

"""
    objective(::RankERGMResult) -> Symbol

`:pseudolikelihood` — the **swap** pseudo-likelihood: the product, over every
(ego, unordered alter pair), of the conditional probability of the observed
relative order given the rest of the rankings. The AlterSwap move takes the role
of the edge toggle in dyadwise MPLE.
"""
objective(::RankERGMResult) = :pseudolikelihood

"""
    is_exact(::RankERGMResult) -> Bool

Always `false`. The swap comparisons overlap — each ranking enters `n − 2` of
the pairwise conditionals — so their product is not the likelihood, and **no
consistency result is claimed** for this estimator (the earlier "consistent
approximation" claim has been withdrawn; see [`fit_ergm_rank`](@ref)).
"""
is_exact(::RankERGMResult) = false

"""
    se_method(result::RankERGMResult) -> Symbol

What the reported standard errors ACTUALLY are: `:hessian` (the inverse negative
Hessian of the swap pseudo-likelihood) or `:bootstrap` (the parametric bootstrap
of `fit_ergm_rank(...; se=:bootstrap)`). Read straight off the fit.
"""
se_method(result::RankERGMResult) = result.se_type

# A `RankNetwork` carries complete orderings, not a dyad mask: there is no
# unobserved-tie concept for the estimator to treat one way or the other.
missing_method(::RankERGMResult) = :none

function approximations(result::RankERGMResult)
    # The point-estimate caveat holds for every rank fit, however the standard
    # errors were computed: `se=:bootstrap` replaces the covariance, not θ̂.
    out = String[
        "swap pseudo-likelihood: the (ego, alter-pair) swap conditionals are " *
        "multiplied as if independent, but they overlap (each ranking enters " *
        "n − 2 comparisons), so this is not the likelihood and no consistency " *
        "result is claimed for the estimator",
    ]
    if result.se_type === :bootstrap
        push!(out, "standard errors are a parametric bootstrap of the swap MPLE " *
                   "(simulate rank networks at θ̂ with the AlterSwap sampler, refit, " *
                   "empirical covariance): they do NOT treat the overlapping swap " *
                   "comparisons as independent, but they are Monte-Carlo estimates " *
                   "and assume the fitted model generated the data")
    else
        push!(out, "inverse-Hessian standard errors of the naive swap pseudo-likelihood: " *
                   "they ignore the dependence between the overlapping comparisons and are " *
                   "expected anticonservative (too small). Treat them as a rough guide, not " *
                   "calibrated inference — or refit with `se=:bootstrap`")
    end
    return out
end

# StatsAPI interface: methods on the shared statistics generics, so results
# interoperate with StatsBase/GLM-style tooling (`coef(fit)`, `vcov(fit)`, ...)

# Number of pseudo-likelihood contributions: one per (ego, unordered alter
# pair) conditional
_n_comparisons(rnet::RankNetwork) = rnet.n * (rnet.n - 1) * (rnet.n - 2) ÷ 2

StatsAPI.coef(result::RankERGMResult) = result.coefficients
StatsAPI.stderror(result::RankERGMResult) = result.std_errors
StatsAPI.vcov(result::RankERGMResult) = result.vcov
StatsAPI.loglikelihood(result::RankERGMResult) = result.loglik
StatsAPI.nobs(result::RankERGMResult) = _n_comparisons(result.model.network)
StatsAPI.dof(result::RankERGMResult) = length(result.coefficients)

"""
    fit_ergm_rank(rnet::RankNetwork, terms; kwargs...) -> RankERGMResult

Fit a rank-order ERGM by **swap-based maximum pseudo-likelihood**: for
each ego `i` and each unordered alter pair {j, k}, the conditional
probability of the observed relative order of j and k given the rest of
the rankings is logistic in `θ'[g(y) − g(y with j,k swapped)]`. The
product of these conditionals is maximized by Newton-Raphson with
step-halving.

This is the natural rank analogue of dyadwise MPLE (the AlterSwap move
takes the role of the edge toggle). It is fast, and it is *not* the MCMC
MLE that `ergm.rank` computes: it maximizes a pseudo-likelihood formed by
multiplying pairwise-swap conditionals that are not independent, so the
two estimators generally disagree.

!!! warning "What is and is not claimed"
    **No consistency result is established here.** Earlier versions of this
    docstring called swap-MPLE a "consistent approximation" to the MCMC MLE;
    that claim was unqualified by any asymptotic regime or assumptions and
    has been withdrawn. For dyad-independent-analogue models (where the
    swap conditionals really are the model's conditionals) the
    pseudo-likelihood coincides with the likelihood; outside that case the
    estimator's large-sample behaviour in this setting has not been
    characterized, and MPLE for dependent ERGMs is known to be biased in
    finite samples.

    **The default standard errors are the inverse observed pseudo-Hessian** —
    the curvature of the pseudo-log-likelihood, which treats the overlapping
    swap comparisons as independent. They therefore ignore the dependence
    between comparisons and are expected to be **anticonservative** (too
    small, giving over-narrow intervals and anti-conservative tests) under
    dependence. Treat them as a rough guide, not calibrated inference — or
    pass `se=:bootstrap` (below), which does not make that assumption.

The pseudo-log-likelihood is maximized with the shared `ERGM.newton_fit`
Newton–Raphson-with-step-halving optimizer.

[`ergm_rank`](@ref) is the R-faithful alias (matching the `ergm.rank`
package); [`fit_rank_ergm`](@ref) is a legacy alias.

# Standard errors

- `se=:hessian` (default) — the inverse negative pseudo-Hessian; see the warning
  above.
- `se=:bootstrap` — parametric bootstrap: simulate `n_boot` rank networks from
  the fitted model at θ̂ with [`simulate_rank_ergm`](@ref) (AlterSwap
  Metropolis), refit the swap MPLE on each, and report the empirical covariance
  of the refits. The point estimates are unchanged; only the covariance is
  replaced. This is the same option, with the same keywords and the same
  semantics, as `ERGM.mple`'s, and it runs on the ONE shared
  `Networks.bootstrap_cov` loop.

# Keyword Arguments
- `maxiter::Int=100`, `tol::Float64=1e-8` (passed to `newton_fit`)
- `se::Symbol=:hessian`: `:hessian` or `:bootstrap` (above)
- `n_boot::Int=100`: number of bootstrap replicates (`se=:bootstrap` only)
- `boot_burnin::Int=500`, `boot_interval::Int=50`: MCMC controls for the
  bootstrap simulations (the [`simulate_rank_ergm`](@ref) defaults)
- `rng::AbstractRNG=Random.default_rng()`: source of the bootstrap randomness —
  a fixed `rng` reproduces the standard errors exactly
"""
function fit_ergm_rank(rnet::RankNetwork, terms::Vector{<:AbstractERGMTerm};
                       maxiter::Int=100, tol::Float64=1e-8,
                       se::Symbol=:hessian,
                       n_boot::Int=100,
                       boot_burnin::Int=500,
                       boot_interval::Int=50,
                       rng::Random.AbstractRNG=Random.default_rng())
    se in (:hessian, :bootstrap) ||
        throw(ArgumentError("se must be :hessian or :bootstrap, got :$se"))
    is_valid_ranking(rnet) ||
        throw(ArgumentError("network is not a valid complete ranking: " *
                            something(_ranking_violation(rnet.ranks), "")))

    model = RankERGMModel(collect(AbstractERGMTerm, terms), copy(rnet),
                          CompleteOrderReference())

    fit = _rank_mple_fit(model.terms, rnet; maxiter=maxiter, tol=tol)

    vcov, std_errors = fit.vcov, fit.se
    if se === :bootstrap
        vcov, std_errors = _rank_bootstrap_cov(model, fit.θ; n_boot=n_boot,
                                               boot_burnin=boot_burnin,
                                               boot_interval=boot_interval,
                                               maxiter=maxiter, tol=tol, rng=rng)
    end

    return RankERGMResult(model, fit.θ, std_errors, vcov, fit.loglik,
                          fit.converged, se)
end

# Core swap MPLE: the design vectors d = g(y) − g(y with j,k swapped) over every
# (ego, unordered alter pair), maximized by the shared Newton optimizer. Shared
# by `fit_ergm_rank` and by the parametric bootstrap's refits.
function _rank_mple_fit(terms::Vector{AbstractERGMTerm}, rnet::RankNetwork;
                        maxiter::Int=100, tol::Float64=1e-8)
    p = length(terms)
    D = _rank_design(terms, copy(rnet))

    # The swap pseudo-likelihood IS a logistic likelihood on the D rows with the
    # response identically TRUE — the observed order is always the "success" —
    # so the derivatives come from the shared `ERGM.logistic_derivatives` (review
    # finding 15): gemv/gemm over the whole design (η = Dβ, ∇ = D'(1−p),
    # −H = D'WD), not a per-comparison `d * d'` outer product allocating a p×p
    # matrix on every one of the size(D, 1) rows of every Newton evaluation.
    # Never paste the loop back in; ERGMMulti and TERGM run on the same one.
    derivatives = logistic_derivatives(D, trues(size(D, 1)))
    return newton_fit(derivatives, zeros(p); maxiter=maxiter, tol=tol)
end

# The swap design: for each (ego, unordered alter pair {j,k}) the difference
# d = g(y_observed) − g(y_swapped), as ONE dense (comparisons × p) matrix — the
# derivatives are BLAS over the whole thing, so a vector-of-vectors would just be
# a scatter to copy out of.
function _rank_design(terms::Vector{AbstractERGMTerm}, work::RankNetwork)
    n = work.n
    p = length(terms)
    rows = Vector{Float64}[]
    for ego in 1:n
        alters = [v for v in 1:n if v != ego]
        for a in 1:length(alters), b in (a+1):length(alters)
            push!(rows, -_swap_delta(terms, work, ego, alters[a], alters[b]))
        end
    end
    D = Matrix{Float64}(undef, length(rows), p)
    for (r, d) in enumerate(rows)
        @inbounds for k in 1:p
            D[r, k] = d[k]
        end
    end
    return D
end

# Parametric-bootstrap covariance of the swap MPLE: simulate `n_boot` rank
# networks at θ̂ with the AlterSwap sampler, refit the swap MPLE on each, take
# the empirical covariance. The loop is the shared `Networks.bootstrap_cov`; this
# supplies only the two callbacks that are ERGMRank's.
function _rank_bootstrap_cov(model::RankERGMModel, θ̂::Vector{Float64};
                             n_boot::Int, boot_burnin::Int, boot_interval::Int,
                             maxiter::Int, tol::Float64,
                             rng::Random.AbstractRNG)
    simulate(rng, B) = simulate_rank_ergm(model.network, model.terms, θ̂;
                                          n_sim=B, burnin=boot_burnin,
                                          interval=boot_interval, rng=rng)

    refit(sim::RankNetwork) =
        _rank_mple_fit(model.terms, sim; maxiter=maxiter, tol=tol).θ

    boot = bootstrap_cov(refit, simulate, θ̂; n_boot=n_boot, rng=rng)
    return boot.vcov, boot.se
end

"""
    ergm_rank(rnet::RankNetwork, terms; kwargs...) -> RankERGMResult

R-faithful alias for [`fit_ergm_rank`](@ref) (the same function), matching
the R `ergm.rank` package name.
"""
const ergm_rank = fit_ergm_rank

"""
    fit_rank_ergm(rnet::RankNetwork, terms; kwargs...) -> RankERGMResult

Alias for [`fit_ergm_rank`](@ref), kept for backward compatibility.
"""
const fit_rank_ergm = fit_ergm_rank

# =============================================================================
# Simulation
# =============================================================================

"""
    simulate_rank_ergm(rnet, terms, θ; n_sim=1, burnin=500, interval=50,
                       rng=Random.default_rng()) -> Vector{RankNetwork}

Simulate rank networks from a rank-order ERGM by Metropolis sampling with
the **AlterSwap** proposal (as in ergm.rank): pick a random ego and two
random alters, propose swapping their ranks, and accept with probability
`min(1, exp(θ'Δg))`. Every state visited is a valid complete ranking, and
the chain targets `P(y) ∝ exp(θ'g(y))` on the complete-ordering space
(the proposal is symmetric, so the CompleteOrder reference cancels).
"""
function simulate_rank_ergm(rnet::RankNetwork, terms::Vector{<:AbstractERGMTerm},
                            θ::Vector{Float64};
                            n_sim::Int=1,
                            burnin::Int=500,
                            interval::Int=50,
                            rng::Random.AbstractRNG=Random.default_rng())
    is_valid_ranking(rnet) ||
        throw(ArgumentError("starting network is not a valid complete ranking"))
    length(θ) == length(terms) ||
        throw(ArgumentError("θ must have one coefficient per term"))

    current = copy(rnet)
    n = current.n
    n >= 3 || throw(ArgumentError("need at least 3 actors to swap alters"))
    draws = RankNetwork[]

    for step in 1:(burnin + n_sim * interval)
        ego = rand(rng, 1:n)
        j = rand(rng, 1:n)
        while j == ego
            j = rand(rng, 1:n)
        end
        k = rand(rng, 1:n)
        while k == ego || k == j
            k = rand(rng, 1:n)
        end

        delta = _swap_delta(terms, current, ego, j, k)
        if log(rand(rng)) < dot(θ, delta)
            swap_ranks!(current, ego, j, k)
        end

        if step > burnin && (step - burnin) % interval == 0
            push!(draws, copy(current))
        end
    end

    return draws
end

"""
    simulate_rank_ergm(result::RankERGMResult; kwargs...) -> Vector{RankNetwork}

Simulate from a fitted rank-order ERGM.
"""
function simulate_rank_ergm(result::RankERGMResult; kwargs...)
    return simulate_rank_ergm(result.model.network, result.model.terms,
                              result.coefficients; kwargs...)
end

# =============================================================================
# Goodness of fit
# =============================================================================

"""
    gof(result::RankERGMResult; n_sim=100, burnin=500, interval=50,
        rng=Random.default_rng()) -> GOFResult

Goodness-of-fit assessment of a fitted rank-order ERGM: rank networks are
simulated from the fitted model with [`simulate_rank_ergm`](@ref) (AlterSwap
Metropolis sampling) and the observed model statistics are compared with
their simulated distributions.

This is a method of the shared `Networks.gof` generic; it returns the shared
`Networks.GOFResult` (observed value, simulation envelope, and two-sided
Monte-Carlo p-value per statistic).

# Keyword Arguments
- `n_sim::Int=100`: Number of simulated rank networks
- `burnin`, `interval`, `rng`: passed to [`simulate_rank_ergm`](@ref)
"""
function gof(result::RankERGMResult; n_sim::Int=100, burnin::Int=500,
             interval::Int=50, rng::Random.AbstractRNG=Random.default_rng())
    rnet = result.model.network
    terms = result.model.terms
    sims = simulate_rank_ergm(result; n_sim=n_sim, burnin=burnin,
                              interval=interval, rng=rng)

    obs_stats = [compute(term, rnet) for term in terms]
    sim_stats = [compute(term, s) for s in sims, term in terms]
    stats = GOFStatistic("model statistics", [name(term) for term in terms],
                         obs_stats, sim_stats)

    return GOFResult([stats]; model="Rank-Order ERGM")
end

end # module
