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
# Shared presentation infrastructure (Network.jl): the ONE `gof` generic all
# model packages extend, plus the common coefficient-table printer and
# GOF containers
import Network: gof, print_coeftable, GOFStatistic, GOFResult
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

# Goodness of fit (method of the shared Network.jl `gof` generic)
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
swap-based pseudo-log-likelihood (see [`ergm_rank`](@ref)); `vcov` is the
inverse negative Hessian of the pseudo-log-likelihood at the optimum.
"""
struct RankERGMResult
    model::RankERGMModel
    coefficients::Vector{Float64}
    std_errors::Vector{Float64}
    vcov::Matrix{Float64}
    loglik::Float64
    converged::Bool
end

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
    println(io)
    z = result.coefficients ./ result.std_errors
    print_coeftable(io, [name(term) for term in result.model.terms],
                    result.coefficients, result.std_errors, _z_pvalues(z);
                    z_values=z)
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
takes the role of the edge toggle). It is a consistent, fast approximation
to the MCMC MLE of `ergm.rank`; like all pseudo-likelihoods it understates
uncertainty for strongly dependent models.

The pseudo-log-likelihood is maximized with the shared `ERGM.newton_fit`
Newton–Raphson-with-step-halving optimizer.

[`ergm_rank`](@ref) is the R-faithful alias (matching the `ergm.rank`
package); [`fit_rank_ergm`](@ref) is a legacy alias.

# Keyword Arguments
- `maxiter::Int=100`, `tol::Float64=1e-8` (passed to `newton_fit`)
"""
function fit_ergm_rank(rnet::RankNetwork, terms::Vector{<:AbstractERGMTerm};
                       maxiter::Int=100, tol::Float64=1e-8)
    is_valid_ranking(rnet) ||
        throw(ArgumentError("network is not a valid complete ranking: " *
                            something(_ranking_violation(rnet.ranks), "")))

    model = RankERGMModel(collect(AbstractERGMTerm, terms), copy(rnet),
                          CompleteOrderReference())
    n = rnet.n
    p = length(terms)
    work = copy(rnet)

    # Design vectors: for each (ego, {j,k}) the difference
    # d = g(y_observed) − g(y_swapped)
    D = Vector{Vector{Float64}}()
    for ego in 1:n
        alters = [v for v in 1:n if v != ego]
        for a in 1:length(alters), b in (a+1):length(alters)
            push!(D, -_swap_delta(model.terms, work, ego, alters[a], alters[b]))
        end
    end

    function derivatives(β)
        ll = 0.0
        grad = zeros(p)
        hess = zeros(p, p)
        for d in D
            η = dot(β, d)
            pr = 1.0 / (1.0 + exp(-η))       # P(observed order | rest)
            # log pr computed stably
            ll += η < 0 ? η - log1p(exp(η)) : -log1p(exp(-η))
            grad .+= d .* (1.0 - pr)
            hess .-= (pr * (1.0 - pr)) .* (d * d')
        end
        return ll, grad, hess
    end

    # Maximize with the shared Newton-with-step-halving optimizer
    fit = newton_fit(derivatives, zeros(p); maxiter=maxiter, tol=tol)

    return RankERGMResult(model, fit.θ, fit.se, fit.vcov, fit.loglik,
                          fit.converged)
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

This is a method of the shared `Network.gof` generic; it returns the shared
`Network.GOFResult` (observed value, simulation envelope, and two-sided
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
