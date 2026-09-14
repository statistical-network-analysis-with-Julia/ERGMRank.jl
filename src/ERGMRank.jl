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

# The statistic protocol (`name`/`compute` are Networks.jl generics that
# ERGM.jl re-exports) and the ONE Metropolis toggle kernel every ERGM-family
# sampler runs on (panel 2026-09, item 28): `simulate_rank_ergm` supplies the
# AlterSwap move, its change statistics and its state mutation as callables.
# `mcmc_convergence` (the t-ratio/Hotelling tests) and the `MCMLEConvergence`
# report type are ERGM's `public` MCMLE machinery, which the rank MCMLE
# (`method=:mcmle`) runs on rather than re-implementing (item 24).
import ERGM: name, compute, mh_toggle!, mcmc_convergence, MCMLEConvergence
# Shared presentation infrastructure (Networks.jl): the ONE `gof` generic all
# model packages extend, the GOF containers, the coefficient table that
# `coeftable(fit)` returns and `show` prints, the floored z→p helper and the
# ONE `se=` validator (panel 2026-09, items 13, 15 and 28)
import Networks: gof, GOFStatistic, GOFResult, CoefficientTable, coeftable,
                 z_pvalues, check_se

# The ONE shared bootstrap loop (Networks.jl `src/bootstrap.jl`): simulate,
# refit, empirical covariance. `se=:bootstrap` supplies the two callbacks; the
# loop, the threading and the rng discipline are not reimplemented here.
import Networks: bootstrap_cov

# The ONE Newton optimizer and logistic-likelihood kernel of the ecosystem
# (Networks.jl `src/newton.jl`, `public`; `ERGM.newton_fit` is the same
# binding). The swap MPLE is a logistic regression on the swap-difference rows
# with the response identically `true`, so it has no loop of its own.
import Networks: newton_fit, logistic_derivatives

# The shared result-metadata protocol (Networks.jl `src/results.jl`): the
# generic accessors that say what a fit actually did. Imported by name because
# ERGMRank adds methods for `RankERGMResult`; `fit_metadata(fit)` collects them.
import Networks: estimand, objective, is_exact, se_method, missing_method,
                 approximations, fit_metadata
# The conversion contract (Networks.jl `src/conversion.jl`, `src/missing.jl`)
# for the `Network` → `RankNetwork` adapter: the missing-dyad guard and the
# two traits it is declared through, and the report a lossy conversion
# returns on request. `Network` and its queries are used qualified or by name;
# ERGMRank does not re-export Networks.
import Networks: supports_missing, missing_policies, require_observed,
                 ConversionReport, record_drop!
using Networks: Network, nv, has_edge, is_directed, get_edge_attribute,
                list_edge_attributes, list_vertex_attributes,
                list_network_attributes
import StatsAPI
import StatsAPI: coef, stderror, vcov, loglikelihood, nobs, dof, aic, bic, confint

# Core types
export RankNetwork, RankERGMModel, RankERGMResult
export CompleteOrderReference

# Rank access and manipulation
export get_rank, set_rank!, swap_ranks!, is_valid_ranking
export as_rank_network, rank_matrix

# The teaching dataset the estimator claims rest on (R ergm.rank's newcomb)
export newcomb_week1

# The statistic protocol: `compute(term, rnet)` and `name(term)` are the ONE
# `Networks.compute`/`Networks.name` (reached through ERGM.jl, which
# re-exports them; `ERGMRank.compute === Networks.compute` is pinned), so
# `using ERGMRank` alone evaluates a rank statistic
export compute, name

# The shared result-metadata protocol (Networks.jl), re-exported so
# `approximations(fit)` works with just `using ERGMRank`; the same bindings
# every fitting package in the ecosystem exports
export fit_metadata, approximations, estimand, objective, is_exact, se_method,
       missing_method

# Terms (matching R ergm.rank) and their swap change statistic
export RankDeference, RankNonconformity, RankNodeICov
export RankInconsistency, RankEdgeCov
export swap_change

# Estimation and simulation
export fit_ergm_rank, ergm_rank, fit_rank_ergm
export simulate_rank_ergm

# Goodness of fit (method of the shared Networks.jl `gof` generic)
export gof

# The ecosystem's StatsAPI surface (re-exported so `coef(fit)` etc. work with
# just `using ERGMRank`; `coeftable` is the ONE `StatsAPI.coeftable` binding
# that Networks.jl re-exports)
export coef, stderror, vcov, confint, loglikelihood, nobs, dof, aic, bic, coeftable

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

`RankNetwork(ranks::Matrix{Int}; validate=true)` validates the invariant
(an `ArgumentError` naming the ego at fault otherwise); [`as_rank_network`](@ref)
is the checked conversion from any integer matrix or from a directed
`Networks.Network` with a rank edge attribute; [`RankNetwork(n)`](@ref) builds
the index-order ranking. A `RankNetwork` does not wrap a `Network`: it holds
complete orderings, not dyads, so there is no missing-dyad mask (see
`as_rank_network` for what an unobserved rank means here).

# Example
```julia
using ERGMRank
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
rnet = RankNetwork(m)
rnet.n                        # 4
get_rank(rnet, 1, 2)          # 3 — ego 1 ranks actor 2 highest
is_valid_ranking(rnet)        # true
rnet                          # RankNetwork(4 actors)
```
"""
struct RankNetwork
    n::Int
    ranks::Matrix{Int}

    function RankNetwork(ranks::Matrix{Int}; validate::Bool=true)
        n = size(ranks, 1)
        size(ranks, 2) == n || throw(ArgumentError(
            "RankNetwork: the rank matrix must be square (got $(size(ranks))): row i " *
            "holds ego i's ranks of every other actor"))
        if validate
            msg = _ranking_violation(ranks)
            isnothing(msg) || throw(ArgumentError(
                "RankNetwork: not a valid complete ranking — $msg (greater = higher " *
                "standing; see as_rank_network)"))
        end
        new(n, copy(ranks))
    end
end

"""
    RankNetwork(n::Int)

Create a rank network with `n` actors in which every ego ranks the alters
in index order: ego `i` gives the alter with the smallest index rank 1 and
the alter with the largest index rank `n − 1` (greater = higher standing).
A convenient starting state for [`simulate_rank_ergm`](@ref); randomize it
with [`swap_ranks!`](@ref) moves.

# Example
```julia
using ERGMRank
rnet = RankNetwork(4)
rank_matrix(rnet)             # [0 1 2 3; 1 0 2 3; 1 2 0 3; 1 2 3 0]
get_rank(rnet, 2, 4)          # 3: ego 2 ranks actor 4 highest
is_valid_ranking(rnet)        # true
```
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
structural invariant of complete rank-order data. Only [`set_rank!`](@ref)
can break it (the constructors validate, [`swap_ranks!`](@ref) preserves
it); `fit_ergm_rank` and `simulate_rank_ergm` refuse a network on which this
is `false`.

# Example
```julia
using ERGMRank
rnet = RankNetwork(4)
is_valid_ranking(rnet)                 # true
set_rank!(rnet, 1, 2, 3)               # ego 1 now ranks actors 2 and 4 both 3
is_valid_ranking(rnet)                 # false
set_rank!(rnet, 1, 2, 1)               # restored
is_valid_ranking(rnet)                 # true
```
"""
is_valid_ranking(rnet::RankNetwork) = isnothing(_ranking_violation(rnet.ranks))

Base.copy(rnet::RankNetwork) = RankNetwork(copy(rnet.ranks); validate=false)

# Display. Two-arg `show` is the ONE-LINE form Base uses inside containers (a
# `Vector{RankNetwork}` of draws from `simulate_rank_ergm`) and for
# `print`/`string`; the block a user expects for a top-level value is the
# `MIME"text/plain"` method the REPL calls — Networks.jl's convention for
# `Network` (one multi-line 2-arg method garbles every container).
function Base.show(io::IO, rnet::RankNetwork)
    print(io, "RankNetwork(", rnet.n, " actors)")
end

function Base.show(io::IO, ::MIME"text/plain", rnet::RankNetwork)
    n = rnet.n
    println(io, "RankNetwork: complete rankings by ", n, " actors")
    println(io, "  Actors: ", n)
    println(io, "  Alters ranked per ego: ", n - 1, " (ranks 1:", n - 1,
            ", greater = higher standing)")
    println(io, "  Swap comparisons: ", n < 3 ? 0 : _n_comparisons(rnet),
            " (ego × unordered alter pair)")
    println(io, "  Valid complete ranking: ", is_valid_ranking(rnet))
    if n <= 12
        println(io, "  Ranks (row = ego, column = alter, diagonal 0):")
        width = max(1, ndigits(max(n - 1, 1)))
        for i in 1:n
            print(io, "    ")
            for j in 1:n
                print(io, lpad(rnet.ranks[i, j], width), j < n ? " " : "")
            end
            i < n && println(io)
        end
    else
        print(io, "  Ranks: ", n, "×", n, " matrix — `rank_matrix(rnet)`")
    end
end

"""
    get_rank(rnet::RankNetwork, i, j) -> Int

The rank ego `i` assigns alter `j` (greater = higher standing); 0 for
`i == j`.

# Example
```julia
using ERGMRank
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
rnet = as_rank_network(m)
get_rank(rnet, 1, 2)                   # 3: ego 1 ranks actor 2 highest
get_rank(rnet, 1, 4)                   # 1: ...and actor 4 lowest
get_rank(rnet, 1, 2) > get_rank(rnet, 1, 3)   # true: 1 ranks 2 over 3
get_rank(rnet, 2, 2)                   # 0: no self-rank
```
"""
get_rank(rnet::RankNetwork, i::Int, j::Int) = rnet.ranks[i, j]

"""
    set_rank!(rnet::RankNetwork, i, j, rank)

Set the rank ego `i` assigns alter `j`. This can transiently break the
per-ego permutation invariant; prefer [`swap_ranks!`](@ref), which
preserves it. Validate afterwards with [`is_valid_ranking`](@ref). A
self-rank (`i == j`) and a rank outside `1:(n-1)` are refused with an
`ArgumentError`.

# Example
```julia
using ERGMRank
rnet = RankNetwork(4)                  # ego 1's row: 0 1 2 3
set_rank!(rnet, 1, 2, 3)               # duplicates rank 3 in row 1...
is_valid_ranking(rnet)                 # false
set_rank!(rnet, 1, 4, 1)               # ...so move the old 3 to the freed 1
is_valid_ranking(rnet)                 # true
rank_matrix(rnet)[1, :]                # [0, 3, 2, 1]
```
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
complete-ordering invariant. It is the proposal of [`simulate_rank_ergm`](@ref)
and the perturbation behind every swap-MPLE comparison; [`swap_change`](@ref)
gives a term's change under it without mutating anything. `ego ∈ {j, k}` is
refused with an `ArgumentError`.

# Example
```julia
using ERGMRank
rnet = RankNetwork(4)                  # ego 1's row: 0 1 2 3
swap_ranks!(rnet, 1, 2, 4)             # ego 1 exchanges its ranks of 2 and 4
rank_matrix(rnet)[1, :]                # [0, 3, 2, 1]
is_valid_ranking(rnet)                 # true — always
swap_ranks!(rnet, 1, 2, 4)             # swapping back restores the ranking
rnet == RankNetwork(4) || rank_matrix(rnet) == rank_matrix(RankNetwork(4))   # true
```
"""
function swap_ranks!(rnet::RankNetwork, ego::Int, j::Int, k::Int)
    (ego == j || ego == k) && throw(ArgumentError("ego cannot swap its own rank"))
    rnet.ranks[ego, j], rnet.ranks[ego, k] = rnet.ranks[ego, k], rnet.ranks[ego, j]
    return rnet
end

"""
    as_rank_network(mat::AbstractMatrix) -> RankNetwork

Build a `RankNetwork` from a matrix of rank values (row `i` holds ego
`i`'s ranks of the alters, greater = higher standing). The matrix must hold
**complete** orderings: integer entries, a zero diagonal, and each row's
off-diagonal a permutation of `1:(n-1)`.

A matrix containing `missing` (or `nothing`) is refused with an
`ArgumentError`: a `RankNetwork` cannot represent an unobserved rank, and
there is no `missing=` policy to select because a rank has no face value to
condition on — unlike a binary tie, an absent rank cannot be read as "0" (see
the Networks.jl missing-data guide for how the rest of the ecosystem treats
unobserved dyads). Drop the actors whose rankings were not observed, or
supply a complete ordering.

# Example
```julia
using ERGMRank
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
rnet = as_rank_network(m)
get_rank(rnet, 1, 2)                   # 3
rank_matrix(rnet) == m                 # true
```
"""
function as_rank_network(mat::AbstractMatrix)
    any(x -> x === missing || x === nothing, mat) &&
        throw(ArgumentError(
            "as_rank_network: the rank matrix contains missing/nothing entries. " *
            "A RankNetwork holds COMPLETE orderings (every ego ranks every alter) " *
            "and cannot represent an unobserved rank; there is no `missing=` " *
            "policy to pass because a rank has no face value to condition on " *
            "(see the Networks.jl missing-data guide, docs/src/guide/missing_data.md). " *
            "Drop the actors whose rankings were not observed, or supply a " *
            "complete rank matrix."))
    ranks = try
        Matrix{Int}(mat)
    catch err
        err isa Union{InexactError, MethodError} || rethrow()
        throw(ArgumentError(
            "as_rank_network: rank values must be integers (each row a " *
            "permutation of 1:(n-1), greater = higher standing); got a matrix " *
            "with element type $(eltype(mat))"))
    end
    return RankNetwork(ranks)
end

"""
    as_rank_network(net::Network; attr=:rank, missing=:error, report=false)
        -> RankNetwork  (or (RankNetwork, ConversionReport) with report=true)

Build a `RankNetwork` from a **directed** `Networks.Network` whose edge
attribute `attr` holds the rank ego `i` assigns alter `j` on the arc `i → j`
(greater = higher standing) — the Julia counterpart of R's
`as.matrix(nw, attrname = "rank")` on `ergm.rank`'s `newcomb` networks. The
conversion honours the ecosystem conversion contract (Networks.jl
`docs/src/guide/conversion_invariants.md`): what a `RankNetwork` can hold is
preserved, what it cannot is rejected or reported.

- **Preserved**: the actor set (vertex `i` is ego/alter `i`) and the ranks.
- **Rejected** (an `ArgumentError` that says why):
  - an undirected network — a ranking is ego-specific, so `i`'s rank of `j`
    and `j`'s rank of `i` must be two different arcs;
  - a two-mode network — every ego must rank every alter;
  - a **masked (unobserved) dyad** — `require_observed(net, missing;
    context="as_rank_network")`; `missing_policies(as_rank_network) ==
    (:error,)`: there is **no `:face` policy**, because a `RankNetwork` holds
    complete orderings and a masked dyad has no rank to read at face value
    (unlike a binary tie, an absent rank cannot be read as "0"). Drop the
    actors whose rankings were not observed, or `clear_missing_dyads!` once
    every dyad really has been observed;
  - a dyad `i → j` with no arc or no `attr` value, or a non-integer value —
    the message names the ego, the alter and the attribute;
  - an ego whose ranks are not a permutation of `1:(n-1)` (the
    `RankNetwork` invariant) — the message names the ego and the attribute.
- **Dropped and reported** (`report=true` returns `(rnet, ConversionReport)`
  with one `record_drop!` entry each): every *other* edge attribute, every
  vertex attribute and every network attribute — a `RankNetwork` carries
  none of them; and self-loop arcs, which have no rank (the diagonal is 0).
  `is_lossless(report)` is `true` exactly when `attr` was the network's only
  attribute and it had no loops.

`supports_missing(as_rank_network)` is `true`: the routine has considered
missingness, and its answer is to refuse.

# Example
```julia
using ERGMRank, Networks
net = network(3)                       # directed
ranks = [0 2 1;
         1 0 2;
         2 1 0]
for i in 1:3, j in 1:3
    i == j && continue
    add_edge!(net, i, j)
    set_edge_attribute!(net, :rank, i, j, ranks[i, j])
end
rnet = as_rank_network(net)            # attr=:rank
rank_matrix(rnet) == ranks             # true

set_vertex_attribute!(net, :sex, ["m", "f", "m"])
rnet, rep = as_rank_network(net; report=true)
dropped_fields(rep)                    # [:sex]: a RankNetwork has no vertex attributes
is_lossless(rep)                       # false

set_missing_dyad!(net, 1, 2)           # ego 1's rank of 2 is unobserved
try
    as_rank_network(net)
catch err
    err isa ArgumentError              # true: no face value to condition on
end
```
"""
function as_rank_network(net::Network{T}; attr::Symbol=:rank, missing::Symbol=:error,
                         report::Bool=false) where {T}
    is_directed(net) || throw(ArgumentError(
        "as_rank_network: the network is undirected, but a ranking is ego-specific " *
        "— ego i's rank of j and ego j's rank of i are two different arcs, i → j " *
        "and j → i. Build a DIRECTED Network (`network(n; directed=true)`) with " *
        "the rank of every ordered pair on its own arc."))
    net.bipartite === nothing || throw(ArgumentError(
        "as_rank_network: the network is two-mode, but a RankNetwork holds COMPLETE " *
        "orderings — every ego ranks every other actor, and within-mode dyads are " *
        "structurally impossible in a two-mode network. Supply a one-mode directed " *
        "Network."))
    missing === :error || throw(ArgumentError(
        "as_rank_network: missing=$(repr(missing)) is not offered; the only policy is " *
        ":error (`missing_policies(as_rank_network) == (:error,)`). A masked dyad is " *
        "unobserved, and a RankNetwork holds COMPLETE orderings: an unobserved rank " *
        "has no face value to condition on (unlike a binary tie, an absent rank " *
        "cannot be read as \"0\"), so there is no `:face`. Drop the actors whose " *
        "rankings were not observed, or `clear_missing_dyads!(net)` once every dyad " *
        "really has been observed."))
    # The guard of the missing-data contract. `face_ok=false`: the error must
    # not suggest a `missing=:face` this routine does not (and cannot) offer.
    require_observed(net, :error; context="as_rank_network", face_ok=false)

    n = nv(net)
    ranks = zeros(Int, n, n)
    has_loops = false
    for i in 1:n, j in 1:n
        if i == j
            has_loops |= has_edge(net, i, i)
            continue
        end
        has_edge(net, i, j) || throw(ArgumentError(
            "as_rank_network: ego $i has no arc to actor $j, so its rank of $j is " *
            "unknown. A RankNetwork holds COMPLETE orderings: every ego must rank " *
            "every alter, one arc i → j per ordered pair carrying the rank in the " *
            "edge attribute $(repr(attr)) (greater = higher standing)."))
        v = get_edge_attribute(net, attr, i, j)
        v === nothing && throw(ArgumentError(
            "as_rank_network: the arc $i → $j carries no edge attribute $(repr(attr)), " *
            "so ego $i's rank of actor $j is unknown (`list_edge_attributes(net)` = " *
            "$(list_edge_attributes(net))). Pass the attribute holding the ranks as " *
            "`attr=`, or set it on every arc."))
        r = try
            Int(v)
        catch err
            err isa Union{InexactError, MethodError} || rethrow()
            throw(ArgumentError(
                "as_rank_network: ego $i's rank of actor $j in edge attribute " *
                "$(repr(attr)) is $(repr(v)) ($(typeof(v))), not an integer; ranks " *
                "must be integers, each ego's a permutation of 1:$(n - 1) (greater = " *
                "higher standing)."))
        end
        ranks[i, j] = r
    end
    violation = _ranking_violation(ranks)
    violation === nothing || throw(ArgumentError(
        "as_rank_network: edge attribute $(repr(attr)) is not a complete ranking: " *
        "$violation (each ego must assign the alters the ranks 1:$(n - 1) once each, " *
        "greater = higher standing)."))
    rnet = RankNetwork(ranks; validate=false)
    report || return rnet

    rep = ConversionReport(:Network, :RankNetwork)
    for a in sort!(list_edge_attributes(net))
        a === attr && continue
        record_drop!(rep, a, "edge attribute $(repr(a)): a RankNetwork carries only " *
                             "the ranks (edge attribute $(repr(attr)))")
    end
    for a in sort!(list_vertex_attributes(net))
        record_drop!(rep, a, "vertex attribute $(repr(a)): a RankNetwork has no vertex " *
                             "attributes (keep it beside the RankNetwork, e.g. as a " *
                             "RankNodeICov covariate)")
    end
    for a in sort!(list_network_attributes(net))
        record_drop!(rep, a, "network attribute $(repr(a)): a RankNetwork has no " *
                             "network attributes")
    end
    has_loops && record_drop!(rep, :self_loops, "self-loop arcs: a RankNetwork has " *
                                                "no self-ranks (the diagonal is 0)")
    return rnet, rep
end

# The declaration half of the missing-data contract: `as_rank_network` has
# considered masked dyads, and its only policy is to refuse them.
supports_missing(::typeof(as_rank_network)) = true
missing_policies(::typeof(as_rank_network)) = (:error,)

"""
    rank_matrix(rnet::RankNetwork) -> Matrix{Int}

The matrix of rank values (a copy): row `i` is ego `i`'s ranks of the
alters (greater = higher standing), the diagonal 0. The inverse of
[`as_rank_network`](@ref) on a matrix.

# Example
```julia
using ERGMRank
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
rnet = as_rank_network(m)
rank_matrix(rnet) == m                 # true
rank_matrix(rnet) === rnet.ranks       # false: a copy, safe to edit
```
"""
rank_matrix(rnet::RankNetwork) = copy(rnet.ranks)

"""
    newcomb_week1() -> RankNetwork

Newcomb's fraternity (Newcomb 1961), week 1: 17 men who had just moved into
a University of Michigan fraternity house, each ranking the other 16 by
liking — the `newcomb[[1]]` network of R `ergm.rank`, read with
`as.matrix(newcomb[[1]], attrname = "rank")`, with **greater value = higher
standing** (ergm.rank's convention: `y[i,j] > y[i,k]` means `i` ranks `j`
over `k`). It is the dataset every `ergm.rank` example fits, and the one the
golden fixture `test/fixtures/newcomb_rank.toml` (generated by
`test/fixtures/r/newcomb_rank.R` from ergm.rank 4.1.2) freezes: the matrix
below is that frozen ranking, verbatim, and the test suite asserts the two
are identical. Every Newcomb number quoted in the README and the estimation
guide — the swap-MPLE `[-0.1409, -0.00585]`, the MCMLE `[-0.1531,
-0.00659]`, the observed statistics `(844, 12748)` — is reproducible from it.

Weeks 2–15 are not bundled; `Networks.load_dataset` does not carry Newcomb
yet, so this is the one loader.

# Example
```julia
using ERGMRank
rnet = newcomb_week1()
rnet.n                                            # 17
is_valid_ranking(rnet)                            # true
compute(RankDeference(), rnet) == 844             # true: ergm.rank's summary()
compute(RankNonconformity(:all), rnet) == 12748   # true
fit = fit_ergm_rank(rnet, [RankDeference(), RankNonconformity()])   # the swap-MPLE, ~15 ms
round.(coef(fit); digits=4) == [-0.1409, -0.0059]                   # true
```

# Reference
Newcomb, T. M. (1961). *The Acquaintance Process.* New York: Holt, Rinehart
& Winston. Distributed as `newcomb` in R `ergm.rank` (Krivitsky & Butts).
"""
function newcomb_week1()
    # ergm.rank::newcomb[[1]], as.matrix(., attrname = "rank"); row = ego,
    # column = alter, greater = higher standing (the fixture's `ranks`)
    m = [
     0  7 12 11 10  4 13 14 15 16  3  9  1  5  8  6  2;
     8  0 16  1 11 12  2 14 10 13 15  6  7  9  5  3  4;
    13 10  0  7  8 11  9 15  6  5  2  1 16 12  4 14  3;
    13  1 15  0 14  4  3 16 12  7  6  9  8 11 10  5  2;
    14 10 11  7  0 16 12  4  5  6  2  3 13 15  8  9  1;
     7 13 11  3 15  0 10  2  4 16 14  5  1 12  9  8  6;
    15  4 11  3 16  8  0  6  9 10  5  2 14 12 13  7  1;
     9  8 16  7 10  1 14  0 11  3  2  5  4 15 12 13  6;
     6 16  8 14 13 11  4 15  0  7  1  2  9  5 12 10  3;
     2 16  9 14 11  4  3 10  7  0 15  8 12 13  1  6  5;
    12  7  4  8  6 14  9 16  3 13  0  2 10 15 11  5  1;
    15 11  2  6  5 14  7 13 10  4  3  0 16  8  9 12  1;
     1 15 16  7  4  2 12 14 13  8  6 11  0 10  3  9  5;
    14  5  8  6 13  9  2 16  1  3 12  7 15  0  4 11 10;
    16  9  4  8  1 13 11 12  6  2  3  5 10 15  0 14  7;
     8 11 15  3 13 16 14 12  1  9  2  6 10  7  5  0  4;
     9 15 10  2  4 11  5 12  3  7  8  1  6 16 14 13  0]
    return RankNetwork(m)
end

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
maintained by the AlterSwap move. It is the only reference measure this
package offers, and every [`RankERGMModel`](@ref) carries one.

# Example
```julia
using ERGMRank
ref = CompleteOrderReference()
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
fit = fit_ergm_rank(as_rank_network(m), [RankDeference()])
fit.model.reference == ref             # true: the reference of every rank fit
```
"""
struct CompleteOrderReference end

# =============================================================================
# Terms (statistics follow ergm.rank's wtchangestats_rank.c)
# =============================================================================

"""
    RankDeference <: AbstractERGMTerm

Deference (aversion), `rank.deference` in ergm.rank: the number of ordered
triples (ego `i`, deferred-to `l`, other `j`) such that `l` ranks `j` over
`i` while `i` ranks `l` over `j`. A negative coefficient is aversion to
deference. Implements `compute` and [`swap_change`](@ref).

# Example
```julia
using ERGMRank
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
rnet = as_rank_network(m)
compute(RankDeference(), rnet)         # 6.0 (R ergm.rank: summary(nw ~ rank.deference))
name(RankDeference())                  # "rank.deference"
swap_change(RankDeference(), rnet, 1, 2, 3)   # Δ if ego 1 swaps its ranks of 2 and 3
```
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

!!! warning "Not implemented: ergm.rank's other `rank.nonconformity` variants"
    `ergm.rank`'s `rank.nonconformity(to = "local1")`, `"local2"`,
    `"geometric"` and `"thresholds"` have no counterpart here. Only `:all`
    and `:localAND` exist; any other variant is refused at construction with
    `ArgumentError: RankNonconformity: variant must be :all or :localAND (got
    :local1); ergm.rank's local1/local2/geometric/thresholds variants are not
    implemented in ERGMRank.jl`.

# Example
```julia
using ERGMRank
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
rnet = as_rank_network(m)
compute(RankNonconformity(), rnet)             # 10.0 (R: rank.nonconformity("all"))
compute(RankNonconformity(:localAND), rnet)    # 4.0  (R: rank.nonconformity("localAND"))
name(RankNonconformity(:localAND))             # "rank.nonconformity.localAND"
try
    RankNonconformity(:local1)
catch err
    err isa ArgumentError                      # true: not implemented
end
```
"""
struct RankNonconformity <: AbstractERGMTerm
    variant::Symbol

    function RankNonconformity(variant::Symbol=:all)
        variant in (:all, :localAND) || throw(ArgumentError(
            "RankNonconformity: variant must be :all or :localAND (got $(repr(variant))); " *
            "ergm.rank's local1/local2/geometric/thresholds variants are not " *
            "implemented in ERGMRank.jl"))
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
be ranked higher. A covariate whose length is not the number of actors is
refused by `compute` and `swap_change` (an `ArgumentError` quoting both).

# Fields
- `x::Vector{Float64}`: Actor-level covariate
- `label::String`: Name for output (default "x")

# Example
```julia
using ERGMRank
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
rnet = as_rank_network(m)
compute(RankNodeICov([10, 20, 30, 40]), rnet)        # -40.0 (R: rank.nodeicov)
name(RankNodeICov([10, 20, 30, 40]; label="age"))   # "rank.nodeicov.age"
```
"""
struct RankNodeICov <: AbstractERGMTerm
    x::Vector{Float64}
    label::String

    RankNodeICov(x::AbstractVector{<:Real}; label::String="x") =
        new(Float64.(x), label)
end

# ERGM.jl's `NodeCov(:age)` spelling — an attribute NAME — is what an R user
# tries first; a RankNetwork carries no attributes, so say where the values
# come from instead of a bare MethodError
RankNodeICov(attr::Union{Symbol, AbstractString}; kwargs...) = throw(ArgumentError(
    "RankNodeICov($(repr(attr))): a RankNetwork carries no vertex attributes, so pass " *
    "the covariate VALUES themselves (one per actor, in actor order), e.g. " *
    "RankNodeICov(vertex_attribute_vector(net, $(repr(Symbol(attr))), Float64); " *
    "label=$(repr(string(attr)))) from the Network the ranking was read from, or " *
    "RankNodeICov([20, 30, 40, 50]; label=$(repr(string(attr))))"))

name(t::RankNodeICov) = "rank.nodeicov.$(t.label)"

# One message for a covariate of the wrong size, shared by `compute` and
# `swap_change` (and hence the fitter, the sampler and `gof`), naming the term
_bad_nodeicov_length(t::RankNodeICov, n::Int) =
    "RankNodeICov($(repr(t.label))): the covariate has $(length(t.x)) values but the " *
    "network has $n actors; pass one value per actor, in actor order"
_bad_matrix_size(what::AbstractString, sz::Tuple, n::Int) =
    "$what is $(sz[1])×$(sz[2]) but the network has $n actors; pass a $n×$n matrix " *
    "with row i = ego i, column j = alter j"

function compute(t::RankNodeICov, rnet::RankNetwork)
    n = rnet.n
    length(t.x) == n || throw(ArgumentError(_bad_nodeicov_length(t, n)))
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
(`(y_ij > y_ik) ≠ (ref_ij > ref_ik)`). Useful for drift from a prior wave
or an exogenous ordering. Fitting `RankInconsistency(rnet)` to `rnet`
itself puts the statistic at the boundary of its attainable range (zero
inconsistency): the coefficient is then fixed at `-Inf` with R's `drop`
semantics — see [`fit_ergm_rank`](@ref).

# Fields
- `ref::Matrix{Int}`: Reference rank matrix (same convention)

# Example
```julia
using ERGMRank
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
rnet = as_rank_network(m)
compute(RankInconsistency(rnet), rnet)          # 0.0: no disagreement with itself
other = copy(rnet); swap_ranks!(other, 1, 2, 3)
compute(RankInconsistency(rnet), other)         # 2.0: ego 1's (2,3) comparison, both orders
name(RankInconsistency(rank_matrix(rnet)))      # "rank.inconsistency"
```
"""
struct RankInconsistency <: AbstractERGMTerm
    ref::Matrix{Int}
end

RankInconsistency(ref_net::RankNetwork) = RankInconsistency(rank_matrix(ref_net))

name(::RankInconsistency) = "rank.inconsistency"

function compute(t::RankInconsistency, rnet::RankNetwork)
    n = rnet.n
    size(t.ref) == (n, n) || throw(ArgumentError(
        _bad_matrix_size("RankInconsistency: the reference rank matrix", size(t.ref), n)))
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
alter pair (j, k) with j ranked over k, add `cov[i, j] − cov[i, k]`. With
`cov[i, j] = x[j]` it is exactly [`RankNodeICov`](@ref)`(x)`. A covariate
matrix that is not `n × n` is refused by `compute` and `swap_change`.

# Fields
- `cov::Matrix{Float64}`: Dyadic covariate matrix
- `label::String`: Name for output

# Example
```julia
using ERGMRank
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
rnet = as_rank_network(m)
x = [10.0, 20.0, 30.0, 40.0]
cov = repeat(x', 4, 1)                                  # cov[i, j] = x[j]
compute(RankEdgeCov(cov), rnet) == compute(RankNodeICov(x), rnet)   # true: -40.0
name(RankEdgeCov(cov; label="distance"))                # "rank.edgecov.distance"
```
"""
struct RankEdgeCov <: AbstractERGMTerm
    cov::Matrix{Float64}
    label::String

    RankEdgeCov(cov::AbstractMatrix{<:Real}; label::String="cov") =
        new(Float64.(cov), label)
end

RankEdgeCov(attr::Union{Symbol, AbstractString}; kwargs...) = throw(ArgumentError(
    "RankEdgeCov($(repr(attr))): a RankNetwork carries no edge attributes, so pass " *
    "the n×n covariate MATRIX itself (row i = ego i, column j = alter j), e.g. " *
    "RankEdgeCov(edge_attribute_matrix(net, $(repr(Symbol(attr)))); " *
    "label=$(repr(string(attr)))) from the Network the ranking was read from"))

name(t::RankEdgeCov) = "rank.edgecov.$(t.label)"

function compute(t::RankEdgeCov, rnet::RankNetwork)
    n = rnet.n
    size(t.cov) == (n, n) || throw(ArgumentError(
        _bad_matrix_size("RankEdgeCov($(repr(t.label))): the dyadic covariate", size(t.cov), n)))
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

# =============================================================================
# Swap change statistics (ergm.rank's wtchangestats_rank.c change functions)
# =============================================================================
#
# The AlterSwap move — ego swaps the ranks of alters j and k — changes ego's
# relative order of exactly the alter pairs that involve j or k, and only
# those whose other member sits between the two ranks. Every term's change
# statistic is therefore a sum over the O(n) pairs (x, z) of ego's row whose
# comparison flips, times whatever the term multiplies that comparison by;
# the two nonconformity variants additionally loop over the other actor
# whose comparison is being matched (O(n²)). Nothing is recomputed from the
# whole ranking (O(n³)–O(n⁴)), and nothing allocates.

# 1[y'[ego,x] > y'[ego,z]] − 1[y[ego,x] > y[ego,z]] for the swap of j and k:
# −1, 0 or +1 (the pair's order flipped down, stayed, or flipped up)
@inline function _swap_pair_change(y::Matrix{Int}, ego::Int, x::Int, z::Int,
                                   j::Int, k::Int)
    @inbounds begin
        rj = y[ego, j]
        rk = y[ego, k]
        rx = y[ego, x]
        rz = y[ego, z]
    end
    rx′ = x == j ? rk : (x == k ? rj : rx)
    rz′ = z == j ? rk : (z == k ? rj : rz)
    return Int(rx′ > rz′) - Int(rx > rz)
end

# Ego's rank of alter x AFTER the swap of j and k
@inline function _swapped_rank(y::Matrix{Int}, ego::Int, x::Int, j::Int, k::Int)
    @inbounds return x == j ? y[ego, k] : (x == k ? y[ego, j] : y[ego, x])
end

# Σ f(x, z, d) over every ORDERED alter pair (x, z) of `ego` whose relative
# order the swap of j and k changes (x or z in {j, k}; each pair visited
# once), with d = _swap_pair_change(...) ≠ 0. `f` returns the term's
# contribution for that pair (a Float64) and captures only immutable state,
# so the reduction allocates nothing.
@inline function _sum_changed_pairs(f::F, y::Matrix{Int}, n::Int, ego::Int,
                                    j::Int, k::Int) where {F}
    total = 0.0
    # Pairs whose first member is j or k (includes (j, k) and (k, j))
    for x in (j, k), z in 1:n
        (z == ego || z == x) && continue
        d = _swap_pair_change(y, ego, x, z, j, k)
        d == 0 && continue
        total += f(x, z, d)
    end
    # Pairs whose second member is j or k and whose first is neither
    for z in (j, k), x in 1:n
        (x == ego || x == j || x == k) && continue
        d = _swap_pair_change(y, ego, x, z, j, k)
        d == 0 && continue
        total += f(x, z, d)
    end
    return total
end

# The swap's three indices must name an ego and two distinct alters (the same
# refusal `swap_ranks!` makes); out of the hot path so the callers stay
# allocation-free
@noinline function _throw_bad_swap(context::AbstractString, n::Int, ego::Int, j::Int,
                                   k::Int)
    throw(ArgumentError("$context: a swap needs an ego and two DISTINCT alters in " *
                        "1:$n, none equal to the ego (got ego = $ego, j = $j, k = $k)"))
end

@inline function _check_swap(context::AbstractString, rnet::RankNetwork, ego::Int,
                             j::Int, k::Int)
    n = rnet.n
    (1 <= ego <= n && 1 <= j <= n && 1 <= k <= n && ego != j && ego != k && j != k) ||
        _throw_bad_swap(context, n, ego, j, k)
    return nothing
end

"""
    swap_change(term, rnet::RankNetwork, ego, j, k) -> Float64

The change in the statistic of `term` when `ego` swaps the ranks it assigns
alters `j` and `k` — `compute(term, y_swapped) − compute(term, y)` — the
AlterSwap analogue of `ERGM.change_stat`, ported from `ergm.rank`'s
`wtchangestats_rank.c` change functions. It is evaluated from the
comparisons the swap actually touches (the alter pairs of `ego`'s row that
involve `j` or `k`): O(n) for `RankDeference`, `RankNodeICov`,
`RankInconsistency` and `RankEdgeCov`, O(n²) for both `RankNonconformity`
variants — never by recomputing either full statistic — and allocates
nothing. `rnet` is left unchanged.

Every rank term implements this method alongside `compute`; the sampler's
`_swap_delta!`, the swap-MPLE design and `gof` all run on it. A term that
implements only `compute` would make the sampler fall back to nothing: there
is no brute-force default, because a per-step O(n⁴) recomputation is exactly
what this method exists to prevent (the test suite keeps that brute force as
its oracle, `ERGMRank._swap_delta_bruteforce`).

`ego`, `j` and `k` must be distinct actors in `1:n` with `ego ∉ {j, k}`
(an `ArgumentError` otherwise); a covariate of the wrong size is refused as
`compute` refuses it.

# Example
```julia
using ERGMRank
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
rnet = as_rank_network(m)
term = RankDeference()
Δ = swap_change(term, rnet, 1, 2, 3)      # ego 1 swaps its ranks of 2 and 3
swapped = copy(rnet); swap_ranks!(swapped, 1, 2, 3)
Δ == compute(term, swapped) - compute(term, rnet)    # true
```
"""
function swap_change end

function swap_change(::RankDeference, rnet::RankNetwork, ego::Int, j::Int, k::Int)
    _check_swap("swap_change", rnet, ego, j, k)
    y = rnet.ranks
    # Ego's row enters the triple count twice: as v1 (deferring to x = v3,
    # who ranks z = v2 over ego) and as v3 (ego ranks x = v2 over z = v1,
    # who ranks ego over x). Both factors are outside ego's row.
    return _sum_changed_pairs(y, rnet.n, ego, j, k) do x, z, d
        @inbounds d * (Int(y[x, z] > y[x, ego]) + Int(y[z, ego] > y[z, x]))
    end
end

function swap_change(t::RankNonconformity, rnet::RankNetwork, ego::Int, j::Int, k::Int)
    _check_swap("swap_change", rnet, ego, j, k)
    return t.variant == :all ? _swap_change_nonconformity_all(rnet, ego, j, k) :
                               _swap_change_nonconformity_local(rnet, ego, j, k)
end

# Global nonconformity: ego's comparison of (x, z) is matched against every
# other actor u's comparison of the same pair (u ∉ {x, z}); a flip turns an
# agreement into a disagreement (+1) or the reverse (−1).
function _swap_change_nonconformity_all(rnet::RankNetwork, ego::Int, j::Int, k::Int)
    y = rnet.ranks
    n = rnet.n
    total = 0.0
    for u in 1:n
        u == ego && continue
        total += _sum_changed_pairs(y, n, ego, j, k) do x, z, d
            (x == u || z == u) && return 0.0
            @inbounds agreed = (d < 0) == (y[u, x] > y[u, z])   # before the swap
            agreed ? 1.0 : -1.0
        end
    end
    return total
end

# Local nonconformity (Krivitsky & Butts): the summand for (v1 = i, v2, v3 =
# l, v4) is 1[y_il > y_iv2] · 1[y_il > y_iv4] · 1[y_iv2 ≤ y_iv4] · 1[y_lv2 >
# y_lv4]. Ego's row enters as i (three of its own comparisons) and as l (the
# comparison being matched by every other i).
@inline function _local_triple(y::Matrix{Int}, l::Int, v2::Int, v4::Int,
                               rl::Int, r2::Int, r4::Int)
    @inbounds return rl > r2 && rl > r4 && r2 <= r4 && y[l, v2] > y[l, v4]
end

@inline function _local_triple_change(y::Matrix{Int}, ego::Int, l::Int, v2::Int,
                                      v4::Int, j::Int, k::Int)
    @inbounds before = _local_triple(y, l, v2, v4, y[ego, l], y[ego, v2], y[ego, v4])
    after = _local_triple(y, l, v2, v4, _swapped_rank(y, ego, l, j, k),
                          _swapped_rank(y, ego, v2, j, k),
                          _swapped_rank(y, ego, v4, j, k))
    return Int(after) - Int(before)
end

function _swap_change_nonconformity_local(rnet::RankNetwork, ego::Int, j::Int, k::Int)
    y = rnet.ranks
    n = rnet.n
    total = 0
    # Ego as i: the triples (l, v2, v4) of ego's alters that contain j or k,
    # each once. With l ∈ {j, k} every pair below l may cross the threshold,
    # so all ordered pairs are visited; otherwise only pairs containing j or k.
    for l in 1:n
        l == ego && continue
        if l == j || l == k
            for v2 in 1:n
                (v2 == ego || v2 == l) && continue
                for v4 in 1:n
                    (v4 == ego || v4 == l || v4 == v2) && continue
                    total += _local_triple_change(y, ego, l, v2, v4, j, k)
                end
            end
        else
            for v2 in (j, k), v4 in 1:n
                (v4 == ego || v4 == l || v4 == v2) && continue
                total += _local_triple_change(y, ego, l, v2, v4, j, k)
            end
            for v4 in (j, k), v2 in 1:n
                (v2 == ego || v2 == l || v2 == j || v2 == k) && continue
                total += _local_triple_change(y, ego, l, v2, v4, j, k)
            end
        end
    end
    # Ego as l: every other i that ranks ego over both members of a flipped
    # pair (x = v2, z = v4), with x ranked at or below z by i, matched ego's
    # comparison of the pair before the swap iff it disagrees after.
    for i in 1:n
        i == ego && continue
        @inbounds r_ego = y[i, ego]
        total += Int(_sum_changed_pairs(y, n, ego, j, k) do x, z, d
            (x == i || z == i) && return 0.0
            @inbounds counted = r_ego > y[i, x] && r_ego > y[i, z] && y[i, x] <= y[i, z]
            counted ? Float64(d) : 0.0
        end)
    end
    return Float64(total)
end

function swap_change(t::RankNodeICov, rnet::RankNetwork, ego::Int, j::Int, k::Int)
    _check_swap("swap_change", rnet, ego, j, k)
    n = rnet.n
    length(t.x) == n || throw(ArgumentError(_bad_nodeicov_length(t, n)))
    x = t.x
    return _sum_changed_pairs(rnet.ranks, n, ego, j, k) do a, b, d
        @inbounds d * (x[a] - x[b])
    end
end

function swap_change(t::RankInconsistency, rnet::RankNetwork, ego::Int, j::Int, k::Int)
    _check_swap("swap_change", rnet, ego, j, k)
    n = rnet.n
    size(t.ref) == (n, n) || throw(ArgumentError(
        _bad_matrix_size("RankInconsistency: the reference rank matrix", size(t.ref), n)))
    r = t.ref
    return _sum_changed_pairs(rnet.ranks, n, ego, j, k) do x, z, d
        # A flip turns agreement with the reference into disagreement (+1)
        # or disagreement into agreement (−1)
        @inbounds agreed = (d < 0) == (r[ego, x] > r[ego, z])
        agreed ? 1.0 : -1.0
    end
end

function swap_change(t::RankEdgeCov, rnet::RankNetwork, ego::Int, j::Int, k::Int)
    _check_swap("swap_change", rnet, ego, j, k)
    n = rnet.n
    size(t.cov) == (n, n) || throw(ArgumentError(
        _bad_matrix_size("RankEdgeCov($(repr(t.label))): the dyadic covariate", size(t.cov), n)))
    c = t.cov
    return _sum_changed_pairs(rnet.ranks, n, ego, j, k) do x, z, d
        @inbounds d * (c[ego, x] - c[ego, z])
    end
end

# Change in the statistic VECTOR from swapping ego's ranks of j and k,
# written into `delta` (the sampler's workspace): g(y_swapped) − g(y), one
# `swap_change` per term. Generated per term count so the body is the same
# straight-line sequence of statically dispatched calls for ANY number of
# terms — `map` over a tuple falls back to a boxed Vector{Any} from 32
# elements on (ERGM.jl's `_change_stat_tuple` pattern). 0 bytes per call over
# a term tuple, pinned at p = 2 and p = 8.
@generated function _swap_delta!(delta::AbstractVector{Float64}, terms::T,
                                 rnet::RankNetwork, ego::Int, j::Int, k::Int) where {T<:Tuple}
    p = length(T.parameters)
    word = p == 1 ? "term" : "terms"
    body = [:(@inbounds delta[$i] = swap_change(terms[$i], rnet, ego, j, k)) for i in 1:p]
    return quote
        length(delta) == $p || throw(ArgumentError(
            "_swap_delta!: delta has length $(length(delta)) but the model has $($p) $($word)"))
        $(body...)
        return delta
    end
end

# Allocating convenience form over a term tuple or vector
_swap_delta(terms, rnet::RankNetwork, ego::Int, j::Int, k::Int) =
    _swap_delta!(Vector{Float64}(undef, length(terms)), Tuple(terms), rnet, ego, j, k)

# The brute-force oracle: g(y_swapped) − g(y) from two full `compute`
# evaluations (swap in place, restore). Test-only — it is what every
# `swap_change` method is asserted equal to — and never on a hot path.
function _swap_delta_bruteforce(terms, rnet::RankNetwork, ego::Int, j::Int, k::Int)
    before = [compute(t, rnet) for t in terms]
    swap_ranks!(rnet, ego, j, k)
    after = [compute(t, rnet) for t in terms]
    swap_ranks!(rnet, ego, j, k)
    return after .- before
end

# =============================================================================
# Model and estimation
# =============================================================================

"""
    RankERGMModel

Rank-order ERGM specification: terms plus the observed `RankNetwork`
under the `CompleteOrderReference`. Built by [`fit_ergm_rank`](@ref) (a
copy of the network, the terms as a `Vector{AbstractERGMTerm}`) and carried
on the [`RankERGMResult`](@ref) as `fit.model`, so `simulate_rank_ergm(fit)`
and `gof(fit)` know what was fitted to what.

# Fields
- `terms::Vector{AbstractERGMTerm}`: the model's rank terms
- `network::RankNetwork`: the observed ranking
- `reference::CompleteOrderReference`: the reference measure

# Example
```julia
using ERGMRank
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
fit = fit_ergm_rank(as_rank_network(m), [RankDeference()])
fit.model isa RankERGMModel            # true
name.(fit.model.terms)                 # ["rank.deference"]
fit.model.network.n                    # 4
string(fit.model)                      # "RankERGMModel(1 term, 4 actors, CompleteOrder)"
```
"""
struct RankERGMModel
    terms::Vector{AbstractERGMTerm}
    network::RankNetwork
    reference::CompleteOrderReference
end

# Networks.jl's display convention (as for `RankNetwork` and `RankERGMResult`):
# the 2-arg `show` is the one-line form containers use, the `MIME"text/plain"`
# method the REPL calls lists the terms.
function Base.show(io::IO, m::RankERGMModel)
    p = length(m.terms)
    print(io, "RankERGMModel(", p, p == 1 ? " term" : " terms", ", ",
          m.network.n, " actors, CompleteOrder)")
end

function Base.show(io::IO, ::MIME"text/plain", m::RankERGMModel)
    p = length(m.terms)
    println(io, "Rank-Order ERGM model: ", p, p == 1 ? " term" : " terms", " on ",
            m.network.n, " actors")
    println(io, "  Reference: CompleteOrder (uniform over complete orderings)")
    println(io, "  Network: ", m.network, " (", _n_comparisons(m.network),
            " swap comparisons)")
    print(io, "  Terms:")
    for t in m.terms
        print(io, "\n    ", name(t))
    end
end

"""
    RankERGMResult

Results from fitting a rank-order ERGM with [`fit_ergm_rank`](@ref).

`method` records which estimator produced it:

- `:mple` — the swap-based maximum pseudo-likelihood; `loglik` is then the
  maximized swap pseudo-log-likelihood.
- `:mcmle` — the MCMC maximum likelihood estimate (the estimator of R
  `ergm.rank`); `loglik` is then the path-sampling (bridge) estimate of the
  log-likelihood, `NaN` when the fit was run with `bridge_rungs=0`.
  `mcmc_convergence` carries the convergence report of the final MCMC sample
  (an `ERGM.MCMLEConvergence`: iterations, step length, per-statistic
  t-ratios, Hotelling p-value, effective sample size; `nothing` for `:mple`),
  `mcmc_samples` the statistics sampled at the returned coefficients, and
  `bridge_rungs` the number of path-sampling segments behind `loglik`.

`se_type` records how `std_errors`/`vcov` were ACTUALLY obtained:

- `:hessian` — the inverse negative Hessian of the swap pseudo-likelihood. The
  swap comparisons overlap, so these are expected anticonservative.
- `:bootstrap` — the parametric bootstrap of `fit_ergm_rank(...; se=:bootstrap)`
  (simulate rank networks at θ̂ with the AlterSwap sampler, refit, empirical
  covariance). `boot_replicates` then holds the `n_boot × p` matrix of refitted
  coefficients — a replicate on which the swap MPLE did not exist (a statistic
  at the boundary of its attainable range in the SIMULATED ranking) is a `NaN`
  row, excluded from the covariance and counted by `approximations`.
- `:fisher` (`:mcmle` only) — the inverse Fisher information estimated from
  the final MCMC sample, `V = Σ̂⁻¹`, **plus the Monte-Carlo component**
  `V·Σ_mc·V` of the estimating equation (Hunter & Handcock 2006, §3.3; the
  same covariance as `ERGM.mcmle`'s). `vcov_fisher` holds the Fisher part
  alone, so `sqrt.(diag(vcov .- vcov_fisher))` is the Monte-Carlo standard
  error and `show` prints R's "MCMC %" column.

`n_kept` is the number of swap comparisons the finite coefficients were
estimated on: every comparison (`nobs`), or, after a boundary statistic was
dropped, the comparisons the dropped statistics do not change — the sample
size `bic` uses, as R's `logLik` `nobs` attribute does.

`se_type` is what `Networks.se_method(fit)` reports, and what `show` reads
before deciding whether the anticonservatism caveat is still true of this fit.

The full StatsAPI surface (`coef`, `stderror`, `vcov`, `confint`,
`loglikelihood`, `nobs`, `dof`, `aic`, `bic`, `coeftable`) and the shared
result-metadata protocol (`fit_metadata`, `approximations`, `estimand`,
`objective`, `is_exact`, `se_method`, `missing_method`) are defined on it.
The REPL prints the R-style block (`show(io, MIME"text/plain"(), fit)`:
estimator, log-likelihood, convergence, the coefficient table, the caveats);
`print`/`string` give the one-line form.

# Example
```julia
using ERGMRank
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
fit = fit_ergm_rank(as_rank_network(m), [RankDeference()])
fit isa RankERGMResult                 # true
fit.method                             # :mple
fit.converged                          # true
length(coef(fit)) == length(stderror(fit)) == 1    # true
string(fit)                            # "RankERGMResult(swap-MPLE, 1 term, converged)"
```
"""
struct RankERGMResult
    model::RankERGMModel
    coefficients::Vector{Float64}
    std_errors::Vector{Float64}
    vcov::Matrix{Float64}
    loglik::Float64
    converged::Bool
    se_type::Symbol
    boot_replicates::Union{Nothing, Matrix{Float64}}
    n_kept::Int
    method::Symbol
    mcmc_convergence::Union{Nothing, MCMLEConvergence}
    mcmc_samples::Union{Nothing, Matrix{Float64}}
    vcov_fisher::Union{Nothing, Matrix{Float64}}
    bridge_rungs::Int

    function RankERGMResult(model, coefficients, std_errors, vcov, loglik, converged,
                            se_type, boot_replicates, n_kept, method,
                            mcmc_convergence, mcmc_samples, vcov_fisher, bridge_rungs)
        method in (:mple, :mcmle) || throw(ArgumentError(
            "RankERGMResult: method must be :mple or :mcmle (got $(repr(method)))"))
        new(model, coefficients, std_errors, vcov, loglik, converged, se_type,
            boot_replicates, n_kept, method, mcmc_convergence, mcmc_samples,
            vcov_fisher, bridge_rungs)
    end
end

# Backwards-compatible constructors: a result built without an `se_type`
# reports the inverse-Hessian standard errors it in fact had; one built
# without a replicate matrix or a kept-comparison count had no bootstrap and
# no dropped statistic; one built without a `method` is a swap-MPLE fit (the
# only estimator before 0.2), with no MCMC sample and no bridge.
RankERGMResult(model, coefficients, std_errors, vcov, loglik, converged) =
    RankERGMResult(model, coefficients, std_errors, vcov, loglik, converged,
                   :hessian)
RankERGMResult(model, coefficients, std_errors, vcov, loglik, converged, se_type) =
    RankERGMResult(model, coefficients, std_errors, vcov, loglik, converged,
                   se_type, nothing, _n_comparisons(model.network))
RankERGMResult(model, coefficients, std_errors, vcov, loglik, converged, se_type,
               boot_replicates, n_kept) =
    RankERGMResult(model, coefficients, std_errors, vcov, loglik, converged,
                   se_type, boot_replicates, n_kept, :mple, nothing, nothing,
                   nothing, 0)

# Number of pseudo-likelihood contributions: one per (ego, unordered alter
# pair) conditional — and the size of the AlterSwap proposal space, the rank
# analogue of ERGM's dyad count
_n_comparisons(rnet::RankNetwork) = rnet.n * (rnet.n - 1) * (rnet.n - 2) ÷ 2

"""
    _mcmc_defaults(rnet::RankNetwork) -> (burnin, interval)

THE burn-in/interval rule behind every AlterSwap sampler default in this
package (`simulate_rank_ergm`, `gof`, the `se=:bootstrap` refits): the ONE
dyad-scaled rule of ERGM.jl, `ERGM._mcmc_defaults(n)` — `burnin = 20 n` and
`interval = max(100, n ÷ 10)` — applied to the size of the swap proposal
space `n = n_actors (n_actors − 1)(n_actors − 2) / 2` (the number of (ego,
unordered alter pair) swaps, the rank analogue of the dyad count: a chain
that proposes one swap per step needs a number of steps proportional to
that count to move every comparison a bounded number of times). A fixed
budget (the pre-0.2 `500`/`50`) was far too small for a 17-actor ranking
(2,040 swaps) and needlessly large for a 4-actor one (12).

# Example
```julia
using ERGMRank
ERGMRank._mcmc_defaults(RankNetwork(4))     # (burnin = 240, interval = 100): 12 swaps
ERGMRank._mcmc_defaults(RankNetwork(17))    # (burnin = 40800, interval = 204): 2040 swaps
```
"""
_mcmc_defaults(rnet::RankNetwork) = ERGM._mcmc_defaults(_n_comparisons(rnet))

# Resolve `burnin`/`interval` keywords that default to `nothing`: an explicit
# integer is honoured as given, `nothing` becomes the swap-scaled default
function _resolve_mcmc_controls(rnet::RankNetwork, burnin, interval)
    if burnin === nothing || interval === nothing
        d = _mcmc_defaults(rnet)
        burnin = something(burnin, d.burnin)
        interval = something(interval, d.interval)
    end
    return Int(burnin), Int(interval)
end

# The (z, p) columns of the coefficient table: `Networks.z_pvalues` (floored at
# floatmin, never 0.0; NaN standard errors give NaN p-values), with a
# coefficient fixed at ∓Inf by a boundary statistic reported as R prints it —
# z = ∓Inf, p = 0 — rather than the NaN its zero standard error would give.
function _z_and_p(result::RankERGMResult)
    zp = z_pvalues(result.coefficients, result.std_errors)
    z, p = zp.z, zp.p
    for k in eachindex(result.coefficients)
        if isinf(result.coefficients[k])
            z[k] = result.coefficients[k]
            p[k] = 0.0
        end
    end
    return z, p
end

# Three significant digits for the diagnostics quoted in warnings and caveats
_fmt3(x::Real) = isfinite(x) ? string(round(x; sigdigits=3)) : string(x)

# THE non-convergence sentence: printed by `show` under `Converged: false`
# and listed by `approximations`, so they cannot disagree. Under `:mcmle` it
# quotes the convergence report of the final sample (max t-ratio, Hotelling
# p, step length, iterations) exactly as ERGM.mcmle does. Under `:mple` three
# ways a fit ends unconverged: a coefficient with no comparison left to
# estimate it on (NaN, after a drop), a separated design (the
# pseudo-likelihood has no finite maximum), or Newton exhausting `maxiter`.
function _nonconvergence_caveat(result::RankERGMResult)
    if result.method === :mcmle
        c = result.mcmc_convergence
        detail = c === nothing ? "" :
            " (max t-ratio $(_fmt3(maximum(c.t_ratios))), Hotelling p " *
            "$(_fmt3(c.hotelling_p)), step length γ $(_fmt3(c.step_length)) " *
            "after $(c.iterations) iteration$(c.iterations == 1 ? "" : "s"))"
        return "MCMLE did not converge$detail: the sampled statistics at the " *
               "returned coefficients are not yet indistinguishable from the " *
               "observed ones, so the point estimates and standard errors are " *
               "unreliable — increase maxiter/n_samples/burnin, or refit with " *
               "init=coef(fit)"
    end
    names = [name(t) for t in result.model.terms]
    nan = [names[k] for k in eachindex(names) if isnan(result.coefficients[k])]
    if !isempty(nan)
        return "coefficient(s) $(join(nan, ", ")) are NaN: not identified — every " *
               "swap comparison changes a statistic that was dropped at the " *
               "boundary of its attainable range, so no comparison is left to " *
               "estimate them on"
    end
    return "the swap-MPLE Newton iteration did not converge: either the swap " *
           "pseudo-likelihood has no finite maximum (perfect separation — a " *
           "combination of the statistics no single swap can lower, R ergm: " *
           "\"The MPLE does not exist!\") or Newton hit maxiter; the coefficients " *
           "are the last iterate and the standard errors and p-values are " *
           "meaningless"
end

# A coefficient fixed at ∓Inf by a boundary statistic (R's `drop`): said in
# `show` and in `approximations`, from the coefficients themselves
function _fixed_coefficient_note(result::RankERGMResult)
    names = [name(t) for t in result.model.terms]
    lo = [names[k] for k in eachindex(names) if result.coefficients[k] == -Inf]
    hi = [names[k] for k in eachindex(names) if result.coefficients[k] == Inf]
    isempty(lo) && isempty(hi) && return nothing
    parts = String[]
    isempty(lo) || push!(parts, "$(join(lo, ", ")) fixed at -Inf (the observed " *
                                "ranking is at the smallest value the statistic " *
                                "can attain under any single swap)")
    isempty(hi) || push!(parts, "$(join(hi, ", ")) fixed at +Inf (the observed " *
                                "ranking is at the largest value the statistic " *
                                "can attain under any single swap)")
    return "coefficient(s) " * join(parts, "; ") * ": no finite swap-MPLE exists; " *
           "the other coefficients are estimated on the $(result.n_kept) swap " *
           "comparisons these statistics do not change, as R ergm does (drop=TRUE)"
end

# Standard errors the Hessian could not deliver (`newton_fit` returns NaN, with
# a warning, when the pseudo-Hessian at the solution is not negative definite)
function _undefined_se_note(result::RankERGMResult)
    any(k -> isnan(result.std_errors[k]) && isfinite(result.coefficients[k]),
        eachindex(result.coefficients)) || return nothing
    result.method === :mcmle &&
        return "standard errors are undefined (NaN): the covariance of the " *
               "statistics sampled at the returned coefficients could not be " *
               "inverted (collinear statistics, or a collapsed sampler)"
    return "standard errors are undefined (NaN): the pseudo-Hessian at the " *
           "solution is not negative definite — a coefficient is not identified " *
           "by the swap comparisons (a statistic no single swap changes, or two " *
           "statistics that change identically)"
end

# Bootstrap replicates without a finite swap MPLE (a boundary statistic in the
# SIMULATED ranking) were excluded from the covariance: said in `show` and in
# `approximations`, from the replicate matrix itself
function _boot_exclusion_note(result::RankERGMResult)
    reps = result.boot_replicates
    reps === nothing && return nothing
    n_boot = size(reps, 1)
    n_dropped = count(b -> !all(isfinite, view(reps, b, :)), 1:n_boot)
    n_dropped == 0 && return nothing
    return "$n_dropped of the $n_boot parametric-bootstrap refits had no finite " *
           "swap MPLE (a statistic at the boundary of its attainable range in the " *
           "simulated ranking) and were excluded from the standard errors, which " *
           "are the empirical covariance of the remaining $(n_boot - n_dropped) " *
           "refits (fit.boot_replicates)"
end

# R's "MCMC %" (`ergm:::summary.ergm`): `round(100 * (tot.se - mod.se) /
# tot.se)` — the share of the TOTAL standard error that the Monte-Carlo term
# adds over the Fisher part, rounded to an integer; NaN where either is NaN.
function _mcmc_percent(result::RankERGMResult)
    result.vcov_fisher === nothing && return fill(NaN, length(result.std_errors))
    se_fisher = sqrt.(max.(diag(result.vcov_fisher), 0.0))
    return [isfinite(se) && isfinite(f) && se > 0 ? round(Int, 100 * (se - f) / se) : NaN
            for (f, se) in zip(se_fisher, result.std_errors)]
end

# The sentence naming the estimator, shared by `show` and `approximations`
_estimation_label(result::RankERGMResult) =
    result.method === :mcmle ?
        "MCMC MLE (AlterSwap Metropolis, " *
        (result.bridge_rungs == 0 ? "no bridge" :
         "$(result.bridge_rungs) bridge rung$(result.bridge_rungs == 1 ? "" : "s")") * ")" :
        "swap pseudo-likelihood (swap-MPLE)"

# Display. Two-arg `show` is the ONE-LINE form (inside containers, `print`,
# `string`); the R-style block is the `MIME"text/plain"` method the REPL
# calls for a top-level value — Networks.jl's convention (one multi-line 2-arg
# method garbles every container of results).
function Base.show(io::IO, result::RankERGMResult)
    p = length(result.coefficients)
    print(io, "RankERGMResult(", result.method === :mcmle ? "MCMC MLE" : "swap-MPLE",
          ", ", p, p == 1 ? " term" : " terms", ", ",
          result.converged ? "converged" : "NOT converged", ")")
end

function Base.show(io::IO, ::MIME"text/plain", result::RankERGMResult)
    println(io, "Rank-Order ERGM Results")
    println(io, "=======================")
    println(io, "Reference: CompleteOrder")
    println(io, "Estimation: ", _estimation_label(result))
    if result.method === :mcmle
        if isnan(result.loglik)
            # `bridge_rungs=0`: the path-sampling estimate was skipped on request
            println(io, "Log-likelihood: not estimated (bridge_rungs=0)")
            println(io, "AIC: not estimated, BIC: not estimated")
        else
            println(io, "Log-likelihood: $(round(result.loglik, digits=4)) ",
                        "(path-sampling bridge estimate)")
            println(io, "AIC: $(round(aic(result), digits=2)), ",
                        "BIC: $(round(bic(result), digits=2))")
        end
    elseif isnan(result.loglik)
        # No swap comparison left to evaluate the pseudo-likelihood on
        # (every one changes a statistic dropped at the boundary)
        println(io, "Pseudo-log-likelihood: not defined (no swap comparison left: ",
                    "every one changes a statistic dropped at the boundary of its ",
                    "attainable range)")
        println(io, "Pseudo-AIC: not defined, pseudo-BIC: not defined")
    else
        println(io, "Pseudo-log-likelihood: $(round(result.loglik, digits=4))")
        println(io, "Pseudo-AIC: $(round(aic(result), digits=2)), ",
                    "pseudo-BIC: $(round(bic(result), digits=2))")
    end
    println(io, "Converged: $(result.converged)")
    if !result.converged
        # The caveat sits right under the verdict: an unconverged fit must
        # never look like a fit with a footnote
        println(io, "  ", _nonconvergence_caveat(result))
    end
    println(io, "Std. errors: ",
            result.se_type === :bootstrap ? "parametric bootstrap" :
            result.se_type === :fisher ?
                "inverse Fisher information from the final MCMC sample, plus " *
                "the Monte-Carlo component" :
                "inverse pseudo-Hessian")
    for note in (_fixed_coefficient_note(result), _undefined_se_note(result),
                 _boot_exclusion_note(result))
        note === nothing || println(io, "  ", note)
    end
    println(io)
    # The printed table IS `coeftable(result)` (a Networks.CoefficientTable
    # rendered through the shared `print_coeftable`), so what is shown and
    # what is inspected cannot diverge.
    show(io, coeftable(result))

    if result.method === :mcmle
        # R's `summary.ergm` "MCMC %" column, as ERGM.jl prints it
        shares = _mcmc_percent(result)
        names = [name(t) for t in result.model.terms]
        println(io)
        println(io, "MCMC % of the standard error (100·(se − se_fisher)/se): ",
                join(("$(names[k]) $(shares[k])" for k in eachindex(names)), ", "))
        println(io)
        println(io, "Note: this model was fit by MCMC maximum likelihood (the estimator of R")
        println(io, "ergm.rank). The estimates carry Monte-Carlo error, included in the")
        println(io, "standard errors; the log-likelihood (and AIC/BIC) is a path-sampling")
        println(io, "bridge estimate.")
        return nothing
    end

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
    objective(result::RankERGMResult) -> Symbol

`:likelihood` for a `method=:mcmle` fit — the MCMC maximum likelihood estimate
solves `E_θ[g(Y)] = g(y_obs)` on the complete-ordering space, the estimator of
R `ergm.rank`. `:pseudolikelihood` for a `method=:mple` fit — the **swap**
pseudo-likelihood: the product, over every (ego, unordered alter pair {j, k}),
of the conditional probability of the observed ranking `y` against the single
alternative `y_swapped` (ego's ranks of j and k exchanged), i.e. `P(y | {y,
y_swapped})`, the AlterSwap move taking the role of the edge toggle in
dyadwise MPLE. That product is never the likelihood — not even for a term
that decomposes over egos, such as `RankNodeICov` (see [`fit_ergm_rank`](@ref)).

# Example
```julia
using ERGMRank
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
fit = fit_ergm_rank(as_rank_network(m), [RankDeference()])
objective(fit)                         # :pseudolikelihood
fit_metadata(fit).objective            # the same, through the shared protocol
```
"""
objective(result::RankERGMResult) =
    result.method === :mcmle ? :likelihood : :pseudolikelihood

"""
    is_exact(::RankERGMResult) -> Bool

Always `false`, for one of two reasons. A `method=:mple` fit maximizes the
swap pseudo-likelihood: the swap comparisons overlap — each ranking enters
`n − 2` of the pairwise conditionals — so their product is not the likelihood,
and **no consistency result is claimed** for that estimator (the earlier
"consistent approximation" claim has been withdrawn; see
[`fit_ergm_rank`](@ref)). A `method=:mcmle` fit targets the likelihood, but
it is a **Monte-Carlo estimate** of the MLE: the estimating equation is solved
against a finite MCMC sample (the Monte-Carlo error is included in the
standard errors) and the log-likelihood is a path-sampling bridge estimate,
so the numbers change with the `rng` and the MCMC budget.

# Example
```julia
using ERGMRank
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
fit = fit_ergm_rank(as_rank_network(m), [RankDeference()])
is_exact(fit)                          # false: a swap pseudo-likelihood
approximations(fit)[1]                 # says so in words
```
"""
is_exact(::RankERGMResult) = false

"""
    se_method(result::RankERGMResult) -> Symbol

What the reported standard errors ACTUALLY are: `:hessian` (the inverse negative
Hessian of the swap pseudo-likelihood), `:bootstrap` (the parametric bootstrap
of `fit_ergm_rank(...; se=:bootstrap)`) or `:fisher` (`method=:mcmle`: the
inverse Fisher information from the final MCMC sample plus the Monte-Carlo
component). Read straight off the fit.

# Example
```julia
using ERGMRank, Random
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
rnet = as_rank_network(m)
se_method(fit_ergm_rank(rnet, [RankDeference()]))                    # :hessian
se_method(fit_ergm_rank(rnet, [RankDeference()]; se=:bootstrap,
                        n_boot=10, rng=Xoshiro(1)))                   # :bootstrap
```
"""
se_method(result::RankERGMResult) = result.se_type

# A `RankNetwork` carries complete orderings, not a dyad mask: there is no
# unobserved-tie concept for the estimator to treat one way or the other.
missing_method(::RankERGMResult) = :none

function approximations(result::RankERGMResult)
    result.method === :mcmle && return _approximations_mcmle(result)
    # The point-estimate caveat holds for every swap-MPLE fit, however the
    # standard errors were computed: `se=:bootstrap` replaces the covariance,
    # not θ̂.
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
    # Non-convergence, a dropped boundary statistic, undefined standard errors
    # and excluded bootstrap replicates are part of what the fit actually did,
    # so they are reported here as well as warned about at fit time (never
    # only in a log line).
    result.converged || push!(out, _nonconvergence_caveat(result))
    for note in (_fixed_coefficient_note(result), _undefined_se_note(result),
                 _boot_exclusion_note(result))
        note === nothing || push!(out, note)
    end
    return out
end

# What a `method=:mcmle` fit did NOT do exactly: the likelihood is met through
# an MCMC sample (Monte-Carlo error, included in the standard errors), the
# log-likelihood is a bridge estimate (or was skipped), and non-convergence —
# with the diagnostics of the final sample — is part of the record.
function _approximations_mcmle(result::RankERGMResult)
    out = String[
        "MCMC MLE: the likelihood is approximated by an MCMC sample of the " *
        "AlterSwap chain (n_samples draws per iteration), so the estimates carry " *
        "Monte-Carlo error — included in the standard errors as the V·Σ_mc·V " *
        "component (fit.vcov_fisher is the Fisher part alone; show prints R's " *
        "MCMC %)",
    ]
    if isnan(result.loglik)
        push!(out, "log-likelihood not estimated (bridge_rungs=0): loglikelihood, " *
                   "AIC and BIC are NaN")
    else
        push!(out, "the reported log-likelihood (and AIC/BIC) is a path-sampling " *
                   "bridge estimate along θ_u = u·θ̂ from the uniform ordering " *
                   "model (θ = 0, log Z = n·log((n−1)!)) with $(result.bridge_rungs) " *
                   "rungs — a Monte-Carlo quantity with its own error")
    end
    result.converged || push!(out, _nonconvergence_caveat(result))
    note = _undefined_se_note(result)
    note === nothing || push!(out, note)
    return out
end

# ============================================================================
# The StatsAPI surface: methods on the shared statistics generics, so results
# interoperate with StatsBase/GLM-style tooling (`coef(fit)`, `vcov(fit)`, ...).
# `Networks.check_statsapi(fit; strict=true)` pins all ten verbs in the tests.
# ============================================================================

StatsAPI.coef(result::RankERGMResult) = result.coefficients
StatsAPI.stderror(result::RankERGMResult) = result.std_errors
StatsAPI.vcov(result::RankERGMResult) = result.vcov
StatsAPI.loglikelihood(result::RankERGMResult) = result.loglik
# R's `logLik` df: a coefficient fixed at ∓Inf by a boundary statistic (R's
# drop) is not an estimated parameter
StatsAPI.dof(result::RankERGMResult) = count(isfinite, result.coefficients)

# The ordered dyads of an n-actor ranking, R's `network.dyadcount` of a
# directed network
_n_dyads(rnet::RankNetwork) = rnet.n * (rnet.n - 1)

"""
    nobs(result::RankERGMResult) -> Int

The sample size the estimator's information criteria are computed on, which
differs between the two estimators (a method of `StatsAPI.nobs`):

- `method=:mple` — the number of swap comparisons, `n(n−1)(n−2)/2` (one
  per ego and unordered alter pair): the observations of the swap
  pseudo-likelihood, each a logistic conditional. After a boundary statistic
  was dropped, `result.n_kept` (the comparisons the dropped statistics do
  not change) is what `bic` uses, as R's `logLik` `nobs` attribute does.
- `method=:mcmle` — the number of ordered dyads, `n(n−1)`: R `ergm.rank`'s
  convention (`nobs.ergm` is `network.dyadcount`), so `bic` of an MCMLE fit
  is on R's scale (`log(272)` on Newcomb's 17 actors, not `log(2040)`).
  `result.n_kept` equals it.

# Example
```julia
using ERGMRank, Random
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
rnet = as_rank_network(m)
nobs(fit_ergm_rank(rnet, [RankDeference()])) == 12              # true: 4·3·2/2 swap comparisons
mle = fit_ergm_rank(rnet, [RankNodeICov([1, 2, 3, 4])]; method=:mcmle,
                    n_samples=64, bridge_rungs=0, rng=Xoshiro(1))
nobs(mle) == 12                                                 # true: 4·3 dyads (R's nobs)
```
"""
StatsAPI.nobs(result::RankERGMResult) =
    result.method === :mcmle ? _n_dyads(result.model.network) :
                               _n_comparisons(result.model.network)

# One docstring for the pair (both exported; `@doc` binds it to each method)
const _AIC_BIC_DOC = """
    aic(result::RankERGMResult) -> Float64
    bic(result::RankERGMResult) -> Float64

`-2 loglikelihood + 2 dof` and `-2 loglikelihood + dof · log(n_kept)`, where
`dof` counts the finite coefficients and `n_kept` is the sample size of the
estimator — [`nobs`](@ref)`(result)`: the swap comparisons of a `:mple` fit
(or fewer after a boundary statistic was dropped, R's `logLik` `nobs`
attribute) and the `n(n−1)` ordered dyads of a `:mcmle` fit (R's
`nobs.ergm`).

For a `method=:mcmle` fit `loglikelihood` is the path-sampling (bridge)
estimate of the **absolute** log-likelihood `θ̂'g(y) − log Z(θ̂)`, with its own
Monte-Carlo error, so these are the AIC/BIC of the MCMC MLE; they are `NaN`
when the fit was run with `bridge_rungs=0` (the estimate skipped on request,
recorded in `approximations`).

!!! note "R's `logLik` is relative to the uniform-ordering model"
    R `ergm.rank`'s `logLik(fit)` is **not** the absolute log-likelihood: for
    a valued ERGM `ergm` defines the null (θ = 0) model's likelihood as 0
    ("Null model likelihood calculation is not implemented for valued ERGMs"),
    so `logLik(fit)` is the bridge-sampled ratio `log L(θ̂) − log L(0)`.
    ERGMRank reports the absolute value, using the exact normalizer of the
    uniform ordering model, `log Z(0) = n·log((n−1)!)`. The two are related
    by a constant of the ranking size alone:

    `R's logLik = loglikelihood(fit) + n·log((n−1)!)` and R's `AIC`/`BIC` are
    `aic(fit)`/`bic(fit)` shifted by `−2n·log((n−1)!)` (`log(1296) ≈ 7.17` on
    4 actors; `17·log(16!) ≈ 521` on Newcomb's 17, so AIC/BIC differ by
    ≈ 1042 there). Differences between models fitted to the same ranking —
    the only comparison either convention licenses — are unaffected. The
    golden fixture asserts the identity against `ergm.rank`'s `logLik`
    within both bridges' Monte-Carlo error.

For a `method=:mple` fit they are the **pseudo**-AIC and **pseudo**-BIC of the
swap pseudo-likelihood — the product of overlapping swap conditionals, not
the likelihood — so they rank swap-MPLE fits against each other on the same
network and nothing else, and are **not comparable** to the `:mcmle` values
or to R's. When no swap comparison is left to estimate on (every one changes
a statistic dropped at the boundary of its attainable range, `n_kept == 0`)
the pseudo-log-likelihood is undefined and all three are `NaN`.

# Example
```julia
using ERGMRank
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
fit = fit_ergm_rank(as_rank_network(m), [RankDeference()])
aic(fit) == -2 * loglikelihood(fit) + 2 * dof(fit)              # true
bic(fit) == -2 * loglikelihood(fit) + dof(fit) * log(nobs(fit))  # true
ERGMRank._log_n_orderings(4) ≈ 4 * log(6)    # true: R's logLik − loglikelihood(mle) on 4 actors
```
"""

@doc _AIC_BIC_DOC StatsAPI.aic(result::RankERGMResult) = -2 * result.loglik + 2 * dof(result)

@doc _AIC_BIC_DOC function StatsAPI.bic(result::RankERGMResult)
    k = dof(result)
    return k == 0 ? -2 * result.loglik + 0.0 : -2 * result.loglik + k * log(result.n_kept)
end

"""
    confint(result::RankERGMResult; level=0.95) -> Matrix{Float64}

Normal-theory (Wald) confidence limits `θ̂ ± z_{(1+level)/2} · se`, one row
per coefficient with the lower limit in column 1 and the upper in column 2
(a method of `StatsAPI.confint`). The standard errors are the ones the fit
reports — `se_method(result)` says whether they are the inverse
pseudo-Hessian (expected anticonservative, so these intervals are
over-narrow) or the parametric bootstrap of `se=:bootstrap`.

# Example
```julia
using ERGMRank
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
fit = fit_ergm_rank(as_rank_network(m), [RankDeference()])
ci = confint(fit)                             # 1×2
ci[1, 1] < coef(fit)[1] < ci[1, 2]            # true
size(confint(fit; level=0.9)) == (1, 2)       # true
```
"""
function StatsAPI.confint(result::RankERGMResult; level::Real=0.95)
    0 < level < 1 || throw(ArgumentError("confint: level must be in (0, 1) (got $level)"))
    q = quantile(Normal(), 1 - (1 - level) / 2)
    θ, se = result.coefficients, result.std_errors
    return hcat(θ .- q .* se, θ .+ q .* se)
end

"""
    coeftable(result::RankERGMResult) -> Networks.CoefficientTable

The R-style coefficient table (`Estimate`, `Std.Error`, `z value`,
`Pr(>|z|)`) as an inspectable `Networks.CoefficientTable` — exactly the table
`show(result)` prints, built from the same vectors (a method of
`StatsAPI.coeftable`). The p-values come from `Networks.z_pvalues` (two-sided
normal, floored at `floatmin(Float64)` so a finite z never prints as `0.0`); a
coefficient fixed at ∓Inf by a boundary statistic has z = ∓Inf and p = 0, as R
prints it. Rows can be read by index or by term name.

# Example
```julia
using ERGMRank
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
fit = fit_ergm_rank(as_rank_network(m), [RankDeference(), RankNodeICov([10, 20, 30, 40])])
fit.converged                                       # true
tbl = coeftable(fit)
tbl["rank.deference"].estimate == coef(fit)[1]     # true
tbl[2].std_error == stderror(fit)[2]                # true
```
"""
function StatsAPI.coeftable(result::RankERGMResult)
    z, p = _z_and_p(result)
    return CoefficientTable([name(t) for t in result.model.terms],
                            result.coefficients, result.std_errors;
                            z_values=z, p_values=p)
end

# One message for the "too few actors" refusal, shared by the fitter and the
# sampler: with fewer than 3 actors no ego has two alters to compare, so there
# is no swap and no pseudo-likelihood contribution at all.
_too_few_actors(context::AbstractString, n::Int) =
    "$context: a rank network needs at least 3 actors (an ego must have two " *
    "alters to compare — the swap comparisons are the observations); got n = $n"

# The rank terms this package knows, for the message below
const _RANK_TERM_NAMES = "RankDeference, RankNonconformity, RankNodeICov, " *
                         "RankInconsistency, RankEdgeCov"

# A term is a rank term iff it implements the term contract — `compute(term,
# rnet)` AND `swap_change(term, rnet, ego, j, k)`. ERGM.jl's binary terms
# (`Edges()`, `Mutual()`, ...) subtype `AbstractERGMTerm` too but have no
# rank statistic; mixing one into a rank model used to surface as a raw
# `MethodError` from deep inside the design builder. Checked once, up front,
# by the fitter and the sampler (`gof` runs on a fitted model's terms).
_is_rank_term(t) = hasmethod(compute, Tuple{typeof(t), RankNetwork}) &&
                   hasmethod(swap_change, Tuple{typeof(t), RankNetwork, Int, Int, Int})

function _check_rank_terms(context::AbstractString, terms)
    for t in terms
        _is_rank_term(t) && continue
        what = t isa AbstractERGMTerm ? "$(nameof(typeof(t))) is not a rank term" :
               "$(repr(t)) ($(typeof(t))) is not a term"
        throw(ArgumentError(
            "$context: $what; a rank model takes $_RANK_TERM_NAMES (a term must " *
            "implement compute(term, ::RankNetwork) and swap_change) — ERGM.jl's " *
            "binary terms such as Edges() or Mutual() have no rank statistic. " *
            "ergm.rank's `rank.` terms map to those five (README: Terms)."))
    end
    return nothing
end

# The arguments an `ergm.rank` user reaches for that are not a `RankNetwork`
# and a term VECTOR: one term, a tuple of terms, a `Network`, a matrix. Each
# gets a sentence naming the fix instead of a bare `MethodError`.
_not_a_rank_network(context::AbstractString, x) =
    ArgumentError("$context: the first argument is a $(typeof(x)), not a " *
                  "RankNetwork; call as_rank_network(...) first — " *
                  "`as_rank_network(net; attr=:rank)` for a directed Network whose " *
                  "arcs carry the ranks (R's as.matrix(nw, attrname=\"rank\")), " *
                  "`as_rank_network(m)` for a rank matrix (row i = ego i's ranks, " *
                  "greater = higher standing)")

"""
    fit_ergm_rank(rnet::RankNetwork, terms; method=:mple, kwargs...) -> RankERGMResult

Fit a rank-order ERGM `P(y) ∝ exp(θ'g(y))` on the complete-ordering space
(the `CompleteOrderReference`), by one of two estimators:

- **`method=:mple`** (default) — **swap-based maximum pseudo-likelihood**:
  for each ego `i` and each unordered alter pair {j, k}, the conditional
  probability of the observed ranking `y` against the single alternative
  `y_swapped` (ego's ranks of j and k exchanged) — `P(y | {y, y_swapped})`,
  the two-state conditional, *not* "the order of j and k given the rest of
  the rankings": a non-adjacent swap also reorders j and k relative to the
  alters ranked between them — is logistic in `θ'[g(y) − g(y_swapped)]`,
  and the product of these conditionals is maximized by Newton-Raphson
  with step-halving. Fast and deterministic; a *different estimator* from
  the MLE (see the warning below).
- **`method=:mcmle`** — the **MCMC maximum likelihood estimate**, the
  estimator of R `ergm.rank`: starting from the swap-MPLE (or `init`), each
  iteration samples the model statistics along an AlterSwap Metropolis chain
  at the current θ and takes a Hummel-style partial Newton step toward the
  observed statistics; convergence is declared when the sampled statistics
  are statistically indistinguishable from the observed ones (per-statistic
  t-ratios and a Hotelling T² test, `ERGM.mcmc_convergence`). Standard errors
  are the inverse Fisher information of the final sample plus the
  Monte-Carlo component; the log-likelihood is a path-sampling bridge
  estimate. See the "MCMC MLE" section below. On Newcomb's fraternity ranks
  it reproduces `ergm.rank`'s coefficients within the Monte-Carlo spread of
  the two implementations (asserted by the provenanced golden fixture).

`ergm.rank`'s `rank.deference + rank.nonconformity("all")` on Newcomb week 1
gives the MCMLE `[-0.1531, -0.00659]`; the swap-MPLE of the same model is
`[-0.1409, -0.00585]` — 16× and 13× R's own seed-to-seed spread apart, i.e.
a systematic (not Monte-Carlo) difference, though only 0.30 and 0.43 of an
MLE standard error. Use `method=:mcmle` when the MLE itself is the target.

The swap-MPLE is the natural rank analogue of dyadwise MPLE (the AlterSwap
move takes the role of the edge toggle). It is fast, and it is *not* the
MCMC MLE that `ergm.rank` computes: it maximizes a pseudo-likelihood formed
by multiplying pairwise-swap conditionals that are not independent, so the
two estimators generally disagree.

!!! warning "What is and is not claimed for the swap-MPLE"
    **No consistency result is established here.** Earlier versions of this
    docstring called swap-MPLE a "consistent approximation" to the MCMC MLE;
    that claim was unqualified by any asymptotic regime or assumptions and
    has been withdrawn. **The swap pseudo-likelihood never equals the
    likelihood** — there is no rank analogue of dyad independence under
    which the product of two-state swap conditionals is the likelihood: the
    comparisons within one ego's row must form a total order, so they are
    never independent, even for a term that decomposes over egos. On the
    4-actor example network with `RankNodeICov([1, 2, 3, 4])`, exact
    enumeration of all 1296 orderings gives the MLE `θ = −0.0767`
    (log-likelihood `−7.015`) while the swap-MPLE is `−0.0880`
    (pseudo-log-likelihood `−8.057`); `method=:mcmle` run to a tight
    convergence threshold lands at `−0.0772`, the swap-MPLE never moves
    (pinned by the test suite). The estimator's
    large-sample behaviour in this setting has not been characterized, and
    MPLE for dependent ERGMs is known to be biased in finite samples.

    **The default standard errors are the inverse observed pseudo-Hessian** —
    the curvature of the pseudo-log-likelihood, which treats the overlapping
    swap comparisons as independent. They therefore ignore the dependence
    between comparisons and are expected to be **anticonservative** (too
    small, giving over-narrow intervals and anti-conservative tests) under
    dependence. Treat them as a rough guide, not calibrated inference — or
    pass `se=:bootstrap` (below), which does not make that assumption.

The swap pseudo-likelihood is a logistic likelihood on the swap-difference
rows with the response identically `true`, so it is maximized with the
ecosystem's shared `Networks.logistic_derivatives` kernel and
`Networks.newton_fit` Newton–Raphson-with-step-halving optimizer (the same
bindings ERGM.jl's MPLE runs on).

[`ergm_rank`](@ref) is the R-faithful alias (matching the `ergm.rank`
package); [`fit_rank_ergm`](@ref) is a legacy alias.

# Standard errors

- `se=:hessian` (default) — the inverse negative pseudo-Hessian; see the warning
  above. When the pseudo-Hessian at the solution is not negative definite (a
  coefficient the swap comparisons do not identify) the standard errors are
  `NaN`, with a warning from `newton_fit` and an entry in
  `approximations` — never a finite number from an indefinite matrix.
- `se=:bootstrap` — parametric bootstrap: simulate `n_boot` rank networks from
  the fitted model at θ̂ with [`simulate_rank_ergm`](@ref) (AlterSwap
  Metropolis), refit the swap MPLE on each, and report the empirical covariance
  of the refits. The point estimates are unchanged; only the covariance is
  replaced. This is the same option, with the same keywords and the same
  semantics, as `ERGM.mple`'s, and it runs on the ONE shared
  `Networks.bootstrap_cov` loop. A replicate on which the swap MPLE does not
  exist (a statistic at the boundary of its attainable range in the *simulated*
  ranking) is **excluded** from the covariance: it is a `NaN` row of
  `fit.boot_replicates`, the exclusion is warned about once and recorded in
  `approximations(fit)`, and fewer than 2 finite refits is an `ArgumentError`.
  `se=:bootstrap` is refused (an `ArgumentError`) when a coefficient of the
  fit itself is fixed at ±Inf, because no ranking can be simulated at an
  infinite coefficient.

# Boundary statistics (R's `drop`)

A statistic whose observed value no single swap can lower (or raise) has no
finite swap-MPLE: the pseudo-log-likelihood increases monotonically as its
coefficient goes to `-Inf` (or `+Inf`). `RankInconsistency(rnet)` fitted to
`rnet` itself is the textbook case (the observed ranking is *at* zero
inconsistency). As R `ergm` does under its default `drop=TRUE`, such a
coefficient is **fixed at ∓Inf with standard error 0** (z = ∓Inf, p = 0), a
warning quotes R's sentence ("observed statistic(s) … are at their smallest
attainable values"), and the remaining coefficients are estimated on the swap
comparisons the dropped statistics do not change — the exact limit of the
pseudo-likelihood. `dof(fit)` counts only the finite coefficients and
`approximations` records the drop. A separated statistic used to
"converge" silently to a large finite value with a meaningless standard error.

# Non-convergence

A swap-MPLE fit that exhausts `maxiter` is returned with `converged == false`,
a warning at fit time, a caveat printed directly under `Converged: false`, and
an entry in `approximations`: the coefficients are the last Newton iterate and
the standard errors are unreliable.

# MCMC MLE (`method=:mcmle`)

The estimator of R `ergm.rank`. Each iteration draws `n_samples` statistics
vectors along the AlterSwap Metropolis chain at the current θ (the ONE
`ERGM.mh_toggle!` kernel, the running statistics kept current from the
accepted swaps' change statistics — never a recomputation per draw), then
takes the Hummel partial Newton step `θ += γ · Σ̂⁻¹ (g_obs − ḡ)`: the step
length `γ` starts at `gamma0` and grows (at most doubling per iteration) while
the observed statistics lie outside the sampled cloud (the 95 % Mahalanobis
radius), reaching 1 once the cloud covers them, and each step is capped at
Euclidean norm `max_step_norm`. Convergence is declared only at `γ = 1` and
when both `ERGM.mcmc_convergence` tests pass: every per-statistic t-ratio
`|g_obs − ḡ| / sd(g)` below `conv_threshold` and a Hotelling T² test of the
mean difference (with an autocorrelation-adjusted, Geyer effective sample
size) non-significant at `hotelling_alpha`. The final sample at the returned
θ̂ gives `fit.mcmc_convergence` (the same report, recomputed) and the
standard errors.

**Non-convergence is loud**: a fit that exhausts `maxiter` MCMLE iterations
warns with the last max t-ratio, Hotelling p-value and step length,
`approximations(fit)` lists the caveat with those numbers, and `show` prints
it under `Converged: false`. Continue such a fit with `init=coef(fit)`.

**Standard errors** (`se=:fisher`, the only option under `:mcmle`) are
`vcov = V + V·Σ_mc·V` with `V = Σ̂⁻¹` the inverse covariance of the final
sample (the Fisher information) and `Σ_mc` the Geyer initial-sequence
covariance of the sampled mean — the Monte-Carlo component of the estimating
equation (Hunter & Handcock 2006 §3.3), exactly as `ERGM.mcmle` reports and
as R's `summary.ergm` "MCMC %" column quotes. `fit.vcov_fisher` is `V` alone.

**Log-likelihood.** `loglik = θ̂'g_obs − log Z(θ̂)` with `log Z(θ̂) = log Z(0)
+ ∫₀¹ E_{uθ̂}[g]'θ̂ du` — path sampling along `θ_u = u·θ̂` from the uniform
ordering model, whose normalizer is exact, `log Z(0) = n · log((n − 1)!)`.
The integral is the trapezoid rule over `bridge_rungs` segments, each grid
point one seeded chain of `bridge_samples` draws (the `ERGM._bridge_logZ`
estimator, with the uniform ranking model as the exact reference). It is
run only after the final sample, so coefficients and standard errors are
bit-identical with and without it; **`bridge_rungs=0` skips it** and
`loglik`, `aic`, `bic` are `NaN`, `show` prints "not estimated" and
`approximations` records it.

**Boundary and separated starts.** A statistic at the boundary of its
attainable range under single swaps (the swap-MPLE start would be `∓Inf`)
has no swap-MPLE to start from: refused with an `ArgumentError` unless
`init=` is supplied, in which case a warning says the MLE may not exist and
the convergence tests decide. A separated swap-MPLE (R: "The MPLE does not
exist!") is refused the same way, pointing at `init=`.

**Reproducibility.** All randomness flows through `rng`; with `n_chains > 1`
the `n_samples` draws are split over chains seeded from `rng` in order and
run on separate tasks, so a fit depends only on `rng` and `n_chains`, never
on the thread count (`n_chains` never defaults to `Threads.nthreads()`).

# Keyword Arguments

Common:
- `method::Symbol=:mple`: `:mple` or `:mcmle` (anything else is an
  `ArgumentError` naming both)
- `maxiter::Int`: the iteration cap — Newton iterations of the swap-MPLE
  (default 100) or MCMLE iterations (default 20)
- `se::Symbol`: `:hessian` (default) or `:bootstrap` under `:mple`; `:fisher`
  (default, and the only option) under `:mcmle`; anything else is an
  `ArgumentError` from the shared `Networks.check_se` naming the method's
  vocabulary
- `rng::AbstractRNG=Random.default_rng()`: source of all randomness (the
  bootstrap, the MCMC chains, the bridge) — a fixed `rng` reproduces the fit
  exactly, on any thread count
- `verbose::Bool=false`: print MCMLE progress

Swap-MPLE (`method=:mple`):
- `tol::Float64=1e-8` (passed to `newton_fit`)
- `n_boot::Int=100`: number of bootstrap replicates (`se=:bootstrap` only;
  at least 2)
- `boot_burnin=nothing`, `boot_interval=nothing`: MCMC controls for the
  bootstrap simulations; `nothing` resolves to the swap-scaled defaults of
  [`simulate_rank_ergm`](@ref) — `burnin = 20 n_swaps`, `interval = max(100,
  n_swaps ÷ 10)` with `n_swaps = n (n − 1)(n − 2) / 2` (ERGM.jl's dyad-scaled
  rule applied to the swap count) — and an explicit integer is honoured as given

MCMC MLE (`method=:mcmle`; the vocabulary of `ERGM.mcmle`):
- `n_samples::Int=1024`: MCMC draws per iteration, in total over the chains
- `burnin=nothing`, `interval=nothing`: steps discarded per chain and steps
  between draws; `nothing` resolves to the swap-scaled rule above
- `n_chains::Int=1`: independent chains per iteration (and for the final
  sample), each burned in from the observed ranking, seeded from `rng`
- `init=nothing`: starting coefficients (default: the swap-MPLE)
- `gamma0::Float64=0.1`: initial Hummel step length
- `max_step_norm::Float64=5.0`: cap on the norm of each Newton step
- `conv_threshold::Float64=0.1`, `hotelling_alpha::Float64=0.05`: the
  convergence tests
- `bridge_rungs::Int=16`, `bridge_samples=nothing` (default `n_samples`):
  the log-likelihood bridge; `bridge_rungs=0` skips it

# Errors
An empty term list, a network with fewer than 3 actors (no ego has two alters
to compare) and an invalid ranking are refused with an `ArgumentError` that
says so. So are a term that is not a rank term — ERGM.jl's binary `Edges()`,
`Mutual()`, … have no rank statistic; the message lists the five rank terms
— and a `Network` or a rank matrix in place of the `RankNetwork` (the
message says to call [`as_rank_network`](@ref) first). A single term, or a
tuple of terms, is accepted in place of the vector:
`fit_ergm_rank(rnet, RankDeference())`.

# Example
```julia
using ERGMRank, Random
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
rnet = as_rank_network(m)
# (RankDeference() + RankNonconformity() on these 4 actors is perfectly
# separated — the swap-MPLE does not exist, and the fit says so; see below)
fit = fit_ergm_rank(rnet, [RankDeference(), RankNodeICov([10, 20, 30, 40])])
fit.converged                     # true
round.(coef(fit); digits=4)       # [-0.0658, -0.0082]: two swap-MPLE coefficients
coeftable(fit)                    # the table `show(fit)` prints
approximations(fit)               # what the estimator did NOT do

x = [1.0, 2.0, 3.0, 4.0]
mle = fit_ergm_rank(rnet, [RankNodeICov(x)]; method=:mcmle, n_samples=512,
                    rng=Xoshiro(1))
mle.method                        # :mcmle
mle.mcmc_convergence.step_length  # 1.0 at convergence
isfinite(loglikelihood(mle))      # true: the bridge estimate (bridge_rungs=16)
```
"""
function fit_ergm_rank(rnet::RankNetwork, terms::Vector{<:AbstractERGMTerm};
                       method::Symbol=:mple,
                       maxiter::Union{Nothing,Int}=nothing, tol::Float64=1e-8,
                       se::Union{Nothing,Symbol}=nothing,
                       n_boot::Int=100,
                       boot_burnin::Union{Nothing,Int}=nothing,
                       boot_interval::Union{Nothing,Int}=nothing,
                       n_samples::Int=1024,
                       burnin::Union{Nothing,Int}=nothing,
                       interval::Union{Nothing,Int}=nothing,
                       n_chains::Int=1,
                       init::Union{Nothing,AbstractVector{<:Real}}=nothing,
                       gamma0::Float64=0.1,
                       max_step_norm::Float64=5.0,
                       conv_threshold::Float64=0.1,
                       hotelling_alpha::Float64=0.05,
                       bridge_rungs::Int=16,
                       bridge_samples::Union{Nothing,Int}=nothing,
                       verbose::Bool=false,
                       rng::Random.AbstractRNG=Random.default_rng())
    method in (:mple, :mcmle) || throw(ArgumentError(
        "fit_ergm_rank: method must be one of (:mple, :mcmle) (got $(repr(method))): " *
        ":mple is the swap pseudo-likelihood, :mcmle the MCMC maximum likelihood " *
        "estimate of R ergm.rank"))
    # `se` resolves per method, and is validated against that method's
    # vocabulary by the ONE shared validator
    if method === :mcmle
        se = check_se(something(se, :fisher), (:fisher,); context="fit_ergm_rank(method=:mcmle)")
        maxiter = something(maxiter, 20)
    else
        se = check_se(something(se, :hessian), (:hessian, :bootstrap); context="fit_ergm_rank")
        maxiter = something(maxiter, 100)
    end
    isempty(terms) &&
        throw(ArgumentError("fit_ergm_rank: the model has no terms; pass at least " *
                            "one rank term, e.g. [RankDeference()]"))
    _check_rank_terms("fit_ergm_rank", terms)
    rnet.n >= 3 || throw(ArgumentError(_too_few_actors("fit_ergm_rank", rnet.n)))
    is_valid_ranking(rnet) ||
        throw(ArgumentError("fit_ergm_rank: network is not a valid complete ranking: " *
                            something(_ranking_violation(rnet.ranks), "")))

    model = RankERGMModel(collect(AbstractERGMTerm, terms), copy(rnet),
                          CompleteOrderReference())

    if method === :mcmle
        return _rank_mcmle(model; n_samples=n_samples, burnin=burnin,
                           interval=interval, maxiter=maxiter, n_chains=n_chains,
                           init=init, gamma0=gamma0, max_step_norm=max_step_norm,
                           conv_threshold=conv_threshold,
                           hotelling_alpha=hotelling_alpha,
                           bridge_rungs=bridge_rungs, bridge_samples=bridge_samples,
                           verbose=verbose, rng=rng)
    end

    fit = _rank_mple_fit(model.terms, rnet; maxiter=maxiter, tol=tol)
    if any(isnan, fit.θ)
        nan = [name(t) for (t, c) in zip(model.terms, fit.θ) if isnan(c)]
        @warn "fit_ergm_rank: coefficient(s) $(join(nan, ", ")) are not identified " *
              "and are returned as NaN with converged == false: every swap " *
              "comparison changes a statistic that sits at the boundary of its " *
              "attainable range (dropped, coefficient fixed at ±Inf), so no " *
              "comparison is left to estimate them on. Remove the boundary " *
              "statistic from the model."
    elseif fit.separated
        _warn_separated_rank()
    elseif !fit.converged
        @warn "fit_ergm_rank: the swap-MPLE Newton iteration did not converge in " *
              "maxiter=$maxiter (|gradient| = $(round(fit.grad_norm; sigdigits=3))); " *
              "the coefficients are the last iterate and the standard errors are " *
              "unreliable — increase maxiter, or check the model for a statistic " *
              "at the boundary of its attainable range"
    end

    vcov, std_errors = fit.vcov, fit.se
    boot_replicates = nothing
    if se === :bootstrap
        # A coefficient fixed at ∓Inf cannot be simulated from (the sampler's
        # θ'Δ would be NaN wherever the dropped statistic changes), so there
        # is nothing to bootstrap: say so instead of returning NaN errors.
        if any(isinf, fit.θ)
            fixed = [name(t) for (t, c) in zip(model.terms, fit.θ) if isinf(c)]
            throw(ArgumentError(
                "fit_ergm_rank: se=:bootstrap is not available when a coefficient " *
                "is fixed at ±Inf by a statistic at the boundary of its attainable " *
                "range ($(join(fixed, ", "))): a ranking cannot be simulated at an " *
                "infinite coefficient. Remove the term (as R's drop=TRUE does) or " *
                "keep the default se=:hessian, which reports standard error 0 for " *
                "the fixed coefficient and the inverse-Hessian errors of the rest."))
        end
        burnin, interval = _resolve_mcmc_controls(rnet, boot_burnin, boot_interval)
        vcov, std_errors, boot_replicates =
            _rank_bootstrap_cov(model, fit.θ; n_boot=n_boot,
                                boot_burnin=burnin,
                                boot_interval=interval,
                                maxiter=maxiter, tol=tol, rng=rng)
    end

    return RankERGMResult(model, fit.θ, std_errors, vcov, fit.loglik,
                          fit.converged, se, boot_replicates, fit.n_kept)
end

# One term, or a tuple of terms, instead of a vector: wrapped. A `Network` or
# a matrix instead of a `RankNetwork`: refused with the conversion named.
fit_ergm_rank(rnet::RankNetwork, term::AbstractERGMTerm; kwargs...) =
    fit_ergm_rank(rnet, AbstractERGMTerm[term]; kwargs...)
fit_ergm_rank(rnet::RankNetwork, terms::Tuple; kwargs...) =
    fit_ergm_rank(rnet, collect(AbstractERGMTerm, terms); kwargs...)
fit_ergm_rank(x::Union{Network, AbstractMatrix}, terms; kwargs...) =
    throw(_not_a_rank_network("fit_ergm_rank", x))

# Core swap MPLE: the design rows d = g(y) − g(y with j,k swapped) over every
# (ego, unordered alter pair), maximized by the shared Newton optimizer. Shared
# by `fit_ergm_rank` and by the parametric bootstrap's refits (`warn=false`:
# a boundary statistic in a SIMULATED replicate is not a fact about the user's
# data, and `_rank_bootstrap_cov` reports those replicates once, in aggregate).
#
# Returns `(θ, se, vcov, loglik, converged, separated, grad_norm, n_kept)`:
# `separated` says the pseudo-likelihood has no finite maximum (then
# `converged` is false), `grad_norm` is ‖∇ℓ(θ)‖ at the returned iterate
# (quoted by the non-convergence warning) and `n_kept` the number of
# comparisons the finite coefficients were fitted on (every row, or the rows
# the dropped columns do not touch).
function _rank_mple_fit(terms::Vector{AbstractERGMTerm}, rnet::RankNetwork;
                        maxiter::Int=100, tol::Float64=1e-8, warn::Bool=true)
    p = length(terms)
    D = _rank_design(terms, copy(rnet))
    m = size(D, 1)

    # The response is identically TRUE, so a column whose nonzero swap
    # differences all share one sign is a statistic at the boundary of its
    # attainable range: the observed ranking minimizes (`:min`, all d < 0) or
    # maximizes (`:max`, all d > 0) it over every single swap, the gradient
    # keeps one sign at every θ and no finite MPLE exists. R's drop semantics
    # — ∓Inf, SE 0, the rest fitted on the untouched rows — through the same
    # `public` helpers `ERGM.mple` uses; the rows here are swap comparisons,
    # not dyads, and ergm.rank has no drop of its own, so the sentence's two
    # variable clauses say so.
    ones_m = ones(m)
    boundary = ERGM._boundary_columns_iterated(D, ones_m, ones_m)
    if !isempty(boundary)
        warn && ERGM._warn_boundary([name(t) for t in terms], boundary;
                                    context="fit_ergm_rank", noun="swap comparisons",
                                    note="R ergm reports the same for its binary terms; " *
                                         "ergm.rank's swap pseudo-likelihood has no drop")
        return _rank_mple_fit_dropped(D, boundary, p; maxiter=maxiter, tol=tol)
    end

    # The swap pseudo-likelihood IS a logistic likelihood on the D rows with the
    # response identically TRUE — the observed order is always the "success" —
    # so the derivatives come from the shared `Networks.logistic_derivatives`
    # (review finding 15): gemv/gemm over the whole design (η = Dβ, ∇ = D'(1−p),
    # −H = D'WD), not a per-comparison `d * d'` outer product allocating a p×p
    # matrix on every one of the size(D, 1) rows of every Newton evaluation.
    # Never paste the loop back in; ERGMMulti and TERGM run on the same one.
    derivatives = logistic_derivatives(D, trues(m))
    fit = newton_fit(derivatives, zeros(p); maxiter=maxiter, tol=tol)
    # `newton_fit`'s verdict is scale-free (the Newton decrement beside the
    # gradient norm), so a rounding-level stall at the maximum of this
    # large-Hessian design is converged without a local shim.
    converged = fit.converged
    # A design on which the pseudo-likelihood has no finite maximum for a
    # reason the column test cannot see — a COMBINATION of statistics that no
    # single swap lowers (quasi-complete separation; the 4-actor test network
    # under deference + nonconformity is one) — is returned with
    # `converged = false` instead of the point where Newton met its objective
    # tolerance on the flat asymptote (θ ≈ 10 with a standard error of 7,000).
    # The predicate is ERGM.mple's (`ERGM._separated`, `public`).
    separated = ERGM._separated(derivatives, D, ones_m, ones_m, fit.θ, fit.se)
    return (θ=fit.θ, se=fit.se, vcov=fit.vcov, loglik=fit.loglik,
            converged=converged && !separated, separated=separated,
            grad_norm=_grad_norm(derivatives, fit.θ), n_kept=m)
end

_grad_norm(derivatives, θ) = norm(derivatives(θ)[2])

# R's sentence (mple.existence) for the separated case, in rank terms
function _warn_separated_rank()
    @warn "fit_ergm_rank: the swap-MPLE does not exist (perfect separation): the " *
          "swap pseudo-likelihood has no finite maximum — some combination of " *
          "the model's statistics is never lowered by any single swap of the " *
          "observed ranking — and the returned coefficients are the point at " *
          "which Newton stopped on its flat asymptote: arbitrarily large, with " *
          "meaningless standard errors. R ergm warns \"The MPLE does not exist!\" " *
          "for the same design. The fit is returned with `converged == false`; " *
          "remove a term, or collect more actors (the 4-actor example network " *
          "separates under two terms)."
    return nothing
end

# The reduced fit behind a boundary statistic (see `_rank_mple_fit`): columns
# in `boundary` fixed at ∓Inf, the rest fit on the rows where every dropped
# column is zero, then padded back to the full parameter vector — the exact
# limit of the pseudo-likelihood as the dropped coefficients go to ∓Inf (their
# rows' probabilities go to 1 and contribute nothing).
function _rank_mple_fit_dropped(D::Matrix{Float64}, boundary::Vector{Tuple{Int,Symbol}},
                                p::Int; maxiter::Int, tol::Float64)
    dropped = Dict(boundary)
    keep_cols = [j for j in 1:p if !haskey(dropped, j)]
    keep_rows = [r for r in 1:size(D, 1) if all(D[r, j] == 0 for j in keys(dropped))]

    θ = zeros(p); se = zeros(p); V = zeros(p, p)
    # No comparison left (every one changes a dropped statistic): the swap
    # pseudo-likelihood has nothing to be evaluated on, so its maximum is
    # undefined — NaN, never the 0.0 of an empty product, which would print
    # as "pseudo-AIC: 0.0" (a perfect fit) beside a NaN coefficient. When
    # every column is dropped but some comparisons change no statistic
    # (keep_rows nonempty, keep_cols empty), each of those rows has
    # probability σ(0) = ½, so the exact limit of the pseudo-likelihood is
    # n_kept·log(½) — what `logistic_derivatives` over those rows reports
    # too — never the 0.0 of an empty product.
    n_kept = length(keep_rows)
    loglik = n_kept == 0 ? NaN : n_kept * log(0.5)
    converged = true
    separated = false
    grad_norm = 0.0
    if !isempty(keep_cols) && isempty(keep_rows)
        # Every comparison changes a dropped statistic: nothing is left to
        # estimate the other coefficients on. NaN, not a Newton "solution" on
        # an empty design (whose zero Hessian would give converged = false and
        # a NaN standard error with no explanation).
        θ[keep_cols] .= NaN
        se[keep_cols] .= NaN
        V[keep_cols, :] .= NaN
        V[:, keep_cols] .= NaN
        converged = false
        grad_norm = NaN
    elseif !isempty(keep_cols)
        Dr = D[keep_rows, keep_cols]
        ones_r = ones(length(keep_rows))
        derivatives = logistic_derivatives(Dr, trues(size(Dr, 1)))
        fit = newton_fit(derivatives, zeros(length(keep_cols)); maxiter=maxiter, tol=tol)
        θ[keep_cols] = fit.θ
        se[keep_cols] = fit.se
        V[keep_cols, keep_cols] = fit.vcov
        loglik = fit.loglik
        separated = ERGM._separated(derivatives, Dr, ones_r, ones_r, fit.θ, fit.se)
        converged = fit.converged && !separated
        grad_norm = _grad_norm(derivatives, fit.θ)
    end
    for (j, side) in boundary
        θ[j] = side === :min ? -Inf : Inf
    end
    return (θ=θ, se=se, vcov=V, loglik=loglik, converged=converged,
            separated=separated, grad_norm=grad_norm, n_kept=n_kept)
end

# The swap design: for each (ego, unordered alter pair {j,k}) the difference
# d = g(y_observed) − g(y_swapped), as ONE dense (comparisons × p) matrix — the
# derivatives are BLAS over the whole thing, so a vector-of-vectors would just be
# a scatter to copy out of. Rows run ego-major, alter pairs in index order.
#
# `Tuple(terms)` is a function barrier: the loop below specializes on the
# tuple's type, so each row is one statically dispatched `_swap_delta!`
# (0 bytes, O(n)–O(n²) per row from the per-term `swap_change`); the whole
# builder allocates the matrix, the delta workspace and nothing per row.
function _rank_design(terms::Vector{AbstractERGMTerm}, work::RankNetwork)
    D = Matrix{Float64}(undef, _n_comparisons(work), length(terms))
    return _rank_design!(D, Tuple(terms), work)
end

function _rank_design!(D::Matrix{Float64}, ts::Tuple, work::RankNetwork)
    n = work.n
    p = length(ts)
    size(D) == (_n_comparisons(work), p) || throw(ArgumentError(
        "_rank_design!: D is $(size(D)) but the design is $((_n_comparisons(work), p))"))
    delta = Vector{Float64}(undef, p)
    r = 0
    for ego in 1:n
        for j in 1:n, k in (j + 1):n
            (j == ego || k == ego) && continue
            r += 1
            _swap_delta!(delta, ts, work, ego, j, k)
            @inbounds for c in 1:p
                D[r, c] = -delta[c]
            end
        end
    end
    return D
end

# Parametric-bootstrap covariance of the swap MPLE: simulate `n_boot` rank
# networks at θ̂ with the AlterSwap sampler, refit the swap MPLE on each, take
# the empirical covariance. The loop is the shared `Networks.bootstrap_cov`; this
# supplies only the two callbacks that are ERGMRank's — and the exclusion of
# replicates without a finite refit (the round-3 pattern of `ERGM.mple`).
function _rank_bootstrap_cov(model::RankERGMModel, θ̂::Vector{Float64};
                             n_boot::Int, boot_burnin::Int, boot_interval::Int,
                             maxiter::Int, tol::Float64,
                             rng::Random.AbstractRNG)
    simulate(rng, B) = simulate_rank_ergm(model.network, model.terms, θ̂;
                                          n_sim=B, burnin=boot_burnin,
                                          interval=boot_interval, rng=rng)

    # A replicate on which the swap MPLE does not exist — a simulated ranking
    # that sits at the boundary of a statistic's attainable range (its refit
    # is ∓Inf), or a Newton iteration that did not converge — has no finite
    # refit: its row is NaN, excluded from the covariance below and counted.
    # The refit is silent (`warn=false`): R's boundary sentence would
    # otherwise fire once per replicate about a SIMULATED ranking, and
    # `bootstrap_cov` runs the refits on every thread.
    function refit(sim::RankNetwork)
        r = _rank_mple_fit(model.terms, sim; maxiter=maxiter, tol=tol, warn=false)
        return r.converged && all(isfinite, r.θ) ? r.θ : fill(NaN, length(θ̂))
    end

    boot = bootstrap_cov(refit, simulate, θ̂; n_boot=n_boot, rng=rng)
    replicates = boot.replicates
    ok = [all(isfinite, view(replicates, b, :)) for b in 1:n_boot]
    n_ok = count(ok)
    n_ok == n_boot && return boot.vcov, boot.se, replicates

    n_ok >= 2 || throw(ArgumentError(
        "fit_ergm_rank: se=:bootstrap — only $n_ok of the $n_boot bootstrap refits " *
        "had a finite swap MPLE (the others simulated a ranking on which a " *
        "statistic sits at the boundary of its attainable range); a covariance " *
        "needs at least 2. The model is near-degenerate at its swap-MPLE: " *
        "increase n_boot or simplify the model."))
    @warn "fit_ergm_rank: se=:bootstrap — $(n_boot - n_ok) of the $n_boot bootstrap " *
          "refits had no finite swap MPLE (the simulated ranking put a statistic " *
          "at the boundary of its attainable range — e.g. zero inconsistency with " *
          "the reference ranking) and were excluded; the standard errors are the " *
          "empirical covariance of the $n_ok finite refits. This is about the " *
          "simulated replicates, not about the observed ranking. " *
          "`fit.boot_replicates` holds every refit (NaN rows excluded); " *
          "`approximations(fit)` records the exclusion."
    V = Matrix{Float64}(cov(replicates[ok, :]))
    return V, sqrt.(max.(diag(V), 0.0)), replicates
end

# =============================================================================
# MCMC maximum likelihood (`method=:mcmle`): the estimator of R ergm.rank
# =============================================================================
#
# The iteration is ERGM.mcmle's (Hummel step length, Mahalanobis radius,
# singular-covariance stop, `mcmc_convergence` tests, final-sample covariance
# through `ERGM._mcmle_covariance`) with the AlterSwap chain as the sampler and
# the swap-MPLE as the start. Nothing statistical is re-derived here: the
# convergence tests and the covariance are ERGM's bindings, the kernel is
# `mh_toggle!`, the proposal and change statistics are `simulate_rank_ergm`'s.

# One AlterSwap chain at θ, recording the model statistics at every sampling
# point as a row of an `n_samples × p` matrix. The running statistics are kept
# current by adding the accepted swap's change statistics — the kernel's
# `delta` workspace, already filled by `change!` — so a draw costs nothing
# beyond the swap itself (no `compute` per sample). `terms` is a tuple: the
# closures and the kernel loop specialize on it. The proposal draws from
# `rng` in the same order as `simulate_rank_ergm`.
function _rank_stats_chain(current::RankNetwork, terms::Tuple, θ::Vector{Float64},
                           n_samples::Int, burnin::Int, interval::Int,
                           rng::Random.AbstractRNG)
    n = current.n
    p = length(θ)
    delta = Vector{Float64}(undef, p)
    stats = Float64[compute(t, current) for t in terms]
    samples = Matrix{Float64}(undef, n_samples, p)

    propose = rng -> _propose_swap(rng, n)
    change! = function (delta, move)
        ego, j, k = move
        _swap_delta!(delta, terms, current, ego, j, k)
        return false
    end
    apply! = function (move, removal)
        ego, j, k = move
        swap_ranks!(current, ego, j, k)
        @inbounds for c in 1:p
            stats[c] += delta[c]
        end
        return nothing
    end
    on_sample = function (k)
        @inbounds for c in 1:p
            samples[k, c] = stats[c]
        end
        return nothing
    end

    mh_toggle!(rng, θ, delta, propose, change!, apply!, on_sample;
               burnin=burnin, interval=interval, n_samples=n_samples)
    return samples
end

# `n_samples` statistics draws at θ over `n_chains` chains, every chain
# starting from `rnet` (ERGM `_mcmc_sample`'s contract): one chain runs on the
# caller's `rng`; several split the draws (the first `n_samples % n_chains`
# chains get one extra), draw one seed each from `rng` in order, run on
# separate tasks and are concatenated in chain order — so the result depends
# on `rng` and `n_chains` only, never on the thread count. `chain_lengths`
# gives the consecutive block lengths for the chain-aware ESS.
function _rank_mcmc_sample(rnet::RankNetwork, terms::Tuple, θ::Vector{Float64},
                           n_samples::Int, burnin::Int, interval::Int;
                           rng::Random.AbstractRNG, n_chains::Int=1)
    n_chains = clamp(n_chains, 1, n_samples)
    if n_chains == 1
        samples = _rank_stats_chain(copy(rnet), terms, θ, n_samples, burnin,
                                    interval, rng)
        return samples, [n_samples]
    end
    counts = fill(n_samples ÷ n_chains, n_chains)
    for c in 1:(n_samples % n_chains)
        counts[c] += 1
    end
    seeds = rand(rng, UInt64, n_chains)
    chain_stats = Vector{Matrix{Float64}}(undef, n_chains)
    @sync for c in 1:n_chains
        Threads.@spawn begin
            chain_stats[c] = _rank_stats_chain(copy(rnet), terms, θ, counts[c],
                                               burnin, interval,
                                               Random.Xoshiro(seeds[c]))
        end
    end
    samples = Matrix{Float64}(undef, n_samples, length(θ))
    offset = 0
    for c in 1:n_chains
        samples[(offset + 1):(offset + counts[c]), :] = chain_stats[c]
        offset += counts[c]
    end
    return samples, counts
end

# log Z(0) of the complete-ordering model: every ego orders n − 1 alters
# uniformly, so there are ((n − 1)!)^n orderings. Exact — the reference the
# bridge integrates from.
_log_n_orderings(n::Int) = Float64(n * log(factorial(big(n - 1))))

# Path-sampling (bridge) estimate of log Z(θ) − log Z(0) along θ_u = u·θ,
# u ∈ [0, 1]: d/du log Z(θ_u) = E_{θ_u}[g]'θ (the thermodynamic identity),
# integrated by the trapezoid rule over `nrungs` segments with E_{θ_u}[g]
# estimated by one seeded AlterSwap chain per grid point (`ERGM._bridge_logZ`,
# with the uniform ordering model — exact normalizer — as the reference).
# Rungs run on separate tasks, each on its own `Xoshiro(seed)` drawn from
# `rng` in order, so the estimate is thread-count independent. θ = 0 is exact
# (the ratio is 0) and draws nothing.
function _rank_bridge_logZ_ratio(rnet::RankNetwork, terms::Tuple, θ::Vector{Float64};
                                 nrungs::Int, n_samples::Int, burnin::Int,
                                 interval::Int, rng::Random.AbstractRNG)
    nrungs >= 1 || throw(ArgumentError("nrungs must be at least 1 (bridge_rungs=0 " *
                                       "skips the log-likelihood estimate)"))
    all(iszero, θ) && return 0.0
    us = range(0.0, 1.0; length=nrungs + 1)
    seeds = rand(rng, UInt64, length(us))
    contrib = Vector{Float64}(undef, length(us))
    @sync for k in eachindex(us)
        Threads.@spawn begin
            θu = us[k] .* θ
            stats = _rank_stats_chain(copy(rnet), terms, θu, n_samples, burnin,
                                      interval, Random.Xoshiro(seeds[k]))
            contrib[k] = dot(θ, vec(mean(stats, dims=1)))
        end
    end
    h = 1.0 / nrungs
    return h * (0.5 * contrib[1] + sum(@view contrib[2:end-1]) + 0.5 * contrib[end])
end

# The bridge log-likelihood θ'g_obs − log Z(θ), with log Z(θ) = log Z(0) +
# [log Z(θ) − log Z(0)]: the exact uniform-model normalizer plus the path
# integral above. `_rank_mcmle`'s loglik, and what the AIC/BIC read.
function _rank_bridge_loglik(rnet::RankNetwork, terms::Tuple, θ::Vector{Float64},
                             obs_stats::Vector{Float64};
                             nrungs::Int, n_samples::Int, burnin::Int,
                             interval::Int, rng::Random.AbstractRNG)
    ratio = _rank_bridge_logZ_ratio(rnet, terms, θ; nrungs=nrungs,
                                    n_samples=n_samples, burnin=burnin,
                                    interval=interval, rng=rng)
    return dot(θ, obs_stats) - ratio - _log_n_orderings(rnet.n)
end

# R's boundary sentence for a start the MCMLE cannot take: the swap-MPLE has a
# coefficient fixed at ∓Inf, so there is no finite starting point
function _refuse_boundary_start(fixed::Vector{String}, init_given::Bool)
    msg = "fit_ergm_rank(method=:mcmle): the observed statistic(s) $(join(fixed, ", ")) " *
          "are at the boundary of their attainable range under every single swap of " *
          "the observed ranking (the swap-MPLE used as the starting point fixes the " *
          "coefficient at ±Inf; R ergm: \"observed statistic(s) ... are at their " *
          "smallest attainable values\"). The MLE may not exist"
    if init_given
        @warn msg * "; proceeding from `init=` — the convergence tests decide, and " *
                    "a fit returned with converged == false must not be interpreted."
        return nothing
    end
    throw(ArgumentError(msg * ". Remove the term (as R's drop=TRUE does), or supply " *
                        "a finite starting point with `init=` to let the convergence " *
                        "tests decide."))
end

function _rank_mcmle(model::RankERGMModel;
                     n_samples::Int, burnin::Union{Nothing,Int},
                     interval::Union{Nothing,Int}, maxiter::Int, n_chains::Int,
                     init::Union{Nothing,AbstractVector{<:Real}},
                     gamma0::Float64, max_step_norm::Float64,
                     conv_threshold::Float64, hotelling_alpha::Float64,
                     bridge_rungs::Int, bridge_samples::Union{Nothing,Int},
                     verbose::Bool, rng::Random.AbstractRNG)
    n_samples >= 2 || throw(ArgumentError(
        "fit_ergm_rank(method=:mcmle): n_samples must be ≥ 2 (got $n_samples)"))
    n_chains >= 1 || throw(ArgumentError(
        "fit_ergm_rank(method=:mcmle): n_chains must be ≥ 1 (got $n_chains)"))
    maxiter >= 1 || throw(ArgumentError(
        "fit_ergm_rank(method=:mcmle): maxiter must be ≥ 1 (got $maxiter)"))
    bridge_rungs >= 0 || throw(ArgumentError(
        "fit_ergm_rank(method=:mcmle): bridge_rungs must be ≥ 0 (got $bridge_rungs); " *
        "0 skips the log-likelihood estimate (loglik/AIC/BIC are then NaN)"))

    rnet = model.network
    terms = model.terms
    ts = Tuple(terms)
    p = length(terms)
    term_names = [name(t) for t in terms]
    burnin, interval = _resolve_mcmc_controls(rnet, burnin, interval)

    # The start: the swap-MPLE, as ERGM.mcmle starts from the MPLE. Its own
    # warnings are silenced (a boundary or separated start is refused below
    # with a sentence that names the way out).
    verbose && println("Getting initial estimates via swap-MPLE...")
    start = _rank_mple_fit(terms, rnet; maxiter=100, tol=1e-8, warn=false)
    fixed = [term_names[k] for k in 1:p if isinf(start.θ[k])]
    isempty(fixed) || _refuse_boundary_start(fixed, init !== nothing)
    if init === nothing
        start.separated && throw(ArgumentError(
            "fit_ergm_rank(method=:mcmle): the swap-MPLE used as the starting point " *
            "does not exist (perfect separation: a combination of the model's " *
            "statistics is never lowered by any single swap of the observed ranking, " *
            "so the swap pseudo-likelihood has no finite maximum; R ergm warns \"The " *
            "MPLE does not exist!\"). The observed statistics are likely on the " *
            "boundary of the model's convex hull, where no MLE exists either. Remove " *
            "a term, collect more actors, or supply a starting point with `init=`."))
        start.converged || @warn "fit_ergm_rank(method=:mcmle): the swap-MPLE used " *
            "as the starting point did not converge within its Newton iteration " *
            "cap; starting from its last iterate."
        θ = copy(start.θ)
    else
        length(init) == p || throw(ArgumentError(
            "fit_ergm_rank(method=:mcmle): init has length $(length(init)) but the " *
            "model has $p terms"))
        all(isfinite, init) || throw(ArgumentError(
            "fit_ergm_rank(method=:mcmle): init must be finite (got $init)"))
        θ = Vector{Float64}(init)
    end

    obs_stats = Float64[compute(t, rnet) for t in terms]
    chisq_cut = quantile(Chisq(p), 0.95)

    converged = false
    γ = gamma0
    iterations = 0
    for iter in 1:maxiter
        iterations = iter
        verbose && println("MCMLE iteration $iter (step length γ = $(round(γ, digits=3)))...")

        samples, chain_lengths = _rank_mcmc_sample(rnet, ts, θ, n_samples, burnin,
                                                   interval; rng=rng, n_chains=n_chains)
        mean_stats = vec(mean(samples, dims=1))
        cov_stats = cov(samples)
        sd_stats = sqrt.(max.(diag(cov_stats), 0.0))
        ERGM._warn_degenerate_stats(sd_stats, term_names)
        diff = obs_stats .- mean_stats

        F = cholesky(Symmetric(cov_stats); check=false)
        if !issuccess(F)
            source = iter == 1 ? "the swap-MPLE start" :
                                 "the iteration-$(iter - 1) MCMLE update"
            @warn "fit_ergm_rank(method=:mcmle): the covariance matrix of the sampled " *
                  "statistics is singular at iteration $iter (collinear statistics, a " *
                  "degenerate model, or a collapsed sampler). MCMLE cannot take " *
                  "further Newton steps; the returned coefficients are $source, " *
                  "unrefined, and standard errors will be NaN. Check the model for " *
                  "degeneracy or redundant terms."
            break
        end

        # Squared Mahalanobis distance of the observed statistics from the
        # sampled cloud, and the Hummel step-length adaptation (ERGM.mcmle)
        d2 = max(dot(diff, F \ diff), 0.0)
        if d2 <= chisq_cut
            γ = 1.0
        else
            γ = clamp(min(sqrt(chisq_cut / d2), 2.0 * γ), 0.01, 1.0)
        end

        if γ == 1.0
            tests = mcmc_convergence(samples, obs_stats;
                                     conv_threshold=conv_threshold,
                                     hotelling_alpha=hotelling_alpha,
                                     chain_lengths=chain_lengths)
            if tests.converged
                converged = true
                verbose && println("Converged at iteration $iter (max t-ratio " *
                                   "$(round(maximum(tests.t_ratios), digits=4)), " *
                                   "Hotelling T² p-value " *
                                   "$(round(tests.hotelling_p, digits=4)))")
                break
            end
        end

        # Partial Newton step toward x_γ = γ·obs + (1−γ)·mean, capped in norm
        step = F \ (γ .* diff)
        step_norm = norm(step)
        step_norm > max_step_norm && (step .*= max_step_norm / step_norm)
        θ .+= step
    end

    # Final sample at the returned coefficients: the standard errors and the
    # recorded convergence report
    final_samples, chain_lengths = _rank_mcmc_sample(rnet, ts, θ, n_samples, burnin,
                                                     interval; rng=rng,
                                                     n_chains=n_chains)
    tests = mcmc_convergence(final_samples, obs_stats;
                             conv_threshold=conv_threshold,
                             hotelling_alpha=hotelling_alpha,
                             chain_lengths=chain_lengths)
    convergence = MCMLEConvergence((iterations, γ, tests.t_ratios,
                                    tests.hotelling_p, tests.n_eff))
    # V = Σ̂⁻¹ plus the Monte-Carlo component V·Σ_mc·V — ERGM.mcmle's
    # covariance, from ERGM's `public` binding
    vcov_fisher, V, std_errors, _ =
        ERGM._mcmle_covariance(final_samples, chain_lengths, nothing, nothing, p)

    converged || @warn "fit_ergm_rank(method=:mcmle): MCMLE did not converge in " *
        "maxiter=$maxiter iterations (last max t-ratio " *
        "$(_fmt3(maximum(tests.t_ratios))), Hotelling p $(_fmt3(tests.hotelling_p)), " *
        "step length γ $(_fmt3(γ))): the estimates are the last iterate and the " *
        "standard errors are unreliable; increase maxiter/n_samples/burnin, or " *
        "refit from these coefficients (init=coef(fit))"

    # The bridge runs last, so it consumes randomness after everything above:
    # coefficients and standard errors are bit-identical with and without it
    loglik = if bridge_rungs == 0
        NaN
    else
        verbose && println("Estimating the log-likelihood ($bridge_rungs bridge rungs)...")
        _rank_bridge_loglik(rnet, ts, θ, obs_stats; nrungs=bridge_rungs,
                            n_samples=something(bridge_samples, n_samples),
                            burnin=burnin, interval=interval, rng=rng)
    end

    # `n_kept` (the sample size `bic` uses) is R's `nobs.ergm` for the
    # likelihood: the n(n−1) ordered dyads, not the swap comparisons
    return RankERGMResult(model, θ, std_errors, V, loglik, converged, :fisher,
                          nothing, _n_dyads(rnet), :mcmle, convergence,
                          final_samples, vcov_fisher, bridge_rungs)
end

"""
    ergm_rank(rnet::RankNetwork, terms; kwargs...) -> RankERGMResult

R-faithful alias for [`fit_ergm_rank`](@ref) (the same function, `===`),
matching the R `ergm.rank` package name — the ecosystem convention of one
`fit_<model>` name and one statnet-style name per model package.

# Example
```julia
using ERGMRank
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
rnet = as_rank_network(m)
ergm_rank === fit_ergm_rank            # true
coef(ergm_rank(rnet, [RankDeference()])) == coef(fit_ergm_rank(rnet, [RankDeference()]))   # true
```
"""
const ergm_rank = fit_ergm_rank

# A deprecated BINDING (not a wrapper function), spelled out as
# `Base.@deprecate_binding` expands it — docstring first, so that attaching it
# does not itself touch the deprecated binding: `fit_rank_ergm ===
# fit_ergm_rank` still holds, and the first access in a session prints
# Julia's binding deprecation warning (with the `_dep_message_` text) under
# `--depwarn=yes`, which `Pkg.test` sets — Julia warns once per deprecated
# binding, not on every access.
"""
    fit_rank_ergm(rnet::RankNetwork, terms; kwargs...) -> RankERGMResult

**Deprecated** legacy alias for [`fit_ergm_rank`](@ref) (the same function,
`===`), from before the ecosystem settled on `fit_<model>` naming. It is a
deprecated binding (`Base.@deprecate_binding`): the first access in a
session prints `WARNING: Use of ERGMRank.fit_rank_ergm is deprecated, use
fit_ergm_rank instead.` when deprecation warnings are on (`julia
--depwarn=yes`, which `Pkg.test` sets; they are off in an interactive
session by default — Julia warns once per deprecated binding), and
the binding is removed in 0.3. Call [`fit_ergm_rank`](@ref) or
[`ergm_rank`](@ref) instead.

# Example
```julia
using ERGMRank
fit_rank_ergm === fit_ergm_rank        # true — with a deprecation warning: use fit_ergm_rank
```
"""
const fit_rank_ergm = fit_ergm_rank
const _dep_message_fit_rank_ergm = ", use fit_ergm_rank instead."
Base.deprecate(@__MODULE__, :fit_rank_ergm)

# =============================================================================
# Simulation
# =============================================================================

# A non-finite coefficient cannot be simulated from: θ'Δ is NaN (or ∓Inf) at
# every proposal that changes the statistic, so `mh_toggle!` rejects every
# move and the "draws" are copies of the starting ranking — which `gof` would
# then print as a perfect fit (every simulated statistic equal to the observed
# one, MC p-value 1). The package itself produces such fits (a statistic at
# the boundary of its attainable range is fixed at ∓Inf, an unidentified
# coefficient is NaN), so every entry point that simulates refuses them up
# front, naming the terms and the fix — the same refusal as `se=:bootstrap`.
function _check_finite_coefficients(context::AbstractString, terms, θ, fitted::Bool)
    all(isfinite, θ) && return nothing
    fixed = String[]; unidentified = String[]
    for (t, c) in zip(terms, θ)
        isnan(c) ? push!(unidentified, name(t)) : isinf(c) && push!(fixed, name(t))
    end
    parts = String[]
    isempty(fixed) || push!(parts, "coefficient(s) $(join(fixed, ", ")) fixed at ±Inf " *
        "by a statistic at the boundary of its attainable range")
    isempty(unidentified) || push!(parts, "coefficient(s) $(join(unidentified, ", ")) " *
        "NaN (not identified)")
    origin = fitted ?
        "the fit has $(join(parts, " and ")) — see `fit.converged` and `approximations(fit)`" :
        "θ has $(join(parts, " and ")) (got $(collect(θ)))"
    throw(ArgumentError(
        "$context: $origin: a ranking cannot be simulated at a non-finite " *
        "coefficient (the sampler would reject every swap and return copies of the " *
        "starting ranking, which `gof` would report as a perfect fit). Remove the " *
        "term, as R's drop=TRUE does, and refit."))
end

"""
    simulate_rank_ergm(rnet, terms, θ; n_sim=1, burnin=nothing, interval=nothing,
                       rng=Random.default_rng()) -> Vector{RankNetwork}

Simulate rank networks from a rank-order ERGM by Metropolis sampling with
the **AlterSwap** proposal (as in ergm.rank): pick a random ego and two
random alters, propose swapping their ranks, and accept with probability
`min(1, exp(θ'Δg))`. Every state visited is a valid complete ranking, and
the chain targets `P(y) ∝ exp(θ'g(y))` on the complete-ordering space
(the proposal is symmetric, so the CompleteOrder reference cancels).

The loop is the ecosystem's shared Metropolis kernel `ERGM.mh_toggle!`
(accept/reject arithmetic, burn-in, thinning), fed the AlterSwap move
`(ego, j, k)`, its change statistics and `swap_ranks!` as callables. `Δg`
is the per-term [`swap_change`](@ref) — evaluated from the comparisons the
swap touches, O(n) per term (O(n²) for nonconformity), never a
recomputation of the full statistics — written in place into the kernel's
workspace, so a step allocates nothing. A swap has no "removal" direction
(`Δg` is already the signed difference), so the kernel never negates the
log-ratio. The sampled sequence for a given `rng` is bit-identical to the
hand-written loop this kernel replaced.

# Burn-in and thinning defaults

`burnin` and `interval` default to `nothing`, resolved by the ecosystem's
one dyad-scaled rule (ERGM.jl's, `burnin = 20 n`, `interval = max(100, n ÷
10)`) applied to the size of the swap proposal space `n_swaps = n (n − 1)(n
− 2) / 2`: **`burnin = 20 n_swaps`** steps and **`interval = max(100,
n_swaps ÷ 10)`** steps between recorded draws. A chain that proposes one
swap per step needs a number of steps proportional to the number of swaps
to move every comparison a bounded number of times, so a fixed budget was
wrong at both ends: for 4 actors (12 swaps) the defaults are `(240, 100)`;
for 17 (2,040 swaps) they are `(40800, 204)`. An explicit integer is
honoured as given.

All randomness flows through `rng`: a fixed `rng` reproduces the draws. `θ`
is any real vector (`[0]`, a range) of one coefficient per term; `terms` may
also be a single term or a tuple. `n_sim < 1`, fewer than 3 actors, an
invalid starting ranking, a `θ` whose length differs from `terms`, a term
that is not a rank term (ERGM.jl's binary terms) and a `Network`/matrix in
place of the `RankNetwork` are refused with an `ArgumentError` that names
the fix.

# Example
```julia
using ERGMRank, Random
rnet = RankNetwork(5)
draws = simulate_rank_ergm(rnet, [RankDeference()], [-0.5];
                           n_sim=20, burnin=200, interval=10, rng=Xoshiro(1))
length(draws)                     # 20
all(is_valid_ranking, draws)      # true
scaled = simulate_rank_ergm(rnet, [RankDeference()], [-0.5]; n_sim=2, rng=Xoshiro(1))
length(scaled)                    # 2, after burnin = 20 · 30 = 600 steps
```
"""
function simulate_rank_ergm(rnet::RankNetwork, terms::Vector{<:AbstractERGMTerm},
                            θ::AbstractVector{<:Real};
                            n_sim::Int=1,
                            burnin::Union{Nothing,Int}=nothing,
                            interval::Union{Nothing,Int}=nothing,
                            rng::Random.AbstractRNG=Random.default_rng())
    _check_rank_terms("simulate_rank_ergm", terms)
    is_valid_ranking(rnet) ||
        throw(ArgumentError("simulate_rank_ergm: starting network is not a valid " *
                            "complete ranking: " *
                            something(_ranking_violation(rnet.ranks), "")))
    length(θ) == length(terms) ||
        throw(ArgumentError("simulate_rank_ergm: θ has $(length(θ)) coefficients " *
                            "but the model has $(length(terms)) " *
                            "$(length(terms) == 1 ? "term" : "terms")"))
    _check_finite_coefficients("simulate_rank_ergm", terms, θ, false)
    n_sim >= 1 ||
        throw(ArgumentError("simulate_rank_ergm: n_sim must be at least 1 (got $n_sim)"))
    rnet.n >= 3 || throw(ArgumentError(_too_few_actors("simulate_rank_ergm", rnet.n)))

    burnin, interval = _resolve_mcmc_controls(rnet, burnin, interval)
    # Function barrier: the term tuple's type is only known at runtime, and
    # everything below specializes on it (allocation-free steps).
    return _simulate_rank(copy(rnet), Tuple(terms), Vector{Float64}(θ),
                          n_sim, burnin, interval, rng)
end

simulate_rank_ergm(rnet::RankNetwork, term::AbstractERGMTerm, θ; kwargs...) =
    simulate_rank_ergm(rnet, AbstractERGMTerm[term], θ; kwargs...)
simulate_rank_ergm(rnet::RankNetwork, terms::Tuple, θ; kwargs...) =
    simulate_rank_ergm(rnet, collect(AbstractERGMTerm, terms), θ; kwargs...)
simulate_rank_ergm(x::Union{Network, AbstractMatrix}, terms, θ; kwargs...) =
    throw(_not_a_rank_network("simulate_rank_ergm", x))

# THE AlterSwap proposal (the ONE draw order every chain in this package
# uses — `simulate_rank_ergm`, the MCMLE's statistics chains, the bridge): an
# ego and two distinct alters, drawn in the same order (and so with the same
# rng consumption) as the loop the kernel replaced, so the sampled sequence
# stays bit-identical to it.
@inline function _propose_swap(rng::Random.AbstractRNG, n::Int)
    ego = rand(rng, 1:n)
    j = rand(rng, 1:n)
    while j == ego
        j = rand(rng, 1:n)
    end
    k = rand(rng, 1:n)
    while k == ego || k == j
        k = rand(rng, 1:n)
    end
    return (ego, j, k)
end

function _simulate_rank(current::RankNetwork, terms::Tuple, θ::Vector{Float64},
                        n_sim::Int, burnin::Int, interval::Int,
                        rng::Random.AbstractRNG)
    n = current.n
    delta = Vector{Float64}(undef, length(θ))
    draws = Vector{RankNetwork}()
    sizehint!(draws, n_sim)

    # The AlterSwap proposal: an ego and two distinct alters, drawn in the
    # same order (and so with the same rng consumption) as the loop the
    # kernel replaced
    propose = rng -> _propose_swap(rng, n)
    # Δg of the swap, already signed: never a "removal"
    change! = function (delta, move)
        ego, j, k = move
        _swap_delta!(delta, terms, current, ego, j, k)
        return false
    end
    apply! = function (move, removal)
        ego, j, k = move
        swap_ranks!(current, ego, j, k)
        return nothing
    end
    on_sample = function (k)
        push!(draws, copy(current))
        return nothing
    end

    mh_toggle!(rng, θ, delta, propose, change!, apply!, on_sample;
               burnin=burnin, interval=interval, n_samples=n_sim)
    return draws
end

"""
    simulate_rank_ergm(result::RankERGMResult; kwargs...) -> Vector{RankNetwork}

Simulate from a fitted rank-order ERGM (the fitted coefficients, terms and
observed ranking as the starting state); keywords as above.

# Errors
A fit with a non-finite coefficient — one fixed at ±Inf by a statistic at the
boundary of its attainable range, or `NaN` (not identified, `converged ==
false`) — is refused with an `ArgumentError` naming the term(s): the sampler
would reject every swap and return copies of the observed ranking. Remove the
term (R's `drop=TRUE`) and refit.
"""
function simulate_rank_ergm(result::RankERGMResult; kwargs...)
    _check_finite_coefficients("simulate_rank_ergm", result.model.terms,
                               result.coefficients, true)
    return simulate_rank_ergm(result.model.network, result.model.terms,
                              result.coefficients; kwargs...)
end

# =============================================================================
# Goodness of fit
# =============================================================================

"""
    gof(result::RankERGMResult; n_sim=100, burnin=nothing, interval=nothing,
        rng=Random.default_rng()) -> GOFResult

Goodness-of-fit assessment of a fitted rank-order ERGM: rank networks are
simulated from the fitted model with [`simulate_rank_ergm`](@ref) (AlterSwap
Metropolis sampling) and the observed model statistics are compared with
their simulated distributions.

This is a method of the shared `Networks.gof` generic; it returns the shared
`Networks.GOFResult` (observed value, simulation envelope, and two-sided
Monte-Carlo p-value per statistic).

# Keyword Arguments
- `n_sim::Int=100`: Number of simulated rank networks (at least 1)
- `burnin`, `interval`, `rng`: passed to [`simulate_rank_ergm`](@ref); `nothing`
  (the default) resolves to its swap-scaled rule — `burnin = 20 n_swaps`,
  `interval = max(100, n_swaps ÷ 10)` with `n_swaps = n (n − 1)(n − 2) / 2`
  — and an explicit integer is honoured

# Errors
`n_sim < 1` is refused. So is a fit with a non-finite coefficient — one fixed
at ±Inf by a statistic at the boundary of its attainable range, or `NaN` (not
identified, `converged == false`): the sampler would reject every swap, the
"simulations" would all equal the observed ranking, and the table would show
a perfect fit (every simulated statistic equal to the observed one, MC
p-value 1). The `ArgumentError` names the term(s) and says to remove them, as
R's `drop=TRUE` does, and refit.

# Example
```julia
using ERGMRank, Random
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
fit = fit_ergm_rank(as_rank_network(m), [RankDeference()])
g = gof(fit; n_sim=50, burnin=100, interval=10, rng=Xoshiro(1))
g.statistics[1].labels            # ["rank.deference"]
```
"""
function gof(result::RankERGMResult; n_sim::Int=100,
             burnin::Union{Nothing,Int}=nothing,
             interval::Union{Nothing,Int}=nothing,
             rng::Random.AbstractRNG=Random.default_rng())
    n_sim >= 1 || throw(ArgumentError("gof: n_sim must be at least 1 (got $n_sim)"))
    rnet = result.model.network
    terms = result.model.terms
    _check_finite_coefficients("gof", terms, result.coefficients, true)
    sims = simulate_rank_ergm(result; n_sim=n_sim, burnin=burnin,
                              interval=interval, rng=rng)

    obs_stats = [compute(term, rnet) for term in terms]
    sim_stats = [compute(term, s) for s in sims, term in terms]
    stats = GOFStatistic("model statistics", [name(term) for term in terms],
                         obs_stats, sim_stats)

    return GOFResult([stats]; model="Rank-Order ERGM")
end

end # module
