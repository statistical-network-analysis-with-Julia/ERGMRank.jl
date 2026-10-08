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

# The statistic protocol (`name`/`compute` are NetworkCore.jl generics that
# ERGM.jl re-exports) and the ONE Metropolis toggle kernel every ERGM-family
# sampler runs on: `simulate_rank_ergm` supplies the
# AlterSwap move, its change statistics and its state mutation as callables.
# `mcmc_convergence` (the t-ratio/Hotelling tests) and the `MCMLEConvergence`
# report type are ERGM's `public` MCMLE machinery, which the rank MCMLE
# (`method=:mcmle`) runs on rather than re-implementing.
import ERGM: name, compute, mh_toggle!, mcmc_convergence, MCMLEConvergence
# Shared presentation infrastructure (NetworkCore.jl): the ONE `gof` generic all
# model packages extend, the GOF containers, the coefficient table that
# `coeftable(fit)` returns and `show` prints, the floored z→p helper and the
# ONE `se=` validator
import NetworkCore: gof, GOFStatistic, GOFResult, CoefficientTable, coeftable,
                 z_pvalues, check_se

# The ONE shared bootstrap loop (NetworkCore.jl `src/bootstrap.jl`): simulate,
# refit, empirical covariance. `se=:bootstrap` supplies the two callbacks; the
# loop, the threading and the rng discipline are not reimplemented here.
import NetworkCore: bootstrap_cov
# The shared task runner: waits for every task and rethrows the first failure
# as the task's own exception (never a TaskFailedException)
import NetworkCore: spawn_all
# The shared separation policy: one warning sentence and the caveat
# `approximations` lists. The verdict itself (an exact test on the design) is
# computed once, by `ERGM.Extension.mple_fit_design`, and returned with the fit
import NetworkCore: warn_separation, separation_caveat

# The ONE Newton optimizer and logistic-likelihood kernel of the ecosystem
# (NetworkCore.jl `src/newton.jl`, `public`; `ERGM.newton_fit` is the same
# binding). The swap MPLE is a logistic regression on the swap-difference rows
# with the response identically `true`, so it has no loop of its own.
import NetworkCore: newton_fit, logistic_derivatives

# The shared result-metadata protocol (NetworkCore.jl `src/results.jl`): the
# generic accessors that say what a fit actually did. Imported by name because
# ERGMRank adds methods for `RankERGMResult`; `fit_metadata(fit)` collects them.
import NetworkCore: estimand, objective, is_exact, se_method, missing_method,
                 approximations, fit_metadata
# The conversion contract (NetworkCore.jl `src/conversion.jl`, `src/missing.jl`)
# for the `Network` → `RankNetwork` adapter: the missing-dyad guard and the
# two traits it is declared through, and the report a lossy conversion
# returns on request. `Network` and its queries are used qualified or by name;
# ERGMRank does not re-export NetworkCore.
import NetworkCore: supports_missing, missing_policies, require_observed,
                 ConversionReport, record_drop!
using NetworkCore: Network, nv, has_edge, is_directed, get_edge_attribute,
                list_edge_attributes, list_vertex_attributes,
                list_network_attributes
import StatsAPI
import StatsAPI: coef, coefnames, stderror, vcov, loglikelihood, nobs, dof, aic, bic, confint

# Core types
export RankNetwork, RankERGMModel, RankERGMResult
export CompleteOrderReference

# Rank access and manipulation
export get_rank, set_rank!, swap_ranks!, is_valid_ranking
export as_rank_network, rank_matrix

# The teaching dataset the estimator claims rest on (R ergm.rank's newcomb)
export newcomb_week1

# The statistic protocol: `compute(term, rnet)` and `name(term)` are the ONE
# `NetworkCore.compute`/`NetworkCore.name` (reached through ERGM.jl, which
# re-exports them; `ERGMRank.compute === NetworkCore.compute` is pinned), so
# `using ERGMRank` alone evaluates a rank statistic
export compute, name

# The shared result-metadata protocol (NetworkCore.jl), re-exported so
# `approximations(fit)` works with just `using ERGMRank`; the same bindings
# every fitting package in the ecosystem exports
export fit_metadata, approximations, estimand, objective, is_exact, se_method,
       missing_method

# Terms (matching R ergm.rank) and their swap change statistic
export RankDeference, RankNonconformity, RankNodeICov
export RankInconsistency, RankEdgeCov
export swap_change

# Estimation and simulation
export fit_ergm_rank, ergm_rank
export simulate_rank_ergm

# Goodness of fit (method of the shared NetworkCore.jl `gof` generic)
export gof

# The ecosystem's StatsAPI surface (re-exported so `coef(fit)` etc. work with
# just `using ERGMRank`; `coeftable` is the ONE `StatsAPI.coeftable` binding
# that NetworkCore.jl re-exports)
export coef, coefnames, stderror, vcov, confint, loglikelihood, nobs, dof, aic, bic,
       coeftable

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
`NetworkCore.Network` with a rank edge attribute; [`RankNetwork(n)`](@ref) builds
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
# `MIME"text/plain"` method the REPL calls — NetworkCore.jl's convention for
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
the NetworkCore.jl missing-data guide for how the rest of the ecosystem treats
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
            "(see the NetworkCore.jl missing-data guide, docs/src/guide/missing_data.md). " *
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

Build a `RankNetwork` from a **directed** `NetworkCore.Network` whose edge
attribute `attr` holds the rank ego `i` assigns alter `j` on the arc `i → j`
(greater = higher standing) — the Julia counterpart of R's
`as.matrix(nw, attrname = "rank")` on `ergm.rank`'s `newcomb` networks. The
conversion honours the ecosystem conversion contract (NetworkCore.jl
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
using ERGMRank, NetworkCore
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

Weeks 2–15 are not bundled; `NetworkCore.load_dataset` does not carry Newcomb
yet, so this is the one loader.

# Example
```julia
using ERGMRank
rnet = newcomb_week1()
rnet.n                                            # 17
is_valid_ranking(rnet)                            # true
compute(RankDeference(), rnet) == 844             # true: ergm.rank's summary()
compute(RankNonconformity(:all), rnet) == 12748   # true
pl = fit_ergm_rank(rnet, [RankDeference(), RankNonconformity()]; method=:mple)   # swap-MPLE, ~15 ms
round.(coef(pl); digits=4) == [-0.1409, -0.0059]                    # true
# fit_ergm_rank(rnet, [RankDeference(), RankNonconformity()]) is the MCMC MLE,
# ergm.rank's estimator: ≈ [-0.153, -0.0066] in 10–30 s
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
fit = fit_ergm_rank(as_rank_network(m), [RankDeference()]; method=:mple)
fit.model.reference == ref             # true: the reference of every rank fit
```
"""
struct CompleteOrderReference end

# =============================================================================
# Terms
# =============================================================================
#
# PROVENANCE of the summary statistics in this section (the `compute`
# methods and the `weights`/`wtcenter` handling of `RankInconsistency`).
# They implement the term definitions of Krivitsky & Butts (2017,
# "Exponential-family random graph models for rank-order relational data",
# Sociological Methodology 47(1)) as documented for ergm.rank's terms
# (`rank.deference`, `rank.nonconformity`, `rank.nodeicov`,
# `rank.inconsistency`, `rank.edgecov`) and in the docstrings below. They were
# written independently for this package from those definitions; no code of
# ergm.rank was consulted or translated. Agreement with ergm.rank is checked
# by output only: golden fixtures of R's `summary()` (test/fixtures/
# rank_terms.toml, rank_inconsistency_weights.toml) at 1e-9, and a
# brute-force enumeration of each definition in the test suite. The swap
# change statistics of the next section are this package's own as well.

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

# Ordered triples (ego, peer, third) of distinct actors in which the ego ranks
# the peer above the third actor while the peer ranks that third actor above
# the ego.
function compute(::RankDeference, rnet::RankNetwork)
    y, n = rnet.ranks, rnet.n
    hits = 0
    @inbounds for ego in 1:n, peer in 1:n
        peer == ego && continue
        peer_from_ego = y[ego, peer]      # the peer's standing with the ego
        ego_from_peer = y[peer, ego]      # the ego's standing with the peer
        for third in 1:n
            (third == ego || third == peer) && continue
            hits += (y[ego, third] < peer_from_ego) & (y[peer, third] > ego_from_peer)
        end
    end
    return Float64(hits)
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

# `:all`: Σ over ordered alter pairs (a, b) and unordered pairs of egos {e, f}
# that both rank a and b, of 1[(y_ea > y_eb) ≠ (y_fa > y_fb)]. For a fixed
# (a, b) the egos split into those that put a above b and the rest; every
# disagreeing pair of egos takes one actor from each side, so (a, b)
# contributes the product of the two group sizes. O(n³).
function _nonconformity_all(y::Matrix{Int}, n::Int)
    total = 0
    @inbounds for a in 1:n, b in 1:n
        a == b && continue
        a_first = 0
        for e in 1:n
            (e == a || e == b) && continue
            a_first += y[e, a] > y[e, b]
        end
        total += a_first * (n - 2 - a_first)
    end
    return total
end

# `:localAND`: an ego disagrees with a guide whom it ranks above both alters j
# and k, the guide putting j above k while the ego puts k at least as high as
# j. Counted over (ego, guide, j, k), all distinct.
function _nonconformity_local_and(y::Matrix{Int}, n::Int)
    total = 0
    @inbounds for ego in 1:n, guide in 1:n
        guide == ego && continue
        cut = y[ego, guide]               # j and k must both rank below the guide
        for j in 1:n
            (j == ego || j == guide || y[ego, j] >= cut) && continue
            for k in 1:n
                (k == ego || k == guide || k == j || y[ego, k] >= cut) && continue
                total += (y[guide, j] > y[guide, k]) & (y[ego, k] >= y[ego, j])
            end
        end
    end
    return total
end

function compute(t::RankNonconformity, rnet::RankNetwork)
    count = t.variant == :all ? _nonconformity_all(rnet.ranks, rnet.n) :
                                _nonconformity_local_and(rnet.ranks, rnet.n)
    return Float64(count)
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

# The covariate terms add, over every ego and ordered alter pair (j, k) with j
# ranked above k, c(ego, j) − c(ego, k). Collected per alter a, c(ego, a)
# enters with + once for each alter the ego ranks below a and with − once for
# each alter it ranks above a, so its multiplier is Σ_b sign(y[ego, a] − y[ego, b]).
@inline function _net_standing(y::Matrix{Int}, n::Int, ego::Int, a::Int)
    s = 0
    @inbounds for b in 1:n
        (b == ego || b == a) && continue
        s += sign(y[ego, a] - y[ego, b])
    end
    return s
end

function compute(t::RankNodeICov, rnet::RankNetwork)
    n = rnet.n
    length(t.x) == n || throw(ArgumentError(_bad_nodeicov_length(t, n)))
    total = 0.0
    for ego in 1:n, a in 1:n
        a == ego && continue
        total += t.x[a] * _net_standing(rnet.ranks, n, ego, a)
    end
    return total
end

"""
    RankInconsistency(ref; weights=nothing, wtname=nothing, wtcenter=false) <: AbstractERGMTerm

(Weighted) inconsistency, `rank.inconsistency(x, attrname, weights, wtname,
wtcenter)` in ergm.rank: the number of ego–alter pair comparisons on which the
network disagrees with a reference ranking
(`(y_ij > y_ik) ≠ (ref_ij > ref_ik)`), each comparison optionally weighted.
Useful for drift from a prior wave or an exogenous ordering. Fitting
`RankInconsistency(rnet)` to `rnet` itself puts the statistic at the boundary
of its attainable range (zero inconsistency), where no finite estimate
exists — see [`fit_ergm_rank`](@ref).

# Arguments
- `ref`: the reference ranking — a rank matrix (same convention as the
  network's) or a `RankNetwork`.
- `weights`: `nothing` (every comparison counts 1), an `n × n × n` array whose
  `[i, j, k]` element weighs ego `i`'s comparison of alters `j` and `k`, or a
  function `(i, j, k) -> weight` (evaluated on every triple of distinct
  actors). The statistic is then
  `Σ_{i} Σ_{j ≠ k} weights[i, j, k] · 1[(y_ij > y_ik) ≠ (ref_ij > ref_ik)]`
  over ordered alter pairs, exactly as in ergm.rank.
- `wtcenter`: if `true` the weights are centred at their mean before use — as
  in R, the mean of the function's values over the triples of distinct
  actors, or of every entry of an array that is not `NaN`/`missing` (so pass
  `NaN` in the `i == j`, `i == k`, `j == k` cells of an array to keep them out
  of the mean, as R's `NA`). `NaN`/`missing` weights count as 0.
- `wtname`: a label for the weights; the term is then named
  `rank.inconsistency:<wtname>` (with a trailing `c` when centred), following
  R's `inconsistency.rank:<wtname>c`.

# Fields
- `ref::Matrix{Int}`: reference rank matrix
- `weights::Union{Nothing, Array{Float64,3}}`: the weights actually used
  (centred, invalid cells zero), or `nothing`
- `label::String`: the term's name

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
w = RankInconsistency(rnet; weights=(i, j, k) -> i, wtname="ego")
compute(w, other) == 2.0                        # true: both comparisons weigh 1 (ego 1)
name(w)                                         # "rank.inconsistency:ego"
```
"""
struct RankInconsistency <: AbstractERGMTerm
    ref::Matrix{Int}
    weights::Union{Nothing, Array{Float64,3}}
    label::String
end

const _WEIGHTS_FORMS = "an n×n×n array or a function (i, j, k) -> weight"

function RankInconsistency(ref::AbstractMatrix{<:Integer};
                           weights=nothing,
                           wtname::Union{Nothing, AbstractString, Symbol}=nothing,
                           wtcenter::Bool=false)
    reference = Matrix{Int}(ref)          # a copy: editing `ref` later leaves the term alone
    if weights === nothing
        (wtname === nothing && !wtcenter) || throw(ArgumentError(
            "RankInconsistency: wtname= and wtcenter= qualify weights=, which was not " *
            "given; pass weights= ($_WEIGHTS_FORMS) or leave them out"))
        return RankInconsistency(reference, nothing, "rank.inconsistency")
    end
    table = _inconsistency_weights(weights, size(reference, 1), wtcenter)
    # R names the term inconsistency.rank:<wtname>, with a trailing "c" when
    # centred; without a wtname the name stays plain, centred or not
    label = wtname === nothing ? "rank.inconsistency" :
            string("rank.inconsistency:", wtname, wtcenter ? "c" : "")
    return RankInconsistency(reference, table, label)
end

RankInconsistency(ref_net::RankNetwork; kwargs...) =
    RankInconsistency(rank_matrix(ref_net); kwargs...)

# The weight of each comparison (ego i; alters j, k) as an n×n×n Float64 array.
# A weight that is not available (NaN or `missing` in an array; every cell with
# a repeated index when the weights come from a function) is left out of the
# mean that `wtcenter=true` subtracts, and then weighs 0.
function _inconsistency_weights(weights, n::Int, wtcenter::Bool)
    table = _weight_table(weights, n)          # NaN = not available
    if wtcenter
        available = filter(!isnan, table)
        isempty(available) || (table .-= mean(available))
    end
    table[isnan.(table)] .= 0.0
    return table
end

function _weight_table(weights::AbstractArray, n::Int)
    size(weights) == (n, n, n) || throw(ArgumentError(
        "RankInconsistency: the weights array has size $(size(weights)), but the " *
        "reference ranking has $n actors; pass a $n×$n×$n array whose [i, j, k] entry " *
        "weighs ego i's comparison of alters j and k, or a function (i, j, k) -> weight"))
    table = Array{Float64,3}(undef, n, n, n)
    for (cell, value) in zip(eachindex(table), weights)
        table[cell] = _weight_value(value, nothing)
    end
    return table
end

function _weight_table(weights, n::Int)
    applicable(weights, 1, 2, 3) || throw(ArgumentError(
        "RankInconsistency: weights must be $_WEIGHTS_FORMS (got a $(typeof(weights)))"))
    table = fill(NaN, n, n, n)
    for k in 1:n, j in 1:n
        j == k && continue
        for i in 1:n
            (i == j || i == k) && continue
            table[i, j, k] = _weight_value(weights(i, j, k), (i, j, k))
        end
    end
    return table
end

# One weight, checked: NaN or `missing` → NaN (not available); a finite real → itself.
function _weight_value(value, triple)
    value === missing && return NaN
    if !(value isa Real)
        at = triple === nothing ? "" : " at (i, j, k) = $triple"
        throw(ArgumentError("RankInconsistency: a weight must be a real number " *
                            "(got $(repr(value))$at)"))
    end
    isnan(value) && return NaN
    isfinite(value) || throw(ArgumentError(
        "RankInconsistency: a weight must be finite (got $value); NaN or missing " *
        "marks a comparison that carries no weight"))
    return Float64(value)
end

name(t::RankInconsistency) = t.label

_bad_weights_size(t::RankInconsistency, n::Int) =
    "RankInconsistency: the weights array is $(join(size(t.weights), "×")) but the " *
    "network has $n actors; pass an $n×$n×$n array"

# Σ over egos and ordered pairs of distinct alters (j, k) of
# weight(ego, j, k) · 1[(y_ej > y_ek) ≠ (ref_ej > ref_ek)], every weight 1
# when no weights were given.
_comparison_weight(::Nothing, ego::Int, j::Int, k::Int) = 1.0
_comparison_weight(w::Array{Float64,3}, ego::Int, j::Int, k::Int) = @inbounds w[ego, j, k]

function _disagreement(y::Matrix{Int}, ref::Matrix{Int}, w, n::Int)
    total = 0.0
    @inbounds for ego in 1:n, j in 1:n
        j == ego && continue
        for k in 1:n
            (k == ego || k == j) && continue
            if (y[ego, j] > y[ego, k]) != (ref[ego, j] > ref[ego, k])
                total += _comparison_weight(w, ego, j, k)
            end
        end
    end
    return total
end

function compute(t::RankInconsistency, rnet::RankNetwork)
    n = rnet.n
    size(t.ref) == (n, n) || throw(ArgumentError(
        _bad_matrix_size("RankInconsistency: the reference rank matrix", size(t.ref), n)))
    t.weights === nothing || size(t.weights) == (n, n, n) ||
        throw(ArgumentError(_bad_weights_size(t, n)))
    return _disagreement(rnet.ranks, t.ref, t.weights, n)
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
    total = 0.0
    for ego in 1:n, a in 1:n
        a == ego && continue
        total += t.cov[ego, a] * _net_standing(rnet.ranks, n, ego, a)
    end
    return total
end

# =============================================================================
# Swap change statistics
# =============================================================================
#
# PROVENANCE. The swap change statistics below are derived independently
# from the term definitions (the docstrings above; Krivitsky & Butts 2017) as
# a swap move — the change of a statistic when one ego exchanges the ranks of
# two alters — which has no counterpart in ergm.rank's single-cell change
# functions. Each method is validated against the brute-force difference of
# two `compute` evaluations.
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
AlterSwap analogue of `ERGM.change_stat`. (The methods are derived
independently from the term definitions as a swap move, which has no
counterpart in R `ergm.rank`'s single-cell change functions.) It is
evaluated from the
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
    # A deference triple (i, l, j): i ranks l over j while l ranks j over i.
    # The swapping actor's row holds two of its comparisons: as the ego i
    # (its comparison of l = x over j = z, weighted by l's comparison of j
    # over i) and as l (its comparison of j = x over i = z, weighted by i's
    # comparison of l over j). Both weights lie outside the swapping row.
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

# Local nonconformity, in the notation of its definition (Krivitsky & Butts
# 2017; the `RankNonconformity` docstring): ego i, an actor l whom i ranks
# above both alters j and k, with l ranking j over k while i ranks k at least
# as high as j. One summand, its factors in the order the definition reads:
#   1[y_il > y_ij] · 1[y_il > y_ik] · 1[y_lj > y_lk] · 1[y_ik ≥ y_ij].
# `r_il`, `r_ij`, `r_ik` are i's ranks of l, j and k (before or after a swap).
@inline function _local_and_summand(y::Matrix{Int}, l::Int, j::Int, k::Int,
                                    r_il::Int, r_ij::Int, r_ik::Int)
    @inbounds return r_il > r_ij && r_il > r_ik && y[l, j] > y[l, k] && r_ik >= r_ij
end

# The change of one summand with i as the ego, when i swaps its ranks of the
# alters a and b
@inline function _local_and_change(y::Matrix{Int}, i::Int, l::Int, j::Int, k::Int,
                                   a::Int, b::Int)
    @inbounds before = _local_and_summand(y, l, j, k, y[i, l], y[i, j], y[i, k])
    after = _local_and_summand(y, l, j, k, _swapped_rank(y, i, l, a, b),
                               _swapped_rank(y, i, j, a, b),
                               _swapped_rank(y, i, k, a, b))
    return Int(after) - Int(before)
end

# The swapping actor's row enters the statistic in two roles: as the ego i
# (the three comparisons y_il : y_ij, y_il : y_ik and y_ik : y_ij) and as the
# actor l (the comparison y_lj : y_lk, matched by every other ego).
_swap_change_nonconformity_local(rnet::RankNetwork, ego::Int, a::Int, b::Int) =
    Float64(_local_and_change_as_i(rnet.ranks, rnet.n, ego, a, b) +
            _local_and_change_as_l(rnet.ranks, rnet.n, ego, a, b))

# The ego i swaps its ranks of a and b: the summands (l, j, k) whose actors
# include a or b, each once. With l ∈ {a, b}, i's rank of l moves, so every
# (j, k) may cross the threshold y_il; otherwise only j or k in {a, b}.
function _local_and_change_as_i(y::Matrix{Int}, n::Int, i::Int, a::Int, b::Int)
    total = 0
    for l in 1:n
        l == i && continue
        if l == a || l == b
            for j in 1:n
                (j == i || j == l) && continue
                for k in 1:n
                    (k == i || k == l || k == j) && continue
                    total += _local_and_change(y, i, l, j, k, a, b)
                end
            end
        else
            for j in (a, b), k in 1:n
                (k == i || k == l || k == j) && continue
                total += _local_and_change(y, i, l, j, k, a, b)
            end
            for k in (a, b), j in 1:n
                (j == i || j == l || j == a || j == b) && continue
                total += _local_and_change(y, i, l, j, k, a, b)
            end
        end
    end
    return total
end

# The actor l swaps its ranks of a and b: the swap changes l's comparison of
# the pairs (j, k) it flips by d = Δ1[y_lj > y_lk], and every other ego i
# that ranks l above both j and k and ranks k at least as high as j counts d
function _local_and_change_as_l(y::Matrix{Int}, n::Int, l::Int, a::Int, b::Int)
    total = 0
    for i in 1:n
        i == l && continue
        @inbounds r_il = y[i, l]
        total += Int(_sum_changed_pairs(y, n, l, a, b) do j, k, d
            (j == i || k == i) && return 0.0
            @inbounds counted = r_il > y[i, j] && r_il > y[i, k] && y[i, k] >= y[i, j]
            counted ? Float64(d) : 0.0
        end)
    end
    return total
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
    w = t.weights
    if w === nothing
        return _sum_changed_pairs(rnet.ranks, n, ego, j, k) do x, z, d
            # A flip turns agreement with the reference into disagreement (+1)
            # or disagreement into agreement (−1)
            @inbounds agreed = (d < 0) == (r[ego, x] > r[ego, z])
            agreed ? 1.0 : -1.0
        end
    end
    wt = w::Array{Float64,3}
    size(wt) == (n, n, n) || throw(ArgumentError(_bad_weights_size(t, n)))
    # Weighted: the flipped comparison (x, z) of ego carries weights[ego, x, z]
    return _sum_changed_pairs(rnet.ranks, n, ego, j, k) do x, z, d
        @inbounds agreed = (d < 0) == (r[ego, x] > r[ego, z])
        @inbounds agreed ? wt[ego, x, z] : -wt[ego, x, z]
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
fit = fit_ergm_rank(as_rank_network(m), [RankDeference()]; method=:mple)
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

# NetworkCore.jl's display convention (as for `RankNetwork` and `RankERGMResult`):
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
  `termination` the verdict of the stopping rule behind `converged` (a
  NamedTuple `(rule, p_value, precision, confidence, n_samples)`, as
  `ERGM.ERGMResult` carries; `nothing` for `:mple`), `mcmc_samples` the
  statistics sampled at the returned coefficients, and `bridge_rungs` the
  number of path-sampling segments behind `loglik`.

`se_type` records how `std_errors`/`vcov` were ACTUALLY obtained:

- `:hessian` — the inverse negative Hessian of the swap pseudo-likelihood. The
  swap comparisons overlap, so these are anticonservative (2–3.9× too small
  on Newcomb's fraternity). `inference_withheld` is `true` when they are the
  default of a `method=:mple` fit: the z values and p-values of
  `coeftable`/`show` are then `NaN` and `confint` refuses, because no test or
  interval should be built on them; it is `false` when `se=:hessian` was
  passed explicitly (the written opt-in to the naive Wald table).
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

`separated_terms` names the coefficients of a separated swap-MPLE (a
combination of statistics no single swap of the observed ranking lowers, so
the pseudo-likelihood has no finite maximum; decided by NetworkCore's exact
verdict on the swap design). It is empty otherwise. When it is not empty the
fit has `converged == false`, its z values and p-values are `NaN`, `confint`
returns `NaN` and `se=:bootstrap` is refused, as the ecosystem's separation
policy prescribes.

`se_type` is what `NetworkCore.se_method(fit)` reports, and what `show` reads
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
fit = fit_ergm_rank(as_rank_network(m), [RankDeference()]; method=:mple)
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
    inference_withheld::Bool
    termination::Union{Nothing, NamedTuple}
    separated_terms::Vector{String}

    function RankERGMResult(model, coefficients, std_errors, vcov, loglik, converged,
                            se_type, boot_replicates, n_kept, method,
                            mcmc_convergence, mcmc_samples, vcov_fisher, bridge_rungs,
                            inference_withheld::Bool=false,
                            termination::Union{Nothing, NamedTuple}=nothing,
                            separated_terms::Vector{String}=String[])
        method in (:mple, :mcmle) || throw(ArgumentError(
            "RankERGMResult: method must be :mple or :mcmle (got $(repr(method)))"))
        new(model, coefficients, std_errors, vcov, loglik, converged, se_type,
            boot_replicates, n_kept, method, mcmc_convergence, mcmc_samples,
            vcov_fisher, bridge_rungs, inference_withheld, termination, separated_terms)
    end
end

# The stopping rule's verdict in words (ERGM.jl prints the same line)
function _termination_detail(t::NamedTuple)
    t.rule === :confidence ?
        "confidence rule, p = $(_fmt3(t.p_value)) against " *
        "$(_fmt3(1 - t.confidence)) (precision $(t.precision), " *
        "$(t.n_samples) draws)" :
        "t-ratio and Hotelling rule, Hotelling p = $(_fmt3(t.p_value)) " *
        "($(t.n_samples) draws)"
end

# Number of pseudo-likelihood contributions: one per (ego, unordered alter
# pair) conditional — and the size of the AlterSwap proposal space, the rank
# analogue of ERGM's dyad count
_n_comparisons(rnet::RankNetwork) = rnet.n * (rnet.n - 1) * (rnet.n - 2) ÷ 2

"""
    _mcmc_defaults(rnet::RankNetwork) -> (burnin, interval)

THE burn-in/interval rule behind every AlterSwap sampler default in this
package (`simulate_rank_ergm`, `gof`, the `se=:bootstrap` refits): the ONE
dyad-scaled rule of ERGM.jl, `ERGM.Extension.mcmc_defaults(n)` — `burnin = 20 n` and
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
_mcmc_defaults(rnet::RankNetwork) = ERGM.Extension.mcmc_defaults(_n_comparisons(rnet))

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

# The (z, p) columns of the coefficient table: `NetworkCore.z_pvalues` (floored at
# floatmin, never 0.0; NaN standard errors give NaN p-values), with a
# coefficient fixed at ∓Inf by a boundary statistic reported as R prints it —
# z = ∓Inf, p = 0 — rather than the NaN its zero standard error would give.
function _z_and_p(result::RankERGMResult)
    zp = z_pvalues(result.coefficients, result.std_errors)
    z, p = zp.z, zp.p
    if !isempty(result.separated_terms)
        # The separation policy: no inference on a fit with no finite maximum
        fill!(z, NaN)
        fill!(p, NaN)
        return z, p
    end
    for k in eachindex(result.coefficients)
        if isinf(result.coefficients[k])
            z[k] = result.coefficients[k]
            p[k] = 0.0
        elseif result.inference_withheld
            # A swap-MPLE fit with the default `se`: no z or p is built on the naive
            # pseudo-Hessian standard errors (see `fit_ergm_rank`)
            z[k] = NaN
            p[k] = NaN
        end
    end
    return z, p
end

# THE sentence for the withheld inference of a swap-MPLE fit with the default `se`, shared
# by `approximations` and (as its error) `confint`
const _WITHHELD_NOTE =
    "z values, p-values and confidence intervals withheld: the inverse " *
    "pseudo-Hessian standard errors of the swap pseudo-likelihood treat the " *
    "overlapping swap comparisons as independent and are too small (2–3.9× " *
    "narrower than the MLE's on Newcomb's fraternity ranks), so no test or " *
    "interval is built on them — refit with method=:mcmle (the default) or " *
    "se=:bootstrap; se=:hessian opts in to the naive Wald table"

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
            " (" * (result.termination === nothing ? "" :
                    _termination_detail(result.termination) * "; ") *
            "max t-ratio $(_fmt3(maximum(c.t_ratios))), Hotelling p " *
            "$(_fmt3(c.hotelling_p)), step length γ $(_fmt3(c.step_length)) " *
            "after $(c.iterations) iteration$(c.iterations == 1 ? "" : "s"))"
        return "MCMLE did not converge$detail: the sampled statistics at the " *
               "returned coefficients are not yet indistinguishable from the " *
               "observed ones, so the point estimates and standard errors are " *
               "unreliable — increase maxiter/n_samples/burnin, or refit with " *
               "init=coef(fit)"
    end
    if !isempty(result.separated_terms)
        return "the swap-MPLE did not converge: " *
               separation_caveat(result.separated_terms)
    end
    names = [name(t) for t in result.model.terms]
    nan = [names[k] for k in eachindex(names) if isnan(result.coefficients[k])]
    if !isempty(nan) && result.n_kept == 0
        return "coefficient(s) $(join(nan, ", ")) are NaN: not identified — every " *
               "swap comparison changes a statistic that was dropped at the " *
               "boundary of its attainable range, so no comparison is left to " *
               "estimate them on"
    end
    return "the swap-MPLE Newton iteration did not converge (Newton hit maxiter); " *
           "the coefficients are the last iterate and the standard errors and " *
           "p-values are meaningless"
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

# A coefficient that is NaN in a converged fit: its statistic is not
# identified on the swap comparisons fitted (no swap changes it, or it is a
# linear combination of the preceding statistics there), R's NA
function _unidentified_note(result::RankERGMResult)
    result.converged || return nothing
    names = [name(t) for t in result.model.terms]
    nan = [names[k] for k in eachindex(names) if isnan(result.coefficients[k])]
    isempty(nan) && return nothing
    return "coefficient(s) $(join(nan, ", ")) are NaN: not identified on the swap " *
           "comparisons fitted (no single swap of the observed ranking changes the " *
           "statistic, or it is a linear combination of the preceding statistics " *
           "there; R reports NA); the other coefficients are estimated without them"
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
# The ecosystem's one sentence for a bootstrap that excluded replicates without
# a finite refit, used verbatim by the warning, `show` and `approximations`
const _BOOT_EXCLUSION_BIAS =
    "The standard errors are conditional on a finite refit: the excluded " *
    "replicates are the extreme ones, so the standard errors are biased downward."

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
           "refits (fit.boot_replicates). " * _BOOT_EXCLUSION_BIAS
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
# calls for a top-level value — NetworkCore.jl's convention (one multi-line 2-arg
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
    elseif result.method === :mcmle && result.termination !== nothing
        println(io, "  Termination: ", _termination_detail(result.termination))
    end
    println(io, "Std. errors: ",
            result.se_type === :bootstrap ? "parametric bootstrap" :
            result.se_type === :fisher ?
                "inverse Fisher information from the final MCMC sample, plus " *
                "the Monte-Carlo component" :
                "inverse pseudo-Hessian")
    for note in (_fixed_coefficient_note(result), _unidentified_note(result),
                 _undefined_se_note(result), _boot_exclusion_note(result))
        note === nothing || println(io, "  ", note)
    end
    println(io)
    # The printed table IS `coeftable(result)` (a NetworkCore.CoefficientTable
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
        println(io, "point estimates are those of a pseudo-likelihood (not the MCMC MLE that R")
        println(io, "ergm.rank computes; they differ from it systematically). The standard")
        println(io, "errors are a parametric bootstrap: they do not treat the overlapping swap")
        println(io, "comparisons as independent, and describe the swap-MPLE, not the MLE.")
    elseif result.inference_withheld
        println(io, "Note: z values and p-values are not reported (NaN). This model was fit by")
        println(io, "swap-based maximum pseudolikelihood; the standard errors shown are the")
        println(io, "inverse pseudo-Hessian ones, which treat the overlapping swap comparisons")
        println(io, "as independent and are anticonservative (too small: 2-3.9x narrower than")
        println(io, "the MLE's on Newcomb's fraternity ranks), so no test or interval is built")
        println(io, "on them. For inference refit with method=:mcmle (the default; R")
        println(io, "ergm.rank's estimator) or se=:bootstrap; se=:hessian requests the naive")
        println(io, "Wald table explicitly.")
    else
        println(io, "Warning: this model was fit by swap-based maximum pseudolikelihood.")
        println(io, "The pairwise-swap comparisons overlap, so the standard errors (inverse")
        println(io, "pseudo-Hessian) ignore that dependence and are expected to be")
        println(io, "anticonservative; the p-values should be treated as a rough guide.")
        println(io, "Refit with `se=:bootstrap` for a parametric-bootstrap covariance.")
    end
end

# ============================================================================
# The shared result-metadata protocol (NetworkCore.jl `src/results.jl`)
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
fit = fit_ergm_rank(as_rank_network(m), [RankDeference()]; method=:mple)
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
fit = fit_ergm_rank(as_rank_network(m), [RankDeference()]; method=:mple)
is_exact(fit)                          # false: a swap pseudo-likelihood
approximations(fit)[1]                 # says so in words
```
"""
is_exact(::RankERGMResult) = false

"""
    se_method(result::RankERGMResult) -> Symbol

What the reported standard errors ACTUALLY are: `:fisher` (the default
`method=:mcmle`: the inverse Fisher information from the final MCMC sample plus
the Monte-Carlo component), `:hessian` (`method=:mple`: the inverse negative
Hessian of the swap pseudo-likelihood) or `:bootstrap` (the parametric
bootstrap of `fit_ergm_rank(...; method=:mple, se=:bootstrap)`). Read straight
off the fit.

# Example
```julia
using ERGMRank, Random
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
rnet = as_rank_network(m)
se_method(fit_ergm_rank(rnet, [RankDeference()]; method=:mple))      # :hessian
se_method(fit_ergm_rank(rnet, [RankDeference()]; method=:mple, se=:bootstrap,
                        n_boot=10, rng=Xoshiro(1)))                   # :bootstrap
se_method(fit_ergm_rank(rnet, [RankNodeICov([1, 2, 3, 4])]; n_samples=256,
                        bridge_rungs=0, rng=Xoshiro(1)))              # :fisher (the default MCMC MLE)
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
        result.inference_withheld && push!(out, _WITHHELD_NOTE)
    end
    # Non-convergence, a dropped boundary statistic, undefined standard errors
    # and excluded bootstrap replicates are part of what the fit actually did,
    # so they are reported here as well as warned about at fit time (never
    # only in a log line).
    result.converged || push!(out, _nonconvergence_caveat(result))
    for note in (_fixed_coefficient_note(result), _unidentified_note(result),
                 _undefined_se_note(result), _boot_exclusion_note(result))
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
        "AlterSwap chain (" *
        (result.termination === nothing ? "n_samples draws per iteration" :
         "$(result.termination.n_samples) draws in the final sample; stopped by the " *
         _termination_detail(result.termination)) * "), so the estimates carry " *
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
# `NetworkCore.check_statsapi(fit; strict=true)` pins all ten verbs in the tests.
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
nobs(fit_ergm_rank(rnet, [RankDeference()]; method=:mple)) == 12   # true: 4·3·2/2 swap comparisons
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
fit = fit_ergm_rank(as_rank_network(m), [RankDeference()]; method=:mple)
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
reports — `se_method(result)` says whether they are the MCMC MLE's
(`:fisher`), the parametric bootstrap of `se=:bootstrap`, or the inverse
pseudo-Hessian of a swap-MPLE fit. For the **default** swap-MPLE fit
(`method=:mple` without `se=`), whose pseudo-Hessian standard errors are too
small, `confint` refuses with an `ArgumentError` (see [`fit_ergm_rank`](@ref));
with an explicit `se=:hessian` the naive, over-narrow intervals are returned.

# Example
```julia
using ERGMRank, Random
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
rnet = as_rank_network(m)
fit = fit_ergm_rank(rnet, [RankNodeICov([1, 2, 3, 4])]; n_samples=256,
                    bridge_rungs=0, rng=Xoshiro(1))
ci = confint(fit)                             # 1×2
ci[1, 1] < coef(fit)[1] < ci[1, 2]            # true
size(confint(fit; level=0.9)) == (1, 2)       # true
pl = fit_ergm_rank(rnet, [RankDeference()]; method=:mple)
try
    confint(pl)
catch err
    err isa ArgumentError                     # true: withheld for a swap-MPLE fit with the default `se`
end
```
"""
function StatsAPI.confint(result::RankERGMResult; level::Real=0.95)
    0 < level < 1 || throw(ArgumentError("confint: level must be in (0, 1) (got $level)"))
    # The separation policy: a fit with no finite maximum has no interval
    isempty(result.separated_terms) ||
        return fill(NaN, length(result.coefficients), 2)
    result.inference_withheld && throw(ArgumentError(
        "confint: no interval is reported for a swap-MPLE fit with the default `se` — " *
        _WITHHELD_NOTE))
    q = quantile(Normal(), 1 - (1 - level) / 2)
    θ, se = result.coefficients, result.std_errors
    return hcat(θ .- q .* se, θ .+ q .* se)
end

"""
    coefnames(result::RankERGMResult) -> Vector{String}

The coefficient labels, in `coef(result)` order: the term names
(`rank.deference`, `rank.nonconformity.localAND`, …), the labels
[`coeftable`](@ref) and `show` print. A method of `StatsAPI.coefnames`
(R's `names(coef(fit))`); ERGMRank's labels are `rank.`-prefixed where
ergm.rank prints `deference`, `nonconformity`, … (see the README). A fresh
vector on every call, so changing it leaves the fit alone.

# Example
```julia
using ERGMRank
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
fit = fit_ergm_rank(as_rank_network(m),
                    [RankDeference(), RankNodeICov([10, 20, 30, 40]; label="age")];
                    method=:mple)
coefnames(fit) == ["rank.deference", "rank.nodeicov.age"]   # true
coefnames(fit) == coeftable(fit).names                      # true
```
"""
StatsAPI.coefnames(result::RankERGMResult) = String[name(t) for t in result.model.terms]

"""
    coeftable(result::RankERGMResult) -> NetworkCore.CoefficientTable

The R-style coefficient table (`Estimate`, `Std.Error`, `z value`,
`Pr(>|z|)`) as an inspectable `NetworkCore.CoefficientTable` — exactly the table
`show(result)` prints, built from the same vectors (a method of
`StatsAPI.coeftable`). The p-values come from `NetworkCore.z_pvalues` (two-sided
normal, floored at `floatmin(Float64)` so a finite z never prints as `0.0`); a
coefficient fixed at ∓Inf by a boundary statistic has z = ∓Inf and p = 0, as R
prints it. For a swap-MPLE fit with the default `se` (`method=:mple` without `se=`) the z
and p columns are `NaN`: no test is built on its pseudo-Hessian standard
errors (see [`fit_ergm_rank`](@ref)). Rows can be read by index or by term
name.

# Example
```julia
using ERGMRank
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
fit = fit_ergm_rank(as_rank_network(m), [RankDeference(), RankNodeICov([10, 20, 30, 40])];
                    method=:mple)
fit.converged                                       # true
tbl = coeftable(fit)
tbl["rank.deference"].estimate == coef(fit)[1]     # true
tbl[2].std_error == stderror(fit)[2]                # true
```
"""
function StatsAPI.coeftable(result::RankERGMResult)
    z, p = _z_and_p(result)
    return CoefficientTable(coefnames(result),
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
    fit_ergm_rank(rnet::RankNetwork, terms; method=:mcmle, kwargs...) -> RankERGMResult

Fit a rank-order ERGM `P(y) ∝ exp(θ'g(y))` on the complete-ordering space
(the `CompleteOrderReference`), by one of two estimators:

- **`method=:mcmle`** (default) — the **MCMC maximum likelihood estimate**,
  the estimator of R `ergm.rank`: starting from the swap-MPLE (or `init`),
  each iteration samples the model statistics along an AlterSwap Metropolis
  chain at the current θ and takes a Hummel-style partial Newton step toward
  the observed statistics; convergence is declared by R ergm 4's confidence
  stopping rule (the iteration is ERGM.jl's `ERGM.Extension.mcmle_solve`, the one
  `ERGM.mcmle` runs).
  Standard errors are the inverse Fisher information of the final sample plus
  the Monte-Carlo component; the log-likelihood is a path-sampling bridge
  estimate. See the "MCMC MLE" section below. On Newcomb's fraternity ranks
  it reproduces `ergm.rank`'s coefficients within `ergm.rank`'s own
  Monte-Carlo spread (asserted by the provenanced golden fixtures, for a
  two-term and a three-term model), in 10–30 s at the default budget.
- **`method=:mple`** — **swap-based maximum pseudo-likelihood**, an estimator
  of this package that `ergm.rank` does not have: for each ego `i` and each
  unordered alter pair {j, k}, the conditional probability of the observed
  ranking `y` against the single alternative `y_swapped` (ego's ranks of j
  and k exchanged) — `P(y | {y, y_swapped})`, the two-state conditional,
  *not* "the order of j and k given the rest of the rankings": a non-adjacent
  swap also reorders j and k relative to the alters ranked between them — is
  logistic in `θ'[g(y) − g(y_swapped)]`, and the product of these
  conditionals is maximized by Newton-Raphson with step-halving. Fast
  (milliseconds) and deterministic; a *different estimator* from the MLE (see
  the warning below). It is the MCMLE's starting point, and useful for
  screening models.

**How far the swap-MPLE is from the MLE depends on the model.** On Newcomb
week 1, `rank.deference + rank.nonconformity("all")`: MCMLE (R)
`[-0.1531, -0.00659]`, swap-MPLE `[-0.1409, -0.00585]` — 16× and 13× R's own
seed-to-seed spread apart, i.e. a systematic (not Monte-Carlo) difference,
0.30 and 0.43 of an MLE standard error. On week 2 with
`rank.deference + rank.nonconformity("localAND") + rank.inconsistency(week 1)`:
MCMLE (R) `[-0.2053, -0.0044, -0.1377]`, swap-MPLE `[-0.1636, -0.0100,
-0.1263]` — 0.89, 0.83 and 0.76 of an MLE standard error, the nonconformity
coefficient 2.3× the MLE's. In both, the pseudo-Hessian standard errors are
2–3.9× too small.

!!! warning "What is and is not claimed for the swap-MPLE"
    **No consistency result is established here.** **The swap
    pseudo-likelihood never equals the likelihood** — there is no rank
    analogue of dyad independence under which the product of two-state swap
    conditionals is the likelihood: the comparisons within one ego's row
    must form a total order, so they are never independent, even for a term
    that decomposes over egos. On the 4-actor example network with
    `RankNodeICov([1, 2, 3, 4])`, exact enumeration of all 1296 orderings
    gives the MLE `θ = −0.0767` (log-likelihood `−7.015`) while the
    swap-MPLE is `−0.0880` (pseudo-log-likelihood `−8.057`); `method=:mcmle`
    lands on the MLE up to Monte-Carlo error (pinned by the test suite). The
    estimator's large-sample behaviour in this setting has not been
    characterized, and MPLE for dependent ERGMs is known to be biased in
    finite samples.

[`ergm_rank`](@ref) is the statnet-style name (after the `ergm.rank`
package) of the same function.

# Standard errors and inference of the swap-MPLE

The inverse negative pseudo-Hessian treats the overlapping swap comparisons
as independent, so the standard errors it gives are too small. Under
`method=:mple`:

- **default (`se` not given)** — the point estimates and the pseudo-Hessian
  standard errors are reported, **but no inference built on them**: the z
  values and p-values of `coeftable`/`show` are `NaN` (with a note saying
  why), `confint` refuses with an `ArgumentError`, and `approximations(fit)`
  records it (`fit.inference_withheld`).
- `se=:hessian` passed **explicitly** — the written opt-in to the naive Wald
  table (z, p and intervals from the pseudo-Hessian), printed with the
  anticonservatism warning.
- `se=:bootstrap` — parametric bootstrap: simulate `n_boot` rank networks from
  the fitted model at θ̂ with [`simulate_rank_ergm`](@ref) (AlterSwap
  Metropolis), refit the swap MPLE on each, and report the empirical covariance
  of the refits, with z, p and intervals. The point estimates are unchanged;
  only the covariance is replaced. It measures the sampling variability of
  the swap-MPLE — in simulation its 95 % intervals covered the true
  coefficients at close to the nominal rate where the Hessian ones did not
  (see the estimation guide) — but it does not remove the swap-MPLE's bias
  relative to the MLE. This is the same option, with the same keywords and
  the same semantics, as `ERGM.mple`'s, and it runs on the ONE shared
  `NetworkCore.bootstrap_cov` loop. The replicates are successive draws of one
  chain: at the default `boot_interval` the thinning is ESS-aware (draws
  whose model statistics have an effective sample size below `n_boot/2` are
  taken again, once, at an interval up to 8× longer; see [`gof`](@ref)). A
  replicate on which the swap MPLE does not exist (a statistic at the
  boundary of its attainable range in the *simulated* ranking) is
  **excluded** from the covariance: it is a `NaN` row of
  `fit.boot_replicates`, the exclusion is warned about once and recorded in
  `approximations(fit)`, and fewer than 2 finite refits is an
  `ArgumentError`. `se=:bootstrap` is refused (an `ArgumentError`) when a
  coefficient of the fit itself is fixed at ±Inf, because no ranking can be
  simulated at an infinite coefficient.

When the pseudo-Hessian at the solution is not negative definite (a
coefficient the swap comparisons do not identify) the standard errors are
`NaN`, with a warning from `newton_fit` and an entry in `approximations` —
never a finite number from an indefinite matrix.

The swap pseudo-likelihood is a logistic likelihood on the swap-difference
rows with the response identically `true`, so it is maximized with the
ecosystem's shared `NetworkCore.logistic_derivatives` kernel and
`NetworkCore.newton_fit` Newton–Raphson-with-step-halving optimizer (the same
bindings ERGM.jl's MPLE runs on).

# Boundary statistics under `method=:mple` (R's `drop`)

A statistic whose observed value no single swap can lower (or raise) has no
finite swap-MPLE: the pseudo-log-likelihood increases monotonically as its
coefficient goes to `-Inf` (or `+Inf`). `RankInconsistency(rnet)` fitted to
`rnet` itself is the textbook case (the observed ranking is *at* zero
inconsistency). As R `ergm` does under its default `drop=TRUE`, such a
coefficient is **fixed at ∓Inf with standard error 0** (z = ∓Inf, p = 0), a
warning quotes R's sentence ("observed statistic(s) … are at their smallest
attainable values"), and the remaining coefficients are estimated on the swap
comparisons the dropped statistics do not change — the exact limit of the
pseudo-likelihood. A statistic observed at an end of its attainable range
(a count observed at 0) is dropped too, even when no swap changes it.
`dof(fit)` counts only the finite coefficients and `approximations` records
the drop. A statistic that no swap changes, or that is a linear combination
of the preceding statistics on the comparisons fitted, has no identifiable
coefficient: it is reported as `NaN` (R's `NA`), with a warning, and the
others are estimated without it.

# Non-convergence

A swap-MPLE fit that exhausts `maxiter` is returned with `converged == false`,
a warning at fit time, a caveat printed directly under `Converged: false`, and
an entry in `approximations`: the coefficients are the last Newton iterate and
the standard errors are unreliable.

# MCMC MLE (`method=:mcmle`, the default)

The estimator of R `ergm.rank`. Each iteration draws `n_samples` statistics
vectors along the AlterSwap Metropolis chain at the current θ (the ONE
`ERGM.mh_toggle!` kernel, the running statistics kept current from the
accepted swaps' change statistics — never a recomputation per draw), then
takes the Hummel partial Newton step `θ += γ · Σ̂⁻¹ (g_obs − ḡ)`: the step
length `γ` starts at `gamma0` and grows (at most doubling per iteration) while
the observed statistics lie outside the sampled cloud (the 95 % Mahalanobis
radius), reaching 1 once the cloud covers them, and each step is capped at
Euclidean norm `max_step_norm`. Convergence is declared only at `γ = 1` and
by the stopping rule `termination`: `:confidence` (default) is R ergm 4's
equivalence test — the estimating equations at the updated coefficients
(this sample importance-reweighted to them) lie with `conv_confidence`
confidence inside the tolerance ellipsoid `x'(conv_precision·Σ̂)⁻¹x ≤ 1`,
and a test that fails near the solution enlarges the next sample (at most
doubling, up to `max_n_samples`); `:hotelling` is the fixed-sample-size rule
of `ERGM.mcmc_convergence` (every t-ratio `|g_obs − ḡ| / sd(g)` below
`conv_threshold` and a Hotelling T² test non-significant at
`hotelling_alpha`). The step, the stopping rule, the boost, the final sample
and the covariance are ERGM.jl's `ERGM.Extension.mcmle_solve` — the iteration
`ERGM.mcmle` runs — with the AlterSwap chain as the sampler.

**A Monte-Carlo Newton step is taken at every iteration, the first
included** (statnet's order, and `ERGM.mcmle`'s): a fit whose start already
passes the tests is still refined by one step from that sample, so the
swap-MPLE start is never returned as the MCMLE. The returned θ̂ is the update
from the sample that passed, and that sample is the final sample: the
standard errors, `fit.mcmc_samples` and `fit.mcmc_convergence` come from it.
`fit.termination` records the rule, its p-value and the size of that sample.
A fit that ends unconverged draws a fresh sample at the returned
coefficients for them instead.

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
The integral is composite Simpson's rule over `bridge_rungs` segments (an odd
number is raised by one), each grid point one seeded chain of `bridge_samples` draws (the `ERGM.Extension.bridge_integrate`
estimator, with the uniform ranking model as the exact reference). It is
run only after the final sample, so coefficients and standard errors are
bit-identical with and without it; **`bridge_rungs=0` skips it** and
`loglik`, `aic`, `bic` are `NaN`, `show` prints "not estimated" and
`approximations` records it.

**Boundary statistics (R's `drop`).** A statistic whose observed value is
an end of its attainable range — a count (nonconformity, the unweighted
inconsistency, deference) observed at 0 — has no finite MLE. R `ergm`'s
`drop=TRUE` would fix its coefficient at `∓Inf` and estimate the rest by MCMC
on the rankings that share the observed value, but the AlterSwap sampler
cannot be held there: those rankings are not connected by single swaps (on 4
actors the 138 rankings with local nonconformity 0 fall into 22 swap classes),
so the held chain would fit a smaller model. The MCMLE therefore refuses such
a model with an `ArgumentError` that says so and points at `method=:mple`,
whose pseudo-likelihood applies the drop exactly. (ergm.rank's terms declare
no bound, so R does not drop here either: its sampler stops moving.)
`drop=false` refuses the same models with the strict-mode message.

**Boundary and separated starts.** A statistic at the boundary of its range
only under single swaps of the observed ranking (the swap-MPLE start would be
`∓Inf`, but the observed value is not an end of the attainable range) is a
local extremum, which does not show that no MLE exists: refused with an
`ArgumentError` unless `init=` is supplied, in which case a warning says so
and the convergence tests decide. A separated swap-MPLE (R: "The MPLE does
not exist!") and a coefficient the swap-MPLE cannot identify (`NaN`: no swap
changes its statistic, or the statistic is a linear combination of the
preceding ones) are refused the same way, pointing at `init=`.

**Reproducibility.** All randomness flows through `rng`; with `n_chains > 1`
the `n_samples` draws are split over chains seeded from `rng` in order and
run on separate tasks, so a fit depends only on `rng` and `n_chains`, never
on the thread count (`n_chains` never defaults to `Threads.nthreads()`).

# Keyword Arguments

Common:
- `method::Symbol=:mcmle`: `:mcmle` or `:mple` (anything else is an
  `ArgumentError` naming both)
- `maxiter::Int`: the iteration cap — MCMLE iterations (default 60, R ergm's
  `MCMLE.maxit`) or Newton iterations of the swap-MPLE (default 100)
- `se`: `:fisher` (default, and the only option) under `:mcmle`; under
  `:mple`, not given (pseudo-Hessian standard errors, inference withheld),
  `:hessian` (the naive Wald table, opted into) or `:bootstrap`; anything
  else is an `ArgumentError` from the shared `NetworkCore.check_se` naming the
  method's vocabulary
- `rng::AbstractRNG=Random.default_rng()`: source of all randomness (the
  bootstrap, the MCMC chains, the bridge) — a fixed `rng` reproduces the fit
  exactly, on any thread count
- `verbose::Bool=false`: print MCMLE progress
- `drop::Bool=true`: R's `control.ergm(drop=)`. Under `:mple` a statistic at
  the boundary of its attainable range is fixed at `∓Inf` and the rest
  estimated (above); `drop=false` refuses the model with an `ArgumentError`
  naming the statistics, before any fitting. Under `:mcmle` such a model is
  refused either way (above); `drop=false` gives the strict-mode message

Swap-MPLE (`method=:mple`):
- `tol::Float64=1e-8` (passed to `newton_fit`)
- `n_boot::Int=100`: number of bootstrap replicates (`se=:bootstrap` only;
  at least 2)
- `boot_burnin=nothing`, `boot_interval=nothing`: MCMC controls for the
  bootstrap simulations; `nothing` resolves to the swap-scaled defaults of
  [`simulate_rank_ergm`](@ref) — `burnin = 20 n_swaps`, `interval = max(100,
  n_swaps ÷ 10)` with `n_swaps = n (n − 1)(n − 2) / 2` (ERGM.jl's dyad-scaled
  rule applied to the swap count); with the default `boot_interval` the
  thinning is ESS-aware (above), an explicit integer is honoured as given

MCMC MLE (`method=:mcmle`; the vocabulary of `ERGM.mcmle`):
- `n_samples::Int=1024`: MCMC draws per iteration, in total over the chains
  (the confidence rule may boost it, up to `max_n_samples`, default
  `16·n_samples`)
- `termination::Symbol=:confidence`: the stopping rule — R ergm 4's
  equivalence test (`conv_precision=0.1`, `conv_confidence=0.99`: the
  estimating equations at the updated coefficients lie, with 99 % confidence,
  inside the tolerance ellipsoid `x'(0.1·Σ)⁻¹x ≤ 1`; a failed test near the
  solution enlarges the next sample), or `:hotelling`, the t-ratio +
  Hotelling rule at a fixed sample size (`conv_threshold`, `hotelling_alpha`)
- `burnin=nothing`, `interval=nothing`: steps discarded per chain and steps
  between draws; `nothing` resolves to the swap-scaled rule above
- `n_chains::Int=1`: independent chains per iteration (and for the final
  sample), each burned in from the observed ranking, seeded from `rng`
- `init=nothing`: starting coefficients (default: the swap-MPLE)
- `gamma0::Float64=0.1`: initial Hummel step length
- `max_step_norm::Float64=5.0`: cap on the norm of each Newton step
- `conv_threshold::Float64=0.1`, `hotelling_alpha::Float64=0.05`: the
  `:hotelling` rule's tests (also what `fit.mcmc_convergence` reports on the
  final sample under either rule)
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
x = [1.0, 2.0, 3.0, 4.0]
mle = fit_ergm_rank(rnet, [RankNodeICov(x)]; n_samples=512, rng=Xoshiro(1))
mle.method                        # :mcmle — the default, R ergm.rank's estimator
mle.mcmc_convergence.step_length  # 1.0 at convergence
isfinite(loglikelihood(mle))      # true: the bridge estimate (bridge_rungs=16)
coeftable(mle)                    # the table `show(mle)` prints
approximations(mle)               # what the estimator did NOT do exactly

# The swap pseudo-likelihood: fast and deterministic, a different estimator.
# (RankDeference() + RankNonconformity() on these 4 actors is perfectly
# separated — the swap-MPLE does not exist, and the fit says so)
pl = fit_ergm_rank(rnet, [RankDeference(), RankNodeICov([10, 20, 30, 40])]; method=:mple)
pl.converged                      # true
round.(coef(pl); digits=4)        # [-0.0658, -0.0082]: two swap-MPLE coefficients
pl.inference_withheld             # true: z, p and confint are not reported by default
```
"""
function fit_ergm_rank(rnet::RankNetwork, terms::Vector{<:AbstractERGMTerm};
                       method::Symbol=:mcmle,
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
                       termination::Symbol=:confidence,
                       conv_precision::Float64=0.1,
                       conv_confidence::Float64=0.99,
                       max_n_samples::Union{Nothing,Int}=nothing,
                       bridge_rungs::Int=16,
                       bridge_samples::Union{Nothing,Int}=nothing,
                       drop::Bool=true,
                       verbose::Bool=false,
                       rng::Random.AbstractRNG=Random.default_rng())
    method in (:mple, :mcmle) || throw(ArgumentError(
        "fit_ergm_rank: method must be one of (:mple, :mcmle) (got $(repr(method))): " *
        ":mple is the swap pseudo-likelihood, :mcmle the MCMC maximum likelihood " *
        "estimate of R ergm.rank"))
    # `se` resolves per method, and is validated against that method's
    # vocabulary by the ONE shared validator
    # Under `:mple`, `se=nothing` (the default) is `:hessian` with the
    # inference built on it withheld; an explicit `se=:hessian` opts in to the
    # naive Wald table (ERGM.mple's pattern — for a rank model EVERY swap-MPLE
    # is a pseudo-likelihood of dependent comparisons, so there is no exempt
    # case)
    naive_opt_in = se === :hessian
    if method === :mcmle
        se in (:hessian, :bootstrap) && throw(ArgumentError(
            "fit_ergm_rank(method=:mcmle): se must be one of (:fisher,) (got " *
            "$(repr(se))) — the MCMC MLE's standard errors are the inverse Fisher " *
            "information plus the Monte-Carlo component; :hessian and :bootstrap " *
            "belong to the swap pseudo-likelihood: pass method=:mple with them"))
        se = check_se(something(se, :fisher), (:fisher,); context="fit_ergm_rank(method=:mcmle)")
        maxiter = something(maxiter, 60)
    else
        se = check_se(something(se, :hessian), (:hessian, :bootstrap);
                      context="fit_ergm_rank(method=:mple)")
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
                           termination=termination, conv_precision=conv_precision,
                           conv_confidence=conv_confidence,
                           max_n_samples=max_n_samples,
                           bridge_rungs=bridge_rungs, bridge_samples=bridge_samples,
                           drop=drop, verbose=verbose, rng=rng)
    end

    fit = _rank_mple_fit(model.terms, rnet; maxiter=maxiter, tol=tol, drop=drop)
    if fit.unidentified || fit.separated
        # warned by `_rank_mple_fit`, naming the terms
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
                "keep the default standard errors, which report standard error 0 for " *
                "the fixed coefficient and the inverse-Hessian errors of the rest."))
        end
        # A separated swap-MPLE is where Newton stopped, not an estimate: there
        # is no fitted model to simulate the replicates from
        fit.separated && throw(ArgumentError(
            "fit_ergm_rank: se=:bootstrap is not available for a separated swap-MPLE " *
            "(the coefficients on $(join(fit.separated_terms, ", ")) run to ±Inf, so " *
            "the returned values are where Newton stopped, not a model to simulate " *
            "from). Remove or merge the separating term(s), or fit the MCMC MLE with " *
            "method=:mcmle and a starting point init=."))
        burnin, interval = _resolve_mcmc_controls(rnet, boot_burnin, boot_interval)
        vcov, std_errors, boot_replicates =
            _rank_bootstrap_cov(model, fit.θ; n_boot=n_boot,
                                boot_burnin=burnin,
                                boot_interval=interval,
                                adaptive=boot_interval === nothing,
                                maxiter=maxiter, tol=tol, rng=rng)
    end

    return RankERGMResult(model, fit.θ, std_errors, vcov, fit.loglik,
                          fit.converged, se, boot_replicates, fit.n_kept, :mple,
                          nothing, nothing, nothing, 0,
                          se === :hessian && !naive_opt_in, nothing,
                          fit.separated_terms)
end

# One term, or a tuple of terms, instead of a vector: wrapped. A `Network` or
# a matrix instead of a `RankNetwork`: refused with the conversion named.
fit_ergm_rank(rnet::RankNetwork, term::AbstractERGMTerm; kwargs...) =
    fit_ergm_rank(rnet, AbstractERGMTerm[term]; kwargs...)
fit_ergm_rank(rnet::RankNetwork, terms::Tuple; kwargs...) =
    fit_ergm_rank(rnet, collect(AbstractERGMTerm, terms); kwargs...)
fit_ergm_rank(x::Union{Network, AbstractMatrix}, terms; kwargs...) =
    throw(_not_a_rank_network("fit_ergm_rank", x))

# The attainable range of a rank statistic, `(lo, hi)`, declared through
# ERGM.jl's extension-API generic `ERGM.Extension.attainable_range` (R ergm's
# `minval`/`maxval`, which `ergm.checkextreme.model` compares the observed
# statistics with: an observed value equal to `lo` fixes the coefficient at
# -Inf, one equal to `hi` at +Inf, R's default `drop=TRUE`), so that
# `ERGM.Extension.extreme_statistics(terms, rnet)` reads them like ERGM.jl's
# own. ergm.rank's terms declare none (its MCMLE runs on, and on an observed
# `rank.inconsistency` of 0 stops with "Unconstrained MCMC sampling did not
# mix at all"), so these are the bounds the definitions give: the counts
# (deference, both nonconformity variants, the unweighted inconsistency, and
# a weighted one whose weights are all ≥ 0) cannot go below 0. No upper bound
# is declared, and the covariate terms declare none (ERGM.jl's default):
# there, only the swap design can show a statistic at a bound
# (`_rank_mple_fit`), which under the MCMLE is a swap-local extremum, not
# proof that no MLE exists.
ERGM.Extension.attainable_range(::Union{RankDeference,RankNonconformity}, ::RankNetwork) =
    (0.0, Inf)
ERGM.Extension.attainable_range(t::RankInconsistency, ::RankNetwork) =
    (t.weights === nothing || all(>=(0), t.weights)) ? (0.0, Inf) : (-Inf, Inf)

# Core swap MPLE: the design rows d = g(y) − g(y with j,k swapped) over every
# (ego, unordered alter pair), fitted by ERGM.jl's one pseudo-likelihood fitter
# `ERGM.Extension.mple_fit_design`. Shared by `fit_ergm_rank`, the MCMLE's start and
# the parametric bootstrap's refits (`warn=false`: a boundary statistic in a
# SIMULATED replicate is not a fact about the user's data, and
# `_rank_bootstrap_cov` reports those replicates once, in aggregate).
#
# Returns `(θ, se, vcov, loglik, converged, separated, separated_terms,
# grad_norm, n_kept, unidentified)`: `separated` says the pseudo-likelihood has
# no finite maximum (then `converged` is false), `grad_norm` is ‖∇ℓ(θ)‖ at the
# returned iterate (quoted by the non-convergence warning), `n_kept` the number
# of comparisons the finite coefficients were fitted on (every row, or the
# rows the dropped columns do not touch) and `unidentified` that every
# comparison changes a dropped statistic, so nothing is left to estimate the
# other coefficients on (they are NaN, `converged` false).
function _rank_mple_fit(terms::Vector{AbstractERGMTerm}, rnet::RankNetwork;
                        maxiter::Int=100, tol::Float64=1e-8, warn::Bool=true,
                        drop::Bool=true,
                        extreme::Vector{Tuple{Int,Symbol}}=ERGM.Extension.extreme_statistics(terms, rnet))
    p = length(terms)
    names = [name(t) for t in terms]
    D = _rank_design(terms, copy(rnet))
    m = size(D, 1)
    ones_m = ones(m)

    # The response is identically TRUE, so a column whose nonzero swap
    # differences all share one sign is a statistic at the boundary of its
    # attainable range: the observed ranking minimizes (`:min`, all d < 0) or
    # maximizes (`:max`, all d > 0) it over every single swap, the gradient
    # keeps one sign at every θ and no finite MPLE exists. `extreme` seeds the
    # test with the statistics whose observed value is the end of their
    # attainable range (`ERGM.Extension.extreme_statistics`), which the design cannot
    # show when no swap changes them. R's drop semantics — ∓Inf, SE 0, the
    # rest fitted on the untouched rows — through ERGM.jl's one fitter; the
    # rows here are swap comparisons, not dyads, and ergm.rank has no drop of
    # its own, so the warning (emitted here, with those two words) says so.
    boundary = ERGM.Extension.boundary_columns(D, ones_m, ones_m; fixed=extreme)
    drop || isempty(boundary) ||
        _refuse_no_drop(names, boundary; context="fit_ergm_rank(method=:mple)")
    warn && !isempty(boundary) &&
        ERGM.Extension.warn_boundary(names, boundary; context="fit_ergm_rank",
                                     noun="swap comparisons",
                                     note="R ergm reports the same for its binary " *
                                          "terms; ergm.rank's swap pseudo-likelihood " *
                                          "has no drop")

    # The swap pseudo-likelihood IS a logistic likelihood on the D rows with
    # the response identically TRUE — the observed order is always the
    # "success" — so it is ERGM.jl's compressed-row fitter with every row a
    # class of one comparison that is a success: the drop, the aliased columns
    # (a statistic no swap changes, or a linear combination of the preceding
    # ones: NaN, R's NA), the shared `NetworkCore.newton_fit` /
    # `logistic_derivatives` kernel and NetworkCore's exact separation verdict
    # on the design actually fitted. Never paste a Newton or logistic loop
    # back in. Its warnings are this package's (rows are comparisons).
    # The fit returns what it dropped, aliased and fitted, the rows it fitted
    # on and its separation verdict, so nothing is recomputed here.
    r = ERGM.Extension.mple_fit_design(D, ones_m, ones_m, names; maxiter=maxiter, tol=tol,
                                       warn=false, context="fit_ergm_rank",
                                       estimate="swap-MPLE", noun="swap comparisons",
                                       extreme=boundary)
    θ = r.coefficients
    rows = r.kept_rows
    fitted = r.fitted
    nan = r.aliased
    n_kept = length(rows)
    unidentified = n_kept == 0 && !isempty(nan)
    # No coefficient fitted: every row has probability σ(0) = ½, so the exact
    # limit of the pseudo-likelihood is n_kept·log(½) — and with no row left
    # it is undefined (NaN), never the 0.0 of an empty product, which would
    # print as "Pseudo-AIC: 0.0" (a perfect fit)
    loglik = !isempty(fitted) ? r.loglik : n_kept == 0 ? NaN : n_kept * log(0.5)
    converged = r.converged && !unidentified
    grad_norm = 0.0
    if !isempty(fitted)
        # the ecosystem's one separation message, from the verdict the fit
        # computed on the design it actually fitted
        r.separated && warn &&
            warn_separation("fit_ergm_rank", r.verdict, names[fitted];
                            estimate="swap-MPLE", note=_SEPARATION_NOTE)
        r.converged || r.separated ||
            (grad_norm = _grad_norm(logistic_derivatives(D[rows, fitted], trues(n_kept)),
                                    θ[fitted]))
    elseif unidentified
        grad_norm = NaN
    end
    if warn && !isempty(nan)
        unidentified ? _warn_unidentified(names[nan]) : _warn_aliased_rank(names[nan])
    end
    return (θ=θ, se=r.std_errors, vcov=r.var_cov, loglik=loglik, converged=converged,
            separated=r.separated, separated_terms=r.separated_terms,
            grad_norm=grad_norm, n_kept=n_kept, unidentified=unidentified)
end

_grad_norm(derivatives, θ) = norm(derivatives(θ)[2])

# Appended to the shared separation warning: what separation means for a
# swap design, and R's own sentence
const _SEPARATION_NOTE =
    "For a swap design this means some combination of the model's statistics is " *
    "never lowered by any single swap of the observed ranking. R ergm warns " *
    "\"The MPLE does not exist!\" for a separated MPLE design. The 4-actor " *
    "example network separates under two terms."

_warn_unidentified(nan::Vector{String}) =
    @warn "fit_ergm_rank: coefficient(s) $(join(nan, ", ")) are not identified " *
          "and are returned as NaN with converged == false: every swap " *
          "comparison changes a statistic that sits at the boundary of its " *
          "attainable range (dropped, coefficient fixed at ±Inf), so no " *
          "comparison is left to estimate them on. Remove the boundary " *
          "statistic from the model."

# R's two sentences for a statistic without a coefficient (`ergm`'s "not
# varying", glm's linear dependence), in the swap design's words
_warn_aliased_rank(nan::Vector{String}) =
    @warn "fit_ergm_rank: statistic(s) $(join(nan, ", ")) are not identified on " *
          "the swap comparisons fitted — no single swap of the observed ranking " *
          "changes them, or they are linear combinations of the preceding " *
          "statistics there — so their coefficients are reported as NaN and the " *
          "other coefficients are estimated without them (R ergm warns \"Model " *
          "statistics ... are not varying\" or \"linear dependence\", and reports " *
          "NA). Remove the term(s)."

# `drop=false`: a statistic at the boundary of its attainable range is refused
# instead of fixed at ∓Inf (R's `drop=FALSE` keeps the term and fits a model
# whose "MLE is poorly defined"; that is not implemented)
function _refuse_no_drop(names::Vector{String}, boundary::Vector{Tuple{Int,Symbol}};
                         context::AbstractString,
                         remedy::AbstractString="Use the default drop=true — the " *
                             "coefficient fixed at ±Inf and the rest estimated, as R " *
                             "ergm does — or remove the term(s).")
    lo = [names[j] for (j, s) in boundary if s === :min]
    hi = [names[j] for (j, s) in boundary if s === :max]
    parts = String[]
    isempty(lo) || push!(parts, "observed statistic(s) $(join(lo, ", ")) are at their " *
                                "smallest attainable values (coefficient -Inf)")
    isempty(hi) || push!(parts, "observed statistic(s) $(join(hi, ", ")) are at their " *
                                "largest attainable values (coefficient +Inf)")
    throw(ArgumentError(
        "$context: " * join(parts, "; ") * ". No finite estimate exists, and " *
        "drop=false asks to keep such a statistic in the model (R's `drop=FALSE`, " *
        "whose \"MLE is poorly defined\"), which is not implemented. $remedy"))
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
# the empirical covariance. The loop is the shared `NetworkCore.bootstrap_cov`; this
# supplies only the two callbacks that are ERGMRank's — and the exclusion of
# replicates without a finite refit, as `ERGM.mple`'s bootstrap excludes them.
function _rank_bootstrap_cov(model::RankERGMModel, θ̂::Vector{Float64};
                             n_boot::Int, boot_burnin::Int, boot_interval::Int,
                             adaptive::Bool=false,
                             maxiter::Int, tol::Float64,
                             rng::Random.AbstractRNG)
    # The replicates are successive draws of ONE chain, and the covariance
    # below treats them as independent: at the default interval the thinning
    # is ESS-aware (`_simulate_thinned`), so autocorrelated draws are redrawn
    # at a longer interval instead of understating the bootstrap's variance
    simulate(rng, B) = _simulate_thinned(model.network, model.terms, θ̂;
                                         n_sim=B, burnin=boot_burnin,
                                         interval=boot_interval, adaptive=adaptive,
                                         rng=rng, context="fit_ergm_rank: se=:bootstrap",
                                         what="bootstrap replicates",
                                         keyword="boot_interval").draws

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
          "empirical covariance of the $n_ok finite refits. " * _BOOT_EXCLUSION_BIAS *
          " This is about the simulated replicates, not about the observed ranking. " *
          "`fit.boot_replicates` holds every refit (NaN rows excluded); " *
          "`approximations(fit)` records the exclusion."
    V = Matrix{Float64}(cov(replicates[ok, :]))
    return V, sqrt.(max.(diag(V), 0.0)), replicates
end

# =============================================================================
# MCMC maximum likelihood (`method=:mcmle`): the estimator of R ergm.rank
# =============================================================================
#
# The iteration is ERGM.mcmle's, CALLED: `ERGM.Extension.mcmle_solve` (Hummel step,
# confidence or Hotelling stopping rule, sample boost, singular-covariance
# stop, final sample, Fisher + Monte-Carlo covariance) with the AlterSwap
# chain as the `draw` callback and the swap-MPLE as the start, and
# `ERGM.Extension.bridge_integrate` for the path-sampling integral. Nothing
# statistical is re-derived or copied here: the kernel is `mh_toggle!`, the
# proposal and change statistics are `simulate_rank_ergm`'s.

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
    spawn_all(n_chains) do c
        chain_stats[c] = _rank_stats_chain(copy(rnet), terms, θ, counts[c],
                                           burnin, interval,
                                           Random.Xoshiro(seeds[c]))
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
# u ∈ [0, 1]: d/du log Z(θ_u) = E_{θ_u}[g]'θ (the thermodynamic identity).
# The integral is ERGM.jl's `ERGM.Extension.bridge_integrate` (composite Simpson's
# rule over `nrungs` segments, an odd count raised by one); this supplies the
# sampler — one seeded AlterSwap chain per grid point — and the exact
# reference, the uniform ordering model. Rungs run on separate tasks, each on
# its own `Xoshiro(seed)` drawn from `rng` in order, so the estimate is
# thread-count independent. θ = 0 is exact (the ratio is 0).
function _rank_bridge_logZ_ratio(rnet::RankNetwork, terms::Tuple, θ::Vector{Float64};
                                 nrungs::Int, n_samples::Int, burnin::Int,
                                 interval::Int, rng::Random.AbstractRNG)
    nrungs >= 1 || throw(ArgumentError("nrungs must be at least 1 (bridge_rungs=0 " *
                                       "skips the log-likelihood estimate)"))
    all(iszero, θ) && return 0.0
    m = iseven(nrungs) ? nrungs : nrungs + 1
    seeds = rand(rng, UInt64, m + 1)
    rung_mean(θu, k) = vec(mean(_rank_stats_chain(copy(rnet), terms, Vector{Float64}(θu),
                                                  n_samples, burnin, interval,
                                                  Random.Xoshiro(seeds[k])), dims=1))
    return ERGM.Extension.bridge_integrate(rung_mean, zeros(length(θ)), θ; rungs=m, threaded=true)
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

# A start the MCMLE cannot take: the swap-MPLE fixes a coefficient at ∓Inf
# although the observed statistic is NOT at an end of its attainable range —
# the observed ranking is extreme only among its single-swap neighbours, which
# does not show that no MLE exists (a statistic at its attainable bound is
# held there instead, R's drop)
function _refuse_boundary_start(fixed::Vector{String}, init_given::Bool)
    msg = "fit_ergm_rank(method=:mcmle): the observed statistic(s) $(join(fixed, ", ")) " *
          "are at the boundary of their range under every single swap of the " *
          "observed ranking (the swap-MPLE used as the starting point fixes the " *
          "coefficient at ±Inf), but not at a bound of their attainable range: the " *
          "observed ranking may be a local extremum only, and the MLE may exist"
    if init_given
        @warn msg * "; proceeding from `init=` — the convergence tests decide, and " *
                    "a fit returned with converged == false must not be interpreted."
        return nothing
    end
    throw(ArgumentError(msg * ". Supply a finite starting point with `init=` to let " *
                        "the convergence tests decide, or remove the term. " *
                        "(`method=:mple` reports the swap pseudo-likelihood's own " *
                        "answer: that coefficient fixed at ±Inf, the others estimated " *
                        "on the comparisons it does not change.)"))
end

# A statistic observed at an end of its attainable range has no finite MLE.
# R ergm's answer (`drop=TRUE`) fixes the coefficient at ∓Inf and estimates
# the rest by MCMC on the networks that share the observed value — a sampler
# held on that level set. For rankings the AlterSwap chain cannot do that: the
# rankings sharing an extreme value are not connected by single swaps (on 4
# actors the 24 rankings with nonconformity 0 are 24 isolated points, and the
# 138 with local nonconformity 0 fall into 22 swap classes), so a held chain
# would estimate a model restricted to the observed ranking's class, not the
# drop. Refused, saying why and pointing at the swap-MPLE, which does drop.
function _refuse_extreme_mcmle(names::Vector{String}, extreme::Vector{Tuple{Int,Symbol}})
    lo = [names[j] for (j, s) in extreme if s === :min]
    hi = [names[j] for (j, s) in extreme if s === :max]
    parts = String[]
    isempty(lo) || push!(parts, "$(join(lo, ", ")) at the smallest value it can take")
    isempty(hi) || push!(parts, "$(join(hi, ", ")) at the largest value it can take")
    throw(ArgumentError(
        "fit_ergm_rank(method=:mcmle): the observed statistic(s) are at an end of " *
        "their attainable range (" * join(parts, "; ") * "), so no finite MLE exists. " *
        "R ergm's drop=TRUE would fix the coefficient at ±Inf and estimate the rest " *
        "on the rankings that share the observed value, but the AlterSwap sampler " *
        "cannot be held there: those rankings are not connected by single swaps, so " *
        "the estimate would be that of a smaller model (ergm.rank itself does not " *
        "drop; its sampler stops moving). Fit method=:mple, whose pseudo-likelihood " *
        "applies R's drop exactly (the coefficient fixed at ±Inf, the rest estimated " *
        "on the swap comparisons that leave the statistic unchanged), or remove the " *
        "term."))
end

function _rank_mcmle(model::RankERGMModel;
                     n_samples::Int, burnin::Union{Nothing,Int},
                     interval::Union{Nothing,Int}, maxiter::Int, n_chains::Int,
                     init::Union{Nothing,AbstractVector{<:Real}},
                     gamma0::Float64, max_step_norm::Float64,
                     conv_threshold::Float64, hotelling_alpha::Float64,
                     termination::Symbol, conv_precision::Float64,
                     conv_confidence::Float64,
                     max_n_samples::Union{Nothing,Int},
                     bridge_rungs::Int, bridge_samples::Union{Nothing,Int},
                     drop::Bool=true, verbose::Bool, rng::Random.AbstractRNG)
    n_samples >= 2 || throw(ArgumentError(
        "fit_ergm_rank(method=:mcmle): n_samples must be ≥ 2 (got $n_samples)"))
    n_chains >= 1 || throw(ArgumentError(
        "fit_ergm_rank(method=:mcmle): n_chains must be ≥ 1 (got $n_chains)"))
    maxiter >= 1 || throw(ArgumentError(
        "fit_ergm_rank(method=:mcmle): maxiter must be ≥ 1 (got $maxiter)"))
    bridge_rungs >= 0 || throw(ArgumentError(
        "fit_ergm_rank(method=:mcmle): bridge_rungs must be ≥ 0 (got $bridge_rungs); " *
        "0 skips the log-likelihood estimate (loglik/AIC/BIC are then NaN)"))
    termination in (:confidence, :hotelling) || throw(ArgumentError(
        "fit_ergm_rank(method=:mcmle): termination must be :confidence (R ergm 4's " *
        "stopping rule, the default) or :hotelling (got $(repr(termination)))"))
    max_n_samples === nothing || max_n_samples >= n_samples || throw(ArgumentError(
        "fit_ergm_rank(method=:mcmle): max_n_samples ($max_n_samples) is below " *
        "n_samples ($n_samples)"))

    rnet = model.network
    terms = model.terms
    ts = Tuple(terms)
    p = length(terms)
    term_names = [name(t) for t in terms]
    burnin, interval = _resolve_mcmc_controls(rnet, burnin, interval)

    # Statistics at an end of their attainable range: no finite MLE, and R's
    # drop cannot be carried out by this sampler (`_refuse_extreme_mcmle`);
    # drop=false is the strict refusal of the same models
    extreme = ERGM.Extension.extreme_statistics(terms, rnet)
    if !isempty(extreme)
        drop || _refuse_no_drop(term_names, extreme; context="fit_ergm_rank(method=:mcmle)",
                                remedy="Fit method=:mple, whose default drop=true fixes " *
                                       "the coefficient at ±Inf and estimates the rest, " *
                                       "or remove the term(s).")
        _refuse_extreme_mcmle(term_names, extreme)
    end

    # The start: the swap-MPLE, as ERGM.mcmle starts from the MPLE. Its own
    # warnings are silenced (a boundary, unidentified or separated start is
    # refused below with a sentence that names the way out).
    verbose && println("Getting initial estimates via swap-MPLE...")
    start = _rank_mple_fit(terms, rnet; maxiter=100, tol=1e-8, warn=false,
                           extreme=extreme)
    fixed = [term_names[k] for k in 1:p if isinf(start.θ[k])]
    isempty(fixed) || _refuse_boundary_start(fixed, init !== nothing)
    if init === nothing
        nan = [term_names[k] for k in 1:p if isnan(start.θ[k])]
        isempty(nan) || throw(ArgumentError(
            "fit_ergm_rank(method=:mcmle): the coefficient(s) of $(join(nan, ", ")) " *
            "have no starting value: on the swap comparisons of the observed ranking " *
            "no swap changes these statistics, or they are linear combinations of " *
            "the preceding ones, so the swap-MPLE cannot identify them (R ergm: " *
            "\"not varying\" / \"linear dependence\"). Remove the term(s), or " *
            "supply a starting point with `init=`."))
        start.separated && throw(ArgumentError(
            "fit_ergm_rank(method=:mcmle): the swap-MPLE used as the starting point " *
            "does not exist (separation of $(join(start.separated_terms, ", ")): a " *
            "combination of the model's " *
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
    # The iteration is ERGM.jl's `ERGM.Extension.mcmle_solve` — the one `ERGM.mcmle`
    # runs: a Hummel-stepped Monte-Carlo Newton update at EVERY iteration (the
    # first included, so the swap-MPLE start is never returned as "the
    # MCMLE"), R ergm 4's confidence stopping rule on the updated coefficients
    # with its sample-size boost (or the t-ratio + Hotelling rule), the final
    # sample, and the Fisher + Monte-Carlo covariance. This package supplies
    # the sampler: `n` AlterSwap draws at θ over `n_chains` chains seeded from
    # `rng` in order.
    draw(θd, n) = begin
        samples, chain_lengths = _rank_mcmc_sample(rnet, ts, Vector{Float64}(θd), n,
                                                   burnin, interval; rng=rng,
                                                   n_chains=n_chains)
        (samples=samples, chain_lengths=chain_lengths)
    end
    sol = ERGM.Extension.mcmle_solve(draw, θ; labels=term_names, n_samples=n_samples,
                            maxiter=maxiter, termination=termination,
                            conv_precision=conv_precision,
                            conv_confidence=conv_confidence,
                            conv_threshold=conv_threshold,
                            hotelling_alpha=hotelling_alpha, gamma0=gamma0,
                            max_step_norm=max_step_norm,
                            max_n_samples=something(max_n_samples, 16 * n_samples),
                            target=obs_stats, verbose=verbose,
                            context="fit_ergm_rank(method=:mcmle)")
    θ = sol.coef
    converged = sol.converged
    tests = sol.tests
    final_samples = Matrix{Float64}(sol.final.samples)
    convergence = MCMLEConvergence((sol.iterations, sol.step_length, tests.t_ratios,
                                    tests.hotelling_p, tests.n_eff))
    term = (rule=termination, p_value=sol.termination_p, precision=conv_precision,
            confidence=conv_confidence, n_samples=size(final_samples, 1))
    vcov_fisher, V, std_errors = sol.vcov_fisher, sol.vcov, sol.se

    converged || @warn "fit_ergm_rank(method=:mcmle): MCMLE did not converge in " *
        "maxiter=$maxiter iterations ($(_termination_detail(term)); last max t-ratio " *
        "$(_fmt3(maximum(tests.t_ratios))), Hotelling p $(_fmt3(tests.hotelling_p)), " *
        "step length γ $(_fmt3(sol.step_length))): the estimates are the last iterate " *
        "and the standard errors are unreliable; increase maxiter/n_samples/burnin, " *
        "or refit from these coefficients (init=coef(fit))"

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
                          final_samples, vcov_fisher, bridge_rungs, false, term)
end

"""
    ergm_rank(rnet::RankNetwork, terms; kwargs...) -> RankERGMResult

The statnet-style name of [`fit_ergm_rank`](@ref) (the same function, `===`),
after the R `ergm.rank` package — the ecosystem convention of one
`fit_<model>` name and one statnet-style name per model package. Like R's
`ergm(..., response=, reference=~CompleteOrder)`, it fits the MCMC MLE by
default.

# Example
```julia
using ERGMRank
m = [0 3 2 1;
     3 0 1 2;
     1 3 0 2;
     2 1 3 0]
rnet = as_rank_network(m)
ergm_rank === fit_ergm_rank            # true
fit = ergm_rank(rnet, [RankNodeICov([1, 2, 3, 4])]; n_samples=256, bridge_rungs=0)
fit.method                             # :mcmle — the MCMC MLE, R ergm.rank's estimator
```
"""
const ergm_rank = fit_ergm_rank

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
# ESS-aware thinning of the draws behind `gof` and `se=:bootstrap`
# =============================================================================

# Below this many draws an effective sample size cannot be measured usefully
const _ESS_MIN_DRAWS = 20

# The model statistics of each draw, one row per draw
_draw_statistics(draws::Vector{RankNetwork}, terms) =
    Float64[compute(t, d) for d in draws, t in terms]

# Effective sample size of successive draws of one chain: the smallest, over
# the model statistics, of the Geyer initial-sequence estimate — ERGM.jl's
# `mcmc_convergence(...).n_eff` (public), clamped to [2, n]. NaN when it
# cannot be measured (too few draws, or a statistic that never moved).
function _draws_ess(stats::Matrix{Float64})
    n = size(stats, 1)
    n >= _ESS_MIN_DRAWS || return NaN
    any(j -> allequal(view(stats, :, j)), axes(stats, 2)) && return NaN
    return Float64(mcmc_convergence(stats, vec(mean(stats, dims=1))).n_eff)
end

# `n_sim` draws at θ for a consumer that treats them as independent (the GOF
# envelopes and p-values; the bootstrap covariance). With `adaptive` (the
# caller's interval was the default) the thinning is ESS-aware, as ERGM.jl's
# `gof` is: when the effective sample size of the model statistics over the
# draws is below n_sim/2, the draws are taken again, once, at an interval
# scaled up by n_sim/ESS (at most 8×); whatever is left below n_sim/4 is
# warned about. Returns the draws, their model statistics, the interval used
# and the measured ESS.
function _simulate_thinned(rnet::RankNetwork, terms, θ;
                           n_sim::Int, burnin::Int, interval::Int, adaptive::Bool,
                           rng::Random.AbstractRNG, context::AbstractString,
                           what::AbstractString, keyword::AbstractString)
    draws = simulate_rank_ergm(rnet, terms, θ; n_sim=n_sim, burnin=burnin,
                               interval=interval, rng=rng)
    stats = _draw_statistics(draws, terms)
    ess = _draws_ess(stats)
    if adaptive && isfinite(ess) && ess < n_sim / 2
        interval *= min(8, ceil(Int, n_sim / max(ess, 1.0)))
        draws = simulate_rank_ergm(rnet, terms, θ; n_sim=n_sim, burnin=burnin,
                                   interval=interval, rng=rng)
        stats = _draw_statistics(draws, terms)
        ess = _draws_ess(stats)
    end
    isfinite(ess) && ess < n_sim / 4 && @warn "$context: the $n_sim $what are " *
        "autocorrelated (effective sample size of the model statistics " *
        "$(round(Int, ess)) at interval $interval), so they carry less information " *
        "than $n_sim independent draws would; pass a larger `$keyword`."
    return (draws=draws, stats=stats, interval=interval, ess=ess)
end

# =============================================================================
# Goodness of fit
# =============================================================================

# --- Auxiliary statistics: features of a ranking that are NOT model terms ----

# Mean rank each actor receives (column means of the rank matrix), sorted
# ascending: the popularity profile, the rank analogue of a degree distribution
function _received_rank_profile!(out::AbstractVector{Float64}, rnet::RankNetwork)
    n = rnet.n
    y = rnet.ranks
    @inbounds for j in 1:n
        total = 0
        for i in 1:n
            total += y[i, j]
        end
        out[j] = total / (n - 1)
    end
    sort!(out)
    return out
end

# Number of unordered pairs {i, j} with |y_ij − y_ji| = d, d = 0:(n−2): how
# far apart the two members of a dyad rank each other (reciprocity of standing)
function _dyad_rank_difference!(out::AbstractVector{Float64}, rnet::RankNetwork)
    n = rnet.n
    y = rnet.ranks
    fill!(out, 0.0)
    @inbounds for i in 1:n, j in (i + 1):n
        out[abs(y[i, j] - y[j, i]) + 1] += 1.0
    end
    return out
end

const _TAU_LABELS = ["[-1,-0.6)", "[-0.6,-0.2)", "[-0.2,0.2]", "(0.2,0.6]", "(0.6,1]"]

_tau_bin(τ::Float64) = τ < -0.6 ? 1 : τ < -0.2 ? 2 : τ <= 0.2 ? 3 : τ <= 0.6 ? 4 : 5

# Number of unordered ego pairs {i, l} by Kendall's τ between their rankings
# of the n − 2 alters they share, in five bins: how alike two egos rank the
# others (the distribution behind the global nonconformity count)
function _ego_agreement!(out::AbstractVector{Float64}, rnet::RankNetwork)
    n = rnet.n
    y = rnet.ranks
    fill!(out, 0.0)
    npairs = (n - 2) * (n - 3) ÷ 2
    @inbounds for i in 1:n, l in (i + 1):n
        concordant = 0
        for a in 1:n
            (a == i || a == l) && continue
            for b in (a + 1):n
                (b == i || b == l) && continue
                concordant += (y[i, a] > y[i, b]) == (y[l, a] > y[l, b])
            end
        end
        τ = (2 * concordant - npairs) / npairs
        out[_tau_bin(τ)] += 1.0
    end
    return out
end

# The structural rank statistics that are not covariates; those not already in
# the model are evaluated as auxiliaries
_structural_auxiliaries(terms) = AbstractERGMTerm[
    t for t in (RankDeference(), RankNonconformity(:all), RankNonconformity(:localAND))
    if !any(m -> name(m) == name(t), terms)]

"""
    gof(result::RankERGMResult; n_sim=100, burnin=nothing, interval=nothing,
        rng=Random.default_rng()) -> GOFResult

Goodness-of-fit assessment of a fitted rank-order ERGM: rank networks are
simulated from the fitted model with [`simulate_rank_ergm`](@ref) (AlterSwap
Metropolis sampling) and features of the observed ranking are compared with
their simulated distributions.

This is a method of the shared `NetworkCore.gof` generic; it returns the shared
`NetworkCore.GOFResult` (observed value, simulation envelope, and two-sided
Monte-Carlo p-value per statistic), with these panels:

- `"model statistics"` — the model's own terms. **For a `method=:mcmle` fit
  these are matched in expectation by construction** (the MLE solves
  `E[g(Y)] = g(y)`), so their p-values only check that the fit converged and
  say nothing about how well the model describes the ranking; for a
  `method=:mple` fit they show how far the swap pseudo-likelihood estimate
  is from reproducing them. The fit of the model is judged on the auxiliary
  panels below, none of which is a model statistic:
- `"structural statistics (not in the model)"` — those of `rank.deference`,
  `rank.nonconformity` and `rank.nonconformity.localAND` the model does not
  contain (absent when it contains all three).
- `"mean received rank (sorted)"` — each actor's mean received rank, sorted
  ascending (labels `1`…`n`): the popularity profile, the rank analogue of a
  degree distribution.
- `"dyadic rank difference"` — the number of dyads {i, j} with
  `|y_ij − y_ji| = d`, `d = 0, …, n − 2`: reciprocity of standing.
- `"ego agreement (Kendall tau)"` — the number of ego pairs by Kendall's τ
  between their rankings of the alters they share, in five bins (needs at
  least 4 actors).

# Keyword Arguments
- `n_sim::Int=100`: Number of simulated rank networks (at least 1)
- `burnin`, `interval`, `rng`: passed to [`simulate_rank_ergm`](@ref); `nothing`
  (the default) resolves to its swap-scaled rule — `burnin = 20 n_swaps`,
  `interval = max(100, n_swaps ÷ 10)` with `n_swaps = n (n − 1)(n − 2) / 2`
  — and an explicit integer is honoured. With the default `interval` the
  thinning is **ESS-aware** (as in `ERGM.gof`): the envelopes and p-values
  treat the draws as independent, so when the effective sample size of the
  model statistics over the `n_sim` draws (at least 20) is below `n_sim/2`
  the draws are taken again, once, at an interval scaled up by `n_sim/ESS`
  (at most 8×); an effective sample size still below `n_sim/4` is warned
  about.

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
fit = fit_ergm_rank(as_rank_network(m), [RankDeference()]; method=:mple)
g = gof(fit; n_sim=50, burnin=100, interval=10, rng=Xoshiro(1))
g.statistics[1].labels            # ["rank.deference"]
[s.name for s in g.statistics]    # the model statistics and four auxiliary panels
```
"""
function gof(result::RankERGMResult; n_sim::Int=100,
             burnin::Union{Nothing,Int}=nothing,
             interval::Union{Nothing,Int}=nothing,
             rng::Random.AbstractRNG=Random.default_rng())
    n_sim >= 1 || throw(ArgumentError("gof: n_sim must be at least 1 (got $n_sim)"))
    rnet = result.model.network
    terms = result.model.terms
    n = rnet.n
    _check_finite_coefficients("gof", terms, result.coefficients, true)
    adaptive = interval === nothing
    burnin, interval = _resolve_mcmc_controls(rnet, burnin, interval)
    sim = _simulate_thinned(rnet, terms, result.coefficients; n_sim=n_sim,
                            burnin=burnin, interval=interval, adaptive=adaptive,
                            rng=rng, context="gof", what="simulated rankings",
                            keyword="interval")
    sims = sim.draws

    panels = GOFStatistic[]
    obs_stats = [compute(term, rnet) for term in terms]
    push!(panels, GOFStatistic("model statistics", [name(term) for term in terms],
                               obs_stats, sim.stats))

    aux = _structural_auxiliaries(terms)
    if !isempty(aux)
        push!(panels, GOFStatistic("structural statistics (not in the model)",
                                   [name(t) for t in aux],
                                   [compute(t, rnet) for t in aux],
                                   Float64[compute(t, s) for s in sims, t in aux]))
    end

    # One auxiliary panel from a fill-in-place statistic of `width` values
    function panel(title, labels, fillfn, width)
        obs = fillfn(Vector{Float64}(undef, width), rnet)
        simulated = Matrix{Float64}(undef, n_sim, width)
        for (r, s) in enumerate(sims)
            fillfn(view(simulated, r, :), s)
        end
        return GOFStatistic(title, labels, obs, simulated)
    end
    push!(panels, panel("mean received rank (sorted)", string.(1:n),
                        _received_rank_profile!, n))
    push!(panels, panel("dyadic rank difference", string.(0:(n - 2)),
                        _dyad_rank_difference!, n - 1))
    n >= 4 && push!(panels, panel("ego agreement (Kendall tau)", _TAU_LABELS,
                                  _ego_agreement!, length(_TAU_LABELS)))

    return GOFResult(panels; model="Rank-Order ERGM")
end

# =============================================================================
# Precompile workload: the README path (fit by both estimators, print, the
# bootstrap, simulate, gof) on the 4-actor example, so the first call in a
# session does not pay for compiling it. Everything random runs on a fixed
# `Xoshiro`, and warnings of the deliberately tiny MCMC budget are discarded.
# =============================================================================

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    workload_ranks = [0 3 2 1;
                      3 0 1 2;
                      1 3 0 2;
                      2 1 3 0]
    @compile_workload begin
        Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
            rnet = as_rank_network(workload_ranks)
            terms = [RankDeference(), RankNodeICov([1.0, 2.0, 3.0, 4.0])]
            rng = Random.Xoshiro(1)
            io = IOBuffer()
            fit = fit_ergm_rank(rnet, terms; n_samples=64, burnin=50, interval=5,
                                bridge_rungs=2, bridge_samples=16, rng=rng)
            show(io, MIME("text/plain"), fit)
            show(io, fit)
            coeftable(fit); confint(fit); approximations(fit)
            pl = fit_ergm_rank(rnet, terms; method=:mple)
            show(io, MIME("text/plain"), pl)
            fit_ergm_rank(rnet, terms; method=:mple, se=:bootstrap, n_boot=4,
                          boot_burnin=20, boot_interval=5, rng=rng)
            simulate_rank_ergm(pl; n_sim=2, burnin=20, interval=5, rng=rng)
            show(io, MIME("text/plain"),
                 gof(pl; n_sim=4, burnin=20, interval=5, rng=rng))
        end
    end
end

end # module
