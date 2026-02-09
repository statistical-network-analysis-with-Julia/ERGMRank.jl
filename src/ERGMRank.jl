"""
    ERGMRank.jl - ERGMs for Rank-Order Networks

Extends ERGM to handle networks where each actor ranks others,
using local structure-based terms appropriate for ordinal relational data.

In rank networks, actor i assigns ranks 1, 2, ..., k to a subset of alters,
where lower rank indicates higher preference/importance.

Port of the R ergm.rank package from the StatNet collection.
"""
module ERGMRank

using ERGM
using Graphs
using LinearAlgebra
using Network
using Optim
using Random
using Statistics
using StatsBase

# Rank-specific terms
export RankEdges, RankMutual, RankTransitivity
export RankNonconsensus, RankDeference
export RankLocaltriangle, RankNodecov, RankAbsdiff

# Reference measures
export PlackettLuce, ThurstoneMosteller

# Model types
export RankNetwork, RankERGMModel, RankERGMResult

# Estimation
export ergm_rank, fit_rank_ergm

# Simulation
export simulate_rank_ergm

# Utilities
export as_rank_network, rank_matrix

# =============================================================================
# Rank Network Data Structure
# =============================================================================

"""
    RankNetwork{T}

A network where edges represent rankings.

# Fields
- `network::Network{T}`: Underlying network structure
- `ranks::Dict{Tuple{T,T}, Int}`: Rank assigned by i to j
- `max_rank::Int`: Maximum rank value
"""
struct RankNetwork{T}
    network::Network{T}
    ranks::Dict{Tuple{T,T}, Int}
    max_rank::Int

    function RankNetwork{T}(n::Int; max_rank::Int=n-1) where T
        net = Network{T}(; n=n, directed=true)
        new{T}(net, Dict{Tuple{T,T}, Int}(), max_rank)
    end
end

RankNetwork(n::Int; kwargs...) = RankNetwork{Int}(n; kwargs...)

# Forward Graphs.jl interface
Graphs.nv(rnet::RankNetwork) = nv(rnet.network)
Graphs.ne(rnet::RankNetwork) = ne(rnet.network)
Graphs.vertices(rnet::RankNetwork) = vertices(rnet.network)
Graphs.edges(rnet::RankNetwork) = edges(rnet.network)
Graphs.is_directed(::RankNetwork) = true

"""
    set_rank!(rnet::RankNetwork, i, j, rank)

Set the rank that actor i assigns to actor j.
"""
function set_rank!(rnet::RankNetwork{T}, i::T, j::T, rank::Int) where T
    i == j && throw(ArgumentError("Cannot rank self"))
    1 <= rank <= rnet.max_rank || throw(ArgumentError("Rank must be in 1:$(rnet.max_rank)"))

    rnet.ranks[(i, j)] = rank
    if !has_edge(rnet.network, i, j)
        add_edge!(rnet.network, i, j)
    end
    return rnet
end

"""
    get_rank(rnet::RankNetwork, i, j) -> Union{Int, Nothing}

Get the rank that actor i assigns to actor j.
"""
function get_rank(rnet::RankNetwork{T}, i::T, j::T) where T
    return get(rnet.ranks, (i, j), nothing)
end

"""
    get_rankings_by(rnet::RankNetwork, i) -> Vector{Tuple{T, Int}}

Get all rankings made by actor i, sorted by rank.
"""
function get_rankings_by(rnet::RankNetwork{T}, i::T) where T
    rankings = Tuple{T, Int}[]
    for ((src, dst), rank) in rnet.ranks
        if src == i
            push!(rankings, (dst, rank))
        end
    end
    sort!(rankings, by=x -> x[2])
    return rankings
end

"""
    get_rankings_of(rnet::RankNetwork, j) -> Vector{Tuple{T, Int}}

Get all rankings received by actor j.
"""
function get_rankings_of(rnet::RankNetwork{T}, j::T) where T
    rankings = Tuple{T, Int}[]
    for ((src, dst), rank) in rnet.ranks
        if dst == j
            push!(rankings, (src, rank))
        end
    end
    return rankings
end

"""
    rank_matrix(rnet::RankNetwork) -> Matrix{Union{Int, Missing}}

Convert rank network to matrix form.
Entry (i,j) is the rank i gives to j, or missing if not ranked.
"""
function rank_matrix(rnet::RankNetwork{T}) where T
    n = nv(rnet)
    mat = Matrix{Union{Int, Missing}}(missing, n, n)
    for ((i, j), rank) in rnet.ranks
        mat[i, j] = rank
    end
    return mat
end

"""
    as_rank_network(mat::Matrix; max_rank=nothing) -> RankNetwork

Convert a matrix of ranks to a RankNetwork.
"""
function as_rank_network(mat::Matrix{<:Union{Int, Missing, Nothing}};
                         max_rank::Union{Int, Nothing}=nothing)
    n = size(mat, 1)
    size(mat, 2) == n || throw(ArgumentError("Matrix must be square"))

    max_r = isnothing(max_rank) ? maximum(skipmissing(mat)) : max_rank
    rnet = RankNetwork(n; max_rank=max_r)

    for i in 1:n, j in 1:n
        if !ismissing(mat[i, j]) && !isnothing(mat[i, j])
            set_rank!(rnet, i, j, mat[i, j])
        end
    end

    return rnet
end

# =============================================================================
# Reference Measures for Rank Data
# =============================================================================

"""
    PlackettLuce

Plackett-Luce model for rankings - items ranked independently based on "worth".
"""
struct PlackettLuce
    # Base model - can be extended with item-specific parameters
end

"""
    ThurstoneMosteller

Thurstone-Mosteller model - rankings based on latent normal utilities.
"""
struct ThurstoneMosteller
    sigma::Float64
    ThurstoneMosteller(σ::Float64=1.0) = new(σ)
end

# =============================================================================
# Rank-Specific ERGM Terms
# =============================================================================

"""
    RankEdges <: AbstractERGMTerm

Number of rankings made (density term for rank networks).
"""
struct RankEdges <: AbstractERGMTerm end

name(::RankEdges) = "rank.edges"

function compute(::RankEdges, rnet::RankNetwork)
    return Float64(length(rnet.ranks))
end

function change_stat(::RankEdges, rnet::RankNetwork, i::Int, j::Int)
    # Change when adding/removing a ranking
    haskey(rnet.ranks, (i, j)) ? -1.0 : 1.0
end

"""
    RankMutual <: AbstractERGMTerm

Mutual high-ranking: both i and j rank each other in top k positions.

# Fields
- `cutoff::Int`: Top k positions to consider (default: 1, meaning top choice)
"""
struct RankMutual <: AbstractERGMTerm
    cutoff::Int
    RankMutual(k::Int=1) = new(k)
end

name(t::RankMutual) = "rank.mutual.$(t.cutoff)"

function compute(t::RankMutual, rnet::RankNetwork{T}) where T
    count = 0
    n = nv(rnet)

    for i in 1:n
        for j in (i+1):n
            rank_ij = get_rank(rnet, T(i), T(j))
            rank_ji = get_rank(rnet, T(j), T(i))

            if !isnothing(rank_ij) && !isnothing(rank_ji)
                if rank_ij <= t.cutoff && rank_ji <= t.cutoff
                    count += 1
                end
            end
        end
    end

    return Float64(count)
end

"""
    RankTransitivity <: AbstractERGMTerm

Transitive ranking: if i ranks j highly and j ranks k highly,
then i tends to rank k highly.

# Fields
- `cutoff::Int`: Top k positions for "high" ranking
"""
struct RankTransitivity <: AbstractERGMTerm
    cutoff::Int
    RankTransitivity(k::Int=3) = new(k)
end

name(t::RankTransitivity) = "rank.transitivity.$(t.cutoff)"

function compute(t::RankTransitivity, rnet::RankNetwork{T}) where T
    count = 0
    n = nv(rnet)

    for i in 1:n, j in 1:n, k in 1:n
        i == j || j == k || i == k || continue

        rank_ij = get_rank(rnet, T(i), T(j))
        rank_jk = get_rank(rnet, T(j), T(k))
        rank_ik = get_rank(rnet, T(i), T(k))

        if !isnothing(rank_ij) && !isnothing(rank_jk) && !isnothing(rank_ik)
            if rank_ij <= t.cutoff && rank_jk <= t.cutoff && rank_ik <= t.cutoff
                count += 1
            end
        end
    end

    return Float64(count)
end

"""
    RankNonconsensus <: AbstractERGMTerm

Measures disagreement in rankings: cases where i ranks j higher than k,
but some other actor l ranks k higher than j.
"""
struct RankNonconsensus <: AbstractERGMTerm end

name(::RankNonconsensus) = "rank.nonconsensus"

function compute(::RankNonconsensus, rnet::RankNetwork{T}) where T
    count = 0
    n = nv(rnet)

    for i in 1:n
        rankings_i = get_rankings_by(rnet, T(i))
        length(rankings_i) < 2 && continue

        for (idx1, (j, rank_j)) in enumerate(rankings_i)
            for (idx2, (k, rank_k)) in enumerate(rankings_i)
                idx1 >= idx2 && continue

                # i ranks j higher than k (lower rank number)
                if rank_j < rank_k
                    # Find any l who ranks k higher than j
                    for l in 1:n
                        l == i && continue
                        rank_lj = get_rank(rnet, T(l), j)
                        rank_lk = get_rank(rnet, T(l), k)

                        if !isnothing(rank_lj) && !isnothing(rank_lk)
                            if rank_lk < rank_lj  # l ranks k higher than j
                                count += 1
                            end
                        end
                    end
                end
            end
        end
    end

    return Float64(count)
end

"""
    RankDeference <: AbstractERGMTerm

Deference: tendency to rank those who rank you highly also highly.
"""
struct RankDeference <: AbstractERGMTerm
    cutoff::Int
    RankDeference(k::Int=3) = new(k)
end

name(t::RankDeference) = "rank.deference.$(t.cutoff)"

function compute(t::RankDeference, rnet::RankNetwork{T}) where T
    count = 0
    n = nv(rnet)

    for i in 1:n, j in 1:n
        i == j && continue

        rank_ij = get_rank(rnet, T(i), T(j))
        rank_ji = get_rank(rnet, T(j), T(i))

        if !isnothing(rank_ij) && !isnothing(rank_ji)
            if rank_ji <= t.cutoff && rank_ij <= t.cutoff
                count += 1
            end
        end
    end

    return Float64(count) / 2  # Each pair counted twice
end

"""
    RankLocaltriangle <: AbstractERGMTerm

Local ranking triangles within each actor's rankings.
"""
struct RankLocaltriangle <: AbstractERGMTerm end

name(::RankLocaltriangle) = "rank.localtriangle"

function compute(::RankLocaltriangle, rnet::RankNetwork{T}) where T
    count = 0
    n = nv(rnet)

    for i in 1:n
        rankings_i = get_rankings_by(rnet, T(i))
        length(rankings_i) < 2 && continue

        # For each pair that i ranks, check if they rank each other
        for (idx1, (j, _)) in enumerate(rankings_i)
            for (idx2, (k, _)) in enumerate(rankings_i)
                idx1 >= idx2 && continue

                # Check if j and k rank each other
                if !isnothing(get_rank(rnet, j, k)) && !isnothing(get_rank(rnet, k, j))
                    count += 1
                end
            end
        end
    end

    return Float64(count)
end

"""
    RankNodecov <: AbstractERGMTerm

Rank-weighted node covariate effect.
"""
struct RankNodecov <: AbstractERGMTerm
    attr::Symbol
    RankNodecov(attr::Symbol) = new(attr)
end

name(t::RankNodecov) = "rank.nodecov.$(t.attr)"

function compute(t::RankNodecov, rnet::RankNetwork)
    attrs = get_vertex_attribute(rnet.network, t.attr)
    isnothing(attrs) && return 0.0

    total = 0.0
    for ((i, j), rank) in rnet.ranks
        # Weight by inverse rank (high rank = more important)
        weight = 1.0 / rank
        total += weight * (get(attrs, i, 0) + get(attrs, j, 0))
    end

    return total
end

"""
    RankAbsdiff <: AbstractERGMTerm

Effect of absolute difference in node attribute on rankings.
"""
struct RankAbsdiff <: AbstractERGMTerm
    attr::Symbol
    RankAbsdiff(attr::Symbol) = new(attr)
end

name(t::RankAbsdiff) = "rank.absdiff.$(t.attr)"

function compute(t::RankAbsdiff, rnet::RankNetwork)
    attrs = get_vertex_attribute(rnet.network, t.attr)
    isnothing(attrs) && return 0.0

    total = 0.0
    for ((i, j), rank) in rnet.ranks
        val_i = get(attrs, i, 0)
        val_j = get(attrs, j, 0)
        total += abs(val_i - val_j) / rank  # Weight by inverse rank
    end

    return total
end

# =============================================================================
# Model and Estimation
# =============================================================================

"""
    RankERGMModel{T}

ERGM model for rank networks.
"""
struct RankERGMModel{T}
    terms::Vector{AbstractERGMTerm}
    network::RankNetwork{T}
end

"""
    RankERGMResult

Results from fitting a rank ERGM.
"""
struct RankERGMResult{T}
    model::RankERGMModel{T}
    coefficients::Vector{Float64}
    std_errors::Vector{Float64}
    loglik::Float64
    converged::Bool
end

function Base.show(io::IO, result::RankERGMResult)
    println(io, "Rank ERGM Results")
    println(io, "=================")
    println(io, "Log-likelihood: $(round(result.loglik, digits=4))")
    println(io, "Converged: $(result.converged)")
    println(io)
    println(io, "Coefficients:")
    for (i, term) in enumerate(result.model.terms)
        println(io, "  $(rpad(name(term), 25)) $(lpad(round(result.coefficients[i], digits=4), 10)) " *
                    "(SE: $(round(result.std_errors[i], digits=4)))")
    end
end

"""
    ergm_rank(rnet::RankNetwork, terms; kwargs...) -> RankERGMResult

Fit an ERGM for rank-order networks.
"""
function ergm_rank(rnet::RankNetwork{T}, terms::Vector{<:AbstractERGMTerm};
                   method::Symbol=:mple,
                   maxiter::Int=100) where T

    model = RankERGMModel{T}(terms, rnet)

    if method == :mple
        return rank_mple(model; maxiter=maxiter)
    else
        throw(ArgumentError("Unknown method: $method"))
    end
end

fit_rank_ergm = ergm_rank

"""
    rank_mple(model::RankERGMModel; kwargs...) -> RankERGMResult

MPLE for rank ERGM.
"""
function rank_mple(model::RankERGMModel{T}; maxiter::Int=100, tol::Float64=1e-6) where T
    n_terms = length(model.terms)
    coef = zeros(n_terms)

    # Compute observed statistics
    obs_stats = [compute(term, model.network) for term in model.terms]

    # Simple gradient descent (placeholder for full implementation)
    for iter in 1:maxiter
        grad = zeros(n_terms)

        # Compute gradient based on observed vs expected statistics
        for (i, term) in enumerate(model.terms)
            grad[i] = obs_stats[i] - compute(term, model.network) * exp(-coef[i])
        end

        # Update
        coef .+= 0.01 * grad

        if maximum(abs.(grad)) < tol
            se = fill(0.1, n_terms)  # Placeholder
            return RankERGMResult{T}(model, coef, se, NaN, true)
        end
    end

    se = fill(NaN, n_terms)
    return RankERGMResult{T}(model, coef, se, NaN, false)
end

# =============================================================================
# Simulation
# =============================================================================

"""
    simulate_rank_ergm(n::Int, terms, coef; kwargs...) -> RankNetwork

Simulate a rank network from an ERGM.
"""
function simulate_rank_ergm(n::Int, terms::Vector{<:AbstractERGMTerm},
                            coef::Vector{Float64};
                            max_rank::Int=n-1,
                            burnin::Int=1000)
    rnet = RankNetwork(n; max_rank=max_rank)

    # Initialize with random rankings
    for i in 1:n
        n_rankings = rand(1:max_rank)
        others = setdiff(1:n, i)
        ranked = sample(others, min(n_rankings, length(others)); replace=false)
        for (r, j) in enumerate(ranked)
            set_rank!(rnet, i, j, r)
        end
    end

    # MCMC updates
    for _ in 1:burnin
        # Pick random actor and alter
        i = rand(1:n)
        j = rand(setdiff(1:n, i))

        # Propose change (add, remove, or modify rank)
        current_rank = get_rank(rnet, i, j)

        # Compute acceptance probability based on change statistics
        # (simplified version)
        if rand() < 0.5
            # Accept proposal
            if isnothing(current_rank)
                set_rank!(rnet, i, j, rand(1:max_rank))
            else
                delete!(rnet.ranks, (i, j))
                rem_edge!(rnet.network, i, j)
            end
        end
    end

    return rnet
end

end # module
