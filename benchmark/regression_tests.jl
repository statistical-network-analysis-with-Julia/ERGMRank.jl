#!/usr/bin/env julia
# benchmark/regression_tests.jl — allocation-regression assertions for the
# ERGMRank.jl hot loops. Standalone; run with
#     julia --project=benchmark benchmark/regression_tests.jl
#
# The per-swap change statistics (`swap_change`, the innermost loop of both
# the swap-MPLE design and every AlterSwap Metropolis proposal) and the
# Metropolis step on `ERGM.mh_toggle!` are allocation-free (measured 0 bytes
# for every term below). Any allocation appearing here is a performance
# regression — these tests assert the loops STAY allocation-free rather than
# tracking a noisy byte budget; the design builder is bounded by the matrix
# it must return plus a fixed workspace.

using ERGMRank
using ERGM
using Networks
using Random
using Test

function random_rank_network(rng::AbstractRNG, n::Int)
    m = zeros(Int, n, n)
    for i in 1:n
        perm = randperm(rng, n - 1)
        c = 0
        for j in 1:n
            i == j && continue
            c += 1
            m[i, j] = perm[c]
        end
    end
    return RankNetwork(m)
end

"Bytes allocated by `swap_change` on a pre-warmed call, worst over `swaps`."
function max_allocs_swap_change(term, rnet, swaps)
    worst = 0
    for (ego, j, k) in swaps
        swap_change(term, rnet, ego, j, k)                 # warm up / compile
        worst = max(worst, @allocated swap_change(term, rnet, ego, j, k))
    end
    return worst
end

"Bytes allocated by `_swap_delta!` over a term tuple on a pre-warmed call."
function delta_allocs(delta, ts, rnet, ego, j, k)
    ERGMRank._swap_delta!(delta, ts, rnet, ego, j, k)
    return @allocated ERGMRank._swap_delta!(delta, ts, rnet, ego, j, k)
end

"Bytes allocated by a sampler run of `B` burn-in steps (one draw)."
function sampler_allocs(rnet, terms, θ, B)
    simulate_rank_ergm(rnet, terms, θ; n_sim=1, burnin=B, interval=1, rng=Random.Xoshiro(1))
    return @allocated simulate_rank_ergm(rnet, terms, θ; n_sim=1, burnin=B, interval=1,
                                         rng=Random.Xoshiro(1))
end

"Bytes allocated by `_rank_design` on a pre-warmed call (the work copy is outside)."
function design_allocs(terms, rnet)
    work = copy(rnet)
    ERGMRank._rank_design(terms, work)
    return @allocated ERGMRank._rank_design(terms, work)
end

@testset "ERGMRank allocation regressions" begin
    n = 50
    rng = Random.Xoshiro(20260912)
    rnet = random_rank_network(Random.Xoshiro(1), n)
    swaps = Tuple{Int, Int, Int}[]
    while length(swaps) < 25
        ego, j, k = rand(rng, 1:n), rand(rng, 1:n), rand(rng, 1:n)
        (ego == j || ego == k || j == k) || push!(swaps, (ego, j, k))
    end
    terms = AbstractERGMTerm[RankDeference(), RankNonconformity(:all),
                             RankNonconformity(:localAND),
                             RankNodeICov(collect(1.0:n)),
                             RankInconsistency(random_rank_network(Random.Xoshiro(2), n)),
                             RankEdgeCov(rand(Random.Xoshiro(3), n, n))]

    @testset "swap_change is allocation-free" begin
        for term in terms
            @test max_allocs_swap_change(term, rnet, swaps) == 0
        end
    end

    @testset "_swap_delta! is allocation-free over a term tuple (p = 2, 6, 8)" begin
        for ts in (Tuple(terms[1:2]), Tuple(terms),
                   Tuple(AbstractERGMTerm[RankEdgeCov(rand(Random.Xoshiro(i), n, n))
                                          for i in 1:8]))
            delta = zeros(length(ts))
            @test delta_allocs(delta, ts, rnet, 3, 5, 9) == 0
        end
    end

    @testset "A Metropolis step on ERGM.mh_toggle! allocates nothing" begin
        θ = [-0.1, -0.005]
        a1 = sampler_allocs(rnet, terms[1:2], θ, 1_000)
        a2 = sampler_allocs(rnet, terms[1:2], θ, 2_000)
        @test a2 <= a1
    end

    @testset "_rank_design allocates the matrix plus a fixed workspace" begin
        small = random_rank_network(Random.Xoshiro(4), 17)
        m, p = 17 * 16 * 15 ÷ 2, 2
        @test design_allocs(terms[1:2], small) <= 8 * m * p + 4096
    end
end
