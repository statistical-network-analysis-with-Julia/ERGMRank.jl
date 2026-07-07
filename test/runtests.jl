using ERGMRank
using ERGM
using Random
using Statistics
using Test

# The 4-actor test network validated against R ergm.rank 4.1.2
# (higher rank value = higher standing)
function r_test_network()
    m = zeros(Int, 4, 4)
    m[1, 2] = 3; m[1, 3] = 2; m[1, 4] = 1
    m[2, 1] = 3; m[2, 3] = 1; m[2, 4] = 2
    m[3, 1] = 1; m[3, 2] = 3; m[3, 4] = 2
    m[4, 1] = 2; m[4, 2] = 1; m[4, 3] = 3
    return RankNetwork(m)
end

# Random valid rank network: each ego's ranks are a random permutation
function random_rank_network(n; rng=Random.default_rng())
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

@testset "ERGMRank.jl" begin
    @testset "RankNetwork construction and validity" begin
        rnet = RankNetwork(4)
        @test is_valid_ranking(rnet)
        @test get_rank(rnet, 1, 1) == 0

        # Invalid: duplicate ranks within an ego
        bad = zeros(Int, 3, 3)
        bad[1, 2] = 1; bad[1, 3] = 1   # duplicate rank 1
        bad[2, 1] = 1; bad[2, 3] = 2
        bad[3, 1] = 1; bad[3, 2] = 2
        @test_throws ArgumentError RankNetwork(bad)

        # Non-square
        @test_throws ArgumentError RankNetwork(zeros(Int, 2, 3))

        # as_rank_network / rank_matrix round trip
        rt = r_test_network()
        @test as_rank_network(rank_matrix(rt)).ranks == rt.ranks
    end

    @testset "swap_ranks! preserves validity" begin
        rnet = r_test_network()
        swap_ranks!(rnet, 1, 2, 4)
        @test is_valid_ranking(rnet)
        @test get_rank(rnet, 1, 2) == 1
        @test get_rank(rnet, 1, 4) == 3
        @test_throws ArgumentError swap_ranks!(rnet, 1, 1, 2)

        # set_rank! can break validity, and is_valid_ranking detects it
        set_rank!(rnet, 1, 2, 3)
        @test !is_valid_ranking(rnet)
    end

    @testset "Golden master vs R ergm.rank 4.1.2" begin
        rnet = r_test_network()

        @test compute(RankDeference(), rnet) == 6.0
        @test compute(RankNonconformity(:all), rnet) == 10.0
        @test compute(RankNonconformity(:localAND), rnet) == 4.0
        @test compute(RankNodeICov([10.0, 20.0, 30.0, 40.0]), rnet) == -40.0

        # Inconsistency with itself is 0
        @test compute(RankInconsistency(rnet), rnet) == 0.0

        # Inconsistency vs the network with ego 1's ranking reversed
        m2 = rank_matrix(rnet)
        m2[1, 2] = 1; m2[1, 3] = 2; m2[1, 4] = 3
        @test compute(RankInconsistency(m2), rnet) == 6.0
    end

    @testset "RankEdgeCov generalizes RankNodeICov" begin
        rnet = r_test_network()
        x = [10.0, 20.0, 30.0, 40.0]
        cov = [x[j] for i in 1:4, j in 1:4]  # cov[i, j] = x[j]
        @test compute(RankEdgeCov(cov), rnet) ==
              compute(RankNodeICov(x), rnet)
    end

    @testset "Term names" begin
        @test name(RankDeference()) == "rank.deference"
        @test name(RankNonconformity()) == "rank.nonconformity"
        @test name(RankNonconformity(:localAND)) == "rank.nonconformity.localAND"
        @test name(RankNodeICov([1.0]; label="wealth")) == "rank.nodeicov.wealth"
        @test name(RankInconsistency(zeros(Int, 2, 2))) == "rank.inconsistency"
        @test_throws ArgumentError RankNonconformity(:bogus)
    end

    @testset "Simulation targets the model" begin
        rng = Random.Xoshiro(42)
        n = 6
        x = collect(1.0:n) ./ n
        start = random_rank_network(n; rng=rng)
        term = RankNodeICov(x)

        # Positive attractiveness coefficient must raise the statistic
        # relative to the uniform (θ = 0) model
        sims_pos = simulate_rank_ergm(start, [term], [1.0];
                                      n_sim=40, burnin=400, interval=20, rng=rng)
        sims_zero = simulate_rank_ergm(start, [term], [0.0];
                                       n_sim=40, burnin=400, interval=20, rng=rng)
        @test all(is_valid_ranking, sims_pos)
        @test all(is_valid_ranking, sims_zero)

        m_pos = mean(compute(term, s) for s in sims_pos)
        m_zero = mean(compute(term, s) for s in sims_zero)
        @test m_pos > m_zero

        # Uniform model: nodeicov statistic has mean 0 by symmetry
        @test abs(m_zero) < 8.0
    end

    @testset "Estimation recovers a known coefficient" begin
        rng = Random.Xoshiro(7)
        n = 8
        x = collect(1.0:n) .- (n + 1) / 2
        term = RankNodeICov(x)
        θ_true = 0.15

        start = random_rank_network(n; rng=rng)
        draws = simulate_rank_ergm(start, [term], [θ_true];
                                   n_sim=1, burnin=4000, interval=1, rng=rng)

        result = ergm_rank(draws[1], [term])
        @test result.converged
        @test isfinite(result.loglik)
        @test result.std_errors[1] > 0
        # Sign and rough magnitude
        @test result.coefficients[1] > 0
        @test result.coefficients[1] ≈ θ_true atol = 0.12
    end

    @testset "Estimator basics" begin
        rnet = r_test_network()
        result = ergm_rank(rnet, [RankDeference()])
        @test result isa RankERGMResult
        @test result.converged
        @test isfinite(result.loglik)
        # Pseudo-log-likelihood of the saturated-null fit is bounded above
        # by 0 (each conditional is a probability)
        @test result.loglik < 0

        @test fit_rank_ergm === ergm_rank

        # Invalid input rejected
        bad = RankNetwork(4)
        set_rank!(bad, 1, 2, bad.ranks[1, 3])  # duplicate
        @test_throws ArgumentError ergm_rank(bad, [RankDeference()])
    end
end
