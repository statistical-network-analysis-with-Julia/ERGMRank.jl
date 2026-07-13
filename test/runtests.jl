using ERGMRank
using ERGM
using Networks
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

    # ------------------------------------------------------------------
    # Allocation regression on the swap-MPLE derivative loop (review finding 15)
    #
    # `_rank_mple_fit` used to carry its own logistic loop with a per-comparison
    # `(pr*(1-pr)) .* (d * d')` inside it: a fresh p×p matrix on every one of the
    # n·(n−1)(n−2)/2 (ego, alter-pair) comparisons of every Newton evaluation.
    # The swap pseudo-likelihood IS a logistic likelihood with the response
    # identically true, so it now runs on the shared, workspace-backed
    # `ERGM.logistic_derivatives` — the same builder ERGMMulti and TERGM use.
    # ------------------------------------------------------------------
    @testset "Swap-MPLE derivative evaluations allocate O(p²), not O(rows · p²)" begin
        terms = AbstractERGMTerm[RankDeference(), RankNonconformity()]
        p = length(terms)

        function evaluation_allocs(n)
            rnet = random_rank_network(n; rng=Random.Xoshiro(12))
            D = ERGMRank._rank_design(terms, copy(rnet))
            d = ERGM.logistic_derivatives(D, trues(size(D, 1)))
            β = fill(-0.1, p)
            d(β)                    # warm up: @allocated on a first call
            return size(D, 1), @allocated d(β)   # would measure compilation
        end

        rows_small, a_small = evaluation_allocs(5)     # 30 comparisons
        rows_big, a_big = evaluation_allocs(17)        # 2040 comparisons
        @test rows_big > 50 * rows_small
        # 68x the comparisons, the same allocations.
        @test a_small <= 512
        @test a_big <= 512
        @test a_big <= a_small + 64

        # ...and the design really is what the fitter fits: the observed order is
        # always the "success", so the response is all-true and the swap
        # pseudo-log-likelihood is the logistic one on those rows.
        rnet = random_rank_network(6; rng=Random.Xoshiro(13))
        D = ERGMRank._rank_design(terms, copy(rnet))
        fit = ergm_rank(rnet, terms)
        @test ERGM.logistic_derivatives(D, trues(size(D, 1)))(fit.coefficients)[1] ≈
              fit.loglik atol = 1e-10
    end

    @testset "StatsAPI accessors" begin
        rnet = r_test_network()
        result = ergm_rank(rnet, [RankDeference(), RankNonconformity()])
        @test coef(result) == result.coefficients
        @test stderror(result) == result.std_errors
        @test vcov(result) == result.vcov
        @test size(vcov(result)) == (2, 2)
        @test all(stderror(result) .≈
                  sqrt.(abs.([vcov(result)[k, k] for k in 1:2])))
        @test loglikelihood(result) == result.loglik
        @test nobs(result) == 4 * 3 * 2 ÷ 2  # ego × unordered alter pairs
        @test dof(result) == 2
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
        @test fit_ergm_rank === ergm_rank

        # Invalid input rejected
        bad = RankNetwork(4)
        set_rank!(bad, 1, 2, bad.ranks[1, 3])  # duplicate
        @test_throws ArgumentError ergm_rank(bad, [RankDeference()])
    end

    @testset "show renders the shared coefficient table" begin
        rnet = r_test_network()
        result = fit_ergm_rank(rnet, [RankDeference(), RankNonconformity()])
        out = sprint(show, result)
        @test occursin("Rank-Order ERGM Results", out)
        @test occursin("Estimate", out)
        @test occursin("Pr(>|z|)", out)
        @test occursin("Signif. codes", out)
        @test occursin("rank.deference", out)
        @test occursin("rank.nonconformity", out)
        # The prose caveat is the twin of what the protocol reports
        @test occursin("anticonservative", out)
    end

    @testset "Result metadata protocol" begin
        rnet = r_test_network()
        result = fit_ergm_rank(rnet, [RankDeference(), RankNonconformity()])

        md = fit_metadata(result)
        @test md.estimand == :rank_ergm
        @test md.objective == :pseudolikelihood
        # Never exact: the swap comparisons overlap, so their product is not the
        # likelihood — and no consistency result is claimed for the estimator
        @test !md.is_exact
        @test md.se_method == :hessian
        @test md.missing_method == :none
        @test md.tie_method == :not_applicable

        # The withdrawn "consistent" claim must not reappear: the caveats say
        # the comparisons are not independent and the SEs are anticonservative
        @test any(occursin("not the likelihood", a) for a in md.approximations)
        @test any(occursin("no consistency result", a) for a in md.approximations)
        @test any(occursin("anticonservative", a) for a in md.approximations)

        # `show` and the protocol agree
        @test occursin("anticonservative", sprint(show, result))
    end

    @testset "gof extends the shared Networks.gof generic" begin
        # One generic across the ecosystem: the method is added to
        # Networks.gof, not a package-local function
        @test ERGMRank.gof === Networks.gof

        rnet = r_test_network()
        result = fit_ergm_rank(rnet, [RankDeference()])
        g = ERGMRank.gof(result; n_sim=10, burnin=50, interval=5,
                         rng=Random.Xoshiro(12))
        @test g isa Networks.GOFResult
        @test Networks.n_simulations(g) == 10
        @test length(g.statistics) == 1
        @test g.statistics[1].name == "model statistics"
        @test g.statistics[1].labels == ["rank.deference"]
        @test g.statistics[1].observed == [compute(RankDeference(), rnet)]
        @test all(p -> 0 < p <= 1, g.statistics[1].p_values)
        # Formatted display renders the shared GOF table
        out = sprint(show, g)
        @test occursin("Goodness-of-fit assessment: Rank-Order ERGM", out)
        @test occursin("MC p-value", out)
    end
    @testset "Robust standard errors: se=:bootstrap" begin
        # Issue #9 / ERGMRank#1: the swap pseudo-likelihood's comparisons overlap
        # (each ranking enters n − 2 of them), so the inverse-pseudo-Hessian SEs
        # are anticonservative — and there was no alternative. `se=:bootstrap` adds
        # a parametric bootstrap (AlterSwap-simulate at θ̂, refit the swap MPLE,
        # empirical covariance) on the ONE shared `Networks.bootstrap_cov` loop,
        # with the same API as `ERGM.mple`.
        rnet = random_rank_network(6; rng=MersenneTwister(1))
        terms = [RankNonconformity(), RankDeference()]

        hess = fit_ergm_rank(rnet, terms)
        boot = fit_ergm_rank(rnet, terms; se=:bootstrap, n_boot=60,
                             rng=MersenneTwister(5))

        # The bootstrap replaces the COVARIANCE, not the point estimate
        @test coef(boot) == coef(hess)
        @test loglikelihood(boot) == loglikelihood(hess)
        @test stderror(boot) != stderror(hess)
        @test vcov(boot) != vcov(hess)
        @test all(isfinite, stderror(boot))

        # Reproducible under a fixed rng
        boot2 = fit_ergm_rank(rnet, terms; se=:bootstrap, n_boot=60,
                              rng=MersenneTwister(5))
        @test stderror(boot2) == stderror(boot)
        @test vcov(boot2) == vcov(boot)
        @test stderror(fit_ergm_rank(rnet, terms; se=:bootstrap, n_boot=60,
                                     rng=MersenneTwister(6))) != stderror(boot)

        # The robust SEs EXCEED the Hessian ones — on both coefficients, and by a
        # wide margin (>2x here). That gap IS the anticonservatism that the
        # withdrawn "consistent" claim was papering over: the swap-Hessian SEs
        # treat n(n−1)(n−2)/2 overlapping comparisons as independent observations.
        @test all(stderror(boot) .> stderror(hess))

        # `se_method` reports what was ACTUALLY used, in both directions
        @test se_method(hess) === :hessian
        @test se_method(boot) === :bootstrap
        @test fit_metadata(hess).se_method === :hessian
        @test fit_metadata(boot).se_method === :bootstrap

        # ... and so does the printed output. No rank fit is exact, so the
        # pseudo-likelihood caveat on the POINT ESTIMATE stays in both; only the
        # anticonservatism claim — which is a claim about the inverse Hessian — is
        # dropped when a bootstrap was actually used.
        out_h = sprint(show, hess)
        out_b = sprint(show, boot)
        @test occursin("inverse pseudo-Hessian", out_h)
        @test occursin("anticonservative", out_h)
        @test occursin("parametric bootstrap", out_b)
        @test !occursin("anticonservative", out_b)
        @test occursin("pseudolikelihood", out_h) && occursin("pseudolikelihood", out_b)
        @test !is_exact(hess) && !is_exact(boot)

        # The approximations list agrees with the printed prose
        @test any(occursin("anticonservative", a) for a in approximations(hess))
        @test !any(occursin("anticonservative", a) for a in approximations(boot))
        @test any(occursin("parametric bootstrap", a) for a in approximations(boot))
        # The swap-pseudo-likelihood caveat on the point estimate survives both
        @test all(any(occursin("no consistency", a) for a in approximations(r))
                  for r in (hess, boot))

        # Unknown se symbols are rejected, not silently ignored
        @test_throws ArgumentError fit_ergm_rank(rnet, terms; se=:sandwich)
        @test_throws ArgumentError fit_ergm_rank(rnet, terms; se=:bootstrap,
                                                 n_boot=1)
    end

    # ------------------------------------------------------------------
    # Golden fixture: a REAL statnet `ergm.rank` fit on Newcomb's fraternity
    # ranks (issue #8). test/fixtures/r/newcomb_rank.R regenerates it (slow:
    # ~17 minutes — a rank-ERGM MCMLE takes ~3 min and the script does six).
    #
    # READ THIS BEFORE TRUSTING THE NUMBERS: THE TWO PACKAGES DO NOT FIT THE SAME
    # ESTIMATOR, AND THIS TESTSET EXISTS TO MEASURE THE GAP, NOT TO HIDE IT.
    #
    #   ergm.rank fits the MCMC MLE.
    #   ERGMRank.jl fits a SWAP PSEUDO-LIKELIHOOD — the product of the logistic
    #   probabilities of each (ego, alter-pair) relative order, multiplied as if
    #   independent, which they are emphatically not.
    #
    # A pseudo-likelihood is not an approximation to the MLE with a small error;
    # it is a different estimator, and this package's own docs decline to claim
    # consistency for it. So the coefficient comparison is @test_broken and the
    # gap is CHARACTERIZED below rather than tolerated. Widening a tolerance until
    # this goes green would be a tolerance chosen to conceal the finding.
    # ------------------------------------------------------------------
    @testset "Golden fixture: statnet ergm.rank on newcomb (swap-MPLE vs MLE)" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "newcomb_rank.toml"))
        @test g.provenance["ergm_rank_version"] == "4.1.2"

        n = Int(g.values["n_actors"])
        M = zeros(Int, n, n)
        for i in 1:n, j in 1:n
            M[i, j] = Int(g.values["ranks"][i][j])
        end
        rnet = RankNetwork(M)
        @test is_valid_ranking(rnet)

        terms = [RankDeference(), RankNonconformity(:all)]

        # --- ASSERTED: the observed sufficient statistics ---------------------
        # A deterministic function of the ranking — no estimator, no Monte Carlo.
        # Machine precision, and a disagreement is a bug in a term formula. (The
        # term values were already golden-tested against ergm.rank 4.1.2 from
        # literals in a comment; this re-pins them from a provenanced file.)
        @test g.values["summary_statistic_names"] == ["deference", "nonconformity"]
        stats = [compute(t, rnet) for t in terms]
        @test check_golden(g, "summary_statistics", stats) ||
              error(golden_report(g, "summary_statistics", stats))

        fit = fit_ergm_rank(rnet, terms)
        @test fit.converged

        r_coef = Float64.(g.values["mle_coefficients"])
        r_se = Float64.(g.values["mle_std_errors"])
        r_sd = Float64.(g.values["mle_seed_sd"])
        gap = abs.(fit.coefficients .- r_coef)

        # --- BROKEN: the swap-MPLE is not the MLE ----------------------------
        # Marked broken, not tolerated. Observed as of ergm.rank 4.1.2:
        #   swap-MPLE : [-0.14091, -0.0058538]
        #   ergm.rank : [-0.15310, -0.0065927]
        @test_broken isapprox(fit.coefficients, r_coef; atol=1e-3)

        # --- ASSERTED: the CHARACTER of the gap -------------------------------
        # This is what the fixture is really for. Two things are true at once and
        # both matter:
        #
        # (1) The gap is SYSTEMATIC, not Monte-Carlo. ergm.rank's own seed-to-seed
        #     sd on these coefficients is 7.4e-4 and 5.6e-5; the gap is 1.2e-2 and
        #     7.4e-4 — 16x and 13x that width. No amount of MCMC would close it.
        #     The two estimators genuinely disagree.
        @test all(gap .> 5 .* r_sd)
        #
        # (2) But the disagreement is SMALL on the scale that matters: 0.30 and
        #     0.43 of an ergm.rank standard error. Same sign, same order, same
        #     substantive story. The swap-MPLE is a different estimator, not a
        #     broken one — and that is exactly the claim the package makes.
        @test all(gap .< 0.6 .* r_se)

        # --- ASSERTED: the pseudo-Hessian SEs are anticonservative, measurably --
        # The package documents this; here is the number. Against ergm.rank's MLE
        # standard errors, the inverse pseudo-Hessian understates by ~3.9x
        # (deference) and ~2.0x (nonconformity) — the overlapping swap comparisons
        # are being multiplied as if independent.
        @test all(fit.std_errors .< r_se)
        @test r_se[1] / fit.std_errors[1] > 2.0

        # ...and `se=:bootstrap` — the honest option — moves most of the way back.
        # Bootstrap: [0.0280, 0.00192] against R's [0.0404, 0.00174]. Still not
        # the MLE's, because it is bootstrapping a different estimator, but no
        # longer understating by a factor of four.
        boot = fit_ergm_rank(rnet, terms; se=:bootstrap, n_boot=60,
                             rng=Random.Xoshiro(7))
        @test boot.coefficients ≈ fit.coefficients atol = 1e-12   # point est unchanged
        @test all(boot.std_errors .> fit.std_errors)
        @test abs(boot.std_errors[1] - r_se[1]) < abs(fit.std_errors[1] - r_se[1])
    end
end
