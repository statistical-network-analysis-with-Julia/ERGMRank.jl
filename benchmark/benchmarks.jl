#!/usr/bin/env julia
# benchmark/benchmarks.jl — BenchmarkTools suite for ERGMRank.jl's hot loops.
#
# Locks in the per-swap change statistics (`swap_change`, O(n) per term and
# O(n²) for the nonconformity variants — never a recomputation of the full
# statistic), the allocation-free AlterSwap Metropolis step on the shared
# `ERGM.mh_toggle!` kernel, the swap-MPLE design builder and the whole
# swap-MPLE fit on a 17-actor ranking (the size of the Newcomb fixture).
#
# Defines the standard `SUITE::BenchmarkGroup`. Run standalone with
#     julia --project=benchmark benchmark/benchmarks.jl
# which tunes + runs the suite, prints one tab-separated `BENCHJL` line per
# benchmark (consumed by the site repo's tools/run_benchmarks.jl), and exits
# non-zero if the scaling assertion fails: `swap_change` is measured on
# rankings of 25 and 50 actors, and the per-term cost must grow no faster
# than the documented order (recomputing a statistic instead — O(n³) or
# O(n⁴) — would show up as an ≥ 8× ratio; O(n) stays ≈ 2×, O(n²) ≈ 4×).

using BenchmarkTools
using ERGMRank
using ERGM
using Networks
using Random

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

const N_SMALL = 25
const N_LARGE = 50
const N_FIT = 17                # the Newcomb fixture's size
const N_SWAPS = 200             # swaps evaluated per benchmark evaluation

"Random complete ranking: each ego's ranks are a random permutation."
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

"Fixed sample of `N_SWAPS` random swaps (ego, j, k) to sweep per evaluation."
function sample_swaps(rng::AbstractRNG, n::Int)
    swaps = Tuple{Int, Int, Int}[]
    while length(swaps) < N_SWAPS
        ego, j, k = rand(rng, 1:n), rand(rng, 1:n), rand(rng, 1:n)
        (ego == j || ego == k || j == k) && continue
        push!(swaps, (ego, j, k))
    end
    return swaps
end

"Sum of swap change statistics over a fixed swap sample."
function sweep_swap_change(term, rnet, swaps)
    s = 0.0
    for (ego, j, k) in swaps
        s += swap_change(term, rnet, ego, j, k)
    end
    return s
end

const NETS = Dict(n => random_rank_network(Random.Xoshiro(n), n) for n in (N_SMALL, N_LARGE))
const SWAPS = Dict(n => sample_swaps(Random.Xoshiro(n + 1), n) for n in (N_SMALL, N_LARGE))
const NET_FIT = random_rank_network(Random.Xoshiro(17), N_FIT)

# (label, term for each size, documented order in n, tolerated ratio at 2n)
terms_for(n) = [
    ("deference", RankDeference(), 1),
    ("nonconformity", RankNonconformity(:all), 2),
    ("nonconformity_localAND", RankNonconformity(:localAND), 2),
    ("nodeicov", RankNodeICov(collect(1.0:n)), 1),
    ("inconsistency", RankInconsistency(random_rank_network(Random.Xoshiro(99), n)), 1),
    ("edgecov", RankEdgeCov(rand(Random.Xoshiro(98), n, n)), 1),
]
# O(n) at 2n is 2×, O(n²) is 4×, and the regression this guards against —
# recomputing a full statistic, O(n³) for deference/nodeicov/inconsistency/
# edgecov and O(n⁴) for nonconformity — is ≥ 8×. The limit sits between the
# documented order and that cliff, with room for cache effects (a 50×50
# rank matrix no longer fits where a 25×25 one did): only a change of
# order trips it.
scaling_limit(order::Int) = order == 1 ? 5.0 : 6.5

const FIT_TERMS = [RankDeference(), RankNonconformity(:all)]

# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

const SUITE = BenchmarkGroup()

let g = addgroup!(SUITE, "swap_change")
    for n in (N_SMALL, N_LARGE)
        for (label, term, _) in terms_for(n)
            g["$(label)_n$(n)"] =
                @benchmarkable sweep_swap_change($term, $(NETS[n]), $(SWAPS[n]))
        end
    end
end

const SAMPLER_TERMS = AbstractERGMTerm[RankDeference(), RankNonconformity(:all)]
const SAMPLER_θ = [-0.1, -0.005]
const STEPS = 1_000

let g = addgroup!(SUITE, "sampler")
    # The Metropolis step on the shared kernel at n = 50 with two terms. A
    # `simulate_rank_ergm` call carries a fixed overhead (validating the
    # start, copying it, the one draw), so the step is measured as the
    # DIFFERENCE between a run of STEPS + 1 steps and a run of 1 step:
    # `main` prints the derived per-step line (time, allocs and bytes are
    # all differences; the step itself allocates nothing).
    g["run_1_step_n$(N_LARGE)"] =
        @benchmarkable simulate_rank_ergm($(NETS[N_LARGE]), $SAMPLER_TERMS, $SAMPLER_θ;
                                          n_sim=1, burnin=1, interval=1,
                                          rng=Random.Xoshiro(1))
    g["run_$(STEPS + 1)_steps_n$(N_LARGE)"] =
        @benchmarkable simulate_rank_ergm($(NETS[N_LARGE]), $SAMPLER_TERMS, $SAMPLER_θ;
                                          n_sim=1, burnin=STEPS + 1, interval=1,
                                          rng=Random.Xoshiro(1))
end

let g = addgroup!(SUITE, "mple")
    terms = collect(AbstractERGMTerm, FIT_TERMS)
    g["rank_design_n$(N_FIT)"] =
        @benchmarkable ERGMRank._rank_design($terms, work) setup=(work = copy(NET_FIT))
    g["fit_ergm_rank_n$(N_FIT)"] =
        @benchmarkable fit_ergm_rank($NET_FIT, $FIT_TERMS)
end

# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

function print_benchjl(results::BenchmarkGroup)
    for (path, trial) in BenchmarkTools.leaves(results)
        est = median(trial)
        println("BENCHJL\t", join(path, "/"), "\t",
                BenchmarkTools.time(est), "\t",
                BenchmarkTools.allocs(est), "\t",
                BenchmarkTools.memory(est))
    end
    # The Metropolis step, derived: (run of STEPS + 1 steps − run of 1 step) / STEPS
    long = median(results["sampler"]["run_$(STEPS + 1)_steps_n$(N_LARGE)"])
    short = median(results["sampler"]["run_1_step_n$(N_LARGE)"])
    per_step(f) = max(0.0, (f(long) - f(short)) / STEPS)
    println("BENCHJL\tsampler/mh_step_n$(N_LARGE)\t",
            per_step(BenchmarkTools.time), "\t",
            round(Int, per_step(BenchmarkTools.allocs)), "\t",
            round(Int, per_step(BenchmarkTools.memory)))
end

"Assert that the per-swap cost of each term grew no faster than its order."
function assert_scaling(results::BenchmarkGroup)
    ok = true
    for (label, _, order) in terms_for(N_SMALL)
        t_small = BenchmarkTools.time(median(results["swap_change"]["$(label)_n$(N_SMALL)"]))
        t_large = BenchmarkTools.time(median(results["swap_change"]["$(label)_n$(N_LARGE)"]))
        ratio = t_large / t_small
        println("SCALING\t", label, "\tn", N_LARGE, "/n", N_SMALL, "\t",
                round(ratio, digits=2))
        if ratio > scaling_limit(order)
            println(stderr, "SCALING FAILURE: $label swap_change is ",
                    round(ratio, digits=2), "x slower at n=$(N_LARGE) than at ",
                    "n=$(N_SMALL) (limit $(scaling_limit(order))x for an O(n^$order) ",
                    "change statistic). The per-swap cost is no longer O(n^$order).")
            ok = false
        end
    end
    return ok
end

function main()
    tune!(SUITE)
    results = run(SUITE; verbose=false, seconds=1)
    print_benchjl(results)
    assert_scaling(results) || exit(1)
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
