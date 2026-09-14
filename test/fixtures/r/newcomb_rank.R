# Golden fixture: statnet `ergm.rank` FITTED OUTPUT on Newcomb's fraternity ranks.
#
# Regenerate from the package root. THIS IS SLOW -- ~17 minutes: each MCMLE fit of
# a rank ERGM takes ~3 minutes, and the script does six of them (one frozen fit
# plus five replication seeds):
#
#   Rscript test/fixtures/r/newcomb_rank.R > test/fixtures/newcomb_rank.toml
#
# WHAT THIS FIXTURE ASSERTS, AND AGAINST WHICH ESTIMATOR.
#
#   ergm.rank fits the MCMC MLE. It samples complete orderings under the
#   CompleteOrder reference with the AlterSwap proposal and solves
#   E_theta[g] = g_obs by MCMC-MLE.
#
#   ERGMRank.jl has two estimators. `fit_ergm_rank(...; method=:mcmle)` is the
#   same MCMC MLE (Hummel-stepped MCMLE on an AlterSwap chain, Fisher-plus-
#   Monte-Carlo standard errors, bridge log-likelihood), and its coefficients
#   and standard errors are ASSERTED against this fixture. The default
#   `method=:mple` is a SWAP PSEUDO-LIKELIHOOD -- for each ego and each
#   unordered pair of alters the logistic probability of the observed relative
#   order given everything else, multiplied across all (ego, pair) comparisons
#   as if they were independent, which they are not -- a DIFFERENT estimator
#   whose gap to the MLE the Julia testset CHARACTERISES (systematic: 16x and
#   13x ergm.rank's own seed-to-seed sd; small: 0.30 and 0.43 of an MLE
#   standard error) but never asserts equality for.
#
# Both MCMLEs are Monte-Carlo estimates of the same MLE, so the honest
# tolerance is a multiple of the two implementations' seed-to-seed spread:
# ergm.rank's (`mle_seed_sd`, five further seeds below) plus ERGMRank.jl's
# (`julia_seed_sd`, measured over five Julia seeds at the same MCMC budget and
# frozen in [tolerance] with its provenance). Widening either beyond what the
# spread justifies would be a tolerance chosen to hide a result.
#
# WHAT IS ASSERTED AT MACHINE PRECISION is the deterministic half: the observed
# sufficient statistics. Those are a function of the ranking alone -- no
# estimator, no Monte Carlo -- and a disagreement there would be a bug in a
# term formula. (ERGMRank.jl already golden-tests its term VALUES against
# ergm.rank 4.1.2; this re-pins them from a provenanced file rather than from
# literals in a comment.)

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(ergm.rank)
})

seed <- 20260713
data(newcomb)

# Newcomb (1961) fraternity: 17 men, 15 weeks, each man ranks the other 16.
# Wave 1. GREATER rank value = HIGHER standing, which is ergm.rank's convention
# and ERGMRank.jl's.
nw <- newcomb[[1]]
n <- network.size(nw)
R <- as.matrix(nw, attrname = "rank")

f <- nw ~ rank.deference + rank.nonconformity("all")
obs <- summary(f, response = "rank")

# ergm() prints MCMLE iteration chatter to stdout, and this script's stdout IS
# the TOML fixture, so unsuppressed it would emit an unparseable file.
fit_once <- function(s) {
  set.seed(s)
  out <- NULL
  invisible(capture.output(
    out <- ergm(f, response = "rank", reference = ~CompleteOrder,
                control = control.ergm(seed = s, MCMC.samplesize = 2048,
                                       MCMC.burnin = 8192,
                                       MCMC.interval = 512)),
    type = "output"))
  out
}

fit <- fit_once(seed)
mle_coef <- coef(fit)
mle_se <- sqrt(diag(vcov(fit)))

# The log-likelihood ergm.rank reports, computed by ergm() at fit time by
# bridge sampling (16 bridges). NOTE THE CONVENTION: for a valued ERGM, ergm
# defines the null (theta = 0) model's likelihood as 0 -- it prints "Null
# model likelihood calculation is not implemented for valued ERGMs at this
# time" -- so logLik(fit) is the bridge-sampled RATIO log L(theta_hat) -
# log L(0), not the absolute log-likelihood. ERGMRank.jl reports the ABSOLUTE
# log-likelihood, subtracting the exact normalizer of the uniform ordering
# model, log Z(0) = n * log((n-1)!). The Julia testset therefore asserts
#   loglikelihood(fit) + n*log((n-1)!)  ~  mle_loglik
# within both bridges' Monte-Carlo error: `mle_loglik_mc_se` is ergm's own
# standard error of its bridge estimate (sqrt of attr(logLik(fit), "vcov"),
# the "MC Std. Err." it prints), and `julia_bridge_sd` in [tolerance] is
# ERGMRank.jl's, measured over seeds. AIC/BIC follow the same convention:
# R's are ERGMRank's shifted by -2 n log((n-1)!), and R's `nobs` for the
# likelihood is the dyad count n(n-1) (`mle_nobs`), which ERGMRank.jl's
# `nobs(fit)` returns for a method=:mcmle fit.
ll <- logLik(fit)
mle_loglik <- as.numeric(ll)
mle_loglik_mc_se <- sqrt(as.numeric(attr(ll, "vcov")))
mle_df <- as.integer(attr(ll, "df"))
mle_nobs <- as.integer(nobs(fit))
mle_aic <- AIC(fit)
mle_bic <- BIC(fit)

# How much does ergm.rank disagree with ITSELF? Five further seeds, same data,
# same model, same MCMC budget. This is the Monte-Carlo width of the MLE, and it
# is the yardstick the Julia-vs-R gap must be reported in: a discrepancy of one
# seed-sd means nothing, a discrepancy of ten means the estimators differ.
rep_seeds <- c(101, 202, 303, 404, 505)
reps <- t(sapply(rep_seeds, function(s) coef(fit_once(s))))
seed_sd <- apply(reps, 2, sd)

num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")
strs <- function(x) paste(sprintf('"%s"', x), collapse = ", ")
rows <- paste(apply(R, 1, function(r) paste0("[", paste(r, collapse = ", "), "]")),
              collapse = ", ")

cat('name = "newcomb_rank"\n\n')

cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('ergm_rank_version = "%s"\n', as.character(packageVersion("ergm.rank"))))
cat(sprintf('ergm_version = "%s"\n', as.character(packageVersion("ergm"))))
cat(sprintf('network_version = "%s"\n', as.character(packageVersion("network"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/newcomb_rank.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "ergm.rank::newcomb[[1]] (Newcomb 1961): 17 fraternity men, week 1, each ranking the other 16; GREATER value = HIGHER standing"\n')
cat('model = "newcomb[[1]] ~ rank.deference + rank.nonconformity(\\"all\\"), response=\\"rank\\", reference=~CompleteOrder"\n')
cat('r_estimator = "MCMC-MLE (AlterSwap proposal)"\n')
cat('julia_estimator = "MCMC-MLE (fit_ergm_rank(...; method=:mcmle), AlterSwap proposal) -- ASSERTED; the default method=:mple swap pseudo-likelihood is a DIFFERENT estimator, characterised but not asserted; see [tolerance]"\n')
cat('mcmc_control = "control.ergm(MCMC.samplesize=2048, MCMC.burnin=8192, MCMC.interval=512)"\n')
cat(sprintf('replication_seeds = "%s"\n', paste(rep_seeds, collapse = ",")))
cat("\n")

cat("[tolerance]\n")
cat("# Observed sufficient statistics: a deterministic function of the ranking.\n")
cat("# No estimator, no Monte Carlo. Machine precision, and a disagreement is a\n")
cat("# bug in a term formula, full stop.\n")
cat("summary_statistics = 1e-9\n")
cat("#\n")
cat("# COEFFICIENTS: ASSERTED against ergm.rank's MCMC MLE, for\n")
cat("# `fit_ergm_rank(...; method=:mcmle)` run at ergm.rank's own MCMC budget\n")
cat("# (`mcmc_control` in [provenance]: n_samples=2048, burnin=8192, interval=512).\n")
cat("#\n")
cat("# Both are Monte-Carlo estimates of the same MLE, so each has a seed-to-seed\n")
cat("# spread and the tolerance is built from both: `mle_seed_sd` ([values]) is\n")
cat("# ergm.rank disagreeing with itself over five further seeds; `julia_seed_sd`\n")
cat("# below is ERGMRank.jl disagreeing with itself over five Julia seeds\n")
cat("# (Xoshiro(1), ..., Xoshiro(5)) at the SAME budget, one chain, the bridge\n")
cat("# skipped (it runs after the final sample and does not touch the\n")
cat("# coefficients), measured 2026-09-12 on ERGMRank.jl 0.2.0-dev:\n")
cat("#   deference     -0.15184 -0.15297 -0.15217 -0.15302 -0.15250  (sd 5.1e-4)\n")
cat("#   nonconformity -0.0067429 -0.0065566 -0.0065854 -0.0066495 -0.0066122  (sd 7.2e-5)\n")
cat("# The Julia mean sits 0.81 and 0.65 ergm.rank seed-sd from `mle_coefficients`;\n")
cat("# the swap-MPLE sat 16x and 13x away. Same estimator, at last.\n")
cat("#\n")
cat("# The testset asserts, on two Julia seeds,\n")
cat("#   |coef - mle_coefficients| <= coefficient_sd_multiple * (mle_seed_sd + julia_seed_sd).\n")
cat("# Justification of the multiple: the difference of two independent Monte-Carlo\n")
cat("# estimates has sd sqrt(sd_R^2 + sd_J^2) <= sd_R + sd_J, so 4 of the sum is\n")
cat("# beyond a 4-sigma band of the difference; `mle_coefficients` is itself one\n")
cat("# draw (1e-5 and 2e-5 from `mle_seed_mean`, absorbed by the same band). The\n")
cat("# largest deviation observed over the five Julia seeds was 1.26e-3 and 1.50e-4,\n")
cat("# a quarter of the band (5.0e-3 and 5.1e-4). Do not raise the multiple: a\n")
cat("# failure at 4 means the estimators disagree, not that the seed was unlucky.\n")
cat("julia_seed_sd = [0.0005105004020318198, 7.216447602442238e-5]\n")
cat("coefficient_sd_multiple = 4\n")
cat("#\n")
cat("# STANDARD ERRORS: asserted within `std_errors_rtol` relative of\n")
cat("# `mle_std_errors`. Both are the inverse Fisher information of the final\n")
cat("# MCMC sample plus the Monte-Carlo-error component (Hunter & Handcock 2006\n")
cat("# section 3.3; ergm's `MCMC %`), so they differ only by the sampling noise of\n")
cat("# a 2048-draw covariance: observed over the five Julia seeds 0.06%-3.0%\n")
cat("# (deference) and 0.1%-3.7% (nonconformity). 15% is five times the largest\n")
cat("# of those and still a quarter of the anticonservatism the swap-MPLE's\n")
cat("# pseudo-Hessian shows (3.9x / 2.0x too narrow), which it must keep failing.\n")
cat("std_errors_rtol = 0.15\n")
cat("#\n")
cat("# LOG-LIKELIHOOD: ergm.rank's logLik is RELATIVE to the uniform-ordering model\n")
cat("# (theta = 0), whose likelihood ergm defines as 0 for a valued ERGM; ERGMRank.jl\n")
cat("# reports the absolute log-likelihood, so the testset asserts\n")
cat("#   |loglikelihood(fit) + n*log((n-1)!) - mle_loglik|\n")
cat("#     <= loglik_sd_multiple * (mle_loglik_mc_se + julia_bridge_sd).\n")
cat("# Both sides are path-sampling (bridge) estimates with Monte-Carlo error:\n")
cat("# `mle_loglik_mc_se` ([values]) is ergm's own standard error of its bridge\n")
cat("# (16 bridges at the fit's MCMC budget); `julia_bridge_sd` is ERGMRank.jl's\n")
cat("# bridge (bridge_rungs=16, bridge_samples=512, the fit's burn-in and interval)\n")
cat("# disagreeing with itself over the five Julia seeds Xoshiro(1..5), measured\n")
cat("# 2026-09-12 on ERGMRank.jl 0.2.0-dev as loglikelihood + 17*log(16!):\n")
cat("#   18.87717 18.90319 18.85990 18.88535 18.85694  (sd 1.9e-2; at\n")
cat("#   bridge_samples=2048: 18.90821 18.90680 18.87066 18.82997 18.87499, sd 3.2e-2 --\n")
cat("#   the error is dominated by the rung chains' mixing, not the draw count)\n")
cat("# The difference of two independent estimates has sd <= the sum of the two,\n")
cat("# so 4 of the sum is beyond a 4-sigma band; the theta-hat mismatch between the\n")
cat("# two fits shifts the log-likelihood by ~1/2 dtheta' I dtheta ~ 2e-3, inside it.\n")
cat("# Do not raise the multiple: a failure at 4 means the bridges disagree, not\n")
cat("# that a seed was unlucky.\n")
cat("julia_bridge_sd = 0.01903536725140555\n")
cat("loglik_sd_multiple = 4\n")
cat("#\n")
cat("# THE SWAP-MPLE (method=:mple, the default) HAS NO TOLERANCE, BY DESIGN. It is\n")
cat("# a different estimator, not a noisy version of the MLE, and ERGMRank.jl's docs\n")
cat("# decline to claim consistency for it. The Julia testset asserts the CHARACTER\n")
cat("# of its gap instead (larger than 5 `mle_seed_sd`, smaller than 0.6 of an MLE\n")
cat("# standard error; pseudo-Hessian SEs narrower than the MLE's), so that a\n")
cat("# change in either direction is noticed.\n")
cat("\n")

cat("[values]\n")
cat("# --- the ranking, frozen. Julia rebuilds it exactly. --------------------\n")
cat(sprintf("n_actors = %d\n", n))
cat(sprintf("ranks = [%s]\n", rows))
cat("\n# --- observed sufficient statistics: deterministic, ASSERTED at 1e-9 ----\n")
cat(sprintf("summary_statistic_names = [%s]\n", strs(names(obs))))
cat(sprintf("summary_statistics = [%s]\n", num(as.numeric(obs))))
cat("\n# --- ergm.rank's MCMC MLE: ASSERTED for method=:mcmle (see [tolerance]) ---\n")
cat("# `fit_ergm_rank(...; method=:mcmle)` at the same MCMC budget must reproduce\n")
cat("# these within the two implementations' seed-to-seed spread; the default\n")
cat("# swap pseudo-likelihood (method=:mple) is measured AGAINST them, not held\n")
cat("# TO them.\n")
cat(sprintf("term_names = [%s]\n", strs(names(mle_coef))))
cat(sprintf("mle_coefficients = [%s]\n", num(as.numeric(mle_coef))))
cat(sprintf("mle_std_errors = [%s]\n", num(as.numeric(mle_se))))
cat("\n# ergm.rank's logLik (RELATIVE to theta = 0; see [tolerance]), its own MC\n")
cat("# standard error, df, nobs (= the n(n-1) ordered dyads, R's convention for\n")
cat("# the likelihood) and the AIC/BIC ergm derives from them.\n")
cat(sprintf("mle_loglik = %s\n", num(mle_loglik)))
cat(sprintf("mle_loglik_mc_se = %s\n", num(mle_loglik_mc_se)))
cat(sprintf("mle_df = %d\n", mle_df))
cat(sprintf("mle_nobs = %d\n", mle_nobs))
cat(sprintf("mle_aic = %s\n", num(mle_aic)))
cat(sprintf("mle_bic = %s\n", num(mle_bic)))
cat("\n# ergm.rank disagreeing with ITSELF over five further seeds. This is the\n")
cat("# Monte-Carlo width of the MLE, half of the MCMLE tolerance above, and the\n")
cat("# unit the swap-MPLE gap is reported in: a gap of one of these means\n")
cat("# nothing; a gap of sixteen means a different estimator.\n")
cat(sprintf("mle_seed_sd = [%s]\n", num(seed_sd)))
cat(sprintf("mle_seed_mean = [%s]\n", num(colMeans(reps))))
