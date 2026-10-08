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
#   ERGMRank.jl has two estimators. `fit_ergm_rank` (the default,
#   `method=:mcmle`) is the same MCMC MLE (Hummel-stepped MCMLE on an AlterSwap
#   chain, Fisher-plus-Monte-Carlo standard errors, bridge log-likelihood),
#   and its coefficients and standard errors are ASSERTED against this fixture.
#   `method=:mple` is a SWAP PSEUDO-LIKELIHOOD -- for each ego and each
#   unordered pair of alters the logistic probability of the observed relative
#   order given everything else, multiplied across all (ego, pair) comparisons
#   as if they were independent, which they are not -- a DIFFERENT estimator
#   whose gap to the MLE the Julia testset CHARACTERISES (systematic: 16x and
#   13x ergm.rank's own seed-to-seed sd; small: 0.30 and 0.43 of an MLE
#   standard error) but never asserts equality for.
#
# Both MCMLEs are Monte-Carlo estimates of the same MLE, so the honest
# tolerance is a multiple of ergm.rank's own seed-to-seed spread
# (`mle_seed_sd`, five further seeds below). If ERGMRank.jl's MCMLE is the
# same estimator, a Julia estimate is one more draw from that spread; the
# tolerance uses no number measured on ERGMRank.jl. Widening it beyond what
# the spread justifies would be a tolerance chosen to hide a result.
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
# the "MC Std. Err." it prints); the tolerance is a multiple of it, as for the
# coefficients. AIC/BIC follow the same convention:
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
rep_fits <- lapply(rep_seeds, fit_once)
reps <- t(sapply(rep_fits, coef))
rep_ses <- t(sapply(rep_fits, function(x) sqrt(diag(vcov(x)))))
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
cat('julia_estimator = "MCMC-MLE (fit_ergm_rank, the default method=:mcmle, AlterSwap proposal) -- ASSERTED; the method=:mple swap pseudo-likelihood is a DIFFERENT estimator, characterised but not asserted; see [tolerance]"\n')
cat('mcmc_control = "control.ergm(MCMC.samplesize=2048, MCMC.burnin=8192, MCMC.interval=512)"\n')
cat(sprintf('replication_seeds = "%s"\n', paste(rep_seeds, collapse = ",")))
cat("\n")

# ---------------------------------------------------------------------------
# Why the equal-precision premise below is stated, not measured here. The
# bands are built from ergm.rank's spread alone: sqrt(2) * sd_R for the
# difference of a Julia estimate and R's frozen one assumes ERGMRank.jl's
# seed-to-seed sd is no larger than R's. The premise can only fail in the
# safe direction: were ERGMRank.jl's sd c * sd_R with c > 1, the band
# 4 * sqrt(2) * sd_R would be 4 * sqrt(2) / sqrt(1 + c^2) of the difference's
# true sd (3.6 at c = 1.2, 3.1 at c = 1.5), so the testset would fail MORE
# often; it can never let a disagreement pass. A noisier Julia estimator
# therefore shows up as a failure, which is what the band is for.
#
# The log-likelihood band is 4 * sqrt(2) * mle_loglik_mc_se = 1.126. An
# earlier form, 4 * (sd_R + sd_J) = 1.039, added ERGMRank.jl's measured
# bridge sd (sd_J = 0.061); no number measured on the package under test may
# enter a tolerance, so it was replaced by the premise sd_J <= sd_R, at the
# cost of an 8 % wider band. With the measured sd_J the band is
# 1.126 / sqrt(0.199^2 + 0.061^2) = 5.4 sd of the difference; under the
# premise alone it is at least 4.
# ---------------------------------------------------------------------------
cat("[tolerance]\n")
cat("# Observed sufficient statistics: a deterministic function of the ranking.\n")
cat("# No estimator, no Monte Carlo. Machine precision, and a disagreement is a\n")
cat("# bug in a term formula, full stop.\n")
cat("summary_statistics = 1e-9\n")
cat("#\n")
cat("# COEFFICIENTS: ASSERTED against ergm.rank's MCMC MLE, for\n")
cat("# `fit_ergm_rank` (method=:mcmle, the default) run at ergm.rank's own MCMC budget\n")
cat("# (`mcmc_control` in [provenance]: n_samples=2048, burnin=8192, interval=512).\n")
cat("#\n")
cat("# Both are Monte-Carlo estimates of the same MLE at the same budget, so the\n")
cat("# yardstick is ergm.rank's own seed-to-seed spread: `mle_seed_sd` ([values]),\n")
cat("# ergm.rank disagreeing with itself over five further seeds. No number in\n")
cat("# this block is measured on ERGMRank.jl. The testset asserts, on two Julia\n")
cat("# seeds,\n")
cat("#   |coef - mle_coefficients| <= coefficient_sd_multiple * sqrt(2) * mle_seed_sd.\n")
cat("# Justification of the rule: if ERGMRank.jl's MCMLE is the same estimator as\n")
cat("# ergm.rank's, a Julia estimate and the frozen R estimate are two independent\n")
cat("# draws with the same Monte-Carlo sd, so their difference has sd\n")
cat("# sqrt(2) * sd_R, and 4 of that is a 4-sigma band. (The testset also asserts\n")
cat("# the tighter |coef - mle_coefficients| < 5 * mle_seed_sd.) Do not raise the\n")
cat("# multiple: a failure at 4 means the estimators disagree, not that the seed\n")
cat("# was unlucky.\n")
cat("coefficient_sd_multiple = 4\n")
cat("#\n")
cat("# STANDARD ERRORS: asserted within `std_errors_rtol` relative of\n")
cat("# `mle_std_errors`. Both are the inverse Fisher information of the final\n")
cat("# MCMC sample plus the Monte-Carlo-error component (Hunter & Handcock 2006\n")
cat("# section 3.3; ergm's `MCMC %`), so they differ only by the sampling noise of\n")
cat("# a 2048-draw covariance. ergm.rank's own relative spread of its standard\n")
cat("# errors over the five replication seeds is `mle_std_errors_rel_sd`\n")
cat("# ([values]); the difference of two such estimates has relative sd about\n")
cat("# sqrt(2) times that, and the testset asserts that 0.15 is at least 4 of\n")
cat("# it. 0.15 is still a quarter of the anticonservatism the swap-MPLE's\n")
cat("# pseudo-Hessian shows (3.9x / 2.0x too narrow), which it must keep failing.\n")
cat("std_errors_rtol = 0.15\n")
cat("#\n")
cat("# LOG-LIKELIHOOD: ergm.rank's logLik is RELATIVE to the uniform-ordering model\n")
cat("# (theta = 0), whose likelihood ergm defines as 0 for a valued ERGM; ERGMRank.jl\n")
cat("# reports the absolute log-likelihood, so the testset asserts\n")
cat("#   |loglikelihood(fit) + n*log((n-1)!) - mle_loglik|\n")
cat("#     <= loglik_sd_multiple * sqrt(2) * mle_loglik_mc_se.\n")
cat("# Both sides are path-sampling (bridge) estimates of the same quantity with\n")
cat("# Monte-Carlo error; `mle_loglik_mc_se` ([values]) is ergm's own standard\n")
cat("# error of its bridge (16 bridges at the fit's MCMC budget). If the Julia\n")
cat("# bridge is no noisier than ergm's, the difference of the two has sd at most\n")
cat("# sqrt(2) * mle_loglik_mc_se, and 4 of that is a 4-sigma band; the theta-hat\n")
cat("# mismatch between the two fits shifts the log-likelihood by about\n")
cat("# 1/2 dtheta' I dtheta ~ 2e-3, inside it. No number in this block is\n")
cat("# measured on ERGMRank.jl. Do not raise the multiple: a failure at 4 means\n")
cat("# the bridges disagree, not that a seed was unlucky.\n")
cat("loglik_sd_multiple = 4\n")
cat("#\n")
cat("# THE SWAP-MPLE (method=:mple) HAS NO TOLERANCE, BY DESIGN. It is\n")
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
cat("# `fit_ergm_rank` (method=:mcmle) at the same MCMC budget must reproduce\n")
cat("# these within ergm.rank's own seed-to-seed spread; the\n")
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
cat("# Monte-Carlo width of the MLE, the yardstick of the MCMLE tolerance above,\n")
cat("# and the unit the swap-MPLE gap is reported in: a gap of one of these means\n")
cat("# nothing; a gap of sixteen means a different estimator.\n")
cat(sprintf("mle_seed_sd = [%s]\n", num(seed_sd)))
cat(sprintf("mle_seed_mean = [%s]\n", num(colMeans(reps))))
cat(sprintf("mle_std_errors_rel_sd = [%s]\n", num(apply(rep_ses, 2, sd) / colMeans(rep_ses))))
