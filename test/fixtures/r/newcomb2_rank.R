# Golden fixture: statnet `ergm.rank` FITTED OUTPUT on Newcomb's fraternity
# ranks, WEEK 2, for a three-term model with the week-1 ranking as the
# reference of `rank.inconsistency`.
#
# Regenerate from the package root. THIS IS SLOW -- ~20 minutes (six MCMLE
# fits of ~3 minutes each: one frozen fit plus five replication seeds):
#
#   Rscript test/fixtures/r/newcomb2_rank.R > test/fixtures/newcomb2_rank.toml
#
# WHY A SECOND FITTED FIXTURE. `newcomb_rank.toml` pins the week-1 two-term
# model, on which ERGMRank.jl's swap pseudo-likelihood (`method=:mple`)
# happens to sit close to the MLE (0.30 and 0.43 of an MLE standard error).
# That closeness is a property of that model, not of the estimator: on THIS
# model the swap-MPLE is 0.7-0.9 of an MLE standard error away, its
# nonconformity coefficient more than twice the MLE's, and its
# pseudo-Hessian standard errors 2-3.3x too narrow. The Julia testset
#   * ASSERTS `fit_ergm_rank` (the default, method=:mcmle) against
#     ergm.rank's replication mean at ergm.rank's MCMC budget, within
#     ergm.rank's own seed-to-seed spread -- a second model, with a
#     different nonconformity variant and an inconsistency term, on which the
#     MCMLE must reproduce ergm.rank;
#   * CHARACTERISES the swap-MPLE's gap (every coefficient more than 0.5 of an
#     MLE standard error away, pseudo-Hessian standard errors about half the
#     MLE's or less), so the documentation's statement that the gap is
#     model-dependent is pinned to a number.
#
# The observed sufficient statistics are deterministic and asserted at 1e-9.

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(ergm.rank)
})

seed <- 20261002
data(newcomb)

nw <- newcomb[[2]]
n <- network.size(nw)
R1 <- as.matrix(newcomb[[1]], attrname = "rank")
R2 <- as.matrix(nw, attrname = "rank")

f <- nw ~ rank.deference + rank.nonconformity("localAND") +
  rank.inconsistency(newcomb[[1]], "rank")
obs <- summary(f, response = "rank")

# ergm() prints MCMLE iteration chatter to stdout, and this script's stdout IS
# the TOML fixture.
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

rep_seeds <- c(101, 202, 303, 404, 505)
rep_fits <- lapply(rep_seeds, fit_once)
reps <- t(sapply(rep_fits, coef))
rep_ses <- t(sapply(rep_fits, function(x) sqrt(diag(vcov(x)))))
seed_sd <- apply(reps, 2, sd)

num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")
strs <- function(x) paste(sprintf('"%s"', x), collapse = ", ")
rows <- function(M) paste(apply(M, 1, function(r) paste0("[", paste(r, collapse = ", "), "]")),
                          collapse = ", ")

cat('name = "newcomb2_rank"\n\n')

cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('ergm_rank_version = "%s"\n', as.character(packageVersion("ergm.rank"))))
cat(sprintf('ergm_version = "%s"\n', as.character(packageVersion("ergm"))))
cat(sprintf('network_version = "%s"\n', as.character(packageVersion("network"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/newcomb2_rank.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "ergm.rank::newcomb[[2]] (Newcomb 1961): 17 fraternity men, week 2, each ranking the other 16, with week 1 (newcomb[[1]]) as the reference ranking; GREATER value = HIGHER standing"\n')
cat('model = "newcomb[[2]] ~ rank.deference + rank.nonconformity(\\"localAND\\") + rank.inconsistency(newcomb[[1]], \\"rank\\"), response=\\"rank\\", reference=~CompleteOrder"\n')
cat('r_estimator = "MCMC-MLE (AlterSwap proposal)"\n')
cat('julia_estimator = "MCMC-MLE (fit_ergm_rank, the default method=:mcmle, AlterSwap proposal) -- ASSERTED; the swap pseudo-likelihood (method=:mple) is a DIFFERENT estimator whose gap is characterised; see [tolerance]"\n')
cat('mcmc_control = "control.ergm(MCMC.samplesize=2048, MCMC.burnin=8192, MCMC.interval=512)"\n')
cat(sprintf('replication_seeds = "%s"\n', paste(rep_seeds, collapse = ",")))
cat("\n")

# ---------------------------------------------------------------------------
# Why the equal-precision premise below is stated, not measured here. The band
# sqrt(1 + 1/5) * sd_R for one Julia draw against the mean of R's five assumes
# ERGMRank.jl's seed-to-seed sd is no larger than R's. The premise can only
# fail in the safe direction: were ERGMRank.jl's sd c * sd_R with c > 1, the
# band would be 4 * sqrt(1.2) / sqrt(c^2 + 0.2) of the difference's true sd
# (3.4 at c = 1.2, 2.8 at c = 1.5), so the testset would fail MORE often; it
# can never let a disagreement pass.
# ---------------------------------------------------------------------------
cat("[tolerance]\n")
cat("# Observed sufficient statistics: deterministic, machine precision.\n")
cat("summary_statistics = 1e-9\n")
cat("#\n")
cat("# COEFFICIENTS: `fit_ergm_rank` (method=:mcmle) at ergm.rank's MCMC budget\n")
cat("# (n_samples=2048, burnin=8192, interval=512, one chain, bridge skipped) is\n")
cat("# asserted on one Julia seed as\n")
cat("#   |coef - mle_seed_mean| <= coefficient_sd_multiple * sqrt(1 + 1/5) * mle_seed_sd,\n")
cat("# where `mle_seed_sd`/`mle_seed_mean` ([values]) are ergm.rank over its five\n")
cat("# replication seeds. No number in this block is measured on ERGMRank.jl.\n")
cat("# Justification of the rule: if ERGMRank.jl's MCMLE is the same estimator as\n")
cat("# ergm.rank's, one Julia estimate is one more draw from ergm.rank's\n")
cat("# seed-to-seed distribution, so its difference from the mean of five\n")
cat("# independent R draws has sd sqrt(1 + 1/5) * sd_R, and 4 of that is a\n")
cat("# 4-sigma band. Do not raise the multiple: a failure at 4 means the\n")
cat("# estimators disagree.\n")
cat("coefficient_sd_multiple = 4\n")
cat("#\n")
cat("# STANDARD ERRORS: within `std_errors_rtol` relative of `mle_std_errors_mean`\n")
cat("# (the mean over ergm.rank's five replication seeds; its own seed-to-seed\n")
cat("# relative spread is listed in [values] as `mle_std_errors_rel_sd`). Both\n")
cat("# are the inverse Fisher information of a 2048-draw sample plus the\n")
cat("# Monte-Carlo component. 0.15 as in newcomb_rank.toml: several times the\n")
cat("# Monte-Carlo noise of ergm.rank's standard errors (the testset asserts\n")
cat("# 0.15 >= 4 * sqrt(2) * mle_std_errors_rel_sd) and far below the swap-MPLE's\n")
cat("# 2-3.3x anticonservatism, which must keep failing it.\n")
cat("std_errors_rtol = 0.15\n")
cat("#\n")
cat("# THE SWAP-MPLE HAS NO TOLERANCE, BY DESIGN: the testset asserts that every\n")
cat("# coefficient is MORE than `mple_gap_min_se` of an MLE standard error from\n")
cat("# `mle_seed_mean` (observed 0.89, 0.83, 0.76) and that its pseudo-Hessian\n")
cat("# standard errors are at least `mple_se_ratio_min` times narrower than\n")
cat("# `mle_std_errors_mean` (observed 3.3, 2.2, 2.0).\n")
cat("mple_gap_min_se = 0.5\n")
cat("mple_se_ratio_min = 1.9\n")
cat("\n")

cat("[values]\n")
cat("# --- the rankings, frozen. Julia rebuilds them exactly. -----------------\n")
cat(sprintf("n_actors = %d\n", n))
cat(sprintf("ranks_week1 = [%s]\n", rows(R1)))
cat(sprintf("ranks_week2 = [%s]\n", rows(R2)))
cat("\n# --- observed sufficient statistics: deterministic, ASSERTED at 1e-9 ----\n")
cat(sprintf("summary_statistic_names = [%s]\n", strs(names(obs))))
cat(sprintf("summary_statistics = [%s]\n", num(as.numeric(obs))))
cat("\n# --- ergm.rank's MCMC MLE (the frozen seed) -------------------------------\n")
cat(sprintf("term_names = [%s]\n", strs(names(mle_coef))))
cat(sprintf("mle_coefficients = [%s]\n", num(as.numeric(mle_coef))))
cat(sprintf("mle_std_errors = [%s]\n", num(as.numeric(mle_se))))
cat("\n# --- ergm.rank over five further seeds -------------------------------------\n")
cat(sprintf("mle_seed_sd = [%s]\n", num(seed_sd)))
cat(sprintf("mle_seed_mean = [%s]\n", num(colMeans(reps))))
cat(sprintf("mle_std_errors_mean = [%s]\n", num(colMeans(rep_ses))))
cat(sprintf("mle_std_errors_rel_sd = [%s]\n", num(apply(rep_ses, 2, sd) / colMeans(rep_ses))))
