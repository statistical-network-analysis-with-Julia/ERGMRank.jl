# Golden fixture: statnet `ergm.rank` FITTED OUTPUT on Newcomb's fraternity ranks.
#
# Regenerate from the package root. THIS IS SLOW -- ~17 minutes: each MCMLE fit of
# a rank ERGM takes ~3 minutes, and the script does six of them (one frozen fit
# plus five replication seeds):
#
#   Rscript test/fixtures/r/newcomb_rank.R > test/fixtures/newcomb_rank.toml
#
# READ THIS BEFORE TRUSTING ANY NUMBER BELOW: THE TWO PACKAGES DO NOT FIT THE SAME
# ESTIMATOR, AND THIS FIXTURE EXISTS TO MEASURE THE GAP, NOT TO PAPER OVER IT.
#
#   ergm.rank fits the MCMC MLE. It samples complete orderings under the
#   CompleteOrder reference with the AlterSwap proposal and solves
#   E_theta[g] = g_obs by MCMC-MLE.
#
#   ERGMRank.jl fits a SWAP PSEUDO-LIKELIHOOD. For each ego and each unordered
#   pair of alters it writes down the logistic probability of the observed
#   relative order given everything else, and multiplies those across all
#   (ego, pair) comparisons as if they were independent. They are not
#   independent -- the comparisons within an ego's ranking overlap heavily -- and
#   ERGMRank.jl's own documentation is explicit that no consistency result is
#   claimed for this estimator.
#
# A pseudo-likelihood is not an approximation to the MLE with a small error; it
# is a different estimator, and for rank data with strongly overlapping
# comparisons there is no reason to expect the two to agree to any particular
# tolerance. So this fixture does NOT assert that ERGMRank.jl reproduces
# ergm.rank's coefficients. It freezes ergm.rank's MLE, freezes ergm.rank's own
# seed-to-seed Monte-Carlo spread, and lets the Julia testset RECORD the
# discrepancy with @test_broken. Choosing a tolerance wide enough to make the
# comparison green would be a tolerance chosen to hide the finding, which is the
# one thing a golden fixture must never be used for.
#
# WHAT *IS* ASSERTED, exactly, is the deterministic half: the observed sufficient
# statistics. Those are a function of the ranking alone -- no estimator, no Monte
# Carlo -- so they get machine precision, and a disagreement there would be a bug
# in a term formula. (ERGMRank.jl already golden-tests its term VALUES against
# ergm.rank 4.1.2; this re-pins them from a provenanced file rather than from
# literals in a comment, and adds the estimator comparison the term tests could
# not make.)

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
cat('julia_estimator = "swap pseudo-likelihood -- a DIFFERENT estimator, not an approximation to the above; see [tolerance]"\n')
cat('mcmc_control = "control.ergm(MCMC.samplesize=2048, MCMC.burnin=8192, MCMC.interval=512)"\n')
cat(sprintf('replication_seeds = "%s"\n', paste(rep_seeds, collapse = ",")))
cat("\n")

cat("[tolerance]\n")
cat("# Observed sufficient statistics: a deterministic function of the ranking.\n")
cat("# No estimator, no Monte Carlo. Machine precision, and a disagreement is a\n")
cat("# bug in a term formula, full stop.\n")
cat("summary_statistics = 1e-9\n")
cat("#\n")
cat("# COEFFICIENTS: THERE IS NO HONEST TOLERANCE HERE, AND THAT IS THE FINDING.\n")
cat("#\n")
cat("# ergm.rank fits the MCMC MLE. ERGMRank.jl fits a SWAP PSEUDO-LIKELIHOOD --\n")
cat("# the product of the logistic probabilities of each (ego, alter-pair)\n")
cat("# relative order, multiplied as if independent, which they emphatically are\n")
cat("# not. That is a different estimator, not a noisy version of the same one,\n")
cat("# and ERGMRank.jl's own docs decline to claim consistency for it.\n")
cat("#\n")
cat("# So no tolerance is stated for the coefficients, and the Julia testset marks\n")
cat("# the comparison @test_broken and PRINTS the gap in units of `mle_seed_sd`\n")
cat("# below -- ergm.rank's own Monte-Carlo width, which is the only scale on\n")
cat("# which 'how far apart are they' means anything. Setting a tolerance wide\n")
cat("# enough to turn that green would be a tolerance chosen to hide the result.\n")
cat("#\n")
cat("# `mle_coefficients` in [values] is therefore FROZEN, not ASSERTED. When\n")
cat("# ERGMRank.jl grows a real MCMLE, delete the @test_broken and assert against\n")
cat("# it at a multiple of `mle_seed_sd`, exactly as the ERGM.jl fixture does.\n")
cat("\n")

cat("[values]\n")
cat("# --- the ranking, frozen. Julia rebuilds it exactly. --------------------\n")
cat(sprintf("n_actors = %d\n", n))
cat(sprintf("ranks = [%s]\n", rows))
cat("\n# --- observed sufficient statistics: deterministic, ASSERTED at 1e-9 ----\n")
cat(sprintf("summary_statistic_names = [%s]\n", strs(names(obs))))
cat(sprintf("summary_statistics = [%s]\n", num(as.numeric(obs))))
cat("\n# --- ergm.rank's MCMC MLE: FROZEN, NOT ASSERTED ------------------------\n")
cat("# ERGMRank.jl does not fit this estimator (see [tolerance]). These are the\n")
cat("# numbers a future MCMLE must reproduce, and the numbers the current swap\n")
cat("# pseudo-likelihood is measured AGAINST rather than held TO.\n")
cat(sprintf("term_names = [%s]\n", strs(names(mle_coef))))
cat(sprintf("mle_coefficients = [%s]\n", num(as.numeric(mle_coef))))
cat(sprintf("mle_std_errors = [%s]\n", num(as.numeric(mle_se))))
cat("\n# ergm.rank disagreeing with ITSELF over five further seeds. This is the\n")
cat("# Monte-Carlo width of the MLE and the unit the Julia-vs-R gap is reported\n")
cat("# in: a gap of one of these means nothing; a gap of a hundred means the two\n")
cat("# estimators are answering different questions.\n")
cat(sprintf("mle_seed_sd = [%s]\n", num(seed_sd)))
cat(sprintf("mle_seed_mean = [%s]\n", num(colMeans(reps))))
