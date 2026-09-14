# Golden fixture: statnet `ergm.rank` SUMMARY STATISTICS of every ERGMRank.jl
# term, on two rankings. Fast (a `summary()` call, seconds). Regenerate from
# the package root:
#
#   Rscript test/fixtures/r/rank_terms.R > test/fixtures/rank_terms.toml
#
# WHAT THIS FIXTURE ASSERTS. The six term statistics of ERGMRank.jl --
# `RankDeference`, `RankNonconformity(:all)`, `RankNonconformity(:localAND)`,
# `RankNodeICov(x)`, `RankInconsistency(ref)`, `RankEdgeCov(cov)` -- are
# ports of ergm.rank's `rank.deference`, `rank.nonconformity("all")`,
# `rank.nonconformity("localAND")`, `rank.nodeicov`, `rank.inconsistency` and
# `rank.edgecov` summary functions (wtchangestats_rank.c). A statistic is a
# deterministic function of the ranking and the covariates -- no estimator,
# no Monte Carlo -- so every value below is asserted at 1e-9 (the counting
# terms and the integer-valued covariate terms are exact integers), and a
# disagreement is a bug in a term formula.
#
# Two rankings:
#   * the 4-actor test network of ERGMRank.jl's README/docstrings (ego 1 ranks
#     2 > 3 > 4, ...), with `x = (10, 20, 30, 40)`, a reference ranking that
#     reverses ego 1's row, and `cov[i, j] = x[j]` (so rank.edgecov(cov) must
#     equal rank.nodeicov(x));
#   * a seeded random 8-actor ranking (each ego a random permutation of 1:7)
#     with integer covariates `x` in -5:5, a random integer `cov` in -3:3, and
#     an independent random reference ranking -- larger, and with every term
#     nonzero and non-degenerate. Everything random is frozen in [values] so
#     Julia rebuilds the inputs exactly.
#
# GREATER rank value = HIGHER standing (ergm.rank's convention and ERGMRank.jl's).

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(ergm.rank)
})

seed <- 20260912

rank_network <- function(m) {
  as.network(m, directed = TRUE, matrix.type = "adjacency",
             ignore.eval = FALSE, names.eval = "rank")
}

stats_of <- function(m, x, ref, cov) {
  nw <- rank_network(m)
  nw %v% "x" <- x
  f <- nw ~ rank.deference + rank.nonconformity("all") +
    rank.nonconformity("localAND") + rank.nodeicov("x") +
    rank.inconsistency(ref) + rank.edgecov(cov)
  summary(f, response = "rank")
}

# --- the 4-actor network -----------------------------------------------------
m4 <- matrix(c(0, 3, 2, 1,
               3, 0, 1, 2,
               1, 3, 0, 2,
               2, 1, 3, 0), 4, 4, byrow = TRUE)
x4 <- c(10, 20, 30, 40)
ref4 <- m4
ref4[1, ] <- c(0, 1, 2, 3)                 # ego 1's ranking reversed
cov4 <- matrix(rep(x4, each = 4), 4, 4)    # cov[i, j] = x[j]
s4 <- stats_of(m4, x4, ref4, cov4)

# --- a seeded random 8-actor ranking ------------------------------------------
set.seed(seed)
n8 <- 8
random_ranking <- function(n) {
  m <- matrix(0L, n, n)
  for (i in seq_len(n)) m[i, -i] <- sample(n - 1)
  m
}
m8 <- random_ranking(n8)
x8 <- sample(-5:5, n8, replace = TRUE)
cov8 <- matrix(sample(-3:3, n8 * n8, replace = TRUE), n8, n8)
ref8 <- random_ranking(n8)
s8 <- stats_of(m8, x8, ref8, cov8)

num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")
ints <- function(x) paste(as.integer(x), collapse = ", ")
strs <- function(x) paste(sprintf('"%s"', x), collapse = ", ")
rows <- function(M) paste(apply(M, 1, function(r) paste0("[", ints(r), "]")),
                          collapse = ", ")

cat('name = "rank_terms"\n\n')

cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('ergm_rank_version = "%s"\n', as.character(packageVersion("ergm.rank"))))
cat(sprintf('ergm_version = "%s"\n', as.character(packageVersion("ergm"))))
cat(sprintf('network_version = "%s"\n', as.character(packageVersion("network"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/rank_terms.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "the 4-actor ERGMRank.jl test network (README/docstrings) and a seeded random 8-actor complete ranking; GREATER value = HIGHER standing"\n')
cat('model = "summary(nw ~ rank.deference + rank.nonconformity(\\"all\\") + rank.nonconformity(\\"localAND\\") + rank.nodeicov(\\"x\\") + rank.inconsistency(ref) + rank.edgecov(cov), response=\\"rank\\")"\n')
cat('what = "summary statistics only: deterministic functions of the ranking and the covariates"\n')
cat("\n")

cat("[tolerance]\n")
cat("# Every value is a deterministic function of a frozen ranking and frozen\n")
cat("# integer covariates: the counting terms are integers and the covariate\n")
cat("# terms are sums of integer differences, so both sides are exact in\n")
cat("# Float64. 1e-9 is machine precision with headroom; a disagreement is a\n")
cat("# bug in a term formula, full stop.\n")
cat("default = 1e-9\n\n")

cat("[values]\n")
cat("# One name per statistic, in the order of the vectors below (ergm.rank's\n")
cat("# summary() names).\n")
cat(sprintf("statistic_names = [%s]\n", strs(names(s4))))
cat("\n# --- the 4-actor network: ranks (row = ego), covariate, reference, cov -----\n")
cat("n4 = 4\n")
cat(sprintf("ranks4 = [%s]\n", rows(m4)))
cat(sprintf("x4 = [%s]\n", ints(x4)))
cat(sprintf("ref4 = [%s]\n", rows(ref4)))
cat(sprintf("cov4 = [%s]\n", rows(cov4)))
cat(sprintf("stats4 = [%s]\n", num(as.numeric(s4))))
cat("\n# --- the seeded 8-actor ranking ---------------------------------------------\n")
cat(sprintf("n8 = %d\n", n8))
cat(sprintf("ranks8 = [%s]\n", rows(m8)))
cat(sprintf("x8 = [%s]\n", ints(x8)))
cat(sprintf("ref8 = [%s]\n", rows(ref8)))
cat(sprintf("cov8 = [%s]\n", rows(cov8)))
cat(sprintf("stats8 = [%s]\n", num(as.numeric(s8))))
