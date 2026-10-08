# Golden fixture: statnet `ergm.rank`'s WEIGHTED `rank.inconsistency(x,
# weights=, wtname=, wtcenter=)` summary statistics. Fast (a `summary()` call,
# seconds). Regenerate from the package root:
#
#   Rscript test/fixtures/r/rank_inconsistency_weights.R > test/fixtures/rank_inconsistency_weights.toml
#
# WHAT THIS FIXTURE ASSERTS. ERGMRank.jl's `RankInconsistency(ref; weights,
# wtname, wtcenter)` computes
#   sum_i sum_{j != k} w[i, j, k] * 1[(y_ij > y_ik) != (ref_ij > ref_ik)]
# with the weights as ergm.rank documents them: a function (i, j, k) is
# evaluated on the triples of distinct actors (NA elsewhere), an array is
# taken as given, `wtcenter=TRUE` subtracts the mean of the non-NA weights,
# and NA becomes 0. This fixture checks that reading against R's `summary()`
# output for each case. Every value is a deterministic
# function of frozen integer inputs, asserted at 1e-9.
#
# Cases, on the two rankings of `rank_terms.R` (the 4-actor network and the
# seeded 8-actor ranking, same seed, so the same `ranks`/`ref`):
#   fun       weights = function(i, j, k) i + 2 * j - k        (asymmetric in j, k)
#   fun_c     the same, wtcenter = TRUE   (mean over distinct triples only)
#   arr       weights = a frozen integer n x n x n array in -2:4
#   arr_c     the same, wtcenter = TRUE   (mean over ALL n^3 entries: none is NA)
#   arr_na_c  the array with NA in every cell with a repeated index,
#             wtcenter = TRUE             (mean over distinct triples only)
#
# The names R gives the five statistics are recorded: R appends
# ":<wtname>" and, when centred, "c". ERGMRank.jl's names are
# "rank.inconsistency:<wtname>[c]".
#
# NOT asserted: ergm.rank's MCMC *change* statistic for the weighted term.
# In ergm.rank 4.1.2 the change statistic of the weighted term tests the
# same comparison weight twice where it should test the weight and its
# mirror, so a comparison whose weight is 0 while its mirror's is not is
# skipped by R's sampler. ERGMRank.jl's `swap_change` is
# pinned to the brute-force difference of two `compute`s instead.

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(ergm.rank)
})

seed <- 20260912

rank_network <- function(m) {
  as.network(m, directed = TRUE, matrix.type = "adjacency",
             ignore.eval = FALSE, names.eval = "rank")
}

wfun <- function(i, j, k) i + 2 * j - k

na_repeated <- function(a) {
  n <- dim(a)[1]
  for (i in 1:n) for (j in 1:n) for (k in 1:n)
    if (i == j || i == k || j == k) a[i, j, k] <- NA
  a
}

stats_of <- function(m, ref, arr) {
  nw <- rank_network(m)
  arr_na <- na_repeated(arr)
  f <- nw ~ rank.inconsistency(ref, weights = wfun, wtname = "fun") +
    rank.inconsistency(ref, weights = wfun, wtname = "fun", wtcenter = TRUE) +
    rank.inconsistency(ref, weights = arr, wtname = "arr") +
    rank.inconsistency(ref, weights = arr, wtname = "arr", wtcenter = TRUE) +
    rank.inconsistency(ref, weights = arr_na, wtname = "arrna", wtcenter = TRUE) +
    rank.inconsistency(ref, weights = arr)
  summary(f, response = "rank")
}

# --- the 4-actor network (as rank_terms.R) -----------------------------------
m4 <- matrix(c(0, 3, 2, 1,
               3, 0, 1, 2,
               1, 3, 0, 2,
               2, 1, 3, 0), 4, 4, byrow = TRUE)
ref4 <- m4
ref4[1, ] <- c(0, 1, 2, 3)

# --- the seeded 8-actor ranking (as rank_terms.R: same seed, same draws) -----
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

# --- the frozen weight arrays (drawn AFTER everything rank_terms.R draws) -----
arr4 <- array(sample(-2:4, 4^3, replace = TRUE), c(4, 4, 4))
arr8 <- array(sample(-2:4, 8^3, replace = TRUE), c(8, 8, 8))

s4 <- stats_of(m4, ref4, arr4)
s8 <- stats_of(m8, ref8, arr8)

num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")
ints <- function(x) paste(as.integer(x), collapse = ", ")
strs <- function(x) paste(sprintf('"%s"', x), collapse = ", ")
rows <- function(M) paste(apply(M, 1, function(r) paste0("[", ints(r), "]")),
                          collapse = ", ")

cat('name = "rank_inconsistency_weights"\n\n')

cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('ergm_rank_version = "%s"\n', as.character(packageVersion("ergm.rank"))))
cat(sprintf('ergm_version = "%s"\n', as.character(packageVersion("ergm"))))
cat(sprintf('network_version = "%s"\n', as.character(packageVersion("network"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/rank_inconsistency_weights.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "the 4-actor ERGMRank.jl test network and the seeded 8-actor ranking of rank_terms.R, with their reference rankings; GREATER value = HIGHER standing"\n')
cat('model = "summary(nw ~ rank.inconsistency(ref, weights=, wtname=, wtcenter=), response=\\"rank\\") for a function and an array of weights, centred and not"\n')
cat('what = "summary statistics only: deterministic functions of the ranking, the reference and the weights"\n')
cat("\n")

cat("[tolerance]\n")
cat("# Sums of frozen integer weights (minus, when centred, their mean): exact in\n")
cat("# Float64 up to the rounding of the mean. 1e-9 is machine precision with\n")
cat("# headroom; a disagreement is a bug in the weighted statistic.\n")
cat("default = 1e-9\n\n")

cat("[values]\n")
cat("# The weights function is (i, j, k) -> i + 2j - k. The arrays are stored\n")
cat("# flat in R's column-major order (first index fastest), as Julia's\n")
cat("# reshape(v, n, n, n) reads them.\n")
cat("# Statistic order: fun, fun centred, arr, arr centred, arr with NA in the\n")
cat("# repeated-index cells centred, arr without a wtname.\n")
cat(sprintf("r_names = [%s]\n", strs(names(s4))))
cat("\nn4 = 4\n")
cat(sprintf("ranks4 = [%s]\n", rows(m4)))
cat(sprintf("ref4 = [%s]\n", rows(ref4)))
cat(sprintf("weights4 = [%s]\n", ints(arr4)))
cat(sprintf("stats4 = [%s]\n", num(as.numeric(s4))))
cat(sprintf("\nn8 = %d\n", n8))
cat(sprintf("ranks8 = [%s]\n", rows(m8)))
cat(sprintf("ref8 = [%s]\n", rows(ref8)))
cat(sprintf("weights8 = [%s]\n", ints(arr8)))
cat(sprintf("stats8 = [%s]\n", num(as.numeric(s8))))
