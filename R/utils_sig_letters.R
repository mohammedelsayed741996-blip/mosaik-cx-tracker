# =============================================================================
# utils_sig_letters.R - column-letter significance tests for banner tables
# -----------------------------------------------------------------------------
# Convention (as in Q / Quantum / SPSS Custom Tables "compare column
# proportions"): every banner column gets a letter (A, B, C ...). Within one
# banner group, each pair of columns is tested; when column X is significantly
# HIGHER than column Y at p < alpha, Y's letter is printed under X's value.
#
#   Proportions: two-proportion z-test, pooled, on EFFECTIVE bases
#                n_eff = (sum w)^2 / sum(w^2), so weighting does not inflate
#                significance.
#                  p_pool = (p1*n1 + p2*n2) / (n1 + n2)
#                  se     = sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
#                  z      = (p1 - p2) / se,  p = 2 * (1 - pnorm(|z|))
#   Means:       z = (m1 - m2) / sqrt(s1^2/n1 + s2^2/n2) on effective bases
#                (used for NPS, the mean of -100 / 0 / +100).
#
# Columns with a base below `min_base` are not tested (and suppressed in the
# tables). The Total column is never tested against its own subgroups:
# callers pass one banner group at a time.
# No multiple-comparison correction, matching common market research practice;
# the table footnote says so.
# =============================================================================

#' Effective base size of a weight vector (Kish).
n_effective <- function(w) {
  w <- w[!is.na(w)]
  if (!length(w)) return(0)
  sum(w)^2 / sum(w^2)
}

#' Weighted proportion of x == 1 (x is 0/1, NA = not in base).
weighted_prop <- function(x, w) {
  ok <- !is.na(x) & !is.na(w)
  if (!any(ok)) return(NA_real_)
  sum(w[ok] * x[ok]) / sum(w[ok])
}

#' Weighted mean and weighted standard deviation.
weighted_mean_sd <- function(x, w) {
  ok <- !is.na(x) & !is.na(w)
  if (!any(ok)) return(c(mean = NA_real_, sd = NA_real_))
  m <- sum(w[ok] * x[ok]) / sum(w[ok])
  v <- sum(w[ok] * (x[ok] - m)^2) / sum(w[ok])
  c(mean = m, sd = sqrt(v))
}

#' Pooled two-proportion z-test. Returns list(z, p, se).
z_test_props <- function(p1, n1, p2, n2) {
  if (any(is.na(c(p1, n1, p2, n2))) || n1 <= 0 || n2 <= 0) return(list(z = NA_real_, p = NA_real_, se = NA_real_))
  pool <- (p1 * n1 + p2 * n2) / (n1 + n2)
  se <- sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
  if (se == 0) return(list(z = NA_real_, p = NA_real_, se = 0))   # both 0% or both 100%
  z <- (p1 - p2) / se
  list(z = z, p = 2 * pnorm(-abs(z)), se = se)
}

#' z-test for two independent means (unpooled). Returns list(z, p, se).
z_test_means <- function(m1, s1, n1, m2, s2, n2) {
  if (any(is.na(c(m1, s1, n1, m2, s2, n2))) || n1 <= 0 || n2 <= 0) return(list(z = NA_real_, p = NA_real_, se = NA_real_))
  se <- sqrt(s1^2 / n1 + s2^2 / n2)
  if (se == 0) return(list(z = NA_real_, p = NA_real_, se = 0))
  z <- (m1 - m2) / se
  list(z = z, p = 2 * pnorm(-abs(z)), se = se)
}

#' Significance letters for one banner group.
#'
#' @param est     estimates per column (proportions 0-1, or means)
#' @param n_eff   effective base per column
#' @param col_letters letters of these columns (e.g. c("B","C","D","E"))
#' @param sd      standard deviations per column (only for type = "mean")
#' @param type    "prop" or "mean"
#' @param alpha   significance level (two-sided)
#' @param min_base columns with n_eff below this are not tested
#' @return character vector: for each column, the letters of the columns it is
#'         significantly higher than ("" if none, NA if not tested)
sig_letters <- function(est, n_eff, col_letters, sd = NULL, type = c("prop", "mean"),
                        alpha = 0.05, min_base = 30) {
  type <- match.arg(type)
  k <- length(est)
  stopifnot(length(n_eff) == k, length(col_letters) == k)
  if (type == "mean") stopifnot(length(sd) == k)
  testable <- !is.na(est) & !is.na(n_eff) & n_eff >= min_base
  out <- ifelse(testable, "", NA_character_)
  for (i in seq_len(k)) {
    if (!testable[i]) next
    hits <- character(0)
    for (j in seq_len(k)) {
      if (i == j || !testable[j]) next
      res <- if (type == "prop") z_test_props(est[i], n_eff[i], est[j], n_eff[j])
             else z_test_means(est[i], sd[i], n_eff[i], est[j], sd[j], n_eff[j])
      if (!is.na(res$p) && res$p < alpha && est[i] > est[j]) hits <- c(hits, col_letters[j])
    }
    out[i] <- paste(sort(hits), collapse = "")
  }
  out
}

#' Letters for n columns: A..Z, then AA, AB ...
column_letters <- function(n) {
  single <- LETTERS
  if (n <= 26) return(single[seq_len(n)])
  c(single, as.vector(outer(single, single, paste0)))[seq_len(n)]
}
