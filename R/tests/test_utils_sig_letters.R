#!/usr/bin/env Rscript
# =============================================================================
# Unit tests for R/utils_sig_letters.R, against hand-calculated examples.
# Run from the repository root:  Rscript R/tests/test_utils_sig_letters.R
# Base R only (no testthat needed); exits with status 1 on any failure.
# =============================================================================
source("R/utils_sig_letters.R")

failures <- 0
check <- function(desc, ok) {
  cat(sprintf("%-62s %s\n", desc, if (isTRUE(ok)) "ok" else "FAIL"))
  if (!isTRUE(ok)) failures <<- failures + 1
}
near <- function(a, b, tol = 1e-5) isTRUE(all.equal(a, b, tolerance = tol, scale = 1))

# -----------------------------------------------------------------------------
# Example 1 - proportions, worked by hand
#   Column A: 60% of n_eff 200;  B: 48% of 150;  C: 55% of 100
#
#   A vs B: p_pool = (0.60*200 + 0.48*150) / 350 = 192 / 350 = 0.548571
#           se     = sqrt(0.548571 * 0.451429 * (1/200 + 1/150))
#                  = sqrt(0.247641 * 0.0116667) = 0.053751
#           z      = (0.60 - 0.48) / 0.053751 = 2.2325  ->  p = 0.0256 < 0.05
#           => A is significantly higher than B: A gets letter "B"
#   A vs C: p_pool = 175 / 300 = 0.583333, se = 0.060381,
#           z = 0.05 / 0.060381 = 0.8281 -> p = 0.4076 (not significant)
#   B vs C: p_pool = 127 / 250 = 0.508, se = 0.064541,
#           z = -0.07 / 0.064541 = -1.0846 -> p = 0.2781 (not significant)
#   Expected letters: A = "B", B = "", C = ""
# -----------------------------------------------------------------------------
r <- z_test_props(0.60, 200, 0.48, 150)
check("A vs B: standard error 0.053751", near(r$se, 0.053751))
check("A vs B: z = 2.232527", near(r$z, 2.232527))
check("A vs B: p = 0.02558", near(r$p, 0.02558))
r <- z_test_props(0.60, 200, 0.55, 100)
check("A vs C: z = 0.828079, p = 0.407626", near(r$z, 0.828079) && near(r$p, 0.407626))
r <- z_test_props(0.48, 150, 0.55, 100)
check("B vs C: z = -1.084574, p = 0.27811", near(r$z, -1.084574) && near(r$p, 0.27811))

lt <- sig_letters(c(0.60, 0.48, 0.55), c(200, 150, 100), c("A", "B", "C"))
check("letters: A = 'B', B = '', C = ''", identical(lt, c("B", "", "")))

# Same test, columns in another order -> letters follow the columns
lt <- sig_letters(c(0.48, 0.60, 0.55), c(150, 200, 100), c("A", "B", "C"))
check("order does not matter: B = 'A'", identical(lt, c("", "A", "")))

# -----------------------------------------------------------------------------
# Example 2 - small bases are not tested
#   Same values, but C has n_eff 25 (< 30): C is NA and never appears as a letter
# -----------------------------------------------------------------------------
lt <- sig_letters(c(0.60, 0.48, 0.90), c(200, 150, 25), c("A", "B", "C"))
check("base < 30 is not tested (NA) and never cited", identical(lt, c("B", "", NA_character_)))

# -----------------------------------------------------------------------------
# Example 3 - 0% vs 0% has no variance: no test, no letter
# -----------------------------------------------------------------------------
lt <- sig_letters(c(0, 0), c(100, 100), c("A", "B"))
check("0% vs 0%: no letters", identical(lt, c("", "")))

# -----------------------------------------------------------------------------
# Example 4 - means (NPS), worked by hand
#   A: NPS +30, sd 60, n_eff 400;  B: NPS +10, sd 65, n_eff 300
#   se = sqrt(60^2/400 + 65^2/300) = sqrt(9 + 14.083333) = 4.804512
#   z  = 20 / 4.804512 = 4.162754 -> p = 0.0000314
# -----------------------------------------------------------------------------
r <- z_test_means(30, 60, 400, 10, 65, 300)
check("means: se = 4.804512, z = 4.162754", near(r$se, 4.804512) && near(r$z, 4.162754))
check("means: p = 0.0000314432", near(r$p, 3.14432e-05, tol = 1e-9))
lt <- sig_letters(c(30, 10), c(400, 300), c("A", "B"), sd = c(60, 65), type = "mean")
check("means letters: A = 'B'", identical(lt, c("B", "")))

# -----------------------------------------------------------------------------
# Example 5 - effective base: weights 1, 1, 2, 2 -> (6^2) / (1+1+4+4) = 3.6
# -----------------------------------------------------------------------------
check("n_effective(c(1,1,2,2)) = 3.6", near(n_effective(c(1, 1, 2, 2)), 3.6))
check("n_effective of equal weights = n", near(n_effective(rep(0.7, 50)), 50))
check("weighted_prop(c(1,0,1), c(1,1,2)) = 0.75", near(weighted_prop(c(1, 0, 1), c(1, 1, 2)), 0.75))
check("column_letters(28) ends with AA, AB", identical(tail(column_letters(28), 2), c("AA", "AB")))

cat(sprintf("\n%d failure(s)\n", failures))
if (failures > 0) quit(status = 1)
