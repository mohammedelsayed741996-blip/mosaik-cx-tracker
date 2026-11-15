#!/usr/bin/env Rscript
# =============================================================================
# 04_weighting.R - Mosaik Bank CX Tracker 2026: rake each wave to Destatis
# population margins and trim the weights.
# -----------------------------------------------------------------------------
# Input   data/clean/tracker_clean.csv         (step 03)
#         config/weighting_targets.csv         (src/build_weighting_targets.py;
#                                               Destatis 12411-0013, 31.12.2025)
# Method  per wave, separately:
#           1. start from equal weights
#           2. rake (iterative proportional fitting) to three margins:
#              age_band (5) x gender_w (2) x region (4), population 18-69
#           3. trim to [0.3, 3.0] (relative to a mean weight of 1), re-rake,
#              and repeat until the weights stay inside the bounds and every
#              margin is within 0.5 points of its target
# Output  data/clean/weights.csv               resp_id, wave, weight
#         data/clean/tracker_weighted.csv      clean file + weight
#         outputs/weighting_diagnostics.csv    margins: sample vs weighted vs target
#         outputs/weighting_report.md          per-wave summary (Kish efficiency etc.)
# Run     Rscript R/04_weighting.R             (from the repository root)
# Needs   install.packages("survey")
# =============================================================================

suppressPackageStartupMessages(library(survey))

# --- settings -----------------------------------------------------------------
CLEAN_CSV    <- "data/clean/tracker_clean.csv"
TARGETS_CSV  <- "config/weighting_targets.csv"
OUT_WEIGHTS  <- "data/clean/weights.csv"
OUT_WEIGHTED <- "data/clean/tracker_weighted.csv"
OUT_DIAG     <- "outputs/weighting_diagnostics.csv"
OUT_REPORT   <- "outputs/weighting_report.md"

MARGINS      <- c("age_band", "gender_w", "region")
TRIM_LOWER   <- 0.3
TRIM_UPPER   <- 3.0
MAX_DIFF_PP  <- 0.5      # max allowed |weighted % - target %| per margin cell
MAX_ROUNDS   <- 25       # rake <-> trim rounds
SEED         <- 42       # only used if gender_w has to be created here

for (f in c(CLEAN_CSV, TARGETS_CSV)) {
  if (!file.exists(f)) stop("Missing ", f, ". Run from the repository root after steps 01-03.")
}
dir.create("outputs", showWarnings = FALSE)

# --- read -----------------------------------------------------------------------
d <- read.csv(CLEAN_CSV, na.strings = "", stringsAsFactors = FALSE, fileEncoding = "UTF-8")
targets <- read.csv(TARGETS_CSV, stringsAsFactors = FALSE)
targets <- targets[targets$variable %in% MARGINS, ]
cat(sprintf("Read %d respondents in %d waves; targets: %s (%s, reference date %s)\n",
            nrow(d), length(unique(d$wave)), TARGETS_CSV, targets$source[1], targets$reference_date[1]))

# --- gender_w ---------------------------------------------------------------------
# Census figures know two sexes (Destatis assigns "divers" / "ohne Angabe" to
# male or female). Step 03 already creates gender_w; this function checks it and
# only builds it, with the same rule, if it is missing.
ensure_gender_w <- function(d) {
  needs <- !("gender_w" %in% names(d)) || any(is.na(d$gender_w))
  if (!needs) {
    stopifnot(all(d$gender_w %in% c(1, 2)))
    n_imp <- sum(!(d$gender %in% c(1, 2)))
    cat(sprintf("gender_w present (from step 03); %d respondents were assigned a weighting gender\n", n_imp))
    return(d)
  }
  set.seed(SEED)
  if (!("gender_w" %in% names(d))) d$gender_w <- NA_integer_
  d$gender_w[d$gender %in% c(1, 2)] <- d$gender[d$gender %in% c(1, 2)]
  todo <- which(is.na(d$gender_w))
  for (key in unique(paste(d$wave[todo], d$age_band[todo]))) {
    rows <- todo[paste(d$wave[todo], d$age_band[todo]) == key]
    ref  <- d$gender[paste(d$wave, d$age_band) == key & d$gender %in% c(1, 2)]
    p_male <- if (length(ref)) mean(ref == 1) else 0.5
    d$gender_w[rows] <- ifelse(runif(length(rows)) < p_male, 1L, 2L)
  }
  cat(sprintf("gender_w created here for %d respondents (seed %d)\n", length(todo), SEED))
  d
}
d <- ensure_gender_w(d)

missing_margin <- sapply(MARGINS, function(v) sum(is.na(d[[v]])))
if (any(missing_margin > 0)) stop("Missing values in weighting variables: ",
                                  paste(names(missing_margin)[missing_margin > 0], collapse = ", "))

# Weighting variables as factors with the target levels, so sample and
# population tables line up exactly
for (v in MARGINS) {
  lv <- sort(unique(targets$level[targets$variable == v]))
  if (!all(unique(d[[v]]) %in% lv)) stop(v, " has levels not in the targets")
  d[[paste0(v, "_f")]] <- factor(d[[v]], levels = lv)
}

# --- helpers -------------------------------------------------------------------------
population_margins <- function(n) {
  lapply(MARGINS, function(v) {
    t <- targets[targets$variable == v, ]
    out <- data.frame(factor(t$level, levels = sort(t$level)), t$share * n)
    names(out) <- c(paste0(v, "_f"), "Freq")
    out
  })
}
sample_margins <- lapply(MARGINS, function(v) as.formula(paste0("~", v, "_f")))

margin_table <- function(dw, w, wave) {
  do.call(rbind, lapply(MARGINS, function(v) {
    t <- targets[targets$variable == v, ]
    unw <- prop.table(table(dw[[paste0(v, "_f")]]))
    wtd <- tapply(w, dw[[paste0(v, "_f")]], sum) / sum(w)
    data.frame(wave = wave, variable = v, level = t$level, label = t$label,
               sample_pct = round(100 * as.numeric(unw[as.character(t$level)]), 2),
               weighted_pct = round(100 * as.numeric(wtd[as.character(t$level)]), 2),
               target_pct = round(100 * t$share, 2))
  }))
}

kish <- function(w) length(w) * sum(w^2) / sum(w)^2      # design effect from weighting

rake_and_trim <- function(dw) {
  des <- svydesign(ids = ~1, weights = ~.one, data = transform(dw, .one = 1))   # equal starting weights
  pm  <- population_margins(nrow(dw))
  for (r in seq_len(MAX_ROUNDS)) {   # r, not "round": never mask base::round
    des <- rake(des, sample.margins = sample_margins, population.margins = pm,
                control = list(maxit = 100, epsilon = 1e-7, verbose = FALSE))
    w <- weights(des)
    w <- w / mean(w)
    inside <- all(w >= TRIM_LOWER - 1e-9 & w <= TRIM_UPPER + 1e-9)
    if (inside) break
    des <- trimWeights(des, lower = TRIM_LOWER * mean(weights(des)),
                       upper = TRIM_UPPER * mean(weights(des)), strict = TRUE)
  }
  w <- weights(des)
  list(w = w / mean(w), rounds = r, converged = inside)
}

# --- weight each wave --------------------------------------------------------------
d$weight <- NA_real_
diag_rows <- list(); summary_rows <- list()
for (wv in sort(unique(d$wave))) {
  idx <- which(d$wave == wv)
  dw  <- d[idx, ]
  res <- rake_and_trim(dw)
  w   <- res$w
  d$weight[idx] <- w

  mt <- margin_table(dw, w, wv)
  mt$diff_pp <- round(mt$weighted_pct - mt$target_pct, 2)
  diag_rows[[length(diag_rows) + 1]] <- mt

  cust <- dw$mosaik_cust == 1
  deff <- kish(w)
  summary_rows[[length(summary_rows) + 1]] <- data.frame(
    wave = wv, n = length(w), rounds = res$rounds,
    min_w = round(min(w), 3), max_w = round(max(w), 3), cv = round(sd(w) / mean(w), 3),
    at_lower = sum(w <= TRIM_LOWER + 0.01), at_upper = sum(w >= TRIM_UPPER - 0.01),   # at or near a bound
    deff = round(deff, 3), efficiency_pct = round(100 / deff, 1), n_eff = round(length(w) / deff),
    cust_n = sum(cust), cust_n_eff = round(sum(cust) / kish(w[cust])),
    max_abs_diff_pp = max(abs(mt$diff_pp)))
}
diag_tab <- do.call(rbind, diag_rows)
summ     <- do.call(rbind, summary_rows)

# --- checks (stop the pipeline if weighting failed) ------------------------------------
problems <- c()
if (any(is.na(d$weight))) problems <- c(problems, "some respondents have no weight")
if (any(d$weight < TRIM_LOWER - 1e-6 | d$weight > TRIM_UPPER + 1e-6))
  problems <- c(problems, sprintf("weights outside [%.1f, %.1f]", TRIM_LOWER, TRIM_UPPER))
bad <- diag_tab[abs(diag_tab$diff_pp) > MAX_DIFF_PP, ]
if (nrow(bad)) problems <- c(problems, sprintf("%d margin cells more than %.1f pt from target",
                                               nrow(bad), MAX_DIFF_PP))
wmeans <- tapply(d$weight, d$wave, mean)
if (any(abs(wmeans - 1) > 1e-6)) problems <- c(problems, "mean weight is not 1 in every wave")

# --- write ---------------------------------------------------------------------------------
write.csv(d[, c("resp_id", "wave", "weight")], OUT_WEIGHTS, row.names = FALSE)
keep <- setdiff(names(d), paste0(MARGINS, "_f"))
write.csv(d[, keep], OUT_WEIGHTED, row.names = FALSE, na = "", fileEncoding = "UTF-8")
write.csv(diag_tab, OUT_DIAG, row.names = FALSE)

md_table <- function(df) {
  c(paste0("| ", paste(names(df), collapse = " | "), " |"),
    paste0("| ", paste(rep("---", ncol(df)), collapse = " | "), " |"),
    apply(df, 1, function(r) paste0("| ", paste(trimws(format(r)), collapse = " | "), " |")))
}
report <- c(
  "# Weighting report: Mosaik Bank CX Tracker 2026", "",
  sprintf("_Generated %s by `R/04_weighting.R`. All survey data is simulated; the targets are official._", Sys.Date()), "",
  sprintf("Each wave is raked separately to %s, trimmed to [%.1f, %.1f] and re-raked until stable.",
          paste(MARGINS, collapse = " x "), TRIM_LOWER, TRIM_UPPER),
  sprintf("Targets: %s, population aged 18-69 on %s (table status %s, retrieved %s).",
          targets$source[1], targets$reference_date[1], targets$source_status[1], targets$retrieved[1]), "",
  "## Weight summary by wave", "",
  "Kish design effect = n * sum(w^2) / sum(w)^2; efficiency = 1 / deff; n_eff = n / deff. ",
  "`cust_n_eff` is the effective base for Mosaik-customer results such as NPS.", "",
  md_table(summ), "",
  "## Margins after weighting (percent)", "",
  md_table(diag_tab), "",
  if (length(problems)) c("## Problems", "", paste0("- ", problems)) else "All checks passed."
)
writeLines(report, OUT_REPORT, useBytes = TRUE)

# --- console summary ----------------------------------------------------------------------
cat("\nWeight summary by wave\n")
print(summ[, c("wave", "n", "min_w", "max_w", "deff", "efficiency_pct", "n_eff",
               "cust_n", "cust_n_eff", "at_upper", "max_abs_diff_pp")], row.names = FALSE)
cat(sprintf("\nWrote %s, %s, %s, %s\n", OUT_WEIGHTS, OUT_WEIGHTED, OUT_DIAG, OUT_REPORT))
if (length(problems)) {
  message("WEIGHTING FAILED: ", paste(problems, collapse = "; "))
  quit(status = 1)
}
cat("All weighting checks passed.\n")
