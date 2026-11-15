#!/usr/bin/env Rscript
# =============================================================================
# 05_tables_sigtests.R - Mosaik Bank CX Tracker 2026: banner tables and NPS trend
# -----------------------------------------------------------------------------
# Input   data/clean/tracker_weighted.csv   (step 04: clean file + weight)
# Output  outputs/tables/banners.xlsx        tab book: one sheet per banner,
#                                            all tables stacked, letters under values
#         outputs/tables/banners_long.csv    every cell in long format (for charts
#                                            and the Python cross-check, step 07)
#         outputs/tables/nps_by_wave.csv     NPS by wave, design-based SE and 95% CI,
#                                            change vs previous wave and vs W1
#         outputs/tables/nps_by_wave_group.csv  the same by main/secondary and age band
# Rules   weighted estimates; significance = column-letter tests within each
#         banner group at p < 0.05 on effective bases (R/utils_sig_letters.R);
#         unweighted base < 50 flagged (*), < 30 suppressed (not shown, not tested)
# Run     Rscript R/05_tables_sigtests.R    (from the repository root)
# Needs   install.packages(c("survey", "openxlsx"))
# =============================================================================

suppressPackageStartupMessages({
  library(survey)
  library(openxlsx)
})
source("R/utils_sig_letters.R")

IN_CSV     <- "data/clean/tracker_weighted.csv"
OUT_DIR    <- "outputs/tables"
ALPHA      <- 0.05
FLAG_BASE  <- 50      # unweighted base below this: shown with *
MIN_BASE   <- 30      # unweighted base below this: suppressed
Z95        <- qnorm(0.975)

if (!file.exists(IN_CSV)) stop("Missing ", IN_CSV, ": run R/04_weighting.R first")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
d <- read.csv(IN_CSV, na.strings = "", stringsAsFactors = FALSE, fileEncoding = "UTF-8")
stopifnot("weight" %in% names(d), !anyNA(d$weight))
LATEST <- max(d$wave)
WAVE_LABEL <- c("1" = "W1 Mar 26", "2" = "W2 Jun 26", "3" = "W3 Sep 26", "4" = "W4 Dec 26")
cat(sprintf("Read %d weighted respondents, waves %s\n", nrow(d), paste(sort(unique(d$wave)), collapse = ", ")))

# =============================================================================
# 1. Banners: groups of columns. Letters run across the whole banner; tests run
#    only within a group (never Total vs its own subgroups).
# =============================================================================
col_def <- function(label, f) list(label = label, f = f)

BANNERS <- list(
  trend = list(
    title = "Trend: all waves",
    groups = list(
      list(name = "Total", cols = list(col_def("Total W1-W4", function(d) rep(TRUE, nrow(d))))),
      list(name = "Wave", cols = lapply(sort(unique(d$wave)), function(w)
        col_def(WAVE_LABEL[[as.character(w)]], function(d) d$wave == w)))
    )),
  latest = list(
    title = sprintf("Profile: %s", WAVE_LABEL[[as.character(LATEST)]]),
    filter = function(d) d$wave == LATEST,
    groups = list(
      list(name = "Total", cols = list(col_def(sprintf("Total %s", WAVE_LABEL[[as.character(LATEST)]]),
                                               function(d) rep(TRUE, nrow(d))))),
      list(name = "Age", cols = list(
        col_def("18-29", function(d) d$age_band == 1), col_def("30-39", function(d) d$age_band == 2),
        col_def("40-49", function(d) d$age_band == 3), col_def("50-59", function(d) d$age_band == 4),
        col_def("60-69", function(d) d$age_band == 5))),
      list(name = "Gender", cols = list(
        col_def("Male", function(d) d$gender %in% 1), col_def("Female", function(d) d$gender %in% 2))),
      list(name = "Mosaik relationship", cols = list(
        col_def("Main bank", function(d) d$mosaik_cust == 1 & d$mosaik_main == 1),
        col_def("Secondary", function(d) d$mosaik_cust == 1 & d$mosaik_main == 0),
        col_def("Not a customer", function(d) d$mosaik_cust == 0))),
      list(name = "Region", cols = list(
        col_def("North", function(d) d$region == 1), col_def("East", function(d) d$region == 2),
        col_def("South", function(d) d$region == 3), col_def("West", function(d) d$region == 4)))
    ))
)

# =============================================================================
# 2. Tables: a base (who is in it) and rows. A row returns 0/1 (proportion) or a
#    number (mean) per respondent; NA = not in this row's base (e.g. don't know).
# =============================================================================
prop_row <- function(label, f) list(label = label, f = f, type = "prop")
mean_row <- function(label, f) list(label = label, f = f, type = "mean")
flag <- function(x) ifelse(is.na(x), NA, as.numeric(x))     # logical -> 0/1, keeps NA
t2b  <- function(x) ifelse(is.na(x), NA, as.numeric(x >= 4))
CUST <- function(d) d$mosaik_cust == 1
ALL  <- function(d) rep(TRUE, nrow(d))

BRANDS <- c("Mosaik Bank", "Volta", "Atlas Direkt", "Sparhaus", "Pebble")
C3_LABELS <- c("App ease", "Transfer speed", "Fees / value for money", "Customer service",
               "Security & trust", "Card features", "Savings interest", "Investing features (from W3)")
C2_LABELS <- c("Fees / value for money", "App", "Customer service", "Security / trust", "Transfers",
               "Card", "Interest / investing", "Other / general")
E1_LABELS <- c("Would switch for a better interest rate", "Prefer a bank with branches",
               "Comfortable banking only by app", "Sustainability matters in bank choice",
               "Invest money regularly", "Security matters more than new features")

TABLES <- list(
  list(id = "T01", title = "C1. Net Promoter Score", base = CUST, base_label = "Mosaik customers",
       rows = list(
         mean_row("NPS (% promoters - % detractors)", function(d) d$nps_score),
         prop_row("Promoters (9-10)", function(d) flag(d$nps_group == 3)),
         prop_row("Passives (7-8)", function(d) flag(d$nps_group == 2)),
         prop_row("Detractors (0-6)", function(d) flag(d$nps_group == 1))),
       note = "NPS tested as a mean of -100 / 0 / +100."),
  list(id = "T02", title = "C3. Satisfaction with Mosaik: top-2-box (satisfied + very satisfied)",
       base = CUST, base_label = "Mosaik customers; don't know excluded per row",
       rows = lapply(1:8, function(i) prop_row(C3_LABELS[i], function(d) d[[paste0("c3_t2b_r", i)]]))),
  list(id = "T03", title = "C4 / C5. Overall satisfaction and switching intention",
       base = CUST, base_label = "Mosaik customers",
       rows = list(
         prop_row("Overall satisfaction: top-2-box", function(d) t2b(d$c4_overall)),
         prop_row("Likely to switch main account (12 months): top-2-box", function(d) t2b(d$c5_switch)))),
  list(id = "T04", title = "C2. Main reason for NPS score (coded open ends, multiple codes possible)",
       base = CUST, base_label = "Mosaik customers",
       rows = lapply(1:8, function(i) prop_row(C2_LABELS[i], function(d) d[[paste0("c2_code_", i)]]))),
  list(id = "T05", title = "D1. Mosaik app use", base = CUST, base_label = "Mosaik customers",
       rows = list(
         prop_row("Daily", function(d) flag(d$d1_app_freq == 1)),
         prop_row("Weekly", function(d) flag(d$d1_app_freq == 2)),
         prop_row("Monthly", function(d) flag(d$d1_app_freq == 3)),
         prop_row("Less than monthly", function(d) flag(d$d1_app_freq == 4)))),
  list(id = "T06", title = "D2 / D3. Customer service contact and resolution",
       base = CUST, base_label = "Mosaik customers (resolution: those who contacted service)",
       rows = list(
         prop_row("Contacted service in the last 3 months", function(d) d$d2_contact),
         prop_row("Resolved at first contact (of those who contacted)",
                  function(d) ifelse(d$d2_contact %in% 1, flag(d$d3_resolved == 1), NA)),
         prop_row("Not resolved (of those who contacted)",
                  function(d) ifelse(d$d2_contact %in% 1, flag(d$d3_resolved == 3), NA)))),
  list(id = "T07", title = "Brand funnel: Mosaik Bank (% of all respondents)", base = ALL,
       base_label = "All respondents",
       rows = list(
         prop_row("Aware", function(d) d$b1_r1),
         prop_row("Would consider", function(d) flag(d$b2_r1 %in% 1)),
         prop_row("Uses (has an account)", function(d) d$a1_r1),
         prop_row("Main bank", function(d) flag(d$main_bank == 1)))),
  list(id = "T08", title = "B1. Brand awareness (aided)", base = ALL, base_label = "All respondents",
       rows = lapply(1:5, function(i) prop_row(BRANDS[i], function(d) d[[paste0("b1_r", i)]]))),
  list(id = "T09", title = "B2. Would consider (% of all respondents)", base = ALL,
       base_label = "All respondents",
       rows = lapply(1:5, function(i) prop_row(BRANDS[i], function(d) flag(d[[paste0("b2_r", i)]] %in% 1)))),
  list(id = "T10", title = "A2. Main bank", base = ALL, base_label = "All respondents",
       rows = c(lapply(1:5, function(i) prop_row(BRANDS[i], function(d) flag(d$main_bank == i))),
                list(prop_row("Other bank", function(d) flag(d$main_bank == 6))))),
  list(id = "T11", title = "E1. Banking attitudes: top-2-box (agree + strongly agree)", base = ALL,
       base_label = "All respondents",
       rows = lapply(1:6, function(i) prop_row(E1_LABELS[i], function(d) t2b(d[[paste0("e1_r", i)]]))))
)

# =============================================================================
# 3. Compute every cell
# =============================================================================
compute_banner <- function(bname, banner) {
  dd <- if (is.null(banner$filter)) d else d[banner$filter(d), ]
  cols <- do.call(c, lapply(banner$groups, function(g)
    lapply(g$cols, function(cl) c(cl, group = g$name))))
  letters_ <- column_letters(length(cols))
  col_masks <- lapply(cols, function(cl) { m <- cl$f(dd); m[is.na(m)] <- FALSE; m })

  out <- list()
  for (tb in TABLES) {
    in_base <- tb$base(dd); in_base[is.na(in_base)] <- FALSE
    for (k in seq_along(tb$rows)) {
      rw <- tb$rows[[k]]
      x  <- rw$f(dd)
      cells <- lapply(seq_along(cols), function(j) {
        m  <- in_base & col_masks[[j]] & !is.na(x)
        n  <- sum(m)
        ne <- n_effective(dd$weight[m])
        if (rw$type == "prop") { est <- weighted_prop(x[m], dd$weight[m]); sdv <- NA_real_ }
        else { ms <- weighted_mean_sd(x[m], dd$weight[m]); est <- ms[["mean"]]; sdv <- ms[["sd"]] }
        suppressed <- n < MIN_BASE
        data.frame(banner = bname, table_id = tb$id, table_title = tb$title, base_label = tb$base_label,
                   row_order = k, row_label = rw$label, type = rw$type,
                   group = cols[[j]]$group, column = cols[[j]]$label, letter = letters_[j],
                   col_order = j, n_unweighted = n, n_eff = round(ne, 1),
                   table_base_unweighted = sum(in_base & col_masks[[j]]),
                   estimate = if (suppressed) NA_real_ else est, sd = if (suppressed) NA_real_ else sdv,
                   small_base = n < FLAG_BASE & !suppressed, suppressed = suppressed,
                   stringsAsFactors = FALSE)
      })
      cells <- do.call(rbind, cells)
      cells$sig <- ""
      for (g in unique(cells$group)) {
        idx <- which(cells$group == g)
        if (length(idx) < 2) next
        cells$sig[idx] <- sig_letters(cells$estimate[idx], cells$n_eff[idx], cells$letter[idx],
                                      sd = cells$sd[idx], type = rw$type, alpha = ALPHA, min_base = 1)
      }
      cells$sig[is.na(cells$sig)] <- ""
      out[[length(out) + 1]] <- cells
    }
  }
  list(cells = do.call(rbind, out), cols = cols, letters = letters_)
}

results <- lapply(names(BANNERS), function(b) compute_banner(b, BANNERS[[b]]))
names(results) <- names(BANNERS)
long <- do.call(rbind, lapply(results, `[[`, "cells"))
write.csv(long, file.path(OUT_DIR, "banners_long.csv"), row.names = FALSE, na = "")

# =============================================================================
# 4. Excel tab book
# =============================================================================
wb <- createWorkbook()
st_title  <- createStyle(fontSize = 14, textDecoration = "bold")
st_note   <- createStyle(fontSize = 9, fontColour = "#595959", wrapText = FALSE)
st_group  <- createStyle(textDecoration = "bold", halign = "center", fgFill = "#1F3864",
                         fontColour = "#FFFFFF", border = "TopBottomLeftRight", borderColour = "#FFFFFF")
st_col    <- createStyle(textDecoration = "bold", halign = "center", wrapText = TRUE, fgFill = "#D9E1F2",
                         border = "TopBottomLeftRight", borderColour = "#FFFFFF")
st_letter <- createStyle(halign = "center", fontColour = "#1F3864", textDecoration = "bold", fgFill = "#D9E1F2")
st_ttitle <- createStyle(textDecoration = "bold", fgFill = "#F2F2F2")
st_base   <- createStyle(fontColour = "#595959", fontSize = 9, numFmt = "#,##0", halign = "center")
st_pct    <- createStyle(numFmt = "0%", halign = "center")
st_nps    <- createStyle(numFmt = "+0;-0;0", halign = "center", textDecoration = "bold")
st_sig    <- createStyle(fontColour = "#C00000", fontSize = 9, halign = "center", textDecoration = "bold")
st_small  <- createStyle(fontColour = "#7F7F7F", textDecoration = "italic")
st_label  <- createStyle(indent = 1)

addWorksheet(wb, "Contents")
contents <- data.frame(Table = sapply(TABLES, `[[`, "id"), Title = sapply(TABLES, `[[`, "title"),
                       Base = sapply(TABLES, `[[`, "base_label"))
writeData(wb, "Contents", "Mosaik Bank CX Tracker 2026 - banner tables", startRow = 1)
addStyle(wb, "Contents", st_title, rows = 1, cols = 1)
writeData(wb, "Contents", "All survey data is simulated. Sheets: one per banner; every table appears on each.",
          startRow = 2)
writeData(wb, "Contents", contents, startRow = 4, headerStyle = st_col)
setColWidths(wb, "Contents", cols = 1:3, widths = c(8, 70, 45))

for (bname in names(BANNERS)) {
  res <- results[[bname]]; banner <- BANNERS[[bname]]
  sh <- substr(bname, 1, 31); addWorksheet(wb, sh)
  ncol_ <- length(res$cols)
  writeData(wb, sh, sprintf("Banner: %s", banner$title), startRow = 1); addStyle(wb, sh, st_title, 1, 1)
  writeData(wb, sh, paste0("Weighted percentages. Letters in red: significantly higher than that column ",
                           "within the same group (p < 0.05, z-test on effective bases, no multiple-comparison ",
                           "correction). * base < ", FLAG_BASE, "; - base < ", MIN_BASE, " (suppressed)."),
            startRow = 2)
  addStyle(wb, sh, st_note, 2, 1)

  # header: group names (merged), column labels, letters
  r0 <- 4; j <- 2
  for (g in banner$groups) {
    k <- length(g$cols)
    writeData(wb, sh, g$name, startRow = r0, startCol = j)
    if (k > 1) mergeCells(wb, sh, cols = j:(j + k - 1), rows = r0)
    addStyle(wb, sh, st_group, rows = r0, cols = j:(j + k - 1), gridExpand = TRUE)
    j <- j + k
  }
  writeData(wb, sh, t(sapply(res$cols, `[[`, "label")), startRow = r0 + 1, startCol = 2, colNames = FALSE)
  writeData(wb, sh, t(res$letters), startRow = r0 + 2, startCol = 2, colNames = FALSE)
  addStyle(wb, sh, st_col, rows = r0 + 1, cols = 2:(ncol_ + 1), gridExpand = TRUE)
  addStyle(wb, sh, st_letter, rows = r0 + 2, cols = 2:(ncol_ + 1), gridExpand = TRUE)
  freezePane(wb, sh, firstActiveRow = r0 + 3, firstActiveCol = 2)

  r <- r0 + 4
  cells <- res$cells
  for (tb in TABLES) {
    tc <- cells[cells$table_id == tb$id, ]
    writeData(wb, sh, sprintf("%s  %s", tb$id, tb$title), startRow = r, startCol = 1)
    addStyle(wb, sh, st_ttitle, rows = r, cols = 1:(ncol_ + 1), gridExpand = TRUE)
    writeData(wb, sh, paste("Base:", tb$base_label, if (!is.null(tb$note)) paste("-", tb$note) else ""),
              startRow = r + 1, startCol = 1)
    addStyle(wb, sh, st_note, rows = r + 1, cols = 1)
    # bases from the first row of the table
    first <- tc[tc$row_order == 1, ][order(tc$col_order[tc$row_order == 1]), ]
    writeData(wb, sh, "Unweighted base", startRow = r + 2, startCol = 1)
    writeData(wb, sh, t(ifelse(first$table_base_unweighted < FLAG_BASE,
                               paste0(format(first$table_base_unweighted, big.mark = ","), "*"),
                               format(first$table_base_unweighted, big.mark = ","))),
              startRow = r + 2, startCol = 2, colNames = FALSE)
    writeData(wb, sh, "Effective base", startRow = r + 3, startCol = 1)
    writeData(wb, sh, t(round(first$n_eff)), startRow = r + 3, startCol = 2, colNames = FALSE)
    addStyle(wb, sh, st_base, rows = (r + 2):(r + 3), cols = 1:(ncol_ + 1), gridExpand = TRUE)
    r <- r + 4
    for (k in sort(unique(tc$row_order))) {
      rc <- tc[tc$row_order == k, ]; rc <- rc[order(rc$col_order), ]
      writeData(wb, sh, rc$row_label[1], startRow = r, startCol = 1)
      addStyle(wb, sh, st_label, rows = r, cols = 1)
      vals <- rc$estimate
      for (jj in seq_along(vals)) {
        if (rc$suppressed[jj]) writeData(wb, sh, "-", startRow = r, startCol = jj + 1)
        else writeData(wb, sh, vals[jj], startRow = r, startCol = jj + 1)
      }
      addStyle(wb, sh, if (rc$type[1] == "mean") st_nps else st_pct, rows = r, cols = 2:(ncol_ + 1),
               gridExpand = TRUE)
      small <- which(rc$small_base)
      if (length(small)) addStyle(wb, sh, st_small, rows = r, cols = small + 1, gridExpand = TRUE, stack = TRUE)
      writeData(wb, sh, t(rc$sig), startRow = r + 1, startCol = 2, colNames = FALSE)
      addStyle(wb, sh, st_sig, rows = r + 1, cols = 2:(ncol_ + 1), gridExpand = TRUE)
      r <- r + 2
    }
    r <- r + 1
  }
  setColWidths(wb, sh, cols = 1, widths = 52)
  setColWidths(wb, sh, cols = 2:(ncol_ + 1), widths = 11)
}

addWorksheet(wb, "Notes")
writeData(wb, "Notes", data.frame(Note = c(
  "Source: Mosaik Bank CX Tracker 2026 (simulated data; brands are fictional).",
  "Weights: raked per wave to Destatis 12411-0013 (31.12.2025), age band x gender x region, trimmed 0.3-3.0.",
  "Percentages are weighted. Bases shown are unweighted; effective base = (sum w)^2 / sum(w^2).",
  "Significance: pooled two-proportion z-test (means: unpooled z-test) on effective bases, p < 0.05, two-sided.",
  "Columns are only compared within their own group; Total is never tested against its subgroups.",
  sprintf("Bases below %d are flagged with * and shown in grey italics; below %d values are suppressed.", FLAG_BASE, MIN_BASE),
  "Row bases can be smaller than the table base when 'don't know' is excluded (C3) or a follow-up applies (D3).")),
  startRow = 1)
setColWidths(wb, "Notes", cols = 1, widths = 120)
saveWorkbook(wb, file.path(OUT_DIR, "banners.xlsx"), overwrite = TRUE)

# =============================================================================
# 5. NPS trend with design-based standard errors
#    Waves are independent samples, so the SE of a change is sqrt(se1^2 + se2^2).
# =============================================================================
des_all <- svydesign(ids = ~1, weights = ~weight, data = d)

nps_trend <- function(group_var = NULL, group_levels = NULL, group_labels = NULL) {
  specs <- if (is.null(group_var)) list(list(label = "Total", keep = rep(TRUE, nrow(d))))
           else lapply(seq_along(group_levels), function(i)
             list(label = group_labels[i], keep = !is.na(d[[group_var]]) & d[[group_var]] == group_levels[i]))
  out <- list()
  for (sp in specs) {
    rows <- list()
    for (w in sort(unique(d$wave))) {
      dom <- d$mosaik_cust == 1 & d$wave == w & sp$keep
      n <- sum(dom)
      sub <- subset(des_all, dom)
      if (n >= MIN_BASE) {
        m  <- svymean(~nps_score, sub, na.rm = TRUE)
        est <- as.numeric(coef(m)); se <- as.numeric(SE(m))
        prom <- weighted_prop(as.numeric(d$nps_group[dom] == 3), d$weight[dom])
        detr <- weighted_prop(as.numeric(d$nps_group[dom] == 1), d$weight[dom])
      } else { est <- se <- prom <- detr <- NA_real_ }
      rows[[length(rows) + 1]] <- data.frame(
        group = if (is.null(group_var)) "Total" else group_var, subgroup = sp$label,
        wave = w, wave_label = WAVE_LABEL[[as.character(w)]],
        n_unweighted = n, n_eff = round(n_effective(d$weight[dom]), 1),
        nps = est, se = se, ci_low = est - Z95 * se, ci_high = est + Z95 * se,
        promoters_pct = 100 * prom, detractors_pct = 100 * detr,
        small_base = n < FLAG_BASE & n >= MIN_BASE, suppressed = n < MIN_BASE)
    }
    tr <- do.call(rbind, rows)
    diff_test <- function(i, j) {
      dlt <- tr$nps[i] - tr$nps[j]; sed <- sqrt(tr$se[i]^2 + tr$se[j]^2)
      c(dlt, 2 * pnorm(-abs(dlt / sed)))
    }
    prev <- t(sapply(seq_len(nrow(tr)), function(i) if (i == 1) c(NA, NA) else diff_test(i, i - 1)))
    first <- t(sapply(seq_len(nrow(tr)), function(i) if (i == 1) c(NA, NA) else diff_test(i, 1)))
    tr$change_vs_prev <- prev[, 1]; tr$p_vs_prev <- prev[, 2]
    tr$sig_vs_prev <- !is.na(tr$p_vs_prev) & tr$p_vs_prev < ALPHA
    tr$change_vs_w1 <- first[, 1]; tr$p_vs_w1 <- first[, 2]
    tr$sig_vs_w1 <- !is.na(tr$p_vs_w1) & tr$p_vs_w1 < ALPHA
    out[[length(out) + 1]] <- tr
  }
  res <- do.call(rbind, out)
  num <- c("nps", "se", "ci_low", "ci_high", "promoters_pct", "detractors_pct",
           "change_vs_prev", "change_vs_w1")
  res[num] <- lapply(res[num], round, 2)
  res$p_vs_prev <- signif(res$p_vs_prev, 4); res$p_vs_w1 <- signif(res$p_vs_w1, 4)
  res
}

nps_total <- nps_trend()
nps_groups <- rbind(
  nps_trend("mosaik_main", c(1, 0), c("Main bank", "Secondary")),
  nps_trend("age_band", 1:5, c("18-29", "30-39", "40-49", "50-59", "60-69")))
write.csv(nps_total, file.path(OUT_DIR, "nps_by_wave.csv"), row.names = FALSE, na = "")
write.csv(nps_groups, file.path(OUT_DIR, "nps_by_wave_group.csv"), row.names = FALSE, na = "")

# =============================================================================
# 6. Console summary
# =============================================================================
cat("\nNPS by wave (Mosaik customers, weighted, 95% CI, design-based SE)\n")
print(nps_total[, c("wave_label", "n_unweighted", "n_eff", "nps", "se", "ci_low", "ci_high",
                    "change_vs_prev", "p_vs_prev", "sig_vs_prev")], row.names = FALSE)
n_sig <- sum(long$sig != "")
cat(sprintf("\nBanner cells: %d (%d with significance letters, %d flagged small, %d suppressed)\n",
            nrow(long), n_sig, sum(long$small_base), sum(long$suppressed)))
cat(sprintf("Wrote %s/banners.xlsx, banners_long.csv, nps_by_wave.csv, nps_by_wave_group.csv\n", OUT_DIR))
