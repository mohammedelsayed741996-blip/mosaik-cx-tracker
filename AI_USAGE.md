# How AI was used in this project

I built this project with an AI assistant (Claude, by Anthropic). It drafted most of the code, the questionnaire and the plan. My job was to decide what to build, check every result, and catch the drafts that were wrong. This log records where AI helped, where it was wrong, and how each part was verified, because "the AI wrote it" is not a reason to trust a number.

> **Before publishing:** add your own entries in the last section, and keep only what you actually did and can explain in an interview.

## Where AI drafted the work

| Area | What AI produced | How it was verified |
| --- | --- | --- |
| Study design | Questionnaire, codebook, routing, the simulated "truth" | Read against the client questions; every variable in the codebook is checked by the scripts |
| Simulator (`00`) | Respondents, planted effects, export damage | Expected NPS computed on a large sample; reproducibility checked by file hashes |
| Ingest (`01`) | Parsing of the four export formats | Stacked file compared cell by cell with the undamaged answers |
| Quality checks (`02`) | Six checks and the removal rule | Scored against the planted bad respondents on three seeds |
| Cleaning (`03`) | Recodes, open-end coding | Coding accuracy against the true reasons; routing checks |
| Weighting targets | Reading the Destatis table | Male + female = total in all 2,496 cells; totals matched a second Destatis source |
| R scripts (`04`, `05`) | Raking, banner tables, significance letters | Hand-calculated unit tests; every number recomputed in pandas (`07`) |
| Drivers, charts (`06`, `08`) | Shapley regression, slides | Shapley tested on data with a known answer; every slide checked by eye |

## Where the AI drafts were wrong

These are the mistakes caught during the build. Each one was found by a check, not by reading the code:

1. **Unrealistic effect sizes.** The first simulator made NPS drop far more than planned, and the driver model had an R² of 0.07, which no market researcher would believe. Fixed by recalibrating the noise and the planted effects against a large-sample expectation.
2. **A sample too small for the story.** With about 360 customers per wave, sampling noise could hide the planted drop or invent a decline that was never planted. The sample was enlarged after testing several seeds.
3. **"fee" matched inside "feel".** The open-end coder put "I feel my money is safe" under fees. Found by comparing codes with the true reasons; fixed with a whole-word match.
4. **A circular quality score.** The quality checks scored 100%, but the truth file applied the same removal rule, so the score only proved the rule was implemented, not that it was good. An outcome-by-respondent-type table was added, and it showed random responders being kept.
5. **Headline titles the data did not support.** Planned slide titles said that under-30s recovered fastest and that main-bank customers drove the NPS drop. The data showed neither difference was significant, and one draft picked an age band of 45 people as the headline. Titles are now written from the data and only claim significant differences.
6. **R code that could not be run during drafting.** R was not available in the drafting environment. One bug, a loop variable named `round` that hid R's `round()` function, was found by reading. **The R scripts must be run and checked by me** (see below).

## Where AI did not help

- Judgement calls: the removal rule, the choice of weighting variables, not correcting for multiple comparisons, dropping age bands from the trend slide. AI proposed options; the choice and the reasoning are mine to defend.
- Facts about the world: population figures came from Destatis, read from the source table, not from the model's memory.

## My own checks

_Fill this in as you run the pipeline. Examples:_

- [ ] Ran `Rscript R/tests/test_utils_sig_letters.R`: ___ failures
- [ ] Ran `R/04_weighting.R`: efficiency per wave ___; any problems ___
- [ ] Ran `src/07_crosscheck.py`: R and pandas agree? ___
- [ ] Opened the deck and checked every title against the tables
- [ ] Something I changed myself after review: ___
