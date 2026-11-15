# Data quality report: Mosaik Bank CX Tracker 2026

_Generated 2026-11-15 by `src/02_quality_checks.py`. All data is simulated._

**569 of 12,000 completes (4.7%) are removed**, leaving 11,431 for analysis. A respondent is removed for any hard flag (attention check, duplicate, inconsistent answers) or for 2+ soft flags (speeder, straightliner, gibberish). One soft flag alone is kept and logged.

## Removals by wave

| Wave | Completes | Removed | Removed % | Kept | Mosaik customers kept |
| --- | --- | --- | --- | --- | --- |
| W1 | 3000 | 136 | 4.5% | 2864 | 636 |
| W2 | 3000 | 142 | 4.7% | 2858 | 616 |
| W3 | 3000 | 145 | 4.8% | 2855 | 643 |
| W4 | 3000 | 146 | 4.9% | 2854 | 613 |

## Flags by wave

A respondent can carry several flags, so the columns do not add up to the removals.

| Flag | W1 | W2 | W3 | W4 | Total | Total % |
| --- | --- | --- | --- | --- | --- | --- |
| Speeder (soft) | 90 | 90 | 90 | 90 | 360 | 3.0% |
| Straightliner (soft) | 75 | 75 | 75 | 75 | 300 | 2.5% |
| Gibberish open end (soft) | 45 | 45 | 45 | 45 | 180 | 1.5% |
| Attention check failed (hard) | 60 | 60 | 60 | 60 | 240 | 2.0% |
| Duplicate panel ID (hard) | 30 | 30 | 30 | 30 | 120 | 1.0% |
| Inconsistent answers (hard) | 30 | 30 | 30 | 30 | 120 | 1.0% |
| Removed | 136 | 142 | 145 | 146 | 569 | 4.7% |

## How soft flags combine

| Soft flags | Respondents | Also a hard flag | Decision |
| --- | --- | --- | --- |
| Gibberish open end | 156 | 0 | kept unless a hard flag |
| Speeder | 228 | 0 | kept unless a hard flag |
| Speeder + Gibberish open end | 24 | 19 | removed |
| Speeder + Straightliner | 108 | 24 | removed |
| Straightliner | 192 | 32 | kept unless a hard flag |

544 respondents with exactly one soft flag are **kept**. They are worth a manual look in a real study.

## Check definitions

| Check | Rule | Type |
| --- | --- | --- |
| Speeder | duration < 40% of the median for the same wave x device x path (customer / non-customer) | soft |
| Straightliner | all valid answers identical in C3 or E1, with 6+ answered (don't know excluded) | soft |
| Gibberish | C2 has < 3 letters, one word repeated, a repeated pattern, a keyboard run of 4+ keys, or no vowels | soft |
| Attention | C3x answered and not 4 | hard |
| Duplicate | same panel ID twice in a wave; the later interview is flagged | hard |
| Inconsistent | breaks any rule below | hard |

### Speed thresholds

| Wave | Device | Path | n | Median (s) | Threshold (s) | Flagged |
| --- | --- | --- | --- | --- | --- | --- |
| W1 | Desktop | customer | 254 | 841 | 336 | 12 |
| W1 | Desktop | non-customer | 999 | 507 | 203 | 24 |
| W1 | Mobile | customer | 422 | 873 | 349 | 23 |
| W1 | Mobile | non-customer | 1119 | 555 | 222 | 28 |
| W1 | Tablet | customer | 60 | 1000 | 400 | 0 |
| W1 | Tablet | non-customer | 146 | 515 | 206 | 3 |
| W2 | Desktop | customer | 248 | 810 | 324 | 10 |
| W2 | Desktop | non-customer | 978 | 514 | 206 | 27 |
| W2 | Mobile | customer | 414 | 902 | 361 | 10 |
| W2 | Mobile | non-customer | 1130 | 558 | 223 | 37 |
| W2 | Tablet | customer | 56 | 812 | 325 | 1 |
| W2 | Tablet | non-customer | 174 | 552 | 221 | 5 |
| W3 | Desktop | customer | 246 | 854 | 342 | 7 |
| W3 | Desktop | non-customer | 977 | 520 | 208 | 26 |
| W3 | Mobile | customer | 434 | 888 | 355 | 13 |
| W3 | Mobile | non-customer | 1118 | 556 | 222 | 41 |
| W3 | Tablet | customer | 61 | 844 | 338 | 2 |
| W3 | Tablet | non-customer | 164 | 562 | 225 | 1 |
| W4 | Desktop | customer | 237 | 833 | 333 | 8 |
| W4 | Desktop | non-customer | 995 | 522 | 209 | 32 |
| W4 | Mobile | customer | 441 | 897 | 359 | 10 |
| W4 | Mobile | non-customer | 1147 | 532 | 213 | 33 |
| W4 | Tablet | customer | 32 | 882 | 353 | 0 |
| W4 | Tablet | non-customer | 148 | 556 | 223 | 7 |

### Consistency rules

| Rule | Description | Respondents |
| --- | --- | --- |
| age_vs_tenure | More than 5 years with Mosaik, but too young to have opened an account 5+ years ago (accounts from age 18) | 48 |
| main_bank_not_used | Main bank (A2) is not among the banks ticked at A1 | 36 |
| c_section_without_mosaik | Answered the Mosaik experience section (C1) without ticking Mosaik at A1 | 36 |
| d3_without_contact | Rated issue resolution (D3) without having contacted service (D2 = no) | 36 |
| consider_without_awareness | Would consider a brand (B2) the respondent said they had not heard of (B1) | 0 |

### Gibberish open ends

| Reason | Respondents | Examples |
| --- | --- | --- |
| keyboard run | 20 | `qwertz`, `asdf` |
| no vowels | 12 | `nsdkjfh` |
| repeated pattern | 34 | `fgfgfg`, `jjjjjjj` |
| same word repeated | 30 | `good good good`, `ok ok ok ok` |
| too few letters | 84 | `x`, `1234`, `...`, `-` |

## Validation against the simulated truth

_Only possible because the data is simulated: `data/_truth/` records which respondents were planted as bad. It was read after all flags were final and changed none of them._

| Flag | Planted | Flagged | Caught | Missed | False alarms | Recall | Precision | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Speeder | 360 | 360 | 360 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| Straightliner | 300 | 300 | 300 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| Attention check failed | 240 | 240 | 240 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| Duplicate panel ID | 120 | 120 | 120 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| Gibberish open end | 180 | 180 | 180 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| Inconsistent answers | 120 | 120 | 120 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| Removed | 569 | 569 | 569 | 0 | 0 | 1.000 | 1.000 | 1.000 |

**Removal decision by wave**

| Wave | Should remove | Removed | Recall | Precision |
| --- | --- | --- | --- | --- |
| W1 | 136 | 136 | 1.000 | 1.000 |
| W2 | 142 | 142 | 1.000 | 1.000 |
| W3 | 145 | 145 | 1.000 | 1.000 |
| W4 | 146 | 146 | 1.000 | 1.000 |

### Does the rule remove the right people?

The scores above compare the flags with a truth that applies the same removal rule, so they show the rule is implemented correctly, not that it is a good rule. This table shows what happened to each type of planted respondent.

| Planted profile | Path | n | Removed | Kept | Removed % |
| --- | --- | --- | --- | --- | --- |
| attention_fail | customer | 165 | 165 | 0 | 100.0% |
| duplicate | non-customer | 93 | 93 | 0 | 100.0% |
| duplicate | customer | 27 | 27 | 0 | 100.0% |
| fast_reader | non-customer | 101 | 0 | 101 | 0.0% |
| fast_reader | customer | 43 | 0 | 43 | 0.0% |
| genuine | non-customer | 8594 | 0 | 8594 | 0.0% |
| genuine | customer | 2293 | 0 | 2293 | 0.0% |
| gibberish | customer | 156 | 0 | 156 | 0.0% |
| inconsistent | customer | 120 | 120 | 0 | 100.0% |
| speeder_random | non-customer | 163 | 79 | 84 | 48.5% |
| speeder_random | customer | 53 | 53 | 0 | 100.0% |
| straightliner | non-customer | 144 | 0 | 144 | 0.0% |
| straightliner | customer | 48 | 32 | 16 | 66.7% |

**Known gap:** 84 random responders who rushed through the survey are kept. They are non-customers, so they never saw the open end or the attention check, and speed is their only flag. Speed alone cannot separate them from fast honest readers (both finish in 15-33% of the median). A next check could test answer coherence, for example E1 answers that contradict each other.


No missed or false flags.

