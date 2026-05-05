# Qualitative Case Findings

The script applied each frozen detector bundle to an already standardized CSV and wrote case slices for:

- missed unsupported answers that were still kept
- unsupported answers that were correctly abstained on
- supported answers that were safely kept
- supported answers that were unnecessarily abstained on

These files help answer a simple question:

How often did the system keep risky answers, and how often did it safely reject them?

## How to read the files

Each output directory contains:

- `all_cases_with_risk.csv`
- `missed_hallucinations.csv`
- `correctly_abstained_unsupported.csv`
- `safe_supported_kept.csv`
- `unnecessary_abstentions.csv`
- `qualitative_summary.txt`

The files are:

- `missed_hallucinations.csv`
  These are unsupported answers that still looked safe enough to keep.
- `safe_supported_kept.csv`
  These are supported answers with low calibrated risk.
- `unnecessary_abstentions.csv`
  These are supported answers that the model rejected even though they were safe.

## PHANTOM in-domain

Source:

- `results/qualitative/phantom_in_domain/qualitative_summary.txt`

Counts:

- total rows: 303
- kept rows: 248
- abstained rows: 55
- unsupported kept: 53
- unsupported abstained: 43
- supported kept: 195
- supported abstained: 12

Interpretation:

- The PHANTOM detector abstains on a meaningful set of risky cases.
- Among 96 unsupported answers, 43 were rejected. This is about 44.8%.
- Among 207 supported answers, 195 were kept. This is about 94.2%.
- This is a useful balance. The model removes many risky answers while keeping most safe ones.

Findings Summary:

- PHANTOM shows the cleanest qualitative behavior in the project.
- The abstention rule is selective instead of overly broad.
- The main remaining failure mode is the 53 unsupported answers that were still kept.

## WikiQA in-domain

Source:

- `results/qualitative/wikiqa_in_domain/qualitative_summary.txt`

Counts:

- total rows: 196
- kept rows: 189
- abstained rows: 7
- unsupported kept: 46
- unsupported abstained: 1
- supported kept: 143
- supported abstained: 6

Interpretation:

- The WikiQA detector is much less selective at the frozen threshold.
- Among 47 unsupported answers, only 1 was rejected. This is about 2.1%.
- Among 149 supported answers, 143 were kept. This is about 96.0%.
- The system is permissive on WikiQA. It keeps most safe answers, but it also keeps almost all unsupported ones.

Findings Summary:

- WikiQA has weaker abstention behavior than PHANTOM.
- The detector still preserves many supported answers.
- The qualitative weakness comes from missed unsupported cases rather than excessive abstention.

## Transfer 1: PHANTOM to WikiQA

Source:

- `results/qualitative/phantom_to_wikiqa/qualitative_summary.txt`

Counts:

- total rows: 1300
- kept rows: 1200
- abstained rows: 100
- unsupported kept: 244
- unsupported abstained: 26
- supported kept: 956
- supported abstained: 74

Interpretation:

- Transfer from PHANTOM to WikiQA keeps too many unsupported answers.
- Among 270 unsupported answers, only 26 were rejected. This is about 9.6%.
- Among 1030 supported answers, 956 were kept. This is about 92.8%.
- The transferred model still behaves conservatively enough to keep many supported answers, but it loses much of its ability to isolate unsupported ones.

Findings Summary:

- The main transfer problem is missed unsupported answers.
- The frozen PHANTOM threshold is still fairly permissive on WikiQA.
- This matches the earlier transfer metrics showing poor cross-dataset ranking and weaker calibration.

## Transfer 2: WikiQA to PHANTOM

Source:

- `results/qualitative/wikiqa_to_phantom/qualitative_summary.txt`

Counts:

- total rows: 2013
- kept rows: 1713
- abstained rows: 300
- unsupported kept: 500
- unsupported abstained: 102
- supported kept: 1213
- supported abstained: 198

Interpretation:

- Transfer from WikiQA to PHANTOM is also weak.
- Among 602 unsupported answers, 102 were rejected. This is about 16.9%.
- Among 1411 supported answers, 1213 were kept. This is about 86.0%.
- This direction abstains more often than PHANTOM to WikiQA, but it still misses many unsupported answers and rejects more supported ones.

Findings Summary:

- This transfer direction is more cautious than PHANTOM to WikiQA.
- That extra caution still does not recover strong unsupported-answer filtering.
- The result supports the conclusion that the frozen source bundle does not travel cleanly across regimes.

## Cross-setting pattern

The four qualitative summaries support the same broad conclusion as the quantitative reports:

- PHANTOM in-domain is the strongest setting.
- WikiQA in-domain is weaker, especially in abstention strength.
- Both transfer settings miss many unsupported answers.
- Transfer weakens the relationship between calibrated risk and actual unsupported behavior.

There is also a clear asymmetry:

- PHANTOM in-domain rejects a substantial share of unsupported answers while keeping most supported ones.
- WikiQA in-domain keeps almost everything, including most unsupported answers.
- WikiQA to PHANTOM becomes more cautious, but that extra caution also increases supported-answer rejection.

## Practical use

These give four useful case types:

- strong in-domain success cases
- strong in-domain failure cases
- transfer failure cases where unsupported answers looked safe
- transfer over-cautious cases where supported answers were rejected

## Main takeaway

The qualitative exports reinforce the main result of the project:

- in-domain PHANTOM produces the best abstention behavior
- WikiQA is weaker in-domain
- transfer in both directions keeps too many unsupported answers

This supports the broader conclusion from the detector, calibration, and transfer analysis:

the learned risk signal is useful within a regime, but it travels poorly across regimes.
