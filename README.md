# Risk-Adjusted Hallucination Detection

A calibrated hallucination-risk system for grounded question answering. The project combines uncertainty and evidence-consistency signals, turns them into a risk score, and uses that score to **abstain on answers that are most likely to be unsupported**.

The goal is not only to detect hallucinations, but to answer a more practical question:

> When should a grounded QA system answer, and when should it refuse because the answer is too risky?

## Highlights

- Evaluated the pipeline on **5,300 grounded-QA examples** across **PHANTOM (4,000)** and **WikiQA (1,300)** using Qwen3-8B generations, Qwen3-14B judging, and `k=5` self-consistency sampling.
- On PHANTOM, the four-signal detector reached **0.773 AUROC, 0.645 AUPRC, 75.9% accuracy, and 63.7% F1**.
- Selective prediction improved PHANTOM answer accuracy from **68.3% at full coverage to 78.6% at 81.9% coverage** by abstaining on the riskiest **18.2%** of answers.
- Calibrated risk estimates reached **0.0734 ECE on PHANTOM** and **0.0725 ECE on WikiQA** using dataset-specific calibration methods.
- Bidirectional transfer experiments showed that the learned risk signals **did not generalize cleanly across datasets**, exposing feature-distribution shift, calibration instability, and threshold non-portability rather than hiding a negative result.

[Full results summary](results/FULL_RESULTS_SUMMARY.md) · [Two-dataset guide](results/two_dataset_results_guide.md)

## Why this matters

A hallucination detector is only useful if its score changes system behavior.

This project therefore goes beyond binary classification. It calibrates the detector score, chooses an abstention threshold on validation data, freezes the detector plus calibration plus threshold, and evaluates the resulting answer-or-abstain policy on held-out data.

On PHANTOM, rejecting the highest-risk answers increased selective accuracy by roughly **10.3 percentage points** while retaining about **81.9%** of answers. That demonstrates a concrete quality/coverage tradeoff rather than only reporting a classifier metric.

The cross-dataset experiments are equally important: performance degraded sharply when a frozen detector was moved between PHANTOM and WikiQA. The result suggests that hallucination risk is strongly dependent on the evidence regime and that calibration and abstention thresholds should not be assumed to transfer unchanged.

## System design

The detector combines four signals:

1. **Token uncertainty** — mean token negative log-likelihood.
2. **Self-consistency disagreement** — disagreement across sampled answers.
3. **Semantic entropy** — uncertainty over answer meanings.
4. **Groundedness / evidence consistency** — how strongly the answer is supported by the provided evidence.

These features feed a logistic-regression detector. The downstream pipeline then compares calibration methods, selects an operating threshold on validation data, freezes the resulting bundle, and evaluates it in-domain and under dataset shift.

```text
Grounded question + evidence
          |
          v
      Qwen3-8B
          |
          +--> answer
          +--> k=5 sampled answers
          |
          v
   Feature extraction
   - token uncertainty
   - disagreement
   - semantic entropy
   - groundedness
          |
          v
 Logistic risk detector
          |
          v
      Calibration
          |
          v
  Answer / Abstain policy
```

## Results

### PHANTOM

PHANTOM is the strongest in-domain result.

| Metric | Result |
| --- | ---: |
| AUROC | **0.7730** |
| AUPRC | **0.6451** |
| Accuracy | **0.7591** |
| F1 | **0.6368** |
| Calibrated ECE | **0.0734** |
| Calibrated Brier score | **0.1652** |

At full coverage, selective accuracy is about **0.6832**. At the frozen operating point:

| Selective metric | Result |
| --- | ---: |
| Coverage | **0.8185** |
| Abstention rate | **0.1815** |
| Selective accuracy | **0.7863** |

The learned coefficient directions are also interpretable: disagreement, token uncertainty, and semantic entropy increase predicted risk, while stronger groundedness decreases it.

### WikiQA

WikiQA is a harder evidence regime.

| Metric | Result |
| --- | ---: |
| AUROC | **0.6927** |
| AUPRC | **0.3434** |
| Accuracy | **0.6837** |
| F1 | **0.5231** |
| Calibrated ECE | **0.0725** |
| Calibrated Brier score | **0.1656** |

At the frozen WikiQA operating point, coverage is **0.9643** with **0.7566 selective accuracy**. The abstention gain is smaller than on PHANTOM, so PHANTOM remains the clearer selective-prediction success case.

### Cross-dataset transfer

The transfer experiments intentionally freeze the source-domain detector, calibration, and threshold before testing on the other dataset.

**PHANTOM -> WikiQA**

- AUROC: `0.3996`
- AUPRC: `0.2020`
- ECE: `0.1561`
- selective accuracy at the frozen source threshold: `0.7960` at `0.9238` coverage

**WikiQA -> PHANTOM**

- AUROC: `0.4294`
- AUPRC: `0.2943`
- ECE: `0.1659`
- selective accuracy at the frozen source threshold: `0.7081` at `0.8510` coverage

The ranking and calibration failures in both directions are the main robustness finding: the useful in-domain risk signals do not remain reliable under evidence-regime shift without adaptation.

## Evaluation discipline

The detector pipeline is designed to avoid common evaluation leakage:

- train/validation/test splits are kept separate;
- standardization statistics are fitted only on training data;
- calibration is fitted on validation data rather than test data;
- the abstention threshold is chosen on validation data;
- the detector, calibrator, and threshold are frozen before transfer evaluation.

The repository includes baseline comparisons, calibration comparisons, reliability diagrams, risk-coverage curves, accuracy-coverage curves, frozen bundles, and transfer diagnostics.

## Repository structure

```text
code/
├── data_gen/         # data preparation, splitting, standardization
├── features/         # uncertainty and evidence features
├── llm_generations/  # answer generation + feature extraction
├── detector/         # baselines, detector, calibration, abstention, transfer
└── analysis/         # cross-dataset diagnostics

data/
└── prepared feature JSON / tables / splits

results/
├── calibration/
├── calibration_compare/
├── phantom_to_wikiqa_transfer/
├── wikiqa_to_phantom_transfer/
└── FULL_RESULTS_SUMMARY.md
```

## Reproduce the main pipeline

Create an environment and install the project requirements:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r code\requirements.txt
```

Prepare PHANTOM and WikiQA subsets:

```powershell
python code\data_gen\main.py --dataset phantom --num-rows 4000
python code\data_gen\main.py --dataset wikiqa --split train --num-rows 1300 --retrieve
```

Generate answers and features:

```powershell
python code\llm_generations\main.py `
  --dataset phantom `
  --data-file <path_to_input_jsonl> `
  --output-file <path_to_output_json> `
  --model qwen3:8b `
  --judge-model qwen3:14b `
  --num-rows 4000 `
  --k 5
```

The standalone, calibration, abstention, and transfer commands are documented in the result guides and scripts under `code/detector/`.

## Data and supporting material

- [Shared data folder](https://drive.google.com/drive/folders/1aHTuwsl0TuDfcwWspzaYjyUueDXrDSw5?usp=sharing)
- [Project poster](https://drive.google.com/file/d/1ZShXs6oNtWmFwnbwolwB9QafHuGERSPC/view?usp=sharing)
- [Full results summary](results/FULL_RESULTS_SUMMARY.md)
- [Two-dataset results guide](results/two_dataset_results_guide.md)

## Scope and limitations

This is a research implementation, not a production safety guarantee. PHANTOM is the strongest in-domain result; WikiQA is weaker, and frozen cross-dataset transfer performs poorly in both directions. Those failures are retained because they are part of the main conclusion: **hallucination-risk calibration and abstention policy are dataset-dependent and need explicit validation under distribution shift.**
