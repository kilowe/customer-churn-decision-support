# Customer Churn Decision Support System (Telco Case)

This repository contains the full data pipeline used in the Master's thesis *"Design and Evaluation of a Machine Learning–Based Decision Support System for Customer Churn"* (Clément Kanku Mukonkole, IU International University, Master of Artificial Intelligence). It compares two logistic regression configurations — a **baseline** model and a **class-weighted ("balanced")** model — under three cost-sensitive threshold scenarios, and turns the result into a three-tier customer risk segmentation with recommended retention actions.

The goal of this repository is **reproducibility**: anyone with Python and the dataset should be able to run the pipeline end to end and obtain the exact same numbers reported in the thesis (chapters 3, 4, and 5).

## What the models actually do

- **Dataset**: IBM Telco Customer Churn (7,043 customers, 21 raw columns, 23 encoded variables after preprocessing).
- **Split**: 64% train / 16% validation / 20% test, stratified on the `Churn` variable, `random_state=42`.
- **Two models trained on the same training set**:
  - `baseline`: `LogisticRegression(class_weight=None)`
  - `balanced`: `LogisticRegression(class_weight='balanced')`
  - Both use `solver='lbfgs'`, `max_iter=3000`.
- **Threshold optimization** (applied after training, on validation-set probabilities): a grid of 91 thresholds from 0.05 to 0.95 (step 0.01) is scanned for each model, under two selection criteria:
  - **max F1**: threshold that maximizes the F1-score.
  - **cost-based**: threshold that minimizes total cost `CFN×FN + CFP×FP`, subject to `recall ≥ 0.70`, under three cost scenarios: `low_incentive` (CFN=300, CFP=30), `medium_incentive` (CFN=300, CFP=60), `high_incentive` (CFN=300, CFP=120).
- **Final evaluation**: the selected thresholds are applied once to the held-out test set to compute classification metrics and total business cost per scenario.
- **Decision support layer**: each test-set customer is assigned a churn probability, a risk tier (`HIGH` / `MEDIUM` / `LOW`) using the balanced model's primary threshold (t=0.62), and a recommended retention action.

## Repository structure

```
data/                          raw and preprocessed data
src/                           pipeline scripts (see run order below)
outputs/                       CSV outputs produced by each script
outputs/reproducibility_supplement/   supporting reproducibility material
docs/                          business context, decision logic, governance notes
requirements.txt               exact library versions
```

## Environment

The table below lists everything that needs to be installed. You do not need to click each link and install packages one by one — that is not how it works in practice. The links are only there as a reference, in case you want to check a version. The actual installation is a single command, explained right after the table, which reads `requirements.txt` and installs every package automatically, at the correct version.

| Parameter | Value | Download / reference |
|---|---|---|
| Python | >=3.9 (tested with 3.9.13) | [python.org/downloads](https://www.python.org/downloads/) |
| pandas | >=2.0 | [pypi.org/project/pandas](https://pypi.org/project/pandas/) |
| numpy | >=1.23 | [pypi.org/project/numpy](https://pypi.org/project/numpy/) |
| scikit-learn | >=1.3 | [pypi.org/project/scikit-learn](https://pypi.org/project/scikit-learn/) |
| matplotlib | >=3.7 | [pypi.org/project/matplotlib](https://pypi.org/project/matplotlib/) |
| random_state | 42 (fixed everywhere: split and both models) | — |

Pandas 2.0 officially supports Python 3.8 through 3.11, so Python 3.9.13 is a valid choice.

### Where to type the install command

The install command is typed in a **terminal**, not in a file or in Python directly. A terminal is a window where you type commands as text instead of clicking icons. It has to be opened **inside the project folder** (the folder that contains `requirements.txt`, `src/`, `data/`), otherwise the command will not find the right files.

**On Windows**: open the project folder in File Explorer, click once in the empty area of the address bar at the top of the window, type `cmd`, and press Enter. A black window opens, already positioned inside that folder.

**On Mac**: open the "Terminal" application (press Cmd+Space, type "Terminal", press Enter), type `cd ` (the letters c and d, followed by one space, nothing after), then drag the project folder from Finder directly into the terminal window — its path will appear automatically — and press Enter.

Once the terminal is open inside the project folder, type the following and press Enter:

```bash
pip install -r requirements.txt
```

This installs every required package automatically, at the correct version. If Python itself is not installed yet, get it first from [python.org/downloads](https://www.python.org/downloads/), then repeat the step above.

## How to reproduce these results

Run the scripts in `src/` in the following order. Each one reads the output of the previous step and writes its own CSV file(s) to `outputs/`.

1. **`preprocessing.py`** — cleans the raw data and encodes the categorical variables (23 final variables). Produces the preprocessed dataset used by every later step.
2. **`validation_pipeline.py`** — performs the 64/16/20 stratified split, trains both models (`baseline` and `balanced`) on the training set, and computes their coefficients and odds ratios.
   Produces: `split_summary.csv`, `validation_probabilities.csv`, `test_probabilities.csv`, `coefficients_odds_ratios.csv`.
3. **`threshold_optimization.py`** — scans the 91-value threshold grid on the validation set for both models, under the max-F1 and cost-based criteria and the three cost scenarios.
   Produces: `threshold_grid_validation.csv`, `selected_thresholds.csv`.
4. **`final_test_evaluation.py`** — applies the thresholds selected in step 3 to the test set and computes final classification metrics and business cost.
   Produces: `Evaluation_Results.csv`, `Business_Cost_Analysis.csv`, `Final_Summary.csv`.
5. **`dss_outputs.py`** — assigns each test-set customer a risk tier and a recommended action, using the balanced model's primary threshold (t=0.62).
   Produces: `DSS_Risk_Output.csv`, `risk_tier_summary.csv`.

### Checking your results against the thesis

Because `random_state=42` is fixed at every stage and no other source of randomness is involved, running the pipeline should reproduce the following numbers exactly — no need to read the full thesis text, just compare these figures:

| File to check | Thesis reference | Key figures to match |
|---|---|---|
| `coefficients_odds_ratios.csv` | Table 6 (chapter 4.5) | Correlation between baseline and balanced coefficients ≈ 0.993 |
| `selected_thresholds.csv` | Table 2 (chapter 4.3) | max-F1 thresholds: baseline t=0.37, balanced t=0.62 |
| `Business_Cost_Analysis.csv` | Table 5 (chapter 4.4) | baseline: 21330 / 35460 / 59100 (low/medium/high); balanced: 21450 / 35520 / 58260 |
| `risk_tier_summary.csv` | Table in chapter 4.6 | churn rate by tier: HIGH 55.34%, MEDIUM 26.67%, LOW 4.42% |

If every number in the table above matches, the pipeline has been fully and independently reproduced.

## Project positioning

This work treats the model as one component of a decision support system, not as an autonomous agent. Predictions are never used directly; they are always passed through an explicit, documented threshold and turned into a recommended action reviewed by a human. The `docs/` folder documents the business context, the governance logic, and the conditions under which the system should not be used.
