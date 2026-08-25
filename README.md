# US Personal Auto — UW & Pricing Prototype (Model-Backed)

Replaces the source workbook's hand-coded frequency/severity/bind formulas
with real trained models, wired into the same two-tab Streamlit UI
(Dashboard + Interface) and the same premium/eligibility rules engine.

| Component | Model |
|---|---|
| Claim Frequency | XGBoost, `count:poisson` objective |
| Claim Severity | Gamma GLM (statsmodels) |
| Bind Propensity | LightGBM Classifier |
| Appetite / Hard-Stops / Premium build / Composite score | Deterministic rules engine (unchanged from workbook's documented logic — kept auditable on purpose) |

## Project layout

```
uw_app/
├── data/
│   ├── export_from_xlsx.py         # xlsx -> scored_dataset_full.parquet
│   ├── scored_dataset_full.parquet # 22,000-row training set (generated)
│   └── model_scored_dataset.parquet# full book scored by trained models (generated)
├── train/
│   ├── feature_config.py           # feature lists per model (no-leakage rules)
│   ├── preprocessing.py            # shared imputers/encoders
│   ├── splits.py                   # time-based train/test split
│   ├── train_frequency_model.py
│   ├── train_severity_model.py
│   ├── train_bind_model.py
│   └── evaluate_models.py          # CONSOLE-ONLY performance report
├── models/                         # trained model + preprocessor artifacts (generated)
├── rules/
│   └── rules_engine.py             # hard-stops, appetite score, premium build, composite score
├── scoring/
│   ├── predictors.py               # loads models, exposes predict_*() methods
│   ├── row_mapper.py               # raw row -> SubmissionInputs
│   └── score_book.py               # batch-scores all 22k rows for the Dashboard tab
├── app/
│   └── app.py                      # Streamlit entrypoint (Dashboard + Interface tabs)
└── requirements.txt
```

## Setup

```bash
cd uw_app
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run — step by step

**1. Export the training data from the source workbook (run once):**
```bash
python data/export_from_xlsx.py --source /path/to/US_Personal_Auto_UW_Pricing_Prototype_dh.xlsx
```

**2. Train all 3 models:**
```bash
python train/train_frequency_model.py
python train/train_severity_model.py
python train/train_bind_model.py
```

**3. Check model performance in the console** (lift charts, deviance, AUC, calibration — this is NOT shown in the app, it's a developer/actuarial sanity check):
```bash
python train/evaluate_models.py
```
Review the lift tables — `avg_actual` should rise roughly monotonically across deciles for frequency/severity, and the bind model's calibration table should show predicted ≈ actual per decile. Don't proceed to the app if these look broken.

**4. Score the full historical book** (produces the dataset the Dashboard tab reads):
```bash
python scoring/score_book.py
```

**5. Launch the app:**
```bash
streamlit run app/app.py
```
Opens at `http://localhost:8501` with two tabs:
- **Dashboard** — portfolio KPIs, filterable by state/band/channel/new-renewal/body type, charts, rate-adequacy exception list.
- **Interface** — live what-if form; submits a hypothetical policy through the 3 trained models + rules engine and returns the full underwriting decision.

## Retraining

Whenever the underlying data changes, rerun steps 1–4 in order — the app
only ever reads the generated parquet/pkl artifacts, so it always reflects
the most recently trained models after a restart (`st.cache_resource` /
`st.cache_data` cache for the life of the Streamlit process — restart the
app after retraining to pick up new artifacts).

## Notes on the rules engine

`rules/rules_engine.py` intentionally stays non-ML — hard-stop eligibility
rules, premium loads (LAE%, Expense%, profit margin), and composite-score
weights (50% loss / 20% bind / 30% appetite) are configuration constants at
the top of the file. Change them there, not in the models.
