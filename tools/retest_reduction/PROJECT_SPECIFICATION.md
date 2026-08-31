# Antigravity Prompt — ATE Retest Recommendation AI Agent

## Objective

Build an **ATE Retest Recommendation AI Agent** for semiconductor ATE failure events.

The system must answer one business question:

> **Will performing a retest on this failed event be beneficial?**

This is **Option B only**.

The ML system must estimate:

`P(RETEST_BENEFICIAL)`

and a separate decision engine must convert that probability into:

- `RETEST`
- `DON'T RETEST`

Do **not** use an LLM as the mathematical decision-maker.

---

## Source of truth

First inspect the actual files in the workspace:

- `ATE_Retest_50_Devices_AI_Dataset.xlsx`
- `ATE_Retest_50_Devices_Month_0_Historical.xlsx`
- `ATE_Retest_50_Devices_Month_6_Historical.xlsx`
- `ATE_Retest_50_Devices_Month_12_NEW_Inference.xlsx`
- `RETEST~2.docx`

Read the DOCX and preserve its terminology, KPI definitions, values, findings, and framing. Do not silently change or reconcile its contents.

Treat the current dataset as a prototype/synthetic dataset, not physically validated semiconductor reliability data.

---

# 1. Longitudinal data design

### Month 0 — Historical

Use for historical learning.

### Month 6 — Historical

Use as additional historical data and temporal validation.

The same devices/events are tracked over time, while parametric values change.

### Month 12 — NEW / Inference

This is the unseen dataset.

The agent must make recommendations using Month 12 inputs **without seeing its actual outcome**.

Desired flow:

```text
Month 0 historical
        +
Month 6 historical
        ↓
      LEARN
        ↓
Month 12 new data
        ↓
P(RETEST_BENEFICIAL)
        ↓
RETEST / DON'T RETEST
        ↓
Explanation + KPI + impact
```

---

# 2. Target

The supervised target is:

`Ground_Truth`

Mapping:

```text
RETEST_BENEFICIAL  -> 1
PERSISTENT_FAILURE -> 0
```

Do not overwrite the original text label.

The model predicts whether the retest will be **beneficial**, not simply whether the first test will pass.

---

# 3. Data leakage — critical

Never use these as prediction features:

- `Ground_Truth`
- `Retest_Result`
- `Final_Result`
- `True_Retest_Pass_Probability`
- `AI_Retest_Probability`
- `AI_Recommendation`

Use a **feature whitelist**.

Historical outcome fields may be used for training/evaluation where appropriate, but Month 12 inference must not use them.

---

# 4. Initial model features

Inspect the actual workbooks before finalizing the feature list.

Candidate pre-retest features:

- `Wafer_ID`
- `ATE_Site`
- `Fail_Test`
- `Fail_Bin`
- `Voltage_V`
- `Temperature_C`
- `First_Result`
- `First_Test_Time_sec`
- `Retest_Time_sec`
- `Test_Month`

`Device_ID` and `Failure_Event` are identifiers/tracking fields and must not automatically become model features because they can encourage memorization.

Only use fields that are actually available before the retest decision.

---

# 5. ML models

Implement three candidate models:

### 1. Logistic Regression
Baseline and interpretable benchmark.

### 2. XGBoost
Primary candidate for larger future tabular datasets.

### 3. Gradient Boosting
Challenger model.

Do not assume XGBoost is automatically the winner.

Compare all three using validation evidence.

---

# 6. Validation strategy

Because this is longitudinal data, prioritize **temporal validation**.

Preferred structure:

```text
Month 0 → training
Month 6 → temporal validation
Month 12 → final unseen inference
```

Do not rely only on random train/test splitting.

The purpose is to simulate:

```text
PAST → LEARN
FUTURE → PREDICT
```

Do not fabricate performance numbers.

---

# 7. Probability calibration

The UI will display:

> Retest-benefit probability: 78%

Therefore probabilities must be treated as probabilities, not arbitrary model scores.

After model selection, implement a calibration layer, such as:

- `CalibratedClassifierCV`

Evaluate calibration where possible using reliability/calibration metrics.

Do not claim a model is calibrated unless validation supports that statement.

---

# 8. Decision threshold

Preserve the probability-threshold framing in the supplied DOCX.

Use:

`RETEST_THRESHOLD = 0.30`

Decision:

```python
if probability >= RETEST_THRESHOLD:
    recommendation = "RETEST"
else:
    recommendation = "DON'T RETEST"
```

Keep the threshold in one configuration location.

Do NOT create blanket rules such as:

```text
Scan -> always RETEST
MBIST -> always DON'T RETEST
IDDQ -> always DON'T RETEST
```

The new system must make event-level probability-based decisions.

---

# 9. Decision engine

Create a separate decision module.

Input:

- calibrated probability
- threshold

Output:

- probability
- threshold
- recommendation
- margin from threshold

Example:

```text
Probability = 0.74
Threshold = 0.30
Margin = +0.44
Recommendation = RETEST
```

---

# 10. Explainability

Use **SHAP** for XGBoost where appropriate.

The UI must answer:

> **Why did the agent recommend RETEST?**

Example:

```text
RETEST
74% retest-benefit probability

Top model contributors:
- Voltage_V
- Fail_Test
- Temperature_C
- Fail_Bin
- Wafer_ID
```

Translate contributions into understandable language.

Never claim a model contribution is physical causation.

Use wording such as:

> “Voltage_V was a strong contributor to the model prediction.”

not:

> “Voltage caused the device to pass.”

---

# 11. Agent architecture

The complete system should contain these layers:

```text
ATE Dataset
    ↓
Data Ingestion
    ↓
Data Validation
    ↓
Feature Engineering
    ↓
Model Training
    ↓
Model Comparison
    ↓
Best Model
    ↓
Probability Calibration
    ↓
P(RETEST_BENEFICIAL)
    ↓
Decision Engine
    ↓
RETEST / DON'T RETEST
    ↓
SHAP / Explanation
    ↓
KPI + Business Impact
    ↓
Antigravity UI
```

The “agent” is this complete orchestration system.

The ML model is the prediction engine.

---

# 12. UI — primary decision screen

Build the main screen around an engineer making a decision.

Show:

### Failure information

- Device ID
- Wafer ID
- ATE Site
- Failure Event
- Fail Test
- Fail Bin
- First Result

### Parameters

- Voltage
- Temperature
- First Test Time
- Retest Time
- Test Month

### AI result

Large cards:

```text
RETEST-BENEFIT PROBABILITY
74%

RECOMMENDATION
RETEST
```

Also display:

- active threshold
- probability margin
- model used
- estimated retest time
- explanation

Example:

```text
RETEST RECOMMENDED

Retest-benefit probability: 74%
Decision threshold: 30%
Estimated retest time: 15.8 sec

Why?
• Voltage_V was a strong model contributor
• Fail_Test was a strong model contributor
• Temperature_C contributed to the prediction
```

---

# 13. Batch / lot analysis screen

Allow the engineer to upload/select the Month 12 dataset and see:

- total failure events
- RETEST recommendations
- DON'T RETEST recommendations
- average retest-benefit probability
- recommendation distribution
- probability distribution
- breakdown by Fail_Test
- breakdown by Wafer_ID
- breakdown by ATE_Site

Allow filtering and sorting.

Allow export of recommendations to CSV/XLSX.

---

# 14. KPI dashboard

Do **not change the supplied DOCX KPI values or definitions**.

The supplied report contains these reference values:

- AI decision accuracy: **70.4%**
- Precision: **69.9%**
- Recall: **82.9%**
- Specificity: **54.5%**
- Unnecessary retests: **25 events (20.0%)**
- Unnecessary retest time: **853.0 sec**
- Missed opportunities: **12 events (9.6%)**

Keep these clearly labeled as:

**Reference / baseline KPIs from supplied report**

Do not replace them with current model values.

Also calculate a separate section:

**Current model evaluation**

---

# 15. Decision-quality metrics

Implement:

- Accuracy
- Precision
- Recall
- Specificity
- TP
- TN
- FP
- FN
- Confusion matrix

Use:

```text
                         Ground Truth
                    Beneficial   Persistent

AI RETEST              TP           FP

AI DON'T RETEST        FN           TN
```

Interpretation:

- TP = correct retest
- FP = unnecessary retest
- FN = missed opportunity
- TN = correct skip

Preserve the supplied report terminology.

---

# 16. Business-impact metrics

Calculate where supported:

### Unnecessary retests

```text
AI = RETEST
Ground truth = PERSISTENT_FAILURE
```

### Missed opportunities

```text
AI = DON'T RETEST
Ground truth = RETEST_BENEFICIAL
```

### Unnecessary retest time

Use `Retest_Time_sec` according to the supplied report methodology.

Do not invent a currency value.

If an ATE cost-per-second is not provided, show time impact rather than fabricated financial savings.

---

# 17. Test-type analysis

Support analysis by:

- Scan
- Func
- MBIST
- IDDQ
- AtSpeed

For each type, calculate where labels are available:

- event count
- RETEST count
- DON'T RETEST count
- accuracy
- precision
- recall
- unnecessary retests
- missed opportunities

Do not hard-code the recommendation by test type.

---

# 18. Wafer and ATE-site analysis

Support:

- Wafer_ID
- ATE_Site

Show:

- event count
- recommendations
- accuracy when labels exist
- error counts
- probability distribution

Treat small groups as directional, not statistically definitive.

---

# 19. Temporal analysis

Show:

```text
Month 0
Month 6
Month 12
```

For historical months:

- parameter distributions
- actual retest-benefit rate
- model predictions
- actual outcomes
- model performance

For Month 12:

- input parameter distributions
- predicted probabilities
- recommendations

Do not expose Month 12 actual outcomes during inference.

---

# 20. Data validation module

Validate:

- required columns
- data types
- missing values
- duplicates
- probability ranges
- ground-truth labels
- inference leakage
- device/event consistency
- schema differences between months

Produce actionable errors.

Example:

```text
Dataset validation failed.

Missing:
Temperature_C

Unexpected outcome field in inference:
Retest_Result
```

Do not silently hide important data issues.

---

# 21. Recommended project structure

Use a modular project:

```text
retest_ai/
│
├── app.py
│
├── config/
│   └── settings.py
│
├── data/
│   ├── ingestion.py
│   ├── validation.py
│   └── preprocessing.py
│
├── models/
│   ├── logistic_model.py
│   ├── xgboost_model.py
│   ├── gradient_boosting.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── calibration.py
│
├── decision/
│   └── decision_engine.py
│
├── explainability/
│   └── shap_explainer.py
│
├── kpis/
│   ├── decision_quality.py
│   ├── business_impact.py
│   └── breakdowns.py
│
├── ui/
│   ├── dashboard.py
│   ├── decision_view.py
│   └── performance_view.py
│
├── tests/
│
├── requirements.txt
│
└── README.md
```

Keep it modular but do not overengineer the small prototype.

---

# 22. Technology

Preferred:

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- SHAP
- openpyxl
- Streamlit

Use suitable visualization libraries where needed.

Do not introduce unnecessary frameworks.

---

# 23. LLM role

Do NOT use an LLM to generate the prediction.

Correct architecture:

```text
ML model
    ↓
probability

Decision engine
    ↓
RETEST / DON'T RETEST

LLM (optional later)
    ↓
natural-language explanation / conversation
```

The LLM may later answer:

> Why did you recommend retest?

or:

> Which test type has the most missed opportunities?

But it must not invent the probability or make the underlying decision.

---

# 24. Model artifacts and reproducibility

Save:

- trained model
- preprocessing pipeline
- calibration object
- feature list
- threshold
- model version
- training datasets
- validation dataset
- hyperparameters
- evaluation metrics

Use fixed random seeds where appropriate.

Example:

```text
retest_xgb_v1
```

---

# 25. Testing

Create tests for:

### Data

- workbook loading
- missing columns
- duplicates
- leakage detection

### Models

- training
- prediction
- probability range [0,1]

### Decision

```text
0.29 -> DON'T RETEST
0.30 -> RETEST
0.74 -> RETEST
```

### KPI

Validate TP/TN/FP/FN and metric calculations.

### UI

Test:

- upload
- validation
- event selection
- recommendation
- batch table
- export

---

# 26. What NOT to build

Do not:

- use deep learning unnecessarily
- use CNN/RNN/LSTM/Transformer as the first model
- use an LLM as the classifier
- use future outcomes as inputs
- use ground truth as a feature
- hard-code test-type decisions
- fabricate performance
- fabricate financial savings
- claim physical causation from SHAP
- modify the supplied DOCX KPI values
- claim XGBoost is proven best on the current small dataset

---

# 27. Development phases

Do NOT generate the entire application blindly.

Work incrementally.

## Phase 1 — Inspect and validate

First inspect:

- workbooks
- columns
- row counts
- devices
- events
- target
- feature candidates
- leakage

Produce a schema/data-quality report.

Do not train until this is confirmed.

## Phase 2 — Baseline ML

Implement:

- Logistic Regression
- XGBoost
- Gradient Boosting

Compare them using temporal validation.

## Phase 3 — Model selection + calibration

Select based on evidence.

Calibrate probability.

## Phase 4 — Decision engine

Implement the 30% threshold.

## Phase 5 — Explainability

Implement SHAP for XGBoost where appropriate.

## Phase 6 — KPI/business engine

Implement decision-quality and time-impact metrics.

## Phase 7 — UI

Build the polished Antigravity UI only after the ML pipeline works.

---

# 28. Acceptance criteria

The prototype is successful when:

### Data

- Month 0 loads
- Month 6 loads
- Month 12 loads
- same device/event tracking is verified
- leakage is prevented

### ML

- all three candidate models work
- temporal validation works
- model comparison works
- probability is generated
- calibration is evaluated

### Decision

Every Month 12 event receives:

```text
P(RETEST_BENEFICIAL)
+
30% threshold
↓
RETEST / DON'T RETEST
```

### Explanation

The UI can answer:

> Why did the model make this recommendation?

### KPI

Current model metrics and supplied report reference KPIs are clearly separated.

### UI

An engineer can:

1. load data
2. select an event
3. see probability
4. see RETEST / DON'T RETEST
5. see explanation
6. see retest time
7. inspect batch recommendations
8. export results

---

# 29. Final product vision

The finished product should be an engineering decision-support system:

```text
┌──────────────────────────────────────────────┐
│          ATE RETEST AI AGENT                 │
├──────────────────────────────────────────────┤
│ Device: DEV023                               │
│ Failure: MBIST_03                            │
│ Fail Bin: 45                                 │
│ Voltage: 0.91 V                              │
│ Temperature: 55°C                            │
│                                              │
│ RETEST-BENEFIT PROBABILITY                   │
│                    74%                       │
│                                              │
│        ★ RECOMMEND: RETEST                   │
│                                              │
│ Threshold: 30%                               │
│ Retest time: 15.8 sec                        │
│                                              │
│ WHY?                                         │
│ • Voltage_V contributed strongly             │
│ • Fail_Test contributed strongly             │
│ • Temperature_C contributed                 │
└──────────────────────────────────────────────┘
```

The primary objective is **not** a visually impressive dashboard.

The primary objective is a **trustworthy, explainable, probability-based ATE retest decision system**.

---

# 30. FIRST COMMAND / FIRST ACTION

Do not start by writing the full application.

First:

1. Inspect every available dataset.
2. Inspect the supplied DOCX.
3. Print the schema for Month 0, Month 6, and Month 12.
4. Confirm device/event counts.
5. Confirm target labels.
6. Identify the exact feature whitelist.
7. Run the leakage check.
8. Explain the proposed training/validation/inference split.
9. STOP and report the findings before implementing the ML model.

Only after this validation phase is correct should you proceed to model implementation.
