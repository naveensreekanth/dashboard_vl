# ATE Retest-Benefit Prediction AI

A supervised machine-learning prototype for semiconductor ATE failure events.

When an ATE test fails, the system uses **only information available before a physical retest**, predicts the probability that a retest would be beneficial, applies a separately defined decision policy, and later scores that recommendation against the actual outcome.

```text
ATE FAIL
    → Pre-retest parameters
    → ML model
    → P(RETEST_BENEFICIAL)
    → DOCX-reference 30% decision policy
    → RETEST / DON'T RETEST
    → Actual outcome later
    → Validation
    → KPI cards
```

The ML model is not an LLM. XGBoost is a candidate classifier, not an "agent". The overall application is an AI agent because it runs prediction, recommendation, and validation as one workflow.

---

## 1. Project purpose

Answer two different questions without mixing them:

1. **Should this failed event be sent for retest?**  
   `P(RETEST_BENEFICIAL)` plus an isolated recommendation.

2. **How well has the AI been making those decisions?**  
   Post-outcome comparison of recommendation vs `Ground_Truth`.

---

## 2. Dataset structure

| File | Role |
|---|---|
| `ATE_Retest_50_Devices_Month_0_Historical.xlsx` | Labeled history. **Train** for temporal validation. |
| `ATE_Retest_50_Devices_Month_6_Historical.xlsx` | Labeled history. **Temporal holdout** (not mixed randomly with Month 0). |
| `ATE_Retest_50_Devices_Month_12_NEW_Inference.xlsx` | Unseen events. **Inference only**. No ground truth required. |
| `ATE_Retest_50_Devices_AI_Dataset.xlsx` | Historical snapshot used to **recompute** the supplied DOCX reference KPIs. |
| `Month_12_PRIVATE_VALIDATION_ONLY.xlsx` | Optional later outcomes. **Never** used as prediction input. |

Final deployment model is trained on Month 0 + Month 6, then used for Month 12 inference. The UI labels this difference.

---

## 3. Pre-retest features (whitelist)

- `Wafer_ID`
- `ATE_Site`
- `Fail_Test`
- `Fail_Bin`
- `First_Result`
- `Voltage_V`
- `Temperature_C`
- `First_Test_Time_sec`
- `Test_Month`

Identifiers kept for tracking/UI only: `Device_ID`, `Failure_Event`.

---

## 4. Leakage prevention

These fields must never enter the prediction feature matrix `X`:

- `Ground_Truth`, `Retest_Result`, `Final_Result`, `Retest_Count`
- `True_Retest_Pass_Probability`, `AI_Retest_Probability`, `AI_Recommendation`
- `Retest_Time_sec` (actual retest execution time)

The API rejects them as inputs. Training workbooks may contain labels; those labels are used only as the supervised target or for later validation, not as features.

---

## 5. Training process

Supervised target: `Ground_Truth`

- `RETEST_BENEFICIAL` → 1
- `PERSISTENT_FAILURE` → 0

Model output: `P(RETEST_BENEFICIAL)` in `[0, 1]`. Probabilities are not rewritten to look better in the UI.

---

## 6. Temporal validation

Train on Month 0, evaluate on Month 6. This is historical temporal validation, not Month 12 performance.

After selection, a deployment model is fit on Month 0 + Month 6 for Month 12 inference.

---

## 7. Model comparison

Candidates: Logistic Regression, Gradient Boosting, XGBoost.

Selection is evidence-based on Month 6 holdout (ROC-AUC, then Brier Score). The UI states why the selected model was chosen. XGBoost is not declared best unless the measured results support it.

Calibration (Brier, Log Loss, reliability buckets) is evaluated, not used to invent a new probability.

A **0.5 evaluation/reporting cutoff** may be shown in an expander for model comparison only. It does **not** produce operational RETEST / DON'T RETEST recommendations.

---

## 8. Probability output

Primary ML output:

`P(RETEST_BENEFICIAL)`  e.g. `0.78` or `78%`

---

## 9. Decision policy

Isolated in `retest_ai/decision/decision_policy.py`.

```text
if P(RETEST_BENEFICIAL) >= 0.30 → RETEST
if P(RETEST_BENEFICIAL) <  0.30 → DON'T RETEST
```

Label used in the UI and API:

**Reference / DOCX decision policy — subject to validation**

This is the 30% logic referenced in `RETEST~2.docx`. It is **not** described as a scientifically proven or permanently approved production threshold. Changing the policy does not require retraining the model. The probability is not modified when the recommendation is applied.

There is no 50% threshold and no threshold slider.

---

## 10. Post-outcome validation

After the actual outcome is known:

```text
AI recommendation + actual Ground_Truth
    → TP Correct Retest
    → FP Unnecessary Retest
    → FN Missed Opportunity
    → TN Correct Skip
```

Month 12 outcome KPIs appear only after outcomes are loaded separately.

---

## 11. KPI calculations

All dynamic (not hard-coded from the DOCX):

- Accuracy = (TP + TN) / Total
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- Specificity = TN / (TN + FP)
- Unnecessary Retests = FP and FP/Total
- Missed Opportunities = FN and FN/Total

DOCX figures such as 70.4% accuracy are historical report values. They are listed for audit and recomputed from the AI dataset. They are not shown as current Month 12 performance.

---

## 12. FastAPI startup

From the project root:

```powershell
uvicorn retest_ai.api.main:app --reload
```

- Service: http://127.0.0.1:8000
- Swagger: http://127.0.0.1:8000/docs

Endpoints: `GET /health`, `GET /model/info`, `POST /predict`, `POST /predict/batch`.

---

## 13. Streamlit startup

From the project root:

```powershell
streamlit run retest_ai/app.py
```

- Single Event: event info, probability, DOCX-reference recommendation, SHAP contributions
- Month 12: event-level probability + recommendation, filters, CSV/Excel export
- Overview: Month 12 prediction KPIs; clickable outcome KPIs and 2x2 matrix on labeled Month 6 data
- Historical Temporal Validation: model comparison
- Reference Report: DOCX listed vs recomputed historical KPIs

---

## 14. Testing

From the project root:

```powershell
python -m unittest discover -s retest_ai/tests -p "test_*.py"
```

---

## 15. Known limitations

- Prototype / synthetic-scale ATE dataset (50 devices, 125 events per month file).
- The 30% recommendation rule is a **DOCX-reference policy subject to validation**, not a certified production threshold.
- Month 12 accuracy cannot be claimed until actual Month 12 outcomes are provided separately.
- SHAP explains model contribution, not physical root cause.
- Temporal holdout uses 125 Month 6 events; calibration buckets are small and should be treated as directional.
