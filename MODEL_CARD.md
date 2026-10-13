# Laerdal Product Hierarchy Classification — Model Card & Experiment Registry

This document serves as the **single source of truth** for:
1. **ML Workflow Lifecycle & Current Status**
2. **Standard Operating Procedures (Checklists & Linked Activities)**
3. **Model Version Cards (v1.0 Baseline vs. v2.0 Augmented)**
4. **Comparative Performance Benchmarks (Held-out Test Set)**

---

## 1. End-to-End ML Workflow Stage Tracker

| Stage | Name | Description | Status | Linked Activity / Deliverable |
| :---: | :--- | :--- | :---: | :--- |
| **01** | **Problem Framing & Business Scope** | Define the 5 target hierarchies, business inputs, and operational constraints. | ✅ **Done** | Jira [LBVOICESER-1516](https://laerdal.atlassian.net/browse/LBVOICESER-1516) |
| **02** | **Data Profiling & Quality Audit** | Audit the 10,237 raw dataset rows, missing values, encoding issues, and leakage risks. | ✅ **Done** | Profiling script; Data drift audit |
| **03** | **Feature Lineage & Exclusions** | Establish feature inclusion/exclusion decisions and eliminate target leakage. | ✅ **Done** | [FEATURE_LOG.md](./FEATURE_LOG.md), Jira [LBVOICESER-1567](https://laerdal.atlassian.net/browse/LBVOICESER-1567) |
| **04** | **Data Preprocessing & Encoding** | Impute nulls, cast types, fit OHE for categories, and embed text with `all-MiniLM-L6-v2`. | ✅ **Done** | Pipeline module ([train_7features_rf.py](./train_7features_rf.py)) |
| **05** | **Model Training & Experimentation** | Train 5 multiclass Random Forest models (600 trees) on stratified 80/20 split. | ✅ **Done** | Jira [LBVOICESER-1564](https://laerdal.atlassian.net/browse/LBVOICESER-1564) |
| **06** | **Evaluation & Head-to-Head Benchmark** | Evaluate Top-1, Top-3, and Top-5 accuracy on the identical held-out test set. | ✅ **Done** | Section 3 of this document |
| **07** | **Packaging, Inference & Deployment** | Export versioned Joblib artifacts, deploy FastAPI & Streamlit in Docker / local venv. | ✅ **Done** | `artifacts/*_rf_7feat_*`, `api.py`, `streamlit_app_v2.py`, Docker & local `.venv` |

---

## 2. Standard Operating Checklist for Every Model Iteration

Whenever an experiment or retraining run is conducted, the following checklist must be satisfied:

- [ ] **Step 1: Check Feature Contract**: Verify inputs against [FEATURE_LOG.md](./FEATURE_LOG.md) to ensure no target leakage columns are included.
- [ ] **Step 2: Preserve Baseline**: Never overwrite existing artifacts (`_rf_3feat_*`). Save new artifacts with distinct versioned tags (e.g., `_rf_7feat_*`).
- [ ] **Step 3: Train on Frozen Split**: Train on 80% split using `random_state=42`, evaluate on the exact 20% held-out test split (1,918 rows).
- [ ] **Step 4: Generate Run Manifest**: Save an automated `run_manifest_{timestamp}.json` recording dataset hash, parameters, and test metrics.
- [ ] **Step 5: Update Model Card**: Update the Model Version Registry (Section 3 below) with accuracy and sample counts.
- [ ] **Step 6: Sync to Jira**: Post the results to [LBVOICESER-1564](https://laerdal.atlassian.net/browse/LBVOICESER-1564) and update parent [LBVOICESER-1516](https://laerdal.atlassian.net/browse/LBVOICESER-1516).

---

## 3. Model Version Registry & Comparative Performance

### Model Version Summary

| Dimension | Version 1.0 (Baseline Model) | Version 2.0 (Augmented Model) |
| :--- | :--- | :--- |
| **Release Date** | 2026-08-11 | 2026-08-11 |
| **Artifact Tag** | `_rf_3feat_*` | `_rf_7feat_*` |
| **Algorithm** | Random Forest (`n_estimators=600`, `min_samples_leaf=1`) | Random Forest (`n_estimators=600`, `min_samples_leaf=1`) |
| **Input Features** | **3 Features**: `GPCBrickCode`, `UNSPSCNumber`, `ProductDescription` | **7 Features**: `GPCBrickCode`, `UNSPSCNumber`, `ProductDescription`, `ProductShortName`, `IsMedical`, `BrandName`, `SubBrandName` |
| **Text Embedding** | `all-MiniLM-L6-v2` (384 dims for `ProductDescription`) | `all-MiniLM-L6-v2` (384 dims for `ProductDescription` + 384 dims for `ProductShortName`, 768 dense dims total) |
| **Training Split** | 80% Stratified (7,669 rows, `random_state=42`) | 80% Stratified (7,669 rows, `random_state=42`) |
| **Held-out Test Split**| 20% Stratified (1,918 rows, `random_state=42`) | 20% Stratified (1,918 rows, `random_state=42`) |
| **Jira Reference** | Baseline Notebook | [LBVOICESER-1564](https://laerdal.atlassian.net/browse/LBVOICESER-1564) |

---

### Head-to-Head Accuracy Comparison (Held-out Test Set, N=1,918)

| Target Label | Metric | v1.0 Baseline (3 Feat) | v2.0 Augmented (7 Feat) | Delta ($\Delta$) | Statistical Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`ProductType`** | Top-1 | **64.2%** | **80.1%** | **+15.9%** | Major statistical gain across service & parts classes |
| | Top-3 | **88.9%** | **95.0%** | **+6.1%** | Near-complete coverage |
| | Top-5 | ~94.0% | **98.5%** | **+4.5%** | Outstanding top-candidate recovery |
| | Weighted F1 | 0.6200 | **0.7911** | **+0.1711** | Highly balanced precision/recall |
| **`ProductGroup`** | Top-1 | **77.8%** | **87.2%** | **+9.4%** | Clear disambiguation |
| | Top-3 | **93.2%** | **95.9%** | **+2.7%** | Near-ceiling accuracy |
| | Top-5 | ~96.5% | **97.2%** | **+0.7%** | Consistent catalog coverage |
| | Weighted F1 | 0.7700 | **0.8664** | **+0.0964** | Sharp reduction in cross-group misclassification |
| **`ProductFamily`** | Top-1 | **71.4%** | **85.7%** | **+14.3%** | Major breakthrough across 63 families |
| | Top-3 | **89.1%** | **95.4%** | **+6.3%** | Reliable suggestion list |
| | Top-5 | ~93.8% | **97.2%** | **+3.4%** | Error rate cut in half |
| | Weighted F1 | 0.5900 | **0.8500** | **+0.2600** | Exceptional recall on rare & mid-frequency families |
| **`ProductLine`** | Top-1 | **68.7%** | **87.1%** | **+18.4%** | Huge performance leap across 145 product lines |
| | Top-3 | **86.4%** | **94.5%** | **+8.1%** | Near-universal top-3 hit rate |
| | Top-5 | ~91.2% | **96.0%** | **+4.8%** | Robust top-5 fallback |
| | Weighted F1 | 0.4900 | **0.8676** | **+0.3776** | Massive improvement on granular lines |
| **`ProductKey`** | Top-1 | **62.5%** | **74.9%** | **+12.4%** | Significant jump on deepest taxonomy level (76 keys) |
| | Top-3 | **83.1%** | **91.4%** | **+8.3%** | High precision in top-3 candidates |
| | Top-5 | ~88.7% | **94.5%** | **+5.8%** | Reliable multi-choice recommendation |
| | Weighted F1 | 0.4500 | **0.7357** | **+0.2857** | Strong disambiguation on product key codes |

*(Note: Baseline figures retrieved from `evaluation/paired_rf_vs_1nn_results.json` evaluated on the exact 1,918 test rows; v2.0 evaluated on identical held-out test rows).*

---

## 4. Intended Use & Known Limitations

### Intended Use
* Suggests top candidate classification taxonomies (`ProductType`, `ProductGroup`, `ProductFamily`, `ProductLine`, `ProductKey`) for master data stewards during new product setup and catalog curation.
* Provides Top-1, Top-3, and Top-5 suggested rankings with calibrated confidence scores.

### Known Limitations
* **Cold-start products**: Products lacking both GS1/UNSPSC numbers and descriptions rely entirely on ShortName; confidence scores are lower for novel product concepts.
* **Rare classes (<10 instances)**: Grouped into `"Other"` during model training to ensure statistical stability.
* **Target leakage guard**: Business workflow inputs matching targets (`ProductCategory`, `ProductPartType`) must remain strictly excluded from inference inputs.
