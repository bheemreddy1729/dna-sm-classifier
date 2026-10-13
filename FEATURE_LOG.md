# Feature Decision Log & Lineage Register

This document tracks all features evaluated, included, and excluded for the **Laerdal Product Hierarchy Classification Models** (`ProductType`, `ProductGroup`, `ProductFamily`, `ProductLine`, `ProductKey`).

> **Single Source of Truth**: For the overall ML lifecycle status, operational checklists, and version comparison tables, see [MODEL_CARD.md](./MODEL_CARD.md).

---

## 1. Feature Decision Matrix

| Parameter / Field Name | Dataset Column | Iteration 1 (3-Feat Baseline) | Iteration 2 (7-Feat Augmented) | Data Type | Preprocessing & Encoding Pipeline | Status / Decision Rationale |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **`ProductNumber`** | `ProductNumber` | Excluded | Excluded | Alphanumeric Identifier | Excluded from feature vector. Carried as row identifier for tracking and UI output mapping. | High cardinality identifier (9,000+ unique values). Feeding to trees causes extreme memorization and catastrophic overfitting. |
| **`GPCBrickCode`** | `GPCBrickCode` | **Included** | **Included** | Categorical | `fillna("Missing")` -> `OneHotEncoder(handle_unknown="ignore")` | GS1 Global Product Classification standard code. Highly predictive category signal when populated. |
| **`UNSPSCNumber`** | `UNSPSCNumber` | **Included** | **Included** | Categorical | Float converted to int string (`42301502.0` -> `"42301502"`), missing filled with `"-1"` -> `OneHotEncoder(handle_unknown="ignore")` | United Nations Standard Products and Services Code. Key taxonomic indicator. |
| **`LongDescription`** | `ProductDescription` | **Included** | **Included** | Free Text | `all-MiniLM-L6-v2` dense embedding (384 dimensions; zero vector if empty) | Core descriptive text explaining product clinical context. Populated in ~41% of records. |
| **`ShortName`** | `ProductShortName` | Excluded | **Included** | Free Text | `all-MiniLM-L6-v2` dense embedding (384 dimensions; zero vector if empty) | **100% populated** across all 10,237 records. Contains concise, brand-anchored product nomenclature (e.g. "Resusci Anne QCPR"). High information density. |
| **`MedicalDevice`** | `IsMedical` | Excluded | **Included** | Boolean / Categorical | Normalized boolean (`True`/`False`), maps workflow `Yes`/`No` -> `OneHotEncoder` | Distinguishes medical devices from non-medical training simulators and consumables. 100% populated. |
| **`BrandName`** | `BrandName` | Excluded | **Included** | Categorical | `fillna("Missing")` -> `OneHotEncoder(handle_unknown="ignore")` | Top-level portfolio anchor (e.g. "SimMan", "Resusci", "Little"). Populated in 5,734 records. |
| **`SubBrandName`** | `SubBrandName` | Excluded | **Included** | Categorical | `fillna("Missing")`, literal `"No Sub-brand"` retained as distinct category -> `OneHotEncoder(handle_unknown="ignore")` | Granular product sub-tier anchor (e.g. "3G", "QCPR"). Populated in 5,734 records. |
| **`ProductLifecycle`** | - | Excluded | Excluded | - | N/A | Excluded per business confirmation (Mudassar). Dataset text occurrences ("Production") in variant fields represent historical manufacturing eras ("Post 2005 Production"), not current lifecycle status. |
| **`ProductPartType`** | - | Excluded | Excluded | - | N/A | No independent dataset column exists. Business proposed using `ProductType`, which causes **direct target leakage**. |
| **`ProductCategory`** | - | Excluded | Excluded | - | N/A | Excluded due to direct semantic overlap with the `ProductType` target (leakage risk). |
| **`VariantDescription`** | `VariantDescription` | Excluded | Excluded | Free Text | Reserved for future ablation experiment | Tracked under sub-task `LBVOICESER-1566` to test in a later candidate run. |
| **`TradeItemUnitDescription`** | `TradeItemUnitDescription` | Excluded | Excluded | Free Text | Reserved for future ablation experiment | Tracked under sub-task `LBVOICESER-1566` to test in a later candidate run. |

---

## 2. Model Experiment Runs & Lineage

| Run ID | Timestamp (UTC) | Model Architecture | Feature Set | Targets | Test Set Accuracy (Top-1 / Top-3) | Artifact Prefix | Jira Reference |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`RUN-001`** | 2026-08-11 | Random Forest (600 trees, `min_samples_leaf=1`) | 3 Features (`GPCBrickCode`, `UNSPSCNumber`, `ProductDescription`) | All 5 targets | Baseline (documented in notebook) | `_rf_3feat_*` | Baseline |
| **`RUN-002`** | *Pending* | Random Forest (600 trees, `min_samples_leaf=1`) | 7 Features (Baseline + `ProductShortName`, `IsMedical`, `BrandName`, `SubBrandName`) | All 5 targets | *To be computed* | `_rf_7feat_*` | [LBVOICESER-1564](https://laerdal.atlassian.net/browse/LBVOICESER-1564) |

---

## 3. Governance & Quality Gates
1. **Target Leakage Prohibition**: No model feature shall be derived from or correlated identically with any of the 5 targets (`ProductType`, `ProductGroup`, `ProductFamily`, `ProductLine`, `ProductKey`).
2. **Artifact Immutability**: All model, encoder, and metadata joblib files must carry their feature count and version in the filename (`_rf_3feat_` vs `_rf_7feat_`).
3. **Execution Manifest**: Every retraining script execution will output a structured JSON manifest containing:
   - Run timestamp (UTC)
   - Git commit hash
   - Dataset filename and SHA-256 hash
   - Feature list and column mappings
   - Top-1, Top-3, and Top-5 accuracy metrics per target
   - Number of training and test samples
