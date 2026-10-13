"""
train_7features_rf.py
=====================
Retraining script for multi-target product taxonomy classification using Random Forest
with an expanded 7-feature pipeline (5 categoricals + 2 sentence-transformer text features).

Target Taxonomy Levels:
- ProductType
- ProductGroup
- ProductFamily
- ProductLine
- ProductKey

Features Used (7):
1. GPCBrickCode (Categorical)
2. UNSPSCNumber (Categorical)
3. IsMedical (Categorical)
4. BrandName (Categorical)
5. SubBrandName (Categorical)
6. ProductDescription (Text, all-MiniLM-L6-v2 embeddings)
7. ProductShortName (Text, all-MiniLM-L6-v2 embeddings)

Features Excluded (Target Leakage):
- ProductCategory (synonymous with ProductType)
- ProductPartType (synonymous with ProductType)
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATASET_XLSX = BASE_DIR / "datasets" / "product_classified_Full.xlsx"
DATASET_CSV = BASE_DIR / "datasets" / "product_classified_Full.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

TARGETS = ["ProductType", "ProductGroup", "ProductFamily", "ProductLine", "ProductKey"]
CAT_FEATURES = ["GPCBrickCode", "UNSPSCNumber", "IsMedical", "BrandName", "SubBrandName"]
TEXT_FEATURES = ["ProductDescription", "ProductShortName"]
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

ARTIFACT_PREFIXES = {
    "ProductType": "product_type",
    "ProductGroup": "product_group",
    "ProductFamily": "product_family",
    "ProductLine": "product_line",
    "ProductKey": "product_key",
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def top_k_accuracy(proba, classes, y_true, k=3):
    class_to_idx = {c: i for i, c in enumerate(classes)}
    true_idx = y_true.map(class_to_idx).to_numpy()
    top_k_idx = np.argsort(proba, axis=1)[:, -k:]
    return np.mean([t in row for t, row in zip(true_idx, top_k_idx)])

def top_k_from_proba(proba_row, classes, k=3):
    order = np.argsort(proba_row)[::-1][:k]
    return [(classes[i], proba_row[i]) for i in order]


def main():
    print("=" * 80)
    print("Starting 7-Feature Random Forest Retraining Pipeline")
    print("=" * 80)

    # 1. Load Dataset
    if DATASET_XLSX.exists():
        print(f"Loading data from {DATASET_XLSX} (sheet 'in')...")
        df_raw = pd.read_excel(DATASET_XLSX, sheet_name="in")
    elif DATASET_CSV.exists():
        print(f"Loading data from {DATASET_CSV}...")
        df_raw = pd.read_csv(DATASET_CSV)
    else:
        print(f"Error: Neither {DATASET_XLSX} nor {DATASET_CSV} found.")
        sys.exit(1)
    print(f"Raw dataset shape: {df_raw.shape}")

    # Retain the exact baseline row-filter: drop rows where baseline triplet is all null
    df = df_raw.dropna(
        subset=["GPCBrickCode", "UNSPSCNumber", "ProductDescription"],
        how="all"
    ).copy()
    print(f"Shape after dropping all-null baseline rows: {df.shape} (dropped {len(df_raw) - len(df)})")

    # 2. Impute & Clean Features
    print("Cleaning and imputing 7 input features...")
    df["GPCBrickCode"] = df["GPCBrickCode"].fillna("Missing").astype(str)
    df["UNSPSCNumber"] = df["UNSPSCNumber"].fillna(-1).astype(int).astype(str)
    df["IsMedical"] = df["IsMedical"].fillna(False).astype(str)
    df["BrandName"] = df["BrandName"].fillna("Missing").astype(str)
    df["SubBrandName"] = df["SubBrandName"].fillna("Missing").astype(str)
    df["ProductDescription"] = df["ProductDescription"].fillna("").astype(str)
    df["ProductShortName"] = df["ProductShortName"].fillna("").astype(str)

    # 3. Text Embeddings
    print(f"Initializing SentenceTransformer: {EMBED_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    embed_dim = embedder.get_sentence_embedding_dimension()
    print(f"Embedding dimension per text field: {embed_dim}")

    # Factorize & embed ProductDescription
    print("Encoding ProductDescription embeddings...")
    codes_desc, uniques_desc = pd.factorize(df["ProductDescription"])
    non_empty_mask_desc = uniques_desc != ""
    non_empty_texts_desc = uniques_desc[non_empty_mask_desc].tolist()

    if non_empty_texts_desc:
        non_empty_emb_desc = embedder.encode(
            non_empty_texts_desc,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=64,
        )
    else:
        non_empty_emb_desc = np.zeros((0, embed_dim), dtype=np.float32)

    unique_embeddings_desc = np.zeros((len(uniques_desc), embed_dim), dtype=np.float32)
    unique_embeddings_desc[non_empty_mask_desc] = non_empty_emb_desc
    X_embed_desc = unique_embeddings_desc[codes_desc]
    print(f"ProductDescription feature matrix shape: {X_embed_desc.shape}")

    # Factorize & embed ProductShortName
    print("Encoding ProductShortName embeddings...")
    codes_short, uniques_short = pd.factorize(df["ProductShortName"])
    non_empty_mask_short = uniques_short != ""
    non_empty_texts_short = uniques_short[non_empty_mask_short].tolist()

    if non_empty_texts_short:
        non_empty_emb_short = embedder.encode(
            non_empty_texts_short,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=64,
        )
    else:
        non_empty_emb_short = np.zeros((0, embed_dim), dtype=np.float32)

    unique_embeddings_short = np.zeros((len(uniques_short), embed_dim), dtype=np.float32)
    unique_embeddings_short[non_empty_mask_short] = non_empty_emb_short
    X_embed_short = unique_embeddings_short[codes_short]
    print(f"ProductShortName feature matrix shape: {X_embed_short.shape}")

    # Combine text embeddings
    X_text = np.hstack([X_embed_desc, X_embed_short])
    print(f"Total combined text feature matrix shape: {X_text.shape}")

    # 4. Target Preparation & Fold Rare Classes
    MIN_CLASS_COUNT = 10
    df_prepared = df.copy()
    for TARGET in TARGETS:
        counts = df_prepared[TARGET].value_counts()
        rare_classes = counts[counts < MIN_CLASS_COUNT].index
        df_prepared[TARGET] = df_prepared[TARGET].replace(rare_classes, "Other")
        print(f"Target '{TARGET}': {df_prepared[TARGET].nunique()} unique classes after folding rare (<{MIN_CLASS_COUNT}) to 'Other'")

    # 5. Train / Test Split and Training Loop
    results = {}

    for TARGET in TARGETS:
        print("\n" + "=" * 60)
        print(f"Training Model for Target: {TARGET}")
        print("=" * 60)

        t0 = time.time()
        # Stratified split using target
        train_idx, test_idx = train_test_split(
            df_prepared.index,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=df_prepared[TARGET],
        )

        df_train = df_prepared.loc[train_idx]
        df_test = df_prepared.loc[test_idx]

        # OneHotEncode 5 categorical features strictly fit on df_train
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat_train = ohe.fit_transform(df_train[CAT_FEATURES])
        X_cat_test = ohe.transform(df_test[CAT_FEATURES])

        # Align text features
        train_pos = df_prepared.index.get_indexer(train_idx)
        test_pos = df_prepared.index.get_indexer(test_idx)
        X_text_train = X_text[train_pos]
        X_text_test = X_text[test_pos]

        # Assemble feature matrices
        X_train = np.hstack([X_cat_train, X_text_train])
        X_test = np.hstack([X_cat_test, X_text_test])
        y_train = df_train[TARGET]
        y_test = df_test[TARGET]

        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        print(f"Fitting RandomForestClassifier (600 trees, random_state={RANDOM_STATE})...")

        rf = RandomForestClassifier(
            n_estimators=600,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        # Predictions & Evaluations
        y_pred = rf.predict(X_test)
        proba_test = rf.predict_proba(X_test)
        classes = rf.classes_

        top1 = top_k_accuracy(proba_test, classes, y_test, 1)
        top3 = top_k_accuracy(proba_test, classes, y_test, 3)
        top5 = top_k_accuracy(proba_test, classes, y_test, 5)
        wf1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        elapsed = time.time() - t0

        print(f"\n--- {TARGET} Results (Elapsed: {elapsed:.1f}s) ---")
        print(f"Top-1 Accuracy : {top1:.4f} ({top1:.1%})")
        print(f"Top-3 Accuracy : {top3:.4f} ({top3:.1%})")
        print(f"Top-5 Accuracy : {top5:.4f} ({top5:.1%})")
        print(f"Weighted F1    : {wf1:.4f}")

        results[TARGET] = {
            "Top-1": top1,
            "Top-3": top3,
            "Top-5": top5,
            "Weighted F1": wf1,
            "Classes": len(classes),
            "Elapsed": elapsed,
        }

        # 6. Save Artifacts
        prefix = ARTIFACT_PREFIXES[TARGET]
        model_path = ARTIFACT_DIR / f"{prefix}_rf_7feat_model.joblib"
        ohe_path = ARTIFACT_DIR / f"{prefix}_rf_7feat_ohe.joblib"
        meta_path = ARTIFACT_DIR / f"{prefix}_rf_7feat_meta.joblib"

        joblib.dump(rf, model_path)
        joblib.dump(ohe, ohe_path)
        joblib.dump(
            {
                "embed_model_name": EMBED_MODEL_NAME,
                "embed_dim": embed_dim,
                "cat_features": CAT_FEATURES,
                "text_features": TEXT_FEATURES,
                "classes": classes.tolist(),
            },
            meta_path,
        )
        print(f"Serialized artifacts to {ARTIFACT_DIR}/{prefix}_rf_7feat_*")

    # 7. Summary Benchmark Table
    print("\n" + "=" * 80)
    print("7-FEATURE MODEL EVALUATION SUMMARY BENCHMARK")
    print("=" * 80)
    summary_df = pd.DataFrame(results).T
    summary_df["Top-1 (%)"] = summary_df["Top-1"].map(lambda x: f"{x:.1%}")
    summary_df["Top-3 (%)"] = summary_df["Top-3"].map(lambda x: f"{x:.1%}")
    summary_df["Top-5 (%)"] = summary_df["Top-5"].map(lambda x: f"{x:.1%}")
    summary_df["Weighted F1"] = summary_df["Weighted F1"].map(lambda x: f"{x:.4f}")
    print(summary_df[["Top-1 (%)", "Top-3 (%)", "Top-5 (%)", "Weighted F1", "Classes", "Elapsed"]])
    print("=" * 80)

if __name__ == "__main__":
    main()
