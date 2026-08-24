import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

import joblib


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


print("\n========================================")
print("PCOS ML MODEL TRAINING")
print("========================================")

print("Project folder:", BASE_DIR)
print("Dataset folder:", DATA_DIR)


# ============================================================
# LOAD DATASET
# ============================================================

csv_path = DATA_DIR / "PCOS_extended_dataset.csv"

if not csv_path.exists():

    raise FileNotFoundError(
        f"""
Dataset not found!

Expected location:
{csv_path}
"""
    )


print("\nLoading dataset...")

df = pd.read_csv(csv_path)


print("Dataset shape:", df.shape)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


print("\nColumns:")

print(
    df.columns.tolist()
)


# ============================================================
# TARGET COLUMN
# ============================================================

TARGET = "PCOS (Y/N)"


if TARGET not in df.columns:

    raise ValueError(
        f"""
Target column '{TARGET}' not found.

Available columns:
{df.columns.tolist()}
"""
    )


# ============================================================
# CLEAN TARGET
# ============================================================

print("\nCleaning target...")


# Convert target to string
df[TARGET] = (
    df[TARGET]
    .astype(str)
    .str.strip()
    .str.upper()
)


print(
    "Original target values:"
)

print(
    df[TARGET].value_counts(
        dropna=False
    )
)


# Support Y/N, Yes/No and 1/0
target_mapping = {

    "Y": 1,
    "YES": 1,
    "1": 1,
    "TRUE": 1,
    "POSITIVE": 1,

    "N": 0,
    "NO": 0,
    "0": 0,
    "FALSE": 0,
    "NEGATIVE": 0

}


df[TARGET] = (
    df[TARGET]
    .map(target_mapping)
)


# Remove rows with invalid target
df = df.dropna(
    subset=[TARGET]
)


df[TARGET] = (
    df[TARGET]
    .astype(int)
)


print(
    "\nFinal target distribution:"
)

print(
    df[TARGET].value_counts()
)


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]


# ============================================================
# REMOVE IDENTIFICATION COLUMNS
# ============================================================

columns_to_remove = [

    "Sl. No",

    "Patient File No.",

    "Blood Group"

]


for column in columns_to_remove:

    if column in X.columns:

        X = X.drop(
            columns=[column]
        )


# ============================================================
# CONVERT YES / NO FEATURES
# ============================================================

yes_no_columns = [

    "Pregnant(Y/N)",

    "Weight gain(Y/N)",

    "hair growth(Y/N)",

    "Skin darkening (Y/N)",

    "Hair loss(Y/N)",

    "Pimples(Y/N)",

    "Fast food (Y/N)",

    "Reg.Exercise(Y/N)"

]


yes_no_mapping = {

    "Y": 1,
    "YES": 1,
    "1": 1,

    "N": 0,
    "NO": 0,
    "0": 0

}


for column in yes_no_columns:

    if column in X.columns:

        X[column] = (

            X[column]
            .astype(str)
            .str.strip()
            .str.upper()
            .map(yes_no_mapping)

        )


# ============================================================
# CYCLE COLUMN
# ============================================================

if "Cycle(R/I)" in X.columns:

    X["Cycle(R/I)"] = (

        X["Cycle(R/I)"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({

            "R": 0,
            "REGULAR": 0,

            "I": 1,
            "IRREGULAR": 1

        })

    )


# ============================================================
# CONVERT REMAINING COLUMNS TO NUMERIC
# ============================================================

print(
    "\nConverting features to numeric..."
)


for column in X.columns:

    # If column is already numeric
    if pd.api.types.is_numeric_dtype(
        X[column]
    ):

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    else:

        # Clean text values
        cleaned = (

            X[column]
            .astype(str)
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .replace(
                {
                    "": np.nan,
                    "nan": np.nan,
                    "NaN": np.nan,
                    "NA": np.nan,
                    "N/A": np.nan,
                    "?": np.nan,
                    "--": np.nan
                }
            )

        )

        converted = pd.to_numeric(
            cleaned,
            errors="coerce"
        )

        # Only replace if useful numeric values exist
        if converted.notna().sum() > 0:

            X[column] = converted

        else:

            # Non-numeric column
            X[column] = np.nan


# ============================================================
# REMOVE COMPLETELY EMPTY COLUMNS
# ============================================================

before_columns = X.shape[1]


X = X.dropna(
    axis=1,
    how="all"
)


after_columns = X.shape[1]


print(
    "\nFeatures before cleaning:",
    before_columns
)

print(
    "Features after cleaning:",
    after_columns
)


# ============================================================
# CHECK FEATURE COUNT
# ============================================================

if X.shape[1] == 0:

    raise ValueError(
        """
No usable features were found.

Please check the dataset values.
"""
    )


print(
    "\nFinal feature count:",
    X.shape[1]
)


print(
    "\nFeatures used:"
)

print(
    X.columns.tolist()
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# RANDOM FOREST PIPELINE
# ============================================================

model = Pipeline([

    (
        "imputer",

        SimpleImputer(
            strategy="median"
        )

    ),

    (
        "classifier",

        RandomForestClassifier(

            n_estimators=300,

            max_depth=12,

            min_samples_split=4,

            random_state=42,

            class_weight="balanced",

            n_jobs=-1

        )

    )

])


# ============================================================
# TRAIN MODEL
# ============================================================

print(
    "\n========================================"
)

print(
    "Training Random Forest..."
)

print(
    "========================================"
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(
    X_test
)


probabilities = (
    model.predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)


roc_auc = roc_auc_score(
    y_test,
    probabilities
)


print(
    "\n========================================"
)

print(
    "MODEL PERFORMANCE"
)

print(
    "========================================"
)


print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


print(
    f"ROC-AUC: {roc_auc:.4f}"
)


print(
    "\nClassification Report:"
)


print(
    classification_report(
        y_test,
        predictions
    )
)


print(
    "\nConfusion Matrix:"
)


print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR /
    "pcos_ml_model.joblib"
)


features_path = (
    MODEL_DIR /
    "pcos_features.joblib"
)


joblib.dump(
    model,
    model_path
)


joblib.dump(
    list(X.columns),
    features_path
)


# ============================================================
# SAVE DATASET INFORMATION
# ============================================================

info = {

    "total_rows": len(df),

    "total_features": X.shape[1],

    "accuracy": accuracy,

    "roc_auc": roc_auc,

    "features": list(X.columns)

}


joblib.dump(
    info,
    MODEL_DIR /
    "model_info.joblib"
)


# ============================================================
# SUCCESS
# ============================================================

print(
    "\n========================================"
)

print(
    "ML TRAINING COMPLETED SUCCESSFULLY!"
)

print(
    "========================================"
)


print(
    "\nModel saved:"
)

print(
    model_path
)


print(
    "\nFeatures saved:"
)

print(
    features_path
)


print(
    "\nModel information saved:"
)

print(
    MODEL_DIR /
    "model_info.joblib"
)