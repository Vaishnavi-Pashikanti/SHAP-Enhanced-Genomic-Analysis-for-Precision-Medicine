# train_model.py
import argparse
import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from data_prep import load_data, choose_features, map_survival_status

# -------------------------------
# Reusable functions
# -------------------------------

def prepare_data(df, target_col="Overall Survival Status", test_size=0.2, random_state=42, k_features=50):
    """Prepare train-test split, preprocessing pipeline, and selector."""
    # Drop rows with missing target
    df = df[df[target_col].notna()].copy()
    df[target_col + "_bin"] = df[target_col].apply(map_survival_status)

    X = choose_features(df)
    y = df[target_col + "_bin"]

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # numeric + categorical
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),

            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ]
    )

    # feature selector
    selector = SelectKBest(f_classif, k=k_features)

    # classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state)

    # full pipeline
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("selector", selector),
        ("clf", clf),
    ])

    return X_train, X_test, y_train, y_test, X.columns.tolist(), pipe, preprocessor, selector, clf


def train_and_save(csv_path, target_col="Overall Survival Status", out_dir="models"):
    print("[train] loading data...")
    df = load_data(csv_path)

    X_train, X_test, y_train, y_test, feature_names, pipe, preprocessor, selector, clf = prepare_data(df, target_col)

    print(f"[prepare] Final train shape={X_train.shape}, test shape={X_test.shape}")
    print("[train] fitting RandomForest...")

    pipe.fit(X_train, y_train)

    # save artifacts
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipe, f"{out_dir}/rf_pipeline.joblib")
    joblib.dump(clf, f"{out_dir}/rf_model.joblib")
    joblib.dump(preprocessor, f"{out_dir}/preprocessor.joblib")
    joblib.dump(selector, f"{out_dir}/selector.joblib")

    with open(f"{out_dir}/selected_feature_names.json", "w") as f:
        json.dump(feature_names, f)

    print(f"[train] model + artifacts saved to {out_dir}/")

    return pipe, (X_train, X_test, y_train, y_test)


# -------------------------------
# CLI entrypoint
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--target", default="Overall Survival Status")
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    train_and_save(args.data_csv, target_col=args.target, out_dir=args.out_dir)
