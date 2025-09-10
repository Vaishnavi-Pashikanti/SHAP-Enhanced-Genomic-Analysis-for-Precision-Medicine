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

def train_and_save(csv_path, target_col="Overall Survival Status", out_dir="models"):
    print("[train] loading data...")
    df = load_data(csv_path)

    # Drop rows with missing target
    df = df[df[target_col].notna()]
    print(f"[train] After dropping missing target: shape={df.shape}")

    # Create binary target column
    df[target_col + "_bin"] = df[target_col].apply(map_survival_status)

    X = choose_features(df)
    y = df[target_col + "_bin"]

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # preprocess numeric + categorical separately
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

    # select top 50 features
    selector = SelectKBest(f_classif, k=50)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("selector", selector),
        ("clf", clf),
    ])

    # fit model
    pipe.fit(X_train, y_train)
    print(f"[prepare] Final train shape={X_train.shape}, test shape={X_test.shape}")
    print("[train] fitting RandomForest...")

    # save artifacts
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipe, f"{out_dir}/rf_pipeline.joblib")
    joblib.dump(clf, f"{out_dir}/rf_model.joblib")
    joblib.dump(preprocessor, f"{out_dir}/preprocessor.joblib")
    joblib.dump(selector, f"{out_dir}/selector.joblib")

    # save feature names before selection
    feature_names = list(X.columns)
    with open(f"{out_dir}/selected_feature_names.json", "w") as f:
        json.dump(feature_names[:50], f)

    print(f"[train] model + artifacts saved to {out_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--target", default="Overall Survival Status")
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    train_and_save(args.data_csv, target_col=args.target, out_dir=args.out_dir)