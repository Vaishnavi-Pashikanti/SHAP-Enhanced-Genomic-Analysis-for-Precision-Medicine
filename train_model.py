# # train_model.py
# import argparse
# import os
# import json
# import joblib
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.impute import SimpleImputer

# from data_prep import load_data, choose_features, map_survival_status

# # -------------------------------
# # Reusable functions
# # -------------------------------

# def prepare_data(df, target_col="Overall Survival Status", test_size=0.2, random_state=42, k_features=50):
#     """Prepare train-test split, preprocessing pipeline, and selector."""
#     # Drop rows with missing target
#     df = df[df[target_col].notna()].copy()
#     df[target_col + "_bin"] = df[target_col].apply(map_survival_status)

#     X = choose_features(df)
#     y = df[target_col + "_bin"]

#     # split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state, stratify=y
#     )

#     # numeric + categorical
#     num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
#     cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", Pipeline([
#                 ("imputer", SimpleImputer(strategy="median")),
#                 ("scaler", StandardScaler())
#             ]), num_cols),

#             ("cat", Pipeline([
#                 ("imputer", SimpleImputer(strategy="most_frequent")),
#                 ("encoder", OneHotEncoder(handle_unknown="ignore"))
#             ]), cat_cols),
#         ]
#     )

#     # feature selector
#     selector = SelectKBest(f_classif, k=k_features)

#     # classifier
#     clf = RandomForestClassifier(n_estimators=200, random_state=random_state)

#     # full pipeline
#     pipe = Pipeline(steps=[
#         ("preprocessor", preprocessor),
#         ("selector", selector),
#         ("clf", clf),
#     ])

#     return X_train, X_test, y_train, y_test, X.columns.tolist(), pipe, preprocessor, selector, clf


# def train_and_save(csv_path, target_col="Overall Survival Status", out_dir="models"):
#     print("[train] loading data...")
#     df = load_data(csv_path)

#     X_train, X_test, y_train, y_test, feature_names, pipe, preprocessor, selector, clf = prepare_data(df, target_col)

#     print(f"[prepare] Final train shape={X_train.shape}, test shape={X_test.shape}")
#     print("[train] fitting RandomForest...")

#     pipe.fit(X_train, y_train)

#     # save artifacts
#     os.makedirs(out_dir, exist_ok=True)
#     joblib.dump(pipe, f"{out_dir}/rf_pipeline.joblib")
#     joblib.dump(clf, f"{out_dir}/rf_model.joblib")
#     joblib.dump(preprocessor, f"{out_dir}/preprocessor.joblib")
#     joblib.dump(selector, f"{out_dir}/selector.joblib")

#     with open(f"{out_dir}/selected_feature_names.json", "w") as f:
#         json.dump(feature_names, f)

#     print(f"[train] model + artifacts saved to {out_dir}/")

#     return pipe, (X_train, X_test, y_train, y_test)


# # -------------------------------
# # CLI entrypoint
# # -------------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data-csv", required=True)
#     parser.add_argument("--target", default="Overall Survival Status")
#     parser.add_argument("--out-dir", default="models")
#     args = parser.parse_args()

#     train_and_save(args.data_csv, target_col=args.target, out_dir=args.out_dir)


# ...existing code...
import argparse
import os
import json
import joblib
import glob
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_prep import load_data, choose_features

def prepare_data(df, target_col="Overall Survival (Months)", test_size=0.2, random_state=42):
    """
    Prepare X, y and train/test split for regression.
    Uses choose_features(df) from data_prep to select input features.
    Drops rows with missing target.
    """
    df = df[df[target_col].notna()].copy()
    if df.shape[0] == 0:
        raise ValueError(f"No rows with non-missing target column: {target_col}")

    X = choose_features(df)
    y = df[target_col].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # identify column types
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols),
        ],
        remainder="drop"
    )

    return X_train, X_test, y_train, y_test, preprocessor

def train_and_save(csv_path, target_col="Overall Survival (Months)", out_dir="models", random_state=42, n_estimators=200):
    print("[train] loading data from:", csv_path)
    df = load_data(csv_path)

    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df, target_col=target_col, random_state=random_state)

    # Build pipeline: preprocessor + regressor
    reg = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("reg", reg),
    ])

    print(f"[train] training on {X_train.shape[0]} rows, validating on {X_test.shape[0]} rows...")
    pipe.fit(X_train, y_train)

    # Predictions and metrics
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"[eval] MAE: {mae:.3f}   RMSE: {rmse:.3f}   R2: {r2:.3f}")

    # Save artifacts
    os.makedirs(out_dir, exist_ok=True)
    pipeline_path = os.path.join(out_dir, "rf_reg_pipeline.joblib")
    model_path = os.path.join(out_dir, "rf_reg_model.joblib")
    preproc_path = os.path.join(out_dir, "preprocessor.joblib")

    joblib.dump(pipe, pipeline_path)
    # save regressor object and preprocessor separately for convenience
    joblib.dump(pipe.named_steps["reg"], model_path)
    joblib.dump(pipe.named_steps["preprocessor"], preproc_path)

    # write a small metadata file
    meta = {
        "pipeline": pipeline_path,
        "model": model_path,
        "preprocessor": preproc_path,
        "target": target_col,
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[train] saved pipeline -> {pipeline_path}")
    print(f"[train] saved model -> {model_path}")
    print(f"[train] saved preprocessor -> {preproc_path}")
    print(f"[train] metadata -> {os.path.join(out_dir, 'metadata.json')}")

    return pipe, (X_train, X_test, y_train, y_test)

# -------------------------------
# CLI entrypoint
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", default=None, help="Path to data CSV (default: data/metabric_clinical.csv or first CSV found in data/)")
    parser.add_argument("--target", default="Overall Survival (Months)")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    # determine csv path if not provided
    if args.data_csv:
        csv_path = args.data_csv
    else:
        default_path = os.path.join("data", "metabric_clinical.csv")
        if os.path.exists(default_path):
            csv_path = default_path
        else:
            csv_files = glob.glob(os.path.join("data", "*.csv"))
            if len(csv_files) == 1:
                csv_path = csv_files[0]
            elif len(csv_files) > 1:
                csv_path = csv_files[0]
                print(f"[warn] multiple CSVs found in data/, using first: {csv_path}")
            else:
                print("[error] no CSV file found in data/ and --data-csv not provided. Place your CSV in the data/ folder or pass --data-csv")
                sys.exit(1)

    print(f"[info] Using data CSV: {csv_path}")
    train_and_save(csv_path, target_col=args.target, out_dir=args.out_dir, random_state=args.random_state, n_estimators=args.n_estimators)
# ...existing code...