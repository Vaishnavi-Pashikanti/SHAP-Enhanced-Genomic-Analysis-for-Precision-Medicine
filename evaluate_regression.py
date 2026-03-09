import argparse
import os
import json

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, median_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)


def _find_target(df, task_hint=None):
    if task_hint == "regression":
        cand = [c for c in df.columns if "Month" in c or "month" in c.lower() or "survival" in c.lower()]
    elif task_hint == "classification":
        cand = [c for c in df.columns if "status" in c.lower() or "survival" in c.lower() and "month" not in c.lower()]
    else:
        cand = df.columns.tolist()
    # heuristics
    for c in cand:
        if "month" in c.lower() or "overall survival (months)" in c.lower():
            return c, "regression"
    for c in df.columns:
        low = c.lower()
        if ("status" in low and ("surviv" in low or "overall" in low)) or low == "overall survival status":
            return c, "classification"
    # fallback
    for c in df.columns:
        if "month" in c.lower():
            return c, "regression"
    raise ValueError("Cannot determine target column automatically. Provide a CSV with a clear regression/classification target.")


def evaluate_regression(df, pipe, out_dir="models", cv=5, thresholds=(6, 12), group_cols=None):
    os.makedirs(out_dir, exist_ok=True)
    # find target
    target_col, _ = _find_target(df, task_hint="regression")
    print(f"[reg] using target: {target_col}")

    df = df.loc[~df[target_col].isna()].reset_index(drop=True)
    X = None
    try:
        from data_prep import choose_features
        X = choose_features(df)
    except Exception:
        X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col].astype(float).to_numpy()

    # predictions (pipeline)
    try:
        y_pred = pipe.predict(X)
    except Exception:
        # manual transform + estimator
        pre = pipe.named_steps.get("preprocessor", None)
        if pre is None:
            raise
        X_trans = pre.transform(X)
        est = None
        for name, step in pipe.named_steps.items():
            if name not in ("preprocessor", "selector", "scaler"):
                est = step
        y_pred = est.predict(X_trans)

    y = np.asarray(y).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y.shape[0] != y_pred.shape[0]:
        m = min(y.shape[0], y_pred.shape[0])
        y, y_pred = y[:m], y_pred[:m]

    # basic metrics
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    mdae = median_absolute_error(y, y_pred)

    # percent within thresholds
    abs_err = np.abs(y - y_pred)
    pct_within = {f"pct_within_{t}": float((abs_err <= t).mean()) for t in thresholds}

    metrics = {
        "mae": float(mae), "mse": float(mse), "rmse": float(rmse), "r2": float(r2), "mdae": float(mdae)
    }
    metrics.update(pct_within)

    # cross-validated estimates (CV on pipeline if it supports fit/predict)
    try:
        kf = KFold(n_splits=cv, shuffle=True, random_state=0)
        # cross-validated predictions
        y_cv_pred = cross_val_predict(pipe, X, y, cv=kf, n_jobs=-1)
        cv_mae = mean_absolute_error(y, y_cv_pred)
        cv_rmse = np.sqrt(mean_squared_error(y, y_cv_pred))
        cv_r2 = r2_score(y, y_cv_pred)
        metrics.update({
            "cv_mae_mean": float(cv_mae),
            "cv_rmse_mean": float(cv_rmse),
            "cv_r2": float(cv_r2)
        })
    except Exception as e:
        print(f"[reg] CV failed: {e}")

    # outlier analysis
    resid = y - y_pred
    df_out = df.copy()
    df_out["_pred"] = y_pred
    df_out["_resid"] = resid
    outliers = df_out.loc[np.abs(resid).argsort()[::-1]].head(20)
    outliers.to_csv(os.path.join(out_dir, "reg_outliers_top20.csv"), index=False)

    # plots
    scatter_path = os.path.join(out_dir, "reg_true_vs_pred.png")
    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_pred, alpha=0.6)
    mn = min(y.min(), y_pred.min()); mx = max(y.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], color="red", linestyle="--")
    plt.xlabel("True"); plt.ylabel("Predicted"); plt.title("True vs Predicted")
    plt.tight_layout(); plt.savefig(scatter_path, dpi=150); plt.close()

    resid_hist = os.path.join(out_dir, "reg_residuals.png")
    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=40, color="C0", alpha=0.8)
    plt.axvline(0, color="k", linestyle="--"); plt.title("Residuals distribution")
    plt.xlabel("Residual (True - Pred)"); plt.tight_layout(); plt.savefig(resid_hist, dpi=150); plt.close()

    resid_vs_pred = os.path.join(out_dir, "reg_residuals_vs_pred.png")
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, resid, alpha=0.6)
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Predicted"); plt.ylabel("Residual (True - Pred)"); plt.title("Residuals vs Predicted")
    plt.tight_layout(); plt.savefig(resid_vs_pred, dpi=150); plt.close()

    # subgroup metrics if requested
    group_results = {}
    if group_cols:
        for col in group_cols:
            if col not in df.columns:
                continue
            grp = df.groupby(col)
            grp_stats = {}
            for name, sub in grp:
                if len(sub) < 10:
                    continue
                try:
                    Xg = X.loc[sub.index]
                except Exception:
                    Xg = X.iloc[sub.index]
                yg = sub[target_col].astype(float).to_numpy()
                try:
                    yp = pipe.predict(Xg)
                except Exception:
                    yp = pipe.predict(Xg)
                grp_stats[str(name)] = {
                    "n": int(len(yg)),
                    "mae": float(mean_absolute_error(yg, yp)),
                    "rmse": float(np.sqrt(mean_squared_error(yg, yp))),
                    "r2": float(r2_score(yg, yp))
                }
            group_results[col] = grp_stats

    # save metrics and artifacts
    with open(os.path.join(out_dir, "regression_eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "regression_group_metrics.json"), "w") as f:
        json.dump(group_results, f, indent=2)

    return metrics, scatter_path, resid_hist, resid_vs_pred, os.path.join(out_dir, "reg_outliers_top20.csv")


def evaluate_classification(df, pipe, out_dir="models", cv=5):
    os.makedirs(out_dir, exist_ok=True)
    # find classification target
    # heuristics: look for binary survival status column
    target_col = None
    for c in df.columns:
        if "status" in c.lower() or ("survival" in c.lower() and "month" not in c.lower()):
            target_col = c
            break
    if target_col is None:
        # try 'Overall Survival Status (binary)' fallback
        raise ValueError("No classification target column found.")
    print(f"[clf] using target: {target_col}")
    df = df.loc[~df[target_col].isna()].reset_index(drop=True)
    y = df[target_col].to_numpy()
    try:
        from data_prep import choose_features
        X = choose_features(df)
    except Exception:
        X = df.drop(columns=[target_col], errors="ignore")

    # predictions
    try:
        y_pred = pipe.predict(X)
        y_proba = pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else None
    except Exception:
        pre = pipe.named_steps.get("preprocessor", None)
        X_trans = pre.transform(X)
        est = None
        for name, step in pipe.named_steps.items():
            if name not in ("preprocessor", "selector", "scaler"):
                est = step
        y_pred = est.predict(X_trans)
        try:
            y_proba = est.predict_proba(X_trans)[:, 1]
        except Exception:
            y_proba = None

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    roc = roc_auc_score(y, y_proba) if y_proba is not None else None

    metrics = {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}
    if roc is not None:
        metrics["roc_auc"] = float(roc)

    # cross-val
    try:
        kf = KFold(n_splits=cv, shuffle=True, random_state=0)
        y_cv_pred = cross_val_predict(pipe, X, y, cv=kf, n_jobs=-1, method="predict")
        acc_cv = accuracy_score(y, y_cv_pred)
        metrics.update({"cv_accuracy": float(acc_cv)})
    except Exception as e:
        print(f"[clf] CV failed: {e}")

    # confusion matrix and report
    cm = confusion_matrix(y, y_pred)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(y, y_pred))

    # save metrics
    with open(os.path.join(out_dir, "classification_eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model", default=os.path.join("models", "rf_reg_pipeline.joblib"))
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--task", choices=["auto", "regression", "classification"], default="auto")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--group-cols", default="", help="comma-separated group columns for subgroup metrics")
    parser.add_argument("--thresholds", default="6,12", help="comma-separated months for pct-within thresholds")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    pipe = joblib.load(args.model)

    task = args.task
    if task == "auto":
        # try to detect
        try:
            _, detected = _find_target(df, None)
            task = detected
        except Exception:
            task = "regression"

    if task == "regression":
        thresholds = tuple(int(x) for x in args.thresholds.split(",") if x.strip())
        groups = [c.strip() for c in args.group_cols.split(",") if c.strip()]
        res = evaluate_regression(df, pipe, out_dir=args.out_dir, cv=args.cv, thresholds=thresholds, group_cols=groups or None)
        print("[main] regression done. Artifacts:", res[1:])
    else:
        res = evaluate_classification(df, pipe, out_dir=args.out_dir, cv=args.cv)
        print("[main] classification done. metrics saved.")

if __name__ == "__main__":
    main()