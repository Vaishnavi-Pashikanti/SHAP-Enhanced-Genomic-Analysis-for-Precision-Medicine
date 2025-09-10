# evaluate_model.py
import argparse
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, RocCurveDisplay
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

from data_prep import load_data, map_survival_status, choose_features


def evaluate(csv_path, target_col, model_dir, cv_folds=5):
    print("[eval] loading data...")
    df = load_data(csv_path)

    # Clean target
    df[target_col + "_bin"] = df[target_col].apply(map_survival_status)
    df = df.dropna(subset=[target_col + "_bin"]).reset_index(drop=True)

    y = df[target_col + "_bin"].astype(int)
    X = choose_features(df)

    print("[eval] loading trained pipeline...")
    pipe = joblib.load(os.path.join(model_dir, "rf_pipeline.joblib"))

    # ------------------------
    # Test set evaluation
    # ------------------------
    y_pred = pipe.predict(X)
    y_prob = pipe.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)

    print("\n[eval] 📊 Metrics (Full Dataset)")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC-AUC : {auc:.3f}\n")
    print("Classification Report:")
    print(classification_report(y, y_pred))

    # Save confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Living", "Deceased"])
    plt.yticks([0, 1], ["Living", "Deceased"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))
    plt.close()
    print(f"[eval] Confusion matrix saved to {model_dir}/confusion_matrix.png")

    # Save ROC curve
    RocCurveDisplay.from_predictions(y, y_prob)
    plt.savefig(os.path.join(model_dir, "roc_curve.png"))
    plt.close()
    print(f"[eval] ROC curve saved to {model_dir}/roc_curve.png")

    # ------------------------
    # Cross-validation (CV)
    # ------------------------
    print(f"\n[eval] Running {cv_folds}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    cv_acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")

    print(f"[eval] CV Accuracy: mean={cv_acc.mean():.3f}, std={cv_acc.std():.3f}")
    print(f"[eval] CV ROC-AUC : mean={cv_auc.mean():.3f}, std={cv_auc.std():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--target", default="Overall Survival Status")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    evaluate(args.data_csv, args.target, args.model_dir, cv_folds=args.cv_folds)
