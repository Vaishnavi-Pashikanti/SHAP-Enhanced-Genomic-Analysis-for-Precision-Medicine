# explain_shap.py
import joblib
import argparse
import shap
import pandas as pd
import numpy as np  # <-- Add this import!
from data_prep import load_data, choose_features, map_survival_status
import shap.plots

def explain_patient(csv_path, patient_idx=0, model_dir="models", target_col="Overall Survival Status"):
    df = load_data(csv_path)
    df[target_col + "_bin"] = df[target_col].apply(map_survival_status)

    # load pipeline
    pipe = joblib.load(f"{model_dir}/rf_pipeline.joblib")

    # get input features (same as training)
    X = choose_features(df)

    # pick patient row
    if isinstance(patient_idx, int):
        patient_row = X.iloc[[patient_idx]]
    else:
        patient_row = X.loc[[patient_idx]]

    # prediction
    prob = pipe.predict_proba(patient_row)[0, 1]
    pred = int(prob >= 0.5)
    print(f"[explain] Predicted class: {pred} (prob={prob:.3f})")

    # Get preprocessor from pipeline
    preprocessor = pipe.named_steps["preprocessor"]

    # Transform patient row through preprocessor only (not selector)
    X_trans = preprocessor.transform(patient_row)

    # Get feature names after preprocessing only
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if hasattr(trans, 'get_feature_names_out'):
                feats = trans.get_feature_names_out(cols)
            elif hasattr(trans, 'get_feature_names'):
                feats = trans.get_feature_names(cols)
            else:
                feats = cols
            feature_names.extend(feats)

    # SHAP explanation
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    shap_values = explainer.shap_values(X_trans)
    # Handle binary vs multiclass
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_pos = shap_values[1]
    else:
        shap_pos = shap_values[0] if isinstance(shap_values, list) else shap_values

    # Debug shapes
    print("X_trans shape:", X_trans.shape)
    print("shap_pos shape:", shap_pos.shape)
    print("feature_names length:", len(feature_names))

    contribs = pd.Series(shap_pos[0, :, 1], index=feature_names)

    print("Top positive contributors (push to class=1):")
    print(contribs.sort_values(ascending=False).head(10))
    print("\nTop negative contributors (push to class=0):")
    print(contribs.sort_values().head(10))

    # save interactive force plot
    # For binary classification, use class 1 base value and SHAP values
    # Robust base_value selection
    if isinstance(explainer.expected_value, (list, tuple, np.ndarray)):
        if len(explainer.expected_value) > 1:
            base_value = explainer.expected_value[1]
        else:
            base_value = explainer.expected_value[0]
    else:
        base_value = explainer.expected_value
    shap_values_for_plot = shap_pos[0, :, 1]

    # Create force plot using new API
    force_plot = shap.plots.force(base_value, shap_values_for_plot, feature_names=feature_names, matplotlib=False)

    # Save as HTML
    shap.save_html(f"{model_dir}/patient_{patient_idx}_force.html", force_plot)
    print(f"[explain] force plot saved to {model_dir}/patient_{patient_idx}_force.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    explain_patient(args.csv, patient_idx=args.idx, model_dir=args.model_dir)