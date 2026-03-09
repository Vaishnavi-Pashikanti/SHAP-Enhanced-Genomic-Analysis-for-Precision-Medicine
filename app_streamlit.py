import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import os
import streamlit.components.v1 as components
import sys

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import matplotlib.pyplot as plt
from reportlab.platypus import Image

MODEL_DIR = "models"

@st.cache_resource
def load_pipeline(model_dir=MODEL_DIR):
    pipe = joblib.load(f"{model_dir}/rf_pipeline.joblib")
    return pipe

st.title("🔬 SHAP-Enhanced METABRIC Explorer")
st.markdown("Upload METABRIC clinical CSV or use the default file in `data/`")

uploaded = st.file_uploader("Upload CSV", type=['csv'])
# initialize df to ensure it's always defined
df = None
if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        st.error("Failed to read uploaded CSV. Ensure it's a valid CSV file.")
        st.stop()
else:
    default_path = os.path.join("data", "metabric_clinical.csv")
    if os.path.exists(default_path):
        try:
            df = pd.read_csv(default_path)
        except Exception:
            st.error(f"Failed to read default CSV at {default_path}.")
            st.stop()
    else:
        df = None

# If no dataframe loaded, show error and exit (works with streamlit and plain python)
if df is None:
    msg = "No data loaded. Upload a METABRIC CSV or place the default file at data/metabric_clinical.csv"
    try:
        st.error(msg)
        st.stop()
    except Exception:
        print(msg)
        sys.exit(1)

# --- FIX: Ensure required columns exist for the pipeline ---
for col in ['Overall Survival Status', 'Overall Survival Status_bin']:
    if col not in df.columns:
        df[col] = np.nan

st.write("Dataset preview:")
st.dataframe(df.head())

pipe = load_pipeline()

st.sidebar.header("Prediction options")
idx = st.sidebar.number_input("Patient row index", min_value=0, max_value=len(df)-1, value=0, step=1)
patient_row = df.iloc[[idx]]

# Import your feature selection and mapping
from data_prep import choose_features

X = choose_features(df)
patient_features = choose_features(patient_row)

# Predict using the pipeline
prob = pipe.predict_proba(patient_features)[0, 1] if hasattr(pipe, 'predict_proba') else pipe.predict(patient_features)[0]
st.subheader("Prediction")
st.write(f"Predicted probability (deceased / positive class): **{prob:.3f}**")
st.write("Interpretation: higher value → higher predicted risk.")

# SHAP explanation
explainer = shap.TreeExplainer(pipe.named_steps["clf"])
# Preprocess for SHAP (preprocessor only, not selector)
preprocessor = pipe.named_steps["preprocessor"]
X_trans = preprocessor.transform(patient_features)

# Get feature names after preprocessing
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

shap_vals = explainer.shap_values(X_trans)
if isinstance(shap_vals, list) and len(shap_vals) > 1:
    sv = shap_vals[1]
else:
    sv = shap_vals[0] if isinstance(shap_vals, list) else shap_vals

# If sv[0] is 2D, select the positive class (column 1)
if sv[0].ndim == 2 and sv[0].shape[1] == 2:
    contribs = pd.Series(sv[0][:, 1], index=feature_names)
else:
    contribs = pd.Series(sv[0], index=feature_names)
st.subheader("Top feature contributors (local)")
st.write(pd.concat([contribs.sort_values(ascending=False)[:10], contribs.sort_values()[:10]]))

# Force plot: embed HTML
st.subheader("SHAP force plot (local explanation)")
# Robust base_value selection
expected_value = explainer.expected_value
if isinstance(expected_value, (list, tuple, np.ndarray)):
    base_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
else:
    base_value = expected_value

# Use only the positive class SHAP values for the force plot
if sv[0].ndim == 2 and sv[0].shape[1] == 2:
    shap_values_1d = sv[0][:, 1]
else:
    shap_values_1d = sv[0]

force_html = shap.plots.force(base_value, shap_values_1d, feature_names=feature_names, matplotlib=False)
shap.save_html("force_plot.html", force_html)
with open("force_plot.html", "r", encoding="utf-8") as f:
    components.html(f.read(), height=450, scrolling=True)



# from weasyprint import HTML
# import tempfile
# --- Build PDF report ---
# --- Generate SHAP force plot as an image ---
force_plot_path = os.path.join(tempfile.gettempdir(), "force_plot.png")

# SHAP force plots are tricky in matplotlib=False mode (they return HTML/JS).
# So we use matplotlib=True to generate a static PNG.
shap.force_plot(
    base_value,
    shap_values_1d,
    feature_names=feature_names,
    matplotlib=True,
    show=False
)
plt.savefig(force_plot_path, bbox_inches="tight", dpi=150)
plt.close()

# --- Build PDF with ReportLab ---
styles = getSampleStyleSheet()
story = []
story.append(Paragraph("METABRIC Patient Prediction Report", styles["Title"]))
story.append(Spacer(1, 12))

# Prediction
story.append(Paragraph(f"Prediction Probability: <b>{prob:.3f}</b>", styles["Normal"]))
story.append(Paragraph("Interpretation: Higher value → higher predicted risk.", styles["Normal"]))
story.append(Spacer(1, 12))

# ...existing code...

# --- Predict using the pipeline (use pipe, not raw clf on raw df) ---
# get prediction and probabilities from the full pipeline (preprocessor + classifier)
predicted = pipe.predict(patient_features)[0]
probs = pipe.predict_proba(patient_features)[0] if hasattr(pipe, "predict_proba") else None

clf = pipe.named_steps["clf"]
classes = getattr(clf, "classes_", None)

# Map numeric class to human label (adjust if your labels differ)
if classes is not None:
    if set(classes) >= {0, 1}:
        label_map = {0: "Living", 1: "Deceased"}
    else:
        label_map = {c: f"Class {c}" for c in classes}
else:
    label_map = {predicted: str(predicted)}

pred_label = label_map.get(predicted, str(predicted))

# probability for "deceased" class if available
prob_deceased = None
if probs is not None and classes is not None:
    try:
        idx_deceased = list(classes).index(1)
        prob_deceased = probs[idx_deceased]
    except Exception:
        # fallback: probability of predicted class
        try:
            prob_deceased = probs[list(classes).index(predicted)]
        except Exception:
            prob_deceased = None

st.subheader("Prediction")
st.write(f"Predicted outcome: **{pred_label}**")
if prob_deceased is not None:
    st.write(f"Estimated probability (Deceased / positive class): **{prob_deceased:.3f}**")
else:
    st.write("Probability score not available for this estimator.")

st.write("Interpretation: this is a model-based risk estimate for overall survival (not a clinical diagnosis).")

# ...existing code...

# Top contributors
story.append(Paragraph("Top Feature Contributors", styles["Heading2"]))
contrib_table = pd.concat([
    contribs.sort_values(ascending=False)[:10],
    contribs.sort_values()[:10]
])
for feat, val in contrib_table.items():
    story.append(Paragraph(f"{feat}: {val:.4f}", styles["Normal"]))
story.append(Spacer(1, 12))

# Insert SHAP force plot
story.append(Paragraph("SHAP Force Plot (Local Explanation)", styles["Heading2"]))
story.append(Image(force_plot_path, width=500, height=200))  # adjust size as needed

# Save PDF
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
    doc = SimpleDocTemplate(tmp_pdf.name)
    doc.build(story)
    tmp_pdf.seek(0)
    pdf_bytes = tmp_pdf.read()

st.download_button(
    label="Download Full Patient Report (PDF)",
    data=pdf_bytes,
    file_name="metabric_patient_report.pdf",
    mime="application/pdf"
)