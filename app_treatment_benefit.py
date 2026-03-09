"""
app_treatment_benefit.py

Streamlit application for precision medicine treatment benefit estimation.

Features:
- Interactive patient selection
- Counterfactual treatment scenario analysis
- SHAP-based explainability
- Treatment benefit ranking
- Clinical guideline context (RAG-ready)
- Decision-support report generation
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from typing import Optional
import sys
import os

# Handle path resolution for modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from treatment_benefit_estimator import (
    TreatmentBenefitEstimator,
    DecisionSupportReport,
    process_patient_for_treatment_benefit
)
from data_prep import load_data, choose_features, map_survival_status

# =============================================================================
# PAGE CONFIG & STATE
# =============================================================================

st.set_page_config(
    page_title="Precision Medicine: Treatment Benefit Estimator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "estimator" not in st.session_state:
    st.session_state.estimator = None
if "df" not in st.session_state:
    st.session_state.df = None
if "feature_names" not in st.session_state:
    st.session_state.feature_names = None
if "selected_patient_idx" not in st.session_state:
    st.session_state.selected_patient_idx = None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@st.cache_resource
def load_estimator():
    """Load treatment benefit estimator."""
    return TreatmentBenefitEstimator()


@st.cache_data
def load_patient_data():
    """Load and preprocess METABRIC data."""
    # Resolve project root robustly (works when this file is at repo root)
    module_dir = Path(__file__).parent
    project_root = module_dir if (module_dir / "data").exists() or (module_dir / "models").exists() else module_dir.parent

    data_path = project_root / "data" / "brca_metabric_clinical_data.csv"
    feature_names_path = project_root / "models" / "selected_feature_names.json"
    
    df = load_data(str(data_path))
    df["Overall Survival Status_bin"] = df["Overall Survival Status"].apply(map_survival_status)
    
    with open(str(feature_names_path), "r") as f:
        feature_names = json.load(f)
    
    return df, feature_names


def format_treatment_scenario(scenario_dict):
    """Format treatment scenario for display."""
    treatments = []
    if scenario_dict["chemotherapy"]:
        treatments.append("Chemotherapy")
    if scenario_dict["hormone_therapy"]:
        treatments.append("Hormone Therapy")
    if scenario_dict["radiotherapy"]:
        treatments.append("Radiotherapy")
    
    return ", ".join(treatments) if treatments else "No Treatment"


def get_benefit_color(category: str) -> str:
    """Map benefit category to color."""
    if "High" in category:
        return "🟢"
    elif "Moderate" in category:
        return "🟡"
    else:
        return "🔴"


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.markdown("---")
    st.subheader("Data & Models")
    
    if st.button("Load Data & Models", use_container_width=True):
        with st.spinner("Loading..."):
            st.session_state.estimator = load_estimator()
            st.session_state.df, st.session_state.feature_names = load_patient_data()
            st.session_state.df = st.session_state.estimator.filter_alive_patients(st.session_state.df)
            st.success(f"✅ Loaded {len(st.session_state.df)} alive patients")
    
    st.markdown("---")
    st.subheader("Patient Selection")
    
    if st.session_state.df is not None:
        # Patient list
        patient_options = [
            f"ID: {row['Patient ID']} | Age: {row['Age at Diagnosis']:.1f} | Stage: {row['Tumor Stage']}"
            for _, row in st.session_state.df.head(50).iterrows()
        ]
        
        selected = st.selectbox(
            "Select Patient:",
            range(min(50, len(st.session_state.df))),
            format_func=lambda i: patient_options[i],
            key="patient_selector"
        )
        
        st.session_state.selected_patient_idx = selected
        
        # Show patient info
        patient_row = st.session_state.df.iloc[selected]
        with st.expander("Patient Details"):
            st.write(f"**Patient ID:** {patient_row['Patient ID']}")
            st.write(f"**Age:** {patient_row['Age at Diagnosis']:.1f} years")
            st.write(f"**Tumor Stage:** {patient_row['Tumor Stage']}")
            st.write(f"**ER Status:** {patient_row['ER Status']}")
            st.write(f"**HER2 Status:** {patient_row['HER2 Status']}")
            st.write(f"**Pam50 Subtype:** {patient_row['Pam50 + Claudin-low subtype']}")
            st.write(f"**Grade:** {patient_row['Neoplasm Histologic Grade']}")
    
    st.markdown("---")
    st.subheader("Analysis Settings")
    
    weight_prob = st.slider(
        "Weight: Survival Probability",
        0.0, 1.0, 0.5, 0.1,
        help="Weight for survival probability in benefit ranking"
    )
    weight_months = 1.0 - weight_prob
    st.write(f"Weight: Survival Months = {weight_months:.1f}")


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.title("🏥 Precision Medicine: Treatment Benefit Estimation")
st.markdown("""
**Decision Support for Alive Breast Cancer Patients**

This system estimates expected survival outcomes under alternative treatment scenarios
to support clinical decision-making. **Not a treatment prescription.**
""")

# Check if data is loaded
if st.session_state.estimator is None or st.session_state.df is None:
    st.warning("⚠️ Please load data and models from the sidebar first.")
    st.stop()

# Get selected patient
patient_idx = st.session_state.selected_patient_idx
patient_row = st.session_state.df.iloc[patient_idx]
patient_id = patient_row["Patient ID"]

# Extract features
X = choose_features(st.session_state.df)
patient_features = X.iloc[patient_idx]

# Estimate benefits
with st.spinner("Analyzing treatment scenarios..."):
    benefits, baseline_pred = st.session_state.estimator.estimate_treatment_benefits(
        patient_features
    )

# Create report
report = DecisionSupportReport(patient_id, benefits, baseline_pred)
summary = report.generate_summary()

# =========================================================================
# TAB 1: DECISION SUPPORT
# =========================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Decision Support",
    "📈 Benefit Comparison",
    "🔬 SHAP Explainability",
    "📋 Report"
])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Baseline Prediction (No Treatment)")
        st.metric(
            "Predicted Survival Probability",
            f"{baseline_pred.survival_probability:.1%}",
            help="Probability of being alive"
        )
    
    with col2:
        st.metric(
            "Predicted Survival Time",
            f"{baseline_pred.predicted_survival_months:.1f} months",
            help="Expected survival duration"
        )
    
    st.markdown("---")
    st.subheader("Treatment Scenarios Ranked by Benefit")
    
    # Rank by weighted benefit
    ranked = sorted(
        benefits,
        key=lambda b: (
            weight_prob * b.survival_probability_benefit +
            weight_months * (b.survival_months_benefit / 120.0)
        ),
        reverse=True
    )
    
    # Display as cards
    for rank, benefit in enumerate(ranked[:4], 1):
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            st.markdown(f"### Rank {rank}")
        
        with col2:
            treatment_str = format_treatment_scenario(benefit.scenario.to_dict())
            st.markdown(f"**{treatment_str}**")
            
            # Benefit metrics
            prob_delta = benefit.survival_probability_benefit
            months_delta = benefit.survival_months_benefit
            
            col_prob, col_months = st.columns(2)
            with col_prob:
                delta_text = "+" if prob_delta >= 0 else ""
                st.caption(f"Survival Prob: {delta_text}{prob_delta:.1%}")
            with col_months:
                delta_text = "+" if months_delta >= 0 else ""
                st.caption(f"Survival Time: {delta_text}{months_delta:.1f} months")
        
        with col3:
            color = get_benefit_color(benefit.benefit_category())
            st.markdown(f"{color} {benefit.benefit_category()}")


# =========================================================================
# TAB 2: BENEFIT COMPARISON
# =========================================================================

with tab2:
    st.subheader("Benefit Comparison Matrix")
    
    benefits_df = report.to_dataframe()
    
    # Create visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Survival Probability Benefit**")
        prob_data = benefits_df.sort_values(
            "survival_probability_benefit",
            ascending=False
        ).head(8)
        
        st.bar_chart(
            data=prob_data.set_index("scenario_id")["survival_probability_benefit"],
            use_container_width=True
        )
    
    with col2:
        st.write("**Survival Months Benefit**")
        months_data = benefits_df.sort_values(
            "survival_months_benefit",
            ascending=False
        ).head(8)
        
        st.bar_chart(
            data=months_data.set_index("scenario_id")["survival_months_benefit"],
            use_container_width=True
        )
    
    st.markdown("---")
    st.subheader("All Scenarios Table")
    
    # Format for display
    display_df = benefits_df.copy()
    display_df["Treatment"] = display_df.apply(
        lambda row: format_treatment_scenario({
            "chemotherapy": row["chemotherapy"],
            "hormone_therapy": row["hormone_therapy"],
            "radiotherapy": row["radiotherapy"]
        }),
        axis=1
    )
    display_df["Surv. Prob. Benefit"] = display_df["survival_probability_benefit"].apply(lambda x: f"{x:+.1%}")
    display_df["Surv. Time Benefit"] = display_df["survival_months_benefit"].apply(lambda x: f"{x:+.1f} mo")
    
    st.dataframe(
        display_df[["Treatment", "Surv. Prob. Benefit", "Surv. Time Benefit", "benefit_category"]],
        use_container_width=True,
        hide_index=True
    )


# =========================================================================
# TAB 3: SHAP EXPLAINABILITY
# =========================================================================

with tab3:
    st.subheader("SHAP Feature Importance")
    st.markdown("""
    **Local SHAP explanations** show which patient features drive survival predictions.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Select scenario for SHAP analysis:**")
        scenario_select = st.selectbox(
            "Scenario",
            options=range(len(ranked[:4])),
            format_func=lambda i: format_treatment_scenario(ranked[i].scenario.to_dict()),
            key="shap_scenario"
        )
        selected_scenario = ranked[scenario_select].scenario
    
    with col2:
        if st.button("Generate SHAP Explanation", use_container_width=True):
            with st.spinner("Computing SHAP values..."):
                try:
                    shap_results = st.session_state.estimator.explain_scenario_with_shap(
                        patient_features,
                        selected_scenario
                    )
                    
                    if "error" not in shap_results:
                        # Display top features
                        st.success("✅ SHAP analysis complete")
                        
                        col_pos, col_neg = st.columns(2)
                        
                        with col_pos:
                            st.write("**↑ Top Features (Increase Survival)**")
                            for feat, val in list(shap_results["top_positive_features"].items())[:5]:
                                st.caption(f"{feat}: {val:.3f}")
                        
                        with col_neg:
                            st.write("**↓ Top Features (Decrease Survival)**")
                            for feat, val in list(shap_results["top_negative_features"].items())[:5]:
                                st.caption(f"{feat}: {val:.3f}")
                    else:
                        st.error(f"⚠️ Error: {shap_results['error']}")
                
                except Exception as e:
                    st.error(f"⚠️ SHAP computation failed: {str(e)}")


# =========================================================================
# TAB 4: REPORT
# =========================================================================

with tab4:
    st.subheader("Clinical Decision-Support Report")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Patient ID", patient_id)
    with col2:
        st.metric("Alive Patients in Cohort", len(st.session_state.df))
    with col3:
        st.metric("Tumor Subtype", patient_row["Pam50 + Claudin-low subtype"])
    
    st.markdown("---")
    
    # Detailed results
    st.subheader("Analysis Results")
    st.json(summary)
    
    st.markdown("---")
    st.subheader("Disclaimer & Clinical Context")
    st.warning(summary["disclaimer"])
    
    st.markdown("""
    ### Important Considerations
    
    **Limitations of This Analysis:**
    - Based on retrospective observational data (METABRIC)
    - Does not account for patient comorbidities or performance status
    - Treatment assignment is non-random (selection bias exists)
    - Time period reflects 2000-2012 standards (clinical guidelines have evolved)
    
    **For Clinical Integration:**
    - Always consult current treatment guidelines (NCCN, ESMO)
    - Consider individual patient health status and preferences
    - Discuss with multidisciplinary oncology team
    - Use as decision support, not as treatment prescription
    
    **Next Steps for Production:**
    - External validation on independent cohorts
    - Fairness assessment across demographic groups
    - Integration with electronic health records
    - Prospective clinical outcome tracking
    """)
    
    st.markdown("---")
    st.subheader("Export Report")
    
    col1, col2 = st.columns(2)
    with col1:
        json_str = json.dumps(summary, indent=2)
        st.download_button(
            label="📥 Download JSON Report",
            data=json_str,
            file_name=f"treatment_benefit_report_{patient_id}.json",
            mime="application/json"
        )
    
    with col2:
        csv_str = report.to_dataframe().to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Results",
            data=csv_str,
            file_name=f"treatment_scenarios_{patient_id}.csv",
            mime="text/csv"
        )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>

**Precision Medicine Treatment Benefit Estimator**  
Built with SHAP explanations for clinical transparency  
METABRIC Dataset | Random Forest Models | Decision Support Framework

⚠️ FOR DECISION SUPPORT ONLY — NOT A TREATMENT PRESCRIPTION
</div>
""", unsafe_allow_html=True)
