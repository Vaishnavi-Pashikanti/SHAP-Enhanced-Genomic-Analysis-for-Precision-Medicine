"""
app_precision_medicine.py

Comprehensive precision medicine application integrating:
- Treatment Benefit Estimation
- SHAP-based Explainability  
- RAG Clinical Context
- Decision Support Framework
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
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
    DecisionSupportReport
)
from rag_clinical_context import RAGClinicalContext
from data_prep import load_data, choose_features, map_survival_status

# =============================================================================
# PAGE CONFIG & STATE
# =============================================================================

st.set_page_config(
    page_title="Precision Medicine: Treatment Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "estimator" not in st.session_state:
    st.session_state.estimator = None
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
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
def load_estimators():
    """Load treatment benefit estimator and RAG engine."""
    return (
        TreatmentBenefitEstimator(),
        RAGClinicalContext()
    )


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


def get_alignment_color(alignment: str) -> str:
    """Map guideline alignment to color."""
    if alignment == "Aligned":
        return "✅"
    elif alignment == "Reasonable":
        return "✓"
    elif alignment == "Unusual":
        return "⚠️"
    else:
        return "❌"


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.markdown("---")
    st.subheader("System Initialization")
    
    if st.button("Initialize System", use_container_width=True):
        with st.spinner("Loading models and data..."):
            st.session_state.estimator, st.session_state.rag_engine = load_estimators()
            st.session_state.df, st.session_state.feature_names = load_patient_data()
            st.session_state.df = st.session_state.estimator.filter_alive_patients(st.session_state.df)
            st.success(f"✅ System ready ({len(st.session_state.df)} alive patients)")
    
    st.markdown("---")
    st.subheader("Patient Selection")
    
    if st.session_state.df is not None:
        # Patient list
        patient_options = [
            f"ID: {row['Patient ID']} | Age: {row['Age at Diagnosis']:.1f} | {row['Pam50 + Claudin-low subtype']}"
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
        with st.expander("Patient Clinical Profile"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Patient ID:** {patient_row['Patient ID']}")
                st.write(f"**Age:** {patient_row['Age at Diagnosis']:.1f} years")
                st.write(f"**ER Status:** {patient_row['ER Status']}")
                st.write(f"**HER2 Status:** {patient_row['HER2 Status']}")
            with col2:
                st.write(f"**Subtype:** {patient_row['Pam50 + Claudin-low subtype']}")
                st.write(f"**Tumor Stage:** {patient_row['Tumor Stage']}")
                st.write(f"**Grade:** {patient_row['Neoplasm Histologic Grade']}")
                st.write(f"**Tumor Size:** {patient_row['Tumor Size']:.1f} mm" if pd.notna(patient_row['Tumor Size']) else "**Tumor Size:** N/A")


# =============================================================================
# MAIN CONTENT - HEADER
# =============================================================================

st.title("🏥 Precision Medicine: Treatment Decision Support")
st.markdown("""
**AI-Powered Decision Support for Personalized Breast Cancer Treatment**

This system combines:
- **Treatment Benefit Estimation**: Counterfactual survival outcome predictions
- **SHAP Explainability**: Feature importance and interpretability
- **Clinical Guidelines**: Evidence-based context from NCCN, ESMO, St. Gallen
- **Patient-Centered Approach**: Decision support, not treatment prescription
""")

# Check if system is initialized
if st.session_state.estimator is None or st.session_state.df is None:
    st.warning("⚠️ Please initialize the system from the sidebar.")
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

# Get guideline context
with st.spinner("Retrieving clinical guidelines..."):
    guideline_summary = st.session_state.rag_engine.generate_patient_guideline_summary(
        patient_id=patient_id,
        age=patient_row["Age at Diagnosis"],
        subtype=str(patient_row["Pam50 + Claudin-low subtype"]),
        stage=str(patient_row["Tumor Stage"]),
        er_status=str(patient_row["ER Status"]),
        her2_status=str(patient_row["HER2 Status"])
    )

# Create report
report = DecisionSupportReport(patient_id, benefits, baseline_pred)
summary = report.generate_summary()

# =============================================================================
# MAIN TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Treatment Benefit Analysis",
    "📋 Clinical Guidelines Context",
    "📈 Scenario Comparison",
    "🔬 SHAP Explanations",
    "📄 Full Report"
])

# =========================================================================
# TAB 1: TREATMENT BENEFIT ANALYSIS
# =========================================================================

with tab1:
    st.subheader("Personalized Treatment Benefit Estimation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Baseline: Survival Probability",
            f"{baseline_pred.survival_probability:.1%}",
            help="Predicted probability of being alive with no treatment"
        )
    
    with col2:
        st.metric(
            "Baseline: Survival Time",
            f"{baseline_pred.predicted_survival_months:.0f} months",
            help="Expected survival duration with no treatment"
        )
    
    with col3:
        st.metric(
            "Patient Subtype",
            str(patient_row["Pam50 + Claudin-low subtype"])
        )
    
    st.markdown("---")
    st.subheader("Treatment Options Ranked by Predicted Benefit")
    
    # Rank by benefit
    ranked = sorted(
        benefits,
        key=lambda b: (
            abs(b.survival_probability_benefit) + 
            abs(b.survival_months_benefit) / 60
        ),
        reverse=True
    )
    
    # Display top scenarios with guidelines
    for rank, benefit in enumerate(ranked[:4], 1):
        scenario = benefit.scenario
        treatment_str = format_treatment_scenario(scenario.to_dict())
        
        # Get guideline alignment for this scenario
        alignment_ctx = st.session_state.rag_engine.evaluate_treatment_alignment(
            subtype=str(patient_row["Pam50 + Claudin-low subtype"]),
            stage=str(patient_row["Tumor Stage"]),
            chemotherapy=scenario.chemotherapy,
            hormone_therapy=scenario.hormone_therapy,
            radiotherapy=scenario.radiotherapy,
            er_status=str(patient_row["ER Status"]),
            her2_status=str(patient_row["HER2 Status"])
        )
        
        with st.container(border=True):
            col1, col2, col3 = st.columns([1, 3, 1.5])
            
            with col1:
                st.markdown(f"### #{rank}")
            
            with col2:
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
                # Benefit color
                benefit_color = get_benefit_color(benefit.benefit_category())
                st.markdown(f"{benefit_color}")
                st.caption(benefit.benefit_category())
                
                # Guideline alignment
                align_color = get_alignment_color(alignment_ctx.guideline_alignment)
                st.markdown(f"{align_color}")
                st.caption(alignment_ctx.guideline_alignment)


# =========================================================================
# TAB 2: CLINICAL GUIDELINES CONTEXT
# =========================================================================

with tab2:
    st.subheader("Evidence-Based Clinical Guidelines")
    
    guideline_display = st.session_state.rag_engine.format_for_clinical_display(guideline_summary)
    
    # Patient profile
    st.write("**Patient Profile:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Age", f"{guideline_display['patient_profile']['age']}")
    with col2:
        st.metric("Subtype", guideline_display['patient_profile']['subtype'])
    with col3:
        st.metric("Stage", guideline_display['patient_profile']['stage'])
    with col4:
        st.metric("Receptors", guideline_display['patient_profile']['receptor_status'])
    
    st.markdown("---")
    
    # Standard treatment
    st.write("**Standard Treatment Recommendations for This Profile:**")
    for i, rec in enumerate(guideline_display['guideline_recommendations']['standard_treatment'], 1):
        st.write(f"{i}. {rec}")
    
    st.markdown("---")
    
    # Age-specific considerations
    st.write("**Age-Specific Treatment Considerations:**")
    for rec in guideline_display['guideline_recommendations']['age_specific']:
        st.write(f"• {rec}")
    
    st.markdown("---")
    
    # Stage-specific considerations
    st.write("**Stage-Specific Considerations:**")
    stage_info = guideline_display['guideline_recommendations']['stage_specific']
    for key, value in stage_info.items():
        st.write(f"• **{key}:** {value}")
    
    st.markdown("---")
    
    # Contraindications
    if guideline_display['contraindications']:
        st.warning("**Clinical Caveats & Contraindications:**")
        for caveat in guideline_display['contraindications']:
            st.write(f"⚠️ {caveat}")
    
    st.markdown("---")
    st.caption(f"Evidence sources: {', '.join(guideline_display['evidence_sources'])}")
    st.info(guideline_display['note'])


# =========================================================================
# TAB 3: SCENARIO COMPARISON
# =========================================================================

with tab3:
    st.subheader("Comparative Treatment Scenario Analysis")
    
    benefits_df = report.to_dataframe()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Survival Probability Benefit vs Baseline**")
        prob_data = benefits_df.sort_values(
            "survival_probability_benefit",
            ascending=False
        )
        
        st.bar_chart(
            data=prob_data.set_index("scenario_id")["survival_probability_benefit"],
            use_container_width=True
        )
    
    with col2:
        st.write("**Survival Months Benefit vs Baseline**")
        months_data = benefits_df.sort_values(
            "survival_months_benefit",
            ascending=False
        )
        
        st.bar_chart(
            data=months_data.set_index("scenario_id")["survival_months_benefit"],
            use_container_width=True
        )
    
    st.markdown("---")
    st.subheader("All Treatment Scenarios")
    
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
    display_df["Surv. Prob. Δ"] = display_df["survival_probability_benefit"].apply(lambda x: f"{x:+.1%}")
    display_df["Surv. Time Δ"] = display_df["survival_months_benefit"].apply(lambda x: f"{x:+.1f} mo")
    display_df["Benefit Category"] = display_df["benefit_category"]
    
    st.dataframe(
        display_df[["Treatment", "Surv. Prob. Δ", "Surv. Time Δ", "Benefit Category"]],
        use_container_width=True,
        hide_index=True
    )


# =========================================================================
# TAB 4: SHAP EXPLANATIONS
# =========================================================================

with tab4:
    st.subheader("Feature Importance & SHAP Explainability")
    
    st.write("""
    **SHAP values** show which patient features most strongly influence survival predictions.
    Positive values increase survival; negative values decrease survival.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Select Treatment Scenario:**")
        scenario_select = st.selectbox(
            "Scenario",
            options=range(min(4, len(ranked))),
            format_func=lambda i: format_treatment_scenario(ranked[i].scenario.to_dict()),
            key="shap_scenario"
        )
        selected_scenario = ranked[scenario_select].scenario
        
        if st.button("Compute SHAP Values", use_container_width=True):
            with st.spinner("Computing feature importance..."):
                try:
                    shap_results = st.session_state.estimator.explain_scenario_with_shap(
                        patient_features,
                        selected_scenario
                    )
                    
                    if "error" not in shap_results:
                        st.session_state.shap_results = shap_results
                        st.success("✅ Complete")
                    else:
                        st.error(f"Error: {shap_results['error']}")
                
                except Exception as e:
                    st.error(f"SHAP computation failed: {str(e)}")
    
    with col2:
        if "shap_results" in st.session_state:
            results = st.session_state.shap_results
            
            col_pos, col_neg = st.columns(2)
            
            with col_pos:
                st.write("**↑ Top Features (Increase Survival)**")
                for feat, val in list(results.get("top_positive_features", {}).items())[:5]:
                    st.caption(f"{feat}: {val:.3f}")
            
            with col_neg:
                st.write("**↓ Top Features (Decrease Survival)**")
                for feat, val in list(results.get("top_negative_features", {}).items())[:5]:
                    st.caption(f"{feat}: {val:.3f}")


# =========================================================================
# TAB 5: FULL REPORT
# =========================================================================

with tab5:
    st.subheader("Comprehensive Clinical Report")
    
    # Patient & baseline info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Patient ID", patient_id)
    with col2:
        st.metric("Subtype", str(patient_row["Pam50 + Claudin-low subtype"]))
    with col3:
        st.metric("Stage", str(patient_row["Tumor Stage"]))
    
    st.markdown("---")
    
    st.subheader("Treatment Benefit Analysis Summary")
    st.json(summary)
    
    st.markdown("---")
    st.subheader("Clinical Guidelines Summary")
    st.json(guideline_display)
    
    st.markdown("---")
    st.subheader("Important Disclaimers")
    
    st.warning(summary["disclaimer"])
    
    st.info("""
    ### Limitations & Clinical Considerations
    
    **Data Limitations:**
    - Retrospective cohort (potential confounding & selection bias)
    - Time period: 2000-2012 (treatment standards evolved)
    - Missing patient health factors (comorbidities, performance status)
    
    **Model Limitations:**
    - Counterfactual scenarios are hypothetical, not causal
    - Observational data cannot establish causal treatment effects
    - Benefit estimates reflect training data patterns
    
    **Clinical Integration:**
    - Use as decision support, NOT as treatment prescription
    - Always consult current guidelines (NCCN, ESMO, St. Gallen)
    - Include patient preferences and multidisciplinary team input
    - Consider individual comorbidities and organ function
    """)
    
    st.markdown("---")
    st.subheader("Export & Archive")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        json_str = json.dumps({
            "decision_support": summary,
            "guideline_context": guideline_display,
            "timestamp": pd.Timestamp.now().isoformat()
        }, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"precision_medicine_report_{patient_id}.json",
            mime="application/json"
        )
    
    with col2:
        csv_str = report.to_dataframe().to_csv(index=False)
        st.download_button(
            label="📥 Download Scenarios",
            data=csv_str,
            file_name=f"treatment_scenarios_{patient_id}.csv",
            mime="text/csv"
        )
    
    with col3:
        st.caption("Reports can be exported for EMR integration")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>

**Precision Medicine: AI-Enhanced Treatment Decision Support**  
Treatment Benefit Estimation + SHAP Explainability + Clinical Guidelines  
Built with RandomForest Models and METABRIC Clinical Data  

⚠️ **FOR DECISION SUPPORT ONLY** — Not a treatment prescription  
Always consult qualified healthcare professionals and current treatment guidelines

</div>
""", unsafe_allow_html=True)
