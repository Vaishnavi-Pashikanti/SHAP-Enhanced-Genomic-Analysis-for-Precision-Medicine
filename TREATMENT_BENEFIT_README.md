# Precision Medicine: Treatment Benefit Estimation Framework

## Overview

This framework implements a **decision-support system for personalized breast cancer treatment** using:
- **Counterfactual Treatment Benefit Estimation**: Predicts survival outcomes under alternative treatment scenarios
- **SHAP Explainability**: Provides interpretable feature importance for clinical transparency
- **RAG Clinical Context**: Retrieves evidence-based guidelines from NCCN, ESMO, St. Gallen
- **Non-Causal Framework**: Decision support, not treatment prescription

## Project Structure

```
metabric-shap-project/
├── treatment_benefit_estimator.py    # Core benefit estimation module
├── rag_clinical_context.py           # Clinical guideline context (RAG)
├── app_treatment_benefit.py          # Streamlit: Treatment benefit analysis
├── app_precision_medicine.py         # Streamlit: Integrated precision medicine app
├── explain_shap.py                   # SHAP explainability utilities
├── data_prep.py                      # Data preprocessing
├── train_model.py                    # Model training
├── data/
│   └── brca_metabric_clinical_data.csv   # METABRIC dataset (2,509 patients)
└── models/
    ├── rf_pipeline.joblib            # Classification model (alive/deceased)
    ├── rf_reg_pipeline.joblib        # Regression model (survival months)
    ├── preprocessor.joblib
    ├── selector.joblib
    └── selected_feature_names.json
```

## Key Components

### 1. Treatment Benefit Estimator (`treatment_benefit_estimator.py`)

**Purpose**: Generate counterfactual treatment scenarios and predict survival outcomes

**Core Classes**:
- `TreatmentScenario`: Represents a treatment configuration (chemotherapy, hormone therapy, radiotherapy)
- `SurvivalPrediction`: Survival prediction for a scenario
- `TreatmentBenefit`: Benefit relative to baseline scenario
- `TreatmentBenefitEstimator`: Main module for analysis
- `DecisionSupportReport`: Clinical report generation

**Key Methods**:
```python
estimator = TreatmentBenefitEstimator()

# Filter alive patients
alive_df = estimator.filter_alive_patients(df)

# Generate all 2^3 = 8 treatment scenarios
scenarios = estimator.generate_treatment_scenarios()

# Estimate benefits for a patient
benefits, baseline_pred = estimator.estimate_treatment_benefits(patient_features)

# Rank by composite benefit
ranked = estimator.rank_treatment_scenarios(benefits)

# SHAP explanations
shap_results = estimator.explain_scenario_with_shap(patient_features, scenario)
```

**Approach**:
1. **Identify Alive Patients**: Focus on forward-looking decision support
2. **Generate Counterfactuals**: Create 8 treatment combinations (2^3)
3. **Predict Outcomes**: Use trained RF classifier + regressor
4. **Compute Benefit**: Δ = predicted_outcome - baseline_outcome
5. **Rank Scenarios**: By weighted survival probability + months benefit
6. **Explain**: SHAP values for feature importance

### 2. RAG Clinical Context (`rag_clinical_context.py`)

**Purpose**: Provide evidence-based clinical guideline context

**Features**:
- **Subtype-Specific Recommendations**: LumA, LumB, Her2, Basal, Normal-like
- **Stage-Based Treatment Intensity**: Stage I, II, III
- **Age-Stratified Implications**: Young (<40), Middle (40-65), Older (>65)
- **Guideline Alignment Assessment**: "Aligned", "Reasonable", "Unusual", "Contraindicated"
- **Evidence Sourcing**: NCCN 2024, ESMO 2021, St. Gallen 2021

**Core Classes**:
- `GuidelineContext`: Comprehensive guideline information for a patient
- `TreatmentScenarioContext`: Guideline alignment for a specific scenario
- `RAGClinicalContext`: Main RAG engine

**Key Methods**:
```python
rag = RAGClinicalContext()

# Get subtype guidelines
subtype_info = rag.get_subtype_context("LumB")

# Evaluate guideline alignment
alignment = rag.evaluate_treatment_alignment(
    subtype="LumB",
    stage="Stage II",
    chemotherapy=True,
    hormone_therapy=True,
    radiotherapy=True
)

# Generate patient guideline summary
summary = rag.generate_patient_guideline_summary(
    patient_id="MB-001",
    age=55,
    subtype="LumB",
    stage="Stage II",
    er_status="Positive",
    her2_status="Negative"
)
```

**Knowledge Base Coverage**:
- **LumA (50-60%)**: ER+, HER2-, low Ki-67 → Endocrine therapy primary
- **LumB (15-20%)**: ER+, HER2-, high Ki-67 → Chemo + endocrine
- **Her2 (15-20%)**: HER2+ → Chemo + HER2-targeted therapy
- **Basal (10-15%)**: TNBC → Chemotherapy + emerging immunotherapy
- **Normal-like (<5%)**: Mixed profile

### 3. Streamlit Applications

#### `app_precision_medicine.py` (Recommended)
**Integrated Application with Full Decision Support**

Features:
- 🏥 Patient selection and clinical profile
- 📊 Treatment benefit ranking (top 4 scenarios)
- 📋 Clinical guidelines context
- 📈 Comparative scenario analysis
- 🔬 SHAP feature importance
- 📄 Comprehensive report generation
- ✅ Guideline alignment indicators

**Usage**:
```bash
streamlit run app_precision_medicine.py
```

Workflow:
1. Initialize system (load models, data, RAG engine)
2. Select patient
3. Review baseline predictions
4. Compare treatment scenarios with guideline alignment
5. Explore SHAP explanations
6. Download report

#### `app_treatment_benefit.py` (Lightweight)
**Focused Treatment Benefit Analysis**

Features:
- Patient selection
- Treatment benefit analysis
- Scenario comparison
- SHAP explanations
- Report export

**Usage**:
```bash
streamlit run app_treatment_benefit.py
```

## Clinical Framework

### Decision Support, Not Prescription

The system provides:
- ✅ **Comparative benefit estimates** for alternative treatment scenarios
- ✅ **Feature importance** explaining which patient characteristics drive predictions
- ✅ **Guideline context** from established treatment standards
- ✅ **Uncertainty acknowledgment** from observational data

The system does NOT:
- ❌ Prescribe treatment
- ❌ Claim causal treatment effects
- ❌ Account for patient comorbidities or performance status
- ❌ Replace clinical judgment

### Limitations Addressed

**Observational Data Bias**:
- Non-random treatment assignment
- Selection bias (sicker patients get more aggressive treatment)
- Confounding by indication
- **Mitigation**: Explicitly frame as decision-support, not causal

**Missing Confounders**:
- Performance status (ECOG, Karnofsky)
- Organ function (cardiac, renal, hepatic)
- Comorbidities (diabetes, hypertension, etc.)
- Social determinants (access, preferences)
- **Mitigation**: Decision support requires clinician judgment

**Data Age**:
- METABRIC spans 2000-2012
- Treatment standards have evolved
- New agents available (CDK4/6 inhibitors, immunotherapy)
- **Mitigation**: RAG module provides contemporary guideline context

### Integration with Clinical Workflow

```
Patient Presentation
       ↓
AI Decision Support System
   ├── Benefit Estimation
   ├── SHAP Explanations
   ├── Guideline Context
   └── Report Generation
       ↓
Clinician Interpretation
   ├── Consider comorbidities
   ├── Integrate patient preferences
   ├── Consult multidisciplinary team
   └── Make clinical decision
       ↓
Treatment Implementation & Follow-up
```

## Usage Examples

### Example 1: Single Patient Analysis

```python
from treatment_benefit_estimator import TreatmentBenefitEstimator, DecisionSupportReport
from data_prep import load_data, choose_features

# Initialize
estimator = TreatmentBenefitEstimator()
df = load_data("data/brca_metabric_clinical_data.csv")

# Filter alive patients
alive_df = estimator.filter_alive_patients(df)

# Get features for first alive patient
X = choose_features(alive_df)
patient_features = X.iloc[0]
patient_id = alive_df.iloc[0]["Patient ID"]

# Estimate benefits
benefits, baseline_pred = estimator.estimate_treatment_benefits(patient_features)

# Generate report
report = DecisionSupportReport(patient_id, benefits, baseline_pred)
summary = report.generate_summary()

print(f"Patient: {patient_id}")
print(f"Baseline Survival Probability: {baseline_pred.survival_probability:.1%}")
print(f"Baseline Survival Time: {baseline_pred.predicted_survival_months:.1f} months")
print(f"\nTop Benefit Scenario: {summary['treatment_scenarios_ranked'][0]}")
```

### Example 2: Batch Patient Analysis

```python
from treatment_benefit_estimator import process_patient_for_treatment_benefit
import json

results = []
for idx, patient_row in alive_df.iterrows():
    X = choose_features(alive_df)
    analysis = process_patient_for_treatment_benefit(
        patient_row,
        estimator,
        X.columns.tolist(),
        patient_id=patient_row["Patient ID"]
    )
    results.append(analysis)

# Save results
with open("treatment_benefit_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Example 3: RAG Clinical Context

```python
from rag_clinical_context import RAGClinicalContext

rag = RAGClinicalContext()

# Get guidelines for a specific patient profile
patient_context = rag.generate_patient_guideline_summary(
    patient_id="MB-0001",
    age=55,
    subtype="LumB",
    stage="Stage II",
    er_status="Positive",
    her2_status="Negative"
)

# Evaluate treatment scenario alignment
scenario_context = rag.evaluate_treatment_alignment(
    subtype="LumB",
    stage="Stage II",
    chemotherapy=True,
    hormone_therapy=True,
    radiotherapy=True,
    er_status="Positive",
    her2_status="Negative"
)

print(f"Guideline Alignment: {scenario_context.guideline_alignment}")
print(f"Evidence Level: {scenario_context.evidence_level}")
print(f"Rationale: {scenario_context.rationale}")
```

## Model Information

### Classification Model (Alive/Deceased)
- **Algorithm**: Random Forest Classifier (200 trees)
- **Target**: Overall Survival Status (0=Living, 1=Deceased)
- **Features**: 31 selected features (SelectKBest)
- **Input**: Preprocessed clinical, pathological, molecular features
- **Output**: P(deceased), used as P(alive) = 1 - P(deceased)

### Regression Model (Survival Months)
- **Algorithm**: Random Forest Regressor
- **Target**: Overall Survival (Months)
- **Features**: Same 31 features
- **Output**: Expected survival duration in months

### Preprocessing Pipeline
1. **Numeric Features**: Median imputation → StandardScaler
2. **Categorical Features**: Mode imputation → OneHotEncoder
3. **Feature Selection**: SelectKBest (f_classif for classification)

## Installation & Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Key Dependencies
- pandas, numpy, scikit-learn
- joblib (model persistence)
- shap (explainability)
- streamlit (UI)
- matplotlib, seaborn (visualization)

### First Run
```bash
# 1. Load data
python data_prep.py

# 2. Train models (optional - pre-trained models included)
python train_model.py --data-csv data/brca_metabric_clinical_data.csv

# 3. Run Streamlit app
streamlit run app_precision_medicine.py
```

## Validation & Fairness

### Recommended Validation Steps

1. **External Cohort Testing**
   - Validate on independent breast cancer datasets
   - Compare to other survival prediction models
   - Assess performance by subtype

2. **Fairness Assessment**
   - Analyze benefit estimates by age groups
   - Evaluate across racial/ethnic groups
   - Check for stage-stratified fairness
   - Assess gender (all female in this dataset)

3. **Clinical Feasibility**
   - Gather clinician feedback on decision support format
   - Test integration with EHRs
   - Evaluate workflow impact

4. **Prospective Outcome Tracking**
   - Follow patients with AI-supported treatment planning
   - Compare to historical cohort
   - Track guideline adherence

## Ethical Considerations

✅ **Strengths**:
- Non-prescriptive: Decision support, not treatment prescription
- Transparent: SHAP explanations for every prediction
- Humble: Explicitly acknowledges limitations and biases
- Evidence-grounded: Integrates clinical guidelines
- Patient-centered: Complements clinician expertise

⚠️ **Ongoing Challenges**:
- Observational bias in historical data
- Difficulty accounting for unmeasured confounders
- Equity concerns across patient populations
- Evolving treatment standards vs. training data vintage

## Future Enhancements

1. **Causal Inference Methods**
   - Propensity score matching to reduce selection bias
   - Instrumental variable approaches
   - Causal forest methods

2. **Expanded RAG Integration**
   - Real-time guideline updates
   - Integration with clinical trial information
   - Biomarker-specific recommendations

3. **Patient Stratification**
   - Subgroup-specific benefit models
   - Tumor biology-driven recommendations
   - Integration with genomic profiling

4. **Real-World Validation**
   - External cohort testing
   - Prospective outcome tracking
   - Fairness auditing

5. **Clinical Integration**
   - EHR plugin development
   - Mobile app for point-of-care decision support
   - Integration with tumor boards

## References

**Clinical Guidelines**:
- NCCN Clinical Practice Guidelines in Oncology: Breast Cancer v2024
- ESMO Breast Cancer Guidelines 2021
- St. Gallen International Expert Consensus 2021

**METABRIC Dataset**:
- Curtis et al. Nature 2012 - Genomic and clinical characterization of breast cancer
- 2,509 patients with clinical, pathological, molecular features
- 20-year follow-up of survival outcomes

**SHAP Explainability**:
- Lundberg & Lee 2017 - "A Unified Approach to Interpreting Model Predictions"
- TreeExplainer for fast model-agnostic feature importance

## Contact & Citation

For questions or feedback: [Add contact information]

If you use this framework, please cite:
```
@misc{precision_medicine_treatment_benefit,
  title={Precision Medicine: Treatment Benefit Estimation Framework},
  author={[Your Name]},
  year={2025},
  url={[Your Repository]}
}
```

---

**⚠️ Important Disclaimer**

This system is for **DECISION SUPPORT ONLY**. It is not a treatment prescription. 

Final treatment decisions must be made by qualified healthcare professionals in consultation with:
- The patient
- Established clinical guidelines (NCCN, ESMO, St. Gallen)
- Multidisciplinary oncology team
- Consideration of comorbidities, performance status, and patient preferences

The system is based on retrospective data with known limitations (confounding, selection bias, missing confounders). Treat predictions as hypothetical comparisons, not causal guarantees.
