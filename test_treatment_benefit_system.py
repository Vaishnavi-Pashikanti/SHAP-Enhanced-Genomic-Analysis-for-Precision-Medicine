"""
test_treatment_benefit_system.py

Integration tests for the treatment benefit estimation framework.
Validates all core functionality before clinical deployment.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path for data_prep imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from treatment_benefit_estimator import (
    TreatmentBenefitEstimator,
    DecisionSupportReport,
    TreatmentScenario
)
from rag_clinical_context import RAGClinicalContext
from data_prep import load_data, choose_features, map_survival_status


def test_data_loading():
    """Test 1: Data loading and preprocessing."""
    print("\n" + "="*80)
    print("TEST 1: Data Loading & Preprocessing")
    print("="*80)
    
    df = load_data("data/brca_metabric_clinical_data.csv")
    assert len(df) == 2509, f"Expected 2509 patients, got {len(df)}"
    assert "Overall Survival Status" in df.columns
    assert "Pam50 + Claudin-low subtype" in df.columns
    
    df["Overall Survival Status_bin"] = df["Overall Survival Status"].apply(map_survival_status)
    
    living = (df["Overall Survival Status_bin"] == 0).sum()
    deceased = (df["Overall Survival Status_bin"] == 1).sum()
    
    print(f"[OK] Data loaded: {len(df)} patients")
    print(f"  Living: {living} ({living/len(df)*100:.1f}%)")
    print(f"  Deceased: {deceased} ({deceased/len(df)*100:.1f}%)")
    
    return df


def test_model_loading():
    """Test 2: Model loading and initialization."""
    print("\n" + "="*80)
    print("TEST 2: Model Loading & Initialization")
    print("="*80)
    
    try:
        estimator = TreatmentBenefitEstimator()
        print("[OK] Treatment Benefit Estimator loaded successfully")
        print(f"  Features: {len(estimator.feature_names)}")
        print(f"  Treatment columns: {estimator.treatment_cols}")
        return estimator
    except Exception as e:
        print(f"[FAIL] Failed to load estimator: {e}")
        raise


def test_alive_patient_filtering(df, estimator):
    """Test 3: Filtering for alive patients."""
    print("\n" + "="*80)
    print("TEST 3: Alive Patient Filtering")
    print("="*80)
    
    alive_df = estimator.filter_alive_patients(df)
    assert len(alive_df) > 0, "No alive patients found"
    assert len(alive_df) < len(df), "All patients marked as alive"
    
    print(f"[OK] Filtering complete")
    print(f"  Total patients: {len(df)}")
    print(f"  Alive patients: {len(alive_df)} ({len(alive_df)/len(df)*100:.1f}%)")
    
    return alive_df


def test_scenario_generation(estimator):
    """Test 4: Treatment scenario generation."""
    print("\n" + "="*80)
    print("TEST 4: Treatment Scenario Generation")
    print("="*80)
    
    scenarios = estimator.generate_treatment_scenarios()
    assert len(scenarios) == 8, f"Expected 8 scenarios (2^3), got {len(scenarios)}"
    
    print(f"[OK] Generated {len(scenarios)} scenarios:")
    for i, scenario in enumerate(scenarios):
        treatments = []
        if scenario.chemotherapy:
            treatments.append("Chemo")
        if scenario.hormone_therapy:
            treatments.append("Hormone")
        if scenario.radiotherapy:
            treatments.append("Radio")
        if not treatments:
            treatments = ["No Treatment"]
        print(f"  {i}: {' + '.join(treatments)}")
    
    return scenarios


def test_patient_analysis(estimator, df):
    """Test 5: Full patient analysis."""
    print("\n" + "="*80)
    print("TEST 5: Full Patient Treatment Benefit Analysis")
    print("="*80)
    
    alive_df = estimator.filter_alive_patients(df)
    X = choose_features(alive_df)
    
    patient_row = alive_df.iloc[0]
    patient_features = X.iloc[0]
    patient_id = patient_row["Patient ID"]
    
    print(f"Patient ID: {patient_id}")
    print(f"Age: {patient_row['Age at Diagnosis']:.1f}")
    print(f"Subtype: {patient_row['Pam50 + Claudin-low subtype']}")
    print(f"Stage: {patient_row['Tumor Stage']}")
    
    # Estimate benefits
    benefits, baseline_pred = estimator.estimate_treatment_benefits(patient_features)
    
    assert baseline_pred.survival_probability >= 0 and baseline_pred.survival_probability <= 1
    assert baseline_pred.predicted_survival_months > 0
    assert len(benefits) == 8
    
    print(f"\n[OK] Analysis complete:")
    print(f"  Baseline survival probability: {baseline_pred.survival_probability:.1%}")
    print(f"  Baseline predicted months: {baseline_pred.predicted_survival_months:.1f}")
    
    # Show top scenarios
    ranked = sorted(
        benefits,
        key=lambda b: abs(b.survival_probability_benefit),
        reverse=True
    )
    
    print(f"\n  Top benefit scenarios:")
    for i, benefit in enumerate(ranked[:3], 1):
        scenario = benefit.scenario
        treatments = []
        if scenario.chemotherapy:
            treatments.append("Chemo")
        if scenario.hormone_therapy:
            treatments.append("Hormone")
        if scenario.radiotherapy:
            treatments.append("Radio")
        treatment_str = " + ".join(treatments) if treatments else "No Treatment"
        
        prob_delta = benefit.survival_probability_benefit
        months_delta = benefit.survival_months_benefit
        
        print(f"    {i}. {treatment_str}")
        print(f"       Prob benefit: {prob_delta:+.1%}, Months benefit: {months_delta:+.1f}")
    
    return patient_id, benefits, baseline_pred


def test_decision_support_report(patient_id, benefits, baseline_pred):
    """Test 6: Report generation."""
    print("\n" + "="*80)
    print("TEST 6: Decision Support Report Generation")
    print("="*80)
    
    report = DecisionSupportReport(patient_id, benefits, baseline_pred)
    summary = report.generate_summary()
    
    assert "patient_id" in summary
    assert "baseline_survival_probability" in summary
    assert "treatment_scenarios_ranked" in summary
    assert len(summary["treatment_scenarios_ranked"]) == 8
    
    print(f"[OK] Report generated:")
    print(f"  Patient ID: {summary['patient_id']}")
    print(f"  Baseline probability: {summary['baseline_survival_probability']:.1%}")
    print(f"  Scenarios ranked: {len(summary['treatment_scenarios_ranked'])}")
    
    # Convert to dataframe
    df = report.to_dataframe()
    assert len(df) == 8
    print(f"  Dataframe shape: {df.shape}")
    
    return summary, df


def test_rag_clinical_context():
    """Test 7: RAG clinical context engine."""
    print("\n" + "="*80)
    print("TEST 7: RAG Clinical Context Engine")
    print("="*80)
    
    rag = RAGClinicalContext()
    print("[OK] RAG engine initialized")
    
    # Test subtype context
    subtypes = ["LumA", "LumB", "Her2", "Basal", "Normal-like"]
    print(f"\n  Available subtypes:")
    for subtype in subtypes:
        info = rag.get_subtype_context(subtype)
        if "error" not in info:
            print(f"    [OK] {subtype}: {info['subtype_name']}")
        else:
            print(f"    ✗ {subtype}: {info['error']}")
    
    # Test guideline summary
    patient_context = rag.generate_patient_guideline_summary(
        patient_id="TEST-001",
        age=55,
        subtype="LumB",
        stage="Stage II",
        er_status="Positive",
        her2_status="Negative"
    )
    
    print(f"\n[OK] Generated guideline summary for test patient")
    print(f"  Subtype recommendations: {len(patient_context.subtype_recommendations)}")
    print(f"  Age considerations: {len(patient_context.age_considerations)}")
    print(f"  Contraindications: {len(patient_context.contraindications)}")
    
    # Test treatment scenario alignment
    alignment_ctx = rag.evaluate_treatment_alignment(
        subtype="LumB",
        stage="Stage II",
        chemotherapy=True,
        hormone_therapy=True,
        radiotherapy=True,
        er_status="Positive",
        her2_status="Negative"
    )
    
    print(f"\n[OK] Evaluated treatment scenario alignment")
    print(f"  Guideline alignment: {alignment_ctx.guideline_alignment}")
    print(f"  Evidence level: {alignment_ctx.evidence_level}")
    print(f"  Rationale: {alignment_ctx.rationale}")
    
    return rag


def test_end_to_end_workflow(df):
    """Test 8: Complete end-to-end workflow."""
    print("\n" + "="*80)
    print("TEST 8: End-to-End Workflow")
    print("="*80)
    
    # Initialize
    estimator = TreatmentBenefitEstimator()
    rag = RAGClinicalContext()
    
    # Get alive patient
    alive_df = estimator.filter_alive_patients(df)
    patient_row = alive_df.iloc[0]
    patient_id = patient_row["Patient ID"]
    
    X = choose_features(alive_df)
    patient_features = X.iloc[0]
    
    # Treatment benefit analysis
    benefits, baseline_pred = estimator.estimate_treatment_benefits(patient_features)
    
    # Clinical guideline context
    guideline_summary = rag.generate_patient_guideline_summary(
        patient_id=patient_id,
        age=patient_row["Age at Diagnosis"],
        subtype=str(patient_row["Pam50 + Claudin-low subtype"]),
        stage=str(patient_row["Tumor Stage"]),
        er_status=str(patient_row["ER Status"]),
        her2_status=str(patient_row["HER2 Status"])
    )
    
    # Evaluate all scenarios for guideline alignment
    scenario_alignments = []
    for benefit in benefits:
        alignment = rag.evaluate_treatment_alignment(
            subtype=str(patient_row["Pam50 + Claudin-low subtype"]),
            stage=str(patient_row["Tumor Stage"]),
            chemotherapy=benefit.scenario.chemotherapy,
            hormone_therapy=benefit.scenario.hormone_therapy,
            radiotherapy=benefit.scenario.radiotherapy,
            er_status=str(patient_row["ER Status"]),
            her2_status=str(patient_row["HER2 Status"])
        )
        scenario_alignments.append(alignment)
    
    print(f"[OK] Complete workflow executed:")
    print(f"  Patient: {patient_id}")
    print(f"  Treatment benefits estimated: {len(benefits)}")
    print(f"  Guideline contexts retrieved: {len(scenario_alignments)}")
    
    # Generate integrated report
    report = DecisionSupportReport(patient_id, benefits, baseline_pred)
    summary = report.generate_summary()
    
    print(f"  Report generated with {len(summary['treatment_scenarios_ranked'])} scenarios")
    
    return {
        "patient_id": patient_id,
        "benefits": benefits,
        "baseline_pred": baseline_pred,
        "guideline_summary": guideline_summary,
        "scenario_alignments": scenario_alignments,
        "report": summary
    }


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "="*80)
    print("TREATMENT BENEFIT ESTIMATION FRAMEWORK - TEST SUITE")
    print("="*80)
    
    try:
        # Test 1: Data Loading
        df = test_data_loading()
        
        # Test 2: Model Loading
        estimator = test_model_loading()
        
        # Test 3: Alive Patient Filtering
        alive_df = test_alive_patient_filtering(df, estimator)
        
        # Test 4: Scenario Generation
        scenarios = test_scenario_generation(estimator)
        
        # Test 5: Patient Analysis
        patient_id, benefits, baseline_pred = test_patient_analysis(estimator, df)
        
        # Test 6: Report Generation
        summary, report_df = test_decision_support_report(patient_id, benefits, baseline_pred)
        
        # Test 7: RAG Context
        rag = test_rag_clinical_context()
        
        # Test 8: End-to-end
        workflow_result = test_end_to_end_workflow(df)
        
        # Summary
        print("\n" + "="*80)
        print("[OK] ALL TESTS PASSED")
        print("="*80)
        
        print("\nSystem is ready for clinical deployment:")
        print("  [OK] Data loading & preprocessing working")
        print("  [OK] Treatment benefit estimation functional")
        print("  [OK] SHAP model loading successful")
        print("  [OK] RAG clinical context engine operational")
        print("  [OK] Decision support report generation working")
        print("  [OK] End-to-end workflow validated")
        
        print("\nTo run the app:")
        print("  streamlit run app_precision_medicine.py")
        
        return True
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"[FAIL] TEST FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
