"""
treatment_benefit_estimator.py

Counterfactual Treatment Benefit Estimation Module for Precision Medicine

This module implements decision-support for alive breast cancer patients by:
1. Identifying alive patients
2. Generating counterfactual treatment scenarios
3. Predicting survival outcomes under each scenario
4. Computing treatment benefit estimates
5. Providing SHAP-based explainability

Non-causal framework: Provides comparative benefit estimates for clinician interpretation,
not treatment prescriptions.
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
import os
from itertools import product
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TreatmentScenario:
    """Represents a single treatment configuration."""
    scenario_id: str
    chemotherapy: bool
    hormone_therapy: bool
    radiotherapy: bool
    
    def to_dict(self) -> Dict:
        return {
            "scenario_id": self.scenario_id,
            "chemotherapy": self.chemotherapy,
            "hormone_therapy": self.hormone_therapy,
            "radiotherapy": self.radiotherapy
        }


@dataclass
class SurvivalPrediction:
    """Survival prediction for a single scenario."""
    scenario: TreatmentScenario
    survival_probability: float  # P(alive)
    predicted_survival_months: float  # Expected survival time
    
    def to_dict(self) -> Dict:
        return {
            **self.scenario.to_dict(),
            "survival_probability": float(self.survival_probability),
            "predicted_survival_months": float(self.predicted_survival_months)
        }


@dataclass
class TreatmentBenefit:
    """Treatment benefit relative to baseline scenario."""
    scenario: TreatmentScenario
    survival_probability_benefit: float  # Delta vs baseline
    survival_months_benefit: float  # Delta vs baseline
    
    def benefit_category(self) -> str:
        """Categorize benefit level."""
        avg_benefit = (
            abs(self.survival_probability_benefit) + 
            abs(self.survival_months_benefit) / 60  # Normalize to 0-1 scale
        ) / 2
        
        if avg_benefit > 0.15:
            return "High predicted benefit"
        elif avg_benefit > 0.05:
            return "Moderate predicted benefit"
        else:
            return "Low or uncertain benefit"
    
    def to_dict(self) -> Dict:
        return {
            **self.scenario.to_dict(),
            "survival_probability_benefit": float(self.survival_probability_benefit),
            "survival_months_benefit": float(self.survival_months_benefit),
            "benefit_category": self.benefit_category()
        }


# ============================================================================
# TREATMENT BENEFIT ESTIMATOR
# ============================================================================

class TreatmentBenefitEstimator:
    """
    Core module for treatment benefit estimation and decision support.
    
    Attributes:
        classifier_pipeline: Trained RandomForest classifier (alive/deceased)
        regressor_pipeline: Trained RandomForest regressor (survival months)
        feature_names: List of feature names used in training
        treatment_cols: Names of treatment columns
    """
    
    def __init__(
        self,
        classifier_pipeline_path: str = "models/rf_pipeline.joblib",
        regressor_pipeline_path: str = "models/rf_reg_pipeline.joblib",
        feature_names_path: str = "models/selected_feature_names.json"
    ):
        """Load pre-trained models and metadata."""
        # Resolve project root and model paths robustly after files moved
        module_dir = Path(__file__).parent
        # Prefer models directory in same folder as this module; otherwise fall back to parent
        project_root = module_dir if (module_dir / "models").exists() else module_dir.parent

        # Use absolute paths for model files
        classifier_abs_path = project_root / classifier_pipeline_path
        regressor_abs_path = project_root / regressor_pipeline_path
        feature_names_abs_path = project_root / feature_names_path
        
        print("[TBE] Loading classification model...")
        self.classifier_pipeline = joblib.load(str(classifier_abs_path))
        
        print("[TBE] Loading regression model...")
        self.regressor_pipeline = joblib.load(str(regressor_abs_path))
        
        print("[TBE] Loading feature names...")
        with open(str(feature_names_abs_path), "r") as f:
            self.feature_names = json.load(f)
        
        # Treatment columns
        self.treatment_cols = ["Chemotherapy", "Hormone Therapy", "Radio Therapy"]
        
        print(f"[TBE] Initialized with {len(self.feature_names)} features")
    
    
    def filter_alive_patients(self, df: pd.DataFrame, 
                            status_col: str = "Overall Survival Status") -> pd.DataFrame:
        """
        Filter for alive patients only (decision-support focus).
        
        Args:
            df: Input dataframe
            status_col: Name of survival status column
            
        Returns:
            Dataframe with only alive patients
        """
        alive_mask = df[status_col].astype(str).str.lower().str.contains(
            'living|alive|0', na=False, regex=True
        )
        alive_df = df[alive_mask].copy()
        print(f"[filter_alive_patients] Found {len(alive_df)} / {len(df)} alive patients")
        return alive_df
    
    
    def generate_treatment_scenarios(self) -> List[TreatmentScenario]:
        """
        Generate all 2^3 = 8 counterfactual treatment scenarios.
        
        Returns:
            List of TreatmentScenario objects
        """
        scenarios = []
        scenario_id = 0
        
        for chemo, hormone, radio in product([False, True], repeat=3):
            scenario = TreatmentScenario(
                scenario_id=f"scenario_{scenario_id:02d}",
                chemotherapy=chemo,
                hormone_therapy=hormone,
                radiotherapy=radio
            )
            scenarios.append(scenario)
            scenario_id += 1
        
        print(f"[generate_treatment_scenarios] Generated {len(scenarios)} scenarios")
        return scenarios
    
    
    def apply_treatment_scenario(
        self,
        patient_features: pd.Series,
        scenario: TreatmentScenario
    ) -> pd.Series:
        """
        Create counterfactual by modifying treatment columns.
        
        Args:
            patient_features: Patient feature vector
            scenario: Target treatment scenario
            
        Returns:
            Modified feature vector with new treatment values
        """
        counterfactual = patient_features.copy()
        
        # Map treatment columns to actual feature values
        treatment_mapping = {
            "Chemotherapy": scenario.chemotherapy,
            "Hormone Therapy": scenario.hormone_therapy,
            "Radio Therapy": scenario.radiotherapy
        }
        
        for col, value in treatment_mapping.items():
            if col in counterfactual.index:
                counterfactual[col] = 1 if value else 0
        
        return counterfactual
    
    
    def predict_survival_for_scenario(
        self,
        patient_features: pd.Series,
        scenario: TreatmentScenario
    ) -> SurvivalPrediction:
        """
        Predict survival outcomes for a treatment scenario.
        
        Args:
            patient_features: Patient feature vector
            scenario: Treatment scenario
            
        Returns:
            SurvivalPrediction with classification and regression outputs
        """
        # Apply treatment modification
        counterfactual = self.apply_treatment_scenario(patient_features, scenario)
        
        # Ensure feature order matches training
        X = pd.DataFrame([counterfactual[self.feature_names]]).fillna(0)
        
        # Classification: P(alive) = 1 - P(deceased)
        clf_proba = self.classifier_pipeline.predict_proba(X)[0, 1]
        survival_probability = 1 - clf_proba  # Class 1 = deceased
        
        # Regression: Expected survival time
        predicted_months = self.regressor_pipeline.predict(X)[0]
        
        return SurvivalPrediction(
            scenario=scenario,
            survival_probability=float(survival_probability),
            predicted_survival_months=float(max(0, predicted_months))  # No negative months
        )
    
    
    def estimate_treatment_benefits(
        self,
        patient_features: pd.Series,
        baseline_scenario_id: int = 0
    ) -> Tuple[List[TreatmentBenefit], SurvivalPrediction]:
        """
        Estimate treatment benefits for a patient across all scenarios.
        
        Args:
            patient_features: Patient feature vector
            baseline_scenario_id: Index of baseline scenario (0 = no treatment)
            
        Returns:
            Tuple of (benefits list, baseline prediction)
        """
        scenarios = self.generate_treatment_scenarios()
        
        # Get baseline prediction
        baseline_scenario = scenarios[baseline_scenario_id]
        baseline_pred = self.predict_survival_for_scenario(patient_features, baseline_scenario)
        
        # Predict for all scenarios
        benefits = []
        for scenario in scenarios:
            pred = self.predict_survival_for_scenario(patient_features, scenario)
            
            # Compute benefit relative to baseline
            surv_prob_delta = pred.survival_probability - baseline_pred.survival_probability
            surv_months_delta = pred.predicted_survival_months - baseline_pred.predicted_survival_months
            
            benefit = TreatmentBenefit(
                scenario=scenario,
                survival_probability_benefit=surv_prob_delta,
                survival_months_benefit=surv_months_delta
            )
            benefits.append(benefit)
        
        return benefits, baseline_pred
    
    
    def rank_treatment_scenarios(
        self,
        benefits: List[TreatmentBenefit],
        weight_probability: float = 0.5,
        weight_months: float = 0.5
    ) -> List[TreatmentBenefit]:
        """
        Rank treatment scenarios by estimated benefit.
        
        Args:
            benefits: List of TreatmentBenefit objects
            weight_probability: Weight for survival probability benefit (0-1)
            weight_months: Weight for survival months benefit (0-1)
            
        Returns:
            Sorted list of benefits (highest to lowest)
        """
        def composite_benefit(b: TreatmentBenefit) -> float:
            # Normalize months to 0-1 scale for fair weighting (max 120 months)
            months_normalized = b.survival_months_benefit / 120.0
            return (
                weight_probability * b.survival_probability_benefit +
                weight_months * months_normalized
            )
        
        ranked = sorted(benefits, key=composite_benefit, reverse=True)
        return ranked
    
    
    def explain_scenario_with_shap(
        self,
        patient_features: pd.Series,
        scenario: TreatmentScenario,
        num_samples: int = 100
    ) -> Dict:
        """
        Generate SHAP explanations for a treatment scenario.
        
        Args:
            patient_features: Patient feature vector
            scenario: Treatment scenario
            num_samples: Number of background samples for SHAP
            
        Returns:
            Dict with SHAP values and feature importances
        """
        # Apply counterfactual
        counterfactual = self.apply_treatment_scenario(patient_features, scenario)
        X = pd.DataFrame([counterfactual[self.feature_names]]).fillna(0)
        
        try:
            # Import shap lazily to avoid module-level import errors
            try:
                import shap
            except Exception as e:
                return {
                    "error": "shap_not_available",
                    "message": f"SHAP import failed: {e}. Install shap and matplotlib or run in an environment with compatible NumPy." 
                }

            # Create SHAP explainer for classifier
            explainer = shap.TreeExplainer(
                self.classifier_pipeline.named_steps["clf"]
            )
            shap_values = explainer.shap_values(X)
            
            # Handle binary classification output
            if isinstance(shap_values, list):
                shap_pos = shap_values[1]  # Class 1 = deceased
            else:
                shap_pos = shap_values
            
            # Get feature contributions
            contributions = pd.Series(
                shap_pos[0],
                index=self.feature_names
            ).sort_values(ascending=False)
            
            return {
                "scenario_id": scenario.scenario_id,
                "top_positive_features": contributions.head(5).to_dict(),
                "top_negative_features": contributions.tail(5).to_dict(),
                "feature_contributions": contributions.to_dict()
            }
        except Exception as e:
            print(f"[explain_scenario_with_shap] Error: {e}")
            return {
                "scenario_id": scenario.scenario_id,
                "error": str(e)
            }


# ============================================================================
# DECISION SUPPORT OUTPUT
# ============================================================================

class DecisionSupportReport:
    """Generates clinician-facing decision-support report."""
    
    def __init__(self, patient_id: str, benefits: List[TreatmentBenefit], 
                 baseline_pred: SurvivalPrediction):
        """
        Args:
            patient_id: Patient identifier
            benefits: List of TreatmentBenefit objects
            baseline_pred: Baseline survival prediction
        """
        self.patient_id = patient_id
        self.benefits = benefits
        self.baseline_pred = baseline_pred
    
    
    def generate_summary(self) -> Dict:
        """Generate decision-support summary."""
        ranked = sorted(
            self.benefits,
            key=lambda b: (
                abs(b.survival_probability_benefit) + 
                abs(b.survival_months_benefit) / 60
            ),
            reverse=True
        )
        
        summary = {
            "patient_id": self.patient_id,
            "baseline_scenario": self.baseline_pred.scenario.to_dict(),
            "baseline_survival_probability": float(self.baseline_pred.survival_probability),
            "baseline_predicted_months": float(self.baseline_pred.predicted_survival_months),
            "treatment_scenarios_ranked": [
                {
                    **b.to_dict(),
                    "rank": i + 1
                }
                for i, b in enumerate(ranked)
            ],
            "disclaimer": (
                "This analysis is for DECISION SUPPORT ONLY. "
                "Final treatment decisions must be made by qualified healthcare professionals "
                "in consultation with the patient, considering comorbidities, performance status, "
                "preferences, and established clinical guidelines."
            )
        }
        
        return summary
    
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert benefits to dataframe for easy viewing."""
        rows = []
        for benefit in self.benefits:
            rows.append(benefit.to_dict())
        return pd.DataFrame(rows)


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def process_patient_for_treatment_benefit(
    patient_row: pd.Series,
    estimator: TreatmentBenefitEstimator,
    feature_names: List[str],
    patient_id: str = "UNKNOWN"
) -> Dict:
    """
    End-to-end processing for a single patient.
    
    Args:
        patient_row: Row from METABRIC data
        estimator: TreatmentBenefitEstimator instance
        feature_names: List of feature names
        patient_id: Patient identifier
        
    Returns:
        Complete treatment benefit analysis
    """
    # Extract features
    patient_features = patient_row[feature_names].copy()
    
    # Estimate benefits
    benefits, baseline_pred = estimator.estimate_treatment_benefits(patient_features)
    
    # Generate report
    report = DecisionSupportReport(patient_id, benefits, baseline_pred)
    
    return {
        "patient_id": patient_id,
        "decision_support": report.generate_summary(),
        "benefits_table": report.to_dataframe().to_dict('records')
    }


if __name__ == "__main__":
    # Example usage
    print("[main] Treatment Benefit Estimator Module")
    
    # Load estimator
    estimator = TreatmentBenefitEstimator()
    
    # Load sample data
    from data_prep import load_data, choose_features, map_survival_status
    
    df = load_data("data/brca_metabric_clinical_data.csv")
    df["Overall Survival Status_bin"] = df["Overall Survival Status"].apply(map_survival_status)
    
    # Filter for alive patients
    alive_df = estimator.filter_alive_patients(df)
    
    # Select first alive patient for demo
    if len(alive_df) > 0:
        X = choose_features(alive_df)
        patient_row = X.iloc[0]
        patient_id = alive_df.iloc[0]["Patient ID"]
        
        print(f"\n[demo] Processing patient: {patient_id}")
        
        # Estimate benefits
        benefits, baseline_pred = estimator.estimate_treatment_benefits(patient_row)
        
        # Display results
        report = DecisionSupportReport(patient_id, benefits, baseline_pred)
        summary = report.generate_summary()
        
        print("\n[demo] DECISION SUPPORT SUMMARY:")
        print(json.dumps(summary, indent=2))
        
        print("\n[demo] TREATMENT BENEFIT TABLE:")
        print(report.to_dataframe().to_string())
