"""
rag_clinical_context.py

Retrieval-Augmented Generation (RAG) module for clinical guideline context.

Role: Provides supporting clinical evidence and guideline recommendations
(NOT used for predictions or treatment ranking).

This module:
1. Retrieves evidence-based treatment rationales
2. Summarizes subtype-specific therapy recommendations
3. Highlights contraindications and clinical caveats
4. Presents context to support clinician decision-making
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ============================================================================
# CLINICAL GUIDELINE KNOWLEDGE BASE
# ============================================================================

GUIDELINE_KNOWLEDGE_BASE = {
    "LumA": {
        "subtype_name": "Luminal A (ER+ HER2-)",
        "prevalence": "50-60% of breast cancers",
        "characteristics": "ER+, HER2-, low Ki-67, best prognosis",
        "standard_treatment": [
            "Endocrine therapy (Tamoxifen, Aromatase Inhibitors) as primary treatment",
            "Chemotherapy for high-risk features (stage III, grade 3, node+, young age)",
            "Radiotherapy per standard post-operative protocols",
            "Consider 10-year endocrine therapy vs 5 years based on risk"
        ],
        "contraindications": [
            "Avoid anthracyclines if low-risk disease",
            "Consider cardiac risk before taxane use"
        ],
        "sources": ["NCCN 2024", "ESMO 2021", "St. Gallen 2021"]
    },
    
    "LumB": {
        "subtype_name": "Luminal B (ER+ HER2-)",
        "prevalence": "15-20% of breast cancers",
        "characteristics": "ER+, HER2-, high Ki-67, intermediate-high risk",
        "standard_treatment": [
            "Chemotherapy followed by endocrine therapy",
            "Anthracycline-based regimens recommended",
            "Taxanes for node-positive disease",
            "Extended endocrine therapy (up to 10 years)",
            "Consider ovarian suppression if premenopausal"
        ],
        "contraindications": [
            "HER2 status must be confirmed (can be HER2+ in 10% cases)",
            "Monitor for anthracycline cardiotoxicity",
            "Assess menopausal status for hormone therapy selection"
        ],
        "sources": ["NCCN 2024", "ESMO 2021"]
    },
    
    "Her2": {
        "subtype_name": "HER2-Enriched (HER2+)",
        "prevalence": "15-20% of breast cancers",
        "characteristics": "HER2+, high risk, historically poor prognosis",
        "standard_treatment": [
            "HER2-targeted therapy mandatory (Trastuzumab, Pertuzumab)",
            "Chemotherapy (usually taxane-based) + HER2-targeted therapy",
            "Newer agents: Trastuzumab deruxtecan (ADC), CDK4/6i consideration",
            "Radiotherapy per standard post-operative protocols",
            "HER2-directed therapy for 1 year (adjuvant/neoadjuvant)"
        ],
        "contraindications": [
            "Cardiotoxicity risk significant (EF monitoring required)",
            "Avoid in LVEF < 50%",
            "Drug interactions with anthracyclines"
        ],
        "sources": ["NCCN 2024", "ESMO 2021", "ASCO 2023"]
    },
    
    "Basal": {
        "subtype_name": "Basal-like (TNBC)",
        "prevalence": "10-15% of breast cancers",
        "characteristics": "ER-, PR-, HER2-, aggressive, early relapse",
        "standard_treatment": [
            "Chemotherapy is primary modality (no endocrine/HER2 targets)",
            "Platinum-based agents for BRCA-mutant patients",
            "Immune checkpoint inhibitors emerging (pembrolizumab, atezolizumab)",
            "Radiotherapy per standard post-operative protocols",
            "Clinical trials for novel agents recommended"
        ],
        "contraindications": [
            "Avoid hormone therapy (ER/PR negative)",
            "Avoid HER2-targeted therapy (not HER2+)",
            "Platinum sensitivity testing may guide therapy"
        ],
        "sources": ["NCCN 2024", "ESMO 2021"]
    },
    
    "Normal-like": {
        "subtype_name": "Normal-like",
        "prevalence": "< 5%",
        "characteristics": "Mixed profile, often low-risk",
        "standard_treatment": [
            "Treatment by hormone and HER2 status",
            "Often ER+ → endocrine therapy primary",
            "Assess with other prognostic markers"
        ],
        "contraindications": [],
        "sources": ["St. Gallen 2021"]
    }
}

STAGE_TREATMENT_CONTEXT = {
    "Stage I": {
        "description": "Early-stage, T1 N0 M0",
        "treatment_intensity": "Low-intermediate",
        "chemotherapy_consideration": "Low-risk: Consider omission; High-risk: Recommended",
        "radiotherapy": "Breast-conserving surgery: Yes; Mastectomy: Consider based on features"
    },
    "Stage II": {
        "description": "T1-2 N1 M0 or T2-3 N0 M0",
        "treatment_intensity": "Intermediate",
        "chemotherapy_consideration": "Generally recommended",
        "radiotherapy": "Usually indicated (post-op breast ± nodes)"
    },
    "Stage III": {
        "description": "Locally advanced, any T N2-3 M0",
        "treatment_intensity": "High",
        "chemotherapy_consideration": "Strongly recommended (neoadjuvant preferred)",
        "radiotherapy": "Strongly indicated (chest wall ± nodes)"
    }
}

AGE_RISK_CONTEXT = {
    "Young (<40)": {
        "risk_factors": "Higher grade, higher proliferation, worse prognosis",
        "treatment_implications": [
            "More aggressive chemotherapy regimens often recommended",
            "Ovarian suppression for premenopausal women on endocrine therapy",
            "Fertility preservation counseling critical"
        ]
    },
    "Middle (40-65)": {
        "risk_factors": "Intermediate presentation",
        "treatment_implications": [
            "Standard treatment per subtype and stage",
            "Assess menopausal status for hormone therapy"
        ]
    },
    "Older (>65)": {
        "risk_factors": "Comorbidity burden, organ dysfunction",
        "treatment_implications": [
            "Geriatric assessment recommended",
            "Consider reduced chemotherapy intensity",
            "Endocrine therapy often preferred",
            "Functional status critical for treatment selection"
        ]
    }
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class GuidelineContext:
    """Structured guideline context for a patient."""
    patient_id: str
    subtype: str
    age: float
    stage: str
    hr_status: str
    her2_status: str
    
    # Context items
    subtype_recommendations: List[str]
    stage_considerations: Dict
    age_considerations: List[str]
    contraindications: List[str]
    evidence_sources: List[str]


@dataclass
class TreatmentScenarioContext:
    """Context for evaluating a specific treatment scenario."""
    scenario_id: str
    treatments: Dict[str, bool]  # {chemo: bool, hormone: bool, radio: bool}
    guideline_alignment: str  # "Aligned", "Reasonable", "Unusual", "Contraindicated"
    rationale: str
    evidence_level: str  # "High", "Moderate", "Low"
    contraindications_present: List[str]
    additional_monitoring: List[str]


# ============================================================================
# RAG CLINICAL CONTEXT ENGINE
# ============================================================================

class RAGClinicalContext:
    """
    Retrieval-Augmented Generation for clinical guideline context.
    
    Purpose: Provide supporting clinical evidence and guideline recommendations
    for the treatment benefit analysis output. NOT used for predictions.
    """
    
    def __init__(self):
        """Initialize knowledge base."""
        self.subtype_kb = GUIDELINE_KNOWLEDGE_BASE
        self.stage_kb = STAGE_TREATMENT_CONTEXT
        self.age_kb = AGE_RISK_CONTEXT
        print("[RAG] Clinical context engine initialized")
    
    
    def get_subtype_context(self, subtype: str) -> Dict:
        """
        Retrieve guideline context for tumor subtype.
        
        Args:
            subtype: Pam50 subtype (LumA, LumB, Her2, Basal, Normal-like)
            
        Returns:
            Dict with guideline recommendations and evidence
        """
        # Normalize subtype names
        subtype_map = {
            "LumA": "LumA",
            "LumB": "LumB",
            "Her2": "Her2",
            "Basal": "Basal",
            "Normal-like": "Normal-like"
        }
        
        normalized = subtype_map.get(subtype, subtype)
        
        if normalized not in self.subtype_kb:
            return {"error": f"Unknown subtype: {subtype}"}
        
        return self.subtype_kb[normalized]
    
    
    def get_stage_context(self, stage: str) -> Dict:
        """Retrieve guideline context for tumor stage."""
        if stage not in self.stage_kb:
            return {"error": f"Unknown stage: {stage}"}
        return self.stage_kb[stage]
    
    
    def get_age_context(self, age: float) -> Tuple[str, Dict]:
        """Categorize age and return risk context."""
        if age < 40:
            category = "Young (<40)"
        elif age < 65:
            category = "Middle (40-65)"
        else:
            category = "Older (>65)"
        
        return category, self.age_kb[category]
    
    
    def evaluate_treatment_alignment(
        self,
        subtype: str,
        stage: str,
        chemotherapy: bool,
        hormone_therapy: bool,
        radiotherapy: bool,
        er_status: Optional[str] = None,
        her2_status: Optional[str] = None
    ) -> TreatmentScenarioContext:
        """
        Evaluate how treatment scenario aligns with clinical guidelines.
        
        Args:
            subtype: Tumor subtype
            stage: Tumor stage
            chemotherapy, hormone_therapy, radiotherapy: Treatment booleans
            er_status, her2_status: Receptor status for verification
            
        Returns:
            TreatmentScenarioContext with alignment assessment
        """
        
        # Build treatment scenario description
        treatments = {
            "chemotherapy": chemotherapy,
            "hormone_therapy": hormone_therapy,
            "radiotherapy": radiotherapy
        }
        
        treatments_list = []
        if chemotherapy:
            treatments_list.append("Chemotherapy")
        if hormone_therapy:
            treatments_list.append("Hormone Therapy")
        if radiotherapy:
            treatments_list.append("Radiotherapy")
        
        treatment_str = ", ".join(treatments_list) if treatments_list else "No Treatment"
        
        # Evaluate alignment
        alignment, rationale, level, contraindications, monitoring = \
            self._assess_alignment(
                subtype, stage, chemotherapy, hormone_therapy, radiotherapy,
                er_status, her2_status
            )
        
        return TreatmentScenarioContext(
            scenario_id=f"{subtype}_{treatment_str.replace(', ', '_')}",
            treatments=treatments,
            guideline_alignment=alignment,
            rationale=rationale,
            evidence_level=level,
            contraindications_present=contraindications,
            additional_monitoring=monitoring
        )
    
    
    def _assess_alignment(
        self,
        subtype: str,
        stage: str,
        chemotherapy: bool,
        hormone_therapy: bool,
        radiotherapy: bool,
        er_status: Optional[str] = None,
        her2_status: Optional[str] = None
    ) -> Tuple[str, str, str, List[str], List[str]]:
        """
        Internal logic for guideline alignment assessment.
        
        Returns:
            (alignment: str, rationale: str, evidence_level: str, 
             contraindications: List[str], monitoring: List[str])
        """
        
        contraindications = []
        monitoring = []
        
        # ===== LUMINAL A =====
        if subtype == "LumA":
            # High-risk: chemo + hormone + radio
            if stage in ["Stage II", "Stage III"]:
                if chemotherapy and hormone_therapy and radiotherapy:
                    return ("Aligned", 
                            "Standard treatment for high-risk Luminal A",
                            "High",
                            contraindications,
                            ["Cardiac function (anthracycline assessment)"])
                elif hormone_therapy and radiotherapy and not chemotherapy:
                    return ("Reasonable",
                            "Endocrine + radiotherapy; chemotherapy omission acceptable for low-risk features",
                            "Moderate",
                            contraindications,
                            [])
            elif stage == "Stage I":
                if hormone_therapy and not chemotherapy:
                    return ("Aligned",
                            "Endocrine therapy standard for low-risk Luminal A",
                            "High",
                            contraindications,
                            [])
        
        # ===== LUMINAL B =====
        if subtype == "LumB":
            if chemotherapy and hormone_therapy:
                if radiotherapy:
                    return ("Aligned",
                            "Standard: chemo-endocrine-radiotherapy for Luminal B",
                            "High",
                            contraindications,
                            ["Cardiac function", "EF baseline"])
                else:
                    return ("Reasonable",
                            "Chemo-endocrine appropriate; radiotherapy per surgical margins",
                            "High",
                            contraindications,
                            ["Cardiac function"])
            elif hormone_therapy and not chemotherapy:
                return ("Unusual",
                        "Endocrine monotherapy not standard for Luminal B without chemo",
                        "Low",
                        ["Consider chemotherapy based on risk factors"],
                        [])
        
        # ===== HER2+ =====
        if subtype == "Her2":
            if chemotherapy:  # HER2-targeted therapy implicit in model
                if radiotherapy:
                    return ("Aligned",
                            "Standard: chemo + HER2-targeted therapy + radiotherapy",
                            "High",
                            contraindications,
                            ["Cardiac function (EF monitoring essential)", "Trastuzumab cardiotoxicity"])
                else:
                    return ("Reasonable",
                            "Chemo + HER2-targeted therapy; radiotherapy per surgical approach",
                            "High",
                            contraindications,
                            ["Cardiac function"])
            else:
                return ("Unusual",
                        "Chemotherapy generally required for HER2+ disease",
                        "Low",
                        ["HER2-targeted therapy strongly recommended"],
                        ["Consider HER2-directed agents"])
        
        # ===== BASAL/TNBC =====
        if subtype == "Basal":
            if chemotherapy:
                if radiotherapy:
                    return ("Aligned",
                            "Chemotherapy + radiotherapy standard for TNBC",
                            "High",
                            contraindications,
                            ["No endocrine/HER2 targets; consider immunotherapy"])
                else:
                    return ("Reasonable",
                            "Chemotherapy standard; radiotherapy per surgical approach",
                            "High",
                            contraindications,
                            [])
            else:
                return ("Unusual",
                        "Chemotherapy is primary modality for TNBC",
                        "Low",
                        ["Chemotherapy strongly recommended"],
                        ["Consider platinum agents; IO therapy emerging"])
            
            if hormone_therapy:
                contraindications.append("ER-negative: avoid hormone therapy")
        
        # Default
        return ("Reasonable",
                f"{treatment_str} for {subtype} {stage}",
                "Moderate",
                contraindications,
                monitoring) if 'treatment_str' in locals() else ("Reasonable",
                f"Assessment pending for {subtype} {stage}",
                "Moderate",
                contraindications,
                monitoring)
    
    
    def generate_patient_guideline_summary(
        self,
        patient_id: str,
        age: float,
        subtype: str,
        stage: str,
        er_status: str,
        her2_status: str
    ) -> GuidelineContext:
        """
        Generate comprehensive guideline context for a patient.
        
        Args:
            patient_id, age, subtype, stage, er_status, her2_status: Patient attributes
            
        Returns:
            GuidelineContext with recommendations
        """
        
        subtype_info = self.get_subtype_context(subtype)
        stage_info = self.get_stage_context(stage)
        age_cat, age_info = self.get_age_context(age)
        
        return GuidelineContext(
            patient_id=patient_id,
            subtype=subtype,
            age=age,
            stage=stage,
            hr_status=er_status,
            her2_status=her2_status,
            subtype_recommendations=subtype_info.get("standard_treatment", []),
            stage_considerations=stage_info,
            age_considerations=age_info.get("treatment_implications", []),
            contraindications=subtype_info.get("contraindications", []),
            evidence_sources=subtype_info.get("sources", [])
        )
    
    
    def format_for_clinical_display(self, context: GuidelineContext) -> Dict:
        """Format context for clinician-facing display."""
        return {
            "patient_id": context.patient_id,
            "patient_profile": {
                "age": f"{context.age:.1f} years",
                "subtype": context.subtype,
                "stage": context.stage,
                "receptor_status": f"ER: {context.hr_status}, HER2: {context.her2_status}"
            },
            "guideline_recommendations": {
                "standard_treatment": context.subtype_recommendations,
                "stage_specific": context.stage_considerations,
                "age_specific": context.age_considerations
            },
            "contraindications": context.contraindications,
            "evidence_sources": context.evidence_sources,
            "note": "This information supports clinical decision-making. "
                    "Always consult current treatment guidelines and multidisciplinary team."
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("[RAG] Clinical Context Module - Example Usage\n")
    
    # Initialize
    rag = RAGClinicalContext()
    
    # Example patient
    patient_context = rag.generate_patient_guideline_summary(
        patient_id="MB-0001",
        age=55.0,
        subtype="LumB",
        stage="Stage II",
        er_status="Positive",
        her2_status="Negative"
    )
    
    print("PATIENT GUIDELINE CONTEXT:")
    import json
    display = rag.format_for_clinical_display(patient_context)
    print(json.dumps(display, indent=2))
    
    print("\n" + "="*80 + "\n")
    
    # Evaluate treatment scenarios
    print("TREATMENT SCENARIO ALIGNMENT:\n")
    
    scenarios = [
        (True, True, True, "Chemo + Hormone + Radio"),
        (True, False, True, "Chemo + Radio (no hormone)"),
        (False, True, False, "Hormone only (no chemo)")
    ]
    
    for chemo, hormone, radio, desc in scenarios:
        ctx = rag.evaluate_treatment_alignment(
            subtype="LumB",
            stage="Stage II",
            chemotherapy=chemo,
            hormone_therapy=hormone,
            radiotherapy=radio,
            er_status="Positive",
            her2_status="Negative"
        )
        
        print(f"**{desc}**")
        print(f"  Alignment: {ctx.guideline_alignment}")
        print(f"  Evidence: {ctx.evidence_level}")
        print(f"  Rationale: {ctx.rationale}")
        if ctx.contraindications_present:
            print(f"  ⚠️ Contraindications: {', '.join(ctx.contraindications_present)}")
        if ctx.additional_monitoring:
            print(f"  📋 Monitoring: {', '.join(ctx.additional_monitoring)}")
        print()
