"""
Treatment Efficacy Analysis
============================

Analyze HIV treatment outcomes and effectiveness.

Key Questions Answered:
1. What factors predict successful viral suppression?
2. How do different ART regimens compare in effectiveness?
3. What are the barriers to treatment adherence?
4. How quickly do patients achieve viral suppression?
5. What are the survival rates and outcomes?
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
import plotly.express as px
import plotly.graph_objects as go
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TreatmentAnalyzer:
    """
    Analyze HIV treatment efficacy and outcomes.
    """
    
    def __init__(
        self,
        patients_data: Optional[pd.DataFrame] = None,
        lab_results_data: Optional[pd.DataFrame] = None,
        treatments_data: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize treatment analyzer.
        
        Args:
            patients_data: Patient demographics and outcomes
            lab_results_data: Lab test results over time
            treatments_data: Treatment regimen data
        """
        self.patients = patients_data
        self.lab_results = lab_results_data
        self.treatments = treatments_data
        
        logger.info("Treatment analyzer initialized")
    
    def analyze_viral_suppression_factors(self) -> pd.DataFrame:
        """
        Identify factors associated with viral suppression success.
        
        Returns:
            DataFrame with factor analysis results
        """
        logger.info("Analyzing factors associated with viral suppression")
        
        if self.patients is None:
            raise ValueError("No patient data loaded.")
        
        # Factors to analyze
        factors = [
            "age",
            "gender",
            "transmission_route",
            "cd4_count_at_diagnosis",
            "who_clinical_stage",
            "treatment_adherence",
            "country_code",
        ]
        
        results = []
        
        for factor in factors:
            if factor not in self.patients.columns:
                continue
            
            if self.patients[factor].dtype in ["object", "category"]:
                # Categorical factor - use chi-square test
                contingency_table = pd.crosstab(
                    self.patients[factor],
                    self.patients["viral_suppression"]
                )
                
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                
                # Calculate suppression rate by category
                suppression_rates = (
                    self.patients.groupby(factor)["viral_suppression"]
                    .agg(["mean", "count"])
                    .round(4)
                )
                
                for category in suppression_rates.index:
                    results.append({
                        "factor": factor,
                        "category": str(category),
                        "suppression_rate": round(suppression_rates.loc[category, "mean"] * 100, 2),
                        "n": int(suppression_rates.loc[category, "count"]),
                        "test_statistic": round(chi2, 4),
                        "p_value": round(p_value, 4),
                        "significant": "Yes" if p_value < 0.05 else "No",
                    })
            
            else:
                # Numeric factor - use t-test
                suppressed = self.patients[self.patients["viral_suppression"] == True][factor]
                not_suppressed = self.patients[self.patients["viral_suppression"] == False][factor]
                
                t_stat, p_value = stats.ttest_ind(suppressed, not_suppressed, nan_policy="omit")
                
                results.append({
                    "factor": factor,
                    "category": "continuous",
                    "mean_suppressed": round(suppressed.mean(), 2),
                    "mean_not_suppressed": round(not_suppressed.mean(), 2),
                    "difference": round(suppressed.mean() - not_suppressed.mean(), 2),
                    "test_statistic": round(t_stat, 4),
                    "p_value": round(p_value, 4),
                    "significant": "Yes" if p_value < 0.05 else "No",
                })
        
        results_df = pd.DataFrame(results)
        logger.info(f"Analyzed {len(factors)} factors for viral suppression")
        
        return results_df
    
    def compare_treatment_regimens(self) -> pd.DataFrame:
        """
        Compare effectiveness of different ART regimens.
        
        Returns:
            DataFrame with regimen comparison
        """
        logger.info("Comparing treatment regimen effectiveness")
        
        if self.patients is None or self.treatments is None:
            raise ValueError("Patient and treatment data required.")
        
        # Merge treatment data with outcomes
        merged = self.patients.merge(
            self.treatments[self.treatments["regimen_line"] == 1],  # First-line regimens
            on="patient_id",
            how="inner"
        )
        
        # Analyze by regimen
        regimen_analysis = (
            merged.groupby("regimen")
            .agg({
                "patient_id": "count",
                "viral_suppression": "mean",
                "treatment_adherence": lambda x: (x == "High").mean(),
                "cd4_count_at_diagnosis": "mean",
                "is_alive": "mean",
            })
            .reset_index()
        )
        
        regimen_analysis.columns = [
            "regimen",
            "n_patients",
            "viral_suppression_rate",
            "high_adherence_rate",
            "mean_cd4_at_start",
            "survival_rate",
        ]
        
        # Round percentages
        regimen_analysis["viral_suppression_rate"] = (
            regimen_analysis["viral_suppression_rate"] * 100
        ).round(2)
        regimen_analysis["high_adherence_rate"] = (
            regimen_analysis["high_adherence_rate"] * 100
        ).round(2)
        regimen_analysis["survival_rate"] = (
            regimen_analysis["survival_rate"] * 100
        ).round(2)
        regimen_analysis["mean_cd4_at_start"] = regimen_analysis["mean_cd4_at_start"].round(0)
        
        # Sort by effectiveness
        regimen_analysis = regimen_analysis.sort_values(
            "viral_suppression_rate", ascending=False
        )
        
        logger.info(f"Compared {len(regimen_analysis)} treatment regimens")
        
        return regimen_analysis
    
    def analyze_time_to_suppression(self) -> pd.DataFrame:
        """
        Analyze time to achieve viral suppression.
        
        Returns:
            DataFrame with time-to-suppression analysis
        """
        logger.info("Analyzing time to viral suppression")
        
        if self.lab_results is None:
            raise ValueError("Lab results data required.")
        
        # Find first suppression date for each patient
        suppressed = self.lab_results[self.lab_results["is_suppressed"] == True]
        
        first_suppression = (
            suppressed.sort_values("test_date")
            .groupby("patient_id")
            .first()
            .reset_index()
        )
        
        # Merge with patient data to get treatment start date
        if self.patients is not None:
            first_suppression = first_suppression.merge(
                self.patients[["patient_id", "treatment_start_date", "treatment_adherence"]],
                on="patient_id",
                how="left"
            )
            
            # Calculate days to suppression
            first_suppression["treatment_start_date"] = pd.to_datetime(
                first_suppression["treatment_start_date"]
            )
            first_suppression["test_date"] = pd.to_datetime(first_suppression["test_date"])
            
            first_suppression["days_to_suppression"] = (
                first_suppression["test_date"] - first_suppression["treatment_start_date"]
            ).dt.days
            
            # Analyze by adherence level
            time_analysis = (
                first_suppression.groupby("treatment_adherence")["days_to_suppression"]
                .agg(["mean", "median", "std", "min", "max", "count"])
                .round(1)
                .reset_index()
            )
            
            logger.info("Time to suppression analysis complete")
            
            return time_analysis
        
        return first_suppression
    
    def analyze_treatment_barriers(self) -> Dict[str, any]:
        """
        Identify barriers to successful treatment.
        
        Returns:
            Dictionary with barrier analysis
        """
        logger.info("Analyzing treatment barriers")
        
        if self.patients is None:
            raise ValueError("Patient data required.")
        
        # Identify patients with poor outcomes
        poor_outcomes = self.patients[
            (self.patients["viral_suppression"] == False) |
            (self.patients["treatment_adherence"] == "Low")
        ]
        
        good_outcomes = self.patients[
            (self.patients["viral_suppression"] == True) &
            (self.patients["treatment_adherence"] == "High")
        ]
        
        barriers = {
            "demographic_barriers": {
                "age": {
                    "poor_outcomes_mean": round(poor_outcomes["age"].mean(), 1),
                    "good_outcomes_mean": round(good_outcomes["age"].mean(), 1),
                },
                "gender_distribution": {
                    "poor_outcomes": poor_outcomes["gender"].value_counts(normalize=True).to_dict(),
                    "good_outcomes": good_outcomes["gender"].value_counts(normalize=True).to_dict(),
                },
            },
            "clinical_barriers": {
                "late_diagnosis": {
                    "poor_outcomes_rate": round(
                        (poor_outcomes["who_clinical_stage"].isin(["Stage 3", "Stage 4"])).mean() * 100, 2
                    ),
                    "good_outcomes_rate": round(
                        (good_outcomes["who_clinical_stage"].isin(["Stage 3", "Stage 4"])).mean() * 100, 2
                    ),
                },
                "low_cd4_at_diagnosis": {
                    "poor_outcomes_mean": round(poor_outcomes["cd4_count_at_diagnosis"].mean(), 0),
                    "good_outcomes_mean": round(good_outcomes["cd4_count_at_diagnosis"].mean(), 0),
                },
            },
            "geographic_barriers": {
                "countries_with_poor_outcomes": poor_outcomes["country_code"].value_counts().head(5).to_dict(),
            },
            "sample_sizes": {
                "poor_outcomes": len(poor_outcomes),
                "good_outcomes": len(good_outcomes),
                "total": len(self.patients),
            },
        }
        
        logger.info("Treatment barrier analysis complete")
        
        return barriers
    
    def perform_survival_analysis(
        self,
        duration_col: str = "days_since_diagnosis",
        event_col: str = "is_alive",
    ) -> Tuple[KaplanMeierFitter, pd.DataFrame]:
        """
        Perform Kaplan-Meier survival analysis.
        
        Args:
            duration_col: Column with duration/time data
            event_col: Column with event indicator (1=event occurred, 0=censored)
        
        Returns:
            Tuple of (fitted KM model, summary statistics)
        """
        logger.info("Performing survival analysis")
        
        if self.patients is None:
            raise ValueError("Patient data required.")
        
        # Prepare data
        data = self.patients.copy()
        
        # Calculate days since diagnosis if not present
        if duration_col not in data.columns:
            data["diagnosis_date"] = pd.to_datetime(data["diagnosis_date"])
            data["days_since_diagnosis"] = (
                pd.Timestamp.now() - data["diagnosis_date"]
            ).dt.days
        
        # Convert is_alive to death event (0=alive, 1=death)
        if event_col == "is_alive":
            data["death_event"] = (~data["is_alive"]).astype(int)
            event_col = "death_event"
        
        # Fit Kaplan-Meier model
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=data[duration_col],
            event_observed=data[event_col],
            label="Overall Survival"
        )
        
        # Generate summary statistics
        survival_summary = pd.DataFrame({
            "time_point_days": [180, 365, 730, 1825],  # 6mo, 1yr, 2yr, 5yr
            "survival_probability": [
                kmf.predict(t) for t in [180, 365, 730, 1825]
            ],
        })
        
        survival_summary["survival_probability"] = (
            survival_summary["survival_probability"] * 100
        ).round(2)
        
        logger.info("Survival analysis complete")
        
        return kmf, survival_summary
    
    def analyze_treatment_switches(self) -> pd.DataFrame:
        """
        Analyze patterns in treatment regimen switches.
        
        Returns:
            DataFrame with switch analysis
        """
        logger.info("Analyzing treatment regimen switches")
        
        if self.treatments is None:
            raise ValueError("Treatment data required.")
        
        # Count regimen switches per patient
        switches = (
            self.treatments.groupby("patient_id")["regimen_line"]
            .max()
            .reset_index()
        )
        switches["n_switches"] = switches["regimen_line"] - 1
        
        # Analyze reasons for switching
        switched_regimens = self.treatments[self.treatments["regimen_line"] > 1]
        
        if len(switched_regimens) > 0:
            switch_reasons = (
                switched_regimens["reason_for_change"]
                .value_counts()
                .reset_index()
            )
            switch_reasons.columns = ["reason", "count"]
            switch_reasons["percentage"] = (
                switch_reasons["count"] / switch_reasons["count"].sum() * 100
            ).round(2)
        else:
            switch_reasons = pd.DataFrame()
        
        # Summary statistics
        summary = {
            "total_patients": switches["patient_id"].nunique(),
            "patients_with_switches": (switches["n_switches"] > 0).sum(),
            "switch_rate": round((switches["n_switches"] > 0).mean() * 100, 2),
            "avg_switches_per_patient": round(switches["n_switches"].mean(), 2),
            "switch_reasons": switch_reasons.to_dict("records") if not switch_reasons.empty else [],
        }
        
        logger.info("Treatment switch analysis complete")
        
        return summary
    
    def generate_treatment_effectiveness_score(self) -> pd.DataFrame:
        """
        Calculate comprehensive treatment effectiveness scores.
        
        Returns:
            DataFrame with effectiveness scores
        """
        logger.info("Calculating treatment effectiveness scores")
        
        if self.patients is None:
            raise ValueError("Patient data required.")
        
        # Calculate multiple effectiveness metrics
        effectiveness = self.patients.groupby("country_code").agg({
            "viral_suppression": "mean",
            "treatment_adherence": lambda x: (x == "High").mean(),
            "is_alive": "mean",
            "patient_id": "count",
        }).reset_index()
        
        effectiveness.columns = [
            "country_code",
            "viral_suppression_rate",
            "high_adherence_rate",
            "survival_rate",
            "n_patients",
        ]
        
        # Calculate composite effectiveness score (0-100)
        effectiveness["effectiveness_score"] = (
            effectiveness["viral_suppression_rate"] * 0.4 +
            effectiveness["high_adherence_rate"] * 0.3 +
            effectiveness["survival_rate"] * 0.3
        ) * 100
        
        effectiveness["effectiveness_score"] = effectiveness["effectiveness_score"].round(2)
        
        # Convert rates to percentages
        for col in ["viral_suppression_rate", "high_adherence_rate", "survival_rate"]:
            effectiveness[col] = (effectiveness[col] * 100).round(2)
        
        # Sort by effectiveness score
        effectiveness = effectiveness.sort_values("effectiveness_score", ascending=False)
        
        logger.info("Treatment effectiveness scores calculated")
        
        return effectiveness


# Example usage
if __name__ == "__main__":
    from src.ingestion.data_generator import HIVDataGenerator
    
    # Generate synthetic data
    generator = HIVDataGenerator()
    datasets = generator.generate_complete_dataset(n_patients=10000)
    
    # Initialize analyzer
    analyzer = TreatmentAnalyzer(
        patients_data=datasets["patients"],
        lab_results_data=datasets["lab_results"],
        treatments_data=datasets["treatments"],
    )
    
    print("\n=== Treatment Efficacy Analysis ===\n")
    
    # 1. Viral suppression factors
    suppression_factors = analyzer.analyze_viral_suppression_factors()
    print("Factors Associated with Viral Suppression:")
    print(suppression_factors)
    
    # 2. Compare regimens
    regimen_comparison = analyzer.compare_treatment_regimens()
    print("\nTreatment Regimen Comparison:")
    print(regimen_comparison)
    
    # 3. Time to suppression
    time_to_suppression = analyzer.analyze_time_to_suppression()
    print("\nTime to Viral Suppression by Adherence:")
    print(time_to_suppression)
    
    # 4. Treatment barriers
    barriers = analyzer.analyze_treatment_barriers()
    print("\nTreatment Barriers:")
    print(f"Late diagnosis in poor outcomes: {barriers['clinical_barriers']['late_diagnosis']['poor_outcomes_rate']}%")
    
    # 5. Effectiveness scores
    effectiveness = analyzer.generate_treatment_effectiveness_score()
    print("\nTreatment Effectiveness Scores by Country:")
    print(effectiveness.head(10))

