"""
Synthetic Data Generator
=========================

Generate realistic synthetic HIV patient data for testing and development.
Maintains statistical properties of real data while ensuring privacy.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
from faker import Faker
from src.utils.logger import get_logger

logger = get_logger(__name__)
fake = Faker()


class HIVDataGenerator:
    """
    Generate synthetic HIV patient and clinical data.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        Faker.seed(seed)
        
        # Configuration for realistic distributions
        self.config = self._setup_distributions()
        
        logger.info("HIV data generator initialized")
    
    def _setup_distributions(self) -> dict:
        """Setup probability distributions based on real-world HIV statistics."""
        return {
            # Age distribution (peaks in 25-44 age group)
            "age_mean": 35,
            "age_std": 12,
            "age_min": 15,
            "age_max": 75,
            
            # Gender distribution
            "gender_dist": {
                "Male": 0.52,
                "Female": 0.47,
                "Other": 0.01,
            },
            
            # Transmission routes (global averages)
            "transmission_routes": {
                "Heterosexual": 0.45,
                "MSM": 0.30,  # Men who have sex with men
                "IDU": 0.10,  # Injection drug use
                "MTCT": 0.05,  # Mother-to-child transmission
                "Blood_Products": 0.02,
                "Unknown": 0.08,
            },
            
            # Geographic distribution (top 10 countries by prevalence)
            "countries": {
                "ZAF": 0.25,  # South Africa
                "NGA": 0.15,  # Nigeria
                "MOZ": 0.10,  # Mozambique
                "IND": 0.10,  # India
                "KEN": 0.08,  # Kenya
                "UGA": 0.08,  # Uganda
                "USA": 0.07,  # United States
                "TZA": 0.07,  # Tanzania
                "ZWE": 0.05,  # Zimbabwe
                "BRA": 0.05,  # Brazil
            },
            
            # CD4 count at diagnosis (cells/μL)
            "cd4_mean": 350,
            "cd4_std": 200,
            "cd4_min": 0,
            "cd4_max": 1500,
            
            # Viral load at diagnosis (copies/mL)
            "viral_load_mean": 50000,
            "viral_load_std": 100000,
            "viral_load_min": 50,
            "viral_load_max": 1000000,
            
            # WHO clinical stages
            "who_stages": {
                "Stage 1": 0.30,
                "Stage 2": 0.35,
                "Stage 3": 0.25,
                "Stage 4": 0.10,
            },
            
            # Treatment outcomes
            "viral_suppression_rate": 0.85,
            "treatment_adherence": {
                "High": 0.70,
                "Medium": 0.20,
                "Low": 0.10,
            },
        }
    
    def generate_patients(self, n_patients: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic patient demographic data.
        
        Args:
            n_patients: Number of patients to generate
        
        Returns:
            DataFrame with patient data
        """
        logger.info(f"Generating {n_patients} synthetic patients")
        
        patients = []
        
        for i in range(n_patients):
            patient_id = f"PAT{str(i+1).zfill(8)}"
            
            # Demographics
            age = int(np.clip(
                np.random.normal(self.config["age_mean"], self.config["age_std"]),
                self.config["age_min"],
                self.config["age_max"]
            ))
            
            gender = np.random.choice(
                list(self.config["gender_dist"].keys()),
                p=list(self.config["gender_dist"].values())
            )
            
            country = np.random.choice(
                list(self.config["countries"].keys()),
                p=list(self.config["countries"].values())
            )
            
            # Diagnosis date (between 2010 and 2024)
            days_ago = np.random.randint(0, 365 * 14)
            diagnosis_date = datetime.now() - timedelta(days=days_ago)
            
            # Transmission route
            transmission_route = np.random.choice(
                list(self.config["transmission_routes"].keys()),
                p=list(self.config["transmission_routes"].values())
            )
            
            # Clinical data at diagnosis
            cd4_count = int(np.clip(
                np.random.normal(self.config["cd4_mean"], self.config["cd4_std"]),
                self.config["cd4_min"],
                self.config["cd4_max"]
            ))
            
            viral_load = int(np.clip(
                np.random.lognormal(np.log(self.config["viral_load_mean"]), 1.5),
                self.config["viral_load_min"],
                self.config["viral_load_max"]
            ))
            
            who_stage = np.random.choice(
                list(self.config["who_stages"].keys()),
                p=list(self.config["who_stages"].values())
            )
            
            # Treatment start date (usually within 30 days of diagnosis)
            treatment_delay = np.random.randint(0, 30)
            treatment_start_date = diagnosis_date + timedelta(days=treatment_delay)
            
            # Treatment adherence
            adherence = np.random.choice(
                list(self.config["treatment_adherence"].keys()),
                p=list(self.config["treatment_adherence"].values())
            )
            
            # Viral suppression (higher chance with high adherence)
            suppression_prob = {
                "High": 0.95,
                "Medium": 0.75,
                "Low": 0.40,
            }
            is_suppressed = np.random.random() < suppression_prob[adherence]
            
            # Current status
            alive_prob = 0.95
            is_alive = np.random.random() < alive_prob
            
            patients.append({
                "patient_id": patient_id,
                "age": age,
                "gender": gender,
                "country_code": country,
                "diagnosis_date": diagnosis_date.strftime("%Y-%m-%d"),
                "transmission_route": transmission_route,
                "cd4_count_at_diagnosis": cd4_count,
                "viral_load_at_diagnosis": viral_load,
                "who_clinical_stage": who_stage,
                "treatment_start_date": treatment_start_date.strftime("%Y-%m-%d"),
                "treatment_adherence": adherence,
                "viral_suppression": is_suppressed,
                "is_alive": is_alive,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
        
        df = pd.DataFrame(patients)
        logger.info(f"Generated {len(df)} patient records")
        
        return df
    
    def generate_lab_results(
        self,
        patient_ids: List[str],
        start_date: datetime,
        end_date: datetime,
        frequency_days: int = 90,
    ) -> pd.DataFrame:
        """
        Generate synthetic lab results (CD4 counts and viral loads) over time.
        
        Args:
            patient_ids: List of patient IDs
            start_date: Start date for lab results
            end_date: End date for lab results
            frequency_days: Days between lab tests
        
        Returns:
            DataFrame with lab results
        """
        logger.info(f"Generating lab results for {len(patient_ids)} patients")
        
        lab_results = []
        
        for patient_id in patient_ids:
            current_date = start_date
            
            # Initial CD4 and viral load
            cd4 = int(np.random.normal(350, 200))
            viral_load = int(np.random.lognormal(np.log(50000), 1.5))
            
            while current_date <= end_date:
                # CD4 count generally improves on treatment
                cd4_change = int(np.random.normal(20, 50))
                cd4 = max(0, min(1500, cd4 + cd4_change))
                
                # Viral load decreases with treatment
                if np.random.random() < 0.8:  # 80% show improvement
                    viral_load = int(viral_load * np.random.uniform(0.5, 0.9))
                else:
                    viral_load = int(viral_load * np.random.uniform(1.0, 1.2))
                
                viral_load = max(50, min(1000000, viral_load))
                
                lab_results.append({
                    "patient_id": patient_id,
                    "test_date": current_date.strftime("%Y-%m-%d"),
                    "cd4_count": cd4,
                    "viral_load": viral_load,
                    "is_suppressed": viral_load < 200,
                })
                
                # Next test date
                current_date += timedelta(days=frequency_days)
        
        df = pd.DataFrame(lab_results)
        logger.info(f"Generated {len(df)} lab result records")
        
        return df
    
    def generate_treatment_regimens(self, patient_ids: List[str]) -> pd.DataFrame:
        """
        Generate synthetic treatment regimen data.
        
        Args:
            patient_ids: List of patient IDs
        
        Returns:
            DataFrame with treatment regimens
        """
        logger.info(f"Generating treatment regimens for {len(patient_ids)} patients")
        
        # Common ART regimens
        regimens = [
            "TDF/3TC/EFV",  # Tenofovir + Lamivudine + Efavirenz
            "TDF/3TC/DTG",  # Tenofovir + Lamivudine + Dolutegravir
            "ABC/3TC/DTG",  # Abacavir + Lamivudine + Dolutegravir
            "TAF/FTC/BIC",  # Tenofovir alafenamide + Emtricitabine + Bictegravir
            "DRV/r + TDF/FTC",  # Darunavir + Tenofovir + Emtricitabine
        ]
        
        treatments = []
        
        for patient_id in patient_ids:
            regimen = np.random.choice(regimens)
            
            # Some patients switch regimens
            n_regimens = int(np.random.choice([1, 2, 3], p=[0.7, 0.25, 0.05]))
            
            for i in range(n_regimens):
                treatments.append({
                    "patient_id": patient_id,
                    "regimen": regimen if i == 0 else np.random.choice(regimens),
                    "regimen_line": i + 1,
                    "start_date": (datetime.now() - timedelta(days=int(365 * (n_regimens - i)))).strftime("%Y-%m-%d"),
                    "end_date": None if i == n_regimens - 1 else (datetime.now() - timedelta(days=int(365 * (n_regimens - i - 1)))).strftime("%Y-%m-%d"),
                    "reason_for_change": None if i == 0 else np.random.choice([
                        "Side effects",
                        "Treatment failure",
                        "Drug interaction",
                        "Simplification",
                        "Drug resistance",
                    ]),
                })
        
        df = pd.DataFrame(treatments)
        logger.info(f"Generated {len(df)} treatment regimen records")
        
        return df
    
    def generate_complete_dataset(self, n_patients: int = 1000) -> dict:
        """
        Generate a complete synthetic dataset including patients, lab results, and treatments.
        
        Args:
            n_patients: Number of patients to generate
        
        Returns:
            Dictionary containing all datasets
        """
        logger.info(f"Generating complete synthetic dataset with {n_patients} patients")
        
        # Generate patients
        patients_df = self.generate_patients(n_patients)
        
        # Generate lab results
        patient_ids = patients_df["patient_id"].tolist()
        start_date = datetime.now() - timedelta(days=365 * 2)  # 2 years of history
        end_date = datetime.now()
        
        lab_results_df = self.generate_lab_results(
            patient_ids, start_date, end_date, frequency_days=90
        )
        
        # Generate treatment regimens
        treatments_df = self.generate_treatment_regimens(patient_ids)
        
        logger.info("Complete synthetic dataset generated successfully")
        
        return {
            "patients": patients_df,
            "lab_results": lab_results_df,
            "treatments": treatments_df,
        }
    
    def save_datasets(self, datasets: dict, output_dir: str = "data/raw"):
        """
        Save generated datasets to files.
        
        Args:
            datasets: Dictionary of datasets
            output_dir: Output directory
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in datasets.items():
            output_path = f"{output_dir}/synthetic_{name}.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved {name} dataset to {output_path}")


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    generator = HIVDataGenerator(seed=42)
    
    # Generate complete dataset
    datasets = generator.generate_complete_dataset(n_patients=5000)
    
    # Display summary
    print("\n=== Synthetic Data Summary ===")
    for name, df in datasets.items():
        print(f"\n{name.upper()}:")
        print(f"  Records: {len(df)}")
        print(f"  Columns: {', '.join(df.columns[:5])}...")
        print(f"\nSample data:\n{df.head(3)}")
    
    # Save datasets
    generator.save_datasets(datasets)
    print("\nDatasets saved to data/raw/")

