#!/usr/bin/env python3
"""
Run HIV Analytics
=================

Script to run comprehensive HIV analytics and generate reports.
"""

import sys
from pathlib import Path
import argparse
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analytics.transmission.transmission_analyzer import TransmissionAnalyzer
from src.analytics.treatment.treatment_analyzer import TreatmentAnalyzer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run HIV medical analytics"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Data directory (default: data/raw)",
    )
    parser.add_argument(
        "--analysis-type",
        type=str,
        choices=["all", "transmission", "treatment"],
        default="all",
        help="Type of analysis to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/generated",
        help="Output directory for reports (default: reports/generated)",
    )
    
    args = parser.parse_args()
    
    logger.info(f"Running HIV analytics: {args.analysis_type}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📊 Loading data...")
    try:
        patients_df = pd.read_parquet(f"{args.data_dir}/synthetic_patients.parquet")
        lab_results_df = pd.read_parquet(f"{args.data_dir}/synthetic_lab_results.parquet")
        treatments_df = pd.read_parquet(f"{args.data_dir}/synthetic_treatments.parquet")
        
        print(f"✅ Loaded {len(patients_df):,} patients")
        print(f"✅ Loaded {len(lab_results_df):,} lab results")
        print(f"✅ Loaded {len(treatments_df):,} treatment records")
    
    except FileNotFoundError:
        print("\n❌ Data files not found!")
        print("Please run: python scripts/generate_synthetic_data.py")
        sys.exit(1)
    
    # Run analyses
    print("\n" + "=" * 60)
    print("RUNNING ANALYSES")
    print("=" * 60)
    
    if args.analysis_type in ["all", "transmission"]:
        run_transmission_analysis(patients_df, args.output_dir)
    
    if args.analysis_type in ["all", "treatment"]:
        run_treatment_analysis(patients_df, lab_results_df, treatments_df, args.output_dir)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE ✅")
    print("=" * 60)
    print(f"\nReports saved to: {args.output_dir}")


def run_transmission_analysis(patients_df, output_dir):
    """Run transmission analysis."""
    print("\n🔬 TRANSMISSION ANALYSIS")
    print("-" * 60)
    
    analyzer = TransmissionAnalyzer(data=patients_df)
    
    # 1. Summary report
    print("\n1. Generating summary report...")
    report = analyzer.generate_summary_report()
    
    print(f"\n   Total Patients: {report['total_patients']:,}")
    print(f"   Date Range: {report['date_range']['start']} to {report['date_range']['end']}")
    print(f"   Most Common Route: {report['transmission_routes']['most_common']} ({report['transmission_routes']['percentage_most_common']}%)")
    
    # 2. Demographic analysis
    print("\n2. Analyzing transmission by demographics...")
    demo_analysis = analyzer.analyze_transmission_by_demographic()
    demo_analysis.to_csv(f"{output_dir}/transmission_demographics.csv", index=False)
    print(f"   ✅ Saved to {output_dir}/transmission_demographics.csv")
    
    # 3. High-risk populations
    print("\n3. Identifying high-risk populations...")
    high_risk = analyzer.identify_high_risk_populations(top_n=10)
    high_risk.to_csv(f"{output_dir}/high_risk_populations.csv", index=False)
    print(f"   ✅ Saved to {output_dir}/high_risk_populations.csv")
    print("\n   Top 3 High-Risk Populations:")
    for i, row in high_risk.head(3).iterrows():
        print(f"   {i+1}. {row['age_group']}, {row['gender']}, {row['transmission_route']}: {row['count']} cases")
    
    # 4. Risk scores
    print("\n4. Calculating transmission risk scores...")
    risk_scores = analyzer.calculate_transmission_risk_scores()
    risk_scores.to_csv(f"{output_dir}/transmission_risk_scores.csv", index=False)
    print(f"   ✅ Saved to {output_dir}/transmission_risk_scores.csv")
    
    print("\n   Risk Scores by Route:")
    for _, row in risk_scores.iterrows():
        print(f"   - {row['transmission_route']}: {row['composite_risk_score']:.1f} ({row['risk_category']})")


def run_treatment_analysis(patients_df, lab_results_df, treatments_df, output_dir):
    """Run treatment efficacy analysis."""
    print("\n💊 TREATMENT EFFICACY ANALYSIS")
    print("-" * 60)
    
    analyzer = TreatmentAnalyzer(
        patients_data=patients_df,
        lab_results_data=lab_results_df,
        treatments_data=treatments_df,
    )
    
    # 1. Viral suppression factors
    print("\n1. Analyzing viral suppression factors...")
    suppression_factors = analyzer.analyze_viral_suppression_factors()
    suppression_factors.to_csv(f"{output_dir}/viral_suppression_factors.csv", index=False)
    print(f"   ✅ Saved to {output_dir}/viral_suppression_factors.csv")
    
    # Display significant factors
    significant = suppression_factors[suppression_factors["significant"] == "Yes"]
    if len(significant) > 0:
        print(f"\n   Found {len(significant)} significant factors:")
        for _, row in significant.head(5).iterrows():
            print(f"   - {row['factor']}: p-value = {row['p_value']:.4f}")
    
    # 2. Regimen comparison
    print("\n2. Comparing treatment regimens...")
    regimen_comparison = analyzer.compare_treatment_regimens()
    regimen_comparison.to_csv(f"{output_dir}/regimen_comparison.csv", index=False)
    print(f"   ✅ Saved to {output_dir}/regimen_comparison.csv")
    
    print("\n   Top 3 Most Effective Regimens:")
    for i, row in regimen_comparison.head(3).iterrows():
        print(f"   {i+1}. {row['regimen']}: {row['viral_suppression_rate']:.1f}% suppression rate")
    
    # 3. Time to suppression
    print("\n3. Analyzing time to viral suppression...")
    time_to_suppression = analyzer.analyze_time_to_suppression()
    time_to_suppression.to_csv(f"{output_dir}/time_to_suppression.csv", index=False)
    print(f"   ✅ Saved to {output_dir}/time_to_suppression.csv")
    
    # 4. Treatment effectiveness
    print("\n4. Calculating treatment effectiveness scores...")
    effectiveness = analyzer.generate_treatment_effectiveness_score()
    effectiveness.to_csv(f"{output_dir}/treatment_effectiveness.csv", index=False)
    print(f"   ✅ Saved to {output_dir}/treatment_effectiveness.csv")
    
    print("\n   Top 5 Countries by Effectiveness:")
    for i, row in effectiveness.head(5).iterrows():
        print(f"   {i+1}. {row['country_code']}: {row['effectiveness_score']:.1f} score")


if __name__ == "__main__":
    main()

