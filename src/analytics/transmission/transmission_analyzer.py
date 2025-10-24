"""
Transmission Route Analysis
============================

Analyze HIV transmission patterns and answer questions about transmission routes.

Key Questions Answered:
1. What are the most common transmission routes by demographic group?
2. How have transmission patterns changed over time?
3. Which populations are at highest risk?
4. What are the geographic variations in transmission routes?
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TransmissionAnalyzer:
    """
    Analyze HIV transmission patterns and routes.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the transmission analyzer.
        
        Args:
            data: DataFrame with patient data including transmission routes
        """
        self.data = data
        logger.info("Transmission analyzer initialized")
    
    def load_data(self, data_path: str):
        """
        Load patient data from file.
        
        Args:
            data_path: Path to data file
        """
        self.data = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(self.data)} records from {data_path}")
    
    def analyze_transmission_by_demographic(
        self,
        demographic_vars: List[str] = ["age_group", "gender", "country_code"],
    ) -> pd.DataFrame:
        """
        Analyze transmission routes by demographic variables.
        
        Args:
            demographic_vars: List of demographic variables to analyze
        
        Returns:
            DataFrame with transmission route counts by demographics
        """
        logger.info(f"Analyzing transmission routes by {demographic_vars}")
        
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Create age groups if not present
        if "age_group" in demographic_vars and "age_group" not in self.data.columns:
            self.data["age_group"] = pd.cut(
                self.data["age"],
                bins=[0, 15, 24, 34, 44, 54, 64, 100],
                labels=["0-14", "15-24", "25-34", "35-44", "45-54", "55-64", "65+"],
            )
        
        # Group by demographics and transmission route
        groupby_vars = demographic_vars + ["transmission_route"]
        
        results = (
            self.data.groupby(groupby_vars)
            .size()
            .reset_index(name="count")
        )
        
        # Calculate percentages within each demographic group
        for var in demographic_vars:
            total_by_group = (
                results.groupby(var)["count"]
                .transform("sum")
            )
            results[f"{var}_percentage"] = (results["count"] / total_by_group * 100).round(2)
        
        logger.info(f"Generated transmission analysis with {len(results)} rows")
        
        return results
    
    def analyze_temporal_trends(
        self,
        frequency: str = "Y",
        transmission_routes: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Analyze how transmission patterns have changed over time.
        
        Args:
            frequency: Time frequency for aggregation ('Y'=yearly, 'M'=monthly, 'Q'=quarterly)
            transmission_routes: Optional list of specific routes to analyze
        
        Returns:
            DataFrame with temporal trends
        """
        logger.info(f"Analyzing temporal trends with {frequency} frequency")
        
        if self.data is None:
            raise ValueError("No data loaded.")
        
        # Ensure diagnosis_date is datetime
        self.data["diagnosis_date"] = pd.to_datetime(self.data["diagnosis_date"])
        
        # Create time period
        self.data["period"] = self.data["diagnosis_date"].dt.to_period(frequency)
        
        # Filter transmission routes if specified
        data = self.data.copy()
        if transmission_routes:
            data = data[data["transmission_route"].isin(transmission_routes)]
        
        # Count by period and transmission route
        trends = (
            data.groupby(["period", "transmission_route"])
            .size()
            .reset_index(name="count")
        )
        
        # Convert period back to timestamp for plotting
        trends["period"] = trends["period"].dt.to_timestamp()
        
        # Calculate percentage by period
        total_by_period = (
            trends.groupby("period")["count"]
            .transform("sum")
        )
        trends["percentage"] = (trends["count"] / total_by_period * 100).round(2)
        
        logger.info(f"Generated temporal trends with {len(trends)} data points")
        
        return trends
    
    def identify_high_risk_populations(
        self,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Identify populations at highest risk based on transmission patterns.
        
        Args:
            top_n: Number of top risk groups to return
        
        Returns:
            DataFrame with high-risk population segments
        """
        logger.info("Identifying high-risk populations")
        
        if self.data is None:
            raise ValueError("No data loaded.")
        
        # Create age groups
        if "age_group" not in self.data.columns:
            self.data["age_group"] = pd.cut(
                self.data["age"],
                bins=[0, 15, 24, 34, 44, 54, 64, 100],
                labels=["0-14", "15-24", "25-34", "35-44", "45-54", "55-64", "65+"],
            )
        
        # Analyze risk by multiple dimensions
        risk_groups = (
            self.data.groupby(["age_group", "gender", "transmission_route", "country_code"])
            .size()
            .reset_index(name="count")
        )
        
        # Calculate incidence rate (assuming population data available)
        # For now, use count as proxy for risk
        risk_groups = risk_groups.sort_values("count", ascending=False).head(top_n)
        
        # Calculate percentage of total
        total_cases = self.data.shape[0]
        risk_groups["percentage_of_total"] = (
            risk_groups["count"] / total_cases * 100
        ).round(2)
        
        logger.info(f"Identified top {len(risk_groups)} high-risk populations")
        
        return risk_groups
    
    def analyze_geographic_patterns(self) -> pd.DataFrame:
        """
        Analyze geographic variations in transmission routes.
        
        Returns:
            DataFrame with geographic transmission patterns
        """
        logger.info("Analyzing geographic transmission patterns")
        
        if self.data is None:
            raise ValueError("No data loaded.")
        
        # Group by country and transmission route
        geo_patterns = (
            self.data.groupby(["country_code", "transmission_route"])
            .size()
            .reset_index(name="count")
        )
        
        # Calculate percentage within each country
        total_by_country = (
            geo_patterns.groupby("country_code")["count"]
            .transform("sum")
        )
        geo_patterns["percentage"] = (
            geo_patterns["count"] / total_by_country * 100
        ).round(2)
        
        # Find dominant transmission route per country
        dominant_route = (
            geo_patterns.loc[geo_patterns.groupby("country_code")["count"].idxmax()]
            [["country_code", "transmission_route", "count"]]
            .rename(columns={"transmission_route": "dominant_route", "count": "dominant_count"})
        )
        
        # Merge back
        geo_patterns = geo_patterns.merge(dominant_route, on="country_code", how="left")
        
        logger.info(f"Generated geographic analysis for {len(geo_patterns['country_code'].unique())} countries")
        
        return geo_patterns
    
    def calculate_transmission_risk_scores(self) -> pd.DataFrame:
        """
        Calculate risk scores for different transmission routes based on multiple factors.
        
        Returns:
            DataFrame with risk scores by transmission route
        """
        logger.info("Calculating transmission risk scores")
        
        if self.data is None:
            raise ValueError("No data loaded.")
        
        # Calculate various risk metrics
        risk_metrics = []
        
        for route in self.data["transmission_route"].unique():
            route_data = self.data[self.data["transmission_route"] == route]
            
            # Metrics
            avg_cd4_at_diagnosis = route_data["cd4_count_at_diagnosis"].mean()
            avg_viral_load = route_data["viral_load_at_diagnosis"].mean()
            late_diagnosis_rate = (route_data["who_clinical_stage"].isin(["Stage 3", "Stage 4"])).mean()
            treatment_adherence_rate = (route_data["treatment_adherence"] == "High").mean()
            viral_suppression_rate = route_data["viral_suppression"].mean()
            
            # Calculate composite risk score (lower is worse)
            # Normalize components to 0-100 scale
            cd4_score = min(100, (avg_cd4_at_diagnosis / 500) * 100)  # 500 is healthy CD4
            adherence_score = treatment_adherence_rate * 100
            suppression_score = viral_suppression_rate * 100
            diagnosis_score = (1 - late_diagnosis_rate) * 100
            
            # Weighted composite score
            composite_score = (
                cd4_score * 0.3 +
                adherence_score * 0.25 +
                suppression_score * 0.25 +
                diagnosis_score * 0.20
            )
            
            risk_metrics.append({
                "transmission_route": route,
                "count": len(route_data),
                "avg_cd4_at_diagnosis": round(avg_cd4_at_diagnosis, 1),
                "avg_viral_load": round(avg_viral_load, 1),
                "late_diagnosis_rate": round(late_diagnosis_rate * 100, 2),
                "treatment_adherence_rate": round(treatment_adherence_rate * 100, 2),
                "viral_suppression_rate": round(viral_suppression_rate * 100, 2),
                "composite_risk_score": round(composite_score, 2),
                "risk_category": "Low" if composite_score >= 75 else "Medium" if composite_score >= 50 else "High",
            })
        
        risk_df = pd.DataFrame(risk_metrics).sort_values("composite_risk_score", ascending=False)
        
        logger.info("Calculated risk scores for all transmission routes")
        
        return risk_df
    
    def visualize_transmission_distribution(self, save_path: Optional[str] = None):
        """
        Create visualization of transmission route distribution.
        
        Args:
            save_path: Optional path to save the figure
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        # Count by transmission route
        route_counts = (
            self.data["transmission_route"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "route", "transmission_route": "count"})
        )
        
        # Create pie chart
        fig = px.pie(
            route_counts,
            values="count",
            names="route",
            title="HIV Transmission Route Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        
        fig.update_traces(textposition="inside", textinfo="percent+label")
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved visualization to {save_path}")
        
        return fig
    
    def visualize_temporal_trends(
        self,
        frequency: str = "Y",
        save_path: Optional[str] = None,
    ):
        """
        Visualize transmission trends over time.
        
        Args:
            frequency: Time frequency
            save_path: Optional path to save
        """
        trends = self.analyze_temporal_trends(frequency)
        
        fig = px.line(
            trends,
            x="period",
            y="percentage",
            color="transmission_route",
            title="HIV Transmission Route Trends Over Time",
            labels={"period": "Time Period", "percentage": "Percentage (%)"},
        )
        
        fig.update_layout(hovermode="x unified")
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved visualization to {save_path}")
        
        return fig
    
    def generate_summary_report(self) -> Dict[str, any]:
        """
        Generate comprehensive summary report on transmission patterns.
        
        Returns:
            Dictionary with summary statistics and insights
        """
        logger.info("Generating transmission analysis summary report")
        
        if self.data is None:
            raise ValueError("No data loaded.")
        
        report = {
            "total_patients": len(self.data),
            "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "date_range": {
                "start": self.data["diagnosis_date"].min(),
                "end": self.data["diagnosis_date"].max(),
            },
            "transmission_routes": {
                "distribution": self.data["transmission_route"].value_counts().to_dict(),
                "most_common": self.data["transmission_route"].value_counts().index[0],
                "percentage_most_common": round(
                    self.data["transmission_route"].value_counts(normalize=True).iloc[0] * 100, 2
                ),
            },
            "demographics": {
                "age_distribution": self.data["age"].describe().to_dict(),
                "gender_distribution": self.data["gender"].value_counts().to_dict(),
                "countries_affected": len(self.data["country_code"].unique()),
            },
            "high_risk_populations": self.identify_high_risk_populations(top_n=5).to_dict("records"),
            "risk_scores": self.calculate_transmission_risk_scores().to_dict("records"),
        }
        
        logger.info("Summary report generated successfully")
        
        return report


# Example usage
if __name__ == "__main__":
    # This would typically load real data
    # For demonstration, we'll use synthetic data
    from src.ingestion.data_generator import HIVDataGenerator
    
    # Generate synthetic data
    generator = HIVDataGenerator()
    datasets = generator.generate_complete_dataset(n_patients=10000)
    
    # Initialize analyzer
    analyzer = TransmissionAnalyzer(data=datasets["patients"])
    
    # Run analyses
    print("\n=== Transmission Analysis ===\n")
    
    # 1. Demographic analysis
    demo_analysis = analyzer.analyze_transmission_by_demographic()
    print("Transmission by Demographics (sample):")
    print(demo_analysis.head(10))
    
    # 2. Temporal trends
    trends = analyzer.analyze_temporal_trends()
    print("\nTemporal Trends (sample):")
    print(trends.head(10))
    
    # 3. High-risk populations
    high_risk = analyzer.identify_high_risk_populations()
    print("\nHigh-Risk Populations:")
    print(high_risk)
    
    # 4. Risk scores
    risk_scores = analyzer.calculate_transmission_risk_scores()
    print("\nTransmission Risk Scores:")
    print(risk_scores)
    
    # 5. Generate summary report
    report = analyzer.generate_summary_report()
    print("\nSummary Report:")
    print(f"Total Patients: {report['total_patients']}")
    print(f"Most Common Route: {report['transmission_routes']['most_common']} ({report['transmission_routes']['percentage_most_common']}%)")

