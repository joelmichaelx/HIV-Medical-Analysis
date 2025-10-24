"""
Test Transmission Analyzer
===========================

Unit tests for the transmission analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from src.analytics.transmission.transmission_analyzer import TransmissionAnalyzer
from src.ingestion.data_generator import HIVDataGenerator


@pytest.fixture
def sample_data():
    """Generate sample HIV patient data for testing."""
    generator = HIVDataGenerator(seed=42)
    datasets = generator.generate_complete_dataset(n_patients=1000)
    return datasets["patients"]


@pytest.fixture
def analyzer(sample_data):
    """Create a TransmissionAnalyzer instance with sample data."""
    return TransmissionAnalyzer(data=sample_data)


class TestTransmissionAnalyzer:
    """Test cases for TransmissionAnalyzer."""
    
    def test_initialization(self, sample_data):
        """Test analyzer initialization."""
        analyzer = TransmissionAnalyzer(data=sample_data)
        assert analyzer.data is not None
        assert len(analyzer.data) == 1000
    
    def test_analyze_transmission_by_demographic(self, analyzer):
        """Test transmission analysis by demographic variables."""
        result = analyzer.analyze_transmission_by_demographic(
            demographic_vars=["gender", "age_group"]
        )
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "gender" in result.columns
        assert "age_group" in result.columns
        assert "transmission_route" in result.columns
        assert "count" in result.columns
    
    def test_analyze_temporal_trends(self, analyzer):
        """Test temporal trend analysis."""
        trends = analyzer.analyze_temporal_trends(frequency="Y")
        
        assert isinstance(trends, pd.DataFrame)
        assert not trends.empty
        assert "period" in trends.columns
        assert "transmission_route" in trends.columns
        assert "count" in trends.columns
        assert "percentage" in trends.columns
    
    def test_identify_high_risk_populations(self, analyzer):
        """Test high-risk population identification."""
        high_risk = analyzer.identify_high_risk_populations(top_n=5)
        
        assert isinstance(high_risk, pd.DataFrame)
        assert len(high_risk) <= 5
        assert "age_group" in high_risk.columns
        assert "gender" in high_risk.columns
        assert "transmission_route" in high_risk.columns
        assert "count" in high_risk.columns
        assert "percentage_of_total" in high_risk.columns
    
    def test_analyze_geographic_patterns(self, analyzer):
        """Test geographic pattern analysis."""
        geo_patterns = analyzer.analyze_geographic_patterns()
        
        assert isinstance(geo_patterns, pd.DataFrame)
        assert not geo_patterns.empty
        assert "country_code" in geo_patterns.columns
        assert "transmission_route" in geo_patterns.columns
        assert "count" in geo_patterns.columns
        assert "percentage" in geo_patterns.columns
    
    def test_calculate_transmission_risk_scores(self, analyzer):
        """Test risk score calculation."""
        risk_scores = analyzer.calculate_transmission_risk_scores()
        
        assert isinstance(risk_scores, pd.DataFrame)
        assert not risk_scores.empty
        assert "transmission_route" in risk_scores.columns
        assert "composite_risk_score" in risk_scores.columns
        assert "risk_category" in risk_scores.columns
        
        # Check risk scores are within valid range
        assert risk_scores["composite_risk_score"].min() >= 0
        assert risk_scores["composite_risk_score"].max() <= 100
    
    def test_generate_summary_report(self, analyzer):
        """Test summary report generation."""
        report = analyzer.generate_summary_report()
        
        assert isinstance(report, dict)
        assert "total_patients" in report
        assert report["total_patients"] == 1000
        assert "transmission_routes" in report
        assert "demographics" in report
        assert "high_risk_populations" in report
        assert "risk_scores" in report
    
    def test_empty_data(self):
        """Test analyzer with empty data."""
        empty_df = pd.DataFrame()
        analyzer = TransmissionAnalyzer(data=empty_df)
        
        with pytest.raises(Exception):
            analyzer.analyze_transmission_by_demographic()
    
    def test_missing_columns(self):
        """Test analyzer with missing columns."""
        incomplete_df = pd.DataFrame({
            "patient_id": [1, 2, 3],
            "age": [25, 35, 45],
        })
        
        analyzer = TransmissionAnalyzer(data=incomplete_df)
        
        # Should handle missing columns gracefully
        with pytest.raises(KeyError):
            analyzer.analyze_transmission_by_demographic()


class TestVisualization:
    """Test visualization methods."""
    
    def test_visualize_transmission_distribution(self, analyzer):
        """Test transmission distribution visualization."""
        fig = analyzer.visualize_transmission_distribution()
        
        assert fig is not None
        # Check that figure has data
        assert len(fig.data) > 0
    
    def test_visualize_temporal_trends(self, analyzer):
        """Test temporal trends visualization."""
        fig = analyzer.visualize_temporal_trends(frequency="Y")
        
        assert fig is not None
        assert len(fig.data) > 0


@pytest.mark.parametrize("n_patients", [100, 500, 1000])
def test_different_sample_sizes(n_patients):
    """Test analyzer with different sample sizes."""
    generator = HIVDataGenerator(seed=42)
    datasets = generator.generate_complete_dataset(n_patients=n_patients)
    
    analyzer = TransmissionAnalyzer(data=datasets["patients"])
    result = analyzer.analyze_transmission_by_demographic()
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


@pytest.mark.parametrize("demographic_var", ["gender", "age_group", "country_code"])
def test_different_demographic_variables(analyzer, demographic_var):
    """Test analysis with different demographic variables."""
    result = analyzer.analyze_transmission_by_demographic([demographic_var])
    
    assert isinstance(result, pd.DataFrame)
    assert demographic_var in result.columns
    assert "transmission_route" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

