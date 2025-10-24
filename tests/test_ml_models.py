"""
Test Machine Learning Models
==============================

Unit tests for ML prediction models.
"""

import pytest
import pandas as pd
import numpy as np
from src.ml.models.viral_suppression_predictor import ViralSuppressionPredictor
from src.ingestion.data_generator import HIVDataGenerator


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    generator = HIVDataGenerator(seed=42)
    datasets = generator.generate_complete_dataset(n_patients=500)
    return datasets["patients"]


@pytest.fixture
def predictor():
    """Create a predictor instance."""
    return ViralSuppressionPredictor(model_type="logistic")  # Fast model for testing


class TestViralSuppressionPredictor:
    """Test cases for viral suppression prediction model."""
    
    def test_initialization(self):
        """Test predictor initialization."""
        predictor = ViralSuppressionPredictor(model_type="xgboost")
        assert predictor.model_type == "xgboost"
        assert predictor.model is None
        assert predictor.feature_names == []
    
    def test_prepare_features(self, predictor, sample_data):
        """Test feature preparation."""
        prepared = predictor.prepare_features(sample_data)
        
        assert isinstance(prepared, pd.DataFrame)
        assert "age_group" in prepared.columns
        assert "days_to_treatment" in prepared.columns
        assert "cd4_category" in prepared.columns
        assert "advanced_stage" in prepared.columns
    
    def test_train(self, predictor, sample_data):
        """Test model training."""
        metrics = predictor.train(sample_data, test_size=0.2, random_state=42)
        
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        
        # Check metrics are within valid range
        for metric, value in metrics.items():
            assert 0 <= value <= 1, f"{metric} should be between 0 and 1"
        
        # Check model is trained
        assert predictor.model is not None
        assert len(predictor.feature_names) > 0
    
    def test_predict(self, predictor, sample_data):
        """Test predictions."""
        # Train model first
        predictor.train(sample_data, test_size=0.2)
        
        # Make predictions on sample
        test_sample = sample_data.head(10)
        predictions = predictor.predict(test_sample)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_predict_proba(self, predictor, sample_data):
        """Test probability predictions."""
        # Train model first
        predictor.train(sample_data, test_size=0.2)
        
        # Make probability predictions
        test_sample = sample_data.head(10)
        probabilities = predictor.predict_proba(test_sample)
        
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (10, 2)  # Binary classification
        
        # Check probabilities sum to 1
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_feature_importance(self, predictor, sample_data):
        """Test feature importance calculation."""
        predictor.train(sample_data, test_size=0.2)
        
        if predictor.feature_importance is not None:
            assert isinstance(predictor.feature_importance, pd.DataFrame)
            assert "feature" in predictor.feature_importance.columns
            assert "importance" in predictor.feature_importance.columns
            assert len(predictor.feature_importance) > 0
    
    def test_save_and_load_model(self, predictor, sample_data, tmp_path):
        """Test model saving and loading."""
        # Train model
        predictor.train(sample_data, test_size=0.2)
        
        # Save model
        model_path = tmp_path / "test_model.joblib"
        predictor.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        new_predictor = ViralSuppressionPredictor()
        new_predictor.load_model(str(model_path))
        
        assert new_predictor.model is not None
        assert new_predictor.model_type == predictor.model_type
        assert new_predictor.feature_names == predictor.feature_names
        
        # Test predictions match
        test_sample = sample_data.head(5)
        predictions1 = predictor.predict(test_sample)
        predictions2 = new_predictor.predict(test_sample)
        
        assert np.array_equal(predictions1, predictions2)
    
    def test_untrained_model_error(self, predictor, sample_data):
        """Test that prediction fails on untrained model."""
        with pytest.raises(ValueError):
            predictor.predict(sample_data)
    
    def test_missing_target_column(self, predictor):
        """Test training with missing target column."""
        bad_data = pd.DataFrame({
            "age": [25, 35, 45],
            "gender": ["Male", "Female", "Male"],
        })
        
        with pytest.raises(KeyError):
            predictor.train(bad_data)


@pytest.mark.parametrize("model_type", ["logistic", "random_forest", "xgboost"])
def test_different_model_types(model_type, sample_data):
    """Test different model types."""
    predictor = ViralSuppressionPredictor(model_type=model_type)
    metrics = predictor.train(sample_data, test_size=0.2)
    
    assert isinstance(metrics, dict)
    assert all(0 <= v <= 1 for v in metrics.values())


@pytest.mark.parametrize("test_size", [0.1, 0.2, 0.3])
def test_different_test_sizes(predictor, sample_data, test_size):
    """Test with different test sizes."""
    metrics = predictor.train(sample_data, test_size=test_size)
    
    assert isinstance(metrics, dict)
    assert all(key in metrics for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"])


def test_model_performance_threshold(sample_data):
    """Test that model meets minimum performance threshold."""
    predictor = ViralSuppressionPredictor(model_type="xgboost")
    metrics = predictor.train(sample_data, test_size=0.2)
    
    # Model should achieve at least 60% accuracy on synthetic data
    assert metrics["accuracy"] >= 0.6, "Model accuracy too low"
    assert metrics["roc_auc"] >= 0.6, "Model ROC-AUC too low"


def test_consistent_predictions(predictor, sample_data):
    """Test that predictions are consistent."""
    predictor.train(sample_data, test_size=0.2)
    
    test_sample = sample_data.head(10)
    
    # Make predictions multiple times
    predictions1 = predictor.predict(test_sample)
    predictions2 = predictor.predict(test_sample)
    
    # Should be identical
    assert np.array_equal(predictions1, predictions2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

