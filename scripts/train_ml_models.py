#!/usr/bin/env python3
"""
Train Machine Learning Models
==============================

Script to train HIV prediction models.
"""

import sys
from pathlib import Path
import argparse
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.models.viral_suppression_predictor import ViralSuppressionPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train HIV prediction models"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/synthetic_patients.parquet",
        help="Path to patient data",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["logistic", "random_forest", "gradient_boosting", "xgboost"],
        default="xgboost",
        help="Type of model to train",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/models/viral_suppression_model.joblib",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing (default: 0.2)",
    )
    
    args = parser.parse_args()
    
    logger.info(f"Training {args.model_type} model")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {args.output_path}")
    
    # Load data
    print("\n📊 Loading data...")
    try:
        data = pd.read_parquet(args.data_path)
        print(f"✅ Loaded {len(data):,} patients")
    except FileNotFoundError:
        print(f"\n❌ Data file not found: {args.data_path}")
        print("Please run: python scripts/generate_synthetic_data.py")
        sys.exit(1)
    
    # Initialize predictor
    print(f"\n🤖 Initializing {args.model_type} model...")
    predictor = ViralSuppressionPredictor(model_type=args.model_type)
    
    # Train model
    print("\n🔄 Training model...")
    metrics = predictor.train(data, test_size=args.test_size)
    
    # Display results
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE")
    print("=" * 60)
    
    print("\n📈 Model Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Feature importance
    if predictor.feature_importance is not None:
        print("\n🎯 Top 5 Most Important Features:")
        for i, row in predictor.feature_importance.head(5).iterrows():
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    print(f"\n💾 Saving model to {args.output_path}...")
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    predictor.save_model(args.output_path)
    
    print("\n✅ Model saved successfully!")
    print("\nTo use this model for predictions:")
    print("  from src.ml.models.viral_suppression_predictor import ViralSuppressionPredictor")
    print("  predictor = ViralSuppressionPredictor()")
    print(f"  predictor.load_model('{args.output_path}')")
    print("  predictions = predictor.predict(new_data)")


if __name__ == "__main__":
    main()

