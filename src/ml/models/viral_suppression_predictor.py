"""
Viral Suppression Prediction Model
===================================

Predict likelihood of achieving viral suppression based on patient characteristics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import xgboost as xgb
import joblib
import shap
from src.utils.logger import get_logger
from src.utils.config import config_manager

logger = get_logger(__name__)


class ViralSuppressionPredictor:
    """
    Machine learning model to predict viral suppression outcomes.
    """
    
    def __init__(self, model_type: str = "xgboost"):
        """
        Initialize the predictor.
        
        Args:
            model_type: Type of model ('logistic', 'random_forest', 'gradient_boosting', 'xgboost')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.feature_importance = None
        
        # Load config
        config = config_manager.get_ml_config()
        self.config = config.get("models", {}).get(model_type, {})
        
        logger.info(f"Initialized viral suppression predictor with {model_type}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for modeling.
        
        Args:
            df: Raw patient DataFrame
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Preparing features for modeling")
        
        # Create a copy
        data = df.copy()
        
        # Create age groups
        data["age_group"] = pd.cut(
            data["age"],
            bins=[0, 25, 35, 45, 55, 100],
            labels=["<25", "25-34", "35-44", "45-54", "55+"],
        )
        
        # Time-based features
        data["diagnosis_date"] = pd.to_datetime(data["diagnosis_date"])
        data["treatment_start_date"] = pd.to_datetime(data["treatment_start_date"])
        
        data["days_to_treatment"] = (
            data["treatment_start_date"] - data["diagnosis_date"]
        ).dt.days
        
        data["diagnosis_year"] = data["diagnosis_date"].dt.year
        data["diagnosis_month"] = data["diagnosis_date"].dt.month
        
        # CD4 categories
        data["cd4_category"] = pd.cut(
            data["cd4_count_at_diagnosis"],
            bins=[0, 200, 350, 500, 2000],
            labels=["<200", "200-349", "350-499", ">=500"],
        )
        
        # Viral load categories
        data["viral_load_category"] = pd.cut(
            data["viral_load_at_diagnosis"],
            bins=[0, 1000, 10000, 100000, 10000000],
            labels=["<1K", "1K-10K", "10K-100K", ">100K"],
        )
        
        # WHO stage binary
        data["advanced_stage"] = data["who_clinical_stage"].isin(["Stage 3", "Stage 4"]).astype(int)
        
        # High-risk transmission route
        data["high_risk_transmission"] = data["transmission_route"].isin(["IDU", "Unknown"]).astype(int)
        
        logger.info(f"Feature engineering complete. Shape: {data.shape}")
        
        return data
    
    def encode_categorical_features(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: DataFrame with categorical features
            categorical_cols: List of categorical column names
            fit: Whether to fit new encoders or use existing ones
        
        Returns:
            DataFrame with encoded features
        """
        data = df.copy()
        
        for col in categorical_cols:
            if col not in data.columns:
                continue
            
            if fit:
                self.label_encoders[col] = LabelEncoder()
                data[col] = self.label_encoders[col].fit_transform(data[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    data[col] = data[col].astype(str)
                    known_classes = set(self.label_encoders[col].classes_)
                    data[col] = data[col].apply(
                        lambda x: x if x in known_classes else self.label_encoders[col].classes_[0]
                    )
                    data[col] = self.label_encoders[col].transform(data[col])
        
        return data
    
    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "viral_suppression",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """
        Train the prediction model.
        
        Args:
            df: Training data
            target_col: Target column name
            test_size: Proportion of data for testing
            random_state: Random seed
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Starting model training")
        
        # Prepare features
        data = self.prepare_features(df)
        
        # Select features
        feature_cols = [
            "age",
            "gender",
            "transmission_route",
            "cd4_count_at_diagnosis",
            "viral_load_at_diagnosis",
            "who_clinical_stage",
            "treatment_adherence",
            "days_to_treatment",
            "advanced_stage",
            "high_risk_transmission",
        ]
        
        # Filter to available columns
        feature_cols = [col for col in feature_cols if col in data.columns]
        
        X = data[feature_cols].copy()
        y = data[target_col]
        
        # Encode categorical features
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        X = self.encode_categorical_features(X, categorical_cols, fit=True)
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if self.model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                class_weight="balanced",
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=random_state,
                class_weight="balanced",
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=random_state,
            )
        elif self.model_type == "xgboost":
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                use_label_encoder=False,
                eval_metric="logloss",
            )
        
        logger.info("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        metrics = self.evaluate(X_test_scaled, y_test)
        
        # Calculate feature importance
        self._calculate_feature_importance(X_train_scaled, feature_cols)
        
        logger.info(f"Model training complete. ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Classification report
        report = classification_report(y_test, y_pred)
        logger.info(f"Classification Report:\n{report}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict viral suppression for new patients.
        
        Args:
            df: Patient data
        
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Prepare features
        data = self.prepare_features(df)
        X = data[self.feature_names].copy()
        
        # Encode
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        X = self.encode_categorical_features(X, categorical_cols, fit=False)
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities of viral suppression.
        
        Args:
            df: Patient data
        
        Returns:
            Array of probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Prepare features
        data = self.prepare_features(df)
        X = data[self.feature_names].copy()
        
        # Encode
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        X = self.encode_categorical_features(X, categorical_cols, fit=False)
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities
    
    def _calculate_feature_importance(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ):
        """
        Calculate feature importance.
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
        """
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            self.feature_importance = pd.DataFrame({
                "feature": feature_names,
                "importance": importances,
            }).sort_values("importance", ascending=False)
            
            logger.info("Top 5 most important features:")
            logger.info(f"\n{self.feature_importance.head()}")
    
    def save_model(self, path: str):
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance,
            "model_type": self.model_type,
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model
        """
        model_data = joblib.load(path)
        
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.label_encoders = model_data["label_encoders"]
        self.feature_names = model_data["feature_names"]
        self.feature_importance = model_data["feature_importance"]
        self.model_type = model_data["model_type"]
        
        logger.info(f"Model loaded from {path}")


# Example usage
if __name__ == "__main__":
    from src.ingestion.data_generator import HIVDataGenerator
    
    # Generate synthetic data
    generator = HIVDataGenerator()
    datasets = generator.generate_complete_dataset(n_patients=10000)
    
    # Initialize predictor
    predictor = ViralSuppressionPredictor(model_type="xgboost")
    
    # Train model
    print("\n=== Training Viral Suppression Prediction Model ===\n")
    metrics = predictor.train(datasets["patients"])
    
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nFeature Importance:")
    print(predictor.feature_importance)
    
    # Make predictions on new data
    sample_patients = datasets["patients"].head(5)
    predictions = predictor.predict(sample_patients)
    probabilities = predictor.predict_proba(sample_patients)
    
    print("\nSample Predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"  Patient {i+1}: Prediction={pred}, Probability={prob[1]:.3f}")
    
    # Save model
    predictor.save_model("data/models/viral_suppression_model.joblib")

