"""
Data Cleaning Module
====================

Clean and standardize raw HIV medical data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from src.utils.logger import get_logger
from src.utils.config import config_manager

logger = get_logger(__name__)


class DataCleaner:
    """
    Clean and standardize raw data.
    """
    
    def __init__(self):
        """Initialize data cleaner with configuration."""
        config = config_manager.get_pipeline_config()
        self.config = config.get("transformation", {}).get("cleaning", {})
        
        logger.info("Data cleaner initialized")
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all cleaning operations to a DataFrame.
        
        Args:
            df: Raw DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting data cleaning for {len(df)} records")
        
        original_count = len(df)
        
        # Remove duplicates
        if self.config.get("remove_duplicates", True):
            df = self._remove_duplicates(df)
        
        # Handle missing values
        if self.config.get("handle_missing_values", True):
            df = self._handle_missing_values(df)
        
        # Remove outliers
        if self.config.get("outlier_detection", True):
            df = self._remove_outliers(df)
        
        # Standardize data types
        df = self._standardize_dtypes(df)
        
        final_count = len(df)
        removed = original_count - final_count
        
        logger.info(f"Data cleaning complete: {final_count} records retained, {removed} removed")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            df: DataFrame
        
        Returns:
            DataFrame without duplicates
        """
        initial_count = len(df)
        
        # Remove exact duplicates
        df = df.drop_duplicates()
        
        # If patient_id exists, keep only the most recent record
        if "patient_id" in df.columns and "created_at" in df.columns:
            df = df.sort_values("created_at", ascending=False)
            df = df.drop_duplicates(subset=["patient_id"], keep="first")
        
        duplicates_removed = initial_count - len(df)
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate records")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values based on column types and configuration.
        
        Args:
            df: DataFrame
        
        Returns:
            DataFrame with missing values handled
        """
        strategy = self.config.get("missing_value_strategy", {})
        
        # Identify column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns
        
        # Handle numeric columns
        numeric_strategy = strategy.get("numeric", "median")
        for col in numeric_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                if numeric_strategy == "mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif numeric_strategy == "median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif numeric_strategy == "mode":
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0, inplace=True)
                elif numeric_strategy == "ffill":
                    df[col].fillna(method="ffill", inplace=True)
                elif numeric_strategy == "bfill":
                    df[col].fillna(method="bfill", inplace=True)
                
                logger.debug(f"Filled {missing_count} missing values in {col} using {numeric_strategy}")
        
        # Handle categorical columns
        categorical_strategy = strategy.get("categorical", "mode")
        for col in categorical_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                if categorical_strategy == "mode":
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                    df[col].fillna(mode_val, inplace=True)
                else:
                    df[col].fillna("Unknown", inplace=True)
                
                logger.debug(f"Filled {missing_count} missing values in {col}")
        
        # Handle datetime columns
        datetime_strategy = strategy.get("datetime", "ffill")
        for col in datetime_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                if datetime_strategy == "ffill":
                    df[col].fillna(method="ffill", inplace=True)
                elif datetime_strategy == "bfill":
                    df[col].fillna(method="bfill", inplace=True)
                
                logger.debug(f"Filled {missing_count} missing values in {col}")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and remove outliers from numeric columns.
        
        Args:
            df: DataFrame
        
        Returns:
            DataFrame with outliers removed
        """
        method = self.config.get("outlier_method", "iqr")
        threshold = self.config.get("outlier_threshold", 3.0)
        
        initial_count = len(df)
        
        # Columns to check for outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Exclude ID columns and binary columns
        exclude_patterns = ["_id", "is_", "has_"]
        numeric_cols = [
            col for col in numeric_cols
            if not any(pattern in col for pattern in exclude_patterns)
        ]
        
        if method == "iqr":
            df = self._remove_outliers_iqr(df, numeric_cols)
        elif method == "zscore":
            df = self._remove_outliers_zscore(df, numeric_cols, threshold)
        
        outliers_removed = initial_count - len(df)
        
        if outliers_removed > 0:
            logger.info(f"Removed {outliers_removed} outlier records using {method} method")
        
        return df
    
    def _remove_outliers_iqr(
        self, df: pd.DataFrame, columns: List[str]
    ) -> pd.DataFrame:
        """
        Remove outliers using Interquartile Range (IQR) method.
        
        Args:
            df: DataFrame
            columns: Columns to check
        
        Returns:
            DataFrame with outliers removed
        """
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter outliers
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df = df[mask]
        
        return df
    
    def _remove_outliers_zscore(
        self, df: pd.DataFrame, columns: List[str], threshold: float
    ) -> pd.DataFrame:
        """
        Remove outliers using Z-score method.
        
        Args:
            df: DataFrame
            columns: Columns to check
            threshold: Z-score threshold
        
        Returns:
            DataFrame with outliers removed
        """
        from scipy import stats
        
        for col in columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            df = df[(z_scores < threshold) | df[col].isna()]
        
        return df
    
    def _standardize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize data types for consistency.
        
        Args:
            df: DataFrame
        
        Returns:
            DataFrame with standardized types
        """
        # Convert date columns
        date_columns = [col for col in df.columns if "date" in col.lower()]
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception as e:
                    logger.warning(f"Could not convert {col} to datetime: {e}")
        
        # Convert boolean columns
        bool_columns = [col for col in df.columns if col.startswith("is_") or col.startswith("has_")]
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        # Convert ID columns to string
        id_columns = [col for col in df.columns if col.endswith("_id")]
        for col in id_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        return df


# Example usage
if __name__ == "__main__":
    # Create sample dirty data
    data = {
        "patient_id": ["P001", "P002", "P003", "P003", "P004"],
        "age": [25, 45, np.nan, 35, 200],  # Missing value and outlier
        "cd4_count": [350, 450, 280, 280, 5000],  # Outlier
        "diagnosis_date": ["2020-01-01", "2021-05-15", None, "2021-05-15", "2023-03-20"],
        "gender": ["Male", "Female", None, "Male", "Female"],
    }
    
    df = pd.DataFrame(data)
    print("Original Data:")
    print(df)
    
    # Clean data
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean(df)
    
    print("\nCleaned Data:")
    print(cleaned_df)

