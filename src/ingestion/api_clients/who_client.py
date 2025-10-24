"""
WHO API Client
==============

Client for accessing WHO Global Health Observatory data.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from src.ingestion.api_clients.base_client import BaseAPIClient
from src.utils.logger import get_logger
from src.utils.config import config_manager

logger = get_logger(__name__)


class WHOClient(BaseAPIClient):
    """
    Client for WHO Global Health Observatory API.
    
    API Documentation: https://www.who.int/data/gho/info/gho-odata-api
    """
    
    def __init__(self):
        """Initialize WHO API client."""
        config = config_manager.get_data_sources_config()
        who_config = config.get("who", {})
        
        super().__init__(
            base_url=who_config.get("base_url", "https://ghoapi.azureedge.net/api"),
            timeout=who_config.get("timeout", 30),
            rate_limit=who_config.get("rate_limit", 100),
            retry_attempts=who_config.get("retry_attempts", 3),
        )
        
        self.indicators = who_config.get("indicators", [])
        logger.info(f"WHO client initialized with {len(self.indicators)} indicators")
    
    def get_indicator_data(
        self,
        indicator_code: str,
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Fetch data for a specific indicator.
        
        Args:
            indicator_code: WHO indicator code (e.g., 'HIV_0000000001')
            countries: List of country codes (ISO3). If None, fetch all countries
            years: List of years. If None, fetch all available years
        
        Returns:
            DataFrame with indicator data
        """
        logger.info(f"Fetching WHO indicator: {indicator_code}")
        
        try:
            # Fetch indicator data
            endpoint = f"{indicator_code}"
            data = self.get(endpoint)
            
            # Extract value data
            if "value" not in data:
                logger.warning(f"No 'value' field in response for {indicator_code}")
                return pd.DataFrame()
            
            records = data["value"]
            
            if not records:
                logger.warning(f"No data returned for indicator {indicator_code}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            # Filter by countries if specified
            if countries and "SpatialDim" in df.columns:
                df = df[df["SpatialDim"].isin(countries)]
            
            # Filter by years if specified
            if years and "TimeDim" in df.columns:
                df = df[df["TimeDim"].isin(years)]
            
            logger.info(f"Fetched {len(df)} records for {indicator_code}")
            
            # Standardize column names
            df = self._standardize_columns(df)
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching indicator {indicator_code}: {e}")
            return pd.DataFrame()
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize WHO column names to consistent format.
        
        Args:
            df: Raw DataFrame from WHO API
        
        Returns:
            DataFrame with standardized columns
        """
        column_mapping = {
            "Id": "id",
            "IndicatorCode": "indicator_code",
            "SpatialDimType": "spatial_dim_type",
            "SpatialDim": "country_code",
            "TimeDimType": "time_dim_type",
            "TimeDim": "year",
            "Dim1Type": "dim1_type",
            "Dim1": "dim1",
            "Dim2Type": "dim2_type",
            "Dim2": "dim2",
            "Dim3Type": "dim3_type",
            "Dim3": "dim3",
            "DataSourceDimType": "data_source_dim_type",
            "DataSourceDim": "data_source",
            "Value": "value",
            "NumericValue": "numeric_value",
            "Low": "low",
            "High": "high",
            "Comments": "comments",
            "Date": "date",
            "TimeDimensionValue": "time_dimension_value",
            "TimeDimensionBegin": "time_dimension_begin",
            "TimeDimensionEnd": "time_dimension_end",
        }
        
        # Rename columns if they exist
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        return df
    
    def get_hiv_prevalence(
        self,
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Get HIV prevalence data.
        
        Args:
            countries: List of country codes (ISO3)
            years: List of years
        
        Returns:
            DataFrame with HIV prevalence data
        """
        return self.get_indicator_data("HIV_0000000001", countries, years)
    
    def get_art_coverage(
        self,
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Get antiretroviral therapy (ART) coverage data.
        
        Args:
            countries: List of country codes (ISO3)
            years: List of years
        
        Returns:
            DataFrame with ART coverage data
        """
        return self.get_indicator_data("HIV_0000000003", countries, years)
    
    def get_new_infections(
        self,
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Get new HIV infections data.
        
        Args:
            countries: List of country codes (ISO3)
            years: List of years
        
        Returns:
            DataFrame with new infections data
        """
        return self.get_indicator_data("HIV_0000000026", countries, years)
    
    def get_aids_deaths(
        self,
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Get AIDS-related deaths data.
        
        Args:
            countries: List of country codes (ISO3)
            years: List of years
        
        Returns:
            DataFrame with AIDS deaths data
        """
        return self.get_indicator_data("HIV_0000000029", countries, years)
    
    def get_all_hiv_indicators(
        self,
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all configured HIV indicators.
        
        Args:
            countries: List of country codes (ISO3)
            years: List of years
        
        Returns:
            Dictionary mapping indicator codes to DataFrames
        """
        logger.info(f"Fetching all {len(self.indicators)} WHO HIV indicators")
        
        results = {}
        
        for indicator in self.indicators:
            try:
                df = self.get_indicator_data(indicator, countries, years)
                if not df.empty:
                    results[indicator] = df
            except Exception as e:
                logger.error(f"Error fetching indicator {indicator}: {e}")
        
        logger.info(f"Successfully fetched {len(results)}/{len(self.indicators)} indicators")
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = WHOClient()
    
    # Fetch HIV prevalence for select countries
    countries = ["USA", "ZAF", "KEN", "UGA", "IND"]  # US, South Africa, Kenya, Uganda, India
    years = [2019, 2020, 2021, 2022, 2023]
    
    # Get prevalence data
    prevalence_df = client.get_hiv_prevalence(countries=countries, years=years)
    print(f"\\nHIV Prevalence Data:\\n{prevalence_df.head()}")
    
    # Get ART coverage
    art_df = client.get_art_coverage(countries=countries, years=years)
    print(f"\\nART Coverage Data:\\n{art_df.head()}")
    
    # Close client
    client.close()

