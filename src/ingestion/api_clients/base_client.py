"""
Base API Client
===============

Base class for all API clients with common functionality.
"""

import time
from typing import Any, Dict, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseAPIClient:
    """
    Base API client with retry logic, rate limiting, and error handling.
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        rate_limit: int = 60,
        retry_attempts: int = 3,
    ):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL for the API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            rate_limit: Maximum requests per minute
            retry_attempts: Number of retry attempts on failure
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.rate_limit = rate_limit
        self.retry_attempts = retry_attempts
        
        # Track request timing for rate limiting
        self.request_times: list[float] = []
        
        # Setup session with retry strategy
        self.session = self._setup_session()
        
        logger.info(f"Initialized API client for {base_url}")
    
    def _setup_session(self) -> requests.Session:
        """
        Setup requests session with retry strategy.
        
        Returns:
            Configured session
        """
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.retry_attempts,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            backoff_factor=1,  # Exponential backoff
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Add default headers
        session.headers.update({
            "User-Agent": "HIV-Medical-Analytics/1.0",
            "Accept": "application/json",
        })
        
        # Add API key if provided
        if self.api_key:
            session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        
        return session
    
    def _rate_limit_check(self):
        """
        Check and enforce rate limiting.
        """
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [
            t for t in self.request_times if current_time - t < 60
        ]
        
        # If at rate limit, wait until oldest request is 1 minute old
        if len(self.request_times) >= self.rate_limit:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                self.request_times = []
        
        # Record this request
        self.request_times.append(current_time)
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a GET request.
        
        Args:
            endpoint: API endpoint (will be appended to base_url)
            params: Query parameters
        
        Returns:
            JSON response as dictionary
        
        Raises:
            requests.RequestException: If request fails after retries
        """
        self._rate_limit_check()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            logger.debug(f"GET request to {url} with params: {params}")
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Received {len(str(data))} bytes from {url}")
            
            return data
        
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {url}: {e}")
            logger.error(f"Response content: {e.response.text if e.response else 'No response'}")
            raise
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise
        
        except ValueError as e:
            logger.error(f"Invalid JSON response from {url}: {e}")
            raise
    
    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make a POST request.
        
        Args:
            endpoint: API endpoint
            data: Form data
            json: JSON data
        
        Returns:
            JSON response as dictionary
        """
        self._rate_limit_check()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            logger.debug(f"POST request to {url}")
            response = self.session.post(
                url, data=data, json=json, timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"POST request failed for {url}: {e}")
            raise
    
    def close(self):
        """Close the session."""
        self.session.close()
        logger.info("API client session closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

