"""
Configuration Management
========================

Load and manage application configurations from YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from dotenv import load_dotenv
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class DatabaseConfig(BaseModel):
    """Database configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(default="hiv_analytics")
    user: str = Field(default="postgres")
    password: str = Field(default="")


class KafkaConfig(BaseModel):
    """Kafka configuration."""
    brokers: list[str] = Field(default_factory=lambda: ["localhost:9092"])
    consumer_group: str = Field(default="hiv-analytics")
    topics: Dict[str, str] = Field(default_factory=dict)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Environment
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    
    # Database
    db_host: str = Field(default="localhost")
    db_port: int = Field(default=5432)
    db_name: str = Field(default="hiv_analytics")
    db_user: str = Field(default="postgres")
    db_password: str = Field(default="")
    
    # MongoDB
    mongo_host: str = Field(default="localhost")
    mongo_port: int = Field(default=27017)
    mongo_db: str = Field(default="hiv_clinical")
    mongo_user: str = Field(default="admin")
    mongo_password: str = Field(default="")
    
    # Redis
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_password: str = Field(default="")
    
    # Kafka
    kafka_broker_1: str = Field(default="localhost:9092")
    kafka_broker_2: str = Field(default="localhost:9093")
    kafka_broker_3: str = Field(default="localhost:9094")
    
    # API Keys
    cdc_api_token: Optional[str] = None
    pubmed_api_key: Optional[str] = None
    
    # Email
    alert_email_primary: Optional[str] = None
    alert_email_secondary: Optional[str] = None
    smtp_server: str = Field(default="smtp.gmail.com")
    smtp_port: int = Field(default=587)
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    
    # Slack
    slack_webhook_url: Optional[str] = None
    
    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    
    # Deployment
    deployment_env: str = Field(default="development")
    
    # Security
    secret_key: Optional[str] = None
    jwt_secret: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class ConfigManager:
    """
    Manage application configurations.
    """
    
    def __init__(self):
        self.settings = Settings()
        self.config_dir = PROJECT_ROOT / "config"
        self._configs: Dict[str, Any] = {}
        
        logger.info(f"Configuration initialized for environment: {self.settings.environment}")
    
    def load_yaml_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_name: Name of the config file (without .yaml extension)
        
        Returns:
            Configuration dictionary
        """
        if config_name in self._configs:
            return self._configs[config_name]
        
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                content = f.read()
                # Replace environment variables
                content = self._replace_env_vars(content)
                config = yaml.safe_load(content)
            
            self._configs[config_name] = config
            logger.info(f"Loaded configuration: {config_name}")
            return config
        
        except Exception as e:
            logger.error(f"Error loading configuration {config_name}: {e}")
            return {}
    
    def _replace_env_vars(self, content: str) -> str:
        """
        Replace ${VAR_NAME} placeholders with environment variables.
        
        Args:
            content: YAML content with placeholders
        
        Returns:
            Content with environment variables replaced
        """
        import re
        
        def replacer(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))
        
        return re.sub(r'\$\{([A-Z_]+)\}', replacer, content)
    
    def get_data_sources_config(self) -> Dict[str, Any]:
        """Get data sources configuration."""
        return self.load_yaml_config("data_sources")
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration."""
        return self.load_yaml_config("pipeline_config")
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Get machine learning configuration."""
        return self.load_yaml_config("ml_config")
    
    def get_database_url(self) -> str:
        """Get PostgreSQL database URL."""
        return f"postgresql://{self.settings.db_user}:{self.settings.db_password}@{self.settings.db_host}:{self.settings.db_port}/{self.settings.db_name}"
    
    def get_mongo_url(self) -> str:
        """Get MongoDB connection URL."""
        if self.settings.mongo_user and self.settings.mongo_password:
            return f"mongodb://{self.settings.mongo_user}:{self.settings.mongo_password}@{self.settings.mongo_host}:{self.settings.mongo_port}/{self.settings.mongo_db}"
        return f"mongodb://{self.settings.mongo_host}:{self.settings.mongo_port}/{self.settings.mongo_db}"
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        if self.settings.redis_password:
            return f"redis://:{self.settings.redis_password}@{self.settings.redis_host}:{self.settings.redis_port}/0"
        return f"redis://{self.settings.redis_host}:{self.settings.redis_port}/0"
    
    def get_kafka_brokers(self) -> list[str]:
        """Get list of Kafka brokers."""
        return [
            self.settings.kafka_broker_1,
            self.settings.kafka_broker_2,
            self.settings.kafka_broker_3,
        ]


# Global config instance
config_manager = ConfigManager()

