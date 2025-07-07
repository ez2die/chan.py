# -*- coding: utf-8 -*-
"""
Configuration module for the cryptocurrency data platform.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OKX API Configuration
    okx_api_key: Optional[str] = Field(None, env="OKX_API_KEY")
    okx_secret: Optional[str] = Field(None, env="OKX_SECRET")
    okx_passphrase: Optional[str] = Field(None, env="OKX_PASSPHRASE")
    okx_sandbox: bool = Field(False, env="OKX_SANDBOX")
    
    # Data Storage Configuration
    data_dir: str = Field("./data", env="DATA_DIR")
    log_dir: str = Field("./logs", env="LOG_DIR")
    
    # Database Configuration
    database_url: str = Field("sqlite:///./data/tasks.db", env="DATABASE_URL")
    
    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    
    # Web Dashboard Configuration
    web_host: str = Field("0.0.0.0", env="WEB_HOST")
    web_port: int = Field(8501, env="WEB_PORT")
    
    # Scheduling Configuration
    scheduler_timezone: str = Field("UTC", env="SCHEDULER_TIMEZONE")
    update_interval_minutes: int = Field(60, env="UPDATE_INTERVAL_MINUTES")
    
    # Data Processing Configuration
    max_days_per_request: int = Field(100, env="MAX_DAYS_PER_REQUEST")
    retry_attempts: int = Field(3, env="RETRY_ATTEMPTS")
    retry_delay_seconds: int = Field(5, env="RETRY_DELAY_SECONDS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for data organization
        data_path = Path(self.data_dir)
        (data_path / "okx").mkdir(parents=True, exist_ok=True)
        (data_path / "okx" / "spot").mkdir(parents=True, exist_ok=True)
        (data_path / "okx" / "swap").mkdir(parents=True, exist_ok=True)
    
    @property
    def okx_credentials(self) -> dict:
        """Get OKX API credentials."""
        return {
            "apiKey": self.okx_api_key,
            "secret": self.okx_secret,
            "password": self.okx_passphrase,
            "sandbox": self.okx_sandbox,
        }
    
    def get_data_path(self, exchange: str, instrument_type: str, symbol: str, timeframe: str) -> Path:
        """Get the file path for storing data."""
        return Path(self.data_dir) / exchange / instrument_type / symbol / timeframe


# Global settings instance
settings = Settings()