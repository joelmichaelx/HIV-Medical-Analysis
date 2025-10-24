"""
Pytest Configuration
====================

Shared fixtures and configuration for all tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def sample_patients():
    """Generate sample patient data for testing."""
    from src.ingestion.data_generator import HIVDataGenerator
    
    generator = HIVDataGenerator(seed=42)
    datasets = generator.generate_complete_dataset(n_patients=100)
    return datasets["patients"]


@pytest.fixture(scope="session")
def sample_lab_results():
    """Generate sample lab results for testing."""
    from src.ingestion.data_generator import HIVDataGenerator
    
    generator = HIVDataGenerator(seed=42)
    datasets = generator.generate_complete_dataset(n_patients=100)
    return datasets["lab_results"]


@pytest.fixture(scope="session")
def sample_treatments():
    """Generate sample treatment data for testing."""
    from src.ingestion.data_generator import HIVDataGenerator
    
    generator = HIVDataGenerator(seed=42)
    datasets = generator.generate_complete_dataset(n_patients=100)
    return datasets["treatments"]


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory for testing."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir

