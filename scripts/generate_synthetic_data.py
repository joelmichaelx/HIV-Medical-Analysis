#!/usr/bin/env python3
"""
Generate Synthetic HIV Data
============================

Script to generate synthetic HIV patient data for testing and development.
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.data_generator import HIVDataGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic HIV patient data"
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=10000,
        help="Number of patients to generate (default: 10000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory (default: data/raw)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    
    logger.info(f"Generating synthetic data for {args.n_patients} patients")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Random seed: {args.seed}")
    
    # Initialize generator
    generator = HIVDataGenerator(seed=args.seed)
    
    # Generate complete dataset
    datasets = generator.generate_complete_dataset(n_patients=args.n_patients)
    
    # Save datasets
    generator.save_datasets(datasets, output_dir=args.output_dir)
    
    # Display summary
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 60)
    
    for name, df in datasets.items():
        print(f"\n{name.upper()}:")
        print(f"  Records: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  File: {args.output_dir}/synthetic_{name}.parquet")
    
    print("\n" + "=" * 60)
    print("\nData generation complete! ✅")
    print(f"\nGenerated {sum(len(df) for df in datasets.values()):,} total records")
    print("\nTo load this data:")
    print("  import pandas as pd")
    print(f"  df = pd.read_parquet('{args.output_dir}/synthetic_patients.parquet')")


if __name__ == "__main__":
    main()

