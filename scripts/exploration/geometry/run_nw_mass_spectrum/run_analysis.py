#!/usr/bin/env python3
"""
New GEF N_w Mass Spectrum Analysis Runner

This script serves as a clean entry point for running the mass spectrum analysis
by leveraging the refactored NwMassSpectrumAnalyzer module.

Author: GEF Research Team
Date: 2025
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from gef.geometry.mass_spectrum_analyzer import NwMassSpectrumAnalyzer

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run GEF N_w mass spectrum analysis using the modular analyzer."
    )
    parser.add_argument(
        'config_path',
        help='Path to the YAML configuration file.'
    )
    args = parser.parse_args()

    try:
        analyzer = NwMassSpectrumAnalyzer(args.config_path)
        results_df, csv_path, plot_path = analyzer.run_full_analysis()

        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {csv_path}")
        print(f"Plot saved to: {plot_path}")

    except Exception as e:
        logging.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
