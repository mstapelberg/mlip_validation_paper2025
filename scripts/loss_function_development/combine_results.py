#!/usr/bin/env python
"""
Combine Results from Multiple Runs
===================================

Helper script to combine results from running compare_loss_functions.py
in different conda environments (for RMCE/RMQE vs standard models).

Usage:
------
    # Step 1: Run in standard nequip env
    python compare_loss_functions.py --models "MSE:..." "TailHuber:..." --output results1
    
    # Step 2: Run in custom nequip fork env
    python compare_loss_functions.py --models "RMCE:..." "RMQE:..." --output results2
    
    # Step 3: Combine
    python combine_results.py --dirs results1 results2 --output combined_results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import shutil


def combine_detailed_results(result_dirs: list) -> pd.DataFrame:
    """Combine detailed_results.csv from multiple directories."""
    dfs = []
    
    for result_dir in result_dirs:
        csv_path = Path(result_dir) / 'detailed_results.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            dfs.append(df)
            print(f"  ✓ Loaded {len(df)} rows from {result_dir}")
        else:
            print(f"  ✗ Not found: {csv_path}")
    
    if not dfs:
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def plot_combined_comparison(combined_df: pd.DataFrame, output_dir: Path):
    """Recreate comparison plots with combined data."""
    from compare_loss_functions import plot_defect_vs_bulk_comparison, create_summary_table
    
    print("\nCreating combined visualizations...")
    plot_defect_vs_bulk_comparison(combined_df, output_dir)
    create_summary_table(combined_df, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Combine results from multiple compare_loss_functions.py runs')
    
    parser.add_argument('--dirs', nargs='+', required=True,
                       help='Result directories to combine')
    
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for combined results')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("COMBINING RESULTS")
    print("="*80)
    print(f"\nInput directories:")
    for d in args.dirs:
        print(f"  - {d}")
    print(f"\nOutput directory: {output_dir}")
    
    # Combine detailed results
    print("\nCombining detailed results...")
    combined_df = combine_detailed_results(args.dirs)
    
    if combined_df is None:
        print("\n✗ No results found to combine!")
        return
    
    print(f"\n✓ Combined {len(combined_df)} total rows")
    print(f"  Models: {combined_df['model'].unique()}")
    
    # Save combined results
    combined_df.to_csv(output_dir / 'detailed_results.csv', index=False)
    print(f"\n✓ Saved combined results: {output_dir / 'detailed_results.csv'}")
    
    # Recreate visualizations
    plot_combined_comparison(combined_df, output_dir)
    
    print("\n" + "="*80)
    print("COMBINING COMPLETE!")
    print("="*80)
    print(f"\nCombined results saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()

