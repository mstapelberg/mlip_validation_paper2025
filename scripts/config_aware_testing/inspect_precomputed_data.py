#!/usr/bin/env python3
"""Inspect the contents of precomputed_data.json to understand what's being saved."""

import json
from pathlib import Path
import pandas as pd

def inspect_precomputed_data(json_path):
    """Print summary of what's in the precomputed data file."""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"=== Precomputed Data Structure ===\n")
    
    # Check each DataFrame
    for key in ['gen_eval_df', 'loss_eval_df', 'loss_groups_eval_df']:
        if key in data and len(data[key]) > 0:
            df = pd.DataFrame(data[key])
            print(f"\n{key}:")
            print(f"  Records: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            
            if 'structure_index' in df.columns:
                print(f"  Structure range: {df['structure_index'].min()} to {df['structure_index'].max()}")
            
            # Show unique values for grouping columns
            for col in ['generation', 'loss_variant', 'loss_group']:
                if col in df.columns:
                    unique_vals = df[col].unique()
                    print(f"  Unique {col}s: {sorted(unique_vals) if isinstance(unique_vals[0], (int, float)) else list(unique_vals)}")
            
            # Show data types and sample stats
            print(f"\n  Sample statistics:")
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in ['e_pa_abs_meV', 'f_rmse', 'p_abs_GPa', 'sigma_rmse_GPa', 'von_mises_abs_GPa']:
                if col in numeric_cols:
                    non_null = df[col].dropna()
                    if len(non_null) > 0:
                        print(f"    {col}: mean={non_null.mean():.3f}, median={non_null.median():.3f}, min={non_null.min():.3f}, max={non_null.max():.3f}")
        else:
            print(f"\n{key}: (empty)")
    
    print(f"\n=== File Size ===")
    file_size = Path(json_path).stat().st_size
    print(f"  Size: {file_size / 1024:.2f} KB")

if __name__ == "__main__":
    import sys
    
    json_path = sys.argv[1] if len(sys.argv) > 1 else "precomputed_data.json"
    
    if not Path(json_path).exists():
        print(f"Error: {json_path} not found")
        sys.exit(1)
    
    inspect_precomputed_data(json_path)

