"""
Quick validation script to check data structure and availability.
Run this first to ensure everything is set up correctly.
"""

import re
import json
import glob
import pandas as pd
from pathlib import Path

# ---------- CONFIG ----------
CSV_PATH = "/home/myless/Packages/mlip_validation_paper2025/data/wandb_data/hpo_training_results.csv"
INFERENCE_DATA_DIR = "/home/myless/Packages/mlip_validation_paper2025/data/wandb_data/hpo_inference_test_analysis"

def extract_hpo_number(name_string):
    """Extract hpo_XXX from the Name column."""
    match = re.search(r'hpo_(\d+)', name_string)
    if match:
        return int(match.group(1))
    return None

print("="*70)
print("DATA VALIDATION CHECK")
print("="*70)

# ---------- CHECK TRAINING DATA ----------
print("\n1. TRAINING DATA")
print("-" * 70)

try:
    df = pd.read_csv(CSV_PATH)
    print(f"✓ CSV loaded successfully: {len(df)} rows")
    print(f"  Columns: {list(df.columns[:5])}... ({len(df.columns)} total)")
    
    # Extract HPO numbers
    df['hpo_num'] = df['Name'].apply(extract_hpo_number)
    unique_hpo = df['hpo_num'].dropna().unique()
    print(f"✓ Unique HPO configs: {len(unique_hpo)}")
    print(f"  Range: hpo_{int(min(unique_hpo)):03d} to hpo_{int(max(unique_hpo)):03d}")
    
    # Check for replicates
    sample_config = unique_hpo[0]
    n_replicates = len(df[df['hpo_num'] == sample_config])
    print(f"✓ Replicates per config (example hpo_{int(sample_config):03d}): {n_replicates}")
    
    # Check required columns
    required_cols = [
        "test0_epoch/forces_rmse",
        "test0_epoch/per_atom_energy_rmse", 
        "test0_epoch/stress_rmse",
        "model.r_max",
        "model.l_max",
        "model.num_layers"
    ]
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"✗ Missing columns: {missing}")
    else:
        print(f"✓ All required columns present")
    
    # Show sample stats
    print(f"\n  Sample statistics (test0_epoch/forces_rmse):")
    print(f"    Mean: {df['test0_epoch/forces_rmse'].mean():.6f}")
    print(f"    Min:  {df['test0_epoch/forces_rmse'].min():.6f}")
    print(f"    Max:  {df['test0_epoch/forces_rmse'].max():.6f}")
    
except Exception as e:
    print(f"✗ Error loading training data: {e}")

# ---------- CHECK BENCHMARK DATA ----------
print("\n2. BENCHMARK DATA")
print("-" * 70)

try:
    pattern = str(Path(INFERENCE_DATA_DIR) / "benchmark_results_hpo_*_compiled.nequip")
    dirs = glob.glob(pattern)
    print(f"✓ Found {len(dirs)} benchmark directories")
    
    # Load one example
    if dirs:
        example_dir = dirs[0]
        match = re.search(r'hpo_(\d+)', example_dir)
        if match:
            hpo_num = int(match.group(1))
            json_path = Path(example_dir) / "benchmark_summary_all_supercells.json"
            
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                print(f"✓ Example: hpo_{hpo_num:03d}")
                print(f"  Supercells tested: {len(data)}")
                print(f"  Supercell sizes: {[entry['supercell'] for entry in data]}")
                print(f"  Atom counts: {[entry['num_atoms'] for entry in data]}")
                
                # Show first entry details
                first = data[0]
                print(f"\n  Sample benchmark (first entry):")
                print(f"    Supercell: {first['supercell']}")
                print(f"    Atoms: {first['num_atoms']}")
                print(f"    Timesteps/s: {first['timesteps_per_s_log']:.2f}")
                print(f"    kAtom-steps/s: {first['katom_steps_per_s_log']:.2f}")
                print(f"    GPU memory: {first['peak_gpu_memory_mib']} MiB")
            else:
                print(f"✗ JSON file not found: {json_path}")
    
    # Count how many configs have benchmark data
    benchmark_hpo_nums = []
    for dir_path in dirs:
        match = re.search(r'hpo_(\d+)', dir_path)
        if match:
            benchmark_hpo_nums.append(int(match.group(1)))
    
    benchmark_hpo_nums = sorted(set(benchmark_hpo_nums))
    print(f"\n✓ Configs with benchmarks: {len(benchmark_hpo_nums)}")
    print(f"  Range: hpo_{benchmark_hpo_nums[0]:03d} to hpo_{benchmark_hpo_nums[-1]:03d}")
    
except Exception as e:
    print(f"✗ Error loading benchmark data: {e}")

# ---------- CHECK OVERLAP ----------
print("\n3. DATA OVERLAP")
print("-" * 70)

try:
    # Configs with training data
    train_set = set(unique_hpo)
    bench_set = set(benchmark_hpo_nums)
    
    overlap = train_set & bench_set
    only_train = train_set - bench_set
    only_bench = bench_set - train_set
    
    print(f"✓ Configs with BOTH training & benchmark: {len(overlap)}")
    print(f"  Overlap percentage: {100*len(overlap)/len(train_set):.1f}%")
    
    if only_train:
        print(f"\n  ⚠ Configs with training but NO benchmark: {len(only_train)}")
        if len(only_train) <= 10:
            print(f"    {sorted(only_train)}")
        else:
            print(f"    First 10: {sorted(list(only_train))[:10]}")
    
    if only_bench:
        print(f"\n  ⚠ Configs with benchmark but NO training: {len(only_bench)}")
        if len(only_bench) <= 10:
            print(f"    {sorted(only_bench)}")
    
except Exception as e:
    print(f"✗ Error checking overlap: {e}")

# ---------- SUMMARY ----------
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

can_run = len(overlap) > 0 if 'overlap' in locals() else False

if can_run:
    print(f"✓ Data validation PASSED")
    print(f"  {len(overlap)} configurations can be analyzed")
    print(f"\nYou can now run:")
    print(f"  python main_analysis_adapted.py")
    print(f"  python scaling_analysis.py")
else:
    print(f"✗ Data validation FAILED")
    print(f"  Please check the errors above")

print("="*70)

