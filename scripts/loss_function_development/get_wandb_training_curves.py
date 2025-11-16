#!/usr/bin/env python
"""
Extract Training Curves from Wandb
===================================

This script downloads training history (loss, validation metrics) from wandb
for your loss function comparison experiments.

IMPORTANT: By default, this script now uses scan_history() to get ALL data points.
This is critical for validation metrics which may be logged less frequently than
training metrics. The default wandb.history() samples ~500-1000 points which can
miss validation data.

Usage:
------
    python get_wandb_training_curves.py \
      --project mnm-shortlab-mit/MLIP2025-Loss-Testing \
      --output training_curves.csv \
      --runs "job_gen_7_2025-05-30_MSE" "job_gen_7_2025-05-30_TailHuber_model_2"
      
    # Or export all runs
    python get_wandb_training_curves.py --project mnm-shortlab-mit/MLIP2025-Loss-Testing
    
    # Limit to specific number of samples (not recommended for validation data)
    python get_wandb_training_curves.py --project mnm-shortlab-mit/MLIP2025-Loss-Testing --samples 1000
"""

import pandas as pd
import wandb
import argparse
import sys


def get_run_history(api, entity_project: str, run_name: str, samples: int = None) -> pd.DataFrame:
    """
    Get training history for a specific run.
    
    Args:
        samples: Number of samples to retrieve. None = all samples (default).
                 By default, wandb.history() samples ~500-1000 points.
                 Set to None to get all data points.
    
    Returns DataFrame with columns like:
    - _step, _runtime, train_loss, val_loss, etc.
    """
    try:
        runs = api.runs(entity_project, filters={"display_name": run_name})
        
        if len(runs) == 0:
            print(f"  ✗ Run '{run_name}' not found")
            return None
        
        run = runs[0]
        print(f"  ✓ Found run: {run.name} (ID: {run.id})")
        
        # Get full history - use scan_history() to get ALL data points
        # This is critical for validation metrics which may be logged less frequently
        print(f"    Fetching history (samples={'all' if samples is None else samples})...")
        
        if samples is None:
            # Use scan_history() to get all data points without sampling
            history_list = []
            for row in run.scan_history():
                history_list.append(row)
            history = pd.DataFrame(history_list)
        else:
            # Use history() with specified sample count
            history = run.history(samples=samples)
        
        if history.empty:
            print(f"    ⚠ Warning: No history data")
            return None
        
        # Add metadata
        history['run_name'] = run_name
        history['run_id'] = run.id
        
        print(f"    → {len(history)} steps, {len(history.columns)} metrics")
        
        # Print available columns for debugging
        val_cols = [c for c in history.columns if 'val' in c.lower() or 'validation' in c.lower()]
        train_cols = [c for c in history.columns if 'train' in c.lower()]
        print(f"    Validation columns: {val_cols[:10]}{'...' if len(val_cols) > 10 else ''}")
        print(f"    Training columns: {train_cols[:10]}{'...' if len(train_cols) > 10 else ''}")
        
        # Check for validation data availability
        if val_cols:
            for col in val_cols[:5]:  # Check first 5 validation columns
                non_null = history[col].notna().sum()
                if non_null > 0:
                    print(f"      {col}: {non_null}/{len(history)} non-null values")
        else:
            print(f"    ⚠ No validation columns found!")
        
        return history
    
    except Exception as e:
        print(f"  ✗ Error getting run '{run_name}': {e}")
        import traceback
        traceback.print_exc()
        return None


def combine_resumed_runs(api, entity_project: str, run_names: list) -> pd.DataFrame:
    """
    Combine history from multiple runs that are continuations of the same training.
    Useful for runs that were stopped and resumed.
    """
    all_history = []
    
    print(f"Combining {len(run_names)} resumed runs:")
    
    for run_name in run_names:
        history = get_run_history(api, entity_project, run_name)
        if history is not None:
            all_history.append(history)
    
    if not all_history:
        return None
    
    # Concatenate and sort by step
    combined = pd.concat(all_history, ignore_index=True)
    combined = combined.sort_values('_step').reset_index(drop=True)
    
    print(f"  → Combined: {len(combined)} total steps")
    
    return combined


def get_all_runs_summary(api, entity_project: str) -> pd.DataFrame:
    """Get summary information for all runs in the project."""
    runs = api.runs(entity_project)
    
    summary_list = []
    config_list = []
    name_list = []
    
    for run in runs:
        summary_list.append(run.summary._json_dict)
        config_list.append({k: v for k, v in run.config.items() if not k.startswith('_')})
        name_list.append(run.name)
    
    df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
    })
    
    return df


def plot_training_curves(histories_dict: dict, output_path: str, 
                         metric: str = 'train_loss'):
    """
    Plot training curves for comparison.
    
    Args:
        histories_dict: {model_name: history_df}
        metric: column name to plot (e.g., 'train_loss', 'val_loss')
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_name, history in histories_dict.items():
        if metric in history.columns:
            # Use _step if available, otherwise index
            x = history['_step'] if '_step' in history.columns else history.index
            y = history[metric].dropna()
            
            ax.plot(x[:len(y)], y, label=model_name, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Training Curves: {metric}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved training curves: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Extract training curves from wandb',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--project', type=str, 
                       default='mnm-shortlab-mit/MLIP2025-Loss-Testing',
                       help='Wandb project in format: entity/project-name (default: mnm-shortlab-mit/MLIP2025-Loss-Testing)')
    
    parser.add_argument('--runs', nargs='*', type=str,
                       help='Specific run names to extract (optional)')
    
    parser.add_argument('--output', type=str, default='training_curves.csv',
                       help='Output CSV file')
    
    parser.add_argument('--plot', type=str, default=None,
                       help='Create training curve plot (saves to this path)')
    
    parser.add_argument('--metric', type=str, default='train_loss',
                       help='Metric to plot (default: train_loss)')
    
    parser.add_argument('--summary-only', action='store_true',
                       help='Only get run summaries, not full history')
    
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of samples to retrieve (default: None = all samples). '
                            'By default wandb samples ~500-1000 points. Set to None for all data.')
    
    args = parser.parse_args()
    
    print("="*80)
    print("WANDB TRAINING CURVE EXTRACTION")
    print("="*80)
    print(f"Project: {args.project}")
    
    # Initialize API
    try:
        api = wandb.Api()
    except Exception as e:
        print(f"\n✗ Error initializing wandb API: {e}")
        print("Make sure you're logged in: wandb login")
        sys.exit(1)
    
    if args.summary_only:
        # Get summary only
        print("\nGetting run summaries...")
        df = get_all_runs_summary(api, args.project)
        df.to_csv(args.output, index=False)
        print(f"\n✓ Saved summaries: {args.output}")
        print(f"  Runs: {len(df)}")
        return
    
    # Get training history
    if args.runs:
        # Specific runs
        print(f"\nGetting history for {len(args.runs)} runs:")
        
        histories = {}
        for run_name in args.runs:
            history = get_run_history(api, args.project, run_name, samples=args.samples)
            if history is not None:
                histories[run_name] = history
        
        if not histories:
            print("\n✗ No runs found!")
            sys.exit(1)
        
        # Combine all histories
        combined = pd.concat(histories.values(), ignore_index=True)
        combined.to_csv(args.output, index=False)
        print(f"\n✓ Saved training history: {args.output}")
        print(f"  Total steps: {len(combined)}")
        
        # Optionally plot
        if args.plot:
            plot_training_curves(histories, args.plot, metric=args.metric)
    
    else:
        # All runs
        print("\nGetting history for ALL runs in project...")
        runs = api.runs(args.project)
        
        all_histories = []
        for run in runs:
            print(f"\n{run.name}")
            try:
                # Use scan_history() to get all data points
                if args.samples is None:
                    print(f"  Fetching all history data...")
                    history_list = []
                    for row in run.scan_history():
                        history_list.append(row)
                    history = pd.DataFrame(history_list)
                else:
                    history = run.history(samples=args.samples)
                
                if not history.empty:
                    history['run_name'] = run.name
                    history['run_id'] = run.id
                    
                    # Print validation column info with details
                    val_cols = [c for c in history.columns if 'val' in c.lower()]
                    if val_cols:
                        print(f"  ✓ {len(history)} steps, {len(val_cols)} validation columns found")
                        # Show details for each validation column
                        for col in val_cols[:10]:  # Show first 10
                            non_null = history[col].notna().sum()
                            if non_null > 0:
                                pct = 100 * non_null / len(history)
                                col_data = history[col].dropna()
                                col_min = col_data.min()
                                col_max = col_data.max()
                                print(f"    {col}: {non_null} points ({pct:.2f}%), range [{col_min:.4f}, {col_max:.4f}]")
                            else:
                                print(f"    {col}: 0 points (empty)")
                    else:
                        print(f"  ✓ {len(history)} steps (⚠ no validation columns found)")
                    
                    all_histories.append(history)
                else:
                    print(f"  ⚠ No history")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
        
        if not all_histories:
            print("\n✗ No training history found!")
            sys.exit(1)
        
        combined = pd.concat(all_histories, ignore_index=True)
        combined.to_csv(args.output, index=False)
        print(f"\n✓ Saved training history: {args.output}")
        print(f"  Total steps: {len(combined)}")


if __name__ == '__main__':
    main()

