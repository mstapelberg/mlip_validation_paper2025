#!/usr/bin/env python
"""
Combined Analysis: Training + Testing + Statistical Significance
=================================================================

Creates publication-quality figures combining:
1. Training curves (from wandb/training_history.csv)
2. Test error analysis (from ensemble comparison)
3. Error distributions (max errors, 95th percentile)
4. Statistical significance tests
5. 3-region analysis (core/shell/bulk)

Usage:
------
    python plot_combined_analysis.py \
      --ensemble_summary ../../results/loss_comparison_ensemble/ensemble_summary.csv \
      --training_history training_history.csv \
      --detailed_results ../../results/loss_comparison_all/detailed_results.csv \
      --output ../../results/combined_analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import argparse
from scipy import stats


def load_training_history(csv_path: str, val_weighted_sums_path: str = None) -> pd.DataFrame:
    """Load training history from wandb export.
    
    Args:
        csv_path: Path to training_history.csv
        val_weighted_sums_path: Optional path to val0_weighted_sums.csv with proper validation data
    """
    df = pd.read_csv(csv_path)
    
    # If val_weighted_sums.csv is provided, use it for validation loss
    if val_weighted_sums_path and Path(val_weighted_sums_path).exists():
        print(f"\n  Found val0_weighted_sums.csv, loading validation data from it...")
        val_df = pd.read_csv(val_weighted_sums_path)
        print(f"    Loaded {len(val_df)} rows from val0_weighted_sums.csv")
        
        # Extract validation data for each model
        val_data_list = []
        
        # Get all columns that contain val0_epoch/weighted_sum (skip MIN/MAX variants)
        val_cols = [c for c in val_df.columns if 'val0_epoch/weighted_sum' in c and '__MIN' not in c and '__MAX' not in c]
        
        print(f"    Found {len(val_cols)} validation columns")
        
        # Get the global step column
        step_col = 'trainer/global_step' if 'trainer/global_step' in val_df.columns else None
        if step_col is None:
            print(f"    ⚠ Could not find trainer/global_step column in val0_weighted_sums.csv")
            val_data_list = []
        else:
            for val_col in val_cols:
                # Extract run name from column (format: "job_gen_7_2025-05-30_MSE_model_0 - val0_epoch/weighted_sum")
                run_name = val_col.split(' - ')[0].strip()
                
                # Map to loss type
                if 'L3' in run_name or 'RMCE' in run_name:
                    loss_type = 'RMCE'
                elif 'L4' in run_name or 'RMQE' in run_name:
                    loss_type = 'RMQE'
                elif 'L2' in run_name or 'RMSE' in run_name:
                    loss_type = 'RMSE'
                elif 'TailHuber' in run_name:
                    loss_type = 'TailHuber'
                elif 'StratHuber' in run_name:
                    loss_type = 'StratHuber'
                elif 'MSE' in run_name:
                    loss_type = 'MSE'
                else:
                    loss_type = 'Unknown'
                
                # Extract non-null validation values
                val_data = val_df[[step_col, val_col]].copy()
                val_data = val_data.dropna(subset=[val_col])
                
                if len(val_data) > 0:
                    val_data['run_name'] = run_name
                    val_data['loss_type'] = loss_type
                    val_data['trainer/global_step'] = val_data[step_col]  # Standardize column name
                    val_data['val_loss'] = val_data[val_col]
                    val_data = val_data[['run_name', 'loss_type', 'trainer/global_step', 'val_loss']]
                    val_data_list.append(val_data)
                    print(f"      ✓ {run_name} ({loss_type}): {len(val_data)} validation points")
        
        if val_data_list:
            # Combine all validation data
            val_combined = pd.concat(val_data_list, ignore_index=True)
            print(f"    Combined {len(val_combined)} validation data points")
            
            # Merge with main dataframe on run_name and trainer/global_step
            # First, ensure main df has trainer/global_step
            if 'trainer/global_step' not in df.columns:
                if '_step' in df.columns:
                    df['trainer/global_step'] = df['_step']
                else:
                    print(f"    ⚠ Could not find step column in main dataframe")
            
            # Update val_loss in main dataframe using the validation data
            # Create a mapping: (run_name, trainer/global_step) -> val_loss
            val_dict = {}
            for _, row in val_combined.iterrows():
                key = (row['run_name'], row['trainer/global_step'])
                val_dict[key] = row['val_loss']
            
            # Update df['val_loss'] where we have data
            def get_val_loss(row):
                if pd.isna(row.get('run_name')) or pd.isna(row.get('trainer/global_step')):
                    return row.get('val_loss', np.nan)
                key = (row['run_name'], row['trainer/global_step'])
                return val_dict.get(key, row.get('val_loss', np.nan))
            
            df['val_loss'] = df.apply(get_val_loss, axis=1)
            
            val_updated = df['val_loss'].notna().sum()
            print(f"    Updated val_loss: {val_updated}/{len(df)} non-null values ({100*val_updated/len(df):.1f}%)")
        else:
            print(f"    ⚠ No validation data extracted from val0_weighted_sums.csv")
    
    # Debug: print available columns
    print(f"\n  Available columns in training_history.csv:")
    print(f"    Total columns: {len(df.columns)}")
    val_cols = [c for c in df.columns if 'val' in c.lower() or 'validation' in c.lower()]
    train_cols = [c for c in df.columns if 'train' in c.lower()]
    print(f"    Validation-related columns: {val_cols[:10]}...")  # Show first 10
    print(f"    Training-related columns: {train_cols[:10]}...")  # Show first 10
    
    # Extract loss type from run_name
    def extract_loss_type(name):
        if pd.isna(name):
            return 'Unknown'
        name_str = str(name)
        # Map L2/L3/L4 to RMSE/RMCE/RMQE
        if 'L3' in name_str or 'RMCE' in name_str:
            return 'RMCE'
        if 'L4' in name_str or 'RMQE' in name_str:
            return 'RMQE'
        if 'L2' in name_str or 'RMSE' in name_str:
            return 'RMSE'
        if 'TailHuber' in name_str or 'tail_huber' in name_str.lower():
            return 'TailHuber'
        if 'StratHuber' in name_str or 'stratified_huber' in name_str.lower():
            return 'StratHuber'
        if 'MSE' in name_str:
            return 'MSE'
        return 'Unknown'
    
    df['loss_type'] = df['run_name'].apply(extract_loss_type)
    
    # Map column names to standard names
    # Training loss: train_loss_step/weighted_sum
    # Validation loss: val0_epoch/weighted_sum
    if 'train_loss_step/weighted_sum' in df.columns:
        df['train_loss'] = df['train_loss_step/weighted_sum']
        print(f"  ✓ Mapped train_loss_step/weighted_sum → train_loss")
    else:
        print(f"  ⚠ train_loss_step/weighted_sum not found!")
    
    # Try to get validation loss
    val_loss_computed = False
    
    if 'val0_epoch/weighted_sum' in df.columns:
        df['val_loss'] = df['val0_epoch/weighted_sum']
        val_non_null = df['val_loss'].notna().sum()
        val_total = len(df)
        val_pct = 100 * val_non_null / val_total if val_total > 0 else 0
        print(f"  ✓ Found val0_epoch/weighted_sum")
        print(f"    Non-null values: {val_non_null}/{val_total} ({val_pct:.1f}%)")
        
        # If weighted_sum is mostly empty (< 5% non-null), compute from components
        if val_pct < 5.0:
            print(f"    ⚠ weighted_sum is mostly empty ({val_pct:.1f}%), computing from components...")
            val_loss_computed = True
        else:
            print(f"    ✓ Using val0_epoch/weighted_sum as validation loss")
    else:
        print(f"  ⚠ val0_epoch/weighted_sum not found, computing from components...")
        val_loss_computed = True
    
    # Compute validation loss from components if needed
    if val_loss_computed or 'val_loss' not in df.columns:
        component_cols = [
            'val0_epoch/forces_rmse',
            'val0_epoch/stress_rmse', 
            'val0_epoch/per_atom_energy_rmse'
        ]
        
        missing_cols = [c for c in component_cols if c not in df.columns]
        if missing_cols:
            print(f"    ⚠ Missing component columns: {missing_cols}")
        
        available_cols = [c for c in component_cols if c in df.columns]
        if available_cols:
            print(f"    Computing validation loss from: {available_cols}")
            
            # Debug: Check component values
            for col in available_cols:
                non_null = df[col].notna().sum()
                non_zero = (df[col] != 0).sum() if non_null > 0 else 0
                if non_null > 0:
                    col_min = df[col].min()
                    col_max = df[col].max()
                    print(f"      {col}: {non_null} non-null, {non_zero} non-zero, range [{col_min:.4f}, {col_max:.4f}]")
            
            # Sum the available components (only where at least one component is non-null)
            # Use fillna(0) to handle NaN, but then check if result is meaningful
            component_sum = df[available_cols].fillna(0).sum(axis=1)
            
            # Check if any component had non-null values for each row
            has_any_component = df[available_cols].notna().any(axis=1)
            
            # Only use sum where at least one component exists, otherwise NaN
            df['val_loss'] = component_sum.where(has_any_component, np.nan)
            
            val_non_null = df['val_loss'].notna().sum()
            val_non_zero = (df['val_loss'] != 0).sum()
            val_total = len(df)
            val_pct = 100 * val_non_null / val_total if val_total > 0 else 0
            print(f"    ✓ Computed val_loss from components")
            print(f"    Non-null values: {val_non_null}/{val_total} ({val_pct:.1f}%)")
            print(f"    Non-zero values: {val_non_zero}/{val_total} ({100*val_non_zero/val_total:.1f}%)")
            
            if val_non_zero == 0:
                print(f"    ⚠ WARNING: All computed val_loss values are zero!")
                print(f"    This suggests the component columns are all zero or NaN.")
                print(f"    Consider checking if 'val0_epoch/weighted_sum' should be used instead.")
        else:
            print(f"    ✗ No component columns available, validation loss will be empty")
            df['val_loss'] = np.nan
    
    return df


def plot_training_curves(training_df: pd.DataFrame, output_dir: Path):
    """Plot training curves for all loss functions."""
    
    print("\n" + "="*80)
    print("PLOTTING TRAINING CURVES")
    print("="*80)
    print(f"  Total rows in training_df: {len(training_df)}")
    print(f"  Columns: {list(training_df.columns)[:15]}...")
    print(f"  Has 'val_loss' column: {'val_loss' in training_df.columns}")
    print(f"  Has 'trainer/global_step' column: {'trainer/global_step' in training_df.columns}")
    print(f"  Has '_step' column: {'_step' in training_df.columns}")
    print(f"  Has 'epoch' column: {'epoch' in training_df.columns}")
    
    if 'val_loss' in training_df.columns:
        val_non_null = training_df['val_loss'].notna().sum()
        val_total = len(training_df)
        print(f"  val_loss non-null: {val_non_null}/{val_total} ({100*val_non_null/val_total:.1f}%)")
        if val_non_null > 0:
            print(f"  val_loss sample values: {training_df['val_loss'].dropna().head(5).tolist()}")
    
    if 'epoch' in training_df.columns:
        epoch_non_null = training_df['epoch'].notna().sum()
        epoch_total = len(training_df)
        print(f"  epoch non-null: {epoch_non_null}/{epoch_total} ({100*epoch_non_null/epoch_total:.1f}%)")
        if epoch_non_null > 0:
            print(f"  epoch range: {training_df['epoch'].min():.1f} to {training_df['epoch'].max():.1f}")
    
    if 'run_name' in training_df.columns:
        print(f"  Unique runs: {training_df['run_name'].nunique()}")
        print(f"  Unique loss_types: {training_df['loss_type'].nunique() if 'loss_type' in training_df.columns else 0}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    loss_types = ['MSE', 'RMSE', 'TailHuber', 'StratHuber', 'RMCE', 'RMQE']
    colors = {
        'MSE': '#2A33C3',        # Blue
        'RMSE': '#A35D00',        # Orange/Brown
        'TailHuber': '#0B7285',   # Teal/Cyan
        'StratHuber': '#8F2D56',  # Magenta/Pink
        'RMCE': '#6E8B00',        # Green
        'RMQE': '#4A5568'         # Dark gray-blue
    }
    
    # Determine x-axis column: use epoch if available, otherwise _step
    use_epoch = 'epoch' in training_df.columns
    x_col = 'epoch' if use_epoch else '_step'
    x_label = 'Epoch' if use_epoch else 'Training Step'
    print(f"\n  Using x-axis column: {x_col}")
    
    # 1. Training loss
    ax = axes[0, 0]
    for loss_type in loss_types:
        data = training_df[training_df['loss_type'] == loss_type].copy()
        if len(data) > 0 and 'train_loss' in data.columns:
            # Filter out NaN values
            data_clean = data.dropna(subset=['train_loss', x_col])
            
            if len(data_clean) > 0:
                # Plot individual runs (light lines)
                for run in data_clean['run_name'].unique():
                    run_data = data_clean[data_clean['run_name'] == run]
                    if len(run_data) > 0:
                        ax.plot(run_data[x_col], run_data['train_loss'], 
                               color=colors.get(loss_type, 'gray'), 
                               alpha=0.58, linewidth=1)
                
                # Plot mean (bold line)
                grouped = data_clean.groupby(x_col)['train_loss'].mean()
                if len(grouped) > 0:
                    ax.plot(grouped.index, grouped.values, 
                           color=colors[loss_type], linewidth=2, 
                           label=loss_type, alpha=0.58)
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Curves', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 2. Validation loss (step-based in wandb)
    ax = axes[0, 1]
    print("\n  Plotting validation curves...")
    # Validation loss is logged at training steps, not epochs
    # Try trainer/global_step first, then _step, then epoch as fallback
    if 'trainer/global_step' in training_df.columns:
        val_x_col = 'trainer/global_step'
        val_x_label = 'Training Step'
    elif '_step' in training_df.columns:
        val_x_col = '_step'
        val_x_label = 'Training Step'
    elif 'epoch' in training_df.columns:
        val_x_col = 'epoch'
        val_x_label = 'Epoch'
    else:
        val_x_col = x_col
        val_x_label = x_label
    print(f"    Using x-axis: {val_x_col}")
    
    validation_plotted = False
    for loss_type in loss_types:
        data = training_df[training_df['loss_type'] == loss_type].copy()
        print(f"\n    {loss_type}:")
        print(f"      Total rows: {len(data)}")
        
        if len(data) == 0:
            print(f"      ✗ No data for this loss type")
            continue
            
        if 'val_loss' not in data.columns:
            print(f"      ✗ 'val_loss' column missing")
            continue
        
        # Check val_loss availability
        val_before_filter = data['val_loss'].notna().sum()
        print(f"      val_loss non-null before filter: {val_before_filter}/{len(data)}")
        
        # Filter to only rows with non-null validation loss (epoch-based data)
        data_clean = data.dropna(subset=['val_loss'])
        print(f"      After dropping NaN val_loss: {len(data_clean)} rows")
        
        # Also need the x-axis column
        if val_x_col in data_clean.columns:
            before_x_filter = len(data_clean)
            data_clean = data_clean.dropna(subset=[val_x_col])
            print(f"      After dropping NaN {val_x_col}: {len(data_clean)} rows (removed {before_x_filter - len(data_clean)})")
        else:
            print(f"      ⚠ {val_x_col} column not found in data!")
        
        if len(data_clean) > 0:
            print(f"      ✓ {len(data_clean)} validation data points")
            print(f"      Unique runs: {data_clean['run_name'].nunique()}")
            print(f"      {val_x_col} range: {data_clean[val_x_col].min():.1f} to {data_clean[val_x_col].max():.1f}")
            print(f"      val_loss range: {data_clean['val_loss'].min():.4f} to {data_clean['val_loss'].max():.4f}")
            
            # Filter out zero values for plotting (they don't show on log scale)
            data_to_plot = data_clean[data_clean['val_loss'] > 0].copy()
            if len(data_to_plot) > 0:
                print(f"      Non-zero val_loss points: {len(data_to_plot)}/{len(data_clean)}")
                validation_plotted = True
                # Plot individual runs (light lines)
                for run in data_to_plot['run_name'].unique():
                    run_data = data_to_plot[data_to_plot['run_name'] == run]
                    if len(run_data) > 0:
                        # Sort by x-axis for proper line plotting
                        run_data = run_data.sort_values(val_x_col)
                        ax.plot(run_data[val_x_col], run_data['val_loss'], 
                               color=colors.get(loss_type, 'gray'), 
                               alpha=0.58, linewidth=1)
                
                # Plot mean (bold line) - group by epoch/x_col
                grouped = data_to_plot.groupby(val_x_col)['val_loss'].mean().sort_index()
                if len(grouped) > 0:
                    ax.plot(grouped.index, grouped.values, 
                           color=colors[loss_type], linewidth=2, 
                           label=loss_type, alpha=0.58)
            else:
                print(f"      ⚠ All val_loss values are zero - cannot plot on log scale")
        else:
            print(f"      ✗ No validation data after filtering")
    
    if not validation_plotted:
        print(f"\n    ⚠ WARNING: No validation curves were plotted!")
    
    ax.set_xlabel(val_x_label, fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Validation Curves', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 3. Loss stability (standard deviation over time)
    ax = axes[1, 0]
    for loss_type in loss_types:
        data = training_df[training_df['loss_type'] == loss_type].copy()
        if len(data) > 0 and 'train_loss' in data.columns and x_col in data.columns:
            # Filter out NaN values
            data_clean = data.dropna(subset=['train_loss', x_col])
            if len(data_clean) > 0:
                # Compute std across runs at each step/epoch
                grouped = data_clean.groupby(x_col)['train_loss'].std()
                if len(grouped) > 0:
                    # Filter out zero values for log scale
                    grouped_positive = grouped[grouped > 0]
                    if len(grouped_positive) > 0:
                        ax.plot(grouped_positive.index, grouped_positive.values, 
                               color=colors[loss_type], linewidth=2, 
                               label=loss_type, alpha=0.58)
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Loss Std Dev (across seeds)', fontsize=12)
    ax.set_title('Training Stability\n(Lower = more reproducible)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 4. Final convergence
    ax = axes[1, 1]
    print("\n  Plotting final convergence...")
    final_losses = []
    loss_type_labels = []
    
    # Use step-based x-axis for validation (same as validation curves)
    if 'trainer/global_step' in training_df.columns:
        val_x_col = 'trainer/global_step'
    elif '_step' in training_df.columns:
        val_x_col = '_step'
    elif 'epoch' in training_df.columns:
        val_x_col = 'epoch'
    else:
        val_x_col = x_col
    print(f"    Using x-axis: {val_x_col}")
    
    for loss_type in loss_types:
        data = training_df[training_df['loss_type'] == loss_type].copy()
        print(f"\n    {loss_type}:")
        
        if len(data) == 0:
            print(f"      ✗ No data for this loss type")
            continue
            
        if 'val_loss' not in data.columns:
            print(f"      ✗ 'val_loss' column missing")
            continue
        
        unique_runs = data['run_name'].unique()
        print(f"      Found {len(unique_runs)} runs")
        
        # Get final validation loss for each run
        for run in unique_runs:
            run_data = data[data['run_name'] == run].copy()
            print(f"        Run: {run}")
            print(f"          Rows: {len(run_data)}")
            
            # Filter to only non-null validation loss
            run_data_clean = run_data.dropna(subset=['val_loss'])
            print(f"          After val_loss filter: {len(run_data_clean)} rows")
            
            if val_x_col in run_data_clean.columns:
                before_x_filter = len(run_data_clean)
                run_data_clean = run_data_clean.dropna(subset=[val_x_col])
                print(f"          After {val_x_col} filter: {len(run_data_clean)} rows")
            else:
                print(f"          ⚠ {val_x_col} column not found!")
            
            if len(run_data_clean) > 0:
                # Sort to get the last value
                run_data_clean = run_data_clean.sort_values(val_x_col)
                final_val = run_data_clean['val_loss'].iloc[-1]
                print(f"          Final {val_x_col}: {run_data_clean[val_x_col].iloc[-1]:.1f}")
                print(f"          Final val_loss: {final_val:.4f}")
                
                # Only add non-zero values (zeros won't show on log scale)
                if not np.isnan(final_val) and np.isfinite(final_val) and final_val > 0:
                    final_losses.append(final_val)
                    loss_type_labels.append(loss_type)
                    print(f"          ✓ Added to final_losses (non-zero)")
                elif final_val == 0:
                    print(f"          ⚠ Final value is zero - skipping (won't show on log scale)")
                else:
                    print(f"          ✗ Final value is NaN or infinite")
            else:
                print(f"          ✗ No data after filtering")
    
    print(f"\n    Total final_losses collected: {len(final_losses)}")
    print(f"    Loss types: {set(loss_type_labels)}")
    
    if final_losses:
        df_final = pd.DataFrame({'loss_type': loss_type_labels, 'final_val_loss': final_losses})
        
        positions = []
        labels = []
        data_to_plot = []
        
        for i, loss_type in enumerate(loss_types):
            subset = df_final[df_final['loss_type'] == loss_type]['final_val_loss'].values
            if len(subset) > 0:
                positions.append(i)
                labels.append(loss_type)
                data_to_plot.append(subset)
        
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                        patch_artist=True, showfliers=True)
        
        for patch, loss_type in zip(bp['boxes'], labels):
            patch.set_facecolor(colors[loss_type])
            patch.set_alpha(0.58)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel('Final Validation Loss', fontsize=12)
        ax.set_title('Final Convergence\n(Across seeds)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved training analysis: {output_dir / 'training_analysis.png'}")
    plt.close()


def plot_error_distributions(detailed_df: pd.DataFrame, output_dir: Path):
    """Plot error distributions showing mean, max, 95th percentile."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Group by loss type
    detailed_df['loss_type'] = detailed_df['model'].apply(
        lambda x: x.split('_model')[0] if '_model' in x else x
    )
    
    loss_types = ['MSE', 'RMSE', 'TailHuber', 'StratHuber', 'RMCE', 'RMQE']
    colors = {
        'MSE': '#2A33C3',        # Blue
        'RMSE': '#A35D00',        # Orange/Brown
        'TailHuber': '#0B7285',   # Teal/Cyan
        'StratHuber': '#8F2D56',  # Magenta/Pink
        'RMCE': '#6E8B00',        # Green
        'RMQE': '#4A5568'         # Dark gray-blue
    }
    
    # 1. Mean errors
    ax = axes[0]
    stats_data = []
    for loss_type in loss_types:
        subset = detailed_df[detailed_df['loss_type'] == loss_type]
        if len(subset) > 0:
            stats_data.append({
                'loss_type': loss_type,
                'mean': subset['defect_mae'].mean(),
                'std': subset['defect_mae'].std()
            })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        bars = ax.bar(stats_df['loss_type'], stats_df['mean'], 
                     yerr=stats_df['std'], capsize=5,
                     color=[colors[lt] for lt in stats_df['loss_type']],
                     alpha=0.58, edgecolor='k', linewidth=1.5)
        ax.set_ylabel('Defect MAE (eV/Å)', fontsize=12, fontweight='bold')
        ax.set_title('Mean Error\n(with std dev)', fontsize=13, fontweight='bold')
        ax.set_xticklabels(stats_df['loss_type'], rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Max errors (worst case)
    ax = axes[1]
    max_data = []
    for loss_type in loss_types:
        subset = detailed_df[detailed_df['loss_type'] == loss_type]
        if len(subset) > 0:
            max_data.append({
                'loss_type': loss_type,
                'max': subset['defect_mae'].max(),
                'mean_max': subset.groupby('model')['defect_mae'].max().mean()
            })
    
    if max_data:
        max_df = pd.DataFrame(max_data)
        bars = ax.bar(max_df['loss_type'], max_df['mean_max'],
                     color=[colors[lt] for lt in max_df['loss_type']],
                     alpha=0.58, edgecolor='k', linewidth=1.5)
        ax.set_ylabel('Max Defect MAE (eV/Å)', fontsize=12, fontweight='bold')
        ax.set_title('Worst-Case Error\n(Maximum across structures)', fontsize=13, fontweight='bold')
        ax.set_xticklabels(max_df['loss_type'], rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 3. 95th percentile
    ax = axes[2]
    percentile_data = []
    for loss_type in loss_types:
        subset = detailed_df[detailed_df['loss_type'] == loss_type]
        if len(subset) > 0:
            p95 = subset['defect_mae'].quantile(0.95)
            percentile_data.append({
                'loss_type': loss_type,
                'p95': p95
            })
    
    if percentile_data:
        p95_df = pd.DataFrame(percentile_data)
        bars = ax.bar(p95_df['loss_type'], p95_df['p95'],
                     color=[colors[lt] for lt in p95_df['loss_type']],
                     alpha=0.58, edgecolor='k', linewidth=1.5)
        ax.set_ylabel('95th Percentile MAE (eV/Å)', fontsize=12, fontweight='bold')
        ax.set_title('High-Error Structures\n(95th percentile)', fontsize=13, fontweight='bold')
        ax.set_xticklabels(p95_df['loss_type'], rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distributions.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved error distributions: {output_dir / 'error_distributions.png'}")
    plt.close()


def statistical_significance_tests(detailed_df: pd.DataFrame, output_dir: Path):
    """Perform statistical tests comparing loss functions."""
    
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)
    
    results = []
    
    # Extract loss type from model column (same as plot_error_distributions)
    detailed_df = detailed_df.copy()
    detailed_df['loss_type'] = detailed_df['model'].apply(
        lambda x: x.split('_model')[0] if '_model' in x else x
    )
    
    # Get loss types
    loss_types = sorted(detailed_df['loss_type'].unique())
    
    # Pairwise t-tests
    print("\nPairwise t-tests (defect MAE):")
    print("-" * 80)
    
    for i, loss1 in enumerate(loss_types):
        for loss2 in loss_types[i+1:]:
            # Get all defect_mae values for each loss type (across all structures and seeds)
            data1 = detailed_df[detailed_df['loss_type'] == loss1]['defect_mae'].dropna().values
            data2 = detailed_df[detailed_df['loss_type'] == loss2]['defect_mae'].dropna().values
            
            # Need at least 2 data points per group for t-test
            if len(data1) >= 2 and len(data2) >= 2:
                t_stat, p_value = stats.ttest_ind(data1, data2)
                
                # Handle NaN results (can occur if variances are zero)
                if np.isnan(t_stat) or np.isnan(p_value):
                    sig = "ns"
                    print(f"{loss1:12s} vs {loss2:12s}: t=    nan, p=nan {sig:3s}  "
                          f"(Δ={data1.mean()-data2.mean():+.4f})")
                else:
                    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    
                    results.append({
                        'comparison': f'{loss1} vs {loss2}',
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': sig,
                        'mean_diff': data1.mean() - data2.mean(),
                        'n1': len(data1),
                        'n2': len(data2)
                    })
                    
                    print(f"{loss1:12s} vs {loss2:12s}: t={t_stat:7.3f}, p={p_value:.4f} {sig:3s}  "
                          f"(Δ={data1.mean()-data2.mean():+.4f}, n1={len(data1)}, n2={len(data2)})")
            else:
                # Not enough data points
                print(f"{loss1:12s} vs {loss2:12s}: Insufficient data (n1={len(data1)}, n2={len(data2)})")
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / 'statistical_tests.csv', index=False)
        print(f"\n✓ Saved statistical tests: {output_dir / 'statistical_tests.csv'}")
    else:
        print("\n⚠ No valid statistical tests could be performed!")
    
    print("\nSignificance codes: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")
    print("="*80)


def create_publication_figure(ensemble_df: pd.DataFrame, detailed_df: pd.DataFrame,
                              output_dir: Path):
    """Create comprehensive publication-quality figure."""
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    colors = {
        'MSE': '#2A33C3',        # Blue
        'RMSE': '#A35D00',        # Orange/Brown
        'TailHuber': '#0B7285',   # Teal/Cyan
        'StratHuber': '#8F2D56',  # Magenta/Pink
        'RMCE': '#6E8B00',        # Green
        'RMQE': '#4A5568'         # Dark gray-blue
    }
    
    loss_types = ensemble_df['loss_type'].values
    
    # 1. Defect errors with error bars
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(loss_types))
    defect_means = ensemble_df['defect_mae_mean'].values
    defect_stds = ensemble_df['defect_mae_std'].values
    
    bars = ax1.bar(x, defect_means, yerr=defect_stds, capsize=5,
                  color=[colors[lt] for lt in loss_types],
                  alpha=0.58, edgecolor='k', linewidth=1.5)
    ax1.set_ylabel('Defect MAE (eV/Å)', fontsize=11, fontweight='bold')
    ax1.set_title('A. Defect Region Errors', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(loss_types, rotation=15, ha='right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Bulk errors with error bars
    ax2 = fig.add_subplot(gs[0, 1])
    bulk_means = ensemble_df['bulk_mae_mean'].values
    bulk_stds = ensemble_df['bulk_mae_std'].values
    
    bars = ax2.bar(x, bulk_means, yerr=bulk_stds, capsize=5,
                  color=[colors[lt] for lt in loss_types],
                  alpha=0.58, edgecolor='k', linewidth=1.5)
    ax2.set_ylabel('Bulk MAE (eV/Å)', fontsize=11, fontweight='bold')
    ax2.set_title('B. Bulk Region Errors', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(loss_types, rotation=15, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Ratio comparison
    ax3 = fig.add_subplot(gs[0, 2])
    ratio_means = ensemble_df['ratio_mean'].values
    ratio_stds = ensemble_df['ratio_std'].values
    
    bars = ax3.bar(x, ratio_means, yerr=ratio_stds, capsize=5,
                  color=[colors[lt] for lt in loss_types],
                  alpha=0.58, edgecolor='k', linewidth=1.5)
    ax3.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_ylabel('Defect/Bulk Ratio', fontsize=11, fontweight='bold')
    ax3.set_title('C. Error Localization', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(loss_types, rotation=15, ha='right', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4-6. Box plots showing distributions
    detailed_df['loss_type'] = detailed_df['model'].apply(
        lambda x: x.split('_model')[0] if '_model' in x else x
    )
    
    ax4 = fig.add_subplot(gs[1, :])
    
    data_to_plot = []
    labels = []
    for loss_type in loss_types:
        subset = detailed_df[detailed_df['loss_type'] == loss_type]['defect_mae'].values
        if len(subset) > 0:
            data_to_plot.append(subset)
            labels.append(loss_type)
    
    if data_to_plot:
        bp = ax4.boxplot(data_to_plot, labels=labels, widths=0.6,
                        patch_artist=True, showfliers=True)
        
        for patch, label in zip(bp['boxes'], labels):
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.58)
        
        ax4.set_ylabel('Defect MAE (eV/Å)', fontsize=11, fontweight='bold')
        ax4.set_title('D. Error Distribution Across All Structures & Seeds', 
                     fontsize=12, fontweight='bold')
        ax4.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 7. Coefficient of variation (reproducibility)
    ax5 = fig.add_subplot(gs[2, 0])
    cv = (ensemble_df['defect_mae_std'] / ensemble_df['defect_mae_mean']) * 100
    
    bars = ax5.bar(loss_types, cv,
                  color=[colors[lt] for lt in loss_types],
                  alpha=0.58, edgecolor='k', linewidth=1.5)
    ax5.set_ylabel('Coefficient of Variation (%)', fontsize=11, fontweight='bold')
    ax5.set_title('E. Reproducibility\n(Lower = More stable)', fontsize=12, fontweight='bold')
    ax5.set_xticklabels(loss_types, rotation=15, ha='right', fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 8. Summary table
    ax6 = fig.add_subplot(gs[2, 1:])
    ax6.axis('off')
    
    table_data = []
    for _, row in ensemble_df.iterrows():
        table_data.append([
            row['loss_type'],
            f"{row['defect_mae_mean']:.3f}±{row['defect_mae_std']:.3f}",
            f"{row['bulk_mae_mean']:.3f}±{row['bulk_mae_std']:.3f}",
            f"{row['ratio_mean']:.2f}±{row['ratio_std']:.2f}",
            f"{row['n_models']:.0f}"
        ])
    
    table = ax6.table(cellText=table_data,
                     colLabels=['Loss', 'Defect MAE', 'Bulk MAE', 'Ratio', 'N'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code rows
    for i, loss_type in enumerate(loss_types):
        table[(i+1, 0)].set_facecolor(colors[loss_type])
        table[(i+1, 0)].set_alpha(0.58)
    
    ax6.set_title('F. Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.savefig(output_dir / 'publication_figure.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved publication figure: {output_dir / 'publication_figure.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Combined analysis: training + testing + statistics')
    
    parser.add_argument('--ensemble_summary', type=str, required=True,
                       help='Path to ensemble_summary.csv')
    
    parser.add_argument('--detailed_results', type=str, required=True,
                       help='Path to detailed_results.csv')
    
    parser.add_argument('--training_history', type=str, default=None,
                       help='Path to training_history.csv (optional)')
    
    parser.add_argument('--val_weighted_sums', type=str, default=None,
                       help='Path to val0_weighted_sums.csv with validation weighted sums (optional)')
    
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("COMPREHENSIVE LOSS FUNCTION ANALYSIS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    ensemble_df = pd.read_csv(args.ensemble_summary)
    detailed_df = pd.read_csv(args.detailed_results)
    
    print(f"  Ensemble summary: {len(ensemble_df)} loss types")
    print(f"  Detailed results: {len(detailed_df)} data points")
    
    # Training curves
    if args.training_history:
        print(f"  Training history: {args.training_history}")
        val_sums_path = args.val_weighted_sums
        if val_sums_path is None:
            # Try to find it in the same directory as training_history
            training_dir = Path(args.training_history).parent
            default_val_sums = training_dir / 'val0_weighted_sums.csv'
            if default_val_sums.exists():
                val_sums_path = str(default_val_sums)
                print(f"  Found val0_weighted_sums.csv in same directory, using it")
        
        training_df = load_training_history(args.training_history, val_weighted_sums_path=val_sums_path)
        plot_training_curves(training_df, output_dir)
    else:
        print("  ⚠ No training history provided, skipping training curves")
    
    # Error distributions
    print("\nAnalyzing error distributions...")
    plot_error_distributions(detailed_df, output_dir)
    
    # Statistical tests
    print("\nPerforming statistical significance tests...")
    statistical_significance_tests(detailed_df, output_dir)
    
    # Publication figure
    print("\nCreating publication figure...")
    create_publication_figure(ensemble_df, detailed_df, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nKey outputs:")
    print("  - training_analysis.png (if training history provided)")
    print("  - error_distributions.png")
    print("  - publication_figure.png")
    print("  - statistical_tests.csv")


if __name__ == '__main__':
    main()

