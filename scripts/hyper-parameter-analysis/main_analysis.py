import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
CSV_PATH = "wandb_export.csv"  # change this
THROUGHPUT_COL = "throughput"  # samples/sec (or similar)
ERROR_COL = "val_mae"          # change to your main metric (e.g., 'val_rmse')
SEED_COL = "seed"              # set to None if not present
FOLD_COL = "fold"              # set to None if not present
HARDWARE_COL = "gpu_name"      # set to None if homogeneous
HARDWARE_FILTER = None         # e.g., "A100" to filter to a single device
BATCH_COL = "batch_size"       # set to None if homogeneous
BATCH_FILTER = None            # e.g., 128

# Hyperparameters that define a unique configuration:
CONFIG_COLS = [
    "R_max", "L_max", "N_layers", "n_tensor_features", "n_scalar_features", "mlp_width"
]

# Optional: if available, these enrich the plot
PARAMS_COL = "param_count"     # set to None if not available

# Throughput requirement (edit as needed)
REQUIRED_T = 1000.0  # samples/sec
ALPHA = 0.05         # for 95% CI

# ---------- LOAD ----------
df = pd.read_csv(CSV_PATH)

# basic cleaning
df = df.copy()
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[THROUGHPUT_COL, ERROR_COL])

# optional stratification by hardware / batch size for apples-to-apples
if HARDWARE_COL and HARDWARE_FILTER is not None:
    df = df[df[HARDWARE_COL] == HARDWARE_FILTER]
if BATCH_COL and BATCH_FILTER is not None:
    df = df[df[BATCH_COL] == BATCH_FILTER]

# ---------- GROUP KEY ----------
present_config_cols = [c for c in CONFIG_COLS if c in df.columns]
if not present_config_cols:
    raise ValueError("No hyperparameter columns from CONFIG_COLS were found in the CSV.")
group_cols = present_config_cols.copy()

# If you want a stricter definition of a replicate (e.g., seed & fold),
# DO NOT include seed/fold in group_cols; they define replicate units *within* each config.
replicate_id_cols = []
if SEED_COL and SEED_COL in df.columns: replicate_id_cols.append(SEED_COL)
if FOLD_COL and FOLD_COL in df.columns: replicate_id_cols.append(FOLD_COL)

# ---------- AGGREGATE REPLICATES ----------
def agg_ci(series, alpha=ALPHA):
    n = series.count()
    mean = series.mean()
    std = series.std(ddof=1)
    if n <= 1 or pd.isna(std):
        halfwidth = np.nan
    else:
        # Normal approx; for small n you may use t-crit
        z = 1.96 if abs(alpha - 0.05) < 1e-9 else 1.96
        halfwidth = z * std / math.sqrt(n)
    return pd.Series({"mean": mean, "std": std, "n": n, "ci_halfwidth": halfwidth})

agg = df.groupby(group_cols).agg({
    THROUGHPUT_COL: lambda s: agg_ci(s),
    ERROR_COL:      lambda s: agg_ci(s),
    **({PARAMS_COL: ['mean'] } if PARAMS_COL and PARAMS_COL in df.columns else {})
})

# flatten multiindex columns
agg.columns = ['_'.join([c for c in col if c]) for col in agg.columns.values]
agg = agg.reset_index()

# Pessimistic (robust) estimates: worst-case within 95% CI
agg["thr_mean"] = agg[f"{THROUGHPUT_COL}_mean"]
agg["thr_l95"]  = agg[f"{THROUGHPUT_COL}_mean"] - agg[f"{THROUGHPUT_COL}_ci_halfwidth"]

agg["err_mean"] = agg[f"{ERROR_COL}_mean"]
agg["err_u95"]  = agg[f"{ERROR_COL}_mean"] + agg[f"{ERROR_COL}_ci_halfwidth"]

if PARAMS_COL and f"{PARAMS_COL}_mean" in agg.columns:
    agg["params"] = agg[f"{PARAMS_COL}_mean"]
else:
    agg["params"] = np.nan

# ---------- FEASIBILITY FILTER ----------
feasible = agg[agg["thr_l95"] >= REQUIRED_T].copy()

# ---------- 2D PARETO FRONT (error vs throughput) ----------
# A point (e1, t1) is dominated by (e2, t2) if e2 <= e1 and t2 >= t1 with at least one strict.
def pareto_nondominated(df2d, err_col="err_u95", thr_col="thr_l95"):
    A = df2d[[err_col, thr_col]].values
    n = A.shape[0]
    is_dom = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dom[i]:
            continue
        e1, t1 = A[i]
        # dominated if any j has e2 <= e1 and t2 >= t1 with at least one strict
        mask = (A[:,0] <= e1) & (A[:,1] >= t1) & ((A[:,0] < e1) | (A[:,1] > t1))
        mask[i] = False
        if mask.any():
            is_dom[i] = True
    return df2d.loc[~is_dom].copy()

front_all = pareto_nondominated(agg, "err_u95", "thr_l95")
front_feasible = pareto_nondominated(feasible, "err_u95", "thr_l95")

# Best feasible (smallest pessimistic error)
best_row = None
if len(feasible):
    best_idx = feasible["err_u95"].idxmin()
    best_row = feasible.loc[best_idx]

# ---------- PLOT ----------
plt.figure(figsize=(8, 6))
plt.scatter(agg["thr_l95"], agg["err_u95"], s=np.clip(agg["params"].fillna(50)/1000.0, 20, 200), alpha=0.4, label="All configs")
if len(front_all):
    plt.plot(front_all.sort_values("thr_l95")["thr_l95"], front_all.sort_values("thr_l95")["err_u95"], lw=2, label="Pareto front (all)")

# Highlight feasible region
plt.axvline(REQUIRED_T, linestyle="--", linewidth=1.5, label=f"Required throughput T={REQUIRED_T}")

# Feasible points
if len(feasible):
    plt.scatter(feasible["thr_l95"], feasible["err_u95"], edgecolor="k", linewidth=0.5, alpha=0.9, label="Feasible (thr_l95 ≥ T)")

# Frontier in feasible region
if len(front_feasible):
    plt.plot(front_feasible.sort_values("thr_l95")["thr_l95"], front_feasible.sort_values("thr_l95")["err_u95"],
             linewidth=2.5, label="Pareto front (feasible)")

# Best point annotation
if best_row is not None:
    x, y = best_row["thr_l95"], best_row["err_u95"]
    plt.scatter([x], [y], s=180, marker="*", label="Best feasible (min err_u95)")
    # build a short label from config
    label = ", ".join(f"{c}={best_row[c]}" for c in group_cols[:3]) + " …"
    plt.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)

plt.xlabel("Throughput (pessimistic) — lower 95% CI")
plt.ylabel("Validation error (pessimistic) — upper 95% CI")
plt.title("Error vs Throughput with Robust Pareto Front and Feasibility Threshold")
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.show()

# ---------- OPTIONAL: export tables ----------
front_feasible_sorted = front_feasible.sort_values(["err_u95", "thr_l95"], ascending=[True, False])
front_feasible_sorted.to_csv("feasible_pareto.csv", index=False)
if best_row is not None:
    pd.DataFrame([best_row]).to_csv("best_feasible_choice.csv", index=False)

print(f"\nFeasible configs: {len(feasible)} / {len(agg)} total")
if best_row is not None:
    print("Best feasible (min err_u95):")
    print(best_row[group_cols + ['thr_l95','err_u95','params'] if 'params' in best_row else group_cols + ['thr_l95','err_u95']])
