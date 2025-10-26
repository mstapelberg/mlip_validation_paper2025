#!/usr/bin/env python
"""
analyse_neb_results.py
Collects vacancy‑diffusion data from multiple sub‑folders, then
produces Nature‑style summary figures.

Folder layout expected
----------------------
simple_data/
   ◦ comp_A/
       • perf/POSCAR, perf/OUTCAR
       • vac/POSCAR_Start*, POSCAR_End*, OUTCAR_Start*, OUTCAR_End*
       • neb/00.xyz …  (VASP images)
   ◦ comp_B/
       • …
simple_results/
   ◦ comp_A/  (written by your existing MLIP script)
       • mlip_start_vac_opt.traj
       • mlip_end_vac_opt.traj
       • mlip_neb_relaxed.traj  (written in earlier run)
   ◦ …

Run:
    $ python analyse_neb_results.py  # writes *.pdf and *.csv
"""
import os, glob, re, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from ase.io import read                   # ASE I/O  [oai_citation:4‡CAMD Wiki](https://wiki.fysik.dtu.dk/ase/ase/io/io.html?utm_source=chatgpt.com)

# ───────────────────────── Visual defaults ──────────────────────────
WONG8 = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
         '#F0E442', '#56B4E9', '#E69F00', '#000000']  # Wong 2011  [oai_citation:5‡Nature](https://www.nature.com/articles/nmeth.1618?utm_source=chatgpt.com)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',  # robust default on most systems
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif'],
    'font.size'  : 8,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.0,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'axes.unicode_minus': False,
    'axes.prop_cycle': plt.cycler(color=WONG8)
})

# ───────────────────────── Helper functions ─────────────────────────
def _energy_from_atoms(atoms, context_label=""):
    """Return energy as float from an ASE Atoms object using ASE-native data.

    Order of precedence:
      1) atoms.get_potential_energy()
      2) atoms.info in keys ['energy','potential_energy','free_energy','total_energy','E0']
      3) atoms.calc.results in keys ['energy','potential_energy','free_energy']
    """
    # Primary: calculator-backed property
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        pass

    # Fallback: values parsed into info
    if hasattr(atoms, 'info') and isinstance(atoms.info, dict):
        for key in ('energy', 'potential_energy', 'free_energy', 'total_energy', 'E0', 'REF_energy'):
            if key in atoms.info:
                try:
                    return float(atoms.info[key])
                except Exception:
                    continue

    # Fallback: values present in calculator results
    if getattr(atoms, 'calc', None) is not None:
        try:
            results = getattr(atoms.calc, 'results', {}) or {}
            for key in ('energy', 'potential_energy', 'free_energy'):
                if key in results:
                    return float(results[key])
        except Exception:
            pass

    raise ValueError(
        f"Energy not available{(' in ' + context_label) if context_label else ''}. "
        f"info keys: {list(getattr(atoms, 'info', {}).keys())}"
    )

def _read_traj_energies(traj_path):
    """Return list of energies (eV) from an ASE trajectory or OUTCAR file."""
    print(f"DEBUG: Reading energies from {traj_path}")
    try:
        traj = read(traj_path, index=':')
        print(f"DEBUG: Read {len(traj)} structures from {traj_path}")
    except Exception as e:
        print(f"DEBUG: Error reading {traj_path}: {e}")
        raise

    energies = []
    for idx, atoms in enumerate(traj):
        energies.append(_energy_from_atoms(atoms, context_label=f"frame {idx} of {traj_path}"))
    print(f"DEBUG: Extracted {len(energies)} energies: {energies[:3]}...")  # Show first 3
    return energies

def _read_neb_folder(xyz_folder):
    """Return energy profile (zeroed to first image) from numbered *.xyz files."""
    print(f"DEBUG: Reading NEB from {xyz_folder}")
    xyz_files = sorted(glob.glob(os.path.join(xyz_folder, '*.xyz')),
                       key=lambda x: int(re.findall(r'(\d+)\.xyz', x)[0]))
    print(f"DEBUG: Found {len(xyz_files)} .xyz files: {xyz_files[:3]}...")

    energies = []
    for i, f in enumerate(xyz_files):
        try:
            atoms = read(f)
            energy = _energy_from_atoms(atoms, context_label=f)
            energies.append(float(energy))
            print(f"DEBUG: {f} -> energy = {energies[-1]}")
        except Exception as e:
            print(f"DEBUG: Error reading {f}: {e}")
            raise

    en = np.array(energies)
    print(f"DEBUG: NEB energies: {en - en[0]}")
    return en - en[0]

def _read_neb_traj(traj_file):
    """Same as above, but for the MLIP trajectory you already saved."""
    images = read(traj_file, index=':')
    energies = [
        _energy_from_atoms(atoms, context_label=f"MLIP NEB frame {idx} of {traj_file}")
        for idx, atoms in enumerate(images)
    ]
    en = np.array(energies)
    return en - en[0]

def _resolve_mlip_dir_for_root(project_root, comp, results_root):
    """Return an existing MLIP directory for a given composition under a specific results root.

    Handles composition folders that may have a "_NN_to_MM" suffix in `simple_data/`
    but only a base composition folder name in the MLIP results.
    """
    base_comp = re.sub(r'_\d+_to_\d+$', '', comp)

    candidates = [
        os.path.join(project_root, results_root, comp),
        os.path.join(project_root, results_root, base_comp),
    ]

    for path in candidates:
        if os.path.isdir(path):
            print(f"DEBUG: Using MLIP dir: {path}")
            return path

    raise FileNotFoundError(
        f"Could not locate MLIP directory for root '{results_root}'. Tried:\n  " + "\n  ".join(candidates)
    )

def analyse_composition(folder, results_roots=None, results_labels=None):
    """Harvest energy arrays and barrier heights for one composition.

    results_roots/results_labels: parallel lists specifying MLIP results sources.
    If None, defaults to a single ['simple_results_gen_10'] with label ['gen10'].
    """
    comp = os.path.basename(folder.rstrip('/'))
    project_root = os.path.dirname(os.path.abspath(__file__))

    start_candidates = sorted(glob.glob(os.path.join(folder, 'vac', 'OUTCAR_Start*')))
    end_candidates   = sorted(glob.glob(os.path.join(folder, 'vac', 'OUTCAR_End*')))
    if not start_candidates:
        raise FileNotFoundError(f"No OUTCAR_Start* found in {os.path.join(folder, 'vac')}")
    if not end_candidates:
        raise FileNotFoundError(f"No OUTCAR_End* found in {os.path.join(folder, 'vac')}")

    vasp_start = _read_traj_energies(start_candidates[0])
    vasp_end   = _read_traj_energies(end_candidates[0])

    vasp_neb = _read_neb_folder(os.path.join(folder, 'neb'))

    # Configure defaults
    if not results_roots:
        results_roots = ['simple_results_gen_10']
    if not results_labels:
        results_labels = ['gen10']
    assert len(results_roots) == len(results_labels), "results_roots and results_labels must align"

    rec = {
        'composition' : comp,
        'vasp_start_en': vasp_start,
        'vasp_end_en'  : vasp_end,
        'vasp_neb'     : vasp_neb,
        'barrier_vasp' : float(np.max(vasp_neb)),
    }

    for root, label in zip(results_roots, results_labels):
        mlip_dir = _resolve_mlip_dir_for_root(project_root, comp, root)
        mlip_start = _read_traj_energies(os.path.join(mlip_dir, 'mlip_start_vac_opt.traj'))
        mlip_end   = _read_traj_energies(os.path.join(mlip_dir, 'mlip_end_vac_opt.traj'))
        mlip_neb_path = os.path.join(mlip_dir, 'mlip_neb_relaxed.traj')
        if not os.path.isfile(mlip_neb_path):
            raise FileNotFoundError(f"Missing MLIP NEB trajectory for {label}: {mlip_neb_path}")
        mlip_neb = _read_neb_traj(mlip_neb_path)

        rec[f'mlip_start_en_{label}'] = mlip_start
        rec[f'mlip_end_en_{label}']   = mlip_end
        rec[f'mlip_neb_{label}']      = mlip_neb
        rec[f'barrier_mlip_{label}']  = float(np.max(mlip_neb))

    return rec

# ───────────────────────── Plot utilities ───────────────────────────
def parity_plot(df, labels, ax=None):
    ax = ax or plt.gca()
    # Collect all finite barrier values across mlip labels to set limits
    all_vals = [df['barrier_vasp'].to_numpy(dtype=float)]
    for label in labels:
        col = f'barrier_mlip_{label}'
        if col in df.columns:
            all_vals.append(df[col].to_numpy(dtype=float))
    all_concat = np.concatenate([v[np.isfinite(v)] for v in all_vals if v.size]) if all_vals else np.array([1.0])
    lim = 1.1 * (all_concat.max() if all_concat.size else 1.0)

    # Plot parity line
    ax.plot([0, lim], [0, lim], '--', color='k', lw=0.8)

    # Plot points per label
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X']
    for i, label in enumerate(labels):
        col = f'barrier_mlip_{label}'
        if col not in df.columns:
            continue
        ax.scatter(df['barrier_vasp'], df[col], s=30, marker=markers[i % len(markers)], label=label)

    def _fmt(comp):
        m = re.search(r'^(.*?)(?:_(\d+_to_\d+))?$', comp)
        base = comp
        suffix = None
        if m:
            base = m.group(1)
            suffix = m.group(2)
        if suffix and '_to_' in suffix:
            a, b = suffix.split('_to_')
            a = a.strip('_')
            return f"{base}: {a} --> {b}"
        return base
    # Annotate with composition once (next to first label's point)
    if labels:
        first_col = f'barrier_mlip_{labels[0]}'
        for _, row in df.iterrows():
            if first_col in df.columns and np.isfinite(row['barrier_vasp']) and np.isfinite(row[first_col]):
                ax.text(row['barrier_vasp']*1.02, row[first_col]*0.98, _fmt(row['composition']), fontsize=7)

    ax.set_xlabel('Barrier VASP (eV)')
    ax.set_ylabel('Barrier MLIP (eV)')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    if labels:
        ax.legend(frameon=False, fontsize=6)

def neb_multi_panel(df, labels):
    """Return Figure with a 1×N grid of NEB profiles (MLIP vs VASP),
    short but wide rows, and titles inside to save vertical space."""
    n = len(df)
    ncols, nrows = 1, n
    # Wider, shorter rows: width ~6.8in, height per row ~1.3in
    fig = plt.figure(figsize=(6.8, max(1.3*nrows, 1.3)))
    gs = gridspec.GridSpec(nrows, ncols, hspace=0.25, wspace=0.2)

    def _format_comp_label(comp):
        # Transform "Base_A_to_B" -> "Base: A --> B"
        m = re.search(r'^(.*?)(?:_(\d+_to_\d+))?$', comp)
        base = comp
        suffix = None
        if m:
            base = m.group(1)
            suffix = m.group(2)
        if suffix and '_to_' in suffix:
            a, b = suffix.split('_to_')
            a = a.strip('_')
            label = f"{base}: {a} --> {b}"
        else:
            label = base
        return label

    for i, (_, row) in enumerate(df.iterrows()):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(row['vasp_neb'], '--', color='k', lw=1.2, label='VASP')
        for j, label in enumerate(labels):
            col = f'mlip_neb_{label}'
            if col in df.columns:
                ax.plot(row[col], '-', lw=1.0, label=label)
        # Inside title at top-left
        ax.text(0.02, 0.92, _format_comp_label(row['composition']),
                transform=ax.transAxes, va='top', ha='left', fontsize=8, fontweight='bold')
        ax.set_xlabel('Image'); ax.set_ylabel('ΔE (eV)')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Only first subplot shows legend to save space
        if i == 0:
            ax.legend(frameon=False, fontsize=6, loc='upper right')
    return fig

# ────────────────────────── Main routine ────────────────────────────
def main(main_data='./simple_data'):
    records = []
    main_data = os.path.abspath(main_data)
    folders = sorted(glob.glob(os.path.join(main_data, '*')))
    print(f"DEBUG: Found {len(folders)} folders to process")

    # Configure multiple MLIP results roots and labels here
    # Example: compare gen_0 and gen_10
    results_roots  = ['simple_results_gen_0', 'simple_results_gen_10']
    results_labels = ['gen0', 'gen10']

    for i, folder in enumerate(folders):
        print(f"DEBUG: Processing folder {i+1}/{len(folders)}: {folder}")
        try:
            result = analyse_composition(folder, results_roots=results_roots, results_labels=results_labels)
            records.append(result)
            print(f"DEBUG: Successfully processed {folder}")
        except Exception as err:
            print(f'Skipped {folder}: {err}')

    print(f"DEBUG: Collected {len(records)} successful records")
    df = pd.DataFrame(records)          # tidy storage  [oai_citation:6‡Pandas](https://pandas.pydata.org/docs/user_guide/dsintro.html?utm_source=chatgpt.com)

    # Build long-form barriers table
    barrier_rows = []
    for _, row in df.iterrows():
        vasp_neb = row.get('vasp_neb', None)
        vasp_neb_json = json.dumps([float(x) for x in vasp_neb]) if vasp_neb is not None else None
        for label in results_labels:
            col_bar = f'barrier_mlip_{label}'
            col_neb = f'mlip_neb_{label}'
            if col_bar in df.columns and col_neb in df.columns:
                mlip_neb = row[col_neb]
                mlip_neb_json = json.dumps([float(x) for x in mlip_neb]) if mlip_neb is not None else None
                barrier_rows.append({
                    'composition' : row['composition'],
                    'label'       : label,
                    'barrier_vasp': row['barrier_vasp'],
                    'barrier_mlip': row[col_bar],
                    'neb_vasp'    : vasp_neb_json,
                    'neb_mlip'    : mlip_neb_json,
                })
    barriers_df = pd.DataFrame(barrier_rows, columns=['composition','label','barrier_vasp','barrier_mlip','neb_vasp','neb_mlip'])
    barriers_df.to_csv('summary_barriers.csv', index=False)

    # Also emit a compact JSON for sharing without raw trajectories
    # Merge gen0/gen10 per composition with arrays for plotting
    per_comp = {}
    for _, r in barriers_df.iterrows():
        key = r['composition']
        if key not in per_comp:
            per_comp[key] = {
                'composition': key,
                'vasp': json.loads(r['neb_vasp']) if isinstance(r['neb_vasp'], str) else r['neb_vasp'],
                'bar_vasp': float(r['barrier_vasp'])
            }
        lab = str(r['label']).strip().lower()
        mlip_neb = json.loads(r['neb_mlip']) if isinstance(r['neb_mlip'], str) else r['neb_mlip']
        if lab == 'gen0':
            per_comp[key]['mlip_gen0'] = mlip_neb
            per_comp[key]['bar_gen0'] = float(r['barrier_mlip'])
        elif lab in ('gen10','gen_10','gen-10'):
            per_comp[key]['mlip_gen10'] = mlip_neb
            per_comp[key]['bar_gen10'] = float(r['barrier_mlip'])

    minimal_records = list(per_comp.values())
    with open('summary_barriers.min.json', 'w') as f:
        json.dump({ 'records': minimal_records }, f)

    if df.empty:
        print("DEBUG: No valid records to plot; skipping figure generation.")
        return

    # NEB panel
    fig1 = neb_multi_panel(df, labels=results_labels)
    fig1.savefig('neb_profiles_all.png', bbox_inches='tight', dpi=450)

    # Parity plot
    fig2, ax2 = plt.subplots(figsize=(3.3, 3.0))
    parity_plot(df, labels=results_labels, ax=ax2)
    fig2.savefig('barrier_parity.png', bbox_inches='tight', dpi=450)

    if not barriers_df.empty:
        print(barriers_df[['composition', 'label', 'barrier_vasp', 'barrier_mlip']])

if __name__ == '__main__':
    main()
