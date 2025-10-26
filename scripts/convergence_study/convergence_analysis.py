import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ase.io import read

# Publication quality settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4

# Color scheme
COLORS = ['#2A33C3', '#A35D00', '#0B7285', '#8F2D56', '#6E8B00']

def save_data_to_json(data, filename):
    """Save convergence data to JSON file."""
    # Convert data to list of dicts for JSON serialization
    json_data = []
    for value, energy, stress in data:
        item = {'value': value}
        if energy is not None:
            item['energy'] = energy
        if stress is not None:
            item['stress'] = stress
        json_data.append(item)
    
    with open(filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Data saved to {filename}")

def load_data_from_json(filename):
    """Load convergence data from JSON file."""
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'r') as f:
        json_data = json.load(f)
    
    # Convert back to tuple format
    data = []
    for item in json_data:
        value = item['value']
        energy = item.get('energy')
        stress = item.get('stress')
        data.append((value, energy, stress))
    
    return data

def parse_outcar_for_energy_and_stress(outcar_file):
    """
    Parses the OUTCAR file using ASE to extract the total energy and stress tensor from the first step.
    """
    total_energy = None
    stress_tensor = None
    try:
        # Read all ionic steps from OUTCAR
        atoms_list = read(outcar_file, index=':', format='vasp-out')
        
        # Get the first step
        if len(atoms_list) > 0:
            first_atoms = atoms_list[0]
            
            # Get total energy
            total_energy = first_atoms.get_potential_energy()
            
            # Get stress tensor (ASE returns in eV/Å³, convert to GPa)
            # 1 eV/Å³ = 160.21766208 GPa
            stress_voigt = first_atoms.get_stress(voigt=True)  # Returns [xx, yy, zz, yz, xz, xy]
            
            # Convert Voigt notation to full 3x3 tensor
            stress_tensor = np.array([
                [stress_voigt[0], stress_voigt[5], stress_voigt[4]],
                [stress_voigt[5], stress_voigt[1], stress_voigt[3]],
                [stress_voigt[4], stress_voigt[3], stress_voigt[2]]
            ])
            
            # Convert from eV/Å³ to GPa
            stress_tensor *= 160.21766208

    except Exception as e:
        print(f"Error parsing {outcar_file}: {e}")

    return total_energy, stress_tensor

def extract_kspacing_data(root_dir):
    """
    Extracts energy and stress data from k-spacing convergence study.
    Returns combined results as list of (kspacing, energy, stress) tuples.
    """
    pattern = re.compile(r'kspacing-(\d+_\d+)')
    results = []
    num_atoms = None

    for folder_name in os.listdir(root_dir):
        match = pattern.match(folder_name)
        if match:
            kspacing_value = float(match.group(1).replace('_', '.'))
            outcar_path = os.path.join(root_dir, folder_name, 'OUTCAR')

            if os.path.isfile(outcar_path):
                total_energy, stress_tensor = parse_outcar_for_energy_and_stress(outcar_path)
                
                energy_per_atom = None
                stress_norm = None
                
                if total_energy is not None:
                    # Get number of atoms from first structure for normalization
                    if num_atoms is None:
                        try:
                            atoms = read(outcar_path, index=0, format='vasp-out')
                            num_atoms = len(atoms)
                        except Exception:
                            num_atoms = 1
                    
                    # Store energy per atom
                    energy_per_atom = total_energy / num_atoms
                
                if stress_tensor is not None:
                    # Calculate Frobenius norm of stress tensor
                    stress_norm = np.linalg.norm(stress_tensor, 'fro')
                
                if energy_per_atom is not None or stress_norm is not None:
                    results.append((kspacing_value, energy_per_atom, stress_norm))

    return sorted(results)

def extract_encut_data(root_dir):
    """
    Extracts energy and stress data from ENCUT convergence study.
    Returns combined results as list of (encut, energy, stress) tuples.
    """
    pattern = re.compile(r'encut_(\d+)')
    results = []
    num_atoms = None

    for folder_name in os.listdir(root_dir):
        match = pattern.match(folder_name)
        if match:
            encut_value = int(match.group(1))
            outcar_path = os.path.join(root_dir, folder_name, 'OUTCAR')

            if os.path.isfile(outcar_path):
                total_energy, stress_tensor = parse_outcar_for_energy_and_stress(outcar_path)
                
                energy_per_atom = None
                stress_norm = None
                
                if total_energy is not None:
                    # Get number of atoms from first structure for normalization
                    if num_atoms is None:
                        try:
                            atoms = read(outcar_path, index=0, format='vasp-out')
                            num_atoms = len(atoms)
                        except Exception:
                            num_atoms = 1
                    
                    # Store energy per atom
                    energy_per_atom = total_energy / num_atoms
                
                if stress_tensor is not None:
                    # Calculate Frobenius norm of stress tensor
                    stress_norm = np.linalg.norm(stress_tensor, 'fro')
                
                if energy_per_atom is not None or stress_norm is not None:
                    results.append((encut_value, energy_per_atom, stress_norm))

    return sorted(results)

def create_publication_plot(kspacing_data, encut_data, chosen_kspacing=(0.12, 0.14), chosen_encut=520):
    """
    Creates a publication-quality convergence plot.
    kspacing_data and encut_data are lists of (value, energy, stress) tuples.
    """
    # Split combined data into separate energy and stress lists
    ksp_vals = [x[0] for x in kspacing_data]
    ksp_energies = [x[1] for x in kspacing_data]
    ksp_stresses = [x[2] for x in kspacing_data]
    
    encut_vals = [x[0] for x in encut_data]
    enc_energies = [x[1] for x in encut_data]
    enc_stresses = [x[2] for x in encut_data]
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # K-spacing energy convergence
    ax1 = fig.add_subplot(gs[0, 0])
    if ksp_energies and any(e is not None for e in ksp_energies):
        # Calculate energy relative to most converged value
        valid_energies = [e for e in ksp_energies if e is not None]
        if valid_energies:
            ref_energy = min(valid_energies)  # smallest energy (most converged)
            rel_energies = [(e - ref_energy) * 1000 if e is not None else None for e in ksp_energies]
            
            # Filter out None values for plotting
            plot_data = [(v, e) for v, e in zip(ksp_vals, rel_energies) if e is not None]
            if plot_data:
                plot_vals, plot_energies = zip(*plot_data)
                ax1.plot(plot_vals, plot_energies, 'o-', color=COLORS[0], 
                        linewidth=2, markersize=8, markeredgecolor='white', markeredgewidth=1.5)
                ax1.axvspan(chosen_kspacing[0], chosen_kspacing[1], alpha=0.2, color=COLORS[3], 
                           label=f'Chosen: {chosen_kspacing[0]}-{chosen_kspacing[1]} Å⁻¹')
                ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='1 meV/atom threshold')
                ax1.set_xlabel('K-spacing (Å⁻¹)', fontweight='bold')
                ax1.set_ylabel('ΔE (meV/atom)', fontweight='bold')
                ax1.set_title('(a) K-spacing Energy Convergence', fontweight='bold', loc='left')
                ax1.grid(True, alpha=0.3, linestyle=':')
                ax1.legend(frameon=True, fancybox=True, shadow=True)
    
    # K-spacing stress convergence
    ax2 = fig.add_subplot(gs[0, 1])
    if ksp_stresses and any(s is not None for s in ksp_stresses):
        # Filter out None values for plotting
        plot_data = [(v, s) for v, s in zip(ksp_vals, ksp_stresses) if s is not None]
        if plot_data:
            plot_vals, plot_stresses = zip(*plot_data)
            ax2.plot(plot_vals, plot_stresses, 's-', color=COLORS[1], 
                    linewidth=2, markersize=8, markeredgecolor='white', markeredgewidth=1.5)
            ax2.axvspan(chosen_kspacing[0], chosen_kspacing[1], alpha=0.2, color=COLORS[3],
                       label=f'Chosen: {chosen_kspacing[0]}-{chosen_kspacing[1]} Å⁻¹')
            ax2.set_xlabel('K-spacing (Å⁻¹)', fontweight='bold')
            ax2.set_ylabel('||Stress|| (GPa)', fontweight='bold')
            ax2.set_title('(b) K-spacing Stress Convergence', fontweight='bold', loc='left')
            ax2.grid(True, alpha=0.3, linestyle=':')
            ax2.legend(frameon=True, fancybox=True, shadow=True)
    
    # ENCUT energy convergence
    ax3 = fig.add_subplot(gs[1, 0])
    if enc_energies and any(e is not None for e in enc_energies):
        # Calculate energy relative to most converged value
        valid_energies = [e for e in enc_energies if e is not None]
        if valid_energies:
            ref_energy = max(valid_energies)  # highest energy (most converged for ENCUT)
            rel_energies = [(e - ref_energy) * 1000 if e is not None else None for e in enc_energies]
            
            # Filter out None values for plotting
            plot_data = [(v, e) for v, e in zip(encut_vals, rel_energies) if e is not None]
            if plot_data:
                plot_vals, plot_energies = zip(*plot_data)
                ax3.plot(plot_vals, plot_energies, 'o-', color=COLORS[2], 
                        linewidth=2, markersize=8, markeredgecolor='white', markeredgewidth=1.5)
                ax3.axvline(x=chosen_encut, color=COLORS[3], linestyle='-', linewidth=2.5, 
                           alpha=0.7, label=f'Chosen: {chosen_encut} eV')
                ax3.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='1 meV/atom threshold')
                ax3.set_xlabel('ENCUT (eV)', fontweight='bold')
                ax3.set_ylabel('ΔE (meV/atom)', fontweight='bold')
                ax3.set_title('(c) ENCUT Energy Convergence', fontweight='bold', loc='left')
                ax3.grid(True, alpha=0.3, linestyle=':')
                ax3.legend(frameon=True, fancybox=True, shadow=True)
    
    # ENCUT stress convergence
    ax4 = fig.add_subplot(gs[1, 1])
    if enc_stresses and any(s is not None for s in enc_stresses):
        # Filter out None values for plotting
        plot_data = [(v, s) for v, s in zip(encut_vals, enc_stresses) if s is not None]
        if plot_data:
            plot_vals, plot_stresses = zip(*plot_data)
            ax4.plot(plot_vals, plot_stresses, 's-', color=COLORS[4], 
                    linewidth=2, markersize=8, markeredgecolor='white', markeredgewidth=1.5)
            ax4.axvline(x=chosen_encut, color=COLORS[3], linestyle='-', linewidth=2.5, 
                       alpha=0.7, label=f'Chosen: {chosen_encut} eV')
            ax4.set_xlabel('ENCUT (eV)', fontweight='bold')
            ax4.set_ylabel('||Stress|| (GPa)', fontweight='bold')
            #ax4.set_title('(d) ENCUT Stress Convergence', fontweight='bold', loc='left')
            ax4.grid(True, alpha=0.3, linestyle=':')
            ax4.legend(frameon=True, fancybox=True, shadow=True)
    
    # Add overall title
    fig.suptitle('DFT Convergence Study: K-spacing and ENCUT', 
                fontsize=14, fontweight='bold', y=0.995)
    
    return fig

def main():
    # Paths to data directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kpoint_dir = os.path.join(script_dir, '../../data/convergence_studies/kpoint_test')
    stress_dir = os.path.join(script_dir, '../../data/convergence_studies/stress_test')
    
    # Paths to JSON data files
    kspacing_json = os.path.join(script_dir, 'kspacing_data.json')
    encut_json = os.path.join(script_dir, 'encut_data.json')
    
    # Try to load from JSON files first
    ksp_data = load_data_from_json(kspacing_json)
    encut_data = load_data_from_json(encut_json)
    
    # If JSON files don't exist, extract from OUTCAR files
    if ksp_data is None:
        print("="*60)
        print("JSON data not found. Extracting from OUTCAR files...")
        print("="*60)
        ksp_data = extract_kspacing_data(kpoint_dir)
        
        # Save to JSON for future use
        if ksp_data:
            save_data_to_json(ksp_data, kspacing_json)
    else:
        print("Loading k-spacing data from JSON...")
    
    if encut_data is None:
        encut_data = extract_encut_data(stress_dir)
        
        # Save to JSON for future use
        if encut_data:
            save_data_to_json(encut_data, encut_json)
    else:
        print("Loading ENCUT data from JSON...")
    
    # Print extracted data
    print("\n" + "="*60)
    print("K-SPACING RESULTS:")
    print("="*60)
    if ksp_data:
        print(f"\nFound {len(ksp_data)} data points")
        print("\nEnergy values:")
        for ksp, energy, stress in ksp_data:
            if energy is not None:
                print(f"  K-spacing: {ksp:.2f} Å⁻¹, Energy: {energy:.6f} eV/atom")
            if stress is not None:
                print(f"  K-spacing: {ksp:.2f} Å⁻¹, ||Stress||: {stress:.4f} GPa")
    
    print("\n" + "="*60)
    print("ENCUT RESULTS:")
    print("="*60)
    if encut_data:
        print(f"\nFound {len(encut_data)} data points")
        print("\nEnergy values:")
        for encut, energy, stress in encut_data:
            if energy is not None:
                print(f"  ENCUT: {encut} eV, Energy: {energy:.6f} eV/atom")
            if stress is not None:
                print(f"  ENCUT: {encut} eV, ||Stress||: {stress:.4f} GPa")
    
    # Create plot
    print("\n" + "="*60)
    print("Creating publication-quality plot...")
    print("="*60)
    fig = create_publication_plot(ksp_data, encut_data)
    
    # Save plot
    output_path = os.path.join(script_dir, 'convergence_study_publication.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved to: {output_path}")
    
    # Also save as PDF for publication
    output_pdf = os.path.join(script_dir, 'convergence_study_publication.pdf')
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"PDF saved to: {output_pdf}")
    
    plt.close()

if __name__ == "__main__":
    main()

