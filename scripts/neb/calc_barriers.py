from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
from nequip.ase import NequIPCalculator
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from ase.mep import DyNEB 

import numpy as np
import matplotlib.pyplot as plt
import glob
import os



# read the perfect start from nonstatic at the beginning 
# get the end perfect from static 
# get the start vacancy and end vacancy from nonstatic 

def find_removed_indices(perf_pos, vac_pos):
    # Use broadcasting to compare all positions
    # For each atom in perf_pos, check if it is (almost) present in vac_pos
    mask = np.array([
        not np.any(np.all(np.isclose(atom, vac_pos, atol=1e-5), axis=1))
        for atom in perf_pos
    ])
    return np.where(mask)[0]


main_data_path = './simple_data'
results_dir = './simple_results_gen_0'

# Create the main results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

#package_path = './potentials/allegro/exploit_rmax6.00_lmax2_layers2_mlp384.nequip.zip'
#package_path = './potentials/allegro/gen-5-exploit_rmax6.00_lmax2_layers2_mlp384.nequip.zip'
#package_path = './potentials/allegro/gen10_seed0.nequip.zip'
#package_path = './potentials/allegro/catw_lmax2_nlayers2_mlp512_nlh_epoch128.nequip.zip'
package_path = './potentials/allegro/gen0_seed0.nequip.zip'
calc = NequIPCalculator._from_packaged_model(package_path=package_path, device='cuda')

# get all the folders in the main_data_path
folders = glob.glob(os.path.join(main_data_path, '*'))

# get the first folder
for folder in folders:

    print(f" Running analysis for {folder}")
    perf_start = read(os.path.join(folder, 'perf', 'POSCAR'))
    perf_end = read(os.path.join(folder, 'perf', 'OUTCAR'))

    results_path = os.path.join(results_dir, perf_start.get_chemical_formula())
    
    # Create the results directory if it doesn't exist
    os.makedirs(results_path, exist_ok=True)

    # get the start and end structures from poscar files in that folder
    # the start poscar is POSCAR_Start* and the end poscar is POSCAR_End*
    start_vac = read(glob.glob(os.path.join(folder, 'vac', 'POSCAR_Start*'))[0])
    end_vac = read(glob.glob(os.path.join(folder, 'vac', 'POSCAR_End*'))[0])

    # read the start and end structures from the poscar files
    perf_positions = perf_end.positions 
    start_vac_positions = start_vac.positions
    end_vac_positions = end_vac.positions

    # get the indices of the atoms that were removed in the start and end vac poscars
    start_vac_removed_indices = find_removed_indices(perf_positions, start_vac_positions)[0]
    end_vac_removed_indices = find_removed_indices(perf_positions, end_vac_positions)[0]

    # need to handle the shaninigans of the Cr5Ti5V115_17_to_23 folder
    if folder == './simple_data/Cr5Ti5V115_17_to_23':
        start_vac_removed_indices = 17
        end_vac_removed_indices = 24

    print("Indices of atoms removed in start_vac:", start_vac_removed_indices)
    print("Indices of atoms removed in end_vac:", end_vac_removed_indices)


    perf_end.info['config_type'] = 'perf_end'
    start_vac.info['config_type'] = 'start_vac'
    end_vac.info['config_type'] = 'end_vac'

    vasp_perf_start_end_vac_start_atoms = [perf_start,perf_end, start_vac, end_vac]
    write(os.path.join(results_path, 'vasp_perf_start_end_vac_start_atoms.extxyz'), vasp_perf_start_end_vac_start_atoms, format='extxyz')

    # Now we relax the perfect start structure using the MLIP 
    mlip_perf = perf_start.copy()
    mlip_perf.calc = calc

    fcf = FrechetCellFilter(mlip_perf)

    opt = FIRE(fcf, trajectory='mlip_perf_opt.traj')
    opt.run(steps=300, fmax=0.05)

    print(f"MLIP perf energy: {mlip_perf.get_potential_energy()}")
    print(f"VASP perf energy: {perf_end.get_potential_energy()}")


    # make the new atoms objects from the MLIP relaxed perfect start 
    mlip_start_vac = mlip_perf.copy() 
    mlip_end_vac = mlip_perf.copy()
    print(f"Position of the end index {end_vac_removed_indices} in the end structure: {perf_end.positions[end_vac_removed_indices]}")
    print(f"Position of the start index {start_vac_removed_indices} in the start structure: {mlip_start_vac.positions[start_vac_removed_indices]}")

    print(f"Moving atom in end position {perf_end.positions[end_vac_removed_indices]} to start position {mlip_start_vac.positions[start_vac_removed_indices]}")
    mlip_end_vac.positions[end_vac_removed_indices] = mlip_start_vac.positions[start_vac_removed_indices]

    print(f"Position of the end index {end_vac_removed_indices} in the perfect structure: {perf_end.positions[end_vac_removed_indices]}")
    print(f"Position of the end index {end_vac_removed_indices} in the end structure: {mlip_end_vac.positions[end_vac_removed_indices]}")

    # remove the atoms in the start index from both vac structures  

    mlip_start_vac.pop(start_vac_removed_indices)
    mlip_end_vac.pop(start_vac_removed_indices)

    perf_end.info['config_type'] = 'perf_end'
    mlip_start_vac.info['config_type'] = 'mlip_start_vac'
    mlip_end_vac.info['config_type'] = 'mlip_end_vac'

    mlip_selected_atoms = [perf_end, mlip_start_vac, mlip_end_vac]

    write(os.path.join(results_path, 'mlip_perf_end_vac_start_vac_end.extxyz'), mlip_selected_atoms, format='extxyz')


    start_vac_relax = mlip_start_vac.copy()
    end_vac_relax = mlip_end_vac.copy()

    start_vac_relax.calc = calc
    end_vac_relax.calc = calc

    start_opt = FIRE(start_vac_relax, trajectory=os.path.join(results_path, 'mlip_start_vac_opt.traj'))
    end_opt = FIRE(end_vac_relax, trajectory=os.path.join(results_path, 'mlip_end_vac_opt.traj'))

    start_opt.run(steps=250, fmax=0.01)
    end_opt.run(steps=250, fmax=0.01)

    from ase.io import read, write

    start_traj = read(os.path.join(results_path, 'mlip_start_vac_opt.traj'), index=':')
    end_traj = read(os.path.join(results_path, 'mlip_end_vac_opt.traj'), index=':')

    write(os.path.join(results_path, 'mlip_start_vac_relaxed_opt.extxyz'), start_traj, format='extxyz')
    write(os.path.join(results_path, 'mlip_end_vac_relaxed_opt.extxyz'), end_traj, format='extxyz')

    start_energies = []
    for traj in start_traj:
        start_energies.append(traj.get_potential_energy())

    end_energies = []
    for traj in end_traj:
        end_energies.append(traj.get_potential_energy())

    #  get the combined energies of the vasp structures

    vasp_start_outcar = glob.glob(os.path.join(folder, 'vac', 'OUTCAR_Start*'))[0]
    vasp_end_outcar = glob.glob(os.path.join(folder, 'vac', 'OUTCAR_End*'))[0]  
    vasp_start_vac = read(vasp_start_outcar, index=':')
    vasp_end_vac = read(vasp_end_outcar, index=':')

    start_vac_energies = [] 
    end_vac_energies = []

    for i, atoms in enumerate(vasp_start_vac):
        start_vac_energies.append(atoms.get_potential_energy())

    for i, atoms in enumerate(vasp_end_vac):
        end_vac_energies.append(atoms.get_potential_energy())

    plt.figure(figsize=(10, 5))
    plt.plot(start_vac_energies, label='vasp start')
    plt.plot(end_vac_energies, label='vasp end')
    plt.plot(start_energies, label='mlip start')
    plt.plot(end_energies, label='mlip end')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'mlip_vac_vasp_vac_energies.png'))

    # Run the NEB 

    images = []
    images.append(start_vac_relax.copy())

    for i in range(5):
        images.append(start_vac_relax.copy())

    images.append(end_vac_relax.copy())

    print(len(images))

    for image in images:
        image.calc = NequIPCalculator._from_packaged_model(package_path=package_path, device='cuda')

    neb = DyNEB(images)
    neb.interpolate(mic=True)

    dyn = FIRE(neb, trajectory=os.path.join(results_path, 'mlip_neb_relaxed.traj'))
    dyn.run(steps=500, fmax=0.05)

    # Ensure final NEB images carry single-point energies in the saved trajectory
    final_images = [img.copy() for img in images]
    for img in final_images:
        img.calc = NequIPCalculator._from_packaged_model(package_path=package_path, device='cuda')
    for idx, img in enumerate(final_images):
        E = img.get_potential_energy()
        F = img.get_forces()
        try:
            S = img.get_stress()
        except Exception:
            S = None
        if S is not None:
            SinglePointCalculator(img, energy=E, forces=F, stress=S)
        else:
            SinglePointCalculator(img, energy=E, forces=F)
        img.info['config_type'] = f'neb_image_{idx}'
    # Overwrite with enriched frames so downstream analysis can read energies
    write(os.path.join(results_path, 'mlip_neb_relaxed.traj'), final_images)

    neb_energies = []
    for image in images:
        neb_energies.append(image.get_potential_energy())


    # get the vasp energies from the finished neb calc 
    neb_data_path = os.path.join(folder, 'neb')
    # get all the xyz files in the neb_data_path
    xyz_files = glob.glob(os.path.join(neb_data_path, '*.xyz'))

    # Sort the xyz files numerically based on their filename (e.g., 00.xyz, 01.xyz, ...)
    def extract_number(x):
        base = os.path.basename(x)
        num = os.path.splitext(base)[0]
        try:
            return int(num)
        except ValueError:
            return num  # fallback, in case

    xyz_files_sorted = sorted(xyz_files, key=extract_number)

    vasp_energies = []
    # read in the xyz files in sorted order
    for xyz_file in xyz_files_sorted:
        atoms = read(xyz_file)
        vasp_energies.append(atoms.get_potential_energy())

    vasp_energies = np.array([e - vasp_energies[0] for e in vasp_energies])
    neb_energies = np.array([e - neb_energies[0] for e in neb_energies])

    plt.figure(figsize=(10, 5))
    plt.title(f'VASP Barrier = {np.round(np.max(vasp_energies), 3)} eV vs MLIP Barrier = {np.round(np.max(neb_energies), 3)} eV')
    plt.plot(neb_energies, label='MLIP')
    plt.plot(vasp_energies, label='VASP')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'mlip_vasp_neb_energies.png'))




