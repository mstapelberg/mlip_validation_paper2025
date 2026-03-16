from ase.io import read, write

data_path = '../../data/good_atoms_objects.xyz'

atoms_list = read(data_path, ':')

new_atoms_list = []
for atoms in atoms_list:
    temp_atoms = atoms.copy()
    forces = atoms.get_forces()
    energy = atoms.get_potential_energy()
    stress = atoms.get_stress()
    temp_atoms.calc = None
    temp_atoms.arrays['REF_force'] = forces
    temp_atoms.info['REF_energy'] = energy
    temp_atoms.info['REF_stress'] = stress
    new_atoms_list.append(temp_atoms)

write('good_atoms_objects_fixed.xyz', new_atoms_list)