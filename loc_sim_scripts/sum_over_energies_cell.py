'''
Script created by Erick Urquilla at the University of Tennessee, Knoxville.
This script reads a file called cell_i_j_k.h5 and processes the particle data within it.
It sums over all the particles in the same angular direction and returns a file named cell_i_j_k_Esummed.h5.
The cell_i_j_k.h5 file must be located in the same directory where this script is executed.
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.ticker import AutoLocator, AutoMinorLocator, LogLocator
import glob
from scipy.interpolate import griddata
from pathlib import Path
import h5py
import sys
from pathlib import Path

# Where am I running?
try:
    # Normal script
    here = Path(__file__).resolve().parent
except NameError:
    # Notebook / REPL
    here = Path.cwd()

phys_const_path = (here / '..' / 'phys_const').resolve()
sys.path.append(str(phys_const_path))

nsm_plots_path = (here / '..' / 'nsm_plots').resolve()
sys.path.append(str(nsm_plots_path))

nsm_plots_postproc = (here / '..' / 'nsm_instabilities').resolve()
sys.path.append(str(nsm_plots_postproc))

import phys_const as pc
import plot_functions as pf
import functions_angular_crossings as fac
import re


# Search for files matching the pattern cell_*_*_*.h5 in the current directory
cell_files = list(here.glob('cell_*_*_*.h5'))

if not cell_files:
    raise FileNotFoundError("No file matching 'cell_*_*_*.h5' found in the current directory.")

# Use the first matching file
cell_file = cell_files[0]
match = re.match(r'cell_(\d+)_(\d+)_(\d+)\.h5', cell_file.name)
if not match:
    raise ValueError(f"Filename {cell_file.name} does not match the expected pattern.")

cell_index_i = int(match.group(1))
cell_index_j = int(match.group(2))
cell_index_k = int(match.group(3))

print(f"Found cell file: {cell_file.name} with indices i={cell_index_i}, j={cell_index_j}, k={cell_index_k}")

# Dictionary keys
# <KeysViewHDF5 ['N00_Re', 'N00_Rebar', 'N01_Im', 'N01_Imbar', 'N01_Re', 'N01_Rebar', 'N02_Im', 'N02_Imbar', 'N02_Re', 'N02_Rebar', 'N11_Re', 'N11_Rebar', 'N12_Im', 'N12_Imbar', 'N12_Re', 'N12_Rebar', 'N22_Re', 'N22_Rebar', 'TrHN', 'Vphase', 'pos_x', 'pos_y', 'pos_z', 'pupt', 'pupx', 'pupy', 'pupz', 'time', 'x', 'y', 'z']>

particles_dict_this_cell = fac.load_particle_data(cell_index_i, cell_index_j, cell_index_k, './')

px = particles_dict_this_cell['pupx']/particles_dict_this_cell['pupt']
py = particles_dict_this_cell['pupy']/particles_dict_this_cell['pupt']
pz = particles_dict_this_cell['pupz']/particles_dict_this_cell['pupt']
print(f'pz.shape = {pz.shape}')

momentum = np.stack((px, py, pz), axis=-1)
tolerance = 1.0e-4
unique_momentum = fac.get_unique_momentum(momentum, tolerance)
print(f'unique_momentum.shape = {unique_momentum.shape}')

Fluxee_all = particles_dict_this_cell['N00_Re'][:, np.newaxis] * momentum
ee_unique_fluxes, ee_unique_fluxes_mag = fac.compute_unique_fluxes(momentum, Fluxee_all, unique_momentum)

p_unit_x = np.real(ee_unique_fluxes[:,0]) / ee_unique_fluxes_mag
p_unit_y = np.real(ee_unique_fluxes[:,1]) / ee_unique_fluxes_mag
p_unit_z = np.real(ee_unique_fluxes[:,2]) / ee_unique_fluxes_mag

def sum_over_energies(quantity):
    quantity_vector = quantity[:, np.newaxis] * momentum
    quantity_unique_fluxes, quantity_unique_fluxes_mag = fac.compute_unique_fluxes(momentum, quantity_vector, unique_momentum)
    return quantity_unique_fluxes_mag

uu_unique_fluxes_mag = sum_over_energies(particles_dict_this_cell['N11_Re'])
tt_unique_fluxes_mag = sum_over_energies(particles_dict_this_cell['N22_Re'])
eebar_unique_fluxes_mag = sum_over_energies(particles_dict_this_cell['N00_Rebar'])
uubar_unique_fluxes_mag = sum_over_energies(particles_dict_this_cell['N11_Rebar'])
ttbar_unique_fluxes_mag = sum_over_energies(particles_dict_this_cell['N22_Rebar'])

average_energy_ee = np.sum(particles_dict_this_cell['pupt']*particles_dict_this_cell['N00_Re']) / np.sum(particles_dict_this_cell['N00_Re'])
average_energy_uu = np.sum(particles_dict_this_cell['pupt']*particles_dict_this_cell['N11_Re']) / np.sum(particles_dict_this_cell['N11_Re'])
average_energy_tt = np.sum(particles_dict_this_cell['pupt']*particles_dict_this_cell['N22_Re']) / np.sum(particles_dict_this_cell['N22_Re'])
average_energy_eebar = np.sum(particles_dict_this_cell['pupt']*particles_dict_this_cell['N00_Rebar']) / np.sum(particles_dict_this_cell['N00_Rebar'])
average_energy_uubar = np.sum(particles_dict_this_cell['pupt']*particles_dict_this_cell['N11_Rebar']) / np.sum(particles_dict_this_cell['N11_Rebar'])
average_energy_ttbar = np.sum(particles_dict_this_cell['pupt']*particles_dict_this_cell['N22_Rebar']) / np.sum(particles_dict_this_cell['N22_Rebar'])

print('average_energy_ee = ', average_energy_ee)
print('average_energy_uu = ', average_energy_uu)
print('average_energy_tt = ', average_energy_tt)
print('average_energy_eebar = ', average_energy_eebar)
print('average_energy_uubar = ', average_energy_uubar)
print('average_energy_ttbar = ', average_energy_ttbar)

energy = ( average_energy_ee + average_energy_eebar + average_energy_uu ) / 3.0
print('average_energy = ', energy)

pupt = p_unit_x * 0.0 + energy
pupx = p_unit_x * energy
pupy = p_unit_y * energy
pupz = p_unit_z * energy

total_Vphase = np.sum(particles_dict_this_cell['Vphase'])
Vphase_single_particle = total_Vphase / len(ee_unique_fluxes_mag)
print('Vphase_single_particle = ', Vphase_single_particle)
Vphase = np.full_like(ee_unique_fluxes_mag, Vphase_single_particle)

output_filename = f'cell_{cell_index_i}_{cell_index_j}_{cell_index_k}_Esummed.h5'
with h5py.File(output_filename, 'w') as f:
 
    f.create_dataset('N00_Re', data=ee_unique_fluxes_mag)
    f.create_dataset('N11_Re', data=uu_unique_fluxes_mag)
    f.create_dataset('N22_Re', data=tt_unique_fluxes_mag)
    f.create_dataset('N00_Rebar', data=eebar_unique_fluxes_mag)
    f.create_dataset('N11_Rebar', data=uubar_unique_fluxes_mag)
    f.create_dataset('N22_Rebar', data=ttbar_unique_fluxes_mag)

    f.create_dataset('N01_Re', data=np.zeros_like(ee_unique_fluxes_mag))
    f.create_dataset('N01_Im', data=np.zeros_like(ee_unique_fluxes_mag))
    f.create_dataset('N02_Re', data=np.zeros_like(ee_unique_fluxes_mag))
    f.create_dataset('N02_Im', data=np.zeros_like(ee_unique_fluxes_mag))
    f.create_dataset('N12_Re', data=np.zeros_like(ee_unique_fluxes_mag))
    f.create_dataset('N12_Im', data=np.zeros_like(ee_unique_fluxes_mag))
    f.create_dataset('N01_Rebar', data=np.zeros_like(ee_unique_fluxes_mag))
    f.create_dataset('N01_Imbar', data=np.zeros_like(ee_unique_fluxes_mag))
    f.create_dataset('N02_Rebar', data=np.zeros_like(ee_unique_fluxes_mag))
    f.create_dataset('N02_Imbar', data=np.zeros_like(ee_unique_fluxes_mag))
    f.create_dataset('N12_Rebar', data=np.zeros_like(ee_unique_fluxes_mag))
    f.create_dataset('N12_Imbar', data=np.zeros_like(ee_unique_fluxes_mag))

    f.create_dataset('time', data=np.zeros_like(ee_unique_fluxes_mag))
    f.create_dataset('TrHN', data=np.zeros_like(ee_unique_fluxes_mag))
    f.create_dataset('Vphase', data=Vphase)
    f.create_dataset('pos_x', data=np.zeros_like(ee_unique_fluxes_mag))
    f.create_dataset('pos_y', data=np.zeros_like(ee_unique_fluxes_mag))
    f.create_dataset('pos_z', data=np.zeros_like(ee_unique_fluxes_mag))
    f.create_dataset('x', data=np.zeros_like(ee_unique_fluxes_mag))
    f.create_dataset('y', data=np.zeros_like(ee_unique_fluxes_mag))
    f.create_dataset('z', data=np.zeros_like(ee_unique_fluxes_mag))

    f.create_dataset('pupt', data=pupt)
    f.create_dataset('pupx', data=pupx)
    f.create_dataset('pupy', data=pupy)
    f.create_dataset('pupz', data=pupz)