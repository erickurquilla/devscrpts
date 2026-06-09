'''
Script created by Erick Urquilla at the University of Tennessee, Knoxville.

This script processes a file named cell_i_j_k.h5, which contains particle data for a specific simulation cell.
Its primary purpose is to rotate the momentum vectors of all particles in the cell according to user-specified or pre-configured rotation parameters.
The script utilizes routines from the 'rotation_functions' module to perform the rotation, ensuring physical quantities are correctly transformed.

The script reads the input cell_i_j_k.h5 file from the current directory, applies the rotation to relevant datasets (such as momentum components and possibly flavors), and generates a new output file reflecting the rotated state.

Typical output: cell_i_j_k_rotated.h5

Requirements:
    - The script and cell_i_j_k.h5 file must both be located in the same directory.
    - Necessary dependencies include numpy, h5py, and the associated local modules.
    - Old and new basis vectors must be specified.

Usage:
    - This script is intended for use within a workflow analyzing or visualizing rotated momentum distributions and related quantities in simulation data.
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

loc_sim_plots_notebooks = (here / '..' / 'loc_sim_plots_notebooks').resolve()
sys.path.append(str(loc_sim_plots_notebooks))

import phys_const as pc
import plot_functions as pf
import functions_angular_crossings as fac
import rotation_functions as rf
import re

########################################################################################################################
# Define the old and new basis vectors

basis_old = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
basis_new = np.array([[1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0], [-1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0], [0.0, 0.0, 1.0]])

########################################################################################################################

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

theta = np.arccos(pz)
phi = np.arctan2(py, px)
phi = np.where(phi < 0, phi + 2 * np.pi, phi) # Adjusting the angle to be between 0 and 2pi.

theta_rotated, phi_rotated = rf.rotate_theta_phi_from_old_to_new_basis(theta, phi, basis_new, basis_old)

px_rotated = np.sin(theta_rotated) * np.cos(phi_rotated)
py_rotated = np.sin(theta_rotated) * np.sin(phi_rotated)
pz_rotated = np.cos(theta_rotated)

px_rotated *= particles_dict_this_cell['pupt']
py_rotated *= particles_dict_this_cell['pupt']
pz_rotated *= particles_dict_this_cell['pupt']

# Get the original file name and indices for saving the rotated file
rotated_filename = f"cell_{cell_index_i}_{cell_index_j}_{cell_index_k}_rotated.h5"

with h5py.File(cell_file, 'r') as src, h5py.File(rotated_filename, 'w') as dst:

    # Copy all groups and datasets except the momentum fields
    for key in src.keys():
        if key not in ['pupx', 'pupy', 'pupz']:
            src.copy(key, dst)

    # Save the rotated momenta in dst
    dst.create_dataset('pupx', data=px_rotated)
    dst.create_dataset('pupy', data=py_rotated)
    dst.create_dataset('pupz', data=pz_rotated)

print(f"Rotated file saved as {rotated_filename}")