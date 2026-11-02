'''
Author: Erick Urquilla, Department of Physics and Astronomy, University of Tennessee, Knoxville.

Description:
This script computes and generates HDF5 files containing neutrino-related quantities for each cell center in the EMU grid. 
It reads density (rho), electron fraction (Ye), temperature (T), and other grid parameters from the `rho_Ye_T.hdf5` file. 
The script utilizes the NuLib and SFHo EOS tables to calculate weak interaction quantities for neutrinos.

Input Files:
- `rho_Ye_T.hdf5`: Contains grid quantities such as rho, Ye, and T.
- NuLib table: Path specified in the script (`NuLib_SFHo.h5`).
- EOS table: Path specified in the script (`LS220.h5`).

Computed Quantities:
1. Chemical potentials:
    - `mu_nu_e_MeV`: Electron neutrino chemical potential (MeV).
    - `mu_nubar_e_MeV`: Electron anti-neutrino chemical potential (MeV).

2. Equilibrium number densities:
    - `n_nu_e_eq_ccm`: Neutrino equilibrium number density (1/ccm).
    - `n_nubar_e_eq_ccm`: Anti-neutrino equilibrium number density (1/ccm).

3. Absorption opacities:
    - `nu_e_absorption_opacity_cm`: Electron neutrino absorption opacity (1/cm).
    - `nubar_e_absorption_opacity_cm`: Electron anti-neutrino absorption opacity (1/cm).
    - `nu_x_absorption_opacity_cm`: Heavy-lepton neutrino absorption opacity (1/cm).

4. Scattering opacities:
    - `nu_e_scattering_opacity_cm`: Electron neutrino scattering opacity (1/cm).
    - `nubar_e_scattering_opacity_cm`: Electron anti-neutrino scattering opacity (1/cm).
    - `nu_x_scattering_opacity_cm`: Heavy-lepton neutrino scattering opacity (1/cm).

5. Optical depths (1 km scale):
    - `nu_e_absorption_optical_depth_1km`: Electron neutrino absorption optical depth for 1km travel distance.
    - `nubar_e_absorption_optical_depth_1km`: Electron anti-neutrino absorption optical depth for 1km travel distance.
    - `nu_x_absorption_optical_depth_1km`: Heavy-lepton neutrino absorption optical depth for 1km travel distance.
    - `nu_e_scattering_optical_depth_1km`: Electron neutrino scattering optical depth for 1km travel distance.
    - `nubar_e_scattering_optical_depth_1km`: Electron anti-neutrino scattering optical depth for 1km travel distance.
    - `nu_x_scattering_optical_depth_1km`: Heavy-lepton neutrino scattering optical depth for 1km travel distance.

Output:
The computed quantities are saved as HDF5 files, with each file containing the grid's cell center coordinates and the corresponding dataset.

Usage:
- Update the paths for `rho_Ye_T.hdf5`, NuLib, and EOS tables in the script.
- Run the script to generate the output HDF5 files.
'''

# --------------------------------------------------------------------------------

matter_file_path = 'rho_Ye_T.hdf5'
nu_lib_table_dir = "/pscratch/sd/b/bigqke/bigqke_scratch/tables/NuLib_SFHo.h5"
eos_table_dir = "/pscratch/sd/b/bigqke/bigqke_scratch/tables/LS220.h5"

# --------------------------------------------------------------------------------

import numpy as np
import h5py
from multiprocessing import Pool
import time

class CGSUnitsConst:
    eV = 1.60218e-12  # erg

class PhysConst:
    c = 2.99792458e10  # cm/s
    c2 = c * c
    c4 = c2 * c2
    hbar = 1.05457266e-27  # erg s
    hbarc = hbar * c  # erg cm
    hc = hbarc*2*np.pi
    GF = (1.1663787e-5 / (1e9 * 1e9 * CGSUnitsConst.eV * CGSUnitsConst.eV))
    Mp = 1.6726219e-24  # g
    sin2thetaW = 0.23122
    kB = 1.380658e-16  # erg/K
    G = 6.67430e-8 # cm3 g−1 s-2
    Msun = 1.9891e33 # g

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# The following functions are used to interpolate the data in the NuLib tables.
# These function were not written by me, but are part of the NuLib repository in:
# https://github.com/evanoconnor/NuLib/blob/master/src/scripts/interpolate_table.py
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

def interp3d(x,y,z,xt,yt,zt,data):
    shape = np.shape(data)
    nx = shape[0] # Ye
    ny = shape[1] # T
    nz = shape[2] # rho

    #!------  determine spacing parameters of (equidistant!!!) table

    dx    = (xt[nx-1] - xt[0]) / (nx-1)
    dy    = (yt[ny-1] - yt[0]) / (ny-1)
    dz    = (zt[nz-1] - zt[0]) / (nz-1)

    dxi   = 1.0 / dx
    dyi   = 1.0 / dy
    dzi   = 1.0 / dz

    dxyi  = dxi * dyi
    dxzi  = dxi * dzi
    dyzi  = dyi * dzi

    dxyzi = dxi * dyi * dzi

    #------- determine location in (equidistant!!!) table 

    ix = 1 + int( (x - xt[0] - 1e-10) * dxi )
    iy = 1 + int( (y - yt[0] - 1e-10) * dyi )
    iz = 1 + int( (z - zt[0] - 1e-10) * dzi )

    ix = max( 1, min( ix, nx ) )
    iy = max( 1, min( iy, ny ) )
    iz = max( 1, min( iz, nz ) )

    #------- set-up auxiliary arrays for Lagrange interpolation

    delx = xt[ix] - x
    dely = yt[iy] - y
    delz = zt[iz] - z

    corners = np.zeros(8)
    corners[0] = data[ix  , iy  , iz  ]
    corners[1] = data[ix-1, iy  , iz  ]
    corners[2] = data[ix  , iy-1, iz  ]
    corners[3] = data[ix  , iy  , iz-1]
    corners[4] = data[ix-1, iy-1, iz  ]
    corners[5] = data[ix-1, iy  , iz-1]
    corners[6] = data[ix  , iy-1, iz-1]
    corners[7] = data[ix-1, iy-1, iz-1]
    
    # coefficients
    
    a1 = corners[0]
    a2 = dxi   * ( corners[1] - corners[0] )       
    a3 = dyi   * ( corners[2] - corners[0] )       
    a4 = dzi   * ( corners[3] - corners[0] )       
    a5 = dxyi  * ( corners[4] - corners[1] - corners[2] + corners[0] )
    a6 = dxzi  * ( corners[5] - corners[1] - corners[3] + corners[0] )
    a7 = dyzi  * ( corners[6] - corners[2] - corners[3] + corners[0] )
    a8 = dxyzi * ( corners[7] - corners[0] + corners[1] + corners[2] +
                   corners[3] - corners[4] - corners[5] - corners[6] )

    return a1 +  a2 * delx \
        +  a3 * dely \
        +  a4 * delz \
        +  a5 * delx * dely \
        +  a6 * delx * delz \
        +  a7 * dely * delz \
        +  a8 * delx * dely * delz     


def interp2d(x, y, xt, yt, data):
    shape = np.shape(data)
    nx = shape[0] # eta
    ny = shape[1] # T

    #------  determine spacing parameters of (equidistant!!!) table

    dx    = (xt[nx-1] - xt[0]) / (nx-1)
    dy    = (yt[ny-1] - yt[0]) / (ny-1)
    
    dxi   = 1.0 / dx
    dyi   = 1.0 / dy
    
    dxyi  = dxi * dyi

    #------- determine location in (equidistant!!!) table 

    ix = 1 + int( (x - xt[0] - 1e-10) * dxi )
    iy = 1 + int( (y - yt[0] - 1e-10) * dyi )
    
    ix = max( 1, min( ix, nx ) )
    iy = max( 1, min( iy, ny ) )

    #------- set-up auxiliary arrays for Lagrange interpolation
    
    delx = xt[ix] - x
    dely = yt[iy] - y

    corners = np.zeros(4)
    corners[0] = data[ix  , iy  ]
    corners[1] = data[ix-1, iy  ]
    corners[2] = data[ix  , iy-1]
    corners[3] = data[ix-1, iy-1]

    #------ set up coefficients of the interpolation polynomial and

    a1 = corners[0]
    a2 = dxi   * ( corners[1] - corners[0] )       
    a3 = dyi   * ( corners[2] - corners[0] )       
    a4 = dxyi  * ( corners[3] - corners[1] - corners[2] + corners[0] )
    
    return a1 +  a2*delx + a3*dely + a4*delx*dely

    

def interpolate_eas(ig, s, rho, T, Ye, table, datasetname):
#!---------------------------------------------------------------------
#!
#!     purpose: interpolation of a function of three variables in an
#!              equidistant(!!!) table.
#!
#!     method:  8-point Lagrange linear interpolation formula          
#!
#!     ig       group number
#!     s        species number
#!     rho      density (g/ccm)
#!     T        temperature (MeV)
#!     Ye       electron fraction
#!
#!     table    h5py.File()
#!     datasetname  {"absorption_opacity", "emissivities", "scattering_opacity", "scattering_delta"}
#!---------------------------------------------------------------------
    data = table[datasetname]

    zt = np.log10(table["rho_points"])
    yt = np.log10(table["temp_points"])
    xt = np.array(table["ye_points"])

    if(rho<table["rho_points"][0] or rho>table["rho_points"][len(zt)-1] \
       or T<table["temp_points"][0] or T>table["temp_points"][len(yt)-1] \
       or Ye<table["ye_points"][0] or Ye>table["ye_points"][len(xt)-1]):
        print("Warning: outside table range")
        return 0

    x = Ye
    y = np.log10(T)
    z = np.log10(rho)

    return interp3d(Ye, np.log10(T), np.log10(rho), xt, yt, zt, data[ig,s,:,:,:])


def interpolate_kernel(ig_in, s, ig_out, eta, T, table, datasetname):
#!---------------------------------------------------------------------
#!
#!     purpose: interpolation of a function of three variables in an
#!              equidistant(!!!) table.
#!
#!     method:  8-point Lagrange linear interpolation formula          
#!
#!     ig_in    group number
#!     s        species number
#!     ig_out   group number
#!     eta      mue / T (dimensionless)
#!     T        temperature (MeV)
#!
#!     table    h5py.File()
#!     datasetname  {"inelastic_phi0", "inelastic_phi1"}
#!---------------------------------------------------------------------
    data = table[datasetname][ig_in,s,ig_out,:,:]

    xt = np.log10(table["eta_Ipoints"])
    yt = np.log10(table["temp_Ipoints"])

    if(eta<table["eta_Ipoints"][0] or eta>table["eta_Ipoints"][len(xt)-1] \
       or T<table["temp_Ipoints"][0] or T>table["temp_Ipoints"][len(yt)-1]):
        print("Warning: outside table range")
        return 0

    x = np.log10(eta)
    y = np.log10(T)

    return interp2d(x, y, xt, yt, data)

def interpolate_eos(rho, T, Ye, table, datasetname):
    data = table[datasetname]

    zt = table["logrho"]
    yt = table["logtemp"]
    xt = table["ye"]

    z = np.log10(rho)
    y = np.log10(T)
    x = Ye

    return interp3d(x, y, z, xt, yt, zt, data)

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

rho_ye_T_h5py = h5py.File(matter_file_path, 'r')
nulib_h5py = h5py.File(nu_lib_table_dir, 'r')
eos_h5py = h5py.File(eos_table_dir, 'r')

# --------------------------------------------------------------------------------

# number of cells
Nx = np.array(rho_ye_T_h5py['/ncellsx'])
Ny = np.array(rho_ye_T_h5py['/ncellsy'])
Nz = np.array(rho_ye_T_h5py['/ncellsz'])
print(f'Nx = {Nx}, Ny = {Ny}, Nz = {Nz}')

# cell size
dx = ( np.array(rho_ye_T_h5py['/xmax']) - np.array(rho_ye_T_h5py['/xmin']) ) / np.array(rho_ye_T_h5py['/ncellsx']) # cm
dy = ( np.array(rho_ye_T_h5py['/ymax']) - np.array(rho_ye_T_h5py['/ymin']) ) / np.array(rho_ye_T_h5py['/ncellsy']) # cm
dz = ( np.array(rho_ye_T_h5py['/zmax']) - np.array(rho_ye_T_h5py['/zmin']) ) / np.array(rho_ye_T_h5py['/ncellsz']) # cm
print(f'dx = {dx} cm, dy = {dy} cm, dz = {dz} cm')

# cell faces
x = np.linspace(0, dx * Nx, Nx + 1) # cm
y = np.linspace(0, dy * Ny, Ny + 1) # cm
z = np.linspace(0, dz * Nz, Nz + 1) # cm

# cell faces mesh
X, Y, Z = np.meshgrid(x, y, z, indexing='ij') # cm

# cell centers
xc = np.linspace(dx / 2, dx * (Nx - 0.5), Nx) # cm
yc = np.linspace(dy / 2, dy * (Ny - 0.5), Ny) # cm
zc = np.linspace(dz / 2, dz * (Nz - 0.5), Nz) # cm

# cell centers mesh
Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing='ij') # cm

# --------------------------------------------------------------------------------

temperature_MeV = rho_ye_T_h5py['/T_Mev'    ] # Mev
ye              = rho_ye_T_h5py['/Ye'       ] # no units
rho_gccm        = rho_ye_T_h5py['/rho_g|ccm'] # g/ccm

# --------------------------------------------------------------------------------

E_bins_MeV        = np.array(nulib_h5py['/neutrino_energies']) # MeV
E_bins_MeV_bottom = np.array(nulib_h5py['/bin_bottom'       ]) # MeV
E_bins_MeV_top    = np.array(nulib_h5py['/bin_top'          ]) # MeV

delta_E_cubic_bins_MeV_cubic = E_bins_MeV_top**3 - E_bins_MeV_bottom**3 # MeV^3

# --------------------------------------------------------------------------------
# Define vacuum numpy arrays to store data

mu_e_MeV    = np.zeros_like(temperature_MeV) # MeV
mu_nu_e_MeV    = np.zeros_like(temperature_MeV) # MeV
mu_nubar_e_MeV = np.zeros_like(temperature_MeV) # MeV

n_nu_e_eq_ccm                     = np.zeros((E_bins_MeV.shape[0],) + temperature_MeV.shape) # 1/ccm
n_nubar_e_eq_ccm                  = np.zeros((E_bins_MeV.shape[0],) + temperature_MeV.shape) # 1/ccm
n_nu_x_eq_ccm                     = np.zeros((E_bins_MeV.shape[0],) + temperature_MeV.shape) # 1/ccm

nu_e_absorption_opacity_cm    = np.zeros((E_bins_MeV.shape[0],) + temperature_MeV.shape) # 1/cm
nubar_e_absorption_opacity_cm = np.zeros((E_bins_MeV.shape[0],) + temperature_MeV.shape) # 1/cm
nu_x_absorption_opacity_cm    = np.zeros((E_bins_MeV.shape[0],) + temperature_MeV.shape) # 1/cm

nu_e_scattering_opacity_cm    = np.zeros((E_bins_MeV.shape[0],) + temperature_MeV.shape) # 1/cm
nubar_e_scattering_opacity_cm = np.zeros((E_bins_MeV.shape[0],) + temperature_MeV.shape) # 1/cm
nu_x_scattering_opacity_cm    = np.zeros((E_bins_MeV.shape[0],) + temperature_MeV.shape) # 1/cm

# --------------------------------------------------------------------------------

def compute_weak_quantities(rho_this_gccm, ye_this, T_this_MeV, E_bins_this_MeV, delta_E_cubic_bins_this_MeV_cubic):

    mu_e   = interpolate_eos(rho_this_gccm, T_this_MeV, ye_this, eos_h5py, 'mu_e') # MeV
    mu_hat = interpolate_eos(rho_this_gccm, T_this_MeV, ye_this, eos_h5py, 'muhat') # MeV

    mu_nu_e_MeV_this    =           mu_e - mu_hat   # MeV
    mu_nubar_e_MeV_this = - 1.0 * ( mu_e - mu_hat ) # MeV

    n_nu_e_eq_this_ccm    = np.zeros_like(E_bins_this_MeV) # 1/ccm
    n_nubar_e_eq_this_ccm = np.zeros_like(E_bins_this_MeV) # 1/ccm
    n_nu_x_eq_this_ccm    = np.zeros_like(E_bins_this_MeV) # 1/ccm

    nu_e_absorption_opacity_this_cm = np.zeros_like(E_bins_this_MeV) # 1/cm
    nubar_e_absorption_opacity_this_cm = np.zeros_like(E_bins_this_MeV) # 1/cm
    nu_x_absorption_opacity_this_cm = np.zeros_like(E_bins_this_MeV) # 1/cm

    nu_e_scattering_opacity_this_cm = np.zeros_like(E_bins_this_MeV) # 1/cm
    nubar_e_scattering_opacity_this_cm = np.zeros_like(E_bins_this_MeV) # 1/cm
    nu_x_scattering_opacity_this_cm = np.zeros_like(E_bins_this_MeV) # 1/cm
    
    for E_index, E_this in enumerate(E_bins_this_MeV): # MeV

        E_this = E_bins_this_MeV[E_index] # MeV
        delta_E_cubic_MeV_cubic_this = delta_E_cubic_bins_this_MeV_cubic[E_index] # MeV^3

        f_nu_e_eq_this    = 1.0 / ( 1.0 + np.exp( ( E_this - mu_nu_e_MeV_this    ) / T_this_MeV ) )
        f_nubar_e_eq_this = 1.0 / ( 1.0 + np.exp( ( E_this - mu_nubar_e_MeV_this ) / T_this_MeV ) )
        f_nu_x_eq_this    = 1.0 / ( 1.0 + np.exp( ( E_this - 0.0                 ) / T_this_MeV ) )

        n_nu_e_eq_this_ccm   [E_index] = ( 1.0 / PhysConst.hc )**3 * 4.0 * np.pi *  ( ( delta_E_cubic_MeV_cubic_this * ( 1e6 * CGSUnitsConst.eV )**3 ) / 3.0 ) * f_nu_e_eq_this
        n_nubar_e_eq_this_ccm[E_index] = ( 1.0 / PhysConst.hc )**3 * 4.0 * np.pi *  ( ( delta_E_cubic_MeV_cubic_this * ( 1e6 * CGSUnitsConst.eV )**3 ) / 3.0 ) * f_nubar_e_eq_this
        n_nu_x_eq_this_ccm   [E_index] = ( 1.0 / PhysConst.hc )**3 * 4.0 * np.pi *  ( ( delta_E_cubic_MeV_cubic_this * ( 1e6 * CGSUnitsConst.eV )**3 ) / 3.0 ) * f_nu_x_eq_this

        nu_e_absorption_opacity_this_cm   [E_index] = interpolate_eas(E_index, 0, rho_this_gccm, T_this_MeV, ye_this, nulib_h5py, 'absorption_opacity')
        nubar_e_absorption_opacity_this_cm[E_index] = interpolate_eas(E_index, 1, rho_this_gccm, T_this_MeV, ye_this, nulib_h5py, 'absorption_opacity')
        nu_x_absorption_opacity_this_cm   [E_index] = interpolate_eas(E_index, 2, rho_this_gccm, T_this_MeV, ye_this, nulib_h5py, 'absorption_opacity')

        nu_e_scattering_opacity_this_cm   [E_index] = interpolate_eas(E_index, 0, rho_this_gccm, T_this_MeV, ye_this, nulib_h5py, 'scattering_opacity')
        nubar_e_scattering_opacity_this_cm[E_index] = interpolate_eas(E_index, 1, rho_this_gccm, T_this_MeV, ye_this, nulib_h5py, 'scattering_opacity')
        nu_x_scattering_opacity_this_cm   [E_index] = interpolate_eas(E_index, 2, rho_this_gccm, T_this_MeV, ye_this, nulib_h5py, 'scattering_opacity')
    
    return [mu_nu_e_MeV_this,
            mu_nubar_e_MeV_this,
            n_nu_e_eq_this_ccm,
            n_nubar_e_eq_this_ccm,
            n_nu_x_eq_this_ccm,
            nu_e_absorption_opacity_this_cm,
            nubar_e_absorption_opacity_this_cm,
            nu_x_absorption_opacity_this_cm,
            nu_e_scattering_opacity_this_cm,
            nubar_e_scattering_opacity_this_cm,
            nu_x_scattering_opacity_this_cm,
            mu_e]

# --------------------------------------------------------------------------------

# Define a wrapper function for multiprocessing
def compute_weak_quantities_wrapper(args):
    return compute_weak_quantities(*args)

# Prepare arguments for each cell
args_list = []
indices = []

for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            indices.append((i, j, k)) 
            rho_this_gccm = rho_gccm[i, j, k]
            ye_this = ye[i, j, k]
            T_this_MeV = temperature_MeV[i, j, k]
            args_list.append((rho_this_gccm, ye_this, T_this_MeV, E_bins_MeV, delta_E_cubic_bins_MeV_cubic))

# Use multiprocessing to parallelize the computation
start_time = time.time()
with Pool() as pool:
    results = pool.map(compute_weak_quantities_wrapper, args_list)
end_time = time.time()
print(f"Execution time for parallel compute_weak_quantities: {end_time - start_time} seconds")

# Unpack results into respective arrays
for idx, result in enumerate(results):

    i, j, k = indices[idx]

    (mu_nu_e_MeV[i, j, k],
     mu_nubar_e_MeV[i, j, k],
     n_nu_e_eq_ccm[:, i, j, k],
     n_nubar_e_eq_ccm[:, i, j, k],
     n_nu_x_eq_ccm[:, i, j, k],
     nu_e_absorption_opacity_cm[:, i, j, k],
     nubar_e_absorption_opacity_cm[:, i, j, k],
     nu_x_absorption_opacity_cm[:, i, j, k],
     nu_e_scattering_opacity_cm[:, i, j, k],
     nubar_e_scattering_opacity_cm[:, i, j, k],
     nu_x_scattering_opacity_cm[:, i, j, k],
     mu_e_MeV[i,j,k]) = result

# --------------------------------------------------------------------------------

one_km_in_cm = 1e5 # cm

nu_e_absorption_optical_depth_1km = nu_e_absorption_opacity_cm * one_km_in_cm
nubar_e_absorption_optical_depth_1km = nubar_e_absorption_opacity_cm * one_km_in_cm
nu_x_absorption_optical_depth_1km = nu_x_absorption_opacity_cm * one_km_in_cm

nu_e_scattering_optical_depth_1km = nu_e_scattering_opacity_cm * one_km_in_cm
nubar_e_scattering_optical_depth_1km = nubar_e_scattering_opacity_cm * one_km_in_cm
nu_x_scattering_optical_depth_1km = nu_x_scattering_opacity_cm * one_km_in_cm

# --------------------------------------------------------------------------------

def save_output_data(filename, xcenters, ycenters, zcenters, datasetname, dataset):
    with h5py.File(filename, 'w') as output_h5py:
        output_h5py.create_dataset('xcenters', data=xcenters)
        output_h5py.create_dataset('ycenters', data=ycenters)
        output_h5py.create_dataset('zcenters', data=zcenters)
        output_h5py.create_dataset(datasetname, data=dataset)

save_output_data("nu_e_absorption_optical_depth_1km.h5", Xc, Yc, Zc, "nu_e_absorption_optical_depth_1km", nu_e_absorption_optical_depth_1km)
save_output_data("nubar_e_absorption_optical_depth_1km.h5", Xc, Yc, Zc, "nubar_e_absorption_optical_depth_1km", nubar_e_absorption_optical_depth_1km)
save_output_data("nu_x_absorption_optical_depth_1km.h5", Xc, Yc, Zc, "nu_x_absorption_optical_depth_1km", nu_x_absorption_optical_depth_1km)

save_output_data("nu_e_scattering_optical_depth_1km.h5", Xc, Yc, Zc, "nu_e_scattering_optical_depth_1km", nu_e_scattering_optical_depth_1km)
save_output_data("nubar_e_scattering_optical_depth_1km.h5", Xc, Yc, Zc, "nubar_e_scattering_optical_depth_1km", nubar_e_scattering_optical_depth_1km)
save_output_data("nu_x_scattering_optical_depth_1km.h5", Xc, Yc, Zc, "nu_x_scattering_optical_depth_1km", nu_x_scattering_optical_depth_1km)

save_output_data("n_nu_e_eq_ccm.h5", Xc, Yc, Zc, "n_nu_e_eq_ccm", n_nu_e_eq_ccm)
save_output_data("n_nubar_e_eq_ccm.h5", Xc, Yc, Zc, "n_nubar_e_eq_ccm", n_nubar_e_eq_ccm)
save_output_data("n_nu_x_eq_ccm.h5", Xc, Yc, Zc, "n_nu_x_eq_ccm", n_nu_x_eq_ccm)

save_output_data("mu_e_MeV.h5", Xc, Yc, Zc, "mu_e_MeV", mu_e_MeV)
save_output_data("mu_nu_e_MeV.h5", Xc, Yc, Zc, "mu_nu_e_MeV", mu_nu_e_MeV)
save_output_data("mu_nubar_e_MeV.h5", Xc, Yc, Zc, "mu_nubar_e_MeV", mu_nubar_e_MeV)

save_output_data("nu_e_absorption_opacity_cm.h5", Xc, Yc, Zc, "nu_e_absorption_opacity_cm", nu_e_absorption_opacity_cm)
save_output_data("nubar_e_absorption_opacity_cm.h5", Xc, Yc, Zc, "nubar_e_absorption_opacity_cm", nubar_e_absorption_opacity_cm)
save_output_data("nu_x_absorption_opacity_cm.h5", Xc, Yc, Zc, "nu_x_absorption_opacity_cm", nu_x_absorption_opacity_cm)

save_output_data("nu_e_scattering_opacity_cm.h5", Xc, Yc, Zc, "nu_e_scattering_opacity_cm", nu_e_scattering_opacity_cm)
save_output_data("nubar_e_scattering_opacity_cm.h5", Xc, Yc, Zc, "nubar_e_scattering_opacity_cm", nubar_e_scattering_opacity_cm)
save_output_data("nu_x_scattering_opacity_cm.h5", Xc, Yc, Zc, "nu_x_scattering_opacity_cm", nu_x_scattering_opacity_cm)

# --------------------------------------------------------------------------------
# Close the HDF5 files
rho_ye_T_h5py.close()
nulib_h5py.close()
eos_h5py.close()
# --------------------------------------------------------------------------------