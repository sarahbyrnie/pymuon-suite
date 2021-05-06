"""potential.py

ElectrostaticPotential class for Unperturbed Electrostatic Potential
"""

# Python 2-to-3 compatibility code
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

import numpy as np
from ase import io
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.cube import read_cube
from ase.units import Bohr
from parsefmt.fmtreader import FMTReader
from pymuonsuite.io.castep import parse_castep_ppots
from pymuonsuite.utils import make_process_slices
from scipy import constants as cnst
from soprano.utils import silence_stdio

# Coulomb constant
_cK = 1.0/(4.0*np.pi*cnst.epsilon_0)
# Crystal unit conversion - unsure
_crys = 1.0

def _makegrid(lattice, grid):
    """Return the proper Fourier grid given the cell and number of grid points

    | Arguements:
    |   lattice {np.array} -- Unit cell lattice
    |   grid    {np.array} -- Number of grid points
    |
    | Returns:
    |   g_grid  {np.array} -- Fourier grid
    
    """

    inv_latt = np.linalg.inv(lattice.T)*2*np.pi
    fft_grid = np.array(np.meshgrid(*[np.fft.fftfreq(grid[i])*grid[i]
                                      for i in range(3)], indexing='ij'))
    # Uses the g-vector convention in formulas used
    g_grid = np.tensordot(inv_latt, fft_grid, axes=(0, 0))

    return g_grid

def _parse_castep_den(seedname, gw_fac, path):
    """Read in charge density from CASTEP .den_fmt and return it in Fourier
       space ready for use in _chdens2potential

    | Arguements:
    |   seedname {str}  -- The seedname of the CASTEP output files
    |                      (.den_fmt and .castep) used to load the data.
    |   gw_fac {number} -- Factor used to divide the Gaussian width used
    |                      for the ions. The final width will be the
    |                      radius of the pseudopotential divided by this.
    |   path {str}      -- Path in which the CASTEP output files can be found.
    |
    | Raises:
    |   RuntimeError -- CASTEP pseudopotentials were not found
    |
    | Returns:
    |   rho_G {np.array}   -- Potential in Fourier space
    |   g_grid {np.array}  -- Fourier grid of system
    |   cell {np.array}    -- Unit cell lattice
    |   struct {ase.Atoms} -- Unit cell
    
    """

    # Load the electronic density
    seedpath = os.path.join(path, seedname)

    elec_den = FMTReader(seedpath + '.den_fmt')
    with silence_stdio():
        struct = io.read(seedpath + '.castep')

    ppots = parse_castep_ppots(seedpath + '.castep')

    # Override by also grabbing any pseudopotentials found in the .cell
    # file
    cppot = None
    try:
        with silence_stdio():
            cppot = io.read(seedpath + '.cell').calc.cell.species_pot.value
    except IOError:
        pass  # If not available, ignore this
    if cppot is not None:
        ppf = [l.split() for l in cppot.split('\n') if l]
        for el, pppath in ppf:
            f = os.path.join(path, pppath)
            try:
                ppots.update(parse_castep_ppots(f))
            except IOError:
                # File not found
                print('WARNING: pseudopotential file '
                      '{0} not found'.format(f))

    # FFT grid
    cell = np.array(elec_den.real_lattice)
    grid = np.array(elec_den.grid)

    g_grid = _makegrid(cell, grid)

    # Information for the elements, and guarantee zero net charge
    elems = struct.get_chemical_symbols()
    pos = struct.get_positions()
    try:
        q = np.array([ppots[el][0] for el in elems])
        gw = np.array([ppots[el][1]/gw_fac for el in elems])
    except KeyError:
        raise RuntimeError("""Some or all CASTEP pseudopotentials were not
    found. UEP calculation can not go on. Please notice that at the moment only
    ultrasoft pseudopotentials are supported, and if not generated automatically,
    they must be possible to retrieve using the paths in the SPECIES_POT block of
    the .cell file.""")

    # Here we find the Fourier components of the potential due to
    # the valence electrons
    rho = elec_den.data[:, :, :, 0]
    if not np.isclose(np.average(rho), sum(q), 1e-4):
        raise RuntimeError('Cell is not neutral')
    # Put the minus sign for electrons
    rho *= -sum(q)/np.sum(rho)  # Normalise charge
    rhoe_G = np.fft.fftn(rho)
    Gnorm = np.linalg.norm(g_grid, axis=0)

    # Now on to doing the same for ionic components
    rhoi_G = g_grid[0]*0.j
    for i, p in enumerate(pos):
        rhoi_G += (q[i] * np.exp(-1.0j*np.sum(g_grid[:, :, :, :] *
                                              p[:, None, None, None],
                                             axis=0) -
                                0.5*(gw[i] * Gnorm)**2))

    rho_G = rhoi_G + rhoe_G

    return rho_G, g_grid, cell, struct

def _chdens2potential(rho_G, g_grid, cell):
    """Converts charge density into potential in Fourier space

    | Arguements:
    |   rho_G   {np.array} -- Charge density in Fourier space
    |   g_grid  {np.array} -- Fourier grid
    |   cell    {np.array} -- Unit cell lattice
    |
    | Returns:
    |   V_G     {np.array} -- Potential in Fourier space
    
    """

    Gnorm = np.linalg.norm(g_grid, axis=0)
    Gnorm_fixed = np.where(Gnorm > 0, Gnorm, np.inf)
    vol = abs(np.dot(np.cross(cell[:, 0], cell[:, 1]), cell[:, 2]))
    pregrid = (4*np.pi/Gnorm_fixed**2*1.0/vol)
    V_G = (pregrid*rho_G)

    return V_G

class ElectrostaticPotential(object):
    """
    ElectrostaticPotential

    An object storing the elctrostatic potential of a cell.
    """

    def __init__(self, V, a, g_grid, program, is_fourier=True):
        """
        Initialise ElectrostaticPotential object

        Initialise an ElectrostaticPotential object with a 3D numpy array of
        the potential in a cell.

        | Arguments:
        |   V {np.array}      -- Potential in fourier space
        |   a {ase.atoms}     -- Unit cell
        |   g_grid {np.array} -- Fourier grid
        |   program {string}  -- Name of dft software used
        |
        | Keyword Arguments:
        |   is_fourier {boolean} -- Bool indicating if potential is in fourier space
        |                           (default: {True})

        """

        if(is_fourier == False):
            V_G = 2*np.pi*np.fft.fftn(V)
            self._V_G = V_G
        else:
            self._V_G = V

        self._struct = a
        self._g_grid = g_grid
        self._program = program

    @property
    def atoms(self):
        a = self._struct
        return a  # atoms object of unit cell

    def V(self, p, max_process_p=20):
        """Potential

        Compute electrostatic potential at a point or list of points,
        total and split by electronic and ionic contributions.

        Arguments:
            p {np.ndarray} -- List of points to compute potential at.

        Keyword Arguments:
            max_process_p {number} -- Max number of points processed at once.
                                      Lower to trade off speed for memory
                                      (default: {20})

        Returns:
            np.ndarray -- Total potential
        """

        # Return potential at a point or list of points
        p = np.array(p)
        if len(p.shape) == 1:
            p = p[None, :]   # Make it into a list of points

        # The point list is sliced for convenience, to avoid taking too much
        # memory
        N = p.shape[0]
        V = np.zeros(N)

        slices = make_process_slices(N, max_process_p)

        for s in slices:
            # Fourier transform kernel
            ftk = np.exp(1.0j*np.tensordot(self._g_grid, p[s].T, axes=(0, 0)))
            # Compute the electronic potential
            V[s] = np.real(np.sum(self._V_G[:, :, :, None]*ftk,
                                   axis=(0, 1, 2)))

        if(program=='castep'):
            V *= _cK*cnst.e*1e10  # Moving to SI units
        if(program=='crystal'):
            V *= 2*np.pi

        return V

    def dV(self, p, max_process_p=20):
        """Potential gradient

        Compute electrostatic potential gradient at a point or list of
        points, total and split by electronic and ionic contributions.

        Arguments:
            p {np.ndarray} -- List of points to compute potential gradient at.

        Keyword Arguments:
            max_process_p {number} -- Max number of points processed at once.
                                      Lower to trade off speed for memory
                                      (default: {20})

        Returns:
            np.ndarray -- Total potential gradient
        """

        # Return potential gradient at a point or list of points
        p = np.array(p)
        if len(p.shape) == 1:
            p = p[None, :]   # Make it into a list of points

        # The point list is sliced for convenience, to avoid taking too much
        # memory
        N = p.shape[0]
        dV = np.zeros((N, 3))

        slices = make_process_slices(N, max_process_p)

        for s in slices:
            # Fourier transform kernel
            ftk = np.exp(1.0j*np.tensordot(self._g_grid, p[s].T, axes=(0, 0)))
            dftk = 1.0j*self._g_grid[:, :, :, :, None]*ftk[None, :, :, :, :]
            # Compute the electronic potential
            dV[s] = np.real(np.sum(self._V_G[None, :, :, :, None] * dftk,
                                   axis=(1, 2, 3))).T

        dV *= _cK*cnst.e*1e20  # Moving to SI units

        return dV

    def d2V(self, p, max_process_p=20):
        """Potential Hessian

        Compute electrostatic potential Hessian at a point or list of
        points, total and split by electronic and ionic contributions.

        Arguments:
            p {np.ndarray} -- List of points to compute potential Hessian at.

        Keyword Arguments:
            max_process_p {number} -- Max number of points processed at once.
                                      Lower to trade off speed for memory
                                      (default: {20})

        Returns:
            np.ndarray -- Total potential Hessian
        """

        # Return potential Hessian at a point or a list of points

        p = np.array(p)
        if len(p.shape) == 1:
            p = p[None, :]   # Make it into a list of points

        # The point list is sliced for convenience, to avoid taking too much
        # memory
        N = p.shape[0]
        d2V = np.zeros((N, 3, 3))

        slices = make_process_slices(N, max_process_p)
        g2_mat = (self._g_grid[:, None, :, :, :] *
                  self._g_grid[None, :, :, :, :])

        for s in slices:
            # Fourier transform kernel
            ftk = np.exp(1.0j*np.tensordot(self._g_grid, p[s].T, axes=(0, 0)))
            d2ftk = -g2_mat[:, :, :, :, :, None]*ftk[None, None, :, :, :, :]
            # Compute the electronic potential
            d2V[s] = np.real(np.sum(self._V_G[None, None, :, :, :, None] *
                                    d2ftk, axis=(2, 3, 4))).T

        d2V *= _cK*cnst.e*1e30  # Moving to SI units

        return d2V

    @staticmethod
    def from_castep(seedname, path='', gw_fac=3):
        """Returns an ElectrostaticPotential object initialised from CASTEP charge density

    | Arguements:
    |   seedname {str}  -- The seedname of the CASTEP output files
    |                      (.den_fmt and .castep) used to load the data.
    |
    | Keyword arguements:
    |   gw_fac {number} -- Factor used to divide the Gaussian width used
    |                      for the ions. The final width will be the
    |                      radius of the pseudopotential divided by this.
    |                      (default: {3})
    |   path {str}      -- Path in which the CASTEP output files can be found.
    |                      (default: {''})
    |
    | Returns:
    |   potential {ElectrostaticPotential} -- ElectrostaticPotential object initialised
    |                                         from CASTEP output

        """

        rho_G, g_grid, cell, a = _parse_castep_den(seedname, path, gw_fac)
        V_G = _chdens2potential(rho_G, g_grid, cell)
        potential = ElectrostaticPotential(V_G, a, g_grid, program='castep')

        return potential  # return ElectrostaticPotential object initialised from CASTEP

    @staticmethod
    def from_crystal(filename):
        """Returns an ElectrostaticPotential object initialised from CRYSTAL potential

    | Arguements:
    |   filename {str}  -- The filename of the CRYSTAL potential output file
    |                      used to load the data.
    |
    | Returns:
    |   potential {ElectrostaticPotential} -- ElectrostaticPotential object initialised
    |                                         from CRYSTAL output

        """

        with open(filename) as cube_pot:
            pot_dict = read_cube(cube_pot, program='castep')

        # Pull relevant information from cube dictionary
        a = pot_dict['atoms']
        V = pot_dict['data']

        cell = a.cell[:]
        grid = np.array(V.shape)
        g_grid = _makegrid(cell, grid)

        # Unit conversion and transformation of potential to fourier space
        V *= _crys
        V_G = np.fft.fftn(V)

        potential = ElectrostaticPotential(V_G, a, g_grid, program='crystal')

        return potential  # return ElectrostaticPotential object initialised from CRYSTAL
