"""
Units conversions

Lots of more work to do here
"""
import numpy as np
from ase.units import (_Nav, Angstrom, bar, Bohr,
        Hartree, kJ, kg, second)
from typing import List, Tuple
import warnings

##does several units conversions

def meV_per_atom_to_kJ_per_mol(meV_atom: float) -> float:
    """Converts energy from meV/atom to kJ/mol
    Input:
        meV_atom (float):   energy in meV/atom
    Returns:
        kJ_mol (float):     energy in kJ/mol
    """
    return (1 / 1000) * meV_atom * _Nav / kJ


def kJ_per_mol_to_meV_per_atom(kJ_mol: float) -> float:
    """Converts energy from kJ/mol to meV/atom
    Input:
        kJ_mol (float):   energy in kJ_mol
    Returns:
        meV/atom (float):     energy in meV/atom
    """
    return 1000 * kJ_mol * kJ / _Nav

def Htr_bohr_to_eV_A(Htr_bohr: float) -> float:
    """Converts force from Htr/Bohr to eV/A
    Input:
        Htr_bohr (float):   force in Htr/bohr
    Returns:
        eV_A (float)        force in eV/A
    """
    return Htr_bohr * Hartree / Bohr


