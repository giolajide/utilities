"""
Conversions. See Charles T. Campbell's papers on the SCM and HCM
"""
import numpy as np
from ase.build import bulk
from ase import Atoms
from typing import List, Tuple
import warnings
from contact_angle import calculate_contact_angle

YM = 1.22 #J/m2
MIN_RES = 0.25
##does conversions for the SCM and sphere 

def theta_to_adhesion(SCM_theta: int, Ym: float) -> float:
    """
    Use the Young-Dupre equation (which assumes an SCM model)
    to calculate adhesion from contact angle
    """
    return Ym * (1 + np.cos(np.radians(SCM_theta)))

def adhesion_to_theta(adhesion: float, Ym: float = YM) -> float:
    """
    Invert the Young–Dupre relation (SCM model):
      adhesion = Ym * (1 + cos(theta))
    to get theta (in degrees) from adhesion.
    """
    cos_theta = adhesion / Ym - 1.0
    # guard against minor numerical drift outside [−1,1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    return np.degrees(theta_rad)

def _alpha(theta: int) -> float:
    return 1 / (1 + np.cos(np.radians(theta)))

def _beta(theta: int) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category = RuntimeWarning)
        return (2 + np.cos(np.radians(theta)))\
                * (1 - np.cos(np.radians(theta))) / (np.sin(np.radians(theta)))

def _calculate_bulk_density(
        element: str,
        a: float = None,
        ) -> float:
    """
    Helper function to calculate bulk density in atoms/A^3,
    given the element symbol and (optionally) lattice constant
    """
    atoms = bulk(element, cubic = True, a = a)
    volume = atoms.get_volume()

    return len(atoms) / volume


def _natoms_to_sphere_diameter(
        n_atoms: int,
        element: str,
        a: float = None,
        ) -> float:
    """
    Helper function to convert number of atoms (n_atoms) to diameter of curvature (NOT footprint diameter)
    using the formula for a sphere
    given element, n_atoms, and lattice constant (optionally)

    returns diameter in A
    """
    bulk_density = _calculate_bulk_density(
            element = element,
            a = a
            )
    volume = atoms / bulk_density #A^3
    curvature_diameter = ((24 * volume)/(np.pi * 4))**(1/3) #Ang

    return curvature_diameter


def _natoms_to_SCM_diameter(
        n_atoms: int,
        element: str,
        theta: float,
        a: float = None,
        ) -> float:
    """
    Helper function to convert number of atoms (n_atoms) to diameter of curvature (NOT footprint diameter)
    using the formula for a spherical cap
    given element, n_atoms, theta, and lattice constant (optionally)

    returns diameter in A
    """
    bulk_density = _calculate_bulk_density(
            element = element,
            a = a
            )
    volume = n_atoms / bulk_density #A^3
    diameter = ((24 * volume)/(np.pi * _alpha(theta) * _beta(theta)))**(1/3) #A
    curvature_diameter = diameter / np.sin(np.radians(theta)) #A

    return curvature_diameter


def _natoms_to_SCM_footprint_diameter(
        n_atoms: int,
        element: str,
        theta: float,
        a: float = None,
        ) -> float:
    """
    Helper function to convert number of atoms (n_atoms) to footprint diameter

    returns diameter in A
    """

    return np.sin(np.radians(theta)) * _natoms_to_SCM_diameter(n_atoms = n_atoms,
            element = element,
            a = a,
            theta = theta
            )


def _SCM_footprint_diameter_to_natoms(
        footprint_diameter: float,
        element: str,
        theta: float,
        a: float = None,
        ) -> int:
    """
    Helper function to convert SCM FOOTPRINT diameter to number of atoms

    expects diameter in A
    """
    bulk_density = _calculate_bulk_density(
            element = element,
            a = a
            )
    volume = ((footprint_diameter**3)  * np.pi * _alpha(theta) * _beta(theta)) / 24
    return int(volume * bulk_density) #atoms


def SCM_Y(
        diameters: np.ndarray, #nm
        Y_inf: float, #J/m2
        Do: float, #nm
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Given the diameters in nm, Do in nm, and the infinite size limit
    weighted surface energy (i.e. YM) in J/m2, calculate the surface energy at size x
    according to the SCM model , i.e. YM' = YM(1 + Do/D)

    Requires:
        diameters (array-like object):  Footprint diameters, in nm
        Y_inf (float):                  YM in the infinite size limit, in J/m2
        Do (float):                     The value of Do, in nm

    Returns:
        YMd:                            diameter-varying YM, in J/m2
        diameters:                      diameters sorted from lowest to highest, in nm
    """
    diameters = np.array(diameters)
    diameters.sort()
    YMd = Y_inf * (1 + (Do / diameters))
    return YMd, diameters #J/m2; nm


def SCM_exp_Y(
        diameters: np.ndarray, #nm
        Y_inf: float, #J/m2
        Do: float, #nm
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Given the diameters in nm, Do in nm, and the infinite size limit
    weighted surface energy (i.e. YM) in J/m2, calculate the surface energy at size x
    according to the non-truncated SCM model , i.e.
        
                                YM' = YM * exp(Do/D)

    Requires:
        diameters (array-like object):  Footprint diameters, in nm
        Y_inf (float):                  YM in the infinite size limit, in J/m2
        Do (float):                     The value of Do, in nm

    Returns:
        YMd:                            diameter-varying YM, in J/m2
        diameters:                      diameters sorted from lowest to highest, in nm
    """
    diameters = np.array(diameters)
    diameters.sort()
    YMd = Y_inf * np.exp(Do / diameters)
    return YMd, diameters #J/m2; nm


def SCM_Y_2nd_order(
        diameters: np.ndarray, #nm
        Y_inf: float, #J/m2
        Do: float, #nm
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Same as SCM_Y() but uses the 2nd-order form, rather than the 1st-order one
    """
    diameters = np.array(diameters)
    diameters.sort()
    YMd = Y_inf * (1 + (Do / diameters) + ((1/2)*((Do / diameters)**2)))
    return YMd, diameters #J/m2; nm

def SCM_Y_3rd_order(
        diameters: np.ndarray, #nm
        Y_inf: float, #J/m2
        Do: float, #nm
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Same as SCM_Y() but uses the 3rd-order form, rather than the 1st-order one
    """
    diameters = np.array(diameters)
    diameters.sort()
    YMd = Y_inf * (1 + (Do / diameters) + ((1/2)*((Do / diameters)**2)) + ((1/6)*((Do / diameters)**3)))
    return YMd, diameters #J/m2; nm


###the spirit of this function dictates that it be elsewhere, but it'll stay here for now
def get_SCM_diameter_and_theta(
        atoms: Atoms,
        np_element: str,
        min_resolution: float = MIN_RES,
        ) -> float:
    f"""
    Get diameter OF CURVATURE (in Ang) and contact angle by fitting the NP to a spherical cap

    Requires:
        atoms (Atoms):              atoms object. NP on support
        np_element (str):           what element the NP is made of.
        min_resolution (float):     resolution for voxel grid. default = {MIN_RES} Ang

    Returns:
        curvature_diameter (float): diameter of curvature, in Ang
        angle (float):              contact angle, in degrees
    """
    angle, _, footprint_radius = calculate_contact_angle(
            atoms = atoms,
            np_element = np_element,
            min_resolution = min_resolution
            )

    curvature_diameter = 2 * footprint_radius / np.sin(np.radians(angle)) #Ang

    return curvature_diameter, angle #A, degrees




