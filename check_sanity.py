import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from sys import exit, argv
from typing import List, Union, Tuple
from ase import Atoms
from tqdm import tqdm
from ase.neighborlist import natural_cutoffs, NeighborList
from os.path import splitext, basename
from typing import List, Union, Tuple
import functools
try:
    from md_quality_assurance import check_autoionization
except ImportError:
    print("Add the Active_Learning dir to your PATH first")



#custom script from ~/npscripts/
from fit_support import check_exploded, check_inversion

#custom script from ~/Active_Learning/
from md_quality_assurance import (
        check_for_overlapping_atoms, check_for_explosion,
        analyze_neighbors_evolution, track_num_neighbors, check_for_floating_atoms
        )


###check that runs and structures are sane!


AG_MAX_SPACING = 70 #Angstrom
AG_COORD_CUTOFF = 3 #bonds

def not_functional(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"""{function.__name__} is not the best.
            Use Demola's script instead""",
            DeprecationWarning,
            stacklevel = 2
        )
        return function(*args, **kwargs)
    return wrapper


class StructureError(Exception):
    """
    Exception raised for ridiculous systems,
    e.g. supposed water system with len(H) != 2 x len(O)
    """
    pass


def check_water(trajectory: Union[List[Atoms], Atoms]) -> bool:
    """
    Checks if len(H) == 2 X len(O) in every image of supplied structure
    Returns True if so, else False
    """
    if isinstance(trajectory, list):
        return all([
            atoms.symbols.count("H") == 2 * atoms.symbols.count("O") for atoms in trajectory
            ])
    elif isinstance(trajectory, Atoms):
        return trajectory.symbols.count("H") == 2 * trajectory.symbols.count("O")


@not_functional
def evaluate_water_MD(trajectory: List[Atoms], output_name: str = "output",
        nprocs: int = -1) -> None:
    """
    Utility function using the active learning dir to
    check if any of your waters ionized
    """
    output_name = splitext(basename(output_name))[0]

    neighbors = NeighborList(cutoffs = natural_cutoffs(trajectory[0]),
            self_interaction = False, bothways = True)

    result = Parallel(n_jobs = nprocs)(delayed(check_autoionization)(
        atoms = image, nl = neighbors) for image in tqdm(trajectory, total = len(trajectory)))

    surely_suceeded = np.all(np.array([not i for i in result]))

    if surely_suceeded:
        print("All went well.")
        return None

    possibly_problematic_images = np.where(result)

    if possibly_problematic_images:
        warnings.warn("As at April 3rd, 2025, the function flags some good images as bad.",
                category = UserWarning)
        print(f"""There are {possibly_problematic_images[0].shape[0]} possibly ionized images.
        Here they are:\t{possibly_problematic_images[0]}""")
        write(f"{output_name}.traj", [trajectory[i] for i in list(possibly_problematic_images[0])])
        with open(f"{output_name}.log","a") as file:
            file.write(str(possibly_problematic_images[0]))

        return None


def check_Ag_MgO(trajectory: Union[List[Atoms], Atoms], spacing: float = AG_MAX_SPACING,
        coord_cutoff: int = AG_COORD_CUTOFF) -> bool:
    """
    Several checks for the sanity of an Ag/MgO system
    Returns True if the structure MIGHT be rational, else False
    """
    if isinstance(trajectory, list):
        return all([
            (check_exploded(atoms) < spacing) and
            (not check_inversion(atoms)) and
            (not check_for_floating_atoms(atoms, coord_cutoff = coord_cutoff)) and
            (not check_for_overlapping_atoms(atoms))
            for atoms in trajectory
            ])
    elif isinstance(trajectory, Atoms):
        return all([
            (check_exploded(trajectory) < spacing) and
            (not check_inversion(trajectory)) and
            (not check_for_floating_atoms(trajectory, coord_cutoff = coord_cutoff)) and
            (not check_for_overlapping_atoms(trajectory))
            ])




