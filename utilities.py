"""
Random functions, some very useful though"""
import numpy as np
from ase.build import bulk
from ase import Atoms
from typing import List, Tuple
import warnings


LAYER_TOLERANCE = 0.6 #A. Every atom within this z of each other is in the same 'layer'


def calc_fmax(atoms):
    """Calculate Fmax"""
    return np.max(np.linalg.norm(atoms.get_forces(), ord = 2, axis = 1))


def unsigned_vector_angle(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    """
    returns the (unsigned) angle, in degrees, between two 1D vectors
    """
    vector_1 = np.array(vector_1)
    vector_2 = np.array(vector_2)
    if (vector_2.shape != vector_1.shape or vector_1.ndim != 1):
        raise TypeError("Vectors must have the same shape, and must be 1D.")

    numerator = np.dot(vector_1, vector_2)
    denominator = np.linalg.norm(vector_1, ord = 2) * np.linalg.norm(vector_2, ord = 2)
    angle = np.degrees(np.arccos(numerator / denominator))

    return angle


def classify_into_layers(atoms: Atoms, tol: float = LAYER_TOLERANCE) -> Tuple[Atoms, np.ndarray]:
    """
    Classify an atoms object into the layers each atom belongs to

    !!!!atoms object SHOULD BE JUST THE NANOPARTICLE!!!!

    returns:
        atoms:      same atoms object
        layers:     array of layer index, starting from 1
            i.e. layers=[1,5] means atom 0 is in the 1st layer and atom 1 is in the 5th
    """
    zs = atoms.positions[:,2]
    sorting_mask = np.argsort(zs)
    zs_sorted = zs[sorting_mask]

    layers_sorted = np.zeros(zs_sorted.shape, dtype=int)
    current_layer = 1 #interfacelayer = 1
    ref = zs_sorted[0] #lowest z
    layers_sorted[0] = current_layer

    for i in range(1, len(zs_sorted)):
        if abs(zs_sorted[i] - ref) > tol:
            current_layer += 1 #if we are outside of the tolerance, we move to next layer
            ref = zs_sorted[i]
        layers_sorted[i] = current_layer #else we remain on current layer

    # unsort back to original atom order
    layers = np.empty(layers_sorted.shape, dtype=int)
    layers[sorting_mask] = layers_sorted

    return atoms, np.array(layers)
    


