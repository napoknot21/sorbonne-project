from __future__ import annotations


import numpy as np
import matplotlib.pyplot as plt

from src.parameters.


def plot_convergence (N_values, err, title : str, ylabel) :
    """
    
    """
    plt.figure()

    plt.loglog(N_values, err, marker="o")
    plt.grid(True, which="both")

    plt.xlabel("N (number of elements)")
    plt.ylabel(ylabel)

    plt.title(title)

    return None


def plot_meshes (N : int = 50, alphas : tuple = (1, 2, 4, 7, 10)) :
    """
    
    """
    plt.figure()

    for a in alphas :
        x = mesh
    