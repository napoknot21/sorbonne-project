from __future__ import annotations


import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple


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


def plot_meshes (N, alphas, func) :
    """
    Docstring for plot_meshes
    
    :param N: Description
    :type N: int
    :param alphas: Description
    :type alphas: Tuple
    """

    plt.figure()

    for a in alphas :

        x = func(N, a)
        plt.plot(x, 0*x + a, marker=".", linestyle="none", label=f"alpha={a}")

    plt.grid(True, which="both")
    plt.xlabel("x_i")
    plt.ylabel("Alpha (just to separate line)")
    
    plt.title(f"Geometric meshes (N={N})")
    plt.legend()

    return None