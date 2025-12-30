# Finite Element Method (1D) — Uniform and Geometric Meshes

**Author:** Charly MARTIN AVILA  
**Student number:** 21503268  

---

## Overview

This project investigates the numerical solution of a **one-dimensional Poisson problem**
using the **P1 Finite Element Method (FEM)**.
The main focus is on understanding how the **choice of mesh**
(uniform vs geometric) affects accuracy and convergence when the exact solution
has **limited regularity**.

The project was developed at **Master 1 level**, combining theoretical analysis,
numerical methods, and clean Python implementation.

---

## Mathematical Problem

We consider the boundary value problem:

- u''(x) = f(x),   x in (0,1)  
u(0) = u(1) = 0

The exact solution is:
u(x) = x^(3/4) (1 - x)

This solution belongs to H^1_0(0,1) but not to H^2(0,1),
due to a singular derivative at x = 0.
This lack of regularity motivates the use of **geometric meshes**.

---

## Project Structure

.
├── main.py                # Core numerical implementation
├── src/
│   ├── params.py          # Mathematical parameters and exact solution
│   └── utils.py           # Plotting and visualization utilities
├── plots/                 # Generated figures (PNG)
├── report.tex             # LaTeX report
└── README.md

---

## Main Module: main.py

All the **core numerical logic** is implemented in main.py.

### Responsibilities
- Assembly of the 1D FEM stiffness matrix and load vector
- Solution of the linear system
- Computation of errors (energy norm and L2 norm)
- Generation of uniform and geometric meshes
- Numerical experiments (Questions 5–9)

### Key Functions
- FEM_1d_assemble
- FEM_1d_solve
- FEM_1d
- mesh_uniform
- mesh_geometric
- run_uniform
- run_geometric
- errors_for_multiple_alpha
- p1_eval

---

## Parameters Module: src/params.py

This module contains:
- Exact solution and RHS
- Physical and numerical constants
- Default values for N and alpha

---

## Visualization Module: src/utils.py

This module handles:
- Convergence plots
- Mesh visualization
- Error comparison plots
- Solution comparison plots

---

## Conclusion

This project highlights the importance of adapting the discretization strategy
to the analytical properties of the solution.
Geometric meshes significantly improve numerical accuracy for singular solutions,
without increasing computational cost.

---

## Author

Charly MARTIN AVILA  
Student number: 21503268  
Master 1 Numerical Analysis
