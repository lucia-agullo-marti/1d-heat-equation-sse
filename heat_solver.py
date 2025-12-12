"""heat_solver.py

Numerical solvers for the 1‑D heat (diffusion) equation

    ∂u/∂t = α ∂²u/∂x²

Supports:
    * Explicit FTCS scheme (conditionally stable)
    * Implicit Crank–Nicolson scheme (unconditionally stable)

All functions are pure; no global state is used.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import Callable, Literal, Tuple


BoundaryType = Literal["Dirichlet", "Neumann"]
InitialType = Literal["gaussian", "sine", "constant"]
OutputFormat = Literal["csv", "npz"]


def _apply_boundary(u: np.ndarray, bc_type: BoundaryType, bc_value: float = 0.0) -> None:
    """Apply Dirichlet/Neumann boundary conditions in‑place."""
    if bc_type == "Dirichlet":
        u[0] = bc_value
        u[-1] = bc_value
    elif bc_type == "Neumann":
        # Zero‑gradient (∂u/∂x=0) implemented with a first‑order ghost point
        u[0] = u[1]
        u[-1] = u[-2]
    else:
        raise ValueError(f"Unsupported boundary type: {bc_type}")


def _initial_condition(
    x: np.ndarray,
    init_type: InitialType,
    amplitude: float = 1.0,
    centre: float | None = None,
    sigma: float = 0.1,
) -> np.ndarray:
    """Return u(x,0) according to the selected initial condition."""
    if init_type == "gaussian":
        centre = centre if centre is not None else 0.5 * (x[0] + x[-1])
        return amplitude * np.exp(-((x - centre) ** 2) / (2 * sigma**2))
    elif init_type == "sine":
        return amplitude * np.sin(np.pi * x / x[-1])
    elif init_type == "constant":
        return amplitude * np.ones_like(x)
    else:
        raise ValueError(f"Unsupported initial condition type: {init_type}")


def solve_ftcs(
    L: float,
    Nx: int,
    T: float,
    dt: float,
    alpha: float,
    bc_type: BoundaryType,
    init_type: InitialType,
    *,
    bc_value: float = 0.0,
    save_history: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Solve using the explicit Forward Time Central Space (FTCS) scheme.

    Parameters
    ----------
    L, Nx, T, dt, alpha, bc_type, init_type : see README
    bc_value : value for Dirichlet boundaries (ignored for Neumann)
    save_history : if True, also return the full (time, space) solution

    Returns
    -------
    x : spatial grid (Nx,)
    u : solution at final time (Nx,)
    u_hist : (Nt, Nx) if ``save_history`` else ``None``

    Raises
    ------
    ValueError
        If the CFL condition α·dt/dx² > 0.5 (unstable explicit scheme).
    """
    dx = L / (Nx - 1)
    if alpha * dt / dx**2 > 0.5:
        raise ValueError(
            f"CFL condition violated: α·dt/dx² = {alpha*dt/dx**2:.3f} > 0.5. "
            "Reduce dt or increase Nx."
        )

    x = np.linspace(0.0, L, Nx)
    u = _initial_condition(x, init_type)
    _apply_boundary(u, bc_type, bc_value)

    r = alpha * dt / dx**2
    Nt = int(np.ceil(T / dt))
    u_hist = [] if save_history else None

    for n in range(Nt):
        u_new = u.copy()
        # interior update
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        _apply_boundary(u_new, bc_type, bc_value)
        u = u_new
        if save_history:
            u_hist.append(u.copy())

    return x, u, np.array(u_hist) if save_history else None


def solve_crank_nicolson(
    L: float,
    Nx: int,
    T: float,
    dt: float,
    alpha: float,
    bc_type: BoundaryType,
    init_type: InitialType,
    *,
    bc_value: float = 0.0,
    save_history: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Solve using the implicit Crank–Nicolson scheme (unconditionally stable).

    The linear system (I - r/2 A) u^{n+1} = (I + r/2 A) u^{n} is solved at each step,
    where A is the discrete Laplacian with appropriate boundary treatment.
    """
    dx = L / (Nx - 1)
    r = alpha * dt / dx**2
    x = np.linspace(0.0, L, Nx)
    u = _initial_condition(x, init_type)
    _apply_boundary(u, bc_type, bc_value)

    # Build the tridiagonal matrix A (Laplacian) with Dirichlet or Neumann BCs
    main = -2.0 * np.ones(Nx)
    off = np.ones(Nx - 1)

    if bc_type == "Dirichlet":
        main[0] = main[-1] = 1.0
        off[0] = off[-1] = 0.0
    elif bc_type == "Neumann":
        main[0] = main[-1] = -1.0  # ghost-point implementation
        off[0] = off[-1] = 1.0
    else:
        raise ValueError(f"Unsupported boundary type: {bc_type}")

    A = diags([off, main, off], offsets=[-1, 0, 1], format="csc")
    I = diags([np.ones(Nx)], [0], format="csc")
    M_left = I - (r / 2.0) * A
    M_right = I + (r / 2.0) * A

    Nt = int(np.ceil(T / dt))
    u_hist = [] if save_history else None

    for _ in range(Nt):
        b = M_right.dot(u)
        # Enforce Dirichlet values directly in RHS
        if bc_type == "Dirichlet":
            b[0] = bc_value
            b[-1] = bc_value
        # Solve linear system
        u = spsolve(M_left, b)
        # Apply Neumann BCs explicitly (the system already respects them)
        if bc_type == "Neumann":
            u[0] = u[1]
            u[-1] = u[-2]
        if save_history:
            u_hist.append(u.copy())

    return x, u, np.array(u_hist) if save_history else None