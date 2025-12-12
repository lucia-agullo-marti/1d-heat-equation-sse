"""run.py

Command‑line entry point for the 1‑D heat equation solver.

Usage
-----
    python run.py --config config/example.yaml
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import yaml

from heat_solver import solve_ftcs, solve_crank_nicolson
from typing import Literal


def load_config(path: pathlib.Path) -> dict:
    """Load a YAML configuration file and return it as a dict."""
    if not path.is_file():
        sys.exit(f"Configuration file '{path}' does not exist.")
    with path.open("rt", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="1‑D heat equation solver")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ------------------------------------------------------------ parameters
    L = float(cfg["domain"]["L"])
    Nx = int(cfg["domain"]["Nx"])
    T = float(cfg["time"]["T"])
    dt = float(cfg["time"]["dt"])
    alpha = float(cfg["physics"]["alpha"])

    bc_type: Literal["Dirichlet", "Neumann"] = cfg["boundary"]["type"]
    bc_value = float(cfg["boundary"].get("value", 0.0))

    init_type: Literal["gaussian", "sine", "constant"] = cfg["initial"]["type"]
    method = cfg.get("method", "ftcs").lower()

    out_fmt: Literal["csv", "npz"] = cfg.get("output", {}).get("format", "csv")
    save_full = bool(cfg.get("output", {}).get("save_full", False))
    out_dir = pathlib.Path(cfg.get("output", {}).get("directory", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ solver dispatch
    if method == "ftcs":
        solver = solve_ftcs
    elif method == "crank-nicolson":
        solver = solve_crank_nicolson
    else:
        sys.exit(f"Unknown method '{method}'. Choose 'ftcs' or 'crank-nicolson'.")

    try:
        x, u_final, u_hist = solver(
            L,
            Nx,
            T,
            dt,
            alpha,
            bc_type,
            init_type,
            bc_value=bc_value,
            save_history=save_full,
        )
    except ValueError as exc:
        sys.exit(str(exc))

    # ------------------------------------------------------------ write results
    if out_fmt == "csv":
        np.savetxt(out_dir / "x.csv", x, delimiter=",")
        np.savetxt(out_dir / "u_final.csv", u_final, delimiter=",")
        if save_full:
            np.savetxt(out_dir / "u_history.csv", u_hist, delimiter=",")
    elif out_fmt == "npz":
        np.savez(
            out_dir / "solution.npz",
            x=x,
            u_final=u_final,
            u_history=u_hist if save_full else None,
        )
    else:
        sys.exit(f"Unsupported output format '{out_fmt}'.")

    # ------------------------------------------------------------ metadata file
    metadata = {
        "equation": "∂u/∂t = α ∂²u/∂x²",
        "method": method,
        "domain_length": L,
        "grid_points": Nx,
        "time_total": T,
        "time_step": dt,
        "diffusion_coefficient": alpha,
        "boundary_type": bc_type,
        "boundary_value": bc_value,
        "initial_condition": init_type,
        "output_format": out_fmt,
        "saved_full_history": save_full,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Results written to directory '{out_dir}'.")
    print("Metadata stored in 'metadata.json'.")


if __name__ == "__main__":
    main()