"""plot_results.py – visualise the heat‑equation output."""

from __future__ import annotations
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

def main() -> None:
    folder = pathlib.Path("results")          # same folder you used in run.py
    x = np.loadtxt(folder / "x.csv", delimiter=",")
    u = np.loadtxt(folder / "u_final.csv", delimiter=",")
    meta = json.loads((folder / "metadata.json").read_text())

    plt.plot(x, u, label="numerical")
    # analytical reference for the sine mode:
    L = meta["domain_length"]
    alpha = meta["diffusion_coefficient"]
    T = meta["time_total"]
    u_ana = np.sin(np.pi * x / L) * np.exp(-alpha * (np.pi / L) ** 2 * T)
    plt.plot(x, u_ana, "--", label="analytic")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title(f"{meta['method'].upper()} – {meta['boundary_type']} BC")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()