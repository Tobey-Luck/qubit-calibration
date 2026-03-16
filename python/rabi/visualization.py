"""
rabi/visualization.py
---------------------
Plotting utilities for Rabi oscillation data, fits, and Bloch sphere trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import qutip as qt
from typing import Optional

from .simulator import RabiResult
from .fitting import FitResult, rabi_model


def plot_rabi_fit(
    result: RabiResult,
    fit: FitResult,
    noisy_data: Optional[np.ndarray] = None,
    cpp_data: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot Rabi oscillation data alongside the fitted curve and residuals.

    Args:
        result:      RabiResult from simulator
        fit:         FitResult from fitting
        noisy_data:  Noisy measurements to plot as scatter (optional)
        cpp_data:    P(|1>) from C++ solver for comparison (optional)
        save_path:   If provided, save figure to this path

    Returns:
        Matplotlib Figure
    """
    fig = plt.figure(figsize=(10, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)

    ax_main = fig.add_subplot(gs[0])
    ax_res  = fig.add_subplot(gs[1], sharex=ax_main)

    times = result.times
    fitted_curve = rabi_model(
        times,
        fit.omega_rabi_fit,
        fit.amplitude,
        fit.decay_time,
        fit.offset,
        0.0,
    )

    # --- Main panel ---
    ax_main.plot(times, result.excited_pop, color="#2196F3", lw=1.8,
                 label="QuTiP simulation (ideal)", zorder=2)

    if noisy_data is not None:
        ax_main.scatter(times, noisy_data, s=8, color="#FF9800", alpha=0.6,
                        label="Simulated measurements", zorder=3)

    if cpp_data is not None:
        ax_main.plot(times, cpp_data, color="#4CAF50", lw=1.5, ls="--",
                     label="C++ RK4 solver", zorder=4)

    ax_main.plot(times, fitted_curve, color="#F44336", lw=2.0, ls="-.",
                 label=f"Fit: ω = {fit.omega_rabi_fit:.4f} ± {fit.omega_rabi_err:.4f} rad/µs",
                 zorder=5)

    true_omega = result.omega_rabi
    ax_main.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax_main.set_ylabel("P(|1⟩)", fontsize=12)
    ax_main.set_ylim(-0.05, 1.12)
    ax_main.legend(fontsize=9, loc="upper right")
    ax_main.set_title(
        f"Rabi Oscillation Calibration\n"
        f"True ω = {true_omega:.4f} rad/µs  |  "
        f"Fitted ω = {fit.omega_rabi_fit:.4f} rad/µs  |  "
        f"Error = {abs(fit.omega_rabi_fit - true_omega) / true_omega * 100:.2f}%",
        fontsize=11,
    )
    ax_main.tick_params(labelbottom=False)

    # --- Residuals panel ---
    ax_res.scatter(times, fit.residuals * 100, s=5, color="#9C27B0", alpha=0.7)
    ax_res.axhline(0, color="black", lw=0.8)
    ax_res.set_ylabel("Residual\n(%)", fontsize=10)
    ax_res.set_xlabel("Time (µs)", fontsize=12)
    ax_res.set_ylim(-8, 8)

    fig.text(0.01, 0.01,
             f"χ² = {fit.chi_squared:.5f}  |  "
             f"Decay τ = {fit.decay_time:.1f} µs",
             fontsize=8, color="gray")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig