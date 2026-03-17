"""
run_step2.py
------------
Demonstrates Step 2: T1/T2 decoherence via the Lindblad master equation.

Shows two regimes side by side:
    - Realistic:   T1=50us, T2=30us  (similar to superconducting qubits)
    - Exaggerated: T1=5us,  T2=3us   (strong decay, clear visual effect)

For each regime:
    1. Simulate decaying Rabi oscillations with QuTiP (Lindblad)
    2. Add synthetic measurement noise
    3. Fit to extract omega_rabi, T2, and uncertainties
    4. Compare fitted T2 against true T2
    5. Plot results side by side
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

from rabi.simulator import add_measurement_noise
from rabi.fitting import fit_rabi, rabi_model
from rabi.t1t2_noise import (
    DecoherenceParams,
    run_rabi_with_decoherence,
    theoretical_decay_envelope,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

OMEGA_RABI = 2.0    # rad/us, same as step 1
T_MAX      = 40.0   # us, longer window to see decay clearly
N_POINTS   = 400
NOISE      = 0.02
SEED       = 42

REGIMES = {
    "Realistic (T1=50µs, T2=30µs)": DecoherenceParams(T1=50.0, T2=30.0),
    "Exaggerated (T1=5µs, T2=3µs)": DecoherenceParams(T1=5.0,  T2=3.0),
}

# ---------------------------------------------------------------------------
# Simulate, fit, and report for each regime
# ---------------------------------------------------------------------------

results = {}

print("=" * 60)
print("  Qubit Calibration Simulator, Step 2: T1/T2 Decoherence")
print("=" * 60)
print(f"\nTrue Rabi frequency: ω = {OMEGA_RABI:.4f} rad/µs")

for label, decoherence in REGIMES.items():
    print(f"\n--- {label} ---")
    print(f"True T1 = {decoherence.T1:.1f} µs  |  True T2 = {decoherence.T2:.1f} µs")

    # Step 1: Simulate with Lindblad decoherence
    result = run_rabi_with_decoherence(
        omega_rabi=OMEGA_RABI,
        decoherence=decoherence,
        t_max=T_MAX,
        n_points=N_POINTS,
    )
    print(f"[✓] Lindblad simulation complete")

    # Step 2: Add measurement noise
    noisy = add_measurement_noise(result.excited_pop, noise_level=NOISE, seed=SEED)

    # Step 3: Fit, extract omega, T2, amplitude, offset
    fit = fit_rabi(result.times, noisy)
    print(f"[✓] Fit complete")

    # Step 4: Report
    omega_err_pct = abs(fit.omega_rabi_fit - OMEGA_RABI) / OMEGA_RABI * 100
    T2_err_pct    = abs(fit.decay_time - decoherence.T2) / decoherence.T2 * 100

    print(f"\n  Calibration Results:")
    print(f"  ω  true:   {OMEGA_RABI:.6f} rad/µs")
    print(f"  ω  fitted: {fit.omega_rabi_fit:.6f} ± {fit.omega_rabi_err:.6f} rad/µs  "
          f"(error: {omega_err_pct:.3f}%)")
    print(f"  T2 true:   {decoherence.T2:.4f} µs")
    print(f"  T2 fitted: {fit.decay_time:.4f} µs  "
          f"(error: {T2_err_pct:.3f}%)")
    print(f"  χ²:        {fit.chi_squared:.6f}")

    results[label] = {
        "result":      result,
        "noisy":       noisy,
        "fit":         fit,
        "decoherence": decoherence,
    }

# ---------------------------------------------------------------------------
# Plot: side-by-side comparison
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(16, 8))
gs  = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.08, wspace=0.3)

for col, (label, data) in enumerate(results.items()):
    result      = data["result"]
    noisy       = data["noisy"]
    fit         = data["fit"]
    decoherence = data["decoherence"]
    times       = result.times

    fitted_curve = rabi_model(
        times, fit.omega_rabi_fit, fit.amplitude,
        fit.decay_time, fit.offset, 0.0,
    )
    envelope = theoretical_decay_envelope(times, decoherence)

    ax_main = fig.add_subplot(gs[0, col])
    ax_res  = fig.add_subplot(gs[1, col], sharex=ax_main)

    # --- Main panel ---
    ax_main.plot(times, result.excited_pop, color="#2196F3", lw=1.5,
                 label="QuTiP simulation", zorder=2)
    ax_main.scatter(times, noisy, s=5, color="#FF9800", alpha=0.5,
                    label="Simulated measurements", zorder=3)
    ax_main.plot(times, fitted_curve, color="#F44336", lw=2.0, ls="-.",
                 label=f"Fit: ω={fit.omega_rabi_fit:.4f}, T2={fit.decay_time:.2f}µs",
                 zorder=4)
    ax_main.plot(times, envelope, color="#9C27B0", lw=1.2, ls="--",
                 alpha=0.7, label=f"True T2 envelope ({decoherence.T2}µs)", zorder=1)
    ax_main.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.5)

    ax_main.set_ylabel("P(|1⟩)", fontsize=11)
    ax_main.set_ylim(-0.05, 1.12)
    ax_main.legend(fontsize=8, loc="upper right")
    ax_main.set_title(
        f"{label}\n"
        f"ω error: {abs(fit.omega_rabi_fit - OMEGA_RABI)/OMEGA_RABI*100:.3f}%  |  "
        f"T2 error: {abs(fit.decay_time - decoherence.T2)/decoherence.T2*100:.3f}%",
        fontsize=10,
    )
    ax_main.tick_params(labelbottom=False)

    # --- Residuals panel ---
    ax_res.scatter(times, fit.residuals * 100, s=4, color="#9C27B0", alpha=0.6)
    ax_res.axhline(0, color="black", lw=0.8)
    ax_res.set_ylabel("Residual\n(%)", fontsize=9)
    ax_res.set_xlabel("Time (µs)", fontsize=11)
    ax_res.set_ylim(-8, 8)

fig.suptitle(
    f"Rabi Oscillation Calibration with T1/T2 Decoherence\n"
    f"True ω = {OMEGA_RABI} rad/µs",
    fontsize=13, y=0.98
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("rabi_decoherence.png", dpi=150, bbox_inches="tight")

plt.savefig("rabi_decoherence.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\n[✓] Plot saved to rabi_decoherence.png")
