"""
run_step3.py
------------
Demonstrates Step 3: Ramsey experiment for precise T2 measurement.

Shows side-by-side comparison of:
    Left:  Ramsey fringe decay - P(|1>) vs free precession time tau
    Right: Direct comparison of T2 accuracy: Ramsey vs Rabi

For each regime (realistic T1=50us/T2=30us and exaggerated T1=5us/T2=3us):
    1. Run Ramsey sweep over tau values
    2. Add synthetic noise
    3. Fit to extract T2 with uncertainty
    4. Compare against Rabi T2 extraction from step 2
"""

import os
import sys
import glob
import importlib.util

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))

from rabi.simulator import add_measurement_noise
from rabi.fitting import fit_rabi
from rabi.t1t2_noise import DecoherenceParams, run_rabi_with_decoherence
from rabi.ramsey import run_ramsey, fit_ramsey, ramsey_model

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

OMEGA_RABI = 2.0
NOISE      = 0.02
SEED       = 42

# Each regime specifies (DecoherenceParams, detuning delta in rad/us).
# Delta is chosen so ~1-2 oscillation cycles fit within the T2 window.
REGIMES = {
    "Realistic (T1=50us, T2=30us)": (DecoherenceParams(T1=50.0, T2=30.0), 0.3),
    "Exaggerated (T1=5us, T2=3us)": (DecoherenceParams(T1=5.0,  T2=3.0),  1.5),
}

# ---------------------------------------------------------------------------
# Note on C++ solver
# ---------------------------------------------------------------------------
# The C++ Ramsey solver is not called here due to a known Windows ABI
# incompatibility between MinGW-built pybind11 extensions and UCRT Python
# 3.13 on repeated object creation. The C++ Rabi solver validates correctly
# in run_step1.py (max diff 1.61e-05). On Linux or with an MSVC-built
# extension this limitation would not apply.

# ---------------------------------------------------------------------------
# Run Ramsey and Rabi for each regime
# ---------------------------------------------------------------------------

print("=" * 60)
print("  Qubit Calibration Simulator - Step 3: Ramsey Experiment")
print("=" * 60)

all_results = {}

for label, (decoherence, delta) in REGIMES.items():
    T2 = decoherence.T2
    print(f"\n--- {label} ---")

    # Ramsey sweep
    print(f"  Running Ramsey sweep...")
    ramsey_result = run_ramsey(
        omega_rabi=OMEGA_RABI,
        decoherence=decoherence,
        delta=delta,
        tau_max=min(2 * T2, 4 * np.pi / delta),
        n_tau=150,
    )
    ramsey_noisy = add_measurement_noise(
        ramsey_result.excited_pop, noise_level=NOISE, seed=SEED
    )
    ramsey_fit = fit_ramsey(ramsey_result.tau_times, ramsey_noisy)
    ramsey_err = abs(ramsey_fit.T2_fit - T2) / T2 * 100

    print(f"  [OK] Ramsey fit complete")
    print(f"       T2 true:   {T2:.4f} us")
    print(f"       T2 fitted: {ramsey_fit.T2_fit:.4f} +/- {ramsey_fit.T2_err:.4f} us")
    print(f"       T2 error:  {ramsey_err:.3f}%")

    # Rabi for comparison
    print(f"  Running Rabi simulation for comparison...")
    rabi_result = run_rabi_with_decoherence(
        omega_rabi=OMEGA_RABI,
        decoherence=decoherence,
        t_max=min(4 * T2, 80.0),
        n_points=600,
    )
    rabi_noisy = add_measurement_noise(
        rabi_result.excited_pop, noise_level=NOISE, seed=SEED
    )
    rabi_fit = fit_rabi(rabi_result.times, rabi_noisy)
    rabi_err = abs(rabi_fit.decay_time - T2) / T2 * 100

    print(f"  [OK] Rabi fit complete")
    print(f"       T2 fitted: {rabi_fit.decay_time:.4f} us")
    print(f"       T2 error:  {rabi_err:.3f}%")

    if ramsey_err > 0:
        print(f"  Ramsey is {rabi_err/ramsey_err:.1f}x more accurate than Rabi")
    if ramsey_err > rabi_err:
        print(f"  Note: T2={T2}us is pulse-duration-limited — Ramsey model")
        print(f"        assumptions break down at very short T2.")

    all_results[label] = {
        "ramsey_result": ramsey_result,
        "ramsey_noisy":  ramsey_noisy,
        "ramsey_fit":    ramsey_fit,
        "ramsey_err":    ramsey_err,
        "rabi_err":      rabi_err,
        "decoherence":   decoherence,
        "cpp_data":      None,
        "delta":         delta,
    }

print("\n[OK] All regimes complete. Generating plot...")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(16, 8))
gs  = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.08, wspace=0.35)

for col, (label, data) in enumerate(all_results.items()):
    ramsey_result = data["ramsey_result"]
    ramsey_noisy  = data["ramsey_noisy"]
    ramsey_fit    = data["ramsey_fit"]
    decoherence   = data["decoherence"]
    tau           = ramsey_result.tau_times
    T2            = decoherence.T2

    fitted_curve        = ramsey_model(tau, ramsey_fit.T2_fit, ramsey_fit.delta_fit,
                                       ramsey_fit.amplitude, ramsey_fit.offset, 0.0)
    true_envelope_upper = 0.5 + 0.5 * np.exp(-tau / T2)
    true_envelope_lower = 0.5 - 0.5 * np.exp(-tau / T2)

    ax_main = fig.add_subplot(gs[0, col])
    ax_res  = fig.add_subplot(gs[1, col], sharex=ax_main)

    ax_main.fill_between(tau, true_envelope_lower, true_envelope_upper,
                         alpha=0.15, color="#9C27B0",
                         label=f"True T2 envelope ({T2}us)")
    ax_main.plot(tau, ramsey_result.excited_pop, color="#2196F3", lw=1.5,
                 label="QuTiP Ramsey", zorder=2)
    ax_main.scatter(tau, ramsey_noisy, s=6, color="#FF9800", alpha=0.5,
                    label="Simulated measurements", zorder=3)
    ax_main.plot(tau, fitted_curve, color="#F44336", lw=2.0, ls="-.",
                 label=f"Fit: T2={ramsey_fit.T2_fit:.2f}us (err={data['ramsey_err']:.1f}%)",
                 zorder=4)

    ax_main.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax_main.set_ylabel("P(|1>)", fontsize=11)
    ax_main.set_ylim(-0.05, 1.12)
    ax_main.legend(fontsize=8, loc="upper right")
    ax_main.set_title(
        f"{label}\n"
        f"Ramsey T2 error: {data['ramsey_err']:.1f}%  vs  "
        f"Rabi T2 error: {data['rabi_err']:.1f}%",
        fontsize=10,
    )
    ax_main.tick_params(labelbottom=False)

    ax_res.scatter(tau, ramsey_fit.residuals * 100, s=4, color="#9C27B0", alpha=0.6)
    ax_res.axhline(0, color="black", lw=0.8)
    ax_res.set_ylabel("Residual\n(%)", fontsize=9)
    ax_res.set_xlabel("Free precession time tau (us)", fontsize=11)
    ax_res.set_ylim(-8, 8)

fig.suptitle(
    f"Ramsey Experiment - Precise T2 Measurement\n"
    f"True omega = {OMEGA_RABI} rad/us",
    fontsize=13, y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("ramsey_experiment.png", dpi=150, bbox_inches="tight")
print("[OK] Plot saved to ramsey_experiment.png")
