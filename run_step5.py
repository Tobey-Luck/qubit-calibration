"""
run_step5.py
------------
Demonstrates Step 5: Qiskit Aer noise model integration.

Compares three simulation approaches for Rabi and Ramsey experiments:
    1. QuTiP (Lindblad master equation) - our physics simulation
    2. Qiskit Aer + custom noise model (T1/T2 from our parameters)
    3. Qiskit Aer + fake IBM backend (realistic device noise)

This bridges the gap between clean simulation and real hardware,
showing how calibration workflows translate to actual quantum devices.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))

from rabi.simulator import run_rabi, add_measurement_noise
from rabi.fitting import fit_rabi
from rabi.t1t2_noise import DecoherenceParams, run_rabi_with_decoherence
from rabi.ramsey import run_ramsey, fit_ramsey

try:
    from rabi.qiskit_backend import (
        build_custom_noise_model,
        build_fake_backend,
        run_rabi_circuit,
        run_ramsey_circuit,
        QISKIT_AVAILABLE,
    )
except ImportError:
    QISKIT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

OMEGA_RABI = 2.0
T1, T2     = 50.0, 30.0
NOISE      = 0.02
SEED       = 42
SHOTS      = 2048
N_POINTS   = 50   # fewer points for Qiskit (each is a separate circuit run)

print("=" * 60)
print("  Qubit Calibration Simulator - Step 5: Qiskit Integration")
print("=" * 60)

if not QISKIT_AVAILABLE:
    print("\n[ERROR] Qiskit not available. Install with:")
    print("  pip install qiskit qiskit-aer qiskit-ibm-runtime")
    sys.exit(1)

decoherence = DecoherenceParams(T1=T1, T2=T2)

# ---------------------------------------------------------------------------
# QuTiP reference simulations
# ---------------------------------------------------------------------------

print("\n--- QuTiP reference ---")

qutip_rabi = run_rabi_with_decoherence(
    omega_rabi=OMEGA_RABI, decoherence=decoherence,
    t_max=10.0, n_points=200,
)
print(f"  [OK] Rabi simulation complete")

qutip_ramsey = run_ramsey(
    omega_rabi=OMEGA_RABI, decoherence=decoherence,
    delta=0.3, tau_max=40.0, n_tau=150,
)
print(f"  [OK] Ramsey simulation complete")

# ---------------------------------------------------------------------------
# Qiskit Aer: custom noise model
# ---------------------------------------------------------------------------

print("\n--- Qiskit Aer: custom noise model ---")

custom_nm = build_custom_noise_model(T1=T1, T2=T2)
print(f"  [OK] Custom noise model built (T1={T1}us, T2={T2}us)")

print(f"  Running Rabi circuit ({N_POINTS} points, {SHOTS} shots each)...")
qiskit_rabi_custom = run_rabi_circuit(
    omega_rabi=OMEGA_RABI, t_max=10.0, n_points=N_POINTS,
    noise_model=custom_nm, shots=SHOTS, noise_model_name="Custom (T1/T2)"
)
print(f"  [OK] Rabi circuit complete")

print(f"  Running Ramsey circuit ({N_POINTS} points, {SHOTS} shots each)...")
qiskit_ramsey_custom = run_ramsey_circuit(
    delta=0.3, tau_max=40.0, n_tau=N_POINTS,
    T1=T1, T2=T2,
    noise_model=custom_nm, shots=SHOTS, noise_model_name="Custom (T1/T2)"
)
print(f"  [OK] Ramsey circuit complete")

# ---------------------------------------------------------------------------
# Qiskit Aer: fake IBM backend
# ---------------------------------------------------------------------------

print("\n--- Qiskit Aer: fake IBM backend ---")

qiskit_rabi_fake    = None
qiskit_ramsey_fake  = None
fake_backend_name   = "N/A"

try:
    fake_sim, fake_backend_name = build_fake_backend()
    print(f"  [OK] Loaded fake backend: {fake_backend_name}")

    print(f"  Running Rabi circuit on fake backend...")
    qiskit_rabi_fake = run_rabi_circuit(
        omega_rabi=OMEGA_RABI, t_max=10.0, n_points=N_POINTS,
        simulator=fake_sim, shots=SHOTS, noise_model_name=fake_backend_name
    )
    print(f"  [OK] Rabi circuit complete")

    print(f"  Running Ramsey circuit on fake backend...")
    qiskit_ramsey_fake = run_ramsey_circuit(
        delta=0.3, tau_max=40.0, n_tau=N_POINTS,
        T1=T1, T2=T2,
        simulator=fake_sim, shots=SHOTS, noise_model_name=fake_backend_name
    )
    print(f"  [OK] Ramsey circuit complete")

except Exception as e:
    print(f"  [WARN] Fake backend unavailable: {e}")
    print(f"         Install with: pip install qiskit-ibm-runtime")

# ---------------------------------------------------------------------------
# Fit Qiskit results
# ---------------------------------------------------------------------------

print("\n--- Fitting Qiskit results ---")

rabi_fit_custom = fit_rabi(qiskit_rabi_custom.times, qiskit_rabi_custom.excited_pop)
rabi_err_custom = abs(rabi_fit_custom.omega_rabi_fit - OMEGA_RABI) / OMEGA_RABI * 100
print(f"  [OK] Custom noise Rabi fit: omega={rabi_fit_custom.omega_rabi_fit:.4f} "
      f"(error={rabi_err_custom:.2f}%)")

ramsey_fit_custom = fit_ramsey(qiskit_ramsey_custom.times, qiskit_ramsey_custom.excited_pop)
ramsey_err_custom = abs(ramsey_fit_custom.T2_fit - T2) / T2 * 100
print(f"  [OK] Custom noise Ramsey fit: T2={ramsey_fit_custom.T2_fit:.4f}us "
      f"(error={ramsey_err_custom:.2f}%)")

if qiskit_rabi_fake is not None:
    rabi_fit_fake   = fit_rabi(qiskit_rabi_fake.times, qiskit_rabi_fake.excited_pop)
    rabi_err_fake   = abs(rabi_fit_fake.omega_rabi_fit - OMEGA_RABI) / OMEGA_RABI * 100
    ramsey_fit_fake = fit_ramsey(qiskit_ramsey_fake.times, qiskit_ramsey_fake.excited_pop)
    ramsey_err_fake = abs(ramsey_fit_fake.T2_fit - T2) / T2 * 100
    print(f"  [OK] Fake backend Rabi fit: omega={rabi_fit_fake.omega_rabi_fit:.4f} "
          f"(error={rabi_err_fake:.2f}%)")
    print(f"  [OK] Fake backend Ramsey fit: T2={ramsey_fit_fake.T2_fit:.4f}us "
          f"(error={ramsey_err_fake:.2f}%)")

print("\n[OK] All experiments complete. Generating plot...")

# ---------------------------------------------------------------------------
# Plot: 2 rows (Rabi, Ramsey) x 2 columns (custom noise, fake backend)
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

def plot_comparison(ax, qutip_times, qutip_pop, qiskit_result,
                    fit_result, true_param, param_name, units,
                    x_label, title):
    """Plot QuTiP vs Qiskit comparison with fit."""
    ax.plot(qutip_times, qutip_pop, color="#2196F3", lw=1.5,
            label="QuTiP (Lindblad)", zorder=2, alpha=0.8)
    ax.scatter(qiskit_result.times, qiskit_result.excited_pop,
               s=8, color="#F44336", alpha=0.7, zorder=3,
               label=f"Qiskit Aer ({qiskit_result.noise_model_name})")

    if fit_result is not None:
        from rabi.fitting import rabi_model
        fitted = rabi_model(qiskit_result.times, fit_result.omega_rabi_fit,
                           fit_result.amplitude, fit_result.decay_time,
                           fit_result.offset, 0.0) if param_name == "omega" else None
        if fitted is not None:
            ax.plot(qiskit_result.times, fitted, color="#FF9800", lw=2, ls="-.",
                    label=f"Fit: {param_name}={fit_result.omega_rabi_fit:.4f}{units}")

    ax.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("P(|1>)", fontsize=11)
    ax.set_ylim(-0.05, 1.12)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)

# Row 0: Rabi experiments
ax00 = fig.add_subplot(gs[0, 0])
ax00.plot(qutip_rabi.times, qutip_rabi.excited_pop, color="#2196F3", lw=1.5,
          label="QuTiP (Lindblad)", zorder=2, alpha=0.8)
ax00.scatter(qiskit_rabi_custom.times, qiskit_rabi_custom.excited_pop,
             s=8, color="#F44336", alpha=0.7, zorder=3,
             label=f"Qiskit (custom noise, {SHOTS} shots)")
from rabi.fitting import rabi_model as _rabi_model
fitted_custom = _rabi_model(qiskit_rabi_custom.times, rabi_fit_custom.omega_rabi_fit,
                             rabi_fit_custom.amplitude, rabi_fit_custom.decay_time,
                             rabi_fit_custom.offset, 0.0)
ax00.plot(qiskit_rabi_custom.times, fitted_custom, color="#FF9800", lw=2, ls="-.",
          label=f"Fit: omega={rabi_fit_custom.omega_rabi_fit:.4f} (err={rabi_err_custom:.2f}%)")
ax00.set_xlabel("Time (us)", fontsize=11)
ax00.set_ylabel("P(|1>)", fontsize=11)
ax00.set_ylim(-0.05, 1.12)
ax00.legend(fontsize=8)
ax00.set_title(f"Rabi: QuTiP vs Qiskit custom noise\nT1={T1}us, T2={T2}us", fontsize=10)

ax01 = fig.add_subplot(gs[0, 1])
if qiskit_rabi_fake is not None:
    ax01.plot(qutip_rabi.times, qutip_rabi.excited_pop, color="#2196F3", lw=1.5,
              label="QuTiP (Lindblad)", alpha=0.8)
    ax01.scatter(qiskit_rabi_fake.times, qiskit_rabi_fake.excited_pop,
                 s=8, color="#9C27B0", alpha=0.7,
                 label=f"Qiskit ({fake_backend_name}, {SHOTS} shots)")
    fitted_fake = _rabi_model(qiskit_rabi_fake.times, rabi_fit_fake.omega_rabi_fit,
                               rabi_fit_fake.amplitude, rabi_fit_fake.decay_time,
                               rabi_fit_fake.offset, 0.0)
    ax01.plot(qiskit_rabi_fake.times, fitted_fake, color="#FF9800", lw=2, ls="-.",
              label=f"Fit: omega={rabi_fit_fake.omega_rabi_fit:.4f} (err={rabi_err_fake:.2f}%)")
    ax01.set_ylim(-0.05, 1.12)
    ax01.legend(fontsize=8)
    ax01.set_title(f"Rabi: QuTiP vs Qiskit {fake_backend_name}", fontsize=10)
else:
    ax01.text(0.5, 0.5, f"Fake backend\nnot available\n\npip install\nqiskit-ibm-runtime",
              ha="center", va="center", transform=ax01.transAxes, fontsize=12,
              color="gray")
    ax01.set_title("Rabi: Fake IBM backend", fontsize=10)
ax01.set_xlabel("Time (us)", fontsize=11)
ax01.set_ylabel("P(|1>)", fontsize=11)

# Row 1: Ramsey experiments
from rabi.ramsey import ramsey_model as _ramsey_model
ax10 = fig.add_subplot(gs[1, 0])
ax10.plot(qutip_ramsey.tau_times, qutip_ramsey.excited_pop, color="#2196F3", lw=1.5,
          label="QuTiP (analytical)", alpha=0.8)
ax10.scatter(qiskit_ramsey_custom.times, qiskit_ramsey_custom.excited_pop,
             s=8, color="#F44336", alpha=0.7,
             label=f"Qiskit (custom noise, {SHOTS} shots)")
fitted_ramsey_custom = _ramsey_model(qiskit_ramsey_custom.times,
                                      ramsey_fit_custom.T2_fit, ramsey_fit_custom.delta_fit,
                                      ramsey_fit_custom.amplitude, ramsey_fit_custom.offset, 0.0)
ax10.plot(qiskit_ramsey_custom.times, fitted_ramsey_custom, color="#FF9800", lw=2, ls="-.",
          label=f"Fit: T2={ramsey_fit_custom.T2_fit:.2f}us (err={ramsey_err_custom:.2f}%)")
ax10.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.5)
ax10.set_xlabel("Free precession time tau (us)", fontsize=11)
ax10.set_ylabel("P(|1>)", fontsize=11)
ax10.set_ylim(-0.05, 1.12)
ax10.legend(fontsize=8)
ax10.set_title(f"Ramsey: QuTiP vs Qiskit custom noise\nT1={T1}us, T2={T2}us", fontsize=10)

ax11 = fig.add_subplot(gs[1, 1])
if qiskit_ramsey_fake is not None:
    ax11.plot(qutip_ramsey.tau_times, qutip_ramsey.excited_pop, color="#2196F3", lw=1.5,
              label="QuTiP (analytical)", alpha=0.8)
    ax11.scatter(qiskit_ramsey_fake.times, qiskit_ramsey_fake.excited_pop,
                 s=8, color="#9C27B0", alpha=0.7,
                 label=f"Qiskit ({fake_backend_name}, {SHOTS} shots)")
    fitted_ramsey_fake = _ramsey_model(qiskit_ramsey_fake.times,
                                        ramsey_fit_fake.T2_fit, ramsey_fit_fake.delta_fit,
                                        ramsey_fit_fake.amplitude, ramsey_fit_fake.offset, 0.0)
    ax11.plot(qiskit_ramsey_fake.times, fitted_ramsey_fake, color="#FF9800", lw=2, ls="-.",
              label=f"Fit: T2={ramsey_fit_fake.T2_fit:.2f}us (err={ramsey_err_fake:.2f}%)")
    ax11.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax11.set_ylim(-0.05, 1.12)
    ax11.legend(fontsize=8)
    ax11.set_title(f"Ramsey: QuTiP vs Qiskit {fake_backend_name}", fontsize=10)
else:
    ax11.text(0.5, 0.5, f"Fake backend\nnot available\n\npip install\nqiskit-ibm-runtime",
              ha="center", va="center", transform=ax11.transAxes, fontsize=12,
              color="gray")
    ax11.set_title("Ramsey: Fake IBM backend", fontsize=10)
ax11.set_xlabel("Free precession time tau (us)", fontsize=11)
ax11.set_ylabel("P(|1>)", fontsize=11)

fig.suptitle(
    "Qiskit Aer Integration: QuTiP vs Hardware-Realistic Noise Models\n"
    f"omega_rabi={OMEGA_RABI} rad/us, T1={T1}us, T2={T2}us, {SHOTS} shots",
    fontsize=13, y=0.98,
)
plt.savefig("qiskit_comparison.png", dpi=150, bbox_inches="tight")
print("[OK] Plot saved to qiskit_comparison.png")
