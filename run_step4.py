"""
run_step4.py
------------
Demonstrates Step 4: Bayesian parameter estimation via MCMC.

Compares curve_fit (point estimate) against full Bayesian posterior for:
    1. Rabi omega estimation (ideal data)
    2. Ramsey T2 estimation
    3. Joint omega + T2 estimation from decohered Rabi data

For each case, shows side-by-side:
    Left:  Data + curve_fit result with 1-sigma error bar
    Right: Posterior distribution (histogram) with median and credible intervals
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
from rabi.fitting import fit_rabi, rabi_model
from rabi.t1t2_noise import DecoherenceParams, run_rabi_with_decoherence
from rabi.ramsey import run_ramsey, fit_ramsey, ramsey_model
from rabi.bayesian import run_mcmc_rabi, run_mcmc_ramsey, run_mcmc_joint

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

OMEGA_RABI = 2.0
T1, T2     = 50.0, 30.0
NOISE      = 0.02
SEED       = 42
N_WALKERS  = 32
N_BURN     = 500
N_STEPS    = 2000

print("=" * 60)
print("  Qubit Calibration Simulator - Step 4: Bayesian Estimation")
print("=" * 60)

# ---------------------------------------------------------------------------
# Case 1: Rabi omega estimation
# ---------------------------------------------------------------------------

print("\n--- Case 1: Rabi omega estimation ---")

rabi_result = run_rabi(omega_rabi=OMEGA_RABI, t_max=10.0, n_points=200)
rabi_noisy  = add_measurement_noise(rabi_result.excited_pop, noise_level=NOISE, seed=SEED)
rabi_fit    = fit_rabi(rabi_result.times, rabi_noisy)

print(f"  curve_fit: omega = {rabi_fit.omega_rabi_fit:.6f} +/- {rabi_fit.omega_rabi_err:.6f} rad/us")
print(f"  Running MCMC ({N_WALKERS} walkers, {N_BURN} burn-in, {N_STEPS} steps)...")

mcmc_rabi = run_mcmc_rabi(
    rabi_result.times, rabi_noisy,
    sigma=NOISE, n_walkers=N_WALKERS, n_burn=N_BURN, n_steps=N_STEPS, seed=SEED
)
omega_samples = mcmc_rabi.samples[:, 0]
print(f"  Bayesian:  omega = {mcmc_rabi.medians[0]:.6f} "
      f"[{mcmc_rabi.lower_1sigma[0]:.6f}, {mcmc_rabi.upper_1sigma[0]:.6f}] rad/us (68% CI)")
print(f"  Acceptance fraction: {mcmc_rabi.acceptance_fraction:.3f}")

# ---------------------------------------------------------------------------
# Case 2: Ramsey T2 estimation
# ---------------------------------------------------------------------------

print("\n--- Case 2: Ramsey T2 estimation ---")

decoherence   = DecoherenceParams(T1=T1, T2=T2)
ramsey_result = run_ramsey(omega_rabi=OMEGA_RABI, decoherence=decoherence,
                           delta=0.3, tau_max=60.0, n_tau=150)
ramsey_noisy  = add_measurement_noise(ramsey_result.excited_pop, noise_level=NOISE, seed=SEED)
ramsey_fit    = fit_ramsey(ramsey_result.tau_times, ramsey_noisy)

print(f"  curve_fit: T2 = {ramsey_fit.T2_fit:.4f} +/- {ramsey_fit.T2_err:.4f} us")
print(f"  Running MCMC...")

mcmc_ramsey = run_mcmc_ramsey(
    ramsey_result.tau_times, ramsey_noisy,
    sigma=NOISE, n_walkers=N_WALKERS, n_burn=N_BURN, n_steps=N_STEPS, seed=SEED
)
T2_samples = mcmc_ramsey.samples[:, 0]
print(f"  Bayesian:  T2 = {mcmc_ramsey.medians[0]:.4f} "
      f"[{mcmc_ramsey.lower_1sigma[0]:.4f}, {mcmc_ramsey.upper_1sigma[0]:.4f}] us (68% CI)")
print(f"  Acceptance fraction: {mcmc_ramsey.acceptance_fraction:.3f}")

# ---------------------------------------------------------------------------
# Case 3: Joint omega + T2 estimation
# ---------------------------------------------------------------------------

print("\n--- Case 3: Joint omega + T2 from decohered Rabi ---")

joint_result = run_rabi_with_decoherence(
    omega_rabi=OMEGA_RABI, decoherence=decoherence, t_max=60.0, n_points=300
)
joint_noisy = add_measurement_noise(joint_result.excited_pop, noise_level=NOISE, seed=SEED)
joint_fit   = fit_rabi(joint_result.times, joint_noisy)

print(f"  curve_fit: omega = {joint_fit.omega_rabi_fit:.4f} rad/us, "
      f"T2 = {joint_fit.decay_time:.4f} us")
print(f"  Running MCMC...")

mcmc_joint = run_mcmc_joint(
    joint_result.times, joint_noisy,
    sigma=NOISE, n_walkers=N_WALKERS, n_burn=N_BURN, n_steps=N_STEPS, seed=SEED
)
joint_omega_samples = mcmc_joint.samples[:, 0]
joint_T2_samples    = mcmc_joint.samples[:, 1]
print(f"  Bayesian:  omega = {mcmc_joint.medians[0]:.4f} "
      f"[{mcmc_joint.lower_1sigma[0]:.4f}, {mcmc_joint.upper_1sigma[0]:.4f}] rad/us")
print(f"  Bayesian:  T2    = {mcmc_joint.medians[1]:.4f} "
      f"[{mcmc_joint.lower_1sigma[1]:.4f}, {mcmc_joint.upper_1sigma[1]:.4f}] us")
print(f"  Acceptance fraction: {mcmc_joint.acceptance_fraction:.3f}")

print("\n[OK] All MCMC runs complete. Generating plots...")

# ---------------------------------------------------------------------------
# Plot: 3 rows x 2 columns (data+fit | posterior histogram)
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(14, 14))
gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

# Helper: posterior histogram with median and credible intervals
def plot_posterior(ax, samples, true_val, cf_val, cf_err, param_name, units):
    ax.hist(samples, bins=60, density=True, color="#2196F3", alpha=0.7,
            label="Posterior")
    ax.axvline(true_val,  color="#4CAF50", lw=2.0, ls="-",
               label=f"True: {true_val:.4f}")
    ax.axvline(cf_val,    color="#FF9800", lw=2.0, ls="--",
               label=f"curve_fit: {cf_val:.4f} +/- {cf_err:.4f}")
    ax.axvline(np.median(samples), color="#F44336", lw=2.0, ls="-.",
               label=f"Bayesian median: {np.median(samples):.4f}")
    lo = np.percentile(samples, 16)
    hi = np.percentile(samples, 84)
    ax.axvspan(lo, hi, alpha=0.15, color="#F44336", label=f"68% CI: [{lo:.4f}, {hi:.4f}]")
    ax.set_xlabel(f"{param_name} ({units})", fontsize=11)
    ax.set_ylabel("Posterior density", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")


# --- Row 0: Rabi omega ---
ax00 = fig.add_subplot(gs[0, 0])
ax01 = fig.add_subplot(gs[0, 1])

fitted_rabi = rabi_model(rabi_result.times, rabi_fit.omega_rabi_fit,
                         rabi_fit.amplitude, rabi_fit.decay_time,
                         rabi_fit.offset, 0.0)
ax00.plot(rabi_result.times, rabi_result.excited_pop, color="#2196F3", lw=1.5,
          label="QuTiP simulation")
ax00.scatter(rabi_result.times, rabi_noisy, s=5, color="#FF9800", alpha=0.5,
             label="Measurements")
ax00.plot(rabi_result.times, fitted_rabi, color="#F44336", lw=2, ls="-.",
          label=f"curve_fit: omega={rabi_fit.omega_rabi_fit:.4f}")
ax00.set_xlabel("Time (us)", fontsize=11)
ax00.set_ylabel("P(|1>)", fontsize=11)
ax00.legend(fontsize=8)
ax00.set_title("Case 1: Rabi oscillation\ncurve_fit result", fontsize=10)

plot_posterior(ax01, omega_samples, OMEGA_RABI,
               rabi_fit.omega_rabi_fit, rabi_fit.omega_rabi_err,
               "omega_rabi", "rad/us")
ax01.set_title("Case 1: Rabi omega\nBayesian posterior", fontsize=10)

# --- Row 1: Ramsey T2 ---
ax10 = fig.add_subplot(gs[1, 0])
ax11 = fig.add_subplot(gs[1, 1])

fitted_ramsey = ramsey_model(ramsey_result.tau_times, ramsey_fit.T2_fit,
                              ramsey_fit.delta_fit, ramsey_fit.amplitude,
                              ramsey_fit.offset, 0.0)
ax10.plot(ramsey_result.tau_times, ramsey_result.excited_pop, color="#2196F3",
          lw=1.5, label="QuTiP Ramsey")
ax10.scatter(ramsey_result.tau_times, ramsey_noisy, s=5, color="#FF9800",
             alpha=0.5, label="Measurements")
ax10.plot(ramsey_result.tau_times, fitted_ramsey, color="#F44336", lw=2, ls="-.",
          label=f"curve_fit: T2={ramsey_fit.T2_fit:.2f}us")
ax10.set_xlabel("Free precession time tau (us)", fontsize=11)
ax10.set_ylabel("P(|1>)", fontsize=11)
ax10.legend(fontsize=8)
ax10.set_title("Case 2: Ramsey experiment\ncurve_fit result", fontsize=10)

plot_posterior(ax11, T2_samples, T2,
               ramsey_fit.T2_fit, ramsey_fit.T2_err,
               "T2", "us")
ax11.set_title("Case 2: Ramsey T2\nBayesian posterior", fontsize=10)

# --- Row 2: Joint omega + T2 (corner-style 2D scatter) ---
ax20 = fig.add_subplot(gs[2, 0])
ax21 = fig.add_subplot(gs[2, 1])

fitted_joint = rabi_model(joint_result.times, joint_fit.omega_rabi_fit,
                          joint_fit.amplitude, joint_fit.decay_time,
                          joint_fit.offset, 0.0)
ax20.plot(joint_result.times, joint_result.excited_pop, color="#2196F3",
          lw=1.5, label="QuTiP simulation")
ax20.scatter(joint_result.times, joint_noisy, s=3, color="#FF9800",
             alpha=0.4, label="Measurements")
ax20.plot(joint_result.times, fitted_joint, color="#F44336", lw=2, ls="-.",
          label=f"curve_fit: omega={joint_fit.omega_rabi_fit:.3f}, T2={joint_fit.decay_time:.1f}us")
ax20.set_xlabel("Time (us)", fontsize=11)
ax20.set_ylabel("P(|1>)", fontsize=11)
ax20.legend(fontsize=7)
ax20.set_title("Case 3: Decohered Rabi\ncurve_fit result", fontsize=10)

# 2D joint posterior scatter
subsample = np.random.default_rng(0).choice(len(joint_omega_samples),
                                             size=min(3000, len(joint_omega_samples)),
                                             replace=False)
ax21.scatter(joint_omega_samples[subsample], joint_T2_samples[subsample],
             s=1, alpha=0.3, color="#2196F3")
ax21.axvline(OMEGA_RABI, color="#4CAF50", lw=1.5, ls="-", label=f"True omega={OMEGA_RABI}")
ax21.axhline(T2,         color="#4CAF50", lw=1.5, ls="--", label=f"True T2={T2}us")
ax21.axvline(mcmc_joint.medians[0], color="#F44336", lw=1.5, ls="-.",
             label=f"Median omega={mcmc_joint.medians[0]:.3f}")
ax21.axhline(mcmc_joint.medians[1], color="#F44336", lw=1.5, ls=":",
             label=f"Median T2={mcmc_joint.medians[1]:.2f}us")
ax21.set_xlabel("omega_rabi (rad/us)", fontsize=11)
ax21.set_ylabel("T2 (us)", fontsize=11)
ax21.legend(fontsize=7)
ax21.set_title("Case 3: Joint posterior\nomega vs T2 correlation", fontsize=10)

fig.suptitle(
    "Bayesian Parameter Estimation vs curve_fit\n"
    "Full posterior distributions reveal uncertainty structure",
    fontsize=13, y=0.98,
)
plt.savefig("bayesian_estimation.png", dpi=150, bbox_inches="tight")
print("[OK] Plot saved to bayesian_estimation.png")
