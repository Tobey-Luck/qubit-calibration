"""
run_step1.py
------------
Demonstrates the full Step 1 pipeline:

    1. Simulate Rabi oscillations with QuTiP
    2. Add synthetic measurement noise
    3. Fit to extract omega_rabi with uncertainty
    4. Compare against C++ RK4 solver (if built)
    5. Plot results

Run from the project root:
    python run_step1.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Allow imports from python/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

from rabi.simulator import run_rabi, add_measurement_noise
from rabi.fitting import fit_rabi
from rabi.visualization import plot_rabi_fit

os.add_dll_directory(r"C:\msys64\mingw64\bin")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

OMEGA_RABI  = 2.0    # rad/us, this is what we're trying to calibrate
T_MAX       = 10.0   # us
N_POINTS    = 200
NOISE_LEVEL = 0.02   # ~ 500 shot statistics
SEED        = 42

# ---------------------------------------------------------------------------
# Step 1: Simulate ideal Rabi oscillations
# ---------------------------------------------------------------------------

print("=" * 55)
print("  Qubit Calibration Simulator, Step 1: Rabi")
print("=" * 55)
print(f"\nTrue Rabi frequency: ω = {OMEGA_RABI:.4f} rad/µs")
print(f"π-pulse time:        {np.pi / OMEGA_RABI:.4f} µs")

result = run_rabi(omega_rabi=OMEGA_RABI, t_max=T_MAX, n_points=N_POINTS)
print(f"\n[✓] QuTiP simulation complete ({N_POINTS} time points)")

# ---------------------------------------------------------------------------
# Step 2: Add measurement noise
# ---------------------------------------------------------------------------

noisy = add_measurement_noise(result.excited_pop, noise_level=NOISE_LEVEL, seed=SEED)
print(f"[✓] Synthetic noise added (σ = {NOISE_LEVEL}, seed = {SEED})")

# ---------------------------------------------------------------------------
# Step 3: Fit to extract omega_rabi
# ---------------------------------------------------------------------------

fit = fit_rabi(result.times, noisy)

print(f"\n--- Calibration Results ---")
print(f"True  ω:   {OMEGA_RABI:.6f} rad/µs")
print(f"Fitted ω:  {fit.omega_rabi_fit:.6f} ± {fit.omega_rabi_err:.6f} rad/µs")
print(f"Error:     {abs(fit.omega_rabi_fit - OMEGA_RABI) / OMEGA_RABI * 100:.3f}%")
print(f"χ²:        {fit.chi_squared:.6f}")
print(f"Decay τ:   {fit.decay_time:.2f} µs  (expected >> {T_MAX} for ideal case)")

# ---------------------------------------------------------------------------
# Step 4: Try loading C++ solver (optional)
# ---------------------------------------------------------------------------

cpp_data = None

try:
    import glob
    build_path = r"C:\qubit_cal\cpp\build"
    sys.path.insert(0, build_path)
    
    # Handle versioned filename e.g. cpp_solver.cp313-win_amd64.pyd
    pyd_files = glob.glob(os.path.join(build_path, "cpp_solver*.pyd"))
    if pyd_files:
        import importlib.util
        spec = importlib.util.spec_from_file_location("cpp_solver", pyd_files[0])
        cpp_solver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cpp_solver)
        BlochParams = cpp_solver.BlochParams
        BlochState = cpp_solver.BlochState
        solve_bloch = cpp_solver.solve_bloch
    else:
        raise ImportError("No cpp_solver .pyd found")

    params = BlochParams()
    params.omega_rabi = OMEGA_RABI
    params.delta      = 0.0
    params.T1         = 0.0
    params.T2         = 0.0

    initial = BlochState()
    initial.bx = 0.0; initial.by = 0.0; initial.bz = -1.0

    cpp_result = solve_bloch(params, initial, T_MAX, N_POINTS)
    cpp_data = np.array(cpp_result.excited_pop)

    max_diff = np.max(np.abs(cpp_data - result.excited_pop))
    print(f"\n[✓] C++ RK4 solver loaded")
    print(f"    Max deviation from QuTiP: {max_diff:.2e}")
    assert max_diff < 1e-4, "C++ and Python results disagree!"
    print(f"    Validation: PASSED ✓")

except Exception as e:
    print(f"\n[!] C++ solver not loaded: {e}")

# ---------------------------------------------------------------------------
# Step 5: Plot
# ---------------------------------------------------------------------------

fig = plot_rabi_fit(result, fit, noisy_data=noisy, cpp_data=cpp_data,
                    save_path="rabi_calibration.png")
plt.show()
print(f"\n[✓] Plot saved to rabi_calibration.png")
