"""
rabi/ramsey.py
--------------
Implements the Ramsey experiment for precise T2 measurement.

Physics background:
    The Ramsey experiment consists of three stages:

    1. pi/2 pulse: A resonant drive rotates |0> to the equator.
       Duration: t_pi2 = pi / (2 * omega_rabi)

    2. Free precession for time tau: Drive off, qubit precesses under
       detuning delta. T1/T2 decoherence damps the off-diagonal elements.

    3. Second pi/2 pulse: Converts accumulated phase to population.

    The Ramsey signal is:
        P(|1>) = 0.5 * (1 + exp(-tau/T2) * cos(delta * tau))

    Starting near 1.0 at tau=0, decaying to 0.5 at tau >> T2.

    Implementation:
        Stage 1 and 3 (pulses) use mesolve with the drive Hamiltonian.
        Stage 2 (free precession) uses the analytical Lindblad solution:
            rho_00 -> rho_00 + (rho_11 - rho_00) * (1 - exp(-t/T1))
            rho_01 -> rho_01 * exp(-t/T2) * exp(-i*delta*t)
        This avoids numerical issues with time-dependent Hamiltonians
        and gives exact results for the free precession stage.
"""

import numpy as np
import qutip as qt
from dataclasses import dataclass
from typing import Optional

from .t1t2_noise import DecoherenceParams, build_collapse_operators


@dataclass
class RamseyResult:
    """Container for Ramsey experiment output."""
    tau_times: np.ndarray
    excited_pop: np.ndarray
    T2_true: float
    delta: float
    omega_rabi: float


@dataclass
class RamseyFitResult:
    """Container for Ramsey fit output."""
    T2_fit: float
    T2_err: float
    delta_fit: float
    amplitude: float
    offset: float
    residuals: np.ndarray
    chi_squared: float


def ramsey_model(
    tau: np.ndarray,
    T2: float,
    delta: float,
    amplitude: float,
    offset: float,
    phi: float,
) -> np.ndarray:
    """
    Analytical Ramsey fringe model.

        P(|1>) = offset + amplitude * 0.5 * (1 + exp(-tau/T2) * cos(delta*tau + phi))

    At tau=0: P = offset + amplitude (ideal: 1.0)
    At tau>>T2: P -> offset + amplitude*0.5 (ideal: 0.5)
    """
    return offset + amplitude * 0.5 * (1.0 + np.exp(-tau / T2) * np.cos(delta * tau + phi))


def _apply_pi2_pulse_dm(
    rho: qt.Qobj,
    omega_rabi: float,
    c_ops: list,
    n_steps: int = 50,
) -> qt.Qobj:
    """
    Apply a resonant pi/2 pulse to a density matrix state.

    Args:
        rho:        Input density matrix
        omega_rabi: Rabi frequency (rad/us)
        c_ops:      Collapse operators
        n_steps:    Number of time steps

    Returns:
        Density matrix after pi/2 pulse
    """
    t_pi2   = np.pi / (2.0 * omega_rabi)
    H_pulse = (omega_rabi / 2.0) * qt.sigmax()
    times   = np.linspace(0, t_pi2, n_steps + 1)

    result = qt.mesolve(
        H_pulse, rho, times,
        c_ops=c_ops,
        e_ops=[],
        options={"store_states": True},
    )
    return result.states[-1]


def _free_precess_analytical(
    rho: qt.Qobj,
    tau: float,
    delta: float,
    T1: float,
    T2: float,
) -> qt.Qobj:
    """
    Apply free precession analytically using the Lindblad solution.

    During free precession (no drive), the density matrix elements evolve as:
        rho_11(t) = rho_11(0) * exp(-t/T1) + rho_eq * (1 - exp(-t/T1))
        rho_00(t) = 1 - rho_11(t)
        rho_01(t) = rho_01(0) * exp(-t/T2) * exp(-i*delta*t)
        rho_10(t) = rho_10(0) * exp(-t/T2) * exp(+i*delta*t)

    where rho_eq = 0 (ground state equilibrium at low temperature).

    This is exact and avoids numerical issues with time-dependent Hamiltonians.

    Args:
        rho:   Input density matrix (2x2 QuTiP Qobj)
        tau:   Free precession time (us)
        delta: Detuning (rad/us)
        T1:    Energy relaxation time (us); 0 = infinite
        T2:    Dephasing time (us); 0 = infinite

    Returns:
        Density matrix after free precession
    """
    if tau == 0:
        return rho

    # Extract density matrix elements
    rho_arr = rho.full()
    r00 = rho_arr[0, 0]
    r11 = rho_arr[1, 1]
    r01 = rho_arr[0, 1]
    r10 = rho_arr[1, 0]

    # Apply T1 decay
    if T1 > 0:
        exp_t1 = np.exp(-tau / T1)
        r11_new = r11 * exp_t1              # decay toward 0
        r00_new = 1.0 - r11_new
    else:
        r11_new = r11
        r00_new = r00

    # Apply T2 decay and phase rotation
    if T2 > 0:
        exp_t2    = np.exp(-tau / T2)
        phase     = np.exp(-1j * delta * tau)
        r01_new   = r01 * exp_t2 * phase
        r10_new   = r10 * exp_t2 * np.conj(phase)
    else:
        r01_new = r01
        r10_new = r10

    # Reconstruct density matrix
    rho_new = qt.Qobj(
        np.array([[r00_new, r01_new],
                  [r10_new, r11_new]]),
        dims=[[2], [2]]
    )
    return rho_new


def run_ramsey_single(
    tau: float,
    omega_rabi: float,
    delta: float,
    decoherence: DecoherenceParams,
    n_steps_pulse: int = 50,
) -> float:
    """
    Run a single Ramsey experiment for one free precession time tau.

    Implements: |0> --[pi/2]--> free precession(tau) --[pi/2]--> P(|1>)

    Stage 1 and 3 use mesolve for the resonant drive.
    Stage 2 uses the analytical Lindblad solution for free precession.

    Args:
        tau:           Free precession time (us)
        omega_rabi:    Rabi frequency (rad/us)
        delta:         Intentional detuning (rad/us)
        decoherence:   T1/T2 parameters
        n_steps_pulse: RK4 steps per pi/2 pulse

    Returns:
        P(|1>) after the full sequence
    """
    c_ops = build_collapse_operators(decoherence)

    # Start in |0> as density matrix
    state = qt.ket2dm(qt.basis(2, 0))

    # Stage 1: pi/2 pulse
    state = _apply_pi2_pulse_dm(state, omega_rabi, c_ops, n_steps_pulse)

    # Stage 2: analytical free precession
    state = _free_precess_analytical(
        state, tau, delta,
        decoherence.T1, decoherence.T2
    )

    # Stage 3: second pi/2 pulse
    state = _apply_pi2_pulse_dm(state, omega_rabi, c_ops, n_steps_pulse)

    # Measure P(|1>)
    return float(qt.expect(qt.num(2), state))


def run_ramsey(
    omega_rabi: float,
    decoherence: DecoherenceParams,
    delta: float = 0.3,
    tau_max: float = None,
    n_tau: int = 150,
) -> RamseyResult:
    """
    Run a full Ramsey experiment by sweeping over free precession times.

    Args:
        omega_rabi:   Rabi frequency for pi/2 pulses (rad/us)
        decoherence:  T1/T2 parameters
        delta:        Intentional detuning (rad/us); default 0.3
        tau_max:      Maximum free precession time; defaults to 2*T2
        n_tau:        Number of tau values to sweep

    Returns:
        RamseyResult with tau array and P(|1>) at each tau
    """
    if tau_max is None:
        tau_max = 2.0 * decoherence.T2

    tau_times = np.linspace(0, tau_max, n_tau)
    print(f"  Running Ramsey sweep: {n_tau} tau points, tau_max={tau_max:.1f}µs")

    excited_pop = np.array([
        run_ramsey_single(tau, omega_rabi, delta, decoherence)
        for tau in tau_times
    ])

    return RamseyResult(
        tau_times=tau_times,
        excited_pop=excited_pop,
        T2_true=decoherence.T2,
        delta=delta,
        omega_rabi=omega_rabi,
    )


def fit_ramsey(
    tau_times: np.ndarray,
    excited_pop: np.ndarray,
) -> RamseyFitResult:
    """
    Fit the Ramsey fringe pattern to extract T2.

    Args:
        tau_times:    Free precession time array (us)
        excited_pop:  Measured P(|1>) at each tau

    Returns:
        RamseyFitResult with T2, uncertainty, and diagnostics
    """
    from scipy.optimize import curve_fit
    from scipy.signal import periodogram

    n  = len(excited_pop)
    dt = tau_times[1] - tau_times[0]

    # Delta guess: FFT of mean-subtracted signal
    signal        = excited_pop - np.mean(excited_pop)
    freqs, psd    = periodogram(signal, fs=1.0 / dt)
    dominant_freq = freqs[np.argmax(psd[1:]) + 1]
    delta_guess   = dominant_freq * 2 * np.pi

    # T2 guess: amplitude ratio between first and last quarter
    q         = max(n // 4, 5)
    early_amp = np.max(excited_pop[:q]) - np.min(excited_pop[:q])
    late_amp  = np.max(excited_pop[-q:]) - np.min(excited_pop[-q:])
    t_early   = np.mean(tau_times[:q])
    t_late    = np.mean(tau_times[-q:])

    if late_amp > 1e-3 and early_amp > late_amp:
        ratio    = late_amp / early_amp
        T2_guess = (t_late - t_early) / (-np.log(ratio))
        T2_guess = np.clip(T2_guess, tau_times[5], tau_times[-1] * 3)
    else:
        T2_guess = tau_times[-1] / 1.5

    p0 = [T2_guess, delta_guess, 1.0, 0.0, 0.0]

    bounds = (
        [0.1,   0,    0.1, -0.2, -np.pi],
        [1e4,  20.0,  1.5,  0.2,  np.pi],
    )

    popt, pcov = curve_fit(
        ramsey_model,
        tau_times,
        excited_pop,
        p0=p0,
        bounds=bounds,
        maxfev=20_000,
    )

    T2_fit, delta_fit, amp_fit, offset_fit, phi_fit = popt
    perr      = np.sqrt(np.diag(pcov))
    fitted    = ramsey_model(tau_times, *popt)
    residuals = excited_pop - fitted
    chi_sq    = float(np.sum(residuals**2) / (len(tau_times) - len(popt)))

    return RamseyFitResult(
        T2_fit=T2_fit,
        T2_err=perr[0],
        delta_fit=delta_fit,
        amplitude=amp_fit,
        offset=offset_fit,
        residuals=residuals,
        chi_squared=chi_sq,
    )