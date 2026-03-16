"""
rabi/fitting.py
---------------
Extracts the Rabi frequency from noisy P(|1>) data using SciPy curve fitting.

This is the core "calibration" step: given raw experimental data (simulated
here), we recover the system parameter (omega_rabi) that we would need to
know to apply accurate gates on the qubit.

The fit model is a damped sinusoid to account for any residual decay:

    P(|1>) = A * sin^2(omega * t / 2 + phi) * exp(-t / tau) + offset

For the ideal (noiseless) case: A=1, phi=0, tau=inf, offset=0.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import periodogram
from dataclasses import dataclass
from typing import Tuple


@dataclass
class FitResult:
    """Container for curve fit output."""
    omega_rabi_fit: float       # Fitted Rabi frequency (rad/us)
    omega_rabi_err: float       # 1-sigma uncertainty from covariance
    amplitude: float            # Fitted amplitude
    decay_time: float           # Fitted decay time tau (us); inf if undamped
    offset: float               # Fitted DC offset
    residuals: np.ndarray       # Data minus fit
    chi_squared: float          # Goodness of fit


def rabi_model(t: np.ndarray, omega: float, amplitude: float,
               tau: float, offset: float, phi: float) -> np.ndarray:
    """
    Damped Rabi oscillation model.

    Args:
        t:          Time array (us)
        omega:      Rabi frequency (rad/us)
        amplitude:  Oscillation amplitude
        tau:        Decay time constant (us)
        offset:     DC offset
        phi:        Phase offset (rad)

    Returns:
        Modelled P(|1>) at each time point
    """
    envelope = amplitude * np.exp(-t / tau)
    return envelope * np.sin(omega * t / 2.0 + phi) ** 2 + offset


def estimate_initial_params(
    times: np.ndarray,
    excited_pop: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """
    Estimate initial fit parameters using FFT frequency detection.

    Good initial guesses are critical for curve_fit convergence. We use
    the power spectral density to find the dominant oscillation frequency,
    which gives a reliable starting point for omega.

    Args:
        times:        Time array (us)
        excited_pop:  Noisy P(|1>) data

    Returns:
        Tuple of (omega_guess, amplitude_guess, tau_guess, offset_guess, phi_guess)
    """
    dt = times[1] - times[0]

    # FFT on mean-subtracted signal to find dominant frequency
    signal = excited_pop - np.mean(excited_pop)
    freqs, psd = periodogram(signal, fs=1.0 / dt)

    # Dominant frequency in cycles/us -> convert to rad/us for omega
    # Rabi oscillation in P(|1>) = sin^2(omega*t/2) has frequency omega/pi
    dominant_freq = freqs[np.argmax(psd[1:]) + 1]  # skip DC
    omega_guess = dominant_freq * np.pi * 2    # sin^2(omega*t/2) has freq omega/pi

    amplitude_guess = (np.max(excited_pop) - np.min(excited_pop))
    tau_guess = times[-1] * 2.0   # assume slow decay
    offset_guess = np.min(excited_pop)
    phi_guess = 0.0

    return omega_guess, amplitude_guess, tau_guess, offset_guess, phi_guess


def fit_rabi(
    times: np.ndarray,
    excited_pop: np.ndarray,
    sigma: float = None,
) -> FitResult:
    """
    Fit a damped Rabi model to P(|1>) data and extract omega_rabi.

    Args:
        times:        Time array (us)
        excited_pop:  Measured P(|1>) (possibly noisy)
        sigma:        Per-point measurement uncertainty (optional)

    Returns:
        FitResult with fitted parameters and uncertainties
    """
    p0 = estimate_initial_params(times, excited_pop)

    # Parameter bounds: all physical values
    bounds = (
        [0,    0,   0.1,  -0.1, -np.pi],   # lower bounds
        [1e3,  1.5,  1e6,   0.6,  np.pi],  # upper bounds
    )

    popt, pcov = curve_fit(
        rabi_model,
        times,
        excited_pop,
        p0=p0,
        bounds=bounds,
        sigma=sigma,
        absolute_sigma=(sigma is not None),
        maxfev=10_000,
    )

    omega_fit, amp_fit, tau_fit, offset_fit, phi_fit = popt
    perr = np.sqrt(np.diag(pcov))

    fitted = rabi_model(times, *popt)
    residuals = excited_pop - fitted
    chi_squared = float(np.sum(residuals**2) / (len(times) - len(popt)))

    return FitResult(
        omega_rabi_fit=omega_fit,
        omega_rabi_err=perr[0],
        amplitude=amp_fit,
        decay_time=tau_fit,
        offset=offset_fit,
        residuals=residuals,
        chi_squared=chi_squared,
    )
