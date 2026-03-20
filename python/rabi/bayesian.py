"""
rabi/bayesian.py
----------------
Bayesian parameter estimation for qubit calibration using MCMC (emcee).

Why Bayesian over curve_fit?
    curve_fit returns a point estimate and uncertainty based on linearization
    of the model around the best-fit point. This works well when:
      - The posterior is Gaussian (symmetric, unimodal)
      - The model is well-constrained by the data
      - Noise is perfectly Gaussian

    Bayesian MCMC returns the full posterior distribution, which:
      - Reveals non-Gaussian features (skewness, multimodality)
      - Correctly propagates uncertainty when parameters are correlated
      - Naturally incorporates prior physical knowledge
      - Handles cases where data is sparse or noisy more robustly

    In qubit calibration, Bayesian methods are used for:
      - Detecting parameter drift (posterior shifts over time)
      - Fusing multiple experiment types into a single estimate
      - Quantifying confidence in gate fidelity predictions

Implementation:
    We use emcee (Foreman-Mackey et al.) which implements the affine-invariant
    ensemble sampler. This requires:
      1. A log-posterior function: log P(params | data) = log L + log prior
      2. Initial walker positions (seeded from curve_fit estimates)
      3. A burn-in phase to move walkers away from initial positions
      4. A production phase to sample the posterior

    The log-posterior is the sum of:
      - log_likelihood: Gaussian likelihood given model prediction and noise
      - log_prior: encodes physical constraints (positivity, T2 <= 2*T1, etc.)
"""

import numpy as np
import emcee
from dataclasses import dataclass, field
from typing import Callable, Optional

from .fitting import rabi_model, fit_rabi
from .ramsey import ramsey_model, fit_ramsey


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class MCMCResult:
    """
    Container for MCMC sampling output.

    Attributes:
        samples:      Flattened chain of shape (n_samples, n_params)
        param_names:  Names of each parameter
        medians:      Posterior median for each parameter
        lower_1sigma: 16th percentile (1-sigma lower bound)
        upper_1sigma: 84th percentile (1-sigma upper bound)
        lower_2sigma: 2.5th percentile (2-sigma lower bound)
        upper_2sigma: 97.5th percentile (2-sigma upper bound)
        acceptance_fraction: Mean acceptance fraction of walkers (ideal: 0.2-0.5)
        n_walkers:    Number of MCMC walkers used
        n_steps:      Number of steps per walker (after burn-in)
    """
    samples: np.ndarray
    param_names: list
    medians: np.ndarray
    lower_1sigma: np.ndarray
    upper_1sigma: np.ndarray
    lower_2sigma: np.ndarray
    upper_2sigma: np.ndarray
    acceptance_fraction: float
    n_walkers: int
    n_steps: int

    def summary(self) -> str:
        """Return a human-readable summary of the posterior."""
        lines = ["Bayesian Posterior Summary:"]
        lines.append(f"  Walkers: {self.n_walkers}  Steps: {self.n_steps}")
        lines.append(f"  Acceptance fraction: {self.acceptance_fraction:.3f} (ideal: 0.2-0.5)")
        lines.append("")
        for i, name in enumerate(self.param_names):
            med   = self.medians[i]
            lo1   = self.lower_1sigma[i]
            hi1   = self.upper_1sigma[i]
            lines.append(f"  {name:20s}: {med:.6f}  [{lo1:.6f}, {hi1:.6f}]  (68% CI)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rabi omega estimation
# ---------------------------------------------------------------------------

def log_prior_rabi(params: np.ndarray) -> float:
    """
    Log prior for Rabi model parameters.

    Priors encode physical constraints:
      - omega:     Uniform [0.1, 100] rad/us  (must be positive, physically bounded)
      - amplitude: Uniform [0.1, 1.5]          (oscillation amplitude ~ 1)
      - tau:       Uniform [0.1, 1e5] us       (decay time, very wide)
      - offset:    Uniform [-0.2, 0.6]          (DC offset near 0)
      - phi:       Uniform [-pi, pi]            (phase, full range)

    Returns -inf if any parameter is outside its prior range (hard constraint).
    """
    omega, amplitude, tau, offset, phi = params

    if not (0.1   < omega     < 100.0):  return -np.inf
    if not (0.1   < amplitude < 1.5):    return -np.inf
    if not (0.1   < tau       < 1e5):    return -np.inf
    if not (-0.2  < offset    < 0.6):    return -np.inf
    if not (-np.pi < phi      < np.pi):  return -np.inf

    return 0.0  # log of uniform prior = 0 within bounds


def log_likelihood_rabi(
    params: np.ndarray,
    times: np.ndarray,
    data: np.ndarray,
    sigma: float,
) -> float:
    """
    Gaussian log-likelihood for Rabi model.

    log L = -0.5 * sum((data - model)^2 / sigma^2) - N * log(sigma * sqrt(2pi))

    Args:
        params: [omega, amplitude, tau, offset, phi]
        times:  Time array (us)
        data:   Noisy P(|1>) measurements
        sigma:  Measurement noise standard deviation

    Returns:
        Log-likelihood value
    """
    model = rabi_model(times, *params)
    residuals = data - model
    return -0.5 * np.sum(residuals**2 / sigma**2 + np.log(2 * np.pi * sigma**2))


def log_posterior_rabi(
    params: np.ndarray,
    times: np.ndarray,
    data: np.ndarray,
    sigma: float,
) -> float:
    """Log posterior = log prior + log likelihood for Rabi model."""
    lp = log_prior_rabi(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_rabi(params, times, data, sigma)


def run_mcmc_rabi(
    times: np.ndarray,
    data: np.ndarray,
    sigma: float = 0.02,
    n_walkers: int = 32,
    n_burn: int = 500,
    n_steps: int = 2000,
    seed: int = 42,
) -> MCMCResult:
    """
    Run MCMC to estimate Rabi frequency and all model parameters.

    Uses curve_fit estimate as initial walker positions, then runs
    emcee ensemble sampler to explore the full posterior.

    Args:
        times:     Time array (us)
        data:      Noisy P(|1>) measurements
        sigma:     Measurement noise std (default 0.02)
        n_walkers: Number of MCMC walkers (must be >= 2 * n_params)
        n_burn:    Burn-in steps (discarded)
        n_steps:   Production steps per walker
        seed:      Random seed for reproducibility

    Returns:
        MCMCResult with full posterior samples and summary statistics
    """
    # Seed walkers from curve_fit estimate
    fit = fit_rabi(times, data)
    p0_center = np.array([
        fit.omega_rabi_fit,
        fit.amplitude,
        fit.decay_time if np.isfinite(fit.decay_time) and fit.decay_time < 1e4 else 1000.0,
        fit.offset,
        0.0,  # phi
    ])

    n_params = len(p0_center)
    rng = np.random.default_rng(seed)

    # Initialize walkers in a small Gaussian ball around curve_fit estimate
    p0 = p0_center + 1e-3 * rng.standard_normal((n_walkers, n_params))

    # Clip to prior bounds to avoid starting outside prior
    p0[:, 0] = np.clip(p0[:, 0], 0.11, 99.9)   # omega
    p0[:, 1] = np.clip(p0[:, 1], 0.11, 1.49)   # amplitude
    p0[:, 2] = np.clip(p0[:, 2], 0.11, 9999.0) # tau
    p0[:, 3] = np.clip(p0[:, 3], -0.19, 0.59)  # offset
    p0[:, 4] = np.clip(p0[:, 4], -np.pi+0.01, np.pi-0.01)  # phi

    sampler = emcee.EnsembleSampler(
        n_walkers, n_params, log_posterior_rabi,
        args=(times, data, sigma)
    )

    # Burn-in
    state = sampler.run_mcmc(p0, n_burn, progress=False)
    sampler.reset()

    # Production run
    sampler.run_mcmc(state, n_steps, progress=False)

    return _extract_result(
        sampler,
        param_names=["omega_rabi", "amplitude", "tau", "offset", "phi"],
        n_walkers=n_walkers,
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# Ramsey T2 estimation
# ---------------------------------------------------------------------------

def log_prior_ramsey(params: np.ndarray) -> float:
    """
    Log prior for Ramsey model parameters.

      - T2:        Uniform [0.1, 1000] us
      - delta:     Uniform [0.01, 20] rad/us
      - amplitude: Uniform [0.1, 1.5]
      - offset:    Uniform [-0.2, 0.2]
      - phi:       Uniform [-pi, pi]
    """
    T2, delta, amplitude, offset, phi = params

    if not (0.1   < T2        < 1000.0): return -np.inf
    if not (0.01  < delta     < 20.0):   return -np.inf
    if not (0.1   < amplitude < 1.5):    return -np.inf
    if not (-0.2  < offset    < 0.2):    return -np.inf
    if not (-np.pi < phi      < np.pi):  return -np.inf

    return 0.0


def log_likelihood_ramsey(
    params: np.ndarray,
    tau_times: np.ndarray,
    data: np.ndarray,
    sigma: float,
) -> float:
    """Gaussian log-likelihood for Ramsey model."""
    model = ramsey_model(tau_times, *params)
    residuals = data - model
    return -0.5 * np.sum(residuals**2 / sigma**2 + np.log(2 * np.pi * sigma**2))


def log_posterior_ramsey(
    params: np.ndarray,
    tau_times: np.ndarray,
    data: np.ndarray,
    sigma: float,
) -> float:
    """Log posterior for Ramsey model."""
    lp = log_prior_ramsey(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_ramsey(params, tau_times, data, sigma)


def run_mcmc_ramsey(
    tau_times: np.ndarray,
    data: np.ndarray,
    sigma: float = 0.02,
    n_walkers: int = 32,
    n_burn: int = 500,
    n_steps: int = 2000,
    seed: int = 42,
) -> MCMCResult:
    """
    Run MCMC to estimate T2 from Ramsey fringe data.

    Args:
        tau_times: Free precession time array (us)
        data:      Noisy P(|1>) measurements
        sigma:     Measurement noise std
        n_walkers: Number of MCMC walkers
        n_burn:    Burn-in steps
        n_steps:   Production steps per walker
        seed:      Random seed

    Returns:
        MCMCResult with full T2 posterior
    """
    fit = fit_ramsey(tau_times, data)
    p0_center = np.array([
        fit.T2_fit,
        fit.delta_fit,
        fit.amplitude,
        fit.offset,
        0.0,
    ])

    n_params = len(p0_center)
    rng = np.random.default_rng(seed)
    p0  = p0_center + 1e-3 * rng.standard_normal((n_walkers, n_params))

    p0[:, 0] = np.clip(p0[:, 0], 0.11, 999.0)
    p0[:, 1] = np.clip(p0[:, 1], 0.02, 19.9)
    p0[:, 2] = np.clip(p0[:, 2], 0.11, 1.49)
    p0[:, 3] = np.clip(p0[:, 3], -0.19, 0.19)
    p0[:, 4] = np.clip(p0[:, 4], -np.pi+0.01, np.pi-0.01)

    sampler = emcee.EnsembleSampler(
        n_walkers, n_params, log_posterior_ramsey,
        args=(tau_times, data, sigma)
    )

    state = sampler.run_mcmc(p0, n_burn, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, n_steps, progress=False)

    return _extract_result(
        sampler,
        param_names=["T2", "delta", "amplitude", "offset", "phi"],
        n_walkers=n_walkers,
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# Joint Rabi omega + T2 estimation
# ---------------------------------------------------------------------------

def log_prior_joint(params: np.ndarray) -> float:
    """
    Log prior for joint omega + T2 estimation from decohered Rabi data.

    Parameters: [omega, T2, amplitude, offset, phi]

    Physical constraint: T2 > 0 (T1 not explicitly modelled here,
    we treat T2 as the effective decay time of the Rabi envelope).
    """
    omega, T2, amplitude, offset, phi = params

    if not (0.1   < omega     < 100.0):  return -np.inf
    if not (0.1   < T2        < 1000.0): return -np.inf
    if not (0.1   < amplitude < 1.5):    return -np.inf
    if not (-0.2  < offset    < 0.6):    return -np.inf
    if not (-np.pi < phi      < np.pi):  return -np.inf

    return 0.0


def log_posterior_joint(
    params: np.ndarray,
    times: np.ndarray,
    data: np.ndarray,
    sigma: float,
) -> float:
    """
    Log posterior for joint omega + T2 estimation.

    Uses rabi_model which already has a decay envelope (tau parameter),
    so omega maps to params[0] and T2 maps to params[1] (the tau/decay term).
    """
    lp = log_prior_joint(params)
    if not np.isfinite(lp):
        return -np.inf
    # rabi_model signature: (t, omega, amplitude, tau, offset, phi)
    # Joint params: [omega, T2, amplitude, offset, phi]
    # Remap: tau = T2
    omega, T2, amplitude, offset, phi = params
    model     = rabi_model(times, omega, amplitude, T2, offset, phi)
    residuals = data - model
    ll = -0.5 * np.sum(residuals**2 / sigma**2 + np.log(2 * np.pi * sigma**2))
    return lp + ll


def run_mcmc_joint(
    times: np.ndarray,
    data: np.ndarray,
    sigma: float = 0.02,
    n_walkers: int = 32,
    n_burn: int = 500,
    n_steps: int = 2000,
    seed: int = 42,
) -> MCMCResult:
    """
    Run MCMC to jointly estimate omega_rabi and T2 from decohered Rabi data.

    This is the most informative use of Bayesian inference — it reveals
    the correlation between omega and T2, showing how uncertainty in one
    parameter affects the other.

    Args:
        times:     Time array (us)
        data:      Noisy decohered P(|1>) measurements
        sigma:     Measurement noise std
        n_walkers: Number of MCMC walkers
        n_burn:    Burn-in steps
        n_steps:   Production steps per walker
        seed:      Random seed

    Returns:
        MCMCResult with joint omega + T2 posterior
    """
    fit = fit_rabi(times, data)
    p0_center = np.array([
        fit.omega_rabi_fit,
        fit.decay_time if np.isfinite(fit.decay_time) and fit.decay_time < 500.0 else 30.0,
        fit.amplitude,
        fit.offset,
        0.0,
    ])

    n_params = len(p0_center)
    rng = np.random.default_rng(seed)
    p0  = p0_center + 1e-3 * rng.standard_normal((n_walkers, n_params))

    p0[:, 0] = np.clip(p0[:, 0], 0.11, 99.9)
    p0[:, 1] = np.clip(p0[:, 1], 0.11, 499.0)
    p0[:, 2] = np.clip(p0[:, 2], 0.11, 1.49)
    p0[:, 3] = np.clip(p0[:, 3], -0.19, 0.59)
    p0[:, 4] = np.clip(p0[:, 4], -np.pi+0.01, np.pi-0.01)

    sampler = emcee.EnsembleSampler(
        n_walkers, n_params, log_posterior_joint,
        args=(times, data, sigma)
    )

    state = sampler.run_mcmc(p0, n_burn, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, n_steps, progress=False)

    return _extract_result(
        sampler,
        param_names=["omega_rabi", "T2", "amplitude", "offset", "phi"],
        n_walkers=n_walkers,
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _extract_result(
    sampler: emcee.EnsembleSampler,
    param_names: list,
    n_walkers: int,
    n_steps: int,
) -> MCMCResult:
    """Extract posterior statistics from a completed emcee sampler."""
    # Flatten chain: shape (n_walkers * n_steps, n_params)
    samples = sampler.get_chain(flat=True)

    medians      = np.median(samples, axis=0)
    lower_1sigma = np.percentile(samples, 16, axis=0)
    upper_1sigma = np.percentile(samples, 84, axis=0)
    lower_2sigma = np.percentile(samples, 2.5, axis=0)
    upper_2sigma = np.percentile(samples, 97.5, axis=0)

    return MCMCResult(
        samples=samples,
        param_names=param_names,
        medians=medians,
        lower_1sigma=lower_1sigma,
        upper_1sigma=upper_1sigma,
        lower_2sigma=lower_2sigma,
        upper_2sigma=upper_2sigma,
        acceptance_fraction=float(np.mean(sampler.acceptance_fraction)),
        n_walkers=n_walkers,
        n_steps=n_steps,
    )
