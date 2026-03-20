"""
tests/test_bayesian.py
----------------------
Tests for the Bayesian parameter estimation module.

Run with: pytest python/tests/ -v

Note: MCMC tests are marked slow and use small n_steps for speed.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from rabi.simulator import run_rabi, add_measurement_noise
from rabi.t1t2_noise import DecoherenceParams, run_rabi_with_decoherence
from rabi.ramsey import run_ramsey
from rabi.bayesian import (
    log_prior_rabi, log_prior_ramsey, log_prior_joint,
    log_likelihood_rabi, log_likelihood_ramsey,
    run_mcmc_rabi, run_mcmc_ramsey, run_mcmc_joint,
    MCMCResult,
)


# ---------------------------------------------------------------------------
# Prior tests
# ---------------------------------------------------------------------------

class TestPriors:

    def test_rabi_prior_valid_params(self):
        """Valid parameters should return 0 (log of uniform prior)."""
        params = np.array([2.0, 1.0, 100.0, 0.0, 0.0])
        assert log_prior_rabi(params) == 0.0

    def test_rabi_prior_negative_omega(self):
        """Negative omega is unphysical — should return -inf."""
        params = np.array([-1.0, 1.0, 100.0, 0.0, 0.0])
        assert log_prior_rabi(params) == -np.inf

    def test_rabi_prior_large_omega(self):
        """Omega > 100 is outside prior range."""
        params = np.array([200.0, 1.0, 100.0, 0.0, 0.0])
        assert log_prior_rabi(params) == -np.inf

    def test_ramsey_prior_valid(self):
        """Valid Ramsey params should return 0."""
        params = np.array([30.0, 0.3, 1.0, 0.0, 0.0])
        assert log_prior_ramsey(params) == 0.0

    def test_ramsey_prior_negative_T2(self):
        """Negative T2 is unphysical."""
        params = np.array([-5.0, 0.3, 1.0, 0.0, 0.0])
        assert log_prior_ramsey(params) == -np.inf

    def test_joint_prior_valid(self):
        """Valid joint params should return 0."""
        params = np.array([2.0, 30.0, 1.0, 0.0, 0.0])
        assert log_prior_joint(params) == 0.0

    def test_joint_prior_negative_T2(self):
        """Negative T2 is unphysical."""
        params = np.array([2.0, -5.0, 1.0, 0.0, 0.0])
        assert log_prior_joint(params) == -np.inf


# ---------------------------------------------------------------------------
# Likelihood tests
# ---------------------------------------------------------------------------

class TestLikelihood:

    def test_rabi_likelihood_perfect_fit(self):
        """Log-likelihood should be maximized when model matches data exactly."""
        from rabi.fitting import rabi_model
        times  = np.linspace(0, 10, 100)
        params = np.array([2.0, 1.0, 1000.0, 0.0, 0.0])
        data   = rabi_model(times, *params)
        ll     = log_likelihood_rabi(params, times, data, sigma=0.02)
        assert np.isfinite(ll)
        assert ll > -1000.0  # should be high for perfect fit

    def test_rabi_likelihood_bad_fit(self):
        """Log-likelihood should be lower for wrong parameters."""
        from rabi.fitting import rabi_model
        times       = np.linspace(0, 10, 100)
        true_params = np.array([2.0, 1.0, 1000.0, 0.0, 0.0])
        bad_params  = np.array([5.0, 1.0, 1000.0, 0.0, 0.0])
        data        = rabi_model(times, *true_params)
        ll_true = log_likelihood_rabi(true_params, times, data, sigma=0.02)
        ll_bad  = log_likelihood_rabi(bad_params,  times, data, sigma=0.02)
        assert ll_true > ll_bad

    def test_ramsey_likelihood_perfect_fit(self):
        """Ramsey likelihood should be finite and high for perfect fit."""
        from rabi.ramsey import ramsey_model
        tau    = np.linspace(0, 30, 100)
        params = np.array([30.0, 0.3, 1.0, 0.0, 0.0])
        data   = ramsey_model(tau, *params)
        ll     = log_likelihood_ramsey(params, tau, data, sigma=0.02)
        assert np.isfinite(ll)
        assert ll > -1000.0


# ---------------------------------------------------------------------------
# MCMC tests (use small n_steps for speed)
# ---------------------------------------------------------------------------

class TestMCMC:

    @pytest.fixture
    def rabi_data(self):
        result = run_rabi(omega_rabi=2.0, t_max=10.0, n_points=200)
        noisy  = add_measurement_noise(result.excited_pop, noise_level=0.02, seed=42)
        return result.times, noisy

    @pytest.fixture
    def decohered_rabi_data(self):
        p      = DecoherenceParams(T1=50.0, T2=30.0)
        result = run_rabi_with_decoherence(omega_rabi=2.0, decoherence=p,
                                           t_max=60.0, n_points=300)
        noisy  = add_measurement_noise(result.excited_pop, noise_level=0.02, seed=42)
        return result.times, noisy

    @pytest.fixture
    def ramsey_data(self):
        p      = DecoherenceParams(T1=50.0, T2=30.0)
        result = run_ramsey(omega_rabi=2.0, decoherence=p,
                            delta=0.3, tau_max=60.0, n_tau=100)
        noisy  = add_measurement_noise(result.excited_pop, noise_level=0.02, seed=42)
        return result.tau_times, noisy

    def test_mcmc_rabi_returns_result(self, rabi_data):
        """run_mcmc_rabi should return an MCMCResult."""
        times, data = rabi_data
        result = run_mcmc_rabi(times, data, n_walkers=16, n_burn=100, n_steps=200)
        assert isinstance(result, MCMCResult)

    def test_mcmc_rabi_sample_shape(self, rabi_data):
        """Samples should have shape (n_walkers * n_steps, n_params)."""
        times, data = rabi_data
        result = run_mcmc_rabi(times, data, n_walkers=16, n_burn=100, n_steps=200)
        assert result.samples.shape == (16 * 200, 5)

    def test_mcmc_rabi_recovers_omega(self, rabi_data):
        """Posterior median omega should be within 5% of true value."""
        times, data = rabi_data
        result = run_mcmc_rabi(times, data, n_walkers=16, n_burn=200, n_steps=500)
        omega_median = result.medians[0]
        assert omega_median == pytest.approx(2.0, rel=0.05)

    def test_mcmc_rabi_uncertainty_positive(self, rabi_data):
        """Posterior credible interval should have positive width."""
        times, data = rabi_data
        result = run_mcmc_rabi(times, data, n_walkers=16, n_burn=100, n_steps=200)
        assert result.upper_1sigma[0] > result.lower_1sigma[0]

    def test_mcmc_rabi_acceptance_fraction(self, rabi_data):
        """Acceptance fraction should be in healthy range [0.1, 0.9]."""
        times, data = rabi_data
        result = run_mcmc_rabi(times, data, n_walkers=16, n_burn=100, n_steps=200)
        assert 0.1 < result.acceptance_fraction < 0.9

    def test_mcmc_param_names(self, rabi_data):
        """MCMCResult should have correct parameter names."""
        times, data = rabi_data
        result = run_mcmc_rabi(times, data, n_walkers=16, n_burn=100, n_steps=200)
        assert "omega_rabi" in result.param_names

    def test_mcmc_ramsey_recovers_T2(self, ramsey_data):
        """Posterior median T2 should be within 10% of true value."""
        tau_times, data = ramsey_data
        result = run_mcmc_ramsey(tau_times, data, n_walkers=16, n_burn=200, n_steps=500)
        T2_median = result.medians[0]
        assert T2_median == pytest.approx(30.0, rel=0.10)

    def test_mcmc_joint_recovers_omega_and_T2(self, decohered_rabi_data):
        """Joint posterior should recover both omega and T2."""
        times, data = decohered_rabi_data
        result = run_mcmc_joint(times, data, n_walkers=16, n_burn=200, n_steps=500)
        assert result.medians[0] == pytest.approx(2.0, rel=0.05)   # omega
        assert result.medians[1] == pytest.approx(30.0, rel=0.20)  # T2

    def test_mcmc_joint_sample_shape(self, decohered_rabi_data):
        """Joint samples should have 5 parameters."""
        times, data = decohered_rabi_data
        result = run_mcmc_joint(times, data, n_walkers=16, n_burn=100, n_steps=200)
        assert result.samples.shape[1] == 5

    def test_summary_string(self, rabi_data):
        """MCMCResult.summary() should return a non-empty string."""
        times, data = rabi_data
        result = run_mcmc_rabi(times, data, n_walkers=16, n_burn=100, n_steps=200)
        s = result.summary()
        assert isinstance(s, str)
        assert "omega_rabi" in s
        assert len(s) > 50
