"""
tests/test_rabi.py
------------------
Unit tests for the Rabi simulator and curve fitter.

Run with:  pytest python/tests/ -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rabi.simulator import run_rabi, add_measurement_noise
from rabi.fitting import fit_rabi


# ---------------------------------------------------------------------------
# Simulator tests
# ---------------------------------------------------------------------------

class TestRabiSimulator:

    def test_ground_state_at_t0(self):
        """At t=0, qubit starts in |0>, so P(|1>) = 0."""
        result = run_rabi(omega_rabi=1.0, t_max=5.0, n_points=100)
        assert result.excited_pop[0] == pytest.approx(0.0, abs=1e-6)

    def test_pi_pulse(self):
        """At t = pi/omega_rabi, qubit should be fully inverted: P(|1>) ~ 1."""
        omega = 2.0
        t_pi = np.pi / omega
        result = run_rabi(omega_rabi=omega, t_max=t_pi * 2, n_points=500)
        # Find index closest to t_pi
        idx = np.argmin(np.abs(result.times - t_pi))
        assert result.excited_pop[idx] == pytest.approx(1.0, abs=0.01)

    def test_oscillation_frequency(self):
        """FFT of P(|1>) should peak at omega_rabi / (2*pi) cycles/us."""
        omega = 3.0
        result = run_rabi(omega_rabi=omega, t_max=20.0, n_points=1000)
        dt = result.times[1] - result.times[0]
        freqs = np.fft.rfftfreq(len(result.times), d=dt)
        power = np.abs(np.fft.rfft(result.excited_pop - 0.5)) ** 2
        peak_freq = freqs[np.argmax(power[1:]) + 1]
        expected_freq = omega / (2 * np.pi)  # P(|1>) = sin^2(omega*t/2) oscillates at omega/2pi
        assert peak_freq == pytest.approx(expected_freq, rel=0.05)

    def test_analytical_agreement(self):
        """Simulation must match P(|1>) = sin^2(omega*t/2) to within 1e-4."""
        omega = 1.5
        result = run_rabi(omega_rabi=omega, t_max=8.0, n_points=500)
        analytic = np.sin(omega * result.times / 2.0) ** 2
        np.testing.assert_allclose(result.excited_pop, analytic, atol=1e-4)

    def test_output_bounds(self):
        """P(|1>) must stay within [0, 1] for all time."""
        result = run_rabi(omega_rabi=2.5, t_max=15.0, n_points=300)
        assert np.all(result.excited_pop >= -1e-9)
        assert np.all(result.excited_pop <=  1.0 + 1e-9)

    def test_measurement_noise_shape(self):
        """Noisy data should have same shape as clean data."""
        result = run_rabi(omega_rabi=1.0, n_points=100)
        noisy = add_measurement_noise(result.excited_pop, seed=42)
        assert noisy.shape == result.excited_pop.shape

    def test_measurement_noise_bounds(self):
        """Noisy data should stay clipped to [0, 1]."""
        result = run_rabi(omega_rabi=1.0, n_points=200)
        noisy = add_measurement_noise(result.excited_pop, noise_level=0.05, seed=0)
        assert np.all(noisy >= 0.0)
        assert np.all(noisy <= 1.0)


# ---------------------------------------------------------------------------
# Fitting tests
# ---------------------------------------------------------------------------

class TestRabiFitting:

    @pytest.fixture
    def clean_data(self):
        omega = 2.0
        result = run_rabi(omega_rabi=omega, t_max=10.0, n_points=300)
        return result.times, result.excited_pop, omega

    @pytest.fixture
    def noisy_data(self):
        omega = 2.0
        result = run_rabi(omega_rabi=omega, t_max=10.0, n_points=300)
        noisy = add_measurement_noise(result.excited_pop, noise_level=0.02, seed=7)
        return result.times, noisy, omega

    def test_fit_recovers_omega_clean(self, clean_data):
        """On clean data, fit should recover omega to within 0.1%."""
        times, data, true_omega = clean_data
        fit = fit_rabi(times, data)
        assert fit.omega_rabi_fit == pytest.approx(true_omega, rel=0.001)

    def test_fit_recovers_omega_noisy(self, noisy_data):
        """On noisy data (2% noise), fit should recover omega to within 2%."""
        times, data, true_omega = noisy_data
        fit = fit_rabi(times, data)
        assert fit.omega_rabi_fit == pytest.approx(true_omega, rel=0.02)

    def test_fit_uncertainty_is_positive(self, noisy_data):
        """Fit uncertainty (1-sigma) must be a positive finite number."""
        times, data, _ = noisy_data
        fit = fit_rabi(times, data)
        assert fit.omega_rabi_err > 0
        assert np.isfinite(fit.omega_rabi_err)

    def test_residuals_shape(self, noisy_data):
        """Residuals should have same length as input data."""
        times, data, _ = noisy_data
        fit = fit_rabi(times, data)
        assert len(fit.residuals) == len(times)

    def test_chi_squared_reasonable(self, noisy_data):
        """Chi-squared should be close to 1 for well-fitted data with ~2% noise."""
        times, data, _ = noisy_data
        fit = fit_rabi(times, data)
        # chi^2 ~ noise_level^2 = 0.02^2 = 4e-4 for this unnormalised version
        assert fit.chi_squared < 0.01


# ---------------------------------------------------------------------------
# Integration test: simulate -> add noise -> fit -> recover
# ---------------------------------------------------------------------------

class TestEndToEnd:

    @pytest.mark.parametrize("omega", [0.5, 1.0, 2.0, 3.5, 5.0])
    def test_calibration_pipeline(self, omega):
        """Full pipeline: simulate -> noise -> fit, for a range of Rabi frequencies."""
        result = run_rabi(omega_rabi=omega, t_max=4 * np.pi / omega, n_points=400)
        noisy  = add_measurement_noise(result.excited_pop, noise_level=0.02, seed=42)
        fit    = fit_rabi(result.times, noisy)
        assert fit.omega_rabi_fit == pytest.approx(omega, rel=0.03)
