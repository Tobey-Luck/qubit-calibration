"""
tests/test_qiskit_backend.py
-----------------------------
Tests for the Qiskit Aer calibration backend.

Skips automatically if qiskit/qiskit-aer are not installed.
Run with: pytest python/tests/ -v
"""

import numpy as np
import pytest
import sys
import os

# Skip entire module if Qiskit not available
qiskit = pytest.importorskip("qiskit", reason="qiskit not installed")
qiskit_aer = pytest.importorskip("qiskit_aer", reason="qiskit-aer not installed")

from rabi.qiskit_backend import (
    build_custom_noise_model,
    run_rabi_circuit,
    run_ramsey_circuit,
    QiskitCalibResult,
    QISKIT_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Noise model tests
# ---------------------------------------------------------------------------

class TestNoiseModel:

    def test_custom_noise_model_builds(self):
        """Custom noise model should build without error."""
        nm = build_custom_noise_model(T1=50.0, T2=30.0)
        assert nm is not None

    def test_custom_noise_model_type(self):
        """Custom noise model should be a Qiskit NoiseModel."""
        from qiskit_aer.noise import NoiseModel
        nm = build_custom_noise_model(T1=50.0, T2=30.0)
        assert isinstance(nm, NoiseModel)

    def test_custom_noise_model_has_errors(self):
        """Custom noise model should have errors defined on gates."""
        nm = build_custom_noise_model(T1=50.0, T2=30.0)
        assert len(nm.basis_gates) > 0


# ---------------------------------------------------------------------------
# Rabi circuit tests
# ---------------------------------------------------------------------------

class TestRabiCircuit:

    def test_rabi_output_shape(self):
        """Output arrays should have n_points elements."""
        result = run_rabi_circuit(
            omega_rabi=2.0, t_max=5.0, n_points=10,
            noise_model=None, shots=256
        )
        assert len(result.times) == 10
        assert len(result.excited_pop) == 10

    def test_rabi_output_bounds(self):
        """P(|1>) must stay in [0, 1]."""
        result = run_rabi_circuit(
            omega_rabi=2.0, t_max=5.0, n_points=10,
            noise_model=None, shots=512
        )
        assert np.all(result.excited_pop >= 0.0)
        assert np.all(result.excited_pop <= 1.0)

    def test_rabi_ground_state_at_t0(self):
        """At t=0, Rx(0) applied — qubit stays in |0>, P(|1>) ~ 0."""
        result = run_rabi_circuit(
            omega_rabi=2.0, t_max=5.0, n_points=10,
            noise_model=None, shots=1024
        )
        assert result.excited_pop[0] == pytest.approx(0.0, abs=0.05)

    def test_rabi_ideal_matches_theory(self):
        """Ideal Rabi circuit should match P(|1>) = sin^2(omega*t/2)."""
        omega  = 1.0
        t_max  = 2 * np.pi / omega
        result = run_rabi_circuit(
            omega_rabi=omega, t_max=t_max, n_points=20,
            noise_model=None, shots=4096
        )
        analytic = np.sin(omega * result.times / 2.0) ** 2
        # Allow 5% tolerance due to shot noise
        np.testing.assert_allclose(result.excited_pop, analytic, atol=0.05)

    def test_rabi_with_custom_noise(self):
        """Rabi with custom noise should produce valid output."""
        nm = build_custom_noise_model(T1=50.0, T2=30.0)
        result = run_rabi_circuit(
            omega_rabi=2.0, t_max=5.0, n_points=10,
            noise_model=nm, shots=512
        )
        assert isinstance(result, QiskitCalibResult)
        assert np.all(result.excited_pop >= 0.0)
        assert np.all(result.excited_pop <= 1.0)

    def test_rabi_result_type(self):
        """Result should be a QiskitCalibResult."""
        result = run_rabi_circuit(
            omega_rabi=2.0, t_max=5.0, n_points=5,
            noise_model=None, shots=256
        )
        assert isinstance(result, QiskitCalibResult)
        assert result.noise_model_name is not None
        assert result.shots == 256


# ---------------------------------------------------------------------------
# Ramsey circuit tests
# ---------------------------------------------------------------------------

class TestRamseyCircuit:

    def test_ramsey_output_shape(self):
        """Output arrays should have n_tau elements."""
        result = run_ramsey_circuit(
            delta=0.3, tau_max=5.0, n_tau=10,
            T1=50.0, T2=30.0,
            noise_model=None, shots=256
        )
        assert len(result.times) == 10
        assert len(result.excited_pop) == 10

    def test_ramsey_output_bounds(self):
        """P(|1>) must stay in [0, 1]."""
        result = run_ramsey_circuit(
            delta=0.3, tau_max=5.0, n_tau=10,
            T1=50.0, T2=30.0,
            noise_model=None, shots=512
        )
        assert np.all(result.excited_pop >= 0.0)
        assert np.all(result.excited_pop <= 1.0)

    def test_ramsey_with_custom_noise(self):
        """Ramsey with custom noise should produce valid output."""
        nm = build_custom_noise_model(T1=50.0, T2=30.0)
        result = run_ramsey_circuit(
            delta=0.3, tau_max=5.0, n_tau=10,
            T1=50.0, T2=30.0,
            noise_model=nm, shots=512
        )
        assert isinstance(result, QiskitCalibResult)
        assert np.all(result.excited_pop >= 0.0)
        assert np.all(result.excited_pop <= 1.0)

    def test_ramsey_result_type(self):
        """Result should be a QiskitCalibResult."""
        result = run_ramsey_circuit(
            delta=0.3, tau_max=5.0, n_tau=5,
            T1=50.0, T2=30.0,
            noise_model=None, shots=256
        )
        assert isinstance(result, QiskitCalibResult)
