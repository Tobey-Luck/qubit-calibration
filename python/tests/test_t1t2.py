"""
tests/test_t1t2.py
------------------
Unit and integration tests for the T1/T2 decoherence module.

Run with: pytest python/tests/ -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from rabi.simulator import add_measurement_noise
from rabi.fitting import fit_rabi
from rabi.t1t2_noise import (
    DecoherenceParams,
    run_rabi_with_decoherence,
    theoretical_decay_envelope,
    build_collapse_operators,
)


# ---------------------------------------------------------------------------
# DecoherenceParams validation tests
# ---------------------------------------------------------------------------

class TestDecoherenceParams:

    def test_valid_params(self):
        """Valid T1/T2 should construct without error."""
        p = DecoherenceParams(T1=50.0, T2=30.0)
        assert p.T1 == 50.0
        assert p.T2 == 30.0

    def test_t2_equals_2t1_is_valid(self):
        """T2 = 2*T1 is physically valid (pure T1-limited dephasing)."""
        p = DecoherenceParams(T1=10.0, T2=20.0)
        assert p.gamma_phi == pytest.approx(0.0, abs=1e-10)

    def test_t2_exceeds_2t1_raises(self):
        """T2 > 2*T1 violates physics and should raise ValueError."""
        with pytest.raises(ValueError, match="T2 <= 2\\*T1"):
            DecoherenceParams(T1=10.0, T2=25.0)

    def test_negative_t1_raises(self):
        """Negative T1 is unphysical."""
        with pytest.raises(ValueError):
            DecoherenceParams(T1=-5.0, T2=3.0)

    def test_negative_t2_raises(self):
        """Negative T2 is unphysical."""
        with pytest.raises(ValueError):
            DecoherenceParams(T1=10.0, T2=-1.0)

    def test_gamma_1(self):
        """gamma_1 = 1/T1."""
        p = DecoherenceParams(T1=50.0, T2=30.0)
        assert p.gamma_1 == pytest.approx(1.0 / 50.0)

    def test_gamma_phi(self):
        """gamma_phi = 1/T2 - 1/(2*T1)."""
        p = DecoherenceParams(T1=50.0, T2=30.0)
        expected = 1.0 / 30.0 - 1.0 / (2.0 * 50.0)
        assert p.gamma_phi == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Collapse operator tests
# ---------------------------------------------------------------------------

class TestCollapseOperators:

    def test_t1_only_gives_one_operator(self):
        """When T2 = 2*T1, pure dephasing is zero so only C1 is returned."""
        p = DecoherenceParams(T1=10.0, T2=20.0)
        c_ops = build_collapse_operators(p)
        assert len(c_ops) == 1

    def test_t1_and_t2_gives_two_operators(self):
        """When T2 < 2*T1, both C1 and C2 are returned."""
        p = DecoherenceParams(T1=50.0, T2=30.0)
        c_ops = build_collapse_operators(p)
        assert len(c_ops) == 2

    def test_collapse_operators_are_qobj(self):
        """Collapse operators must be QuTiP Qobj instances."""
        import qutip as qt
        p = DecoherenceParams(T1=10.0, T2=6.0)
        c_ops = build_collapse_operators(p)
        for op in c_ops:
            assert isinstance(op, qt.Qobj)


# ---------------------------------------------------------------------------
# Simulation tests
# ---------------------------------------------------------------------------

class TestRabiWithDecoherence:

    def test_ground_state_at_t0(self):
        """At t=0, P(|1>) must be 0 regardless of decoherence."""
        p = DecoherenceParams(T1=50.0, T2=30.0)
        result = run_rabi_with_decoherence(omega_rabi=2.0, decoherence=p)
        assert result.excited_pop[0] == pytest.approx(0.0, abs=1e-4)

    def test_decay_reduces_amplitude(self):
        """Oscillation amplitude at t=T2 should be significantly reduced."""
        T2 = 5.0
        p = DecoherenceParams(T1=10.0, T2=T2)
        result = run_rabi_with_decoherence(
            omega_rabi=2.0, decoherence=p, t_max=20.0, n_points=400
        )
        # At t=T2, amplitude should be reduced by factor e^-1 ~ 0.37
        # Max P(|1>) in second half should be much less than 1
        second_half = result.excited_pop[len(result.excited_pop)//2:]
        assert np.max(second_half) < 0.6

    def test_decohered_less_than_ideal(self):
        """Decohered oscillation peak amplitude should be less than ideal."""
        from rabi.simulator import run_rabi
        p = DecoherenceParams(T1=5.0, T2=3.0)
        ideal   = run_rabi(omega_rabi=2.0, t_max=15.0, n_points=300)
        decohered = run_rabi_with_decoherence(
            omega_rabi=2.0, decoherence=p, t_max=15.0, n_points=300
        )
        # Max P(|1>) should be lower in decohered case
        assert np.max(decohered.excited_pop) < np.max(ideal.excited_pop)

    def test_long_time_equilibrium(self):
        """At t >> T1, qubit should approach thermal equilibrium (~0)."""
        T1 = 3.0
        p = DecoherenceParams(T1=T1, T2=2.0)
        result = run_rabi_with_decoherence(
            omega_rabi=2.0, decoherence=p, t_max=30.0, n_points=500
        )
        # Last 10% of trace should be close to 0.5 (mixed state)
        tail = result.excited_pop[int(0.9 * len(result.excited_pop)):]
        assert np.mean(tail) < 0.6

    def test_output_bounds(self):
        """P(|1>) must stay within [0, 1] even with decoherence."""
        p = DecoherenceParams(T1=5.0, T2=3.0)
        result = run_rabi_with_decoherence(omega_rabi=2.0, decoherence=p)
        assert np.all(result.excited_pop >= -1e-6)
        assert np.all(result.excited_pop <=  1.0 + 1e-6)

    def test_result_has_correct_shape(self):
        """Output arrays should have n_points elements."""
        p = DecoherenceParams(T1=10.0, T2=6.0)
        result = run_rabi_with_decoherence(
            omega_rabi=2.0, decoherence=p, n_points=150
        )
        assert len(result.times) == 150
        assert len(result.excited_pop) == 150


# ---------------------------------------------------------------------------
# Decay envelope tests
# ---------------------------------------------------------------------------

class TestDecayEnvelope:

    def test_envelope_at_t0(self):
        """At t=0, envelope should be 1.0 (no decay yet)."""
        p = DecoherenceParams(T1=10.0, T2=6.0)
        times    = np.array([0.0])
        envelope = theoretical_decay_envelope(times, p)
        assert envelope[0] == pytest.approx(1.0, abs=1e-10)

    def test_envelope_at_T2(self):
        """At t=T2, envelope should be 0.5*(1 + e^-1) ~ 0.684."""
        T2 = 10.0
        p  = DecoherenceParams(T1=20.0, T2=T2)
        times    = np.array([T2])
        envelope = theoretical_decay_envelope(times, p)
        expected = 0.5 * (1 + np.exp(-1.0))
        assert envelope[0] == pytest.approx(expected, rel=1e-6)

    def test_envelope_is_decreasing(self):
        """Decay envelope should be monotonically decreasing."""
        p     = DecoherenceParams(T1=10.0, T2=6.0)
        times = np.linspace(0, 30, 100)
        env   = theoretical_decay_envelope(times, p)
        assert np.all(np.diff(env) <= 0)


# ---------------------------------------------------------------------------
# Integration test: simulate with decoherence -> fit -> recover T2
# ---------------------------------------------------------------------------

class TestDecoherenceFitting:

    @pytest.mark.parametrize("T1,T2", [
        (50.0, 30.0),   # realistic
        (5.0,  3.0),    # exaggerated
        (10.0, 10.0),   # T2 = T1 (Hahn echo limit... ish)
    ])
    def test_fit_recovers_T2(self, T1, T2):
        p      = DecoherenceParams(T1=T1, T2=T2)
        t_max  = min(4 * T2, 80.0)
        result = run_rabi_with_decoherence(
            omega_rabi=2.0, decoherence=p,
            t_max=t_max, n_points=600,
        )
        noisy = add_measurement_noise(result.excited_pop, noise_level=0.02, seed=42)
        fit   = fit_rabi(result.times, noisy)
        assert fit.decay_time == pytest.approx(T2, rel=0.20)  # 20% tolerance

    @pytest.mark.parametrize("T1,T2", [
        (50.0, 30.0),
        (5.0,  3.0),
    ])
    def test_fit_recovers_omega_with_decoherence(self, T1, T2):
        """Rabi frequency recovery should still work with decoherence present."""
        p      = DecoherenceParams(T1=T1, T2=T2)
        t_max  = min(3 * T2, 60.0)
        result = run_rabi_with_decoherence(
            omega_rabi=2.0, decoherence=p,
            t_max=t_max, n_points=400,
        )
        noisy = add_measurement_noise(result.excited_pop, noise_level=0.02, seed=42)
        fit   = fit_rabi(result.times, noisy)

        assert fit.omega_rabi_fit == pytest.approx(2.0, rel=0.03)
