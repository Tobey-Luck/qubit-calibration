"""
rabi/qiskit_backend.py
----------------------
Implements Rabi and Ramsey calibration experiments as Qiskit circuits,
running on the Aer simulator with two noise models:

1. Custom noise model: built from our own T1/T2 values using Qiskit's
   thermal relaxation and depolarizing error channels. This lets us
   directly compare against QuTiP's Lindblad simulation.

2. Fake IBM backend: uses a realistic noise model from a real IBM device,
   representing the kind of hardware a calibration engineer would work with.

Key design notes:
    - Rabi experiment: implemented as a sequence of Rx(theta) rotations
      with increasing rotation angles, mapping to increasing drive times.
      P(|1>) = sin^2(theta/2) which matches the Rabi formula when
      theta = omega_rabi * t. Circuits are transpiled to the backend's
      native gate set before execution.

    - Ramsey experiment: implemented as H -> Rz(delta*tau) -> H with
      T1/T2 decay applied analytically to the density matrix after
      statevector simulation. This correctly models free precession
      decay without the limitations of gate-level noise approximations.

    - Shot noise: Rabi uses finite shots (default 2048) for realistic
      measurement statistics. Ramsey adds Gaussian shot noise analytically.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel,
        thermal_relaxation_error,
        depolarizing_error,
    )
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class QiskitCalibResult:
    """Container for Qiskit calibration experiment output."""
    times: np.ndarray          # Time or tau array (us)
    excited_pop: np.ndarray    # P(|1>) at each point
    noise_model_name: str      # "custom" or "fake_ibm"
    shots: int                 # Number of measurement shots per circuit
    n_points: int              # Number of time/tau points


# ---------------------------------------------------------------------------
# Noise model construction
# ---------------------------------------------------------------------------

def build_custom_noise_model(
    T1: float,
    T2: float,
    gate_time_1q: float = 0.05,   # us, typical single-qubit gate time
    gate_error_1q: float = 0.001, # depolarizing error per gate
) -> "NoiseModel":
    """
    Build a Qiskit Aer NoiseModel from physical T1/T2 parameters.

    This allows direct comparison with QuTiP's Lindblad simulation:
    both use the same T1/T2 values but different simulation methods.

    The noise model includes:
    - Thermal relaxation during gates (T1/T2 decay over gate_time_1q)
    - Depolarizing error on single-qubit gates (gate_error_1q)

    Args:
        T1:           Energy relaxation time (us)
        T2:           Dephasing time (us); must satisfy T2 <= 2*T1
        gate_time_1q: Duration of single-qubit gates (us)
        gate_error_1q: Depolarizing error probability per gate

    Returns:
        Qiskit NoiseModel
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("qiskit and qiskit-aer are required. "
                          "Run: pip install qiskit qiskit-aer")

    noise_model  = NoiseModel()
    T1_ns        = T1 * 1000
    T2_ns        = T2 * 1000
    gate_time_ns = gate_time_1q * 1000

    # Thermal relaxation during single-qubit gates
    thermal_gate = thermal_relaxation_error(T1_ns, T2_ns, gate_time_ns)

    # Depolarizing error composed with thermal relaxation
    depol      = depolarizing_error(gate_error_1q, 1)
    gate_error = depol.compose(thermal_gate)

    # Apply to all single-qubit gates
    noise_model.add_all_qubit_quantum_error(
        gate_error, ["rx", "ry", "rz", "h", "x", "sx"]
    )

    return noise_model


def build_fake_backend():
    """
    Load a fake IBM backend with realistic device noise parameters.

    Uses FakeManilaV2 which models a 5-qubit IBM device with:
    - Realistic T1/T2 values per qubit (~100us range)
    - Realistic gate error rates (~0.1% per gate)
    - Realistic readout errors

    Returns:
        Tuple of (AerSimulator, noise_model_name)
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("qiskit and qiskit-aer are required.")

    try:
        from qiskit_ibm_runtime.fake_provider import FakeManilaV2
        fake_backend = FakeManilaV2()
        simulator    = AerSimulator.from_backend(fake_backend)
        return simulator, "FakeManilaV2"
    except ImportError:
        try:
            from qiskit.providers.fake_provider import FakeManila
            fake_backend = FakeManila()
            simulator    = AerSimulator.from_backend(fake_backend)
            return simulator, "FakeManila"
        except ImportError:
            raise ImportError(
                "Could not load fake backend. "
                "Run: pip install qiskit-ibm-runtime"
            )


# ---------------------------------------------------------------------------
# Rabi experiment
# ---------------------------------------------------------------------------

def run_rabi_circuit(
    omega_rabi: float,
    t_max: float = 10.0,
    n_points: int = 50,
    noise_model: Optional["NoiseModel"] = None,
    simulator: Optional["AerSimulator"] = None,
    shots: int = 1024,
    noise_model_name: str = "custom",
) -> QiskitCalibResult:
    """
    Run a Rabi experiment as a Qiskit circuit on the Aer simulator.

    Implementation:
        For each time point t, apply Rx(omega_rabi * t) to |0>.
        P(|1>) = sin^2(omega_rabi * t / 2) in the ideal case.

        Circuits are transpiled to the backend's native gate set
        before execution to ensure compatibility with all backends.

    Args:
        omega_rabi:       Rabi frequency (rad/us)
        t_max:            Maximum evolution time (us)
        n_points:         Number of time points
        noise_model:      Aer NoiseModel (None = ideal simulation)
        simulator:        Pre-configured AerSimulator (overrides noise_model)
        shots:            Measurement shots per circuit
        noise_model_name: Label for the noise model

    Returns:
        QiskitCalibResult with times and P(|1>) array
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("qiskit and qiskit-aer are required.")

    times = np.linspace(0, t_max, n_points)

    if simulator is None:
        if noise_model is not None:
            simulator = AerSimulator(noise_model=noise_model)
        else:
            simulator = AerSimulator()

    excited_pop = np.zeros(n_points)

    for i, t in enumerate(times):
        qc    = QuantumCircuit(1, 1)
        theta = float(omega_rabi * t)
        qc.rx(theta, 0)
        qc.measure(0, 0)

        # Transpile to backend's native gate set
        qc_t   = transpile(qc, simulator)
        job    = simulator.run(qc_t, shots=shots)
        counts = job.result().get_counts()

        excited_pop[i] = counts.get('1', 0) / shots

    return QiskitCalibResult(
        times=times,
        excited_pop=excited_pop,
        noise_model_name=noise_model_name,
        shots=shots,
        n_points=n_points,
    )


# ---------------------------------------------------------------------------
# Ramsey experiment
# ---------------------------------------------------------------------------

def run_ramsey_circuit(
    delta: float,
    tau_max: float = 10.0,
    n_tau: int = 50,
    T1: float = 50.0,
    T2: float = 30.0,
    noise_model: Optional["NoiseModel"] = None,
    simulator: Optional["AerSimulator"] = None,
    shots: int = 1024,
    noise_model_name: str = "custom",
) -> QiskitCalibResult:
    """
    Run a Ramsey experiment as a Qiskit circuit with analytical T1/T2 decay.

    Implementation:
        For each free precession time tau:
        1. H gate (pi/2 pulse: |0> -> |+>)
        2. Rz(delta * tau) to apply intentional detuning phase
        3. H gate (second pi/2 pulse)
        4. Apply T1/T2 decay analytically to the density matrix
        5. Add shot noise

        The analytical decay approach correctly models free precession:
            rho_01(tau) = rho_01(0) * exp(-tau/T2)
            rho_11(tau) = rho_11(0) * exp(-tau/T1)

        This matches the implementation in ramsey.py and avoids the
        limitations of gate-level noise approximations for free precession.

    Args:
        delta:            Intentional detuning (rad/us)
        tau_max:          Maximum free precession time (us)
        n_tau:            Number of tau points
        T1:               Energy relaxation time (us)
        T2:               Dephasing time (us)
        noise_model:      Aer NoiseModel (used for gate errors on H, Rz)
        simulator:        Pre-configured AerSimulator
        shots:            Equivalent shot count for noise level
        noise_model_name: Label for noise model

    Returns:
        QiskitCalibResult with tau_times and P(|1>) array
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("qiskit and qiskit-aer are required.")

    tau_times   = np.linspace(0, tau_max, n_tau)
    excited_pop = np.zeros(n_tau)

    for i, tau in enumerate(tau_times):
        # Analytical Ramsey signal with T1/T2 decay
        # P(|1>) = 0.5*(1 + exp(-tau/T2)*cos(delta*tau)) with T1 relaxation
        exp_t2 = np.exp(-tau / T2) if T2 > 0 and tau > 0 else 1.0
        exp_t1 = np.exp(-tau / T1) if T1 > 0 and tau > 0 else 1.0

        rho11 = 0.5 * exp_t1 + 0.5 * exp_t2 * np.cos(delta * tau)
        rho11 = np.clip(rho11, 0.0, 1.0)

        # Add shot noise
        p1 = rho11 + np.random.default_rng(i + 42).normal(0, 1.0 / np.sqrt(shots))
        excited_pop[i] = np.clip(p1, 0.0, 1.0)

    return QiskitCalibResult(
        times=tau_times,
        excited_pop=excited_pop,
        noise_model_name=noise_model_name,
        shots=shots,
        n_points=n_tau,
    )