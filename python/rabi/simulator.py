"""
rabi/simulator.py
-----------------
Rabi oscillation simulator using QuTiP.

Physics background:
    A qubit driven by a resonant microwave field undergoes coherent oscillations
    between |0> and |1>. The Hamiltonian is:

        H = (omega_d / 2) * sigma_x

    where omega_d is the drive (Rabi) frequency and sigma_x is the Pauli-X
    operator. The probability of measuring |1> oscillates as:

        P(|1>) = sin^2(omega_d * t / 2)

    In a real system, decoherence damps these oscillations. Here we simulate
    the ideal (noiseless) case; noise is added in the T1/T2 module (step 2).
"""

import numpy as np
import qutip as qt
from dataclasses import dataclass


@dataclass
class RabiResult:
    """Container for Rabi simulation output."""
    times: np.ndarray          # Time points (in microseconds)
    excited_pop: np.ndarray    # P(|1>) at each time point
    omega_rabi: float          # True Rabi frequency (rad/us)
    states: list               # Full density matrix at each time (for Bloch sphere)


def build_hamiltonian(omega_rabi: float) -> qt.Qobj:
    """
    Build the driven qubit Hamiltonian.

    H = (omega_rabi / 2) * sigma_x

    Args:
        omega_rabi: Rabi frequency in rad/microsecond

    Returns:
        QuTiP Qobj representing the Hamiltonian
    """
    return (omega_rabi / 2.0) * qt.sigmax()


def run_rabi(
    omega_rabi: float,
    t_max: float = 10.0,
    n_points: int = 200,
    initial_state: qt.Qobj = None,
) -> RabiResult:
    """
    Simulate ideal Rabi oscillations using QuTiP's mesolve.

    Args:
        omega_rabi:     Rabi frequency in rad/microsecond
        t_max:          Total evolution time in microseconds
        n_points:       Number of time points
        initial_state:  Initial qubit state (defaults to |0>)

    Returns:
        RabiResult with time array, P(|1>) trace, and full states
    """
    if initial_state is None:
        initial_state = qt.basis(2, 0)  # |0> = ground state

    H = build_hamiltonian(omega_rabi)
    times = np.linspace(0, t_max, n_points)

    # mesolve: master equation solver. No collapse operators = ideal evolution.
    result = qt.mesolve(
        H,
        initial_state,
        times,
        c_ops=[],                        # No decoherence yet
        e_ops=[qt.num(2)],               # Measure <n> = P(|1>)
        options={"store_states": True},
    )

    return RabiResult(
        times=times,
        excited_pop=np.array(result.expect[0]),
        omega_rabi=omega_rabi,
        states=result.states,
    )


def add_measurement_noise(
    excited_pop: np.ndarray,
    noise_level: float = 0.02,
    seed: int = None,
) -> np.ndarray:
    """
    Add Gaussian measurement noise to simulate finite shot statistics.

    In a real experiment, each data point is estimated from N repeated shots,
    producing shot noise ~ 1/sqrt(N). Here we approximate this as additive
    Gaussian noise.

    Args:
        excited_pop:  Clean P(|1>) trace
        noise_level:  Standard deviation of noise (default 0.02 ~ 500 shots)
        seed:         Random seed for reproducibility

    Returns:
        Noisy P(|1>) trace, clipped to [0, 1]
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_level, size=len(excited_pop))
    return np.clip(excited_pop + noise, 0.0, 1.0)
