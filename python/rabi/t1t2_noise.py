"""
rabi/t1t2_noise.py
------------------
Adds T1 (energy relaxation) and T2 (dephasing) decoherence to the Rabi
simulation using QuTiP's Lindblad master equation solver.

Physics background:
    In a real qubit, two processes destroy coherence:

    T1, Energy relaxation:
        The qubit spontaneously decays from |1> to |0> at rate gamma_1 = 1/T1.
        This is modelled by the collapse operator:
            C1 = sqrt(1/T1) * sigma_minus
        where sigma_minus = |0><1| is the lowering operator.

    T2, Dephasing:
        The qubit loses phase coherence without energy exchange at rate
        gamma_phi = 1/T2 - 1/(2*T1). This is modelled by:
            C2 = sqrt(gamma_phi) * sigma_z
        where sigma_z is the Pauli-Z operator.

    Note: T2 <= 2*T1 always. The dephasing rate gamma_phi is the *pure*
    dephasing contribution on top of the T1-induced dephasing. If T2 = 2*T1,
    pure dephasing is zero and decoherence is entirely due to energy relaxation.

    The combined effect on P(|1>) is a damped oscillation:
        P(|1>) ~ A * exp(-t/T2) * sin^2(omega*t/2) + offset
"""

import numpy as np
import qutip as qt
from dataclasses import dataclass
from typing import Optional

from .simulator import RabiResult


@dataclass
class DecoherenceParams:
    """
    Container for T1/T2 decoherence parameters.

    Attributes:
        T1: Energy relaxation time in microseconds
        T2: Total dephasing time in microseconds (must be <= 2*T1)
    """
    T1: float
    T2: float

    def __post_init__(self):
        """Validate physical constraints on T1 and T2."""
        if self.T1 <= 0:
            raise ValueError(f"T1 must be positive, got {self.T1}")
        if self.T2 <= 0:
            raise ValueError(f"T2 must be positive, got {self.T2}")
        if self.T2 > 2 * self.T1:
            raise ValueError(
                f"T2 ({self.T2}) cannot exceed 2*T1 ({2*self.T1}). "
                f"This violates the physical constraint T2 <= 2*T1."
            )

    @property
    def gamma_1(self) -> float:
        """Energy relaxation rate (rad/us)."""
        return 1.0 / self.T1

    @property
    def gamma_phi(self) -> float:
        """Pure dephasing rate (rad/us), excluding T1 contribution."""
        return 1.0 / self.T2 - 1.0 / (2.0 * self.T1)


def build_collapse_operators(params: DecoherenceParams) -> list:
    """
    Build QuTiP collapse operators for T1 and T2 decoherence.

    Collapse operators are the core of the Lindblad master equation.
    Each operator C_k contributes a dissipative term to the evolution:

        dρ/dt = -i[H, ρ] + Σ_k (C_k ρ C_k† - ½{C_k†C_k, ρ})

    The first term is unitary (Hamiltonian) evolution.
    The sum is the dissipator, it's what causes decoherence.

    Args:
        params: DecoherenceParams with T1 and T2 values

    Returns:
        List of QuTiP Qobj collapse operators [C1, C2]
    """
    # C1: energy relaxation |1> -> |0>
    # sigma_minus = |0><1| lowers the qubit from excited to ground state
    C1 = np.sqrt(params.gamma_1) * qt.sigmam()

    # C2: pure dephasing (phase randomization without energy exchange)
    # sigma_z causes random phase kicks, destroying off-diagonal coherences
    C2 = np.sqrt(params.gamma_phi) * qt.sigmaz()

    # Only include C2 if pure dephasing rate is non-negligible
    if params.gamma_phi > 1e-10:
        return [C1, C2]
    else:
        return [C1]


def run_rabi_with_decoherence(
    omega_rabi: float,
    decoherence: DecoherenceParams,
    t_max: float = 10.0,
    n_points: int = 200,
    initial_state: Optional[qt.Qobj] = None,
) -> RabiResult:
    """
    Simulate Rabi oscillations with T1/T2 decoherence via Lindblad master equation.

    This extends run_rabi() from simulator.py by adding collapse operators
    that model energy relaxation and dephasing. The result is a decaying
    oscillation rather than a perfect sinusoid.

    Args:
        omega_rabi:    Rabi frequency in rad/microsecond
        decoherence:   DecoherenceParams with T1 and T2
        t_max:         Total evolution time in microseconds
        n_points:      Number of time points
        initial_state: Initial qubit state (defaults to |0>)

    Returns:
        RabiResult with decaying P(|1>) trace
    """
    if initial_state is None:
        initial_state = qt.basis(2, 0)  # |0> ground state

    # Build Hamiltonian (same as ideal case)
    H = (omega_rabi / 2.0) * qt.sigmax()

    # Build collapse operators for decoherence
    c_ops = build_collapse_operators(decoherence)

    times = np.linspace(0, t_max, n_points)

    # mesolve with collapse operators: now solves the full Lindblad equation
    # The density matrix ρ is evolved instead of the state vector |ψ>
    result = qt.mesolve(
        H,
        initial_state,
        times,
        c_ops=c_ops,                     # Non-empty: decoherence is active
        e_ops=[qt.num(2)],               # Still measuring P(|1>)
        options={"store_states": True},
    )

    return RabiResult(
        times=times,
        excited_pop=np.array(result.expect[0]),
        omega_rabi=omega_rabi,
        states=result.states,
    )


def theoretical_decay_envelope(
    times: np.ndarray,
    decoherence: DecoherenceParams,
) -> np.ndarray:
    """
    Compute the theoretical T2 decay envelope for comparison with fits.

    For a resonantly driven qubit, the oscillation amplitude decays as:
        envelope(t) = 0.5 * exp(-t / T2)

    The factor 0.5 comes from the time-average of sin^2 being 0.5,
    the envelope sits at P=0.5 and swings ±0.5 around it.

    Args:
        times:        Time array in microseconds
        decoherence:  DecoherenceParams with T1 and T2

    Returns:
        Upper envelope of P(|1>) decay
    """
    return 0.5 * (1 + np.exp(-times / decoherence.T2))
