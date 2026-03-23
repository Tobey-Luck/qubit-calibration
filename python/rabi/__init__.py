# python/rabi/__init__.py
from .simulator import run_rabi, add_measurement_noise, RabiResult
from .fitting import fit_rabi, FitResult
from .t1t2_noise import DecoherenceParams, run_rabi_with_decoherence
from .ramsey import run_ramsey, fit_ramsey, RamseyResult, RamseyFitResult
from .bayesian import run_mcmc_rabi, run_mcmc_ramsey, run_mcmc_joint, MCMCResult

# Qiskit backend (optional — requires qiskit and qiskit-aer)
try:
    from .qiskit_backend import (
        build_custom_noise_model,
        run_rabi_circuit,
        run_ramsey_circuit,
        QiskitCalibResult,
    )
except ImportError:
    pass
