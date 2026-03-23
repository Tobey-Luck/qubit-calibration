# Qubit Calibration Simulator

A physically grounded qubit calibration simulator demonstrating core workflows
of a quantum computing calibration engineer: simulation, noise modeling, parameter
extraction, and hardware-facing software integration.

## Project Structure

```
qubit_calibration/
├── python/
│   ├── rabi/
│   │   ├── __init__.py
│   │   ├── simulator.py       # QuTiP-based Rabi oscillation simulation
│   │   ├── fitting.py         # SciPy curve fitting + parameter extraction
│   │   ├── t1t2_noise.py      # Lindblad T1/T2 decoherence simulation
│   │   ├── ramsey.py          # Ramsey experiment for precise T2 measurement
│   │   ├── bayesian.py        # Bayesian MCMC parameter estimation (emcee)
│   │   └── qiskit_backend.py  # Qiskit Aer circuit simulation + noise models
│   └── tests/
│       ├── conftest.py              # pytest path configuration
│       ├── test_rabi.py             # 17 tests: Rabi simulation and fitting
│       ├── test_t1t2.py             # 24 tests: T1/T2 decoherence
│       ├── test_ramsey.py           # 16 tests: Ramsey experiment
│       ├── test_bayesian.py         # 20 tests: Bayesian estimation
│       └── test_qiskit_backend.py   # 13 tests: Qiskit Aer integration
├── cpp/
│   ├── include/
│   │   └── bloch_solver.h     # C++ data structures + declarations
│   ├── src/
│   │   ├── bloch_solver.cpp   # RK4 Bloch equation + Ramsey solver
│   │   ├── bindings.cpp       # pybind11 Python bindings
│   │   └── test_bloch_main.cpp  # Standalone C++ validation binary
│   └── CMakeLists.txt
├── run_step1.py    # Rabi calibration demo
├── run_step2.py    # T1/T2 decoherence demo
├── run_step3.py    # Ramsey experiment demo
├── run_step4.py    # Bayesian estimation demo (~5 min)
├── run_step5.py    # Qiskit Aer integration demo
├── pytest.ini      # pytest path configuration
├── requirements.txt
└── README.md
```

## Roadmap

| Step | Description | Tests |
|------|-------------|-------|
| 1 | Rabi oscillation simulation + C++ RK4 solver | 17 |
| 2 | T1/T2 noise via Lindblad master equation | 24 |
| 3 | Ramsey experiment + T2 extraction | 16 |
| 4 | Bayesian parameter estimation (emcee MCMC) | 20 |
| 5 | Qiskit Aer noise model integration | 13 |
| **Total** | | **90** |

## Setup

### Requirements
- Python 3.11+
- CMake 3.15+
- GCC 12+ via MSYS2/MinGW (Windows) or system GCC (Linux/macOS)
- pybind11 (installed via pip)

### Python

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### C++ Solver

#### Windows (MSYS2 + MinGW)

1. Install [MSYS2](https://www.msys2.org/) then open the MSYS2 MinGW 64-bit terminal:
```bash
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-make
```

2. Add `C:\msys64\mingw64\bin` to the top of your system PATH.

3. If your project path contains spaces, create a junction:
```bash
mklink /J C:\qubit_cal "C:\path\to\qubit_calibration"
```

4. Build:
```bash
pip install pybind11
cmake -S "C:\qubit_cal\cpp" -B "C:\qubit_cal\cpp\build" -G "MinGW Makefiles"
mingw32-make -C "C:\qubit_cal\cpp\build"
```

#### macOS / Linux

```bash
pip install pybind11
cd cpp && mkdir -p build && cd build
cmake ..
make
```

### Known Limitation (Windows)

The C++ Ramsey solver is not called from `run_step3.py` due to a Windows ABI
incompatibility between MinGW-built pybind11 extensions and UCRT Python 3.13
on repeated object creation. The C++ Rabi solver validates correctly in
`run_step1.py` (max diff 1.61e-05). On Linux or with an MSVC-built extension
this limitation does not apply.

### Run

```bash
# Step demos
python run_step1.py    # Rabi calibration
python run_step2.py    # T1/T2 decoherence
python run_step3.py    # Ramsey T2 measurement
python run_step4.py    # Bayesian estimation (~5 min)
python run_step5.py    # Qiskit Aer integration

# Full test suite (90 tests)
pytest python/tests/ -v

# Standalone C++ validation
# Windows: C:\qubit_cal\cpp\build\test_bloch_cpp.exe
# Linux:   ./cpp/build/test_bloch_cpp
```

## Physics Background

### Step 1 - Rabi Oscillations
A qubit driven resonantly undergoes coherent oscillations:
```
P(|1>) = sin^2(omega_rabi * t / 2)
```
The Rabi frequency omega_rabi is extracted via SciPy curve fitting with
FFT-seeded initial parameter estimates. A C++ RK4 solver validates against
QuTiP to < 2e-05.

### Step 2 - T1/T2 Decoherence (Lindblad Master Equation)
Two decoherence processes are modelled via collapse operators in QuTiP:
- T1: Energy relaxation — C1 = sqrt(1/T1) * sigma_minus
- T2: Dephasing — C2 = sqrt(gamma_phi) * sigma_z

T2 extracted from Rabi data has ~15% error due to the competing drive.

### Step 3 - Ramsey Experiment
Three-stage pulse sequence: pi/2 -> free precession(tau) -> pi/2.
During free precession, T2 dephasing is applied analytically:
```
rho_01(tau) = rho_01(0) * exp(-tau/T2) * exp(-i*delta*tau)
```
T2 extracted from Ramsey data achieves ~1-2% error — a 10x improvement
over Rabi extraction.

### Step 4 - Bayesian Parameter Estimation (MCMC)
Uses emcee ensemble sampler to explore full posterior distributions over
calibration parameters. Three inference problems:
1. Rabi omega from ideal data
2. T2 from Ramsey fringe data
3. Joint omega + T2 from decohered Rabi data (reveals parameter correlations)

### Step 5 - Qiskit Aer Integration
Runs Rabi calibration experiments as Qiskit circuits on two noise models:
- Custom noise model: built from T1/T2 parameters + gate error rates,
  enabling direct comparison with QuTiP Lindblad simulation
- FakeManilaV2: realistic IBM device noise with hardware-calibrated
  T1/T2 values and gate error rates

The ~9% T2 discrepancy between custom noise and QuTiP is physically
meaningful: gate errors contribute additional dephasing beyond pure T2.

### C++ RK4 Bloch Solver
Integrates the optical Bloch equations using 4th-order Runge-Kutta:
```
dBx/dt =  delta*By - Bx/T2
dBy/dt = -delta*Bx + omega*Bz - By/T2
dBz/dt = -omega*By - (Bz - Bz_eq)/T1
```
Also implements the full Ramsey pulse sequence in C++.

## Results Summary

| Experiment | Parameter | True | Fitted | Error |
|------------|-----------|------|--------|-------|
| Step 1: Rabi (QuTiP) | omega_rabi | 2.0000 rad/us | 1.9997 rad/us | 0.015% |
| Step 2: Rabi + T1/T2 | T2 | 30.0 us | 25.35 us | 15.5% |
| Step 3: Ramsey | T2 | 30.0 us | 30.40 us | 1.3% |
| Step 4: Bayesian Rabi | omega_rabi | 2.0000 rad/us | 1.9996 rad/us | 0.02% |
| Step 4: Bayesian Ramsey | T2 | 30.0 us | 29.88 us | 0.4% |
| Step 5: Rabi (Qiskit custom) | omega_rabi | 2.0000 rad/us | 2.0011 rad/us | 0.05% |
| Step 5: Rabi (FakeManilaV2) | omega_rabi | 2.0000 rad/us | 1.9986 rad/us | 0.07% |
| Step 5: Ramsey (Qiskit) | T2 | 30.0 us | 27.26 us | 9.1%* |

*9% discrepancy reflects gate error contribution to dephasing — physically expected.
