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
│   │   ├── simulator.py      # QuTiP-based Rabi oscillation simulation
│   │   ├── fitting.py        # SciPy curve fitting + parameter extraction
│   │   ├── t1t2_noise.py     # Lindblad T1/T2 decoherence simulation
│   │   ├── ramsey.py         # Ramsey experiment for precise T2 measurement
│   │   └── visualization.py  # Calibration fit plots
│   └── tests/
│       ├── test_rabi.py      # 17 tests: Rabi simulation and fitting
│       ├── test_t1t2.py      # 24 tests: T1/T2 decoherence
│       └── test_ramsey.py    # 16 tests: Ramsey experiment
├── cpp/
│   ├── include/
│   │   └── bloch_solver.h    # C++ data structures + declarations
│   ├── src/
│   │   ├── bloch_solver.cpp  # RK4 Bloch equation + Ramsey solver
│   │   ├── bindings.cpp      # pybind11 Python bindings
│   │   └── test_bloch_main.cpp  # Standalone C++ validation binary
│   └── CMakeLists.txt
├── run_step1.py              # Rabi calibration demo
├── run_step2.py              # T1/T2 decoherence demo
├── run_step3.py              # Ramsey experiment demo
├── requirements.txt
└── README.md
```

## Roadmap

| Step | Description | Status |
|------|-------------|--------|
| 1 | Rabi oscillation simulation + curve fitting | Done |
| 2 | T1/T2 noise via Lindblad master equation | Done |
| 3 | Ramsey experiment + T2 extraction | Done |
| 4 | Bayesian parameter estimation (emcee) | Next |
| 5 | Qiskit Aer noise model integration | Planned |

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

# Full test suite (57 tests)
pytest python/tests/ -v

# Standalone C++ validation
# Windows: C:\qubit_cal\cpp\build\test_bloch_cpp.exe
# Linux:   ./cpp/build/test_bloch_cpp
```

## Physics Background

### Step 1 — Rabi Oscillations
A qubit driven resonantly undergoes coherent oscillations:
```
P(|1>) = sin^2(omega_rabi * t / 2)
```
The Rabi frequency omega_rabi is extracted via SciPy curve fitting with
FFT-seeded initial parameter estimates.

### Step 2 — T1/T2 Decoherence (Lindblad Master Equation)
Two decoherence processes are modelled via collapse operators in QuTiP:
- **T1**: Energy relaxation — C1 = sqrt(1/T1) * sigma_minus
- **T2**: Dephasing — C2 = sqrt(gamma_phi) * sigma_z

The combined signal is a damped oscillation. T2 extracted from Rabi data
has ~15% error due to the competing drive.

### Step 3 — Ramsey Experiment
Three-stage pulse sequence: pi/2 -> free precession(tau) -> pi/2.
During free precession, T2 dephasing is applied analytically:
```
rho_01(tau) = rho_01(0) * exp(-tau/T2) * exp(-i*delta*tau)
```
T2 extracted from Ramsey data achieves ~1-2% error — a 10x improvement
over Rabi extraction, demonstrating why dedicated T2 experiments are used
on real quantum hardware.

### C++ RK4 Bloch Solver
Integrates the optical Bloch equations using 4th-order Runge-Kutta:
```
dBx/dt =  delta*By - Bx/T2
dBy/dt = -delta*Bx + omega*Bz - By/T2
dBz/dt = -omega*By - (Bz - Bz_eq)/T1
```
Agrees with QuTiP to < 2e-05 for Rabi oscillations.

## Results Summary

| Experiment | Parameter | True | Fitted | Error |
|------------|-----------|------|--------|-------|
| Step 1: Rabi | omega_rabi | 2.0000 rad/us | 1.9997 rad/us | 0.015% |
| Step 2: Rabi + T1/T2 | T2 | 30.0 us | 25.35 us | 15.5% |
| Step 3: Ramsey | T2 | 30.0 us | 30.40 us | 1.3% |
