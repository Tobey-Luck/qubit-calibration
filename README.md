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
│   │   └── visualization.py  # Calibration fit plots
│   └── tests/
│       └── test_rabi.py      # 17 pytest unit + integration tests
├── cpp/
│   ├── include/
│   │   └── bloch_solver.h    # C++ data structures + declarations
│   ├── src/
│   │   ├── bloch_solver.cpp  # RK4 Bloch equation solver
│   │   ├── bindings.cpp      # pybind11 Python bindings
│   │   └── test_bloch_main.cpp  # Standalone C++ validation binary
│   └── CMakeLists.txt
├── run_step1.py              # Full Step 1 demo
├── requirements.txt
└── README.md
```

## Roadmap

| Step | Description | Status |
|------|-------------|--------|
| 1 | Rabi oscillation simulation + curve fitting | ✅ Done |
| 2 | T1/T2 noise via Lindblad master equation | 🔜 Next |
| 3 | Ramsey experiment + frequency estimation | 🔜 |
| 4 | Bayesian parameter estimation (emcee) | 🔜 |
| 5 | Qiskit Aer noise model integration | 🔜 |

## Setup

### Requirements
- Python 3.11+
- CMake 3.15+
- GCC 12+ / MSVC 2019+ (C++17 required)
- pybind11 (installed via pip)

### Python

```bash
# Create and activate a virtual environment first (recommended)
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### C++ Solver

#### Windows (MSYS2 + MinGW)

1. Install [MSYS2](https://www.msys2.org/) then open the MSYS2 MinGW 64-bit terminal and run:
```bash
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-make
```

2. Add `C:\msys64\mingw64\bin` to the **top** of your system PATH environment variable.

3. Install pybind11 into your venv:
```bash
pip install pybind11
```

4. If your project path contains spaces, create a junction to avoid CMake issues:
```bash
mklink /J C:\qubit_cal "C:\path\to\qubit_calibration"
```

5. Build:
```bash
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

### Run

```bash
# Full Step 1 demo
python run_step1.py

# Tests (17 tests, all should pass)
pytest python/tests/ -v

# Standalone C++ validation (Windows)
C:\qubit_cal\cpp\build\test_bloch_cpp.exe
```

## Physics Background

### Rabi Oscillations
A qubit driven resonantly by a microwave field undergoes coherent oscillations:

```
P(|1⟩) = sin²(ω_rabi · t / 2)
```

The **Rabi frequency** ω_rabi is a key calibration parameter — it determines
how long to apply a drive to implement a π-pulse (qubit flip) or π/2-pulse
(superposition). In real hardware this must be calibrated regularly as it
drifts with temperature, laser power, and field fluctuations.

### Bloch Equations (C++ solver)
The C++ solver integrates the optical Bloch equations using 4th-order Runge-Kutta:

```
dBx/dt =  δ·By  - Bx/T2
dBy/dt = -δ·Bx + ω·Bz - By/T2
dBz/dt = -ω·By - (Bz - Bz_eq)/T1
```

Setting T1=T2=0 (infinite decoherence times) recovers ideal Rabi oscillations.
The C++ solver agrees with the QuTiP result to < 2×10⁻⁵.

## Step 1 Results

```
True  ω:   2.000000 rad/µs
Fitted ω:  1.999705 ± 0.001207 rad/µs
Error:     0.015%
χ²:        0.000276
C++ max deviation from QuTiP: 1.61e-05  ✓
```

## Key Design Decisions

- **Dual Python/C++ implementation**: Python (QuTiP + SciPy) is the primary
  analysis layer; the C++ RK4 solver is a performant kernel that validates
  against the Python result to < 2×10⁻⁵.
- **Modular layout**: simulator, fitting, and visualization are separated so
  each can be tested and extended independently.
- **Realistic noise model**: Gaussian shot noise parameterised by `noise_level`,
  mapping to physical shot count via σ ≈ 1/√N_shots.
- **FFT-seeded curve fitting**: Initial parameter estimates use the power
  spectral density to find the dominant frequency, ensuring robust convergence
  across a wide range of Rabi frequencies.
