// cpp/src/bindings.cpp
// --------------------
// pybind11 bindings: exposes the C++ Bloch solver to Python.
//
// After building (see README), import in Python as:
//
//     from qubit_calibration.cpp_solver import BlochParams, BlochState, solve_bloch
//
// pybind11 beginner note:
//   PYBIND11_MODULE defines a Python module. py::class_ wraps a C++ struct/class.
//   .def() registers a method or property. def_readwrite() exposes a member variable.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // Automatic std::vector <-> Python list conversion
#include "../include/bloch_solver.h"

namespace py = pybind11;
using namespace qubit_cal;

PYBIND11_MODULE(cpp_solver, m) {
    m.doc() = "C++ RK4 Bloch equation solver for qubit dynamics";

    // --- BlochState ---
    py::class_<BlochState>(m, "BlochState",
        "Bloch vector representing qubit state. Ground=|0>=(0,0,-1), Excited=|1>=(0,0,1)")
        .def(py::init<>())
        .def(py::init([](double bx, double by, double bz) {
            BlochState s; s.bx = bx; s.by = by; s.bz = bz; return s;
        }), py::arg("bx"), py::arg("by"), py::arg("bz"))
        .def_readwrite("bx", &BlochState::bx)
        .def_readwrite("by", &BlochState::by)
        .def_readwrite("bz", &BlochState::bz)
        .def("__repr__", [](const BlochState& s) {
            return "BlochState(bx=" + std::to_string(s.bx) +
                   ", by=" + std::to_string(s.by) +
                   ", bz=" + std::to_string(s.bz) + ")";
        });

    // --- BlochParams ---
    py::class_<BlochParams>(m, "BlochParams", "Physical parameters for Bloch equations")
        .def(py::init<>())
        .def_readwrite("omega_rabi", &BlochParams::omega_rabi,
                       "Rabi frequency in rad/microsecond")
        .def_readwrite("delta",      &BlochParams::delta,
                       "Drive detuning in rad/microsecond (0 = resonant)")
        .def_readwrite("T1",         &BlochParams::T1,
                       "Energy relaxation time in microseconds (0 = ideal/infinite)")
        .def_readwrite("T2",         &BlochParams::T2,
                       "Dephasing time in microseconds (0 = ideal/infinite)")
        .def_readwrite("bz_eq",      &BlochParams::bz_eq,
                       "Equilibrium Bz; -1.0 for ground state")
        .def("__repr__", [](const BlochParams& p) {
            return "BlochParams(omega_rabi=" + std::to_string(p.omega_rabi) +
                   ", delta=" + std::to_string(p.delta) +
                   ", T1=" + std::to_string(p.T1) +
                   ", T2=" + std::to_string(p.T2) + ")";
        });

    // --- SolverResult ---
    py::class_<SolverResult>(m, "SolverResult", "Output of solve_bloch()")
        .def_readonly("times",       &SolverResult::times,
                      "Time points in microseconds")
        .def_readonly("bx",          &SolverResult::bx, "Bloch vector x component")
        .def_readonly("by",          &SolverResult::by, "Bloch vector y component")
        .def_readonly("bz",          &SolverResult::bz, "Bloch vector z component")
        .def_readonly("excited_pop", &SolverResult::excited_pop,
                      "P(|1>) = (1 + Bz) / 2 at each time point");

    // --- solve_bloch ---
    m.def("solve_bloch", &solve_bloch,
        py::arg("params"),
        py::arg("initial_state"),
        py::arg("t_max"),
        py::arg("n_points"),
        R"doc(
        Solve the optical Bloch equations using 4th-order Runge-Kutta.

        Args:
            params:        BlochParams with Rabi frequency, detuning, T1, T2
            initial_state: BlochState (default ground state: bz=-1)
            t_max:         Total simulation time in microseconds
            n_points:      Number of output time points

        Returns:
            SolverResult with times, Bloch vector components, and P(|1>)
        )doc"
    );
}
