// cpp/src/test_bloch_main.cpp
// ---------------------------
// Standalone C++ test: verifies the RK4 solver against the known analytical
// solution for ideal Rabi oscillations (no decoherence, resonant drive).
//
// Analytical result: P(|1>) = sin^2(omega_rabi * t / 2)
//
// Build and run:
//   cd cpp/build && cmake .. && make test_bloch_cpp && ./test_bloch_cpp

#include <iostream>
#include <cmath>
#include <iomanip>
#include "bloch_solver.h"

int main() {
    using namespace qubit_cal;

    const double omega_rabi = 2.0;   // rad/us
    const double t_max      = 10.0;  // us
    const int    n_points   = 500;

    BlochParams params;
    params.omega_rabi = omega_rabi;
    params.delta      = 0.0;   // resonant
    params.T1         = 0.0;   // ideal (no decay)
    params.T2         = 0.0;   // ideal (no dephasing)

    BlochState initial;
    initial.bx = 0.0;
    initial.by = 0.0;
    initial.bz = -1.0;  // ground state |0>

    SolverResult result = solve_bloch(params, initial, t_max, n_points);

    double dt        = t_max / (n_points - 1);
    double max_error = 0.0;
    double rms_error = 0.0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "t (us)    P_RK4      P_analytic  error\n";
    std::cout << "-------   --------   ----------  --------\n";

    for (int i = 0; i < n_points; i += n_points / 20) {
        double t          = result.times[i];
        double p_rk4      = result.excited_pop[i];
        double p_analytic = std::pow(std::sin(omega_rabi * t / 2.0), 2);
        double error      = std::abs(p_rk4 - p_analytic);

        max_error = std::max(max_error, error);
        rms_error += error * error;

        std::cout << t << "  " << p_rk4 << "  " << p_analytic
                  << "  " << error << "\n";
    }

    rms_error = std::sqrt(rms_error / (n_points / 20));

    std::cout << "\n--- Summary ---\n";
    std::cout << "Max error: " << max_error << "\n";
    std::cout << "RMS error: " << rms_error << "\n";

    bool passed = max_error < 1e-6;
    std::cout << "Test: " << (passed ? "PASSED ✓" : "FAILED ✗") << "\n";

    return passed ? 0 : 1;
}
