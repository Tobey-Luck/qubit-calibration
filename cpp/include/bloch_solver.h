// cpp/include/bloch_solver.h
// --------------------------
// Data structures and function declarations for the RK4 Bloch equation solver.

#pragma once
#include <vector>

namespace qubit_cal {

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Bloch vector: represents the qubit state as a 3D unit vector.
/// Ground state |0> = (0, 0, -1), Excited state |1> = (0, 0, 1)
struct BlochState {
    double bx = 0.0;
    double by = 0.0;
    double bz = -1.0;  // Default: ground state
};

/// Physical parameters for the Bloch equations
struct BlochParams {
    double omega_rabi = 1.0;   // Rabi frequency (rad/us)
    double delta      = 0.0;   // Detuning (rad/us); 0 = resonant
    double T1         = 0.0;   // Energy relaxation time (us); 0 = infinite (ideal)
    double T2         = 0.0;   // Dephasing time (us); 0 = infinite (ideal)
    double bz_eq      = -1.0;  // Equilibrium Bz (thermal state; -1 = ground state)
};

/// Output of the ODE solver: full trajectory
struct SolverResult {
    std::vector<double> times;
    std::vector<double> bx;
    std::vector<double> by;
    std::vector<double> bz;
    std::vector<double> excited_pop;  // P(|1>) = (1 + Bz) / 2
};

// ---------------------------------------------------------------------------
// Operator overloads for BlochState arithmetic (used in RK4)
// ---------------------------------------------------------------------------

BlochState operator+(const BlochState& a, const BlochState& b);
BlochState operator*(double scalar, const BlochState& s);

// ---------------------------------------------------------------------------
// Core solver functions
// ---------------------------------------------------------------------------

/// Compute Bloch equation derivatives at a given state
BlochState bloch_deriv(const BlochState& state, const BlochParams& params);

/// Advance state by one RK4 step
BlochState rk4_step(const BlochState& state, const BlochParams& params, double dt);

/// Solve full trajectory from t=0 to t=t_max
SolverResult solve_bloch(
    const BlochParams& params,
    const BlochState&  initial_state,
    double             t_max,
    int                n_points
);

} // namespace qubit_cal
