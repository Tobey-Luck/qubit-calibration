// cpp/src/bloch_solver.cpp
// ------------------------
// Runge-Kutta 4 solver for the optical Bloch equations.
//
// The Bloch equations describe qubit dynamics on the Bloch sphere:
//
//   dBx/dt =  delta * By - Bx / T2
//   dBy/dt = -delta * Bx + omega_rabi * Bz - By / T2
//   dBz/dt = -omega_rabi * By - (Bz - Bz_eq) / T1
//
// where:
//   Bx, By, Bz  = Bloch vector components
//   delta       = detuning (0 for resonant drive)
//   omega_rabi  = Rabi frequency (rad/us)
//   T1          = energy relaxation time (us)
//   T2          = dephasing time (us)
//   Bz_eq       = equilibrium Bz (-1 for ground state)
//
// For step 1 (ideal Rabi): delta=0, T1=T2=infinity -> no decay terms.
// P(|1>) = (1 + Bz) / 2

#include "../include/bloch_solver.h"
#include <cmath>
#include <stdexcept>

namespace qubit_cal {

// ---------------------------------------------------------------------------
// BlochState helpers
// ---------------------------------------------------------------------------

BlochState operator+(const BlochState& a, const BlochState& b) {
    return {a.bx + b.bx, a.by + b.by, a.bz + b.bz};
}

BlochState operator*(double scalar, const BlochState& s) {
    return {scalar * s.bx, scalar * s.by, scalar * s.bz};
}

// ---------------------------------------------------------------------------
// Bloch equation derivative
// ---------------------------------------------------------------------------

BlochState bloch_deriv(const BlochState& state, const BlochParams& p) {
    double bx = state.bx;
    double by = state.by;
    double bz = state.bz;

    // Avoid division by zero for infinite T1/T2 (ideal case)
    double decay_x  = (p.T2 > 0) ? bx / p.T2 : 0.0;
    double decay_y  = (p.T2 > 0) ? by / p.T2 : 0.0;
    double decay_z  = (p.T1 > 0) ? (bz - p.bz_eq) / p.T1 : 0.0;

    double dbx = p.delta * by - decay_x;
    double dby = -p.delta * bx + p.omega_rabi * bz - decay_y;
    double dbz = -p.omega_rabi * by - decay_z;

    return {dbx, dby, dbz};
}

// ---------------------------------------------------------------------------
// Single RK4 step
// ---------------------------------------------------------------------------

BlochState rk4_step(const BlochState& state, const BlochParams& p, double dt) {
    BlochState k1 = bloch_deriv(state,                  p);
    BlochState k2 = bloch_deriv(state + (dt/2.0) * k1, p);
    BlochState k3 = bloch_deriv(state + (dt/2.0) * k2, p);
    BlochState k4 = bloch_deriv(state + dt       * k3, p);

    return state + (dt / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4);
}

// ---------------------------------------------------------------------------
// Full trajectory solver
// ---------------------------------------------------------------------------

SolverResult solve_bloch(
    const BlochParams& params,
    const BlochState&  initial_state,
    double             t_max,
    int                n_points
) {
    if (n_points < 2) {
        throw std::invalid_argument("n_points must be >= 2");
    }
    if (t_max <= 0.0) {
        throw std::invalid_argument("t_max must be positive");
    }

    SolverResult result;
    result.times.resize(n_points);
    result.bx.resize(n_points);
    result.by.resize(n_points);
    result.bz.resize(n_points);
    result.excited_pop.resize(n_points);

    double dt = t_max / (n_points - 1);
    BlochState state = initial_state;

    for (int i = 0; i < n_points; ++i) {
        double t = i * dt;
        result.times[i]       = t;
        result.bx[i]          = state.bx;
        result.by[i]          = state.by;
        result.bz[i]          = state.bz;
        result.excited_pop[i] = (1.0 + state.bz) / 2.0;  // P(|1>) = (1 + Bz) / 2

        if (i < n_points - 1) {
            state = rk4_step(state, params, dt);
        }
    }

    return result;
}

} // namespace qubit_cal
