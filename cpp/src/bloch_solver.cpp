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
// For step 1 (ideal Rabi): delta=0, T1=T2=infinity -> no decay terms.
// P(|1>) = (1 + Bz) / 2

#include "../include/bloch_solver.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

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
        result.excited_pop[i] = (1.0 + state.bz) / 2.0;

        if (i < n_points - 1) {
            state = rk4_step(state, params, dt);
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Ramsey experiment solver
// ---------------------------------------------------------------------------

double ramsey_single(
    double             tau,
    double             omega_rabi,
    double             delta,
    const BlochParams& decoherence_params,
    int                n_steps_pulse,
    int                n_steps_free
) {
    // --- Stage 1: pi/2 pulse ---
    // Resonant drive (delta=0) for t_pi2 = pi / (2 * omega_rabi)
    BlochParams pulse_params;
    pulse_params.omega_rabi = omega_rabi;
    pulse_params.delta      = 0.0;
    pulse_params.T1         = decoherence_params.T1;
    pulse_params.T2         = decoherence_params.T2;
    pulse_params.bz_eq      = decoherence_params.bz_eq;

    double t_pi2    = M_PI / (2.0 * omega_rabi);
    double dt_pulse = t_pi2 / n_steps_pulse;

    BlochState state;  // Ground state (0, 0, -1) by default
    for (int i = 0; i < n_steps_pulse; ++i) {
        state = rk4_step(state, pulse_params, dt_pulse);
    }

    // --- Stage 2: free precession for time tau ---
    // Drive off (omega_rabi=0), detuning and decoherence only
    if (tau > 0.0) {
        BlochParams free_params;
        free_params.omega_rabi = 0.0;
        free_params.delta      = delta;
        free_params.T1         = decoherence_params.T1;
        free_params.T2         = decoherence_params.T2;
        free_params.bz_eq      = decoherence_params.bz_eq;

        int    n_free  = std::max(n_steps_free, (int)(tau * n_steps_free));
        double dt_free = tau / n_free;

        for (int i = 0; i < n_free; ++i) {
            state = rk4_step(state, free_params, dt_free);
        }
    }

    // --- Stage 3: second pi/2 pulse ---
    for (int i = 0; i < n_steps_pulse; ++i) {
        state = rk4_step(state, pulse_params, dt_pulse);
    }

    // Measure P(|1>) = (1 + Bz) / 2
    return (1.0 + state.bz) / 2.0;
}

RamseyResult solve_ramsey(
    double             omega_rabi,
    double             delta,
    const BlochParams& decoherence_params,
    double             tau_max,
    int                n_tau,
    int                n_steps_pulse,
    int                n_steps_free
) {
    RamseyResult result;
    result.tau_times.resize(n_tau);
    result.excited_pop.resize(n_tau);

    double dtau = tau_max / (n_tau - 1);

    for (int i = 0; i < n_tau; ++i) {
        double tau = i * dtau;
        result.tau_times[i]   = tau;
        result.excited_pop[i] = ramsey_single(
            tau, omega_rabi, delta, decoherence_params,
            n_steps_pulse, n_steps_free
        );
    }

    return result;
}

} // namespace qubit_cal
