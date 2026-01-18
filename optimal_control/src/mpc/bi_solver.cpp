#ifndef BI_SOLVER
#define BI_SOLVER

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>
using namespace std;
using namespace std::chrono;

class BicycleProblem {
public:
    BicycleProblem(
            Eigen::VectorXd& state_ref,
            Eigen::MatrixXd& Q,
            Eigen::MatrixXd& R,
            Eigen::VectorXd& u_extreme,
            const size_t n_horizon,
            const double wheel_base,
            const double dt,
            const size_t state_dim,
            const size_t control_dim)
        : state_ref(state_ref),
          Q(Q),
          R(R),
          u_extreme(u_extreme),
          n_horizon(n_horizon),
          l(wheel_base),
          dt(dt),
          state_dim(state_dim),
          control_dim(control_dim) {}

    // Normalize an angle to be between [-pi, pi)
    float normalize_angle(float angle) {
        angle = fmod(angle + M_PI, 2.0 * M_PI);
        if (angle < 0) angle += 2.0 * M_PI;
        return angle - M_PI;
    }

    double f(const Eigen::VectorXd& states, const Eigen::VectorXd& controls) {
        double f_ = 0;
        for (size_t t = 0; t < n_horizon; t++) {
            Eigen::VectorXd state_diff = states.segment(state_dim * (t + 1), state_dim) - state_ref;
            state_diff(2) = normalize_angle(state_diff(2));
            Eigen::VectorXd controls_t = controls.segment(control_dim * t, control_dim);
            f_ += 0.5 * state_diff.transpose() * Q * state_diff;
            f_ += 0.5 * controls_t.transpose() * R * controls_t;
        }
        return f_;
    }

    Eigen::VectorXd dfdx(const Eigen::VectorXd& states, const Eigen::VectorXd& controls) {
        Eigen::VectorXd dfdx_((state_dim + control_dim) * n_horizon);
        for (size_t t = 0; t < n_horizon; t++) {
            Eigen::VectorXd state_diff = states.segment(state_dim * (t + 1), state_dim) - state_ref;
            state_diff(2) = normalize_angle(state_diff(2));
            dfdx_.segment(state_dim * t, state_dim) = Q * state_diff;
        }
        size_t df_offset = state_dim * n_horizon;
        for (size_t t = 0; t < n_horizon; t++) {
            Eigen::VectorXd control_t = controls.segment(control_dim * t, control_dim);
            dfdx_.segment(control_dim * t + df_offset, control_dim) = R * control_t;
        }
        return dfdx_;
    }

    Eigen::MatrixXd d2fdx2(const Eigen::VectorXd& states, const Eigen::VectorXd& controls) {
        size_t n_rows = (state_dim + control_dim) * n_horizon;
        size_t n_cols = (state_dim + control_dim) * n_horizon;
        Eigen::MatrixXd d2fdx2_(n_rows, n_cols);
        d2fdx2_.setZero();
        size_t df2_offset = state_dim * n_horizon;
        for (size_t t = 0; t < n_horizon; t++) {
            d2fdx2_.block(state_dim * t, state_dim * t, state_dim, state_dim) = Q;
            d2fdx2_.block(
                    control_dim * t + df2_offset,
                    control_dim * t + df2_offset,
                    control_dim,
                    control_dim) = R;
        }
        return d2fdx2_;
    }

    Eigen::VectorXd d(const Eigen::VectorXd& state, const Eigen::VectorXd& control) {
        Eigen::VectorXd state_dot(state_dim);
        double theta = state(2);
        double v = state(3);

        state_dot(0) = v * cos(theta);
        state_dot(1) = v * sin(theta);
        state_dot(2) = v / l * tan(state(4));
        state_dot(3) = control(0);
        state_dot(4) = control(1);

        return state_dot;
    }

    Eigen::VectorXd h(const Eigen::VectorXd& states, const Eigen::VectorXd& controls) {
        Eigen::VectorXd h_(state_dim * n_horizon);
        h_.setZero();
        for (size_t t = 0; t < n_horizon; t++) {
            const Eigen::VectorXd current_state = states.segment(state_dim * t, state_dim);
            const Eigen::VectorXd next_state = states.segment(state_dim * (t + 1), state_dim);
            h_.segment(state_dim * t, state_dim) =
                    next_state - current_state -
                    dt * d(current_state, controls.segment(control_dim * t, control_dim));
            h_(state_dim * t + 2) = normalize_angle(h_(state_dim * t + 2));
        }
        return h_;
    }

    Eigen::MatrixXd ddds(const Eigen::VectorXd& state, const Eigen::VectorXd& control) {
        Eigen::MatrixXd ddds_(state_dim, state_dim);
        ddds_.setZero();
        double theta = state(2);
        double v = state(3);
        double phi = state(4);

        ddds_(0, 2) = -v * sin(theta);
        ddds_(0, 3) = cos(theta);
        ddds_(1, 2) = v * cos(theta);
        ddds_(1, 3) = sin(theta);
        ddds_(2, 3) = tan(phi) / l;
        ddds_(2, 4) = v * pow(1.0 / cos(phi), 2) / l;

        return ddds_;
    }

    Eigen::MatrixXd dddu(const Eigen::VectorXd& state, const Eigen::VectorXd& control) {
        Eigen::MatrixXd dddu_(state_dim, control_dim);
        dddu_.setZero();

        dddu_(3, 0) = 1.0;
        dddu_(4, 1) = 1.0;

        return dddu_;
    }

    Eigen::MatrixXd dhds(const Eigen::VectorXd& state, const Eigen::VectorXd& control) {
        Eigen::MatrixXd dhds_(state_dim, state_dim);
        dhds_ = -Eigen::MatrixXd::Identity(state_dim, state_dim);
        dhds_ -= dt * ddds(state, control);
        return dhds_;
    }

    Eigen::MatrixXd dhdu(const Eigen::VectorXd& state, const Eigen::VectorXd& control) {
        Eigen::MatrixXd dhdu_(state_dim, control_dim);
        dhdu_ = -dt * dddu(state, control);
        return dhdu_;
    }

    Eigen::MatrixXd dhdx(const Eigen::VectorXd& states, const Eigen::VectorXd& controls) {
        Eigen::MatrixXd dhdx_(state_dim * n_horizon, (state_dim + control_dim) * n_horizon);
        dhdx_.setZero();

        for (size_t t = 0; t < n_horizon; t++) {
            const Eigen::VectorXd current_state = states.segment(state_dim * t, state_dim);
            const Eigen::VectorXd current_control = controls.segment(control_dim * t, control_dim);
            if (t > 0)
                dhdx_.block(state_dim * t, state_dim * (t - 1), state_dim, state_dim) =
                        dhds(current_state, current_control);
            dhdx_.block(state_dim * t, state_dim * t, state_dim, state_dim) =
                    Eigen::MatrixXd::Identity(state_dim, state_dim);
            dhdx_.block(
                    state_dim * t,
                    state_dim * n_horizon + control_dim * t,
                    state_dim,
                    control_dim) = dhdu(current_state, current_control);
        }
        return dhdx_;
    }

    Eigen::VectorXd g(const Eigen::VectorXd& states, const Eigen::VectorXd& controls) {
        Eigen::VectorXd g_(control_dim * 2 * n_horizon);
        for (size_t t = 0; t < n_horizon; t++) {
            g_.segment(control_dim * t, control_dim) =
                    -u_extreme - controls.segment(control_dim * t, control_dim);
            g_.segment(control_dim * n_horizon + control_dim * t, control_dim) =
                    controls.segment(control_dim * t, control_dim) - u_extreme;
        }
        return g_;
    }

    Eigen::MatrixXd dgdx(const Eigen::VectorXd& states, const Eigen::VectorXd& controls) {
        Eigen::MatrixXd dgdx_(2 * control_dim * n_horizon, (state_dim + control_dim) * n_horizon);
        dgdx_.setZero();

        Eigen::MatrixXd i(control_dim * n_horizon, control_dim * n_horizon);
        i.setIdentity();

        dgdx_.block(0, state_dim * n_horizon, control_dim * n_horizon, control_dim * n_horizon) =
                -i;
        dgdx_.block(
                control_dim * n_horizon,
                state_dim * n_horizon,
                control_dim * n_horizon,
                control_dim * n_horizon) = i;

        return dgdx_;
    }

    bool within_constraints(const Eigen::VectorXd& states, const Eigen::VectorXd& controls) {
        // Check Equality constraints
        Eigen::VectorXd hx = h(states, controls);
        for (size_t t = 0; t < state_dim * n_horizon; t++) {
            if (abs(hx(t)) > 1e-6) return false;
        }

        // Check Inequality constraints
        Eigen::VectorXd gx = g(states, controls);
        for (size_t t = 0; t < 2 * control_dim * n_horizon; t++) {
            if (gx(t) > 0) return false;
        }
        return true;
    }

    size_t n_horizon, state_dim, control_dim;
    Eigen::VectorXd state_ref, u_extreme;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    double l, dt;
};

class ALSolver {
public:
    ALSolver(BicycleProblem problem, size_t n_horizon) : problem(problem), n_horizon(n_horizon) {}

    bool is_pos_def(const Eigen::MatrixXd& H) {
        Eigen::LLT<Eigen::MatrixXd> llt(H);
        return llt.info() == Eigen::Success;
    }

    Eigen::VectorXd dldx(
            const Eigen::VectorXd& states,
            const Eigen::VectorXd& controls,
            const Eigen::VectorXd& nu,
            const Eigen::VectorXd& lmda,
            const double rho) {
        Eigen::MatrixXd dhdx_ = problem.dhdx(states, controls);
        const auto gx = problem.g(states, controls);
        Eigen::VectorXd C = (lmda + rho * gx).cwiseMax(0.0);

        Eigen::VectorXd dfdx_ = problem.dfdx(states, controls);
        // std::cout << "dfdx: " << dfdx_ << "\n";
        Eigen::VectorXd h_ = problem.h(states, controls);
        Eigen::MatrixXd dgdx_ = problem.dgdx(states, controls);

        return dfdx_ + dhdx_.transpose() * (nu + rho * h_) + dgdx_.transpose() * C;
    }

    double al(
            const Eigen::VectorXd& states,
            const Eigen::VectorXd& controls,
            const Eigen::VectorXd& nu,
            const Eigen::VectorXd& lmda,
            const double rho) {
        Eigen::VectorXd C =
                (lmda + rho * problem.g(states, controls)).cwiseMax(0.0).array().square();
        double f_ = problem.f(states, controls);
        // std::cout << "f: " << f_ << "\n";
        Eigen::VectorXd h_ = problem.h(states, controls);
        // std::cout << "h_: " << h_ << "\n";
        double al_ = f_ + nu.transpose() * h_ + 0.5 * rho * h_.norm() +
                     0.5 / rho * (C.sum() - lmda.squaredNorm());
        // std::cout << "al_: " << al_ << "\n";
        return al_;
    }

    void solve_al_subproblem(
            const Eigen::VectorXd& initial_states,
            const Eigen::VectorXd& initial_controls,
            const Eigen::VectorXd& nu,
            const Eigen::VectorXd& lmda,
            const double rho,
            Eigen::VectorXd& final_states,
            Eigen::VectorXd& final_controls) {
        Eigen::VectorXd states = initial_states;
        Eigen::VectorXd controls = initial_controls;

        Eigen::VectorXd g = dldx(states, controls, nu, lmda, rho);
        double prev_g_norm = INFINITY;

        // std::cout << "states: " << states << "\ncontrols: " << controls << "\n";

        while (abs(g.norm() - prev_g_norm) > 1e-8) {
            // while (g.norm() > 1e-3) {
            std::cout << "Err: " << abs(g.norm() - prev_g_norm) << "\n";
            std::cout << "g_norm: " << g.norm() << "\n";
            prev_g_norm = g.norm();

            std::cout << "g: " << g << "\n";

            // Find Active inequality constraints
            Eigen::MatrixXd dgdx_ = problem.dgdx(states, controls);
            // std::cout << "dgdx_: " << dgdx_ << "\n";
            Eigen::VectorXd g_ = problem.g(states, controls);
            // std::cout << "g_: " << g_ << "\n";

            Eigen::VectorXd m = ((lmda + rho * g_).array() > 0.0).cast<double>();
            // std::cout << "m: " << m << "\n";
            Eigen::MatrixXd C = m.asDiagonal() * dgdx_;
            // std::cout << "C: " << C << "\n";

            // Compute Hessian
            Eigen::MatrixXd d2fdx2 = problem.d2fdx2(states, controls);
            Eigen::MatrixXd dhdx_ = problem.dhdx(states, controls);
            Eigen::MatrixXd H = d2fdx2 + rho * dhdx_.transpose() * dhdx_ + rho * C.transpose() * C;

            // Regularize Hessian until PD

            double beta = 1e-6;
            Eigen::MatrixXd I = Eigen::MatrixXd::Identity(H.rows(), H.cols());
            while (!is_pos_def(H)) {
                H += beta * I;
                beta *= 10.0;
                std::cout << "beta: " << beta << "\n";
            }

            // std::cout << "H shape: " << H.rows() << " x " << H.cols() << std::endl;

            // std::cout << "g shape: " << g.rows() << " x " << g.cols() << std::endl;

            // std::cout << "states shape: " << states.rows() << " x " << states.cols() <<
            // std::endl;

            // Solve KKT system
            Eigen::VectorXd dx = -H.ldlt().solve(g);
            std::cout << "dx: " << dx << "\n";
            Eigen::VectorXd dstates(problem.state_dim * (n_horizon + 1));
            dstates.setZero();
            dstates.segment(problem.state_dim, problem.state_dim * n_horizon) =
                    dx.segment(0, problem.state_dim * n_horizon);
            Eigen::VectorXd dcontrols =
                    dx.segment(problem.state_dim * n_horizon, problem.control_dim * n_horizon);

            // Backtracking line search using merit function
            double alpha = 1.0;
            while (al(states + alpha * dstates, controls + alpha * dcontrols, nu, lmda, rho) >
                   al(states, controls, nu, lmda, rho) +
                           0.0001 * alpha * g.transpose() * dx) {  // && alpha > 1e-8) {
                std::cout
                        << "al1: "
                        << al(states + alpha * dstates, controls + alpha * dcontrols, nu, lmda, rho)
                        << "\n";
                std::cout << "al2: "
                          << al(states, controls, nu, lmda, rho) +
                                     0.0001 * alpha * g.transpose() * dx
                          << "\n";
                alpha *= 0.5;
                std::cout << "alpha: " << alpha << "\n";
            }

            states = states + alpha * dstates;
            controls = controls + alpha * dcontrols;

            // std::cout << "states: " << states << "\ncontrols: " << controls << "\n";
            std::cout << "states: \n";
            for (size_t t = 0; t < n_horizon + 1; t++) {
                for (size_t k = 0; k < problem.state_dim; k++)
                    std::cout << states(problem.state_dim * t + k) << ",";
                std::cout << "\n";
            }
            std::cout << "controls: ";  // << solved_controls;
            for (size_t t = 0; t < n_horizon; t++) {
                for (size_t k = 0; k < problem.control_dim; k++)
                    std::cout << controls(problem.control_dim * t + k) << ",";
                std::cout << "\n";
            }
            std::cout << "\n" << std::flush;
            g = dldx(states, controls, nu, lmda, rho);
        }

        final_states = states;
        // final_states = Eigen::VectorXd((n_horizon + 1) * problem.state_dim);
        // final_states.setZero();
        // final_states.segment(0, problem.state_dim) = states.segment(0, problem.state_dim);
        // for (size_t t = 1; t < n_horizon + 1; t++) {
        //     final_states.segment(problem.state_dim * t, problem.state_dim) +=
        //             problem.dt *
        //             problem.d(
        //                     final_states.segment(problem.state_dim * (t - 1), problem.state_dim),
        //                     controls.segment(problem.control_dim * (t - 1), problem.control_dim));
        //     // for (size_t k = 0; k < problem.state_dim; k++)
        //     //     std::cout << final_states_(problem.state_dim * t + k) << ",";
        //     // std::cout << "\n";
        // }
        final_controls = controls;
    }

    void al_step(
            Eigen::VectorXd& states,
            Eigen::VectorXd& controls,
            Eigen::VectorXd& nu,
            Eigen::VectorXd& lmda,
            const double rho) {
        Eigen::VectorXd states_, controls_;
        solve_al_subproblem(states, controls, nu, lmda, rho, states_, controls_);
        states = states_;
        controls = controls_;

        nu = nu + rho * problem.h(states, controls);

        lmda = (lmda + rho * problem.g(states, controls)).cwiseMax(0.0);
    }

    bool is_converged(
            const Eigen::VectorXd& states,
            const Eigen::VectorXd& controls,
            const Eigen::VectorXd& prev_states,
            const Eigen::VectorXd& prev_controls) {
        double f_diff = abs(problem.f(prev_states, prev_controls) - problem.f(states, controls));
        std::cout << "f_diff: " << f_diff << "\n";
        return f_diff <= 1.0;
    }

    void solve(
            const Eigen::VectorXd& initial_states,
            const Eigen::VectorXd& initial_controls,
            double rho,
            Eigen::VectorXd& final_states,
            Eigen::VectorXd& final_controls) {
        Eigen::VectorXd states = initial_states;
        Eigen::VectorXd controls = initial_controls;

        // TODO: Change to lagrange variables warm start
        Eigen::VectorXd nu(problem.state_dim * n_horizon);
        nu.setZero();
        Eigen::VectorXd lmda(2 * problem.control_dim * n_horizon);
        lmda.setZero();

        Eigen::VectorXd prev_states(problem.state_dim * (n_horizon + 1)),
                prev_controls(problem.control_dim * n_horizon);
        prev_states.setConstant(INFINITY);
        prev_controls.setConstant(INFINITY);

        int i = 0;
        while (!is_converged(states, controls, prev_states, prev_controls) ||
               !problem.within_constraints(states, controls)) {
            prev_states = states;
            prev_controls = controls;
            al_step(states, controls, nu, lmda, rho);
            std::cout << "Iters: " << (i++) << "\n";

            if (is_converged(states, controls, prev_states, prev_controls) &&
                !problem.within_constraints(states, controls)) {
                rho = min(rho * 10.0, 1e20);
                std::cout << "Increased rho: " << rho << "\n";
            }

            std::cout << "Final States: \n";
            Eigen::VectorXd final_states_((n_horizon + 1) * problem.state_dim);
            final_states_.setZero();
            for (size_t t = 1; t < n_horizon + 1; t++) {
                final_states_.segment(problem.state_dim * t, problem.state_dim) +=
                        problem.dt *
                        problem.d(
                                final_states_.segment(
                                        problem.state_dim * (t - 1), problem.state_dim),
                                controls.segment(
                                        problem.control_dim * (t - 1), problem.control_dim));
                for (size_t k = 0; k < problem.state_dim; k++)
                    std::cout << final_states_(problem.state_dim * t + k) << ",";
                std::cout << "\n";
            }
        }

        final_states = states;
        final_controls = controls;
    }

    size_t n_horizon;
    BicycleProblem problem;
};

int main(int argc, char* argv[]) {
    int H = 25;
    double wheel_base = 5;

    size_t state_dim = 5;
    size_t control_dim = 2;
    Eigen::MatrixXd Q(state_dim, state_dim);
    // Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Q << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    Eigen::MatrixXd R(control_dim, control_dim);
    R = Eigen::MatrixXd::Identity(control_dim, control_dim);

    double del_t = 0.1;

    Eigen::VectorXd state_ref(state_dim);
    state_ref << 0.0, 0.0, 1.0, 1.0, 0.0;
    Eigen::VectorXd u_extreme(2);
    u_extreme << 1.0, 1.0;

    BicycleProblem problem(
            state_ref, Q, R, u_extreme, H, wheel_base, del_t, state_dim, control_dim);

    Eigen::VectorXd current_states((H + 1) * state_dim);
    current_states.setZero();
    Eigen::VectorXd current_controls(H * control_dim);
    current_controls.setOnes();
    for (int t = 0; t < H; t++) {
        current_states.segment(state_dim * (t + 1), state_dim) =
                current_states.segment(state_dim * t, state_dim) +
                del_t * problem.d(
                                current_states.segment(state_dim * t, state_dim),
                                current_controls.segment(control_dim * t, control_dim));
    }
    std::cout << "Current states: " << current_states << "\n";
    std::cout << "Current controls: " << current_controls << "\n";

    double f = problem.f(current_states, current_controls);
    std::cout << "f_: " << f << "\n";

    Eigen::VectorXd dfdx = problem.dfdx(current_states, current_controls);
    std::cout << "dfdx: " << dfdx << "\n";

    Eigen::MatrixXd d2fdx2 = problem.d2fdx2(current_states, current_controls);
    std::cout << "d2fdx2: " << d2fdx2 << "\n";

    Eigen::VectorXd h = problem.h(current_states, current_controls);
    std::cout << "h: " << h << "\n";

    Eigen::MatrixXd dhdx = problem.dhdx(current_states, current_controls);
    std::cout << "dhdx: " << dhdx << "\n";

    Eigen::VectorXd g = problem.g(current_states, current_controls);
    std::cout << "g: " << g << "\n";

    Eigen::MatrixXd dgdx = problem.dgdx(current_states, current_controls);
    std::cout << "dgdx: " << dgdx << "\n";

    bool within_con = problem.within_constraints(current_states, current_controls);
    std::cout << "within constraints: " << std::boolalpha << within_con << "\n";

    ALSolver solver(problem, H);

    Eigen::VectorXd nu(problem.state_dim * H);
    nu.setZero();
    Eigen::VectorXd lmda(2 * problem.control_dim * H);
    lmda.setZero();

    Eigen::MatrixXd dldx_ = solver.dldx(current_states, current_controls, nu, lmda, 1.0);
    std::cout << "dldx: \n" << dldx_ << "\n";

    double al = solver.al(current_states, current_controls, nu, lmda, 1.0);
    std::cout << "al: " << al << "\n";

    Eigen::VectorXd solved_states(problem.state_dim * (H + 1)),
            solved_controls(problem.control_dim * H);
    // solver.solve_al_subproblem(
    //         current_states, current_controls, nu, lmda, 1.0, solved_states, solved_controls);
    solver.solve(current_states, current_controls, 1.0, solved_states, solved_controls);
    std::cout << "Solved States: \n";  // << solved_states;
    for (size_t t = 0; t < H + 1; t++) {
        for (size_t k = 0; k < problem.state_dim; k++)
            std::cout << solved_states(problem.state_dim * t + k) << ",";
        std::cout << "\n";
    }
    std::cout << "Solved Controls: ";  // << solved_controls;
    for (size_t t = 0; t < H; t++) {
        for (size_t k = 0; k < problem.control_dim; k++)
            std::cout << solved_controls(problem.control_dim * t + k) << ",";
        std::cout << "\n";
    }
    std::cout << "\n" << std::flush;

    std::cout << "Final States: \n";
    Eigen::VectorXd final_states((H + 1) * problem.state_dim);
    final_states.setZero();
    for (size_t t = 1; t < H + 1; t++) {
        final_states.segment(problem.state_dim * t, problem.state_dim) +=
                del_t *
                problem.d(
                        final_states.segment(problem.state_dim * (t - 1), problem.state_dim),
                        current_controls.segment(
                                problem.control_dim * (t - 1), problem.control_dim));
        for (size_t k = 0; k < problem.state_dim; k++)
            std::cout << final_states(problem.state_dim * t + k) << ",";
        std::cout << "\n";
    }

    return 0;
}

#endif