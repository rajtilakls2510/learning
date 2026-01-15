#ifndef IP_SOLVER
#define IP_SOLVER

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>
using namespace std;
using namespace std::chrono;

/**
 * Solve MPC Problem of keeping pendulum stable at s_ref = (x_ref, 0, 0, 0)
 *   min_s,F 0.5 * Sum_t (s_t - s_ref)^T * Q * (s_t - s_ref) + 0.5 * Sum_t R * F_t^2
 *       s.t. s_{t+1} = s_t + dt * d(s_t, F_t)
 *            -F_min <= F_t <= F_max
 */

class InversePendulumProblem {
public:
    InversePendulumProblem(
            Eigen::VectorXd& state_ref,
            Eigen::MatrixXd& Q,
            double R,
            const size_t n_horizon,
            const double f_extreme,
            const double mp,
            const double g,
            const double mc,
            const double l,
            const double dt)
        : state_ref(state_ref),
          Q(Q),
          R(R),
          n_horizon(n_horizon),
          f_extreme(f_extreme),
          mp(mp),
          gv(g),
          mc(mc),
          l(l),
          dt(dt) {}

    double cost(const Eigen::VectorXd& states, const Eigen::VectorXd& controls) {
        // states: (T+1 * 4, 1), controls: (T x 1)
        double f = 0;
        for (size_t t = 0; t < n_horizon; t++) {
            Eigen::Vector4d state_diff = states.segment<4>(4 * (t + 1)) - state_ref;

            // Eigen::VectorXd state_diff = states.row(t + 1).transpose() - state_ref;
            f += 0.5 * state_diff.transpose() * Q * state_diff;
            f += 0.5 * R * pow(controls(t), 2);
        }
        return f;
    }

    Eigen::VectorXd dcostdx(const Eigen::VectorXd& states, const Eigen::VectorXd& controls) {
        Eigen::VectorXd dcdx(5 * n_horizon);
        for (size_t t = 0; t < n_horizon; t++) {
            Eigen::Vector4d state_diff = states.segment<4>(4 * (t + 1)) - state_ref;
            dcdx.segment<4>(t * 4) = Q * state_diff;
        }
        dcdx.segment(4 * n_horizon, controls.size()) = R * controls;
        return dcdx;
    }

    Eigen::MatrixXd d2costdx2(const Eigen::VectorXd& states, const Eigen::VectorXd& controls) {
        Eigen::MatrixXd d2cdx2(5 * n_horizon, 5 * n_horizon);
        d2cdx2.setZero();
        for (size_t t = 0; t < n_horizon; t++) {
            d2cdx2.block<4, 4>(t * 4, t * 4) = Q;
        }
        for (size_t t = 0; t < n_horizon; t++) {
            d2cdx2(4 * n_horizon + t, 4 * n_horizon + t) = R;
        }
        return d2cdx2;
    }

    Eigen::VectorXd d(const Eigen::VectorXd& state, const double& control) {
        Eigen::VectorXd state_dot(4);

        double t_dot = state(3);

        state_dot(0) = state(2);
        state_dot(1) = t_dot;

        double sin_theta = sin(state(1));
        double cos_theta = cos(state(1));

        double theta_dot_dot =
                (gv * sin_theta -
                 cos_theta * (control + mp * l * pow(t_dot, 2) * sin_theta) / (mc + mp)) /
                (l * (4 / 3 - mp * pow(cos_theta, 2) / (mc + mp)));
        double x_dot_dot =
                (control + mp * l * (pow(t_dot, 2) * sin_theta - theta_dot_dot * cos_theta)) /
                (mc + mp);

        state_dot(2) = x_dot_dot;
        state_dot(3) = theta_dot_dot;

        return state_dot;
    }

    Eigen::VectorXd h(const Eigen::VectorXd& states, const Eigen::VectorXd& controls) {
        Eigen::VectorXd h_(4 * n_horizon);
        h_.setZero();
        for (size_t t = 0; t < n_horizon; t++) {
            const Eigen::VectorXd current_state = states.segment<4>(4 * t);
            const Eigen::VectorXd next_state = states.segment<4>(4 * (t + 1));
            h_.segment<4>(4 * t) = next_state - current_state - dt * d(current_state, controls(t));
        }
        return h_;
    }

    Eigen::MatrixXd ddds(const Eigen::VectorXd& state, const double& control) {
        Eigen::MatrixXd ddds_(4, 4);
        ddds_.setZero();
        ddds_(0, 2) = 1.0;
        ddds_(1, 3) = 1.0;

        double sin_theta = sin(state(1));
        double cos_theta = cos(state(1));
        double theta_dot = state(3);

        double m_ = (mc + mp);

        double deno = 4 * m_ - 3 * mp * pow(cos_theta, 2);

        double dtdotdot_dt = 3 *
                                     (gv * cos_theta * m_ + control * sin_theta -
                                      mp * l * pow(theta_dot * cos_theta, 2) +
                                      mp * l * pow(theta_dot * sin_theta, 2)) /
                                     (l * deno) -
                             18 *
                                     (gv * sin_theta * m_ - control * cos_theta -
                                      mp * l * pow(theta_dot, 2) * sin_theta * cos_theta) *
                                     (mp * sin_theta * cos_theta) / (l * pow(deno, 2));

        double dtdotdot_dtdot = -(6 * mp * theta_dot * sin_theta * cos_theta) / deno;

        double theta_dot_dot = 3 *
                               (gv * sin_theta * m_ - control * cos_theta -
                                mp * l * pow(theta_dot, 2) * sin_theta * cos_theta) /
                               (l * deno);

        double dxdotdot_dt = mp * l *
                             (pow(theta_dot, 2) * cos_theta + theta_dot_dot * sin_theta -
                              cos_theta * dtdotdot_dt) /
                             m_;
        double dxdotdot_dtdot =
                mp * l * (2 * theta_dot * sin_theta - cos_theta * dtdotdot_dtdot) / m_;

        ddds_(2, 1) = dxdotdot_dt;
        ddds_(2, 3) = dxdotdot_dtdot;
        ddds_(3, 1) = dtdotdot_dt;
        ddds_(3, 3) = dtdotdot_dtdot;
        return ddds_;
    }

    Eigen::VectorXd dddf(const Eigen::VectorXd& state, const double control) {
        Eigen::VectorXd dddf_(4);
        dddf_.setZero();

        double cos_theta = cos(state(1));
        double m_ = mc + mp;
        double dtdotdot_df = -3.0 * cos_theta / (l * (4 * m_ - 3 * mp * pow(cos_theta, 2)));
        double dxdotdot_df = (1 - mp * l * cos_theta * dtdotdot_df) / m_;

        dddf_(2) = dxdotdot_df;
        dddf_(3) = dtdotdot_df;
        return dddf_;
    }

    Eigen::MatrixXd dhds(const Eigen::VectorXd& state, const double& control) {
        Eigen::MatrixXd dhds_(4, 4);
        dhds_ = -Eigen::Matrix4d::Identity();
        dhds_ -= dt * ddds(state, control);
        return dhds_;
    }

    Eigen::VectorXd dhdf(const Eigen::VectorXd& state, const double& control) {
        Eigen::VectorXd dhdf_(4);
        dhdf_ = -dt * dddf(state, control);
        return dhdf_;
    }

    Eigen::MatrixXd dhdx(const Eigen::VectorXd& states, const Eigen::VectorXd& controls) {
        Eigen::MatrixXd dhdx_(4 * n_horizon, 5 * n_horizon);

        for (size_t t = 0; t < n_horizon; t++) {
            const Eigen::VectorXd current_state = states.segment<4>(4 * t);
            if (t > 0) {
                dhdx_.block<4, 4>(4 * t, 4 * (t - 1)) = dhds(current_state, controls(t));
            }
            dhdx_.block<4, 4>(4 * t, 4 * t) = Eigen::Matrix4d::Identity();
            dhdx_.block<4, 1>(4 * t, 4 * n_horizon + t) = dhdf(current_state, controls(t));
        }
        return dhdx_;
    }

    Eigen::VectorXd g(const Eigen::VectorXd& states, const Eigen::VectorXd& controls) {
        Eigen::VectorXd g_(2 * n_horizon);
        for (size_t t = 0; t < n_horizon; t++) {
            g_(t) = -f_extreme - controls(t);
            g_(n_horizon + t) = controls(t) - f_extreme;
        }
        return g_;
    }

    Eigen::MatrixXd dgdx(const Eigen::VectorXd& states, const Eigen::VectorXd& controls) {
        Eigen::MatrixXd dgdx_(2 * n_horizon, 5 * n_horizon);
        dgdx_.setZero();

        Eigen::MatrixXd i(n_horizon, n_horizon);
        i.setIdentity();

        dgdx_.block(0, n_horizon, n_horizon, n_horizon) = -i;
        dgdx_.block(n_horizon, n_horizon, n_horizon, n_horizon) = i;

        return dgdx_;
    }

    bool within_constraints(const Eigen::VectorXd& states, const Eigen::VectorXd& controls) {
        // Check Equality constraints
        Eigen::VectorXd hx = h(states, controls);
        for (int i = 0; i < 4 * n_horizon; i++) {
            if (abs(hx(i)) > 1e-6) return false;
        }

        // Check Inequality constraints
        Eigen::VectorXd gx = g(states, controls);
        for (int i = 0; i < 2 * n_horizon; i++) {
            if (gx(i) > 0) return false;
        }
        return true;
    }

    size_t state_dim{4}, n_horizon;
    Eigen::VectorXd state_ref;
    Eigen::MatrixXd Q;
    double R, f_extreme;
    double mp, gv, mc, l, dt;
};

class ALSolver {
public:
    ALSolver(InversePendulumProblem problem, int n_horizon)
        : problem(problem), n_horizon(n_horizon) {}

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

        Eigen::VectorXd dfdx_ = problem.dcostdx(states, controls);
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
        double f_ = problem.cost(states, controls);
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
            std::cout << "Err: " << abs(g.norm() - prev_g_norm) << "\n";
            prev_g_norm = g.norm();

            // std::cout << "g: " << g << "\n";

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
            Eigen::MatrixXd d2fdx2 = problem.d2costdx2(states, controls);
            Eigen::MatrixXd dhdx_ = problem.dhdx(states, controls);
            Eigen::MatrixXd H = d2fdx2 + rho * dhdx_.transpose() * dhdx_ + rho * C.transpose() * C;

            // Regularize Hessian until PD

            double beta = 1e-6;
            Eigen::MatrixXd I = Eigen::MatrixXd::Identity(H.rows(), H.cols());
            while (!is_pos_def(H)) {
                H += beta * I;
                beta *= 10.0;
                // std::cout << "beta: " << beta << "\n";
            }

            // std::cout << "H shape: " << H.rows() << " x " << H.cols() << std::endl;

            // std::cout << "g shape: " << g.rows() << " x " << g.cols() << std::endl;

            // std::cout << "states shape: " << states.rows() << " x " << states.cols() <<
            // std::endl;

            // Solve KKT system
            Eigen::VectorXd dx = -H.ldlt().solve(g);
            // std::cout << "dx: " << dx << "\n";
            Eigen::VectorXd dstates(4 * (n_horizon + 1));
            dstates.setZero();
            dstates.segment(4, 4 * n_horizon) = dx.segment(0, 4 * n_horizon);
            Eigen::VectorXd dcontrols = dx.segment(4 * n_horizon, n_horizon);

            // Backtracking line search using merit function
            double alpha = 1.0;
            while (al(states + alpha * dstates, controls + alpha * dcontrols, nu, lmda, rho) >
                   al(states, controls, nu, lmda, rho) +
                           0.01 * alpha * g.transpose() * dx) {  //&& alpha > 1e-8) {
                // std::cout
                //         << "al1: "
                //         << al(states + alpha * dstates, controls + alpha * dcontrols, nu, lmda,
                //         rho)
                //         << "\n";
                // std::cout << "al2: "
                //           << al(states, controls, nu, lmda, rho) * 0.01 * alpha * g.transpose() *
                //           dx
                //           << "\n";
                alpha *= 0.5;
                // std::cout << "alpha: " << alpha << "\n";
            }

            states = states + alpha * dstates;
            controls = controls + alpha * dcontrols;

            // std::cout << "states: " << states << "\ncontrols: " << controls << "\n";
            std::cout << "states: \n";
            for (size_t t = 0; t < n_horizon + 1; t++) {
                std::cout << states(4 * t) << "," << states(4 * t + 1) << "," << states(4 * t + 2)
                          << "," << states(4 * t + 3) << "\n";
            }
            std::cout << "Solved Controls: ";  // << solved_controls;
            for (size_t t = 0; t < n_horizon; t++) {
                std::cout << controls(t) << ",";
            }
            std::cout << "\n" << std::flush;
            g = dldx(states, controls, nu, lmda, rho);
        }

        final_states = states;
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
        return abs(problem.cost(prev_states, prev_controls) - problem.cost(states, controls)) <=
               1e-6;
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
        Eigen::VectorXd nu(4 * n_horizon);
        nu.setZero();
        Eigen::VectorXd lmda(2 * n_horizon);
        lmda.setZero();

        Eigen::VectorXd prev_states(4 * (n_horizon + 1)), prev_controls(n_horizon);
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
                rho = min(rho * 1.0, 1e2);
            }
        }

        final_states = states;
        final_controls = controls;
    }

    InversePendulumProblem problem;
    int n_horizon;
};

int main(int argc, char* argv[]) {
    int H = 50;
    double f_extreme = 5.0;
    Eigen::MatrixXd Q(4, 4);  // = 1 * Eigen::Matrix4d::Identity();
    Q << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    Q = 1.0 * Q;
    double R = 0.0;
    double mp = 1.0, mc = 1.0, l = 1.0, gv = 9.8;
    double del_t = 0.1;

    Eigen::VectorXd state_ref(4);
    state_ref << 0.1, 0.0, 0.0, 0.0;

    InversePendulumProblem problem(state_ref, Q, R, H, f_extreme, mp, gv, mc, l, del_t);

    Eigen::VectorXd current_states((H + 1) * 4);
    current_states.setZero();
    Eigen::VectorXd current_controls(H);
    current_controls.setZero();
    for (int t = 0; t < H; t++) {
        current_states.segment(4 * (t + 1), 4) +=
                del_t * problem.d(current_states.segment(4 * t, 4), current_controls(t));
    }
    // for (int t = 0; t < H + 1; ++t) {
    //     current_states(t * 4 + 0) = H + 1 - t;
    //     current_states(t * 4 + 1) = 0.0;
    //     current_states(t * 4 + 2) = 0.0;
    //     current_states(t * 4 + 3) = 0.0;
    // }
    double f = problem.cost(current_states, current_controls);
    std::cout << "Cost: " << f << "\n";

    Eigen::VectorXd dcdx = problem.dcostdx(current_states, current_controls);
    // std::cout << "dCostdx: " << dcdx << "\n";

    Eigen::MatrixXd d2cdx2 = problem.d2costdx2(current_states, current_controls);

    std::cout << "d2cdx2: \n";
    for (size_t t1 = 0; t1 < 5 * H; t1++) {
        for (size_t t2 = 0; t2 < 5 * H; t2++) std::cout << d2cdx2(t1, t2) << ",";
        std::cout << "\n";
    }

    Eigen::VectorXd g_ = problem.g(current_states, current_controls);
    std::cout << "g:\n" << g_ << "\n";

    Eigen::MatrixXd dgdx_ = problem.dgdx(current_states, current_controls);
    std::cout << "dgdx:\n";
    for (size_t t1 = 0; t1 < 2 * H; t1++) {
        for (size_t t2 = 0; t2 < 2 * H; t2++) std::cout << dgdx_(t1, t2) << ",";
        std::cout << "\n";
    }

    Eigen::VectorXd d_ = problem.d(current_states.segment<4>(0), 1.0);
    std::cout << "d: " << d_ << "\n";

    Eigen::VectorXd h_ = problem.h(current_states, current_controls);
    std::cout << "h: \n";
    for (size_t t = 0; t < H; t++) {
        std::cout << h_(4 * t) << "," << h_(4 * t + 1) << "," << h_(4 * t + 2) << ","
                  << h_(4 * t + 3) << "\n";
    }

    Eigen::MatrixXd ddds_ = problem.ddds(current_states.segment<4>(0), 1.0);
    std::cout << "ddds: \n" << ddds_ << "\n";

    Eigen::VectorXd dddf_ = problem.dddf(current_states.segment<4>(0), 1.0);
    std::cout << "dddf: \n" << dddf_ << "\n";

    Eigen::MatrixXd dhds_ = problem.dhds(current_states.segment<4>(0), 1.0);
    std::cout << "dhds: \n" << dhds_ << "\n";

    Eigen::VectorXd dhdf_ = problem.dhdf(current_states.segment<4>(0), 1.0);
    std::cout << "dhdf: \n" << dhdf_ << "\n";

    Eigen::MatrixXd dhdx_ = problem.dhdx(current_states, current_controls);
    std::cout << "dhdx: \n" << dhdx_ << "\n";

    // // Test Dynamics
    // Eigen::VectorXd state(4);
    // state << 0.0, 0.0, 0.0, 0.0;
    // std::cout << "initial state: " << state(0) << "," << state(1) << "," << state(2) << ","
    //           << state(3) << "\n";
    // for (size_t t = 0; t < H; t++) {
    //     double control = t % 2 == 0 ? 1 : -1;
    //     state = state + del_t * problem.d(state, control);
    //     std::cout << "t: " << t << " control: " << control << " state: " << state(0) << ","
    //               << state(1) << "," << state(2) << "," << state(3) << "\n";
    // }

    ALSolver solver(problem, H);

    Eigen::VectorXd nu(4 * H);
    nu.setZero();
    Eigen::VectorXd lmda(2 * H);
    lmda.setZero();

    Eigen::MatrixXd dldx_ = solver.dldx(current_states, current_controls, nu, lmda, 1.0);
    std::cout << "dldx: \n" << dldx_ << "\n";

    double al = solver.al(current_states, current_controls, nu, lmda, 1.0);
    std::cout << "al: " << al << "\n";

    Eigen::VectorXd solved_states(4 * (H + 1)), solved_controls(H);
    // solver.solve_al_subproblem(
    //         current_states, current_controls, nu, lmda, 1.0, solved_states, solved_controls);
    solver.solve(current_states, current_controls, 0.1, solved_states, solved_controls);
    std::cout << "Solved States: \n";  // << solved_states;
    for (size_t t = 0; t < H + 1; t++) {
        std::cout << solved_states(4 * t) << "," << solved_states(4 * t + 1) << ","
                  << solved_states(4 * t + 2) << "," << solved_states(4 * t + 3) << "\n";
    }
    std::cout << "Solved Controls: ";  // << solved_controls;
    for (size_t t = 0; t < H; t++) {
        std::cout << solved_controls(t) << ",";
    }
    std::cout << "\n" << std::flush;

    return 0;
}

#endif