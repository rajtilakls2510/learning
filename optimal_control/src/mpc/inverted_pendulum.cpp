#ifndef INVERTED_PEN
#define INVERTED_PEN

#include <Eigen/Dense>


class InvertedPendulum {
 public:
  /**
   * Constructor with parameters and initial conditions
   * @param M Mass of the base
   * @param m Mass of the pendulum
   * @param J Moment of inertia of the pendulum
   * @param l Distance from the base to the pendulum
   * @param c Coefficient of viscous friction (base)
   * @param gamma Coefficient of viscous friction (pendulum)
   * @param x_0 Initial conditions
   */
  InvertedPendulum(double M, double m, double J, double l, double c,
                   double gamma, Eigen::VectorXd x_0) : M_(M),
        m_(m),
        J_(J),
        l_(l),
        c_(c),
        gamma_(gamma),
        g_(9.81),
        M_t_(M + m),
        J_t_(J + m * std::pow(l, 2)),
        x_(x_0),
        x_dot_(Eigen::VectorXd(4)),
        previous_time_(0) {
    x_dot_ << 0, 0, 0, 0;
    }

  /**
   * Constructor with default parameters and initial conditions
   * @param x_0 Initial conditions
   */
  InvertedPendulum(Eigen::VectorXd x_0) : InvertedPendulum(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, x_0) {}

  /**
   * Constructor with default parameters and default initial conditions
   */
  InvertedPendulum() : InvertedPendulum(Eigen::VectorXd(4)) {
    x_ << 0, 0, 0, 0;
    }


  /**
   * Updates the state by using the inverted pendulum equations
   * @param time Current simulation time
   * @param u Force applied to the base of the system
   */
  void Update(double time, double u)  {
    // Recover state parameters
    double x     = x_(0); // position of the base
    double theta = x_(1); // angle of the pendulum
    double vx    = x_(2); // velocity of the base
    double omega = x_(3); // angular rate of the pendulum

    // Compute common terms
    double s_t = std::sin(theta);
    double c_t = std::cos(theta);
    double o_2 = std::pow(omega, 2);
    double l_2 = std::pow(l_, 2);

    // Calculate derivatives
    x_dot_(0) = vx;
    x_dot_(1) = omega;
    x_dot_(2) = (-m_ * l_ * s_t * o_2 + m_ * g_ * (m_ * l_2 / J_t_) * s_t * c_t -
                c_ * vx - (gamma_ / J_t_) * m_ * l_ * c_t * omega + u) /
                (M_t_ - m_ * (m_ * l_2 / J_t_) * c_t * c_t);
    x_dot_(3) =
        (-m_ * l_2 * s_t * c_t * o_2 + M_t_ * g_ * l_ * s_t - c_ * l_ * c_t * vx -
        gamma_ * (M_t_ / m_) * omega + l_ * c_t * u) /
        (J_t_ * (M_t_ / m_) - m_ * (l_ * c_t) * (l_ * c_t));
    
    // Apply Euler method to solve differential equations
    double dt = time - previous_time_;
    previous_time_ = time;
    x_ += x_dot_ * dt;
    }

  /**
   * Returns the value of the state vector
   */
  Eigen::VectorXd GetState() const { return x_; }

  /**
   * Linearizes the system around the equilibrium point (theta -> 0)
   */
  void Linearize() {
    const double mu = M_t_ * J_t_ - std::pow((m_ * l_), 2);

    A_ = Eigen::MatrixXd::Zero(4, 4);
    A_(0, 2) = 1;
    A_(1, 3) = 1;
    A_(2, 1) = std::pow((m_ * l_), 2) * g_ / mu;
    A_(2, 2) = -c_ * J_t_ / mu;
    A_(2, 3) = -gamma_ * l_ * m_ / mu;
    A_(3, 1) = M_t_ * m_ * g_ * l_ / mu;
    A_(3, 2) = -c_ * l_ * m_ / mu;
    A_(3, 3) = -gamma_ * M_t_ / mu;

    B_ = Eigen::MatrixXd::Zero(4, 1);
    B_(2, 0) = J_t_ / mu;
    B_(3, 0) = l_ * m_ / mu;

    C_ = Eigen::MatrixXd::Identity(4, 4);
    D_ = Eigen::MatrixXd::Zero(4, 1);
    }

  const double M_;      // mass of the base
  const double m_;      // mass of the pendulum
  const double J_;      // moment of inertia of the pendulum
  const double l_;      // distance from the base to the pendulum
  const double c_;      // coefficient of viscous friction (base)
  const double gamma_;  // coefficient of viscous friction (pendulum)
  const double g_;      // acceleration due to gravity
  const double M_t_;    // total mass
  const double J_t_;    // total inertia

  Eigen::MatrixXd A_;  // dynamics matrix
  Eigen::MatrixXd B_;  // control matrix
  Eigen::MatrixXd C_;  // sensor matrix
  Eigen::MatrixXd D_;  // direct term

 private:
  Eigen::VectorXd x_;      // state vector
  Eigen::VectorXd x_dot_;  // state vector derivative
  double previous_time_;
};



// InvertedPendulum::InvertedPendulum(double M, double m, double J, double l,
//                                    double c, double gamma, Eigen::VectorXd x_0)
//     : M_(M),
//       m_(m),
//       J_(J),
//       l_(l),
//       c_(c),
//       gamma_(gamma),
//       g_(9.81),
//       M_t_(M + m),
//       J_t_(J + m * std::pow(l, 2)),
//       x_(x_0),
//       x_dot_(Eigen::VectorXd(4)),
//       previous_time_(0) {
//   x_dot_ << 0, 0, 0, 0;
// }

// InvertedPendulum::InvertedPendulum(Eigen::VectorXd x_0)
//     : InvertedPendulum(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, x_0) {}

// InvertedPendulum::InvertedPendulum()
//     : InvertedPendulum(Eigen::VectorXd(4)) {
//   x_ << 0, 0, 0, 0;
// }

// void InvertedPendulum::Update(double time, double u) {
//   // Recover state parameters
//   double x     = x_(0); // position of the base
//   double theta = x_(1); // angle of the pendulum
//   double vx    = x_(2); // velocity of the base
//   double omega = x_(3); // angular rate of the pendulum

//   // Compute common terms
//   double s_t = std::sin(theta);
//   double c_t = std::cos(theta);
//   double o_2 = std::pow(omega, 2);
//   double l_2 = std::pow(l_, 2);

//   // Calculate derivatives
//   x_dot_(0) = vx;
//   x_dot_(1) = omega;
//   x_dot_(2) = (-m_ * l_ * s_t * o_2 + m_ * g_ * (m_ * l_2 / J_t_) * s_t * c_t -
//                c_ * vx - (gamma_ / J_t_) * m_ * l_ * c_t * omega + u) /
//               (M_t_ - m_ * (m_ * l_2 / J_t_) * c_t * c_t);
//   x_dot_(3) =
//       (-m_ * l_2 * s_t * c_t * o_2 + M_t_ * g_ * l_ * s_t - c_ * l_ * c_t * vx -
//        gamma_ * (M_t_ / m_) * omega + l_ * c_t * u) /
//       (J_t_ * (M_t_ / m_) - m_ * (l_ * c_t) * (l_ * c_t));
  
//   // Apply Euler method to solve differential equations
//   double dt = time - previous_time_;
//   previous_time_ = time;
//   x_ += x_dot_ * dt;
// }

// Eigen::VectorXd InvertedPendulum::GetState() const { return x_; }

// void InvertedPendulum::Linearize() {
//   const double mu = M_t_ * J_t_ - std::pow((m_ * l_), 2);

//   A_ = Eigen::MatrixXd::Zero(4, 4);
//   A_(0, 2) = 1;
//   A_(1, 3) = 1;
//   A_(2, 1) = std::pow((m_ * l_), 2) * g_ / mu;
//   A_(2, 2) = -c_ * J_t_ / mu;
//   A_(2, 3) = -gamma_ * l_ * m_ / mu;
//   A_(3, 1) = M_t_ * m_ * g_ * l_ / mu;
//   A_(3, 2) = -c_ * l_ * m_ / mu;
//   A_(3, 3) = -gamma_ * M_t_ / mu;

//   B_ = Eigen::MatrixXd::Zero(4, 1);
//   B_(2, 0) = J_t_ / mu;
//   B_(3, 0) = l_ * m_ / mu;

//   C_ = Eigen::MatrixXd::Identity(4, 4);
//   D_ = Eigen::MatrixXd::Zero(4, 1);
// }

#endif