#ifndef PID_H
#define PID_H

class PID {
 public:
  /**
   * Constructor
   */
  PID() {}

  /**
   * Destructor.
   */
  virtual ~PID() {}

  /**
   * Initialize PID.
   * @param (Kp_, Ki_, Kd_) The initial PID coefficients
   */
  void Init(double Kp_, double Ki_, double Kd_) {
    /**
     * Initialize PID coefficients (and errors, if needed)
     */
    Kp = Kp_;
    Ki = Ki_;
    Kd = Kd_;

    p_error = 0;
    i_error = 0;
    d_error = 0;
    }

  /**
   * Update the PID error variables given cross track error.
   * @param cte The current cross track error
   */
  void UpdateError(double time, double cte) {
    /**
     * Update PID errors based on cte.
     */
    double dt = time - previous_time;
    previous_time = time;
    double prev_cte = p_error;

    p_error = cte;
    i_error = dt * i_error + cte;
    d_error = (cte - prev_cte) / dt;
    }

  /**
   * Calculate the total PID error.
   * @output The total PID error
   */
  double TotalError()  {
    /**
     * Calculate and return the total error
     */
    double total = Kp * p_error + Ki * i_error + Kd * d_error;
    return total;  // Add your total error calc here!
    }


 private:
  double previous_time;

  /**
   * PID Errors
   */
  double p_error;
  double i_error;
  double d_error;

  /**
   * PID Coefficients
   */
  double Kp;
  double Ki;
  double Kd;
};


// PID::PID() {}

// PID::~PID() {}

// void PID::Init(double Kp_, double Ki_, double Kd_) {
//   /**
//    * Initialize PID coefficients (and errors, if needed)
//    */
//   Kp = Kp_;
//   Ki = Ki_;
//   Kd = Kd_;

//   p_error = 0;
//   i_error = 0;
//   d_error = 0;
// }

// void PID::UpdateError(double time, double cte) {
//   /**
//    * Update PID errors based on cte.
//    */
//   double dt = time - previous_time;
//   previous_time = time;
//   double prev_cte = p_error;

//   p_error = cte;
//   i_error = dt * i_error + cte;
//   d_error = (cte - prev_cte) / dt;
// }

// double PID::TotalError() {
//   /**
//    * Calculate and return the total error
//    */
//   double total = Kp * p_error + Ki * i_error + Kd * d_error;
//   return total;  // Add your total error calc here!
// }

#endif  // PID_H