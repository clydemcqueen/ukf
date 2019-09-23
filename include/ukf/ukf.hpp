#ifndef UKF_H
#define UKF_H

#include "eigen3/Eigen/Dense"

namespace ukf
{
  using namespace Eigen;

  // Eigen Refs provide a way to pass in a reference to a column or row while avoiding copies.
  // There are 2 forms:
  //    Read-write:  void foo1(Ref<VectorXf> x);
  //    Read-only:   void foo2(const Ref<const VectorXf> &x);
  // From https://eigen.tuxfamily.org/dox/classEigen_1_1Ref.html

  //========================================================================
  // Unscented math
  //========================================================================

  typedef std::function<void(const double dt, const MatrixXd &u, Ref<MatrixXd> x)> TransitionFn;
  typedef std::function<void(const Ref<const MatrixXd> x, Ref<MatrixXd> z)> MeasurementFn;

  typedef std::function<MatrixXd(const Ref<const MatrixXd> &m, const MatrixXd &mean)> ResidualFn;
  typedef std::function<MatrixXd(const MatrixXd &sigma_points, const MatrixXd &Wm)> UnscentedMeanFn;

  void cholesky(const MatrixXd &in, MatrixXd &out);

  void merwe_sigmas(const int state_dim, const double alpha, const double beta, const int kappa,
                    const MatrixXd &x, const MatrixXd &P, MatrixXd &sigma_points, MatrixXd &Wm, MatrixXd &Wc);

  MatrixXd unscented_mean(const MatrixXd &sigma_points, const MatrixXd &Wm);

  MatrixXd unscented_covariance(const ResidualFn &r_x_fn, const MatrixXd &sigma_points, const MatrixXd &Wc,
                                const MatrixXd &x, const MatrixXd &Q);

  void unscented_transform(const ResidualFn &r_x_fn, const UnscentedMeanFn &mean_fn, const MatrixXd &sigma_points,
                           const MatrixXd &Wm, const MatrixXd &Wc, const MatrixXd &Q, MatrixXd &x, MatrixXd &P);

  //========================================================================
  // UnscentedKalmanFilter
  //========================================================================

  class UnscentedKalmanFilter
  {
    // Inputs
    int state_dim_;             // Size of state space
    MatrixXd Q_;                // Process covariance

    // Additional inputs: constants for generating Merwe sigma points
    double alpha_;              // Generally 0≤α≤1, larger value spreads the sigma points further from the mean
    double beta_;               // β=2 is a good choice for Gaussian problems
    int kappa_;                 // κ=0 is a good default

    // Current state
    MatrixXd x_;                // Mean
    MatrixXd P_;                // Covariance

    // State after the predict step
    MatrixXd sigmas_p_;         // Predicted sigma points = f(sigma_points)
    MatrixXd Wm_;               // Weights for computing mean
    MatrixXd Wc_;               // Weights for computing covariance
    MatrixXd x_p_;              // Predicted mean
    MatrixXd P_p_;              // Predicted covariance

    // State after the update step, for diagnostics
    MatrixXd K_;                // Kalman gain

    // These functions must be provided
    TransitionFn f_fn_;         // State transition function
    MeasurementFn h_fn_;        // Measurement function

    // Defaults are provided for these functions, but they can be overridden
    ResidualFn r_x_fn_;         // Compute x - mean
    ResidualFn r_z_fn_;         // Compute z - mean_in_z_space
    UnscentedMeanFn mean_x_fn_; // Compute the mean of the sigma points
    UnscentedMeanFn mean_z_fn_; // Compute the mean of the sigma points in z space

  public:

    explicit UnscentedKalmanFilter(int state_dim, double alpha, double beta, int kappa);

    ~UnscentedKalmanFilter()
    {}

    const auto &x() const
    { return x_; }

    const auto &P() const
    { return P_; }

    const auto &K() const
    { return K_; }

    void set_x(const MatrixXd &x)
    {
      assert(x.rows() == state_dim_ && x.cols() == 1);
      x_ = x;
    }

    void set_P(const MatrixXd &P)
    {
      assert(P.rows() == state_dim_ && P.cols() == state_dim_);
      P_ = P;
    }

    void set_Q(const MatrixXd &Q)
    {
      assert(Q.rows() == state_dim_ && Q.cols() == state_dim_);
      Q_ = Q;
    }

    void set_f_fn(const TransitionFn &f_fn)
    { f_fn_ = f_fn; }

    void set_h_fn(const MeasurementFn &h_fn)
    { h_fn_ = h_fn; }

    void set_r_x_fn(const ResidualFn &r_x_fn)
    { r_x_fn_ = r_x_fn; }

    void set_r_z_fn(const ResidualFn &r_z_fn)
    { r_z_fn_ = r_z_fn; }

    void set_mean_x_fn(const UnscentedMeanFn &mean_x_fn)
    { mean_x_fn_ = mean_x_fn; }

    void set_mean_z_fn(const UnscentedMeanFn &mean_z_fn)
    { mean_z_fn_ = mean_z_fn; }

    bool valid();

    bool predict(double dt, const MatrixXd &u);

    bool update(const MatrixXd &z, const MatrixXd &R);
  };

} // namespace ukf

#endif // UKF_H