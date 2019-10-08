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

  using TransitionFn = std::function<void(const double dt, const VectorXd &u, Ref<VectorXd> x)>;
  using MeasurementFn = std::function<void(const Ref<const VectorXd> x, Ref<VectorXd> z)>;

  using ResidualFn = std::function<VectorXd(const Ref<const VectorXd> &m, const VectorXd &mean)>;
  using UnscentedMeanFn = std::function<VectorXd(const MatrixXd &sigma_points, const RowVectorXd &Wm)>;

  void cholesky(const MatrixXd &in, MatrixXd &out);

  void merwe_sigmas(const int state_dim, const double alpha, const double beta, const int kappa,
                    const MatrixXd &x, const MatrixXd &P, MatrixXd &sigma_points, RowVectorXd &Wm, RowVectorXd &Wc);

  VectorXd residual(const Ref<const VectorXd> &x, const VectorXd &mean);

  VectorXd unscented_mean(const MatrixXd &sigma_points, const RowVectorXd &Wm);

  MatrixXd unscented_covariance(const ResidualFn &r_x_fn, const MatrixXd &sigma_points, const RowVectorXd &Wc,
                                const VectorXd &x);

  void unscented_transform(const ResidualFn &r_x_fn, const UnscentedMeanFn &mean_fn, const MatrixXd &sigma_points,
                           const RowVectorXd &Wm, const RowVectorXd &Wc, VectorXd &x, MatrixXd &P);

  bool valid_x(const VectorXd &x);

  bool valid_P(const MatrixXd &P);

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
    VectorXd x_;                // Mean
    MatrixXd P_;                // Covariance

    // State after the predict step
    MatrixXd sigmas_p_;         // Predicted sigma points = f(sigma_points)
    RowVectorXd Wm_;            // Weights for computing mean
    RowVectorXd Wc_;            // Weights for computing covariance

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

    void set_x(const VectorXd &x)
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

    void predict(double dt, const VectorXd &u);

    void update(const VectorXd &z, const MatrixXd &R);
  };

} // namespace ukf

#endif // UKF_H