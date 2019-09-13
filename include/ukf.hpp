#ifndef UKF_H
#define UKF_H

#include "eigen3/Eigen/Dense"

namespace ukf
{
  using namespace Eigen;

  //========================================================================
  // Unscented math
  //========================================================================

  void cholesky(const MatrixXd &in, MatrixXd &out);

  void merwe_sigmas(const int state_dim, const double alpha, const double beta, const int kappa,
                    const MatrixXd &x, const MatrixXd &P,
                    MatrixXd &sigma_points, MatrixXd &Wm, MatrixXd &Wc);

  MatrixXd unscented_mean(const MatrixXd &sigma_points, const MatrixXd &Wm);

  MatrixXd unscented_covariance(const MatrixXd &sigma_points, const MatrixXd &Wc, const MatrixXd &x, const MatrixXd &Q);

  void unscented_transform(const MatrixXd &sigma_points, const MatrixXd &Wm, const MatrixXd &Wc, const MatrixXd &Q,
                           MatrixXd &x, MatrixXd &P);

  //========================================================================
  // UnscentedKalmanFilter
  //========================================================================

  class UnscentedKalmanFilter
  {
    int state_dim_;           // Size of state space
    int measurement_dim_;     // Size of measurement space

    double alpha_;            // Generally 0≤α≤1, larger value spreads the sigma points further from the mean
    double beta_;             // β=2 is a good choice for Gaussian problems
    int kappa_;               // κ=3−state_dim_ is a good choice

    MatrixXd sigmas_;         // Sigma points
    MatrixXd sigmas_p_;       // Predicted sigma points = f(sigma_points)
    MatrixXd sigmas_z_;       // Sigma points in measurement space = h(f(sigma_points))
    MatrixXd Wm_;             // Weights for computing mean
    MatrixXd Wc_;             // Weights for computing covariance

    MatrixXd x_;              // State mean
    MatrixXd P_;              // State covariance
    MatrixXd Q_;              // Process covariance

    std::function<void(const double, Ref<MatrixXd>)> f_;          // State transition function
    std::function<void(const Ref<MatrixXd>, Ref<MatrixXd>)> h_;   // Measurement function

    std::function<void(const Ref<MatrixXd> x, const Ref<MatrixXd> mean, Ref<MatrixXd> y)> residual_x_;
    std::function<void(const Ref<MatrixXd> z, const Ref<MatrixXd> mean, Ref<MatrixXd> y)> residual_z_;

  public:

    explicit UnscentedKalmanFilter(int state_dim, int measurement_dim) :
      UnscentedKalmanFilter(state_dim, measurement_dim, 0.3, 2, 3 - state_dim)
    {}

    explicit UnscentedKalmanFilter(int state_dim, int measurement_dim, double alpha, double beta, int kappa_offset);

    ~UnscentedKalmanFilter()
    {}

    const auto &x() const
    { return x_; }

    const auto &P() const
    { return P_; }

    void set_x(const MatrixXd &x);

    void set_P(const MatrixXd &P);

    void set_Q(const MatrixXd &Q);

    void set_f(const std::function<void(const double, Ref<MatrixXd>)> &f);

    void set_h(const std::function<void(const Ref<MatrixXd>, Ref<MatrixXd>)> &h);

    void predict(double dt);

    void update(const MatrixXd &z, const MatrixXd &R);
  };

} // namespace ukf

#endif // UKF_H