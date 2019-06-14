#include "ukf.hpp"

#include <cassert>
#include <iostream>

namespace ukf
{
  using namespace Eigen;

  //========================================================================
  // Unscented math
  //========================================================================

  // Move an angle to the region [-M_PI, M_PI]
  constexpr double norm_angle(double a)
  {
    while (a < -M_PI) {
      a += 2 * M_PI;
    }
    while (a > M_PI) {
      a -= 2 * M_PI;
    }

    return a;
  }

  // Square root of a matrix
  void cholesky(const MatrixXd &in, MatrixXd &out)
  {
    LLT<MatrixXd, Lower> lltOfA(in);
    out = lltOfA.matrixL();
  }

  // Generate Merwe scaled sigma points
  void merwe_sigma_points(const int state_dim, const double alpha, const double beta, const int kappa,
                          const MatrixXd &x, const MatrixXd &P,
                          MatrixXd &sigma_points, MatrixXd &Wm, MatrixXd &Wc)
  {
    int num_points = 2 * state_dim + 1;
    double lambda = alpha * alpha * (state_dim + kappa) - state_dim;
    double Wm_0 = lambda / (state_dim + lambda);
    double Wc_0 = lambda / (state_dim + lambda) + 1 - alpha * alpha + beta;
    double v = 1. / (2. * (state_dim + lambda));

    Wm = MatrixXd::Constant(1, num_points, v);
    Wm(0, 0) = Wm_0;

    Wc = MatrixXd::Constant(1, num_points, v);
    Wc(0, 0) = Wc_0;

    sigma_points = MatrixXd::Zero(state_dim, num_points);
    MatrixXd U;
    cholesky((state_dim + lambda) * P, U);
    sigma_points.col(0) = x;
    for (int i = 0; i < state_dim; ++i) {
      sigma_points.col(i + 1) = x + U.col(i);
      sigma_points.col(i + 1 + state_dim) = x - U.col(i);
    }
  }

  // Residual in state space
  // Not useful for angles
  // TODO support custom residual functions
  void residual_x(const MatrixXd &x, const MatrixXd &mean,
                  MatrixXd &y)
  {
    y = x - mean;
  }

  // Residual in measurement space
  // Not useful for angles
  // TODO support custom residual functions
  void residual_z(const MatrixXd &z, const MatrixXd &mean,
                  MatrixXd &y)
  {
    y = z - mean;
  }

  // sum of { Wm[i] * f(sigma[i]) }
  // Not useful for angles
  // TODO support custom mean functions
  void unscented_mean(const MatrixXd &sigma_points, const MatrixXd &Wm,
                      MatrixXd &x)
  {
    x = MatrixXd::Zero(sigma_points.rows(), 1);

    for (int i = 0; i < sigma_points.cols(); ++i) {
      x += Wm(0, i) * sigma_points.col(i);
    }
  }

  // sum of { Wc[i] * (f(sigma[i]) - mean) * (f(sigma[i]) - mean).T } + Q
  void unscented_covariance(const MatrixXd &sigma_points, const MatrixXd &Wc, const MatrixXd &x, const MatrixXd &Q,
                            MatrixXd &P)
  {
    P = MatrixXd::Zero(x.rows(), x.rows());

    for (int i = 0; i < sigma_points.cols(); ++i) {
      MatrixXd y;
      residual_x(sigma_points.col(i), x, y);
      P += Wc(0, i) * (y * y.transpose());
    }

    P += Q;
  }

  // Compute the unscented transform
  void unscented_transform(const MatrixXd &sigma_points, const MatrixXd &Wm, const MatrixXd &Wc, const MatrixXd &Q,
                           MatrixXd &x, MatrixXd &P)
  {
    unscented_mean(sigma_points, Wm, x);
    unscented_covariance(sigma_points, Wc, x, Q, P);
  }

  //========================================================================
  // UnscentedKalmanFilter
  //========================================================================

  UnscentedKalmanFilter::UnscentedKalmanFilter(int state_dim, int measurement_dim,
                                               double alpha, double beta, int kappa_offset) :
    state_dim_{state_dim},
    measurement_dim_{measurement_dim},
    alpha_{alpha},
    beta_{beta},
    kappa_{kappa_offset - measurement_dim}
  {
    assert(state_dim > 0);
    assert(measurement_dim > 0);

    x_ = MatrixXd::Zero(state_dim, 1);
    P_ = MatrixXd::Identity(state_dim, state_dim);
    Q_ = MatrixXd::Identity(state_dim, state_dim);
    f_ = nullptr;
    h_ = nullptr;
  }

  void UnscentedKalmanFilter::set_x(const MatrixXd &x)
  {
    assert(x.rows() == state_dim_ && x.cols() == 1);
    x_ = x;
  }

  void UnscentedKalmanFilter::set_P(const MatrixXd &P)
  {
    assert(P.rows() == state_dim_ && P.cols() == state_dim_);
    P_ = P;
  }

  void UnscentedKalmanFilter::set_Q(const MatrixXd &Q)
  {
    assert(Q.rows() == state_dim_ && Q.cols() == state_dim_);
    Q_ = Q;
  }

  void UnscentedKalmanFilter::set_f(const std::function<void(const double, Ref<MatrixXd>)> &f)
  {
    f_ = f;
  }

  void UnscentedKalmanFilter::set_h(const std::function<void(const Ref<MatrixXd>, Ref<MatrixXd>)> &h)
  {
    h_ = h;
  }

  void UnscentedKalmanFilter::predict(double dt)
  {
    // Generate sigma points
    ukf::merwe_sigma_points(state_dim_, alpha_, beta_, kappa_, x_, P_, sigma_points_, Wm_, Wc_);

    // Predict the state at t + dt for each sigma point
    for (int i = 0; i < sigma_points_.cols(); ++i) {
      f_(dt, sigma_points_.col(i));
    }

    // Find mean and covariance of the predicted sigma points
    ukf::unscented_transform(sigma_points_, Wm_, Wc_, Q_, x_, P_);
  }

  void UnscentedKalmanFilter::update(const MatrixXd &z, const MatrixXd &R)
  {
    assert(z.rows() == measurement_dim_ && z.cols() == 1);
    assert(R.rows() == measurement_dim_ && R.cols() == measurement_dim_);

    // Transform sigma points into measurement space
    MatrixXd sigma_points_z = MatrixXd(measurement_dim_, sigma_points_.cols());
    for (int i = 0; i < sigma_points_.cols(); ++i) {
      h_(sigma_points_.col(i), sigma_points_z.col(i));
    }

    // Find mean and covariance of sigma points in measurement space
    MatrixXd x_z;
    MatrixXd P_z;
    ukf::unscented_transform(sigma_points_z, Wm_, Wc_, R, x_z, P_z);

    // Find cross covariance of the sigma points in the state and measurement spaces
    MatrixXd P_xz = MatrixXd::Zero(state_dim_, measurement_dim_);
    for (int i = 0; i < sigma_points_z.cols(); ++i) {

      MatrixXd y_x;
      residual_x(sigma_points_.col(i), x_, y_x);

      MatrixXd y_z;
      residual_z(sigma_points_z.col(i), x_z, y_z);

      P_xz += Wc_(0, i) * (y_x * y_z.transpose());
    }

    // Kalman gain
    MatrixXd K = P_xz * P_z.inverse();

    // Combine measurement and prediction into a new estimate
    MatrixXd y_z;
    residual_z(z, x_z, y_z);
    x_ = x_ + K * y_z;
    P_ = P_ - K * P_z * K.transpose();
  }

} // namespace ukf