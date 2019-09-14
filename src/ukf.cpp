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
  void merwe_sigmas(const int state_dim, const double alpha, const double beta, const int kappa,
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
  MatrixXd residual_x(const MatrixXd &x, const MatrixXd &mean)
  {
    return x - mean;
  }

  // Residual in measurement space
  // Not useful for angles
  // TODO support custom residual functions
  MatrixXd residual_z(const MatrixXd &z, const MatrixXd &mean)
  {
    return z - mean;
  }

  // sum of { Wm[i] * f(sigma[i]) }
  // Not useful for angles
  // TODO support custom mean functions
  MatrixXd unscented_mean(const MatrixXd &sigma_points, const MatrixXd &Wm)
  {
    MatrixXd x = MatrixXd::Zero(sigma_points.rows(), 1);

    for (int i = 0; i < sigma_points.cols(); ++i) {
      x += Wm(0, i) * sigma_points.col(i);
    }

    return x;
  }

  // sum of { Wc[i] * (f(sigma[i]) - mean) * (f(sigma[i]) - mean).T } + Q
  MatrixXd unscented_covariance(const MatrixXd &sigma_points, const MatrixXd &Wc, const MatrixXd &x, const MatrixXd &Q)
  {
    MatrixXd P = MatrixXd::Zero(x.rows(), x.rows());

    for (int i = 0; i < sigma_points.cols(); ++i) {
      MatrixXd y = residual_x(sigma_points.col(i), x);
      P += Wc(0, i) * (y * y.transpose());
    }

    return P + Q;
  }

  // Compute the unscented transform
  void unscented_transform(const MatrixXd &sigma_points, const MatrixXd &Wm, const MatrixXd &Wc, const MatrixXd &Q,
                           MatrixXd &x, MatrixXd &P)
  {
    x = unscented_mean(sigma_points, Wm);
    P = unscented_covariance(sigma_points, Wc, x, Q);
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
    residual_x_ = nullptr;
    residual_z_ = nullptr;
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

  void UnscentedKalmanFilter::set_f(const std::function<void(const double, Ref<MatrixXd>, const Ref<MatrixXd>)> &f)
  {
    f_ = f;
  }

  void UnscentedKalmanFilter::set_h(const std::function<void(const Ref<MatrixXd>, Ref<MatrixXd>)> &h)
  {
    h_ = h;
  }

  void UnscentedKalmanFilter::predict(double dt, const Ref<MatrixXd> u)
  {
    // Generate sigma points
    ukf::merwe_sigmas(state_dim_, alpha_, beta_, kappa_, x_, P_, sigmas_, Wm_, Wc_);

    // Predict the state at t + dt for each sigma point
    sigmas_p_ = sigmas_;
    for (int i = 0; i < sigmas_p_.cols(); ++i) {
      f_(dt, sigmas_p_.col(i), u);
    }

    // Find mean and covariance of the predicted sigma points
    // TODO save as x_p_ and P_p_
    ukf::unscented_transform(sigmas_p_, Wm_, Wc_, Q_, x_, P_);
  }

  void UnscentedKalmanFilter::update(const MatrixXd &z, const MatrixXd &R)
  {
    assert(z.rows() == measurement_dim_ && z.cols() == 1);
    assert(R.rows() == measurement_dim_ && R.cols() == measurement_dim_);

    // Transform sigma points into measurement space
    sigmas_z_ = MatrixXd(measurement_dim_, sigmas_p_.cols());
    for (int i = 0; i < sigmas_p_.cols(); ++i) {
      h_(sigmas_p_.col(i), sigmas_z_.col(i));
    }

    // Find mean and covariance of sigma points in measurement space
    MatrixXd x_z;
    MatrixXd P_z;
    ukf::unscented_transform(sigmas_z_, Wm_, Wc_, R, x_z, P_z);

    // Find cross covariance of the sigma points in the state and measurement spaces
    MatrixXd P_xz = MatrixXd::Zero(state_dim_, measurement_dim_);
    for (int i = 0; i < sigmas_z_.cols(); ++i) {
      MatrixXd y_p = residual_x(sigmas_p_.col(i), x_);  // TODO x_p_
      MatrixXd y_z = residual_z(sigmas_z_.col(i), x_z);
      P_xz += Wc_(0, i) * (y_p * y_z.transpose());
    }

    // Kalman gain
    MatrixXd K = P_xz * P_z.inverse();

    // Combine measurement and prediction into a new estimate
    MatrixXd y_z = residual_z(z, x_z);
    x_ = x_ + K * y_z; // TODO x_p_
    P_ = P_ - K * P_z * K.transpose();  // TODO P_p_
  }

} // namespace ukf