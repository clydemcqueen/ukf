#include "ukf.hpp"

#include <cassert>
#include <iostream>

namespace ukf
{
  using namespace Eigen;

  //========================================================================
  // Unscented math
  //========================================================================

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

  // sum of { Wm[i] * f(sigma[i]) }
  // Not useful for angles
  MatrixXd unscented_mean(const MatrixXd &sigma_points, const MatrixXd &Wm)
  {
    MatrixXd x = MatrixXd::Zero(sigma_points.rows(), 1);

    for (int i = 0; i < sigma_points.cols(); ++i) {
      x += Wm(0, i) * sigma_points.col(i);
    }

    return x;
  }

  // sum of { Wc[i] * (f(sigma[i]) - mean) * (f(sigma[i]) - mean).T } + Q
  MatrixXd unscented_covariance(const ResidualFn &r_x_fn, const MatrixXd &sigma_points, const MatrixXd &Wc,
                                const MatrixXd &x, const MatrixXd &Q)
  {
    MatrixXd P = MatrixXd::Zero(x.rows(), x.rows());

    for (int i = 0; i < sigma_points.cols(); ++i) {
      MatrixXd y = r_x_fn(sigma_points.col(i), x);
      P += Wc(0, i) * (y * y.transpose());
    }

    return P + Q;
  }

  // Compute the unscented transform
  void unscented_transform(const ResidualFn &r_x_fn, const UnscentedMeanFn &mean_fn, const MatrixXd &sigma_points,
                           const MatrixXd &Wm, const MatrixXd &Wc, const MatrixXd &Q, MatrixXd &x, MatrixXd &P)
  {
    x = mean_fn(sigma_points, Wm);
    P = unscented_covariance(r_x_fn, sigma_points, Wc, x, Q);
  }

  //========================================================================
  // UnscentedKalmanFilter
  //========================================================================

  UnscentedKalmanFilter::UnscentedKalmanFilter(int state_dim, int measurement_dim,
                                               double alpha, double beta, int kappa) :
    state_dim_{state_dim},
    measurement_dim_{measurement_dim},
    alpha_{alpha},
    beta_{beta},
    kappa_{kappa}
  {
    assert(state_dim > 0);
    assert(measurement_dim > 0);

    x_ = MatrixXd::Zero(state_dim, 1);
    P_ = MatrixXd::Identity(state_dim, state_dim);
    Q_ = MatrixXd::Identity(state_dim, state_dim);
    f_fn_ = nullptr;
    h_fn_ = nullptr;

    // The default residual and unscented mean functions are not useful for angles
    r_x_fn_ = [](const Ref<const MatrixXd> &x, const MatrixXd &mean) -> MatrixXd
    {
      return x - mean;
    };
    r_z_fn_ = [](const Ref<const MatrixXd> &z, const MatrixXd &mean) -> MatrixXd
    {
      return z - mean;
    };
    mean_x_fn_ = unscented_mean;
    mean_z_fn_ = unscented_mean;
  }

  void UnscentedKalmanFilter::predict(double dt, const MatrixXd &u)
  {
    assert(f_fn_);

    // Generate sigma points
    ukf::merwe_sigmas(state_dim_, alpha_, beta_, kappa_, x_, P_, sigmas_p_, Wm_, Wc_);

    // Predict the state at t + dt for each sigma point
    for (int i = 0; i < sigmas_p_.cols(); ++i) {
      f_fn_(dt, u, sigmas_p_.col(i));
    }

    // Find mean and covariance of the predicted sigma points
    ukf::unscented_transform(r_x_fn_, mean_x_fn_, sigmas_p_, Wm_, Wc_, Q_, x_p_, P_p_);
  }

  void UnscentedKalmanFilter::update(const MatrixXd &z, const MatrixXd &R)
  {
    assert(h_fn_);
    assert(z.rows() == measurement_dim_ && z.cols() == 1);
    assert(R.rows() == measurement_dim_ && R.cols() == measurement_dim_);

    // Transform sigma points into measurement space
    MatrixXd sigmas_z(measurement_dim_, sigmas_p_.cols());
    for (int i = 0; i < sigmas_p_.cols(); ++i) {
      h_fn_(sigmas_p_.col(i), sigmas_z.col(i));
    }

    // Find mean and covariance of sigma points in measurement space
    MatrixXd x_z;
    MatrixXd P_z;
    ukf::unscented_transform(r_z_fn_, mean_z_fn_, sigmas_z, Wm_, Wc_, R, x_z, P_z);

    // Find cross covariance of the sigma points in the state and measurement spaces
    MatrixXd P_xz = MatrixXd::Zero(state_dim_, measurement_dim_);
    for (int i = 0; i < sigmas_z.cols(); ++i) {
      MatrixXd y_p = r_x_fn_(sigmas_p_.col(i), x_p_);
      MatrixXd y_z = r_z_fn_(sigmas_z.col(i), x_z);
      P_xz += Wc_(0, i) * (y_p * y_z.transpose());
    }

    // Kalman gain
    K_ = P_xz * P_z.inverse();

    // Combine measurement and prediction into a new estimate
    MatrixXd y_z = r_z_fn_(z, x_z);
    x_ = x_p_ + K_ * y_z;
    P_ = P_p_ - K_ * P_z * K_.transpose();
  }

} // namespace ukf