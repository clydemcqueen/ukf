#include "ukf/ukf.hpp"

#include <cassert>
#include <iostream>

namespace ukf
{
using namespace Eigen;

//========================================================================
// Unscented math
//========================================================================

bool is_finite(const MatrixXd & m)
{
  return m.array().isFinite().count() == m.rows() * m.cols();
}

bool valid_x(const VectorXd & x)
{
  if (!is_finite(x)) {
    std::cout << "x is not finite: " << std::endl << x << std::endl;
    return false;
  }

  return true;
}

bool valid_P(const MatrixXd & P)
{
  if (!is_finite(P)) {
    std::cout << "P is not finite: " << std::endl << P << std::endl;
    return false;
  }

#undef CHECK_EIGENVALUES
#ifdef CHECK_EIGENVALUES
  EigenSolver<MatrixXd> eigen_solver(P);
  auto eigen_values = eigen_solver.eigenvalues();
  for (int r = 0; r < eigen_values.rows(); ++r) {
    auto eigen_value = eigen_values(r, 0);
    if (eigen_value.real() < 0) {
      // std::cout << "negative eigenvalue for P: " << std::endl << P << std::endl;
      return false;
    }
  }
#endif

  return true;
}

// Square root of a matrix
void cholesky(const MatrixXd & in, MatrixXd & out)
{
  LLT<MatrixXd, Lower> lltOfA(in);
  out = lltOfA.matrixL();
}

// Generate Merwe scaled sigma points
void merwe_sigmas(const int state_dim, const double alpha, const double beta, const int kappa,
  const MatrixXd & x, const MatrixXd & P,
  MatrixXd & sigma_points, RowVectorXd & Wm, RowVectorXd & Wc)
{
  int num_points = 2 * state_dim + 1;
  double lambda = alpha * alpha * (state_dim + kappa) - state_dim;
  double Wm_0 = lambda / (state_dim + lambda);
  double Wc_0 = lambda / (state_dim + lambda) + 1 - alpha * alpha + beta;
  double v = 1. / (2. * (state_dim + lambda));

  Wm = RowVectorXd::Constant(num_points, v);
  Wm(0) = Wm_0;

  Wc = RowVectorXd::Constant(num_points, v);
  Wc(0) = Wc_0;

  sigma_points = MatrixXd::Zero(state_dim, num_points);
  MatrixXd U;
  cholesky((state_dim + lambda) * P, U);
  sigma_points.col(0) = x;
  for (int i = 0; i < state_dim; ++i) {
    sigma_points.col(i + 1) = x + U.col(i);
    sigma_points.col(i + 1 + state_dim) = x - U.col(i);
  }
}

// Simple residual function
// Not useful for angles
VectorXd residual(const Ref<const VectorXd> & x, const VectorXd & mean)
{
  return x - mean;
}

// sum of { Wm[i] * f(sigma[i]) }
// Not useful for angles
VectorXd unscented_mean(const MatrixXd & sigma_points, const RowVectorXd & Wm)
{
  VectorXd x = VectorXd::Zero(sigma_points.rows());

  for (int i = 0; i < sigma_points.cols(); ++i) {
    x += Wm(i) * sigma_points.col(i);
  }

  return x;
}

// sum of { Wc[i] * (f(sigma[i]) - mean) * (f(sigma[i]) - mean).T }
MatrixXd unscented_covariance(const ResidualFn & r_x_fn, const MatrixXd & sigma_points,
  const RowVectorXd & Wc, const VectorXd & x)
{
  MatrixXd P = MatrixXd::Zero(x.rows(), x.rows());

  for (int i = 0; i < sigma_points.cols(); ++i) {
    VectorXd y = r_x_fn(sigma_points.col(i), x);
    P += Wc(i) * (y * y.transpose());
  }

  // Caller must add Q (or R)
  return P;
}

// Compute the unscented transform
void unscented_transform(const ResidualFn & r_x_fn, const UnscentedMeanFn & mean_fn,
  const MatrixXd & sigma_points, const RowVectorXd & Wm, const RowVectorXd & Wc,
  VectorXd & x, MatrixXd & P)
{
  x = mean_fn(sigma_points, Wm);

  // Caller must add Q (or R)
  P = unscented_covariance(r_x_fn, sigma_points, Wc, x);
}

// Use a Mahalanobis test to reject outliers, expressed in std deviations from x
bool outlier(const VectorXd & y_z, const MatrixXd & P_z_inverse, const double distance)
{
  return y_z.dot(P_z_inverse * y_z) >= distance * distance;
}

//========================================================================
// UnscentedKalmanFilter
//========================================================================

UnscentedKalmanFilter::UnscentedKalmanFilter(int state_dim, double alpha, double beta, int kappa)
  : outlier_distance_{std::numeric_limits<double>::max()},
    state_dim_{state_dim},
    alpha_{alpha},
    beta_{beta},
    kappa_{kappa}
{
  assert(state_dim > 0);

  x_ = VectorXd::Zero(state_dim);
  P_ = MatrixXd::Identity(state_dim, state_dim);
  Q_ = MatrixXd::Identity(state_dim, state_dim);
  f_fn_ = nullptr;
  h_fn_ = nullptr;

  // The default residual and unscented mean functions are not useful for angles
  r_x_fn_ = residual;
  r_z_fn_ = residual;
  mean_x_fn_ = unscented_mean;
  mean_z_fn_ = unscented_mean;
}

bool UnscentedKalmanFilter::valid()
{
  assert(x_.rows() == state_dim_ && x_.cols() == 1);
  assert(P_.rows() == state_dim_ && P_.cols() == state_dim_);

  return valid_x(x_) && valid_P(P_);
}

void UnscentedKalmanFilter::predict(double dt, const VectorXd & u)
{
  assert(f_fn_);

  // Generate sigma points
  ukf::merwe_sigmas(state_dim_, alpha_, beta_, kappa_, x_, P_, sigmas_p_, Wm_, Wc_);

  // Predict the state at t + dt for each sigma point
  for (int i = 0; i < sigmas_p_.cols(); ++i) {
    f_fn_(dt, u, sigmas_p_.col(i));
  }

  // Find mean and covariance of the predicted sigma points
  ukf::unscented_transform(r_x_fn_, mean_x_fn_, sigmas_p_, Wm_, Wc_, x_, P_);

  // Add process noise to the covariance
  P_ += Q_ * dt;
}

bool UnscentedKalmanFilter::update(const VectorXd & z, const MatrixXd & R)
{
  int measurement_dim = z.rows();

  assert(h_fn_);
  assert(z.rows() > 0 && z.cols() == 1);
  assert(R.rows() == measurement_dim && R.cols() == measurement_dim);

  // Transform sigma points into measurement space
  MatrixXd sigmas_z(measurement_dim, sigmas_p_.cols());
  for (int i = 0; i < sigmas_p_.cols(); ++i) {
    h_fn_(sigmas_p_.col(i), sigmas_z.col(i));
  }

  // Find mean and covariance of sigma points in measurement space
  // Measurement noise is not dependent on dt
  VectorXd x_z;
  MatrixXd P_z;
  ukf::unscented_transform(r_z_fn_, mean_z_fn_, sigmas_z, Wm_, Wc_, x_z, P_z);

  // Add measurement noise to the covariance
  P_z += R;

  // Find cross covariance of the sigma points in the state and measurement spaces
  MatrixXd P_xz = MatrixXd::Zero(state_dim_, measurement_dim);
  for (int i = 0; i < sigmas_z.cols(); ++i) {
    VectorXd y_p = r_x_fn_(sigmas_p_.col(i), x_);
    VectorXd y_z = r_z_fn_(sigmas_z.col(i), x_z);
    P_xz += Wc_(i) * (y_p * y_z.transpose());
  }

  // Kalman gain
  MatrixXd P_z_inverse = P_z.inverse();
  K_ = P_xz * P_z_inverse;
  // assert(K_.rows() == state_dim_);
  // assert(K_.cols() == measurement_dim_);

  // Compute the innovation
  MatrixXd y_z = r_z_fn_(z, x_z);

  // Reject outliers
  if (outlier_distance_ < std::numeric_limits<double>::max() &&
    outlier(y_z, P_z_inverse, outlier_distance_)) {
    return false;
  }

  // Combine measurement and prediction into a new estimate
  x_ = x_ + K_ * y_z;
  P_ = P_ - K_ * P_z * K_.transpose();

  return true;
}

} // namespace ukf