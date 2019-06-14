#include <iostream>

#include "ukf.hpp"

using namespace Eigen;

template<typename DerivedA, typename DerivedB>
bool allclose(const DenseBase<DerivedA> &a,
              const DenseBase<DerivedB> &b,
              const typename DerivedA::RealScalar &rtol
              = NumTraits<typename DerivedA::RealScalar>::dummy_precision(),
              const typename DerivedA::RealScalar &atol
              = NumTraits<typename DerivedA::RealScalar>::epsilon())
{
  return ((a.derived() - b.derived()).array().abs()
          <= (atol + rtol * b.derived().array().abs())).all();
}

void test_cholesky()
{
  std::cout << "\n========= CHOLESKY =========\n" << std::endl;

  MatrixXd t(2, 2);
  t << 2, 0.1, 0.1, 3;

  MatrixXd s;
  ukf::cholesky(t, s);

  MatrixXd square = s * s.transpose();

  std::cout << (allclose(t, square) ? "Cholesky OK" : "Cholesky FAIL") << std::endl;
}

void test_generate_sigma_points()
{
  std::cout << "\n========= GENERATE SIGMA POINTS =========\n" << std::endl;

  double alpha{0.3}, beta{2};
  int kappa_offset{3};
  int measurement_dim{1};

  for (int state_dim = 1; state_dim < 5; ++state_dim) {
    std::cout << state_dim << " dimension(s):" << std::endl;

    MatrixXd x = MatrixXd::Zero(state_dim, 1);
    MatrixXd P = MatrixXd::Identity(state_dim, state_dim);

    // Make sure the sizes are correct
    ukf::UnscentedKalmanFilter filter(state_dim, measurement_dim, alpha, beta, kappa_offset);
    filter.set_x(x);
    filter.set_P(P);

    MatrixXd Wm;
    MatrixXd Wc;
    MatrixXd sigma_points;

    ukf::merwe_sigma_points(state_dim, alpha, beta, kappa_offset - state_dim, x, P, sigma_points, Wm, Wc);

    std::cout << "Wm:" << std::endl << Wm << std::endl;
    std::cout << "Wc:" << std::endl << Wc << std::endl;
    std::cout << "Sigma points (column vectors):" << std::endl << sigma_points << std::endl;

    int num_points = 2 * state_dim + 1;
    assert(Wc.rows() == 1 && Wm.rows() == 1 && sigma_points.rows() == x.rows());
    assert(Wc.cols() == num_points && Wm.cols() == num_points && sigma_points.cols() == num_points);
  }
}

void test_unscented_transform()
{
  std::cout << "\n========= UNSCENTED TRANSFORM =========\n" << std::endl;

  double alpha{0.3}, beta{2};
  int kappa_offset{3};
  int measurement_dim{1};

  for (int state_dim = 1; state_dim < 10; ++state_dim) {
    std::cout << state_dim << " dimension(s):" << std::endl;

    MatrixXd x = MatrixXd::Zero(state_dim, 1);
    MatrixXd P = MatrixXd::Identity(state_dim, state_dim);
    MatrixXd Q = MatrixXd::Zero(state_dim, state_dim);

    // Make sure the sizes are correct
    ukf::UnscentedKalmanFilter filter(state_dim, measurement_dim, alpha, beta, kappa_offset);
    filter.set_x(x);
    filter.set_P(P);
    filter.set_Q(Q);

    MatrixXd Wm;
    MatrixXd Wc;
    MatrixXd sigma_points;

    // Generate sigma points
    ukf::merwe_sigma_points(state_dim, alpha, beta, kappa_offset - state_dim, x, P, sigma_points, Wm, Wc);
    // std::cout << "Sigma points (column vectors):" << std::endl << sigma_points << std::endl;

    // Assume process model f() == identity
    MatrixXd sigma_points_f = sigma_points;

    // Compute mean of sigma_points_f
    MatrixXd x_f;
    ukf::unscented_mean(sigma_points_f, Wm, x_f);
    assert(x_f.rows() == x.rows() && x_f.cols() == 1);
    // std::cout << "x_f:" << std::endl << x_f << std::endl;
    std::cout << (allclose(x, x_f) ? "mean OK" : "mean FAIL") << std::endl;

    // Compute covariance of sigma_points_f
    MatrixXd P_f;
    ukf::unscented_covariance(sigma_points_f, Wc, x_f, Q, P_f);
    assert(P_f.rows() == P.rows() && P_f.cols() == P.cols());
    // std::cout << "P_f:" << std::endl << P_f << std::endl;
    std::cout << (allclose(P, P_f) ? "covar OK" : "covar FAIL") << std::endl;
  }
}

void test_filter()
{
  std::cout << "\n========= FILTER =========\n" << std::endl;

  int state_dim{6};
  int measurement_dim{2};

  ukf::UnscentedKalmanFilter filter(state_dim, measurement_dim);
  filter.set_x(MatrixXd::Zero(state_dim, 1));
  filter.set_P(MatrixXd::Identity(state_dim, state_dim));
  filter.set_Q(MatrixXd::Zero(state_dim, state_dim));

  // State transition function
  filter.set_f([](const double dt, Ref<MatrixXd> x)
               {
                 // vx += ax * dt
                 x(1, 0) += x(2, 0) * dt;

                 // x += vx * dt
                 x(0, 0) += x(1, 0) * dt;

                 // vy += ay * dt
                 x(4, 0) += x(5, 0) * dt;

                 // y += vy * dt
                 x(3, 0) += x(4, 0) * dt;
               });

  // Measurement function
  filter.set_h([](const Ref<MatrixXd> x, Ref<MatrixXd> z)
               {
                 // x
                 z(0, 0) = x(0, 0);

                 // y
                 z(1, 0) = x(3, 0);
               });

  MatrixXd z = MatrixXd::Zero(measurement_dim, 1);
  MatrixXd R = MatrixXd::Identity(measurement_dim, measurement_dim);

  for (int i = 0; i < 100; ++i) {
    filter.predict(0.1);
    filter.update(z, R);
  }

  assert(filter.x().rows() == state_dim && filter.x().cols() == 1);
  assert(filter.P().rows() == state_dim && filter.P().cols() == state_dim);

  std::cout << "100 iterations" << std::endl;
  std::cout << "estimated x:" << std::endl << filter.x() << std::endl;
  std::cout << "estimated P:" << std::endl << filter.P() << std::endl;
}

int main(int argc, char **argv)
{
  test_cholesky();
  test_generate_sigma_points();
  test_unscented_transform();
  test_filter();
  return 0;
}