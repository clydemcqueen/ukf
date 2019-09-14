#include <iostream>
#include <fstream>
#include <random>

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

void test_generate_sigmas()
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
    MatrixXd sigmas;

    ukf::merwe_sigmas(state_dim, alpha, beta, kappa_offset - state_dim, x, P, sigmas, Wm, Wc);

    std::cout << "Wm:" << std::endl << Wm << std::endl;
    std::cout << "Wc:" << std::endl << Wc << std::endl;
    std::cout << "Sigma points (column vectors):" << std::endl << sigmas << std::endl;

    int num_points = 2 * state_dim + 1;
    assert(Wc.rows() == 1 && Wm.rows() == 1 && sigmas.rows() == x.rows());
    assert(Wc.cols() == num_points && Wm.cols() == num_points && sigmas.cols() == num_points);
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
    MatrixXd sigmas;

    // Generate sigma points
    ukf::merwe_sigmas(state_dim, alpha, beta, kappa_offset - state_dim, x, P, sigmas, Wm, Wc);

    // Assume process model f() == identity
    MatrixXd sigmas_p = sigmas;

    // Compute mean of sigmas_p
    MatrixXd x_f = ukf::unscented_mean(sigmas_p, Wm);
    assert(x_f.rows() == x.rows() && x_f.cols() == 1);
    std::cout << (allclose(x, x_f) ? "mean OK" : "mean FAIL") << std::endl;

    // Compute covariance of sigmas_p
    MatrixXd P_f = ukf::unscented_covariance(sigmas_p, Wc, x_f, Q);
    assert(P_f.rows() == P.rows() && P_f.cols() == P.cols());
    std::cout << (allclose(P, P_f) ? "covar OK" : "covar FAIL") << std::endl;
  }
}

void test_filter()
{
  std::cout << "\n========= FILTER =========\n" << std::endl;

  // State: [x, vx, ax, y, vy, zy]T
  int state_dim{6};
  int measurement_dim{2};
  int control_dim{0};

  ukf::UnscentedKalmanFilter filter(state_dim, measurement_dim);
  filter.set_x(MatrixXd::Zero(state_dim, 1));
  filter.set_P(MatrixXd::Identity(state_dim, state_dim));
  filter.set_Q(MatrixXd::Zero(state_dim, state_dim));

  // State transition function
  filter.set_f([](const double dt, Ref<MatrixXd> x, const Ref<MatrixXd> u)
               {
                 // Ignore u
                 // ax and ay are discovered

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
  MatrixXd u = MatrixXd::Zero(control_dim, 1);

  for (int i = 0; i < 100; ++i) {
    filter.predict(0.1, u);
    filter.update(z, R);
  }

  assert(filter.x().rows() == state_dim && filter.x().cols() == 1);
  assert(filter.P().rows() == state_dim && filter.P().cols() == state_dim);

  std::cout << "100 iterations" << std::endl;
  std::cout << "estimated x:" << std::endl << filter.x() << std::endl;
  std::cout << "estimated P:" << std::endl << filter.P() << std::endl;
}

void test_1d_drag_filter(bool use_control, std::string filename)
{
  std::cout << "\n========= 1D DRAG FILTER " << use_control << " =========\n" << std::endl;

  int state_dim = use_control ? 2 : 3;      // [x, vx]T or [x, vx, ax]T
  int measurement_dim = 1;                  // [x]T
  int control_dim = use_control ? 1 : 0;    // [thrust_ax]T or []

  // Inputs
  double target_vx = 1.0;
  double drag_constant = 0.1;
  double dt = 1.0;

  ukf::UnscentedKalmanFilter filter(state_dim, measurement_dim);
  filter.set_x(MatrixXd::Zero(state_dim, 1));
  filter.set_P(MatrixXd::Identity(state_dim, state_dim));
  filter.set_Q(MatrixXd::Zero(state_dim, state_dim));

  // State transition function
  if (use_control) {
    filter.set_f([drag_constant](const double dt, Ref<MatrixXd> x, const Ref<MatrixXd> u)
                 {
                   // ax = thrust_ax - vx * vx * drag_constant
                   double ax = u(0, 0) - x(1, 0) * x(1, 0) * drag_constant;

                   // vx += ax * dt
                   x(1, 0) += ax * dt;

                   // x += vx * dt
                   x(0, 0) += x(1, 0) * dt;
                 });
  } else {
    filter.set_f([](const double dt, Ref<MatrixXd> x, const Ref<MatrixXd> u)
                 {
                   // Ignore u
                   // ax is discovered, drag is hidden inside ax

                   // vx += ax * dt
                   x(1, 0) += x(2, 0) * dt;

                   // x += vx * dt
                   x(0, 0) += x(1, 0) * dt;
                 });
  }

  // Measurement function
  filter.set_h([](const Ref<MatrixXd> x, Ref<MatrixXd> z)
               {
                 // x
                 z(0, 0) = x(0, 0);
               });

  MatrixXd z = MatrixXd::Zero(measurement_dim, 1);
  MatrixXd R = MatrixXd::Identity(measurement_dim, measurement_dim);

  // Thruster is always on and generates a constant force
  double thrust_ax = target_vx * target_vx * drag_constant;
  MatrixXd u = MatrixXd::Zero(control_dim, 1);
  if (use_control) {
    u(0, 0) = thrust_ax;
  }

  // Write state so we can use matplotlib later
  std::ofstream f;
  f.open(filename);
  f << "t, actual_x, actual_vx, actual.ax, z, x.x, x.vx, x.ax" << std::endl;

  // Initial state
  double actual_ax = thrust_ax;
  double actual_vx = 0.0;
  double actual_x = 0.0;

  // Add some noise
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0, 1);

  // Run a simulation
  int num_i = 40;
  for (int i = 0; i < num_i; ++i) {
    // Measurement
    z(0, 0) = actual_x + distribution(generator);

    // Predict and update
    filter.predict(dt, u);
    filter.update(z, R);

    // Write to file
    auto x = filter.x();
    double ax = use_control ? 0 : x(2, 0);
    f << i << ", "
      << actual_x << ", "
      << actual_vx << ", "
      << actual_ax << ", "
      << z << ", "
      << x(0, 0) << ", "
      << x(1, 0) << ", "
      << ax << std::endl;

    // Generate new z
    actual_ax = thrust_ax - actual_vx * actual_vx * drag_constant;
    actual_vx += actual_ax * dt;
    actual_x += actual_vx * dt;
  }

  f.close();

  assert(filter.x().rows() == state_dim && filter.x().cols() == 1);
  assert(filter.P().rows() == state_dim && filter.P().cols() == state_dim);

  std::cout << num_i << " iterations" << std::endl;
  std::cout << "estimated x:" << std::endl << filter.x() << std::endl;
  std::cout << "estimated P:" << std::endl << filter.P() << std::endl;
}

int main(int argc, char **argv)
{
  test_cholesky();
  test_generate_sigmas();
  test_unscented_transform();
  test_filter();
  test_1d_drag_filter(false, "ukf_1d_drag_discover_ax");
  test_1d_drag_filter(true, "ukf_1d_drag_control_ax");
  return 0;
}