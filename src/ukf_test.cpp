#include <iostream>
#include <fstream>
#include <random>

#include "ukf/ukf.hpp"

using namespace Eigen;

// Move an angle to the region [-M_PI, M_PI)
double norm_angle(double a)
{
  if (a < -M_PI || a > M_PI) {
    // Force to [-2PI, 2PI)
    a = fmod(a, 2 * M_PI);

    // Move to [-PI, PI)
    if (a < -M_PI) {
      a += 2 * M_PI;
    } else if (a > M_PI) {
      a -= 2 * M_PI;
    }
  }

  return a;
}

template<typename DerivedA, typename DerivedB>
bool allclose(const DenseBase<DerivedA> &a,
              const DenseBase<DerivedB> &b,
              const typename DerivedA::RealScalar &rtol = NumTraits<typename DerivedA::RealScalar>::dummy_precision(),
              const typename DerivedA::RealScalar &atol = NumTraits<typename DerivedA::RealScalar>::epsilon())
{
  return ((a.derived() - b.derived()).array().abs() <= (atol + rtol * b.derived().array().abs())).all();
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

  double alpha{0.1}, beta{2.0};
  int kappa{0};

  for (int state_dim = 1; state_dim < 5; ++state_dim) {
    std::cout << state_dim << " dimension(s):" << std::endl;

    MatrixXd x = MatrixXd::Zero(state_dim, 1);
    MatrixXd P = MatrixXd::Identity(state_dim, state_dim);

    // Make sure the sizes are correct
    ukf::UnscentedKalmanFilter filter(state_dim, alpha, beta, kappa);
    filter.set_x(x);
    filter.set_P(P);

    MatrixXd Wm;
    MatrixXd Wc;
    MatrixXd sigmas;

    ukf::merwe_sigmas(state_dim, alpha, beta, kappa, x, P, sigmas, Wm, Wc);

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
  int kappa{0};
  ukf::ResidualFn residual_x = [](const MatrixXd &x, const MatrixXd &mean) -> MatrixXd
  {
    return x - mean;
  };

  for (int state_dim = 1; state_dim < 10; ++state_dim) {
    std::cout << state_dim << " dimension(s):" << std::endl;

    MatrixXd x = MatrixXd::Zero(state_dim, 1);
    MatrixXd P = MatrixXd::Identity(state_dim, state_dim);
    MatrixXd Q = MatrixXd::Zero(state_dim, state_dim); // Zero process noise, just for this test

    // Make sure the sizes are correct
    ukf::UnscentedKalmanFilter filter(state_dim, alpha, beta, kappa);
    filter.set_x(x);
    filter.set_P(P);
    filter.set_Q(Q);

    MatrixXd Wm;
    MatrixXd Wc;
    MatrixXd sigmas;

    // Generate sigma points
    ukf::merwe_sigmas(state_dim, alpha, beta, kappa, x, P, sigmas, Wm, Wc);

    // Assume process model f() == identity
    MatrixXd sigmas_p = sigmas;

    // Compute mean of sigmas_p
    MatrixXd x_f = ukf::unscented_mean(sigmas_p, Wm);
    assert(x_f.rows() == x.rows() && x_f.cols() == 1);
    std::cout << (allclose(x, x_f) ? "mean OK" : "mean FAIL") << std::endl;

    // Compute covariance of sigmas_p
    MatrixXd P_f = ukf::unscented_covariance(residual_x, sigmas_p, Wc, x_f, Q);
    assert(P_f.rows() == P.rows() && P_f.cols() == P.cols());
    std::cout << (allclose(P, P_f) ? "covar OK" : "covar FAIL") << std::endl;
  }
}

// Simple Newtonian filter with 2 DoF
void test_simple_filter()
{
  std::cout << "\n========= FILTER =========\n" << std::endl;

  // State: [x, vx, ax, y, vy, zy]T
  int state_dim{6};
  int measurement_dim{2};
  int control_dim{0};

  ukf::UnscentedKalmanFilter filter(state_dim, 0.1, 2.0, 0);

  // State transition function
  filter.set_f_fn([](const double dt, const MatrixXd &u, Ref<MatrixXd> x)
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
  filter.set_h_fn([](const Ref<const MatrixXd> &x, Ref<MatrixXd> z)
                  {
                    // x
                    z(0, 0) = x(0, 0);

                    // y
                    z(1, 0) = x(3, 0);
                  });

  MatrixXd z = MatrixXd::Zero(measurement_dim, 1);
  MatrixXd R = MatrixXd::Identity(measurement_dim, measurement_dim);
  MatrixXd u = MatrixXd::Zero(control_dim, 1);

  int num_i = 100;
  for (int i = 0; i < num_i; ++i) {
    if (!filter.predict(0.1, u) || !filter.update(z, R)) {
      std::cout << "INVALID" << std::endl;
      return;
    }
  }

  assert(filter.x().rows() == state_dim && filter.x().cols() == 1);
  assert(filter.P().rows() == state_dim && filter.P().cols() == state_dim);

  std::cout << num_i << " iterations" << std::endl;
  std::cout << "estimated x:" << std::endl << filter.x() << std::endl;
  std::cout << "estimated P:" << std::endl << filter.P() << std::endl;
}

// Implement a filter with drag using one of 2 strategies:
// 1. Pass thrust_ax (acceleration due to thrust) to the transition function, and compute drag in the
//    transition function. This is non-linear.
// 2. Discover ax with a simple Newtonian transition function. This is linear.
//
// In tests the control strategy produces better results.
// The drag constant is a function of the drag coefficient, surface area and mass, and must be known in advance.
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
  double z_stddev = 1.0;

  ukf::UnscentedKalmanFilter filter(state_dim, 0.3, 2.0, 0);
  filter.set_Q(MatrixXd::Identity(state_dim, state_dim) * 0.05);

  // State transition function
  if (use_control) {
    filter.set_f_fn([drag_constant](const double dt, const MatrixXd &u, Ref<MatrixXd> x)
                    {
                      // ax = thrust_ax - vx * vx * drag_constant
                      double ax = u(0, 0) - x(1, 0) * x(1, 0) * drag_constant;

                      // vx += ax * dt
                      x(1, 0) += ax * dt;

                      // x += vx * dt
                      x(0, 0) += x(1, 0) * dt;
                    });
  } else {
    filter.set_f_fn([](const double dt, const MatrixXd &u, Ref<MatrixXd> x)
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
  filter.set_h_fn([](const Ref<const MatrixXd> &x, Ref<MatrixXd> z)
                  {
                    // x
                    z(0, 0) = x(0, 0);
                  });

  MatrixXd z = MatrixXd::Zero(measurement_dim, 1);
  MatrixXd R = MatrixXd::Identity(measurement_dim, measurement_dim) * z_stddev;

  // Thruster is always on and generates a constant force
  double thrust_ax = target_vx * target_vx * drag_constant;
  MatrixXd u = MatrixXd::Zero(control_dim, 1);
  if (use_control) {
    u(0, 0) = thrust_ax;
  }

  // Write state so we can use matplotlib later
  std::ofstream f;
  f.open(filename);
  f << "t, actual_x, actual_vx, actual.ax, z, x.x, x.vx, x.ax, K.x, K.vx, K.ax" << std::endl;

  // Initial state
  double actual_ax = thrust_ax;
  double actual_vx = 0.0;
  double actual_x = 0.0;

  // Add some noise
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0, z_stddev);

  // Run a simulation
  int num_i = 40;
  for (int i = 0; i < num_i; ++i) {
    // Measurement
    z(0, 0) = actual_x + distribution(generator);

    // Predict and update
    if (!filter.predict(dt, u) || !filter.update(z, R)) {
      std::cout << "INVALID" << std::endl;
      return;
    }

    // Write to file
    auto x = filter.x();
    auto K = filter.K();
    double ax = use_control ? 0 : x(2, 0);
    double Ka = use_control ? 0 : K(2, 0);
    f << i << ", "
      << actual_x << ", "
      << actual_vx << ", "
      << actual_ax << ", "
      << z << ", "
      << x(0, 0) << ", "
      << x(1, 0) << ", "
      << ax << ", "
      << K(0, 0) << ", "
      << K(1, 0) << ", "
      << Ka << std::endl;

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

// Simple filter with an angle, which requires custom residual and mean lambdas.
void test_angle_filter()
{
  std::cout << "\n========= ANGLE FILTER =========\n" << std::endl;

  int state_dim{2};           // [y, vy]T
  int measurement_dim{1};     // [y]T
  int control_dim{0};         // Not used

  // Inputs
  double dt = 0.1;
  double z_stddev = 0.5;

  ukf::UnscentedKalmanFilter filter(state_dim, 0.3, 2.0, 0);
  filter.set_Q(MatrixXd::Identity(state_dim, state_dim) * 0.01);

  // State transition function
  filter.set_f_fn([](const double dt, const MatrixXd &u, Ref<MatrixXd> x)
                  {
                    // Ignore u
                    // vy is discovered

                    // y += vy * dt
                    x(0, 0) = norm_angle(x(0, 0) + x(1, 0) * dt);
                  });

  // Measurement function
  filter.set_h_fn([](const Ref<const MatrixXd> &x, Ref<MatrixXd> z)
                  {
                    // y
                    z(0, 0) = x(0, 0);
                  });

  // Residual x function
  filter.set_r_x_fn([](const Ref<const MatrixXd> &x, const MatrixXd &mean) -> MatrixXd
                    {
                      MatrixXd residual = x - mean;
                      residual(0, 0) = norm_angle(residual(0, 0));
                      return residual;
                    });

  // Residual z function
  filter.set_r_z_fn([](const Ref<const MatrixXd> &z, const MatrixXd &mean) -> MatrixXd
                    {
                      MatrixXd residual = z - mean;
                      residual(0, 0) = norm_angle(residual(0, 0));
                      return residual;
                    });

  // The x and z mean functions need to compute the mean of angles, which doesn't have a precise meaning.
  // See https://en.wikipedia.org/wiki/Mean_of_circular_quantities for one idea.

  // Unscented mean x function
  filter.set_mean_x_fn([](const MatrixXd &sigma_points, const MatrixXd &Wm) -> MatrixXd
                       {
                         MatrixXd mean = MatrixXd::Zero(sigma_points.rows(), 1);

                         assert(mean.rows() == 2);

                         double sum_y_sin = 0.0;
                         double sum_y_cos = 0.0;

                         for (int i = 0; i < sigma_points.cols(); ++i) {
                           sum_y_sin += Wm(0, i) * sin(sigma_points(0, i));
                           sum_y_cos += Wm(0, i) * cos(sigma_points(0, i));

                           mean(1, 0) += Wm(0, i) * sigma_points(1, i);
                         }

                         mean(0, 0) = atan2(sum_y_sin, sum_y_cos);

                         return mean;
                       });

  // Unscented mean z function
  filter.set_mean_z_fn([](const MatrixXd &sigma_points, const MatrixXd &Wm) -> MatrixXd
                       {
                         MatrixXd mean = MatrixXd::Zero(sigma_points.rows(), 1);

                         assert(mean.rows() == 1);

                         double sum_y_sin = 0.0;
                         double sum_y_cos = 0.0;

                         for (int i = 0; i < sigma_points.cols(); ++i) {
                           sum_y_sin += Wm(0, i) * sin(sigma_points(0, i));
                           sum_y_cos += Wm(0, i) * cos(sigma_points(0, i));
                         }

                         mean(0, 0) = atan2(sum_y_sin, sum_y_cos);

                         return mean;
                       });

  MatrixXd z = MatrixXd::Zero(measurement_dim, 1);
  MatrixXd R = MatrixXd::Identity(measurement_dim, measurement_dim) * z_stddev;
  MatrixXd u = MatrixXd::Zero(control_dim, 1);

  // Write state so we can use matplotlib later
  std::ofstream f;
  f.open("angle_filter.txt");
  f << "t, actual_y, actual_vy, actual.ay, z, x.y, x.vy, x.ay, K.y, K.vy, K.ay" << std::endl;

  // Initial state
  double actual_ay = 0.0;
  double actual_vy = 1.0;
  double actual_y = 0.0;

  // Add some noise
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0, z_stddev);

  // Run a simulation
  int num_i = 100;
  for (int i = 0; i < num_i; ++i) {
    // Measurement
    z(0, 0) = norm_angle(actual_y + distribution(generator));

    // Predict and update
    // Predict and update
    if (!filter.predict(dt, u) || !filter.update(z, R)) {
      std::cout << "INVALID" << std::endl;
      return;
    }

    // Write to file
    auto x = filter.x();
    auto K = filter.K();
    f << i << ", "
      << actual_y << ", "
      << actual_vy << ", "
      << actual_ay << ", "
      << z << ", "
      << x(0, 0) << ", "
      << x(1, 0) << ", "
      << 0 << ", "
      << K(0, 0) << ", "
      << K(1, 0) << ", "
      << 0 << std::endl;

    // Generate new z
    actual_vy += actual_ay * dt;
    actual_y = norm_angle(actual_y + actual_vy * dt);
  }

  assert(filter.x().rows() == state_dim && filter.x().cols() == 1);
  assert(filter.P().rows() == state_dim && filter.P().cols() == state_dim);

  std::cout << num_i << " iterations" << std::endl;
  std::cout << "estimated x:" << std::endl << filter.x() << std::endl;
  std::cout << "estimated P:" << std::endl << filter.P() << std::endl;
}

// Fuse 2 sensors, one measures 2 variables, the other measures 1
void test_fusion()
{
  std::cout << "\n========= FUSION FILTER =========\n" << std::endl;

  // State: [x, vx, ax, y, vy, zy]T
  int state_dim{6};
  int measure_2_dim{2};
  int measure_1_dim{1};
  int control_dim{0};

  ukf::UnscentedKalmanFilter filter(state_dim, 0.1, 2.0, 0);
  filter.set_Q(MatrixXd::Identity(state_dim, state_dim) * 0.01);

  // State transition function
  auto f_fn = [](const double dt, const MatrixXd &u, Ref<MatrixXd> x)
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
  };

  filter.set_f_fn(f_fn);

  // Measure 1 function
  auto measure_1_fn = [](const Ref<const MatrixXd> &x, Ref<MatrixXd> z)
  {
    // x
    z(0, 0) = x(0, 0);
  };

  // Measure 2 function
  auto measure_2_fn = [](const Ref<const MatrixXd> &x, Ref<MatrixXd> z)
  {
    // x
    z(0, 0) = x(0, 0);

    // y
    z(1, 0) = x(3, 0);
  };

  double z1_mean = -1;
  double z2_mean = 1;
  double z_stddev = 0.01;
  double z_var = z_stddev * z_stddev;

  MatrixXd z1 = MatrixXd::Zero(measure_1_dim, 1);
  MatrixXd z2 = MatrixXd::Zero(measure_2_dim, 1);

  MatrixXd R1 = MatrixXd::Identity(measure_1_dim, measure_1_dim) * z_var;
  MatrixXd R2 = MatrixXd::Identity(measure_2_dim, measure_2_dim) * z_var;

  MatrixXd u = MatrixXd::Zero(control_dim, 1);

  // Add some noise
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0, z_stddev);

  // Run a simulation
  int num_i = 10000;
  std::cout << "Running " << num_i << " iterations..." << std::endl;
  for (int i = 0; i < num_i; ++i) {

    // Predict
    bool ok = filter.predict(0.1, u);

    // Update
    if (ok) {
      // Alternate z1 and z2
      if (i % 3) {
        z1(0) = z1_mean + distribution(generator);
        filter.set_h_fn(measure_1_fn);
        ok = filter.update(z1, R1);
      } else {
        z2(0) = z2_mean + distribution(generator);
        z2(1) = z2_mean + distribution(generator);
        filter.set_h_fn(measure_2_fn);
        ok = filter.update(z2, R2);
      }
    }

    if (!ok) {
      std::cout << "INVALID iteration " << i << std::endl;
      std::cout << filter.x() << std::endl;
      std::cout << filter.P() << std::endl;
      return;
    }

    // TODO write out and graph
    // TODO try different dts, so the measurements "beat" against each other
  }

  assert(filter.x().rows() == state_dim && filter.x().cols() == 1);
  assert(filter.P().rows() == state_dim && filter.P().cols() == state_dim);

  std::cout << "estimated x:" << std::endl << filter.x() << std::endl;
  std::cout << "estimated P:" << std::endl << filter.P() << std::endl;
}

int main(int argc, char **argv)
{
  test_cholesky();
  test_generate_sigmas();
  test_unscented_transform();
  test_simple_filter();
  test_1d_drag_filter(false, "ukf_1d_drag_discover_ax.txt");
  test_1d_drag_filter(true, "ukf_1d_drag_control_ax.txt");
  test_angle_filter();
  test_fusion();
  return 0;
}