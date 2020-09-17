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
bool allclose(const DenseBase<DerivedA> & a,
  const DenseBase<DerivedB> & b,
  const typename DerivedA::RealScalar & rtol = NumTraits<typename DerivedA::RealScalar>::dummy_precision(),
  const typename DerivedA::RealScalar & atol = NumTraits<typename DerivedA::RealScalar>::epsilon())
{
  return ((a.derived() - b.derived()).array().abs() <=
    (atol + rtol * b.derived().array().abs())).all();
}

bool test_valid()
{
  std::cout << "\n========= VALID =========\n" << std::endl;

  MatrixXd good_P = MatrixXd::Zero(2, 2);
  MatrixXd bad_P = MatrixXd::Zero(2, 2);
  good_P << 1, 0.1, 0.1, 2;
  bad_P << 0, 1, -2, -3;
  bool ok = ukf::valid_P(good_P) && !ukf::valid_P(bad_P);
  std::cout << (ok ? "Valid OK" : "Valid FAIL") << std::endl;
  return ok;
}

bool test_cholesky()
{
  std::cout << "\n========= CHOLESKY =========\n" << std::endl;

  MatrixXd t(2, 2);
  t << 2, 0.1, 0.1, 3;

  MatrixXd s;
  ukf::cholesky(t, s);

  MatrixXd square = s * s.transpose();

  bool ok = allclose(t, square);
  std::cout << (ok ? "Cholesky OK" : "Cholesky FAIL") << std::endl;
  return ok;
}

void test_generate_sigmas()
{
  std::cout << "\n========= GENERATE SIGMA POINTS =========\n" << std::endl;

  double alpha{0.1}, beta{2.0};
  int kappa{0};

  for (int state_dim = 1; state_dim < 5; ++state_dim) {
    std::cout << state_dim << " dimension(s):" << std::endl;

    auto x = VectorXd::Zero(state_dim, 1);
    auto P = MatrixXd::Identity(state_dim, state_dim);

    // Make sure the sizes are correct
    ukf::UnscentedKalmanFilter filter(state_dim, alpha, beta, kappa);
    filter.set_x(x);
    filter.set_P(P);

    RowVectorXd Wm;
    RowVectorXd Wc;
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

bool test_unscented_transform()
{
  std::cout << "\n========= UNSCENTED TRANSFORM =========\n" << std::endl;

  double alpha{0.3}, beta{2};
  int kappa{0};
  ukf::ResidualFn residual_x = [](const VectorXd & x, const VectorXd & mean) -> VectorXd
    {
      return x - mean;
    };

  for (int state_dim = 1; state_dim < 10; ++state_dim) {
    std::cout << state_dim << " dimension(s):" << std::endl;

    VectorXd x = VectorXd::Zero(state_dim);
    MatrixXd P = MatrixXd::Identity(state_dim, state_dim);
    MatrixXd Q = MatrixXd::Zero(state_dim, state_dim); // Zero process noise, just for this test

    // Make sure the sizes are correct
    ukf::UnscentedKalmanFilter filter(state_dim, alpha, beta, kappa);
    filter.set_x(x);
    filter.set_P(P);
    filter.set_Q(Q);

    RowVectorXd Wm;
    RowVectorXd Wc;
    MatrixXd sigmas;

    // Generate sigma points
    ukf::merwe_sigmas(state_dim, alpha, beta, kappa, x, P, sigmas, Wm, Wc);

    // Assume process model f() == identity
    MatrixXd sigmas_p = sigmas;

    // Compute mean of sigmas_p
    VectorXd x_f = ukf::unscented_mean(sigmas_p, Wm);
    assert(x_f.rows() == x.rows() && x_f.cols() == 1);
    bool ok = allclose(x, x_f);
    std::cout << (ok ? "mean OK" : "mean FAIL") << std::endl;
    if (!ok) { return false; }

    // Compute covariance of sigmas_p
    MatrixXd P_f = ukf::unscented_covariance(residual_x, sigmas_p, Wc, x_f) + Q;
    assert(P_f.rows() == P.rows() && P_f.cols() == P.cols());
    ok = allclose(P, P_f);
    std::cout << (ok ? "covar OK" : "covar FAIL") << std::endl;
    if (!ok) { return false; }
  }

  return true;
}

// Simple Newtonian filter with 2 DoF
bool test_simple_filter(double q, double sd,
  double outlier_distance = std::numeric_limits<double>::max(),
  bool expect_many_outliers = false)
{
  std::cout << "\n========= FILTER =========\n" << std::endl;

  // State: [x, vx, ax, y, vy, zy]T
  int state_dim{6};
  int measurement_dim{2};
  int control_dim{0};

  ukf::UnscentedKalmanFilter filter(state_dim, 0.1, 2.0, 0);

  // State transition function
  filter.set_f_fn([](const double dt, const VectorXd & u, Ref<VectorXd> x)
    {
      // Ignore u
      // ax and ay are discovered

      // vx += ax * dt
      x(1) += x(2) * dt;

      // x += vx * dt
      x(0) += x(1) * dt;

      // vy += ay * dt
      x(4) += x(5) * dt;

      // y += vy * dt
      x(3) += x(4) * dt;
    });

  // Measurement function
  filter.set_h_fn([](const Ref<const VectorXd> & x, Ref<VectorXd> z)
    {
      // x
      z(0) = x(0);

      // y
      z(1) = x(3);
    });

  // Process noise
  filter.set_Q(MatrixXd::Identity(state_dim, state_dim) * q);

  // Outlier distance
  filter.set_outlier_distance(outlier_distance);

  VectorXd z = VectorXd::Zero(measurement_dim);
  MatrixXd R = MatrixXd::Identity(measurement_dim, measurement_dim) * sd * sd;
  VectorXd u = VectorXd::Zero(control_dim);

  // Add some noise
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0, sd);

  int num_i = 100;
  int num_outliers = 0;
  for (int i = 0; i < num_i; ++i) {
    filter.predict(0.1, u);
    z(0) = 0 + distribution(generator);
    z(1) = 0 + distribution(generator);
    if (!filter.update(z, R)) {
      num_outliers++;
    }
    if (!filter.valid()) {
      std::cout << "INVALID iteration " << i << std::endl;
      return false;
    }
  }

  assert(filter.x().rows() == state_dim && filter.x().cols() == 1);
  assert(filter.P().rows() == state_dim && filter.P().cols() == state_dim);

  std::cout << num_i << " iterations" << std::endl;
  std::cout << "estimated x:" << std::endl << filter.x() << std::endl;
  std::cout << "estimated P:" << std::endl << filter.P() << std::endl;

  if (expect_many_outliers) {
    std::cout << "expected > " << num_i / 2 << " outliers, got " << num_outliers;
  } else {
    std::cout << "expected < " << num_i / 2 << " outliers, got " << num_outliers;
  }

  bool many_outliers = num_outliers > num_i / 2;
  bool ok = (expect_many_outliers && many_outliers) || (!expect_many_outliers && !many_outliers);
  if (ok) {
    std::cout << ", OK" << std::endl;
    return true;
  } else {
    std::cout << ", FAILED" << std::endl;
    return false;
  }
}

// Implement a filter with drag using one of 2 strategies:
// 1. Pass thrust_ax (acceleration due to thrust) to the transition function, and compute drag in the
//    transition function. This is non-linear.
// 2. Discover ax with a simple Newtonian transition function. This is linear.
//
// In tests the control strategy produces better results.
// The drag constant is a function of the drag coefficient, surface area and mass, and must be known in advance.
bool test_1d_drag_filter(bool use_control, const std::string & filename)
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
    filter.set_f_fn([drag_constant](const double dt, const VectorXd & u, Ref<VectorXd> x)
      {
        // ax = thrust_ax - vx * vx * drag_constant
        double ax = u(0, 0) - x(1) * x(1) * drag_constant;

        // vx += ax * dt
        x(1) += ax * dt;

        // x += vx * dt
        x(0) += x(1) * dt;
      });
  } else {
    filter.set_f_fn([](const double dt, const VectorXd & u, Ref<VectorXd> x)
      {
        // Ignore u
        // ax is discovered, drag is hidden inside ax

        // vx += ax * dt
        x(1) += x(2) * dt;

        // x += vx * dt
        x(0) += x(1) * dt;
      });
  }

  // Measurement function
  filter.set_h_fn([](const Ref<const VectorXd> & x, Ref<VectorXd> z)
    {
      // x
      z(0) = x(0);
    });

  VectorXd z = VectorXd::Zero(measurement_dim);
  MatrixXd R = MatrixXd::Identity(measurement_dim, measurement_dim) * z_stddev;

  // Thruster is always on and generates a constant force
  double thrust_ax = target_vx * target_vx * drag_constant;
  VectorXd u = VectorXd::Zero(control_dim);
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
    z(0) = actual_x + distribution(generator);

    // Predict and update
    filter.predict(dt, u);
    filter.update(z, R);
    if (!filter.valid()) {
      std::cout << "INVALID iteration " << i << std::endl;
      return false;
    }

    // Write to file
    auto x = filter.x();
    auto K = filter.K();
    double ax = use_control ? 0 : x(2);
    double Ka = use_control ? 0 : K(2, 0);
    f << i << ", "
      << actual_x << ", "
      << actual_vx << ", "
      << actual_ax << ", "
      << z << ", "
      << x(0) << ", "
      << x(1) << ", "
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

  return true;
}

template<typename T>
constexpr T clamp(const T v, const T min, const T max)
{
  return v > max ? max : (v < min ? min : v);
}

template<typename T>
constexpr T clamp(const T v, const T minmax)
{
  return clamp(v, -minmax, minmax);
}

// Simple filter with an angle, which requires custom residual and mean lambdas.
bool test_angle_filter(int iterations, bool update)
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
  filter.set_f_fn([](const double dt, const VectorXd & u, Ref<VectorXd> x)
    {
      // Ignore u
      // vy is discovered

      // y += vy * dt
      x(0) = norm_angle(x(0) + x(1) * dt);
    });

  // Measurement function
  filter.set_h_fn([](const Ref<const VectorXd> & x, Ref<VectorXd> z)
    {
      // y
      z(0) = x(0);
    });

  // Residual x function
  filter.set_r_x_fn([](const Ref<const VectorXd> & x, const VectorXd & mean) -> VectorXd
    {
      VectorXd residual = x - mean;
      residual(0, 0) = norm_angle(residual(0, 0));
      return residual;
    });

  // Residual z function
  filter.set_r_z_fn([](const Ref<const MatrixXd> & z, const VectorXd & mean) -> VectorXd
    {
      VectorXd residual = z - mean;
      residual(0, 0) = norm_angle(residual(0, 0));
      return residual;
    });

  // The x and z mean functions need to compute the mean of angles, which doesn't have a precise meaning.
  // See https://en.wikipedia.org/wiki/Mean_of_circular_quantities for one idea.

  // Unscented mean x function
  filter.set_mean_x_fn([](const MatrixXd & sigma_points, const RowVectorXd & Wm) -> VectorXd
    {
      VectorXd mean = MatrixXd::Zero(sigma_points.rows(), 1);

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
  filter.set_mean_z_fn([](const MatrixXd & sigma_points, const RowVectorXd & Wm) -> VectorXd
    {
      VectorXd mean = VectorXd::Zero(sigma_points.rows());

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

  VectorXd z = VectorXd::Zero(measurement_dim);
  MatrixXd R = MatrixXd::Identity(measurement_dim, measurement_dim) * z_stddev;
  VectorXd u = VectorXd::Zero(control_dim);

  // Write state so we can use matplotlib later
  std::ofstream f;
  f.open("angle_filter.txt");
  f << "t, actual_y, actual_vy, actual.ay, z, x.y, x.vy, x.ay, K.y, K.vy, K.ay" << std::endl;
  constexpr int PLOT_N = 200;

  // Initial state
  double actual_ay = 0.0;
  double actual_vy = 1.0;
  double actual_y = 0.0;

  // Add some noise
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0, z_stddev);

  // Run a simulation
  for (int i = 0; i < iterations; ++i) {
    // Measurement
    z(0) = norm_angle(actual_y + distribution(generator));

    // Predict and update
    filter.predict(dt, u);
    if (update) {
      filter.update(z, R);
    }
    if (!filter.valid()) {
      std::cout << "INVALID iteration " << i << std::endl;
      return false;
    }

    // If we're not calling update() the process noise will just keep adding to P.
    // This is OK for scalar values, but not OK for angles: if the variance gets to ~2 radians
    // the mean calculation will fail. Cap the covariance.
    MatrixXd P = filter.P();
    P(0, 0) = clamp(P(0, 0), 1.5);
    P(0, 1) = clamp(P(0, 1), 2.0);
    P(1, 0) = clamp(P(1, 0), 2.0);
    P(1, 1) = clamp(P(1, 1), 3.0);
    filter.set_P(P);

    // Write to file
    if (i + PLOT_N > iterations) {
      auto x = filter.x();
      if (update) {
        auto K = filter.K();
        f << i << ", "
          << actual_y << ", "
          << actual_vy << ", "
          << actual_ay << ", "
          << z << ", "
          << x(0) << ", "
          << x(1) << ", "
          << 0 << ", "
          << K(0, 0) << ", "
          << K(1, 0) << ", "
          << 0 << std::endl;
      } else {
        f << i << ", "
          << actual_y << ", "
          << actual_vy << ", "
          << actual_ay << ", "
          << z << ", "
          << x(0) << ", "
          << x(1) << ", "
          << 0 << ", "
          << 0 << ", "
          << 0 << ", "
          << 0 << std::endl;
      }
    }

    // Generate new z
    actual_vy += actual_ay * dt;
    actual_y = norm_angle(actual_y + actual_vy * dt);
  }

  assert(filter.x().rows() == state_dim && filter.x().cols() == 1);
  assert(filter.P().rows() == state_dim && filter.P().cols() == state_dim);

  std::cout << iterations << " iterations" << std::endl;
  std::cout << "estimated x:" << std::endl << filter.x() << std::endl;
  std::cout << "estimated P:" << std::endl << filter.P() << std::endl;

  return true;
}

// Fuse 2 sensors, one measures 2 variables, the other measures 1
bool test_fusion(int iterations, double z_data_s, double z_filter_s, bool measure_1, bool measure_2,
  double q)
{
  std::cout << "\n========= FUSION FILTER =========\n" << std::endl;

  // State: [x, vx, ax, y, vy, zy]T
#define x_x x(0)
#define x_vx x(1)
#define x_ax x(2)
#define x_y x(3)
#define x_vy x(4)
#define x_ay x(5)
  int state_dim{6};
  int measure_2_dim{2};
  int measure_1_dim{1};
  int control_dim{0};

  ukf::UnscentedKalmanFilter filter(state_dim, 0.1, 2.0, 0);
  filter.set_Q(MatrixXd::Identity(state_dim, state_dim) * q);

  // State transition function
  auto f_fn = [](const double dt, const VectorXd & u, Ref<VectorXd> x)
    {
      // Ignore u
      // ax and ay are discovered

      // vx += ax * dt
      x_vx += x_ax * dt;

      // x += vx * dt
      x_x += x_vx * dt;

      // vy += ay * dt
      x_vy += x_ay * dt;

      // y += vy * dt
      x_y += x_vy * dt;
    };

  filter.set_f_fn(f_fn);

  // Measure 1 function: measure x
  auto measure_1_fn = [](const Ref<const VectorXd> & x, Ref<VectorXd> z)
    {
      // x
      z(0) = x_x;
    };

  // Measure 2 function: measure x and y
  auto measure_2_fn = [](const Ref<const VectorXd> & x, Ref<VectorXd> z)
    {
      // x
      z(0) = x_x;

      // y
      z(1) = x_y;
    };

  double x_mean = 100;
  double y_mean = 1000;
  double z1_bias = -1;
  double z2_bias = 1;

  VectorXd z1 = VectorXd::Zero(measure_1_dim);
  VectorXd z2 = VectorXd::Zero(measure_2_dim);

  MatrixXd R1 = MatrixXd::Identity(measure_1_dim, measure_1_dim) * z_filter_s * z_filter_s;
  MatrixXd R2 = MatrixXd::Identity(measure_2_dim, measure_2_dim) * z_filter_s * z_filter_s;

  VectorXd u = VectorXd::Zero(control_dim);

  // Add some noise
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0, z_data_s);

  // Write state so we can use matplotlib later
  std::ofstream f_x;
  f_x.open("fusion_filter_x.txt");
  f_x << "t, actual_x, actual_vx, actual.ax, z, x.x, x.vx, x.ax, K.x, K.vx, K.ax" << std::endl;
  std::ofstream f_y;
  f_y.open("fusion_filter_y.txt");
  f_y << "t, actual_y, actual_vy, actual.ay, z, x.y, x.vy, x.ay, K.y, K.vy, K.ay" << std::endl;
  constexpr int PLOT_N = 200;

  // Run a simulation
  std::cout << "Running " << iterations << " iterations " <<
            ", measure_1 " << measure_1 << ", measure_2 " << measure_2 << std::endl;
  for (int i = 0; i < iterations; ++i) {

    // Predict
    filter.predict(0.1, u);

    // Update
    // Alternate z1 and z2
    double x_measured = 0;
    double y_measured = 0;
    if (i % 3) {
      if (measure_1) {
        x_measured = x_mean + z1_bias + distribution(generator);
        z1(0) = x_measured;
        // y wasn't measured
        filter.set_h_fn(measure_1_fn);
        filter.update(z1, R1);
      }
    } else {
      if (measure_2) {
        x_measured = x_mean + z2_bias + distribution(generator);
        y_measured = y_mean + z2_bias + distribution(generator);
        z2(0) = x_measured;
        z2(1) = y_measured;
        filter.set_h_fn(measure_2_fn);
        filter.update(z2, R2);
      }
    }

    if (!filter.valid()) {
      std::cout << "INVALID iteration " << i << std::endl;
      std::cout << filter.x() << std::endl;
      std::cout << filter.P() << std::endl;
      return false;
    }

    // Plot the last n iterations
    if (i + PLOT_N > iterations) {
      // Write to file
      auto x = filter.x();
      // K isn't valid if we didn't run an update step
      // auto K = filter.K();

      f_x << i << ", "
          << x_mean << ", "
          << 0 << ", "
          << 0 << ", "
          << x_measured << ", "
          << x_x << ", "
          << x_vx << ", "
          << x_ax << ", "
          << 0 << ", "
          << 0 << ", "
          << 0 << std::endl;

      f_y << i << ", "
          << y_mean << ", "
          << 0 << ", "
          << 0 << ", "
          << y_measured << ", "
          << x_y << ", "
          << x_vy << ", "
          << x_ay << ", "
          << 0 << ", "
          << 0 << ", "
          << 0 << std::endl;
    }
  }

  assert(filter.x().rows() == state_dim && filter.x().cols() == 1);
  assert(filter.P().rows() == state_dim && filter.P().cols() == state_dim);

  std::cout << "estimated x:" << std::endl << filter.x() << std::endl;
  std::cout << "estimated P:" << std::endl << filter.P() << std::endl;

  return true;
}

int main(int argc, char ** argv)
{
  test_generate_sigmas();

  bool ok = test_valid() &&
    test_cholesky() &&
    test_unscented_transform() &&
    test_simple_filter(0.01, 0.1) &&
    test_simple_filter(0.01, 0.2, 5.0, false) &&
    test_simple_filter(0.01, 0.2, 1.0, true) &&
    test_1d_drag_filter(false, "ukf_1d_drag_discover_ax.txt") &&
    test_1d_drag_filter(true, "ukf_1d_drag_control_ax.txt") &&
    test_angle_filter(1000, true) &&
    test_angle_filter(1000, false) &&
    test_fusion(100, 0.1, 10.0, true, true, 1.0) &&
    test_fusion(100, 0.1, 10.0, true, false, 1.0) &&
    test_fusion(100, 0.1, 10.0, false, true, 1.0) &&
    test_fusion(100, 0.1, 10.0, false, false, 1.0);

  if (ok) {
    std::cout << std::endl << "PASSED ALL TESTS" << std::endl;
  } else {
    std::cout << std::endl << "FAILED A TEST" << std::endl;
  }

  return ok ? 0 : 1;
}