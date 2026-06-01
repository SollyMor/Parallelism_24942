#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

namespace po = boost::program_options;

namespace
{
  constexpr double kCornerBL = 10.0;
  constexpr double kCornerBR = 20.0;
  constexpr double kCornerTR = 30.0;
  constexpr double kCornerTL = 20.0;
  constexpr int kMaxIterations = 1'000'000;
  constexpr int kErrorCheckPeriod = 10000;

  inline std::size_t idx(int j, int i, int m) noexcept
  {
    return static_cast<std::size_t>(j) * static_cast<std::size_t>(m) +
           static_cast<std::size_t>(i);
  }

  void set_boundary(double *grid, int m, int n)
  {
    if (m <= 0 || n <= 0)
    {
      return;
    }

    grid[idx(0, 0, m)] = kCornerBL;
    grid[idx(0, m - 1, m)] = kCornerBR;
    grid[idx(n - 1, m - 1, m)] = kCornerTR;
    grid[idx(n - 1, 0, m)] = kCornerTL;

    if (m > 2)
    {
      const double denom = static_cast<double>(m - 1);
      for (int i = 1; i < m - 1; ++i)
      {
        const double t = static_cast<double>(i) / denom;
        grid[idx(0, i, m)] = kCornerBL + (kCornerBR - kCornerBL) * t;
        grid[idx(n - 1, i, m)] = kCornerTL + (kCornerTR - kCornerTL) * t;
      }
    }

    if (n > 2)
    {
      const double denom = static_cast<double>(n - 1);
      for (int j = 1; j < n - 1; ++j)
      {
        const double t = static_cast<double>(j) / denom;
        grid[idx(j, 0, m)] = kCornerBL + (kCornerTL - kCornerBL) * t;
        grid[idx(j, m - 1, m)] = kCornerBR + (kCornerTR - kCornerBR) * t;
      }
    }
  }

  void initialize(double *a, double *anew, int m, int n)
  {
    const std::size_t count =
        static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
    std::memset(a, 0, count * sizeof(double));
    std::memset(anew, 0, count * sizeof(double));
    set_boundary(a, m, n);
    set_boundary(anew, m, n);
  }

  double jacobi_step(double *a, double *anew, int m, int n)
  {
    double error = 0.0;
    const int nn = m * n;

#pragma acc parallel loop collapse(2) present(a[0:nn], anew[0:nn]) \
    reduction(max : error) async(1)
    for (int j = 1; j < n - 1; ++j)
    {
      for (int i = 1; i < m - 1; ++i)
      {
        const std::size_t id = idx(j, i, m);
        const double new_val = 0.25 * (a[idx(j, i + 1, m)] + a[idx(j, i - 1, m)] +
                                       a[idx(j - 1, i, m)] + a[idx(j + 1, i, m)]);
        anew[id] = new_val;
        error = std::fmax(error, std::fabs(new_val - a[id]));
      }
    }
#pragma acc wait(1)

    return error;
  }

  void jacobi_step_no_error(double *a, double *anew, int m, int n)
  {
    const int nn = m * n;

#pragma acc parallel loop collapse(2) present(a[0:nn], anew[0:nn]) async(1)
    for (int j = 1; j < n - 1; ++j)
    {
      for (int i = 1; i < m - 1; ++i)
      {
        const std::size_t id = idx(j, i, m);
        anew[id] = 0.25 * (a[idx(j, i + 1, m)] + a[idx(j, i - 1, m)] +
                           a[idx(j - 1, i, m)] + a[idx(j + 1, i, m)]);
      }
    }
  }

  void print_grid(const double *a, int m, int n)
  {
    for (int j = 0; j < n; ++j)
    {
      for (int i = 0; i < m; ++i)
      {
        std::printf("%10.6f", a[idx(j, i, m)]);
        if (i + 1 < m)
        {
          std::printf(" ");
        }
      }
      std::printf("\n");
    }
  }

} // namespace

int main(int argc, char **argv)
{
  try
  {
    int size = 128;
    double tol = 1.0e-6;
    int max_iter = kMaxIterations;
    bool quiet = false;
    int error_check_period = kErrorCheckPeriod;

    po::options_description desc("2D heat equation (five-point Jacobi), OpenACC");
    desc.add_options()("help,h", "print help")(
        "size,s", po::value<int>(&size)->default_value(128), "grid size NxN")(
        "eps,e", po::value<double>(&tol)->default_value(1.0e-6),
        "convergence tolerance")(
        "tol,t", po::value<double>(&tol), "alias for --eps")(
        "max-iters,m", po::value<int>(&max_iter)->default_value(kMaxIterations),
        "maximum iterations")(
        "max-iter,i", po::value<int>(&max_iter), "alias for --max-iters")(
        "check-interval,c", po::value<int>(&error_check_period),
        "error check period (default 10000, same as laplace2d)")(
        "quiet,q", po::bool_switch(&quiet),
        "output: time_sec iter error")(
        "print-grid,p", po::bool_switch(), "print grid (also for N=10,13)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
      std::cout << desc << '\n';
      return 0;
    }

    if (size < 3)
    {
      std::cerr << "Grid size must be at least 3\n";
      return 1;
    }

    if (max_iter < 1)
    {
      std::cerr << "max-iters must be at least 1\n";
      return 1;
    }
    if (max_iter > kMaxIterations)
    {
      std::cerr << "max-iters capped at " << kMaxIterations << '\n';
      max_iter = kMaxIterations;
    }

    if (!vm.count("check-interval"))
    {
      error_check_period = kErrorCheckPeriod;
    }
    if (error_check_period < 1)
    {
      std::cerr << "check-interval must be at least 1\n";
      return 1;
    }

    const int m = size;
    const int n = size;
    const std::size_t count =
        static_cast<std::size_t>(m) * static_cast<std::size_t>(n);

    double *a = new double[count];
    double *anew = new double[count];
    initialize(a, anew, m, n);

    double error = 1.0;
    int iter = 0;

    const auto t0 = std::chrono::steady_clock::now();

#pragma acc enter data copyin(a[0:count], anew[0:count])

    for (iter = 0; iter < max_iter; ++iter)
    {
      const bool check_error = ((iter + 1) % error_check_period == 0) ||
                               (iter + 1 == max_iter);
      if (check_error)
      {
        error = jacobi_step(a, anew, m, n);
      }
      else
      {
        jacobi_step_no_error(a, anew, m, n);
      }
      std::swap(a, anew);
      if (check_error && error <= tol)
      {
        ++iter;
        break;
      }
    }

#pragma acc wait(1)
#pragma acc exit data delete(a[0:count], anew[0:count])

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double>(t1 - t0).count();

    const bool show_grid =
        vm.count("print-grid") || size == 10 || size == 13;

    if (quiet)
    {
      std::cout << std::fixed << std::setprecision(6) << elapsed << ' ' << iter
                << ' ' << std::scientific << error << '\n';
    }
    else
    {
      std::cout << std::scientific << std::setprecision(6);
#ifdef ACC_MODE_STR
      std::cout << "acc_mode=" << ACC_MODE_STR << '\n';
#endif
      std::cout << "grid=" << m << "x" << n << '\n';
      std::cout << "check_interval=" << error_check_period << '\n';
      std::cout << "iterations=" << iter << '\n';
      std::cout << "error=" << error << '\n';
      std::cout << "time_sec=" << std::fixed << std::setprecision(6) << elapsed
                << '\n';
    }

    if (show_grid)
    {
      print_grid(a, m, n);
    }

    delete[] a;
    delete[] anew;
  }
  catch (const std::exception &ex)
  {
    std::cerr << "Error: " << ex.what() << '\n';
    return 1;
  }
  return 0;
}
