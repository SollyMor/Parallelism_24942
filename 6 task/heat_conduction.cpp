/**
 * 2D steady heat / Laplace equation — five-point stencil (Jacobi).
 * Corners: 10, 20, 30, 20 (BL, BR, TR, TL). Edges: linear interpolation.
 * OpenACC (-acc=host | -acc=multicore | -acc=gpu).
 */

#include <boost/program_options.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace po = boost::program_options;

namespace
{
  constexpr double kCornerBL = 10.0;
  constexpr double kCornerBR = 20.0;
  constexpr double kCornerTR = 30.0;
  constexpr double kCornerTL = 20.0;

  inline std::size_t idx(int i, int j, int n) noexcept
  {
    return static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
           static_cast<std::size_t>(j);
  }

  void init_grid(double *u, int n)
  {
    const std::size_t cells =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    const int nm1 = n - 1;
    const double inv = (nm1 > 0) ? (1.0 / static_cast<double>(nm1)) : 0.0;

    for (std::size_t k = 0; k < cells; ++k)
    {
      u[k] = 0.0;
    }

    u[idx(0, 0, n)] = kCornerBL;
    u[idx(nm1, 0, n)] = kCornerBR;
    u[idx(nm1, nm1, n)] = kCornerTR;
    u[idx(0, nm1, n)] = kCornerTL;

    for (int i = 0; i < n; ++i)
    {
      const double t = static_cast<double>(i) * inv;
      u[idx(i, 0, n)] = kCornerBL + (kCornerBR - kCornerBL) * t;
      u[idx(i, nm1, n)] = kCornerTL + (kCornerTR - kCornerTL) * t;
    }
    for (int j = 0; j < n; ++j)
    {
      const double t = static_cast<double>(j) * inv;
      u[idx(0, j, n)] = kCornerBL + (kCornerTL - kCornerBL) * t;
      u[idx(nm1, j, n)] = kCornerBR + (kCornerTR - kCornerBR) * t;
    }
  }

  void print_grid(const double *u, int n)
  {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Grid " << n << "x" << n << ":\n";
    for (int i = 0; i < n; ++i)
    {
      for (int j = 0; j < n; ++j)
      {
        std::cout << std::setw(11) << u[idx(i, j, n)];
        if (j + 1 < n)
        {
          std::cout << ' ';
        }
      }
      std::cout << '\n';
    }
    std::cout << std::flush;
  }

  double max_diff_host(const double *a, const double *b, int n)
  {
    double err = 0.0;
    for (int i = 1; i < n - 1; ++i)
    {
      for (int j = 1; j < n - 1; ++j)
      {
        const std::size_t k = idx(i, j, n);
        const double d = std::fabs(b[k] - a[k]);
        if (d > err)
        {
          err = d;
        }
      }
    }
    return err;
  }

  void jacobi_step_baseline(double *cur, double *next, int n, std::size_t sz)
  {
#pragma acc parallel loop collapse(2) present(cur[:sz], next[:sz])
    for (int i = 1; i < n - 1; ++i)
    {
      for (int j = 1; j < n - 1; ++j)
      {
        const std::size_t c = idx(i, j, n);
        next[c] = 0.25 * (cur[idx(i - 1, j, n)] + cur[idx(i + 1, j, n)] +
                          cur[idx(i, j - 1, n)] + cur[idx(i, j + 1, n)]);
      }
    }
  }

  double jacobi_step_optimized(double *cur, double *next, int n, std::size_t sz)
  {
    double err = 0.0;

#pragma acc parallel loop collapse(2) reduction(max : err) present(cur[:sz], next[:sz])
    for (int i = 1; i < n - 1; ++i)
    {
      for (int j = 1; j < n - 1; ++j)
      {
        const std::size_t c = idx(i, j, n);
        const double v =
            0.25 * (cur[idx(i - 1, j, n)] + cur[idx(i + 1, j, n)] +
                    cur[idx(i, j - 1, n)] + cur[idx(i, j + 1, n)]);
        const double d = std::fabs(v - cur[c]);
        next[c] = v;
        err = std::fmax(err, d);
      }
    }
    return err;
  }

  struct SolveResult
  {
    int iterations = 0;
    double error = 0.0;
    double seconds = 0.0;
  };

  SolveResult solve_baseline(std::vector<double> &u,
                             std::vector<double> &u_new,
                             int n,
                             double eps,
                             int max_iters,
                             double *&solution)
  {
    const std::size_t sz = u.size();
    init_grid(u.data(), n);

    double *cur = u.data();
    double *next = u_new.data();
    double err = std::numeric_limits<double>::infinity();
    int iter = 0;
    const auto t0 = std::chrono::steady_clock::now();

    while (iter < max_iters && err > eps)
    {
#pragma acc data copy(cur[:sz], next[:sz])
      {
        jacobi_step_baseline(cur, next, n, sz);
#pragma acc update host(next[:sz])
      }
      err = max_diff_host(cur, next, n);
      std::swap(cur, next);
      ++iter;
    }

    solution = cur;
    const auto t1 = std::chrono::steady_clock::now();
    return {iter, err,
            std::chrono::duration<double>(t1 - t0).count()};
  }

  SolveResult solve_optimized(std::vector<double> &u,
                              std::vector<double> &u_new,
                              int n,
                              double eps,
                              int max_iters,
                              double *&solution)
  {
    const std::size_t sz = u.size();
    init_grid(u.data(), n);

    double *cur = u.data();
    double *next = u_new.data();
    double err = std::numeric_limits<double>::infinity();
    int iter = 0;
    const auto t0 = std::chrono::steady_clock::now();

#pragma acc data copyin(u[:sz], u_new[:sz])
    {
      while (iter < max_iters && err > eps)
      {
        err = jacobi_step_optimized(cur, next, n, sz);
        std::swap(cur, next);
        ++iter;
      }
#pragma acc update host(cur[:sz])
    }

    solution = cur;
    const auto t1 = std::chrono::steady_clock::now();
    return {iter, err,
            std::chrono::duration<double>(t1 - t0).count()};
  }
} // namespace

int main(int argc, char **argv)
{
  try
  {
    int grid_size = 128;
    double eps = 1e-6;
    int max_iters = 1000000;
    bool optimized = false;
    bool print_grid_flag = false;
    bool quiet = false;

    po::options_description desc("2D heat equation (five-point Jacobi), OpenACC");
    desc.add_options()("help,h", "print help")(
        "size,s", po::value<int>(&grid_size)->default_value(128),
        "grid dimension N (N x N)")(
        "eps,e", po::value<double>(&eps)->default_value(1e-6),
        "convergence tolerance")(
        "max-iters,m", po::value<int>(&max_iters)->default_value(1000000),
        "maximum iterations")(
        "optimized,o", po::bool_switch(&optimized),
        "optimized: persistent device data, on-device residual")(
        "print-grid,p", po::bool_switch(&print_grid_flag),
        "print full grid (auto for N=10 or N=13)")(
        "quiet,q", po::bool_switch(&quiet), "only iterations and error");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
      std::cout << desc << '\n';
      return 0;
    }

    if (grid_size < 3)
    {
      std::cerr << "Grid size must be >= 3\n";
      return 1;
    }

    const int n = grid_size;
    const std::size_t sz = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    std::vector<double> u(sz, 0.0);
    std::vector<double> u_new(sz, 0.0);

    double *solution = u.data();
    const SolveResult result =
        optimized ? solve_optimized(u, u_new, n, eps, max_iters, solution)
                  : solve_baseline(u, u_new, n, eps, max_iters, solution);

    if (!quiet)
    {
      std::cout << std::scientific << std::setprecision(6);
      std::cout << "mode=" << (optimized ? "optimized" : "baseline") << '\n';
      std::cout << "grid=" << n << "x" << n << '\n';
      std::cout << "iterations=" << result.iterations << '\n';
      std::cout << "error=" << result.error << '\n';
      std::cout << "time_sec=" << std::fixed << std::setprecision(6)
                << result.seconds << '\n';
    }
    else
    {
      std::cout << result.iterations << ' ' << result.error << '\n';
    }

    if (print_grid_flag || n == 10 || n == 13)
    {
      print_grid(solution, n);
    }
  }
  catch (const std::exception &ex)
  {
    std::cerr << "Error: " << ex.what() << '\n';
    return 1;
  }

  return 0;
}
