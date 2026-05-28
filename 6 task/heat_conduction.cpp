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

  /** Jacobi + max residual in one pass; arrays already on device (present). */
  double jacobi_step(double *cur, double *next, int n, std::size_t sz)
  {
    double err = 0.0;
    const std::size_t sn = static_cast<std::size_t>(n);

#pragma acc parallel loop collapse(2) reduction(max : err) present(cur[:sz], next[:sz])
    for (int i = 1; i < n - 1; ++i)
    {
      for (int j = 1; j < n - 1; ++j)
      {
        const std::size_t c =
            static_cast<std::size_t>(i) * sn + static_cast<std::size_t>(j);
        const double v =
            0.25 * (cur[c - sn] + cur[c + sn] + cur[c - 1] + cur[c + 1]);
        const double d = std::fabs(v - cur[c]);
        next[c] = v;
        err = std::fmax(err, d);
      }
    }
    return err;
  }

  /** Tiled variant for GPU cache locality (no collapse — conflicts with tile). */
  double jacobi_step_tiled(double *cur, double *next, int n, std::size_t sz)
  {
    double err = 0.0;
    const std::size_t sn = static_cast<std::size_t>(n);

#pragma acc parallel loop tile(32, 32) reduction(max : err) present(cur[:sz], next[:sz])
    for (int i = 1; i < n - 1; ++i)
    {
      for (int j = 1; j < n - 1; ++j)
      {
        const std::size_t c =
            static_cast<std::size_t>(i) * sn + static_cast<std::size_t>(j);
        const double v =
            0.25 * (cur[c - sn] + cur[c + sn] + cur[c - 1] + cur[c + 1]);
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

  /**
   * copyin both grids once (fixed BC on device for u and u_new).
   * Jacobi only updates interiors; edges must be valid before each read.
   * No copyout on exit unless sync_host (benchmark path = zero grid transfers).
   * Fixed u/u_new pointers in kernels — do not swap cur/next (breaks present).
   */
  SolveResult solve_acc(std::vector<double> &u,
                        std::vector<double> &u_new,
                        int n,
                        double eps,
                        int max_iters,
                        double *&solution,
                        bool use_tiled,
                        bool sync_host)
  {
    const std::size_t sz = u.size();
    init_grid(u.data(), n);
    init_grid(u_new.data(), n);

    double err = std::numeric_limits<double>::infinity();
    int iter = 0;
    bool write_to_new = true;

    SolveResult result;

#pragma acc data copyin(u[:sz], u_new[:sz])
    {
      const auto t0 = std::chrono::steady_clock::now();

      while (iter < max_iters && err > eps)
      {
        if (write_to_new)
        {
          err = use_tiled ? jacobi_step_tiled(u.data(), u_new.data(), n, sz)
                          : jacobi_step(u.data(), u_new.data(), n, sz);
        }
        else
        {
          err = use_tiled ? jacobi_step_tiled(u_new.data(), u.data(), n, sz)
                          : jacobi_step(u_new.data(), u.data(), n, sz);
        }
        write_to_new = !write_to_new;
        ++iter;
      }

      const auto t1 = std::chrono::steady_clock::now();
      result.seconds = std::chrono::duration<double>(t1 - t0).count();
      result.iterations = iter;
      result.error = err;
      solution = write_to_new ? u.data() : u_new.data();

      if (sync_host)
      {
#pragma acc update host(solution[:sz])
      }
    }

    return result;
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
        "optimized: copyin/create, tiled Jacobi, host sync only if grid printed")(
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

#ifdef FORCE_OPTIMIZED
    optimized = true;
#endif

    if (grid_size < 3)
    {
      std::cerr << "Grid size must be >= 3\n";
      return 1;
    }

    const int n = grid_size;
    const std::size_t sz =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    std::vector<double> u(sz, 0.0);
    std::vector<double> u_new(sz, 0.0);

    const bool sync_host = print_grid_flag || n == 10 || n == 13;
    double *solution = u.data();

    const SolveResult result =
        solve_acc(u, u_new, n, eps, max_iters, solution, optimized, sync_host);

    if (!quiet)
    {
      std::cout << std::scientific << std::setprecision(6);
      std::cout << "mode=" << (optimized ? "optimized" : "baseline") << '\n';
#ifdef ACC_MODE_STR
      std::cout << "acc_mode=" << ACC_MODE_STR << '\n';
#else
      std::cout << "acc_mode=unknown\n";
#endif
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

    if (sync_host)
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
