#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace po = boost::program_options;

namespace
{
  constexpr double kCornerBL = 10.0;
  constexpr double kCornerBR = 20.0;
  constexpr double kCornerTR = 30.0;
  constexpr double kCornerTL = 20.0;
  constexpr int kDefaultErrorCheckPeriod = 10000;
  constexpr int kMaxIterations = 1'000'000;

  inline std::size_t idx(int row, int col, int n) noexcept
  {
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
           static_cast<std::size_t>(col);
  }

  void init_grid(double *grid, int n)
  {
    const std::size_t count =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    std::memset(grid, 0, count * sizeof(double));

    const int nm1 = n - 1;
    grid[idx(0, 0, n)] = kCornerBL;
    grid[idx(0, nm1, n)] = kCornerBR;
    grid[idx(nm1, nm1, n)] = kCornerTR;
    grid[idx(nm1, 0, n)] = kCornerTL;

    if (n > 2)
    {
      const double inv = 1.0 / static_cast<double>(nm1);
      for (int i = 1; i < nm1; ++i)
      {
        const double t = static_cast<double>(i) * inv;
        grid[idx(0, i, n)] = kCornerBL + (kCornerBR - kCornerBL) * t;
        grid[idx(nm1, i, n)] = kCornerTL + (kCornerTR - kCornerTL) * t;
      }
      for (int j = 1; j < nm1; ++j)
      {
        const double t = static_cast<double>(j) * inv;
        grid[idx(j, 0, n)] = kCornerBL + (kCornerTL - kCornerBL) * t;
        grid[idx(j, nm1, n)] = kCornerBR + (kCornerTR - kCornerBR) * t;
      }
    }
  }

  void print_grid(const double *grid, int n)
  {
    for (int row = 0; row < n; ++row)
    {
      for (int col = 0; col < n; ++col)
      {
        std::printf("%10.6f", grid[idx(row, col, n)]);
        if (col + 1 < n)
        {
          std::printf(" ");
        }
      }
      std::printf("\n");
    }
  }

  /** Many Jacobi steps in one GPU launch (cuts kernel launch overhead). */
  void jacobi_batch_noerr(double *u, double *v, int n, int nn, int steps,
                          bool use_tiled)
  {
    if (steps <= 0)
    {
      return;
    }

    if (use_tiled)
    {
#pragma acc parallel async(1) present(u[0:nn], v[0:nn])
      {
        double *cur = u;
        double *nxt = v;
        for (int s = 0; s < steps; ++s)
        {
#pragma acc loop tile(32, 32)
          for (int row = 1; row < n - 1; ++row)
          {
            for (int col = 1; col < n - 1; ++col)
            {
              const std::size_t id = idx(row, col, n);
              nxt[id] = 0.25 * (cur[idx(row, col + 1, n)] +
                                cur[idx(row, col - 1, n)] +
                                cur[idx(row - 1, col, n)] +
                                cur[idx(row + 1, col, n)]);
            }
          }
          double *const tmp = cur;
          cur = nxt;
          nxt = tmp;
        }
      }
    }
    else
    {
#pragma acc parallel async(1) present(u[0:nn], v[0:nn])
      {
        double *cur = u;
        double *nxt = v;
        for (int s = 0; s < steps; ++s)
        {
#pragma acc loop collapse(2)
          for (int row = 1; row < n - 1; ++row)
          {
            for (int col = 1; col < n - 1; ++col)
            {
              const std::size_t id = idx(row, col, n);
              nxt[id] = 0.25 * (cur[idx(row, col + 1, n)] +
                                cur[idx(row, col - 1, n)] +
                                cur[idx(row - 1, col, n)] +
                                cur[idx(row + 1, col, n)]);
            }
          }
          double *const tmp = cur;
          cur = nxt;
          nxt = tmp;
        }
      }
    }
  }

  double jacobi_step_err(double *cur, double *next, int n, int nn, bool use_tiled)
  {
    double error = 0.0;

    if (use_tiled)
    {
#pragma acc parallel loop tile(32, 32) present(cur[0:nn], next[0:nn]) \
    reduction(max : error) async(1)
      for (int row = 1; row < n - 1; ++row)
      {
        for (int col = 1; col < n - 1; ++col)
        {
          const std::size_t id = idx(row, col, n);
          const double new_val = 0.25 * (cur[idx(row, col + 1, n)] +
                                         cur[idx(row, col - 1, n)] +
                                         cur[idx(row - 1, col, n)] +
                                         cur[idx(row + 1, col, n)]);
          next[id] = new_val;
          error = std::fmax(error, std::fabs(new_val - cur[id]));
        }
      }
    }
    else
    {
#pragma acc parallel loop collapse(2) present(cur[0:nn], next[0:nn]) \
    reduction(max : error) async(1)
      for (int row = 1; row < n - 1; ++row)
      {
        for (int col = 1; col < n - 1; ++col)
        {
          const std::size_t id = idx(row, col, n);
          const double new_val = 0.25 * (cur[idx(row, col + 1, n)] +
                                         cur[idx(row, col - 1, n)] +
                                         cur[idx(row - 1, col, n)] +
                                         cur[idx(row + 1, col, n)]);
          next[id] = new_val;
          error = std::fmax(error, std::fabs(new_val - cur[id]));
        }
      }
    }

    return error;
  }

} // namespace

int main(int argc, char **argv)
{
  try
  {
    int grid_size = 128;
    double eps = 1e-6;
    int max_iters = kMaxIterations;
    bool optimized = false;
    bool tiled = false;
    bool print_grid_flag = false;
    bool quiet = false;
    int check_interval = kDefaultErrorCheckPeriod;

    po::options_description desc("2D heat equation (five-point Jacobi), OpenACC");
    desc.add_options()("help,h", "print help")(
        "size,s", po::value<int>(&grid_size)->default_value(128),
        "grid dimension N (N x N)")(
        "eps,e", po::value<double>(&eps)->default_value(1e-6),
        "convergence tolerance")(
        "max-iters,m", po::value<int>(&max_iters)->default_value(kMaxIterations),
        "maximum iterations")(
        "optimized,o", po::bool_switch(&optimized),
        "check convergence every 100 iterations (default: 10000)")(
        "tiled,t", po::bool_switch(&tiled),
        "use tile(32,32) kernel (can be slower on some GPUs)")(
        "check-interval,c", po::value<int>(&check_interval),
        "host sync / error check every N iterations (default 10000)")(
        "print-grid,p", po::bool_switch(&print_grid_flag),
        "print full grid (auto for N=10 or N=13)")(
        "quiet,q", po::bool_switch(&quiet),
        "compact output: time_sec iter error (like laplace2d)");

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

    if (optimized && !vm.count("check-interval"))
    {
      check_interval = 100;
    }

    if (check_interval < 1)
    {
      std::cerr << "check-interval must be >= 1\n";
      return 1;
    }

    if (grid_size < 3)
    {
      std::cerr << "Grid size must be >= 3\n";
      return 1;
    }

    if (max_iters < 1)
    {
      std::cerr << "max-iters must be >= 1\n";
      return 1;
    }
    if (max_iters > kMaxIterations)
    {
      std::cerr << "max-iters capped at " << kMaxIterations << '\n';
      max_iters = kMaxIterations;
    }

    const int n = grid_size;
    const int nn = n * n;
    const std::size_t count = static_cast<std::size_t>(nn);

    std::vector<double> u(count);
    std::vector<double> u_new(count);
    init_grid(u.data(), n);
    init_grid(u_new.data(), n);

    double *cur = u.data();
    double *next = u_new.data();

    double error = 1.0;
    int iter = 0;

    const auto t0 = std::chrono::steady_clock::now();

#pragma acc enter data copyin(cur[0:count], next[0:count])

    while (iter < max_iters && error > eps)
    {
      const int next_check =
          ((iter / check_interval) + 1) * check_interval;
      const int chunk_end = std::min(next_check, max_iters);
      const int chunk = chunk_end - iter;
      const int noerr_steps = chunk - 1;

      if (noerr_steps > 0)
      {
        jacobi_batch_noerr(cur, next, n, nn, noerr_steps, tiled);
        if (noerr_steps % 2 == 1)
        {
          std::swap(cur, next);
        }
        iter += noerr_steps;
      }

      if (iter >= max_iters)
      {
        break;
      }

#pragma acc wait(1)
      error = jacobi_step_err(cur, next, n, nn, tiled);
#pragma acc wait(1)
      std::swap(cur, next);
      ++iter;

      if (error <= eps)
      {
        break;
      }
    }

    const bool sync_host =
        print_grid_flag || grid_size == 10 || grid_size == 13;
    if (sync_host)
    {
#pragma acc update host(cur[0:count])
    }

#pragma acc exit data delete(cur[0:count], next[0:count])

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double>(t1 - t0).count();

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
      std::cout << "grid=" << n << "x" << n << '\n';
      std::cout << "check_interval=" << check_interval << '\n';
      std::cout << "tiled=" << (tiled ? "on" : "off") << '\n';
      std::cout << "iterations=" << iter << '\n';
      std::cout << "error=" << error << '\n';
      std::cout << "time_sec=" << std::fixed << std::setprecision(6) << elapsed
                << '\n';
    }

    if (sync_host)
    {
      print_grid(cur, n);
    }
  }
  catch (const std::exception &ex)
  {
    std::cerr << "Error: " << ex.what() << '\n';
    return 1;
  }
  return 0;
}
