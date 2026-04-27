#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <omp.h>

struct SolverResult
{
  double elapsed_sec = 0.0;
  int iterations = 0;
  double diff_norm = 0.0;
  double error_norm = 0.0;
};

enum class ScheduleKind
{
  Static,
  Dynamic,
  Guided
};

static std::string to_string_schedule(ScheduleKind kind)
{
  switch (kind)
  {
  case ScheduleKind::Static:
    return "static";
  case ScheduleKind::Dynamic:
    return "dynamic";
  case ScheduleKind::Guided:
    return "guided";
  }
  return "unknown";
}

static omp_sched_t to_omp_schedule(ScheduleKind kind)
{
  switch (kind)
  {
  case ScheduleKind::Static:
    return omp_sched_static;
  case ScheduleKind::Dynamic:
    return omp_sched_dynamic;
  case ScheduleKind::Guided:
    return omp_sched_guided;
  }
  return omp_sched_static;
}

static SolverResult solve_variant1_parallel_for(
    int n, int max_iters, double eps, int threads, double tau_factor)
{
  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);
  std::vector<double> b(n, static_cast<double>(n + 1));
  const double tau = tau_factor / static_cast<double>(n + 1);

  omp_set_num_threads(threads);

  double diff = std::numeric_limits<double>::infinity();
  int it = 0;
  const double t0 = omp_get_wtime();

  while (it < max_iters && diff > eps)
  {
    double sum_x = 0.0;
#pragma omp parallel for reduction(+ : sum_x)
    for (int i = 0; i < n; ++i)
    {
      sum_x += x[i];
    }

#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
      const double ax_i = sum_x + x[i];
      x_new[i] = x[i] - tau * (ax_i - b[i]);
    }

    diff = 0.0;
#pragma omp parallel for reduction(+ : diff)
    for (int i = 0; i < n; ++i)
    {
      const double d = x_new[i] - x[i];
      diff += d * d;
    }
    diff = std::sqrt(diff);

#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
      x[i] = x_new[i];
    }

    ++it;
  }

  const double elapsed = omp_get_wtime() - t0;

  double err = 0.0;
#pragma omp parallel for reduction(+ : err)
  for (int i = 0; i < n; ++i)
  {
    const double d = x[i] - 1.0;
    err += d * d;
  }
  err = std::sqrt(err);

  return SolverResult{elapsed, it, diff, err};
}

static SolverResult solve_variant2_single_parallel(
    int n, int max_iters, double eps, int threads, double tau_factor)
{
  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);
  std::vector<double> b(n, static_cast<double>(n + 1));
  const double tau = tau_factor / static_cast<double>(n + 1);

  omp_set_num_threads(threads);

  double diff = std::numeric_limits<double>::infinity();
  int it = 0;
  const double t0 = omp_get_wtime();

  double sum_x = 0.0;
  double diff_local = 0.0;
  bool stop = false;
#pragma omp parallel shared(x, x_new, b, diff, it, sum_x, diff_local, stop)
  {
    while (true)
    {
#pragma omp single
      {
        stop = false;
        if (it >= max_iters || diff <= eps)
        {
          stop = true;
        }
      }
#pragma omp barrier
      if (stop)
      {
        break;
      }

#pragma omp single
      {
        sum_x = 0.0;
      }
#pragma omp barrier
      double sum_part = 0.0;
#pragma omp for nowait
      for (int i = 0; i < n; ++i)
      {
        sum_part += x[i];
      }
#pragma omp atomic update
      sum_x += sum_part;
#pragma omp barrier

#pragma omp for
      for (int i = 0; i < n; ++i)
      {
        const double ax_i = sum_x + x[i];
        x_new[i] = x[i] - tau * (ax_i - b[i]);
      }

#pragma omp single
      {
        diff_local = 0.0;
      }
#pragma omp barrier
      double diff_part = 0.0;
#pragma omp for nowait
      for (int i = 0; i < n; ++i)
      {
        const double d = x_new[i] - x[i];
        diff_part += d * d;
      }
#pragma omp atomic update
      diff_local += diff_part;
#pragma omp barrier

#pragma omp single
      {
        diff = std::sqrt(diff_local);
        ++it;
      }

#pragma omp for
      for (int i = 0; i < n; ++i)
      {
        x[i] = x_new[i];
      }
#pragma omp barrier
    }
  }

  const double elapsed = omp_get_wtime() - t0;

  double err = 0.0;
#pragma omp parallel for reduction(+ : err)
  for (int i = 0; i < n; ++i)
  {
    const double d = x[i] - 1.0;
    err += d * d;
  }
  err = std::sqrt(err);

  return SolverResult{elapsed, it, diff, err};
}

static SolverResult solve_variant2_runtime_schedule(
    int n, int max_iters, double eps, int threads, double tau_factor, ScheduleKind sched, int chunk)
{
  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);
  std::vector<double> b(n, static_cast<double>(n + 1));
  const double tau = tau_factor / static_cast<double>(n + 1);

  omp_set_num_threads(threads);
  omp_set_schedule(to_omp_schedule(sched), chunk);

  double diff = std::numeric_limits<double>::infinity();
  int it = 0;
  const double t0 = omp_get_wtime();

  double sum_x = 0.0;
  double diff_local = 0.0;
  bool stop = false;
#pragma omp parallel shared(x, x_new, b, diff, it, sum_x, diff_local, stop)
  {
    while (true) // переменную внутрь!!!
    {
#pragma omp single
      {
        stop = false;
        if (it >= max_iters || diff <= eps)
        {
          stop = true;
        }
      }
#pragma omp barrier
      if (stop)
      {
        break;
      }

#pragma omp single
      {
        sum_x = 0.0;
      }
#pragma omp barrier
      double sum_part = 0.0;
#pragma omp for schedule(runtime) nowait
      for (int i = 0; i < n; ++i)
      {
        sum_part += x[i];
      }
#pragma omp atomic update
      sum_x += sum_part;
#pragma omp barrier

#pragma omp for schedule(runtime)
      for (int i = 0; i < n; ++i)
      {
        const double ax_i = sum_x + x[i];
        x_new[i] = x[i] - tau * (ax_i - b[i]);
      }

#pragma omp single
      {
        diff_local = 0.0;
      }
#pragma omp barrier
      double diff_part = 0.0;
#pragma omp for schedule(runtime) nowait
      for (int i = 0; i < n; ++i)
      {
        const double d = x_new[i] - x[i];
        diff_part += d * d;
      }
#pragma omp atomic update
      diff_local += diff_part;
#pragma omp barrier

#pragma omp single
      {
        diff = std::sqrt(diff_local);
        ++it;
      }

#pragma omp for schedule(runtime)
      for (int i = 0; i < n; ++i)
      {
        x[i] = x_new[i];
      }
#pragma omp barrier
    }
  }

  const double elapsed = omp_get_wtime() - t0;

  double err = 0.0;
#pragma omp parallel for reduction(+ : err)
  for (int i = 0; i < n; ++i)
  {
    const double d = x[i] - 1.0;
    err += d * d;
  }
  err = std::sqrt(err);

  return SolverResult{elapsed, it, diff, err};
}

int main(int argc, char **argv)
{
  // Библеотеку буст
  const int n = (argc > 1) ? std::stoi(argv[1]) : 20000;
  const int max_iters = (argc > 2) ? std::stoi(argv[2]) : 5000;
  const double eps = (argc > 3) ? std::stod(argv[3]) : 1e-6;
  const int repeats = (argc > 4) ? std::stoi(argv[4]) : 3;
  const int fixed_threads_for_schedule = (argc > 5) ? std::stoi(argv[5]) : std::max(1, omp_get_max_threads() / 2);
  const double tau_factor = (argc > 6) ? std::stod(argv[6]) : 1e-3;

  const int max_threads = omp_get_max_threads();
  std::cout << "OpenMP max threads: " << max_threads << "\n";
  std::cout << "N=" << n << ", max_iters=" << max_iters << ", eps=" << eps
            << ", repeats=" << repeats << ", tau_factor=" << tau_factor << "\n";

  std::ofstream scaling_csv("scaling_results.csv");
  scaling_csv << "variant,threads,repeat,time_sec,iterations,diff_norm,error_norm\n";

  for (int threads = 1; threads <= max_threads; ++threads)
  {
    for (int r = 1; r <= repeats; ++r)
    {
      const auto res1 = solve_variant1_parallel_for(n, max_iters, eps, threads, tau_factor);
      scaling_csv << "variant1," << threads << "," << r << ","
                  << std::setprecision(10) << res1.elapsed_sec << ","
                  << res1.iterations << "," << res1.diff_norm << "," << res1.error_norm << "\n";
      std::cout << "[v1] threads=" << threads << ", repeat=" << r
                << ", time=" << res1.elapsed_sec << " sec, iters=" << res1.iterations << "\n";

      const auto res2 = solve_variant2_single_parallel(n, max_iters, eps, threads, tau_factor);
      scaling_csv << "variant2," << threads << "," << r << ","
                  << std::setprecision(10) << res2.elapsed_sec << ","
                  << res2.iterations << "," << res2.diff_norm << "," << res2.error_norm << "\n";
      std::cout << "[v2] threads=" << threads << ", repeat=" << r
                << ", time=" << res2.elapsed_sec << " sec, iters=" << res2.iterations << "\n";
    }
  }
  scaling_csv.close();

  const std::vector<ScheduleKind> schedules = {
      ScheduleKind::Static,
      ScheduleKind::Dynamic,
      ScheduleKind::Guided};
  const std::vector<int> chunks = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

  std::ofstream schedule_csv("schedule_results.csv");
  schedule_csv << "schedule,chunk,threads,repeat,time_sec,iterations,diff_norm,error_norm\n";

  const int sched_threads = std::min(max_threads, std::max(1, fixed_threads_for_schedule));
  for (const auto sched : schedules)
  {
    for (const int chunk : chunks)
    {
      for (int r = 1; r <= repeats; ++r)
      {
        const auto res = solve_variant2_runtime_schedule(
            n, max_iters, eps, sched_threads, tau_factor, sched, chunk);
        schedule_csv << to_string_schedule(sched) << "," << chunk << "," << sched_threads << ","
                     << r << "," << std::setprecision(10) << res.elapsed_sec << ","
                     << res.iterations << "," << res.diff_norm << "," << res.error_norm << "\n";
        std::cout << "[sched] " << to_string_schedule(sched)
                  << ", chunk=" << chunk
                  << ", repeat=" << r
                  << ", time=" << res.elapsed_sec << " sec\n";
      }
    }
  }
  schedule_csv.close();

  std::cout << "\nSaved: scaling_results.csv, schedule_results.csv\n";
  return 0;
}
