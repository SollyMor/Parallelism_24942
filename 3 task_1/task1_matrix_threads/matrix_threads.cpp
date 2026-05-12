#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <new>
#include <string>
#include <thread>
#include <vector>

namespace
{
  constexpr int kDefaultRepeats = 100;
  const std::vector<int> kThreadCounts = {1, 2, 4, 7, 8, 16, 20, 40};
  const std::vector<int> kMatrixSizes = {20000, 40000};

  void print_system_info()
  {
    std::cout << "\n=== System information ===\n";
    std::system("lscpu | grep 'Model name'");
    std::system("cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo 'N/A'");
    std::system("numactl --hardware 2>/dev/null | grep -E 'available|node [0-9]+ size' || echo 'NUMA info not available'");
    std::system("cat /etc/os-release 2>/dev/null | grep 'PRETTY_NAME' | cut -d'=' -f2 | tr -d '\"'");
    std::cout << "==========================\n\n";
  }

  template <typename Func>
  void run_threads(int threads, int n, Func func)
  {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));

    for (int tid = 0; tid < threads; ++tid)
    {
      const int begin = static_cast<int>((static_cast<std::int64_t>(n) * tid) / threads);
      const int end = static_cast<int>((static_cast<std::int64_t>(n) * (tid + 1)) / threads);
      workers.emplace_back(func, begin, end);
    }

    for (std::thread &worker : workers)
    {
      worker.join();
    }
  }

  void init_data(std::vector<double> &matrix, std::vector<double> &x, int n, int threads)
  {
    run_threads(threads, n, [&](int row_begin, int row_end)
                {
    for (int i = row_begin; i < row_end; ++i)
    {
      double *row = matrix.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
      for (int j = 0; j < n; ++j)
      {
        row[j] = 1.0 + 0.000001 * static_cast<double>((i + j) % 100);
      }
      x[static_cast<std::size_t>(i)] = 1.0 + 0.00001 * static_cast<double>(i % 100);
    } });
  }

  double multiply_matrix_vector(const std::vector<double> &matrix,
                                const std::vector<double> &x,
                                std::vector<double> &y,
                                int n,
                                int threads)
  {
    const auto start = std::chrono::steady_clock::now();

    run_threads(threads, n, [&](int row_begin, int row_end)
                {
    for (int i = row_begin; i < row_end; ++i)
    {
      const double *row = matrix.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
      double sum = 0.0;
      for (int j = 0; j < n; ++j)
      {
        sum += row[j] * x[static_cast<std::size_t>(j)];
      }
      y[static_cast<std::size_t>(i)] = sum;
    } });

    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    return elapsed.count();
  }

  double checksum(const std::vector<double> &y)
  {
    double sum = 0.0;
    for (double value : y)
    {
      sum += value;
    }
    return sum;
  }
} // namespace

int main(int argc, char **argv)
{
  const int repeats = (argc > 1) ? std::stoi(argv[1]) : kDefaultRepeats;

  print_system_info();
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Repeats for every mode: " << repeats << "\n";

  std::ofstream runs_csv("matrix_threads_runs.csv");
  std::ofstream summary_csv("matrix_threads_summary.csv");
  if (!runs_csv || !summary_csv)
  {
    std::cerr << "Cannot open CSV files for writing\n";
    return 1;
  }

  runs_csv << "size,threads,run,time_sec,checksum\n";
  summary_csv << "size,threads,avg_time_sec,speedup,efficiency,checksum\n";

  for (int n : kMatrixSizes)
  {
    std::cout << "\n=== Matrix " << n << "x" << n << " ===\n";

    std::vector<double> matrix;
    std::vector<double> x;
    std::vector<double> y;
    try
    {
      const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
      matrix.resize(total);
      x.resize(static_cast<std::size_t>(n));
      y.resize(static_cast<std::size_t>(n));
    }
    catch (const std::bad_alloc &)
    {
      std::cerr << "Not enough memory for matrix " << n << "x" << n << "\n";
      return 2;
    }

    init_data(matrix, x, n, std::min(40, static_cast<int>(std::thread::hardware_concurrency())));

    double base_time = 0.0;
    for (int threads : kThreadCounts)
    {
      double total_time = 0.0;
      double last_checksum = 0.0;

      for (int run = 1; run <= repeats; ++run)
      {
        const double elapsed = multiply_matrix_vector(matrix, x, y, n, threads);
        last_checksum = checksum(y);
        total_time += elapsed;

        runs_csv << n << "," << threads << "," << run << ","
                 << elapsed << "," << std::setprecision(12) << last_checksum
                 << std::setprecision(6) << "\n";
      }

      const double avg_time = total_time / static_cast<double>(repeats);
      if (threads == 1)
      {
        base_time = avg_time;
      }

      const double speedup = base_time / avg_time;
      const double efficiency = speedup / static_cast<double>(threads);
      summary_csv << n << "," << threads << "," << avg_time << ","
                  << speedup << "," << efficiency << "," << last_checksum << "\n";

      std::cout << "threads=" << std::setw(2) << threads
                << " avg=" << avg_time
                << " speedup=" << speedup
                << " efficiency=" << efficiency << "\n";
    }
  }

  std::cout << "\nSaved: matrix_threads_runs.csv, matrix_threads_summary.csv\n";
  return 0;
}
