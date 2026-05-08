#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <string>
#include <vector>

#include <omp.h>

namespace
{
constexpr int kDefaultRepeats = 50;
const std::vector<int> kThreadCounts = {1, 2, 4, 7, 8, 16, 20, 40};
const std::vector<int> kMatrixSizes = {20000, 40000};

void print_system_info()
{
  std::cout << "\n=== Информация о вычислительном узле ===\n";
  std::system("lscpu | grep 'Model name'");
  std::system("cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo 'N/A'");
  std::system("numactl --hardware 2>/dev/null | grep -E 'available|node [0-9]+ size' || echo 'NUMA info not available'");
  std::system("cat /etc/os-release 2>/dev/null | grep 'PRETTY_NAME' | cut -d'=' -f2 | tr -d '\"'");
  std::cout << "=====================================\n\n";
}

void init_data(std::vector<double> &matrix, std::vector<double> &x, int n)
{
  const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);

#pragma omp parallel for schedule(static)
  for (std::int64_t idx = 0; idx < static_cast<std::int64_t>(total); ++idx)
  {
    const int row = static_cast<int>(idx / n);
    const int col = static_cast<int>(idx - static_cast<std::int64_t>(row) * n);
    matrix[static_cast<std::size_t>(idx)] = 1.0 + 0.000001 * static_cast<double>((row + col) % 100);
  }

#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i)
  {
    x[i] = 1.0 + 0.00001 * static_cast<double>(i % 100);
  }
}

double multiply_matrix_vector(const std::vector<double> &matrix,
                              const std::vector<double> &x,
                              std::vector<double> &y,
                              int n,
                              int threads)
{
  omp_set_num_threads(threads);
  const double start = omp_get_wtime();

#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i)
  {
    const double *row = matrix.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
    double sum = 0.0;
    for (int j = 0; j < n; ++j)
    {
      sum += row[j] * x[j];
    }
    y[i] = sum;
  }

  return omp_get_wtime() - start;
}

double checksum(const std::vector<double> &y)
{
  double sum = 0.0;
#pragma omp parallel for reduction(+ : sum) schedule(static)
  for (std::int64_t i = 0; i < static_cast<std::int64_t>(y.size()); ++i)
  {
    sum += y[static_cast<std::size_t>(i)];
  }
  return sum;
}
} 

int main(int argc, char **argv)
{
  const int repeats = (argc > 1) ? std::stoi(argv[1]) : kDefaultRepeats;

  print_system_info();
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Количество запусков для каждого режима: " << repeats << "\n";

  std::ofstream runs_csv("matrix_vector_runs.csv");
  std::ofstream summary_csv("matrix_vector_summary.csv");
  if (!runs_csv || !summary_csv)
  {
    std::cerr << "Не удалось открыть CSV файлы для записи\n";
    return 1;
  }

  runs_csv << "size,threads,run,time_sec,checksum\n";
  summary_csv << "size,threads,avg_time_sec,speedup,efficiency,checksum\n";

  for (const int n : kMatrixSizes)
  {
    std::cout << "\n=== Размер матрицы " << n << "x" << n << " ===\n";
    const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);

    std::vector<double> matrix;
    std::vector<double> x;
    std::vector<double> y;
    try
    {
      matrix.resize(total);
      x.resize(static_cast<std::size_t>(n));
      y.resize(static_cast<std::size_t>(n));
    }
    catch (const std::bad_alloc &)
    {
      std::cerr << "Недостаточно памяти для матрицы " << n << "x" << n << "\n";
      return 2;
    }

    init_data(matrix, x, n);

    double base_time = 0.0;
    for (const int threads : kThreadCounts)
    {
      double total_time = 0.0;
      double last_checksum = 0.0;

      for (int run = 1; run <= repeats; ++run)
      {
        const double elapsed = multiply_matrix_vector(matrix, x, y, n, threads);
        last_checksum = checksum(y);
        total_time += elapsed;

        runs_csv << n << "," << threads << "," << run << ","
                 << elapsed << "," << std::setprecision(12) << last_checksum << std::setprecision(6) << "\n";
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

  std::cout << "\nСохранено: matrix_vector_runs.csv, matrix_vector_summary.csv\n";
  return 0;
}

