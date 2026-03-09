#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>

using namespace std;
using namespace std::chrono;

// Получение информации о системе
void print_system_info()
{
  cout << "\n=== Информация о вычислительном узле ===\n";

  system("lscpu | grep 'Model name'");
  system("lscpu | grep 'CPU(s)' | head -1");
  system("lscpu | grep 'NUMA'");

  system("cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo 'N/A'");

  system("numactl --hardware | grep 'available' 2>/dev/null || echo 'NUMA info not available'");
  system("numactl --hardware | grep 'size' 2>/dev/null");

  system("cat /etc/os-release | grep 'PRETTY_NAME' | cut -d'=' -f2 | tr -d '\"' 2>/dev/null");

  cout << "=====================================\n\n";
}

// Параллельная инициализация матрицы
void init_matrix_parallel(double *matrix, size_t rows, size_t cols)
{
#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < rows; i++)
  {
    for (size_t j = 0; j < cols; j++)
    {
      matrix[i * cols + j] = static_cast<double>(i + j);
    }
  }
}

// Параллельная инициализация вектора
void init_vector_parallel(double *vector, size_t size)
{
#pragma omp parallel for
  for (size_t i = 0; i < size; i++)
  {
    vector[i] = static_cast<double>(i);
  }
}

// Последовательное умножение матрицы на вектор
void matrix_vector_mult_serial(const double *matrix, const double *vector,
                               double *result, size_t rows, size_t cols)
{
  for (size_t i = 0; i < rows; i++)
  {
    result[i] = 0.0;
    for (size_t j = 0; j < cols; j++)
    {
      result[i] += matrix[i * cols + j] * vector[j];
    }
  }
}

// Параллельное умножение матрицы на вектор с использованием OpenMP
void matrix_vector_mult_parallel(const double *matrix, const double *vector,
                                 double *result, size_t rows, size_t cols)
{
#pragma omp parallel for
  for (size_t i = 0; i < rows; i++)
  {
    result[i] = 0.0;
    for (size_t j = 0; j < cols; j++)
    {
      result[i] += matrix[i * cols + j] * vector[j];
    }
  }
}

// Функция для замера времени выполнения
double run_benchmark(size_t size, int num_threads, bool parallel_init = true)
{
  size_t rows = size;
  size_t cols = size;

  double *matrix = nullptr;
  double *vector = nullptr;
  double *result = nullptr;

  try
  {
    matrix = new double[rows * cols];
    vector = new double[cols];
    result = new double[rows];
  }
  catch (const bad_alloc &e)
  {
    cerr << "Ошибка выделения памяти для размера " << size << endl;
    delete[] matrix;
    delete[] vector;
    delete[] result;
    return -1.0;
  }
  if (parallel_init && num_threads > 0)
  {
    omp_set_num_threads(num_threads);
    init_matrix_parallel(matrix, rows, cols);
    init_vector_parallel(vector, cols);
  }
  else
  {
    for (size_t i = 0; i < rows; i++)
    {
      for (size_t j = 0; j < cols; j++)
      {
        matrix[i * cols + j] = static_cast<double>(i + j);
      }
    }
    for (size_t j = 0; j < cols; j++)
    {
      vector[j] = static_cast<double>(j);
    }
  }
  if (num_threads > 0)
  {
    omp_set_num_threads(num_threads);
  }

  auto start = high_resolution_clock::now();

  if (num_threads == 0)
  {
    matrix_vector_mult_serial(matrix, vector, result, rows, cols);
  }
  else
  {
    matrix_vector_mult_parallel(matrix, vector, result, rows, cols);
  }

  auto end = high_resolution_clock::now();

  duration<double> elapsed = end - start;
  double seconds = elapsed.count();

  delete[] matrix;
  delete[] vector;
  delete[] result;

  return seconds;
}

// Функция для многократного замера с усреднением
double run_benchmark_averaged(size_t size, int num_threads, bool parallel_init = true, int num_runs = 5)
{
  vector<double> times;

  for (int run = 0; run < num_runs; run++)
  {
    double time = run_benchmark(size, num_threads, parallel_init);
    if (time < 0)
      return -1.0;
    times.push_back(time);
  }

  sort(times.begin(), times.end());
  double sum = 0;
  for (int i = 1; i < num_runs - 1; i++)
  {
    sum += times[i];
  }

  return sum / (num_runs - 2);
}

// Упрощенная функция - только через переменные окружения
void set_binding_policy(const string &policy)
{
  if (policy == "close")
  {
    setenv("OMP_PROC_BIND", "close", 1);
    setenv("OMP_PLACES", "cores", 1);
    cout << "  Установлена политика: CLOSE (потоки привязаны к ближайшим ядрам)\n";
  }
  else if (policy == "spread")
  {
    setenv("OMP_PROC_BIND", "spread", 1);
    setenv("OMP_PLACES", "cores", 1);
    cout << "  Установлена политика: SPREAD (потоки распределены по всем ядрам)\n";
  }
  else if (policy == "master")
  {
    setenv("OMP_PROC_BIND", "master", 1);
    setenv("OMP_PLACES", "cores", 1);
    cout << "  Установлена политика: MASTER (все потоки на одном ядре с мастером)\n";
  }
  else
  {
    setenv("OMP_PROC_BIND", "false", 1);
    cout << "  Установлена политика: NO BIND (без привязки)\n";
  }
}

int main()
{
  print_system_info();

  const size_t sizes[] = {20000, 40000};
  const char *size_labels[] = {"20000 (~3 GiB)", "40000 (~12 GiB)"};
  const int num_threads[] = {2, 4, 7, 8, 16, 20, 40};
  const int num_threads_count = sizeof(num_threads) / sizeof(num_threads[0]);

  const vector<string> binding_policies = {"none", "close", "spread"};
  const string policy_names[] = {"Без привязки", "Close binding", "Spread binding"};

  const int NUM_RUNS = 5;

  cout << "\n=========== ТЕСТИРОВАНИЕ С РАЗНЫМИ ПОЛИТИКАМИ ПРИВЯЗКИ ===========\n";
  cout << "Каждое значение усреднено по " << NUM_RUNS << " замерам (с отбрасыванием мин/макс)\n\n";

  for (size_t policy_idx = 0; policy_idx < binding_policies.size(); policy_idx++)
  {
    string policy = binding_policies[policy_idx];

    cout << "\n----------------------------------------------------------------\n";
    cout << "ПОЛИТИКА: " << policy_names[policy_idx] << "\n";
    cout << "----------------------------------------------------------------\n";

    set_binding_policy(policy);

    for (int s = 0; s < 2; s++)
    {
      size_t size = sizes[s];

      cout << "\n"
           << size_labels[s] << ":\n";
      cout << "------------------------------------------------\n";

      double T1 = run_benchmark_averaged(size, 0, false, NUM_RUNS);
      if (T1 < 0)
        continue;

      cout << "T1 (avg) = " << fixed << setprecision(6) << T1 << " s\n";
      cout << "Потоки | Время (с) avg | Ускорение\n";
      cout << "----------------------------------------\n";

      for (int t = 0; t < num_threads_count; t++)
      {
        int threads = num_threads[t];
        double Tp_avg = run_benchmark_averaged(size, threads, true, NUM_RUNS);

        if (Tp_avg < 0)
          continue;

        double speedup = T1 / Tp_avg;
        printf("%2d | %.6f | %.4f\n", threads, Tp_avg, speedup);
      }
    }
  }

  cout << "\n================================================\n";
  cout << "* Close binding - потоки привязаны к ближайшим ядрам\n";
  cout << "* Spread binding - потоки распределены по всем доступным ядрам\n";

  return 0;
}