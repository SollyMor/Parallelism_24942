#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <string>

// Получение информации о системе
void print_system_info()
{
  std::cout << "\n=== Информация о вычислительном узле ===\n";

  system("lscpu | grep 'Model name'");
  system("lscpu | grep 'CPU(s)' | head -1");
  system("lscpu | grep 'NUMA'");

  system("cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo 'N/A'");

  system("numactl --hardware | grep 'available' 2>/dev/null || echo 'NUMA info not available'");
  system("numactl --hardware | grep 'size' 2>/dev/null");

  system("cat /etc/os-release | grep 'PRETTY_NAME' | cut -d'=' -f2 | tr -d '\"' 2>/dev/null");

  std::cout << "=====================================\n\n";
}

// Вспомогательная функция для форматирования вектора
template <typename T>
std::string vector_to_string(const std::vector<T> &v)
{
  std::string result = "[";
  for (size_t i = 0; i < v.size(); i++)
  {
    if (i > 0)
      result += ", ";
    result += std::to_string(v[i]);
  }
  result += "]";
  return result;
}

// Класс для решения СЛАУ методом простой итерации
class SimpleIterationSolver
{
private:
  int N;
  std::vector<std::vector<double>> A;
  std::vector<double> b;
  std::vector<double> x;
  double epsilon;
  int max_iter;

public:
  SimpleIterationSolver(int size, double eps = 1e-10, int max_it = 10000)
      : N(size), epsilon(eps), max_iter(max_it)
  {
    std::cout << "Инициализация матрицы " << N << "x" << N << "...\n";

    // Инициализация матрицы A: диагональ = 2.0, остальное = 1.0
    A.resize(N, std::vector<double>(N, 1.0));
    for (int i = 0; i < N; i++)
    {
      A[i][i] = 2.0;
    }

    // Инициализация вектора b: все элементы = N+1
    b.resize(N, N + 1.0);

    // Начальное приближение x: все элементы = 0
    x.resize(N, 0.0);

    std::cout << "Инициализация завершена\n";
  }

  // Последовательная версия
  double solve_serial()
  {
    std::vector<double> x_new(N, 0.0);
    double norm_diff;
    int iter = 0;

    auto start = std::chrono::high_resolution_clock::now();

    do
    {
      // Вычисление нового приближения
      for (int i = 0; i < N; i++)
      {
        double sum = 0.0;
        for (int j = 0; j < N; j++)
        {
          if (j != i)
          {
            sum += A[i][j] * x[j];
          }
        }
        x_new[i] = (b[i] - sum) / A[i][i];
      }

      // Вычисление нормы разности
      norm_diff = 0.0;
      for (int i = 0; i < N; i++)
      {
        double diff = x_new[i] - x[i];
        norm_diff += diff * diff;
      }
      norm_diff = std::sqrt(norm_diff);

      // Обновление решения
      x = x_new;
      iter++;

    } while (norm_diff > epsilon && iter < max_iter);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Последовательная версия: " << iter << " итераций, норма невязки: " << norm_diff << std::endl;

    return elapsed.count();
  }

  // Вариант 1: отдельные параллельные секции для каждого цикла
  double solve_parallel_v1(int num_threads)
  {
    omp_set_num_threads(num_threads);

    std::vector<double> x_new(N, 0.0);
    double norm_diff;
    int iter = 0;

    auto start = std::chrono::high_resolution_clock::now();

    do
    {
// Параллельное вычисление нового приближения
#pragma omp parallel for
      for (int i = 0; i < N; i++)
      {
        double sum = 0.0;
        for (int j = 0; j < N; j++)
        {
          if (j != i)
          {
            sum += A[i][j] * x[j];
          }
        }
        x_new[i] = (b[i] - sum) / A[i][i];
      }

      // Параллельное вычисление нормы
      double sum_sq = 0.0;
#pragma omp parallel for reduction(+ : sum_sq)
      for (int i = 0; i < N; i++)
      {
        double diff = x_new[i] - x[i];
        sum_sq += diff * diff;
      }
      norm_diff = std::sqrt(sum_sq);

      // Обновление решения (последовательно)
      x = x_new;
      iter++;

    } while (norm_diff > epsilon && iter < max_iter);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (num_threads == 1)
    {
      std::cout << "Вариант 1: " << iter << " итераций, норма невязки: " << norm_diff << std::endl;
    }

    return elapsed.count();
  }

  // Вариант 2: одна параллельная секция на весь итерационный процесс
  double solve_parallel_v2(int num_threads)
  {
    omp_set_num_threads(num_threads);

    std::vector<double> x_new(N, 0.0);
    double norm_diff;
    int iter = 0;

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel
    {
      do
      {
// Вычисление нового приближения
#pragma omp for
        for (int i = 0; i < N; i++)
        {
          double sum = 0.0;
          for (int j = 0; j < N; j++)
          {
            if (j != i)
            {
              sum += A[i][j] * x[j];
            }
          }
          x_new[i] = (b[i] - sum) / A[i][i];
        }

        // Вычисление нормы
        double local_sum = 0.0;
#pragma omp for reduction(+ : local_sum)
        for (int i = 0; i < N; i++)
        {
          double diff = x_new[i] - x[i];
          local_sum += diff * diff;
        }

#pragma omp single
        {
          norm_diff = std::sqrt(local_sum);
        }

// Обновление решения
#pragma omp single
        {
          x = x_new;
          iter++;
        }

#pragma omp barrier

      } while (norm_diff > epsilon && iter < max_iter);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (num_threads == 1)
    {
      std::cout << "Вариант 2: " << iter << " итераций, норма невязки: " << norm_diff << std::endl;
    }

    return elapsed.count();
  }

  // Исследование schedule
  void test_schedule(int num_threads, int schedule_type, int chunk_size = 0)
  {
    omp_set_num_threads(num_threads);

    std::vector<double> x_new(N, 0.0);
    double norm_diff;
    int iter = 0;
    std::vector<double> x_copy = x;

    auto start = std::chrono::high_resolution_clock::now();

    do
    {
      // Выбор типа schedule
      if (schedule_type == 0)
      { // static
#pragma omp parallel for schedule(static, chunk_size)
        for (int i = 0; i < N; i++)
        {
          double sum = 0.0;
          for (int j = 0; j < N; j++)
          {
            if (j != i)
            {
              sum += A[i][j] * x_copy[j];
            }
          }
          x_new[i] = (b[i] - sum) / A[i][i];
        }
      }
      else if (schedule_type == 1)
      { // dynamic
#pragma omp parallel for schedule(dynamic, chunk_size)
        for (int i = 0; i < N; i++)
        {
          double sum = 0.0;
          for (int j = 0; j < N; j++)
          {
            if (j != i)
            {
              sum += A[i][j] * x_copy[j];
            }
          }
          x_new[i] = (b[i] - sum) / A[i][i];
        }
      }
      else
      { // guided
#pragma omp parallel for schedule(guided, chunk_size)
        for (int i = 0; i < N; i++)
        {
          double sum = 0.0;
          for (int j = 0; j < N; j++)
          {
            if (j != i)
            {
              sum += A[i][j] * x_copy[j];
            }
          }
          x_new[i] = (b[i] - sum) / A[i][i];
        }
      }

      // Вычисление нормы
      double sum_sq = 0.0;
#pragma omp parallel for reduction(+ : sum_sq)
      for (int i = 0; i < N; i++)
      {
        double diff = x_new[i] - x_copy[i];
        sum_sq += diff * diff;
      }
      norm_diff = std::sqrt(sum_sq);

      x_copy = x_new;
      iter++;

    } while (norm_diff > epsilon && iter < max_iter);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::string schedule_name;
    if (schedule_type == 0)
      schedule_name = "static";
    else if (schedule_type == 1)
      schedule_name = "dynamic";
    else
      schedule_name = "guided";

    std::cout << "schedule(" << schedule_name << ", " << (chunk_size > 0 ? std::to_string(chunk_size) : "default")
              << "): " << elapsed.count() << " с, " << iter << " итераций\n";
  }

  // Сброс решения к начальному приближению
  void reset()
  {
    x.assign(N, 0.0);
  }
};

// Функция для замера времени с усреднением для solve_parallel_v1
double run_benchmark_v1(SimpleIterationSolver &solver, int num_threads, int num_runs = 2)
{
  std::vector<double> times;

  for (int run = 0; run < num_runs; run++)
  {
    solver.reset();
    double time = solver.solve_parallel_v1(num_threads);
    times.push_back(time);
  }

  // Усредняем
  double sum = 0;
  for (double t : times)
  {
    sum += t;
  }

  return sum / num_runs;
}

// Функция для замера времени с усреднением для solve_parallel_v2
double run_benchmark_v2(SimpleIterationSolver &solver, int num_threads, int num_runs = 2)
{
  std::vector<double> times;

  for (int run = 0; run < num_runs; run++)
  {
    solver.reset();
    double time = solver.solve_parallel_v2(num_threads);
    times.push_back(time);
  }

  // Усредняем
  double sum = 0;
  for (double t : times)
  {
    sum += t;
  }

  return sum / num_runs;
}

int main()
{
  print_system_info();

  int N = 5000; // Увеличил размер для более реалистичного теста
  double epsilon = 1e-6;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "\n=== РЕШЕНИЕ СЛАУ МЕТОДОМ ПРОСТОЙ ИТЕРАЦИИ ===\n";
  std::cout << "Размер системы: " << N << "x" << N << std::endl;
  std::cout << "Точность: " << epsilon << std::endl;

  SimpleIterationSolver solver(N, epsilon);

  std::vector<int> num_threads = {1, 2, 4, 8, 16};
  int max_threads = omp_get_max_threads();
  std::cout << "Максимальное доступное количество потоков: " << max_threads << std::endl;

  std::vector<int> available_threads;
  for (int t : num_threads)
  {
    if (t <= max_threads)
    {
      available_threads.push_back(t);
    }
  }

  if (available_threads.empty())
  {
    available_threads.push_back(1);
  }

  std::cout << "\n========================================\n";
  std::cout << "ИССЛЕДОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ\n";
  std::cout << "========================================\n";

  std::cout << "\nПоследовательная версия:\n";
  solver.reset();
  double T1 = solver.solve_serial();
  std::cout << "Время: " << T1 << " с\n";

  std::cout << "\n"
            << std::string(100, '=') << std::endl;
  std::cout << "Потоки | Вариант 1 (время) | Ускорение | Вариант 2 (время) | Ускорение | Эффективность V2" << std::endl;
  std::cout << std::string(100, '-') << std::endl;

  std::vector<double> times_v1, times_v2, speedups_v1, speedups_v2;

  for (int threads : available_threads)
  {
    std::cout << "\nТестирование с " << threads << " потоками..." << std::endl;

    double t_v1 = run_benchmark_v1(solver, threads, 2);
    double s_v1 = T1 / t_v1;

    double t_v2 = run_benchmark_v2(solver, threads, 2);
    double s_v2 = T1 / t_v2;
    double eff_v2 = s_v2 / threads * 100;

    std::cout << std::setw(4) << threads << "    | "
              << std::setw(12) << t_v1 << "  | " << std::setw(8) << std::fixed << std::setprecision(4) << s_v1 << " | "
              << std::setw(12) << t_v2 << "  | " << std::setw(8) << s_v2 << " | "
              << std::setw(8) << std::fixed << std::setprecision(2) << eff_v2 << "%" << std::endl;

    times_v1.push_back(t_v1);
    times_v2.push_back(t_v2);
    speedups_v1.push_back(s_v1);
    speedups_v2.push_back(s_v2);
  }

  std::cout << std::string(100, '=') << std::endl;

  std::cout << "\nИССЛЕДОВАНИЕ ПАРАМЕТРОВ SCHEDULE\n";
  std::cout << "Фиксированное число потоков: " << std::min(8, max_threads) << std::endl;

  std::vector<int> chunk_sizes = {1, 10, 100, 1000, 0};

  for (int schedule = 0; schedule < 3; schedule++)
  {
    std::cout << "\n--- schedule type: " << (schedule == 0 ? "static" : schedule == 1 ? "dynamic"
                                                                                      : "guided")
              << " ---\n";
    for (int chunk : chunk_sizes)
    {
      solver.reset();
      solver.test_schedule(std::min(8, max_threads), schedule, chunk);
    }
  }

  // ПРОБЛЕМНОЕ МЕСТО - теперь эти строки точно будут выполнены
  std::cout << "\n\n========== ДАННЫЕ ДЛЯ ГРАФИКОВ ==========\n";
  std::cout << "threads = " << vector_to_string(available_threads) << std::endl;
  std::cout << "times_v1 = " << vector_to_string(times_v1) << std::endl;
  std::cout << "times_v2 = " << vector_to_string(times_v2) << std::endl;
  std::cout << "speedups_v1 = " << vector_to_string(speedups_v1) << std::endl;
  std::cout << "speedups_v2 = " << vector_to_string(speedups_v2) << std::endl;

  std::cout << "\n========== ВЫВОДЫ ==========\n";
  std::cout << "------------------------\n";

  double avg_speedup_v1 = 0, avg_speedup_v2 = 0;
  int count = 0;

  for (size_t i = 1; i < speedups_v1.size(); i++)
  {
    avg_speedup_v1 += speedups_v1[i];
    avg_speedup_v2 += speedups_v2[i];
    count++;
  }

  if (count > 0)
  {
    avg_speedup_v1 /= count;
    avg_speedup_v2 /= count;
  }

  std::cout << "Среднее ускорение (без учета 1 потока) Вариант 1: " << avg_speedup_v1 << std::endl;
  std::cout << "Среднее ускорение (без учета 1 потока) Вариант 2: " << avg_speedup_v2 << std::endl;

  double max_speedup_v1 = 0, max_speedup_v2 = 0;
  int max_threads_v1 = 0, max_threads_v2 = 0;

  for (size_t i = 1; i < speedups_v1.size(); i++)
  {
    if (speedups_v1[i] > max_speedup_v1)
    {
      max_speedup_v1 = speedups_v1[i];
      max_threads_v1 = available_threads[i];
    }
    if (speedups_v2[i] > max_speedup_v2)
    {
      max_speedup_v2 = speedups_v2[i];
      max_threads_v2 = available_threads[i];
    }
  }

  std::cout << "\nМаксимальное ускорение Вариант 1: " << max_speedup_v1
            << " при " << max_threads_v1 << " потоках" << std::endl;
  std::cout << "Максимальное ускорение Вариант 2: " << max_speedup_v2
            << " при " << max_threads_v2 << " потоках" << std::endl;

  std::cout << "\nПрограмма успешно завершена!" << std::endl;

  return 0;
}