#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <string>

const double PI = 3.14159265358979323846;
const double a = -4.0;
const double b = 4.0;
const int nsteps = 40000000;

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

double cpuSecond()
{
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

double func(double x)
{
  return exp(-x * x);
}

double integrate_serial(double (*func)(double), double a, double b, int n)
{
  double h = (b - a) / n;
  double sum = 0.0;

  for (int i = 0; i < n; i++)
    sum += func(a + h * (i + 0.5));

  return sum * h;
}

double integrate_omp_atomic(double (*func)(double), double a, double b, int n)
{
  double h = (b - a) / n;
  double sum = 0.0;

#pragma omp parallel for
  for (int i = 0; i < n; i++)
  {
    double local_sum = func(a + h * (i + 0.5));
#pragma omp atomic
    sum += local_sum;
  }

  return sum * h;
}

double integrate_omp_local(double (*func)(double), double a, double b, int n)
{
  double h = (b - a) / n;
  double sum = 0.0;

#pragma omp parallel
  {
    double local_sum = 0.0;

#pragma omp for
    for (int i = 0; i < n; i++)
    {
      local_sum += func(a + h * (i + 0.5));
    }

#pragma omp atomic
    sum += local_sum;
  }

  return sum * h;
}

double integrate_omp_reduction(double (*func)(double), double a, double b, int n)
{
  double h = (b - a) / n;
  double sum = 0.0;

#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < n; i++)
  {
    sum += func(a + h * (i + 0.5));
  }

  return sum * h;
}

double run_benchmark(double (*integrate_func)(double (*)(double), double, double, int),
                     int num_threads, int num_runs = 3)
{

  std::vector<double> times;

  for (int run = 0; run < num_runs; run++)
  {
    if (num_threads > 0)
    {
      omp_set_num_threads(num_threads);
    }

    double t = cpuSecond();
    double res = integrate_func(func, a, b, nsteps);
    t = cpuSecond() - t;
    times.push_back(t);

    if (run == 0 && num_threads <= 1)
    {
      printf("Result: %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    }
  }

  std::sort(times.begin(), times.end());
  double sum = 0;
  for (int i = 1; i < num_runs - 1; i++)
  {
    sum += times[i];
  }

  return sum / (num_runs - 2);
}

void set_binding_policy(const std::string &policy)
{
  if (policy == "close")
  {
    setenv("OMP_PROC_BIND", "close", 1);
    setenv("OMP_PLACES", "cores", 1);
    std::cout << "  Установлена политика: CLOSE (потоки привязаны к ближайшим ядрам)\n";
  }
  else if (policy == "spread")
  {
    setenv("OMP_PROC_BIND", "spread", 1);
    setenv("OMP_PLACES", "cores", 1);
    std::cout << "  Установлена политика: SPREAD (потоки распределены по всем ядрам)\n";
  }
  else
  {
    setenv("OMP_PROC_BIND", "false", 1);
    std::cout << "  Установлена политика: NO BIND (без привязки)\n";
  }
}

int main()
{
  print_system_info();

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "\n=== ЧИСЛЕННОЕ ИНТЕГРИРОВАНИЕ ===\n";
  printf("Функция: exp(-x*x) на [%.1f, %.1f]\n", a, b);
  printf("Количество точек: %d\n", nsteps);
  printf("Точное значение интеграла: %.12f (sqrt(PI))\n", sqrt(PI));

  const int num_threads[] = {2, 4, 7, 8, 16, 20, 40};
  const int num_threads_count = sizeof(num_threads) / sizeof(num_threads[0]);

  const std::vector<std::string> policies = {"none", "close", "spread"};
  const std::string policy_names[] = {"Без привязки", "Close binding", "Spread binding"};

  const int NUM_RUNS = 5;

  std::cout << "\n========================================\n";
  std::cout << "РЕЗУЛЬТАТЫ (усреднено по " << NUM_RUNS << " запускам)\n";
  std::cout << "========================================\n";

  std::vector<std::vector<double>> all_times(policies.size(), std::vector<double>(num_threads_count));
  std::vector<std::vector<double>> all_speedups(policies.size(), std::vector<double>(num_threads_count));

  std::cout << "\nПоследовательное выполнение:\n";
  double T1 = run_benchmark(integrate_serial, 0, NUM_RUNS);
  printf("T1 = %.6f с\n", T1);

  for (size_t p = 0; p < policies.size(); p++)
  {
    std::cout << "\n----------------------------------------------------------------\n";
    std::cout << "ПОЛИТИКА: " << policy_names[p] << "\n";
    std::cout << "----------------------------------------------------------------\n";

    set_binding_policy(policies[p]);

    printf("%4s | %12s | %10s\n", "Потоки", "Время (с)", "Ускорение");
    std::cout << std::string(40, '-') << "\n";

    for (int t = 0; t < num_threads_count; t++)
    {
      int threads = num_threads[t];
      double time = run_benchmark(integrate_omp_local, threads, NUM_RUNS);
      double speedup = T1 / time;

      all_times[p][t] = time;
      all_speedups[p][t] = speedup;

      printf("%4d | %12.6f | %10.4f\n", threads, time, speedup);
    }
  }

  // Сводная таблица ускорений
  std::cout << "====================СВОДНАЯ ТАБЛИЦА УСКОРЕНИЙ S(p)====================\n";

  printf("%12s |", "Политика");
  for (int t = 0; t < num_threads_count; t++)
  {
    printf(" %4d |", num_threads[t]);
  }
  std::cout << "\n"
            << std::string(70, '-') << "\n";

  for (size_t p = 0; p < policies.size(); p++)
  {
    printf("%12s |", policy_names[p].c_str());
    for (int t = 0; t < num_threads_count; t++)
    {
      printf(" %4.2f |", all_speedups[p][t]);
    }
    std::cout << "\n";
  }

  // Вывод для построения графика в Python
  std::cout << "\n========== ДАННЫЕ ДЛЯ ГРАФИКА ==========\n";

  std::cout << "threads = [2, 4, 7, 8, 16, 20, 40]\n";
  for (size_t p = 0; p < policies.size(); p++)
  {
    std::cout << "speedup_" << policies[p] << " = [";
    for (int t = 1; t < num_threads_count; t++)
    {
      if (t > 1)
        std::cout << ", ";
      printf("%.4f", all_speedups[p][t]);
    }
    std::cout << "]\n";
  }

  std::cout << "\n====================== ВЫВОДЫ О МАСШТАБИРУЕМОСТИ ======================\n";

  for (size_t p = 0; p < policies.size(); p++)
  {
    std::cout << "\n"
              << policy_names[p] << ":\n";

    double max_speedup = all_speedups[p][num_threads_count - 1];
    double efficiency = max_speedup / num_threads[num_threads_count - 1] * 100;

    printf("  Максимальное ускорение: %.2f при %d потоках\n",
           max_speedup, num_threads[num_threads_count - 1]);
    printf("  Эффективность: %.1f%%\n", efficiency);

    if (efficiency > 80)
    {
      std::cout << "  Отличная масштабируемость\n";
    }
    else if (efficiency > 50)
    {
      std::cout << "  Хорошая масштабируемость\n";
    }
    else
    {
      std::cout << "  Удовлетворительная масштабируемость\n";
    }
  }

  return 0;
}