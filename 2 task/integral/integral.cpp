#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <omp.h>

namespace
{
constexpr int kNsteps = 40000000;
constexpr int kDefaultRepeats = 50;
constexpr double kIntervalStart = -4.0;
constexpr double kIntervalEnd = 4.0;
const std::vector<int> kThreadCounts = {1, 2, 4, 7, 8, 16, 20, 40};

void print_system_info()
{
  std::cout << "\n=== Информация о вычислительном узле ===\n";
  std::system("lscpu | grep 'Model name'");
  std::system("cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo 'N/A'");
  std::system("numactl --hardware 2>/dev/null | grep -E 'available|node [0-9]+ size' || echo 'NUMA info not available'");
  std::system("cat /etc/os-release 2>/dev/null | grep 'PRETTY_NAME' | cut -d'=' -f2 | tr -d '\"'");
  std::cout << "=====================================\n\n";
}

double target_function(double x)
{
  return std::exp(-x * x);
}

double integrate_serial(double (*func)(double), double a, double b, int nsteps)
{
  const double h = (b - a) / static_cast<double>(nsteps);
  double sum = 0.0;

  for (int i = 0; i < nsteps; ++i)
  {
    sum += func(a + h * (static_cast<double>(i) + 0.5));
  }

  return sum * h;
}

double integrate_omp(double (*func)(double), double a, double b, int nsteps, int threads)
{
  const double h = (b - a) / static_cast<double>(nsteps);
  double sum = 0.0;
  omp_set_num_threads(threads);

#pragma omp parallel
  {
    double local_sum = 0.0;

#pragma omp for schedule(static)
    for (int i = 0; i < nsteps; ++i)
    {
      local_sum += func(a + h * (static_cast<double>(i) + 0.5));
    }

#pragma omp atomic update
    sum += local_sum;
  }

  return sum * h;
}

struct Measurement
{
  double time_sec = 0.0;
  double value = 0.0;
};

Measurement measure_serial()
{
  const double start = omp_get_wtime();
  const double value = integrate_serial(target_function, kIntervalStart, kIntervalEnd, kNsteps);
  return {omp_get_wtime() - start, value};
}

Measurement measure_parallel(int threads)
{
  const double start = omp_get_wtime();
  const double value = integrate_omp(target_function, kIntervalStart, kIntervalEnd, kNsteps, threads);
  return {omp_get_wtime() - start, value};
}
} // namespace

int main(int argc, char **argv)
{
  const int repeats = (argc > 1) ? std::stoi(argv[1]) : kDefaultRepeats;

  print_system_info();
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Интегрирование функции exp(-x*x) на отрезке ["
            << kIntervalStart << ", " << kIntervalEnd << "]\n";
  std::cout << "nsteps=" << kNsteps << ", запусков=" << repeats << "\n";

  std::ofstream runs_csv("integration_runs.csv");
  std::ofstream summary_csv("integration_summary.csv");
  if (!runs_csv || !summary_csv)
  {
    std::cerr << "Не удалось открыть CSV файлы для записи\n";
    return 1;
  }

  runs_csv << "method,threads,run,time_sec,value\n";
  summary_csv << "method,threads,avg_time_sec,speedup,efficiency,value\n";

  double serial_total = 0.0;
  double serial_value = 0.0;
  for (int run = 1; run <= repeats; ++run)
  {
    const Measurement m = measure_serial();
    serial_total += m.time_sec;
    serial_value = m.value;
    runs_csv << "serial,1," << run << "," << m.time_sec << "," << std::setprecision(12)
             << m.value << std::setprecision(6) << "\n";
  }

  const double serial_avg = serial_total / static_cast<double>(repeats);
  summary_csv << "serial,1," << serial_avg << ",1,1," << std::setprecision(12)
              << serial_value << std::setprecision(6) << "\n";

  double best_time = std::numeric_limits<double>::max();
  int best_threads = 1;
  double best_speedup = 1.0;
  double best_efficiency = 0.0;
  int best_efficiency_threads = 1;
  double best_efficiency_time = serial_avg;
  double best_efficiency_speedup = 1.0;

  for (const int threads : kThreadCounts)
  {
    double total_time = 0.0;
    double value = 0.0;

    for (int run = 1; run <= repeats; ++run)
    {
      const Measurement m = measure_parallel(threads);
      total_time += m.time_sec;
      value = m.value;
      runs_csv << "omp_atomic_local," << threads << "," << run << ","
               << m.time_sec << "," << std::setprecision(12) << m.value << std::setprecision(6) << "\n";
    }

    const double avg_time = total_time / static_cast<double>(repeats);
    const double speedup = serial_avg / avg_time;
    const double efficiency = speedup / static_cast<double>(threads);
    summary_csv << "omp_atomic_local," << threads << "," << avg_time << ","
                << speedup << "," << efficiency << "," << std::setprecision(12)
                << value << std::setprecision(6) << "\n";

    if (avg_time < best_time)
    {
      best_time = avg_time;
      best_threads = threads;
      best_speedup = speedup;
    }
    if (threads > 1 && efficiency > best_efficiency)
    {
      best_efficiency = efficiency;
      best_efficiency_threads = threads;
      best_efficiency_time = avg_time;
      best_efficiency_speedup = speedup;
    }

    std::cout << "threads=" << std::setw(2) << threads
              << " avg=" << avg_time
              << " speedup=" << speedup
              << " efficiency=" << efficiency << "\n";
  }

  std::cout << "\nМинимальное время / максимальная скорость: " << best_threads
            << " потоков, среднее время " << best_time
            << " сек, ускорение " << best_speedup << "\n";
  std::cout << "Максимальный КПД среди параллельных запусков: " << best_efficiency_threads
            << " потоков, среднее время " << best_efficiency_time
            << " сек, ускорение " << best_efficiency_speedup
            << ", КПД " << best_efficiency << "\n";
  std::cout << "Сохранено: integration_runs.csv, integration_summary.csv\n";
  return 0;
}
#if 0
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
#include <fstream>

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
    }
    else if (policy == "spread")
    {
        setenv("OMP_PROC_BIND", "spread", 1);
        setenv("OMP_PLACES", "cores", 1);
    }
    else
    {
        setenv("OMP_PROC_BIND", "false", 1);
    }
}

int main()
{
    print_system_info();

    const int num_threads[] = {2, 4, 7, 8, 16, 20, 40};
    const int num_threads_count = sizeof(num_threads) / sizeof(num_threads[0]);

    const std::vector<std::string> policies = {"none", "close", "spread"};
    const std::string policy_names[] = {"none", "close", "spread"};

    const int NUM_RUNS = 5;

    std::ofstream csv_file("integration_results.csv");
    if (!csv_file.is_open())
    {
        std::cerr << "Ошибка: не удалось создать файл integration_results.csv\n";
        return 1;
    }

    csv_file << "Policy,Threads,Time_Seconds,Speedup\n";

    double T1 = run_benchmark(integrate_serial, 0, NUM_RUNS);

    csv_file << "serial,0," << T1 << ",1.0\n";

    for (size_t p = 0; p < policies.size(); p++)
    {
        set_binding_policy(policies[p]);

        for (int t = 0; t < num_threads_count; t++)
        {
            int threads = num_threads[t];
            double time = run_benchmark(integrate_omp_local, threads, NUM_RUNS);
            double speedup = T1 / time;

            csv_file << policy_names[p] << ","
                     << threads << ","
                     << time << ","
                     << speedup << "\n";
        }
    }

    csv_file.close();
    std::cout << "Формат CSV: Policy,Threads,Time_Seconds,Speedup\n";

    return 0;
}
#endif