#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <numeric>

const double EPSILON = 1e-13;
const double TAU = -1e-5;
const int N = 7100;
const int NUM_RUNS = 5;

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

void init_system(std::vector<double> &a, std::vector<double> &b, std::vector<double> &x, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            a[i * n + j] = 1.0 + (i == j ? 1.0 : 0.0);
        }
        b[i] = static_cast<double>(n + 1);
        x[i] = 0.0;
    }
}

void solve_serial(const std::vector<double> &a, const std::vector<double> &b,
                  std::vector<double> &x, size_t n, int /*num_threads*/)
{
    std::vector<double> diff(n);
    double b_len_sq = 0.0;
    double dev = 0.0;

    for (size_t i = 0; i < n; i++)
    {
        b_len_sq += b[i] * b[i];
        diff[i] = 0.0;
    }
    dev = b_len_sq;

    while (std::sqrt(dev / b_len_sq) >= EPSILON)
    {
        dev = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            x[i] = x[i] - TAU * diff[i];
        }
        for (size_t i = 0; i < n; ++i)
        {
            double sum_ax = 0.0;
            for (size_t j = 0; j < n; ++j)
                sum_ax += a[i * n + j] * x[j];
            diff[i] = b[i] - sum_ax;
            dev += diff[i] * diff[i];
        }
    }
}

void solve_parallel_1(const std::vector<double> &a, const std::vector<double> &b,
                      std::vector<double> &x, size_t n, int num_threads)
{
    std::vector<double> diff(n);
    double b_len_sq = 0.0;
    double dev = 0.0;

    for (size_t i = 0; i < n; ++i)
    {
        b_len_sq += b[i] * b[i];
        diff[i] = 0.0;
    }
    dev = b_len_sq;

    while (std::sqrt(dev / b_len_sq) >= EPSILON)
    {
        dev = 0.0;
#pragma omp parallel for num_threads(num_threads)
        for (size_t i = 0; i < n; ++i)
        {
            x[i] = x[i] - TAU * diff[i];
        }
#pragma omp parallel for num_threads(num_threads) reduction(+ : dev)
        for (size_t i = 0; i < n; ++i)
        {
            double sum_ax = 0.0;
            for (size_t j = 0; j < n; ++j)
                sum_ax += a[i * n + j] * x[j];
            diff[i] = b[i] - sum_ax;
            dev += diff[i] * diff[i];
        }
    }
}

void solve_parallel_2(const std::vector<double> &a, const std::vector<double> &b,
                      std::vector<double> &x, size_t n, int num_threads)
{
    std::vector<double> diff(n);
    double b_len_sq = 0.0;

    for (size_t i = 0; i < n; ++i)
    {
        b_len_sq += b[i] * b[i];
        diff[i] = 0.0;
    }

    double dev = 0.0;
    bool stop = false;

#pragma omp parallel num_threads(num_threads) shared(a, b, x, diff, dev, stop) firstprivate(b_len_sq)
    {
        while (!stop)
        {
#pragma omp single
            {
                dev = 0.0;
            }
#pragma omp for schedule(runtime)
            for (size_t i = 0; i < n; ++i)
            {
                x[i] = x[i] - TAU * diff[i];
            }
#pragma omp for schedule(runtime) reduction(+ : dev)
            for (size_t i = 0; i < n; ++i)
            {
                double sum_ax = 0.0;
                for (size_t j = 0; j < n; ++j)
                {
                    sum_ax += a[i * n + j] * x[j];
                }
                diff[i] = b[i] - sum_ax;
                dev += diff[i] * diff[i];
            }
#pragma omp single
            {
                stop = (std::sqrt(dev / b_len_sq) < EPSILON);
            }
#pragma omp barrier
        }
    }
}

double run_single_measurement(void (*solve)(const std::vector<double> &, const std::vector<double> &,
                                            std::vector<double> &, size_t, int),
                              size_t n, int num_threads)
{
    std::vector<double> a(n * n);
    std::vector<double> b(n);
    std::vector<double> x(n);
    init_system(a, b, x, n);

    auto start = std::chrono::steady_clock::now();
    solve(a, b, x, n, num_threads);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    return elapsed.count();
}

double run_benchmark_averaged(void (*solve)(const std::vector<double> &, const std::vector<double> &,
                                            std::vector<double> &, size_t, int),
                              size_t n, int num_threads, int num_runs = NUM_RUNS)
{
    double sum = 0.0;

    for (int run = 0; run < num_runs; run++)
    {
        double time = run_single_measurement(solve, n, num_threads);
        sum += time;
    }

    return sum / num_runs;
}

int main()
{
    print_system_info();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n=== РЕШЕНИЕ СЛАУ МЕТОДОМ ПРОСТЫХ ИТЕРАЦИЙ ===\n";
    std::cout << "Размер матрицы: " << N << "×" << N << "\n";
    std::cout << "Точность: " << EPSILON << "\n";
    std::cout << "Количество запусков: " << NUM_RUNS << " (простое среднее)\n";

    const std::vector<int> num_threads = {1, 2, 4, 7, 8, 16, 20, 40};
    const int num_threads_count = num_threads.size();

    std::cout << "\n=== ФАЗА 1: СРАВНЕНИЕ АЛГОРИТМОВ ===\n";

    std::cout << "\nПараллельные алгоритмы:\n";
    std::cout << std::string(100, '-') << "\n";
    printf("%8s | %15s | %15s | %15s | %12s | %12s\n",
           "Потоки", "T1 (с)", "Время 1 (с)", "Время 2 (с)", "Ускор 1", "Ускор 2");
    std::cout << std::string(100, '-') << "\n";

    std::vector<double> times1, times2, speedups1, speedups2;

    for (int threads : num_threads)
    {
        double T1 = run_benchmark_averaged(solve_serial, N, 1, NUM_RUNS);
        double time1 = run_benchmark_averaged(solve_parallel_1, N, threads, NUM_RUNS);
        double time2 = run_benchmark_averaged(solve_parallel_2, N, threads, NUM_RUNS);

        double speedup1 = T1 / time1;
        double speedup2 = T1 / time2;

        times1.push_back(time1);
        times2.push_back(time2);
        speedups1.push_back(speedup1);
        speedups2.push_back(speedup2);

        printf("%8d | %15.6f | %15.6f | %15.6f | %12.4f | %12.4f\n",
               threads, T1, time1, time2, speedup1, speedup2);
    }

    std::cout << "\n=== ФАЗА 2: ИССЛЕДОВАНИЕ РАСПИСАНИЙ (SCHEDULE) ===\n";
    std::cout << "Используется алгоритм 2 с " << num_threads[4] << " потоками\n\n";

    struct ScheduleConfig
    {
        omp_sched_t kind;
        int chunk;
        const char *name;
    };

    const std::vector<ScheduleConfig> configs = {
        {omp_sched_static, 1, "static,1"},
        {omp_sched_static, 4, "static,4"},
        {omp_sched_static, 100, "static,100"},
        {omp_sched_static, N / num_threads[4], "static,N/T"},
        {omp_sched_dynamic, 1, "dynamic,1"},
        {omp_sched_dynamic, 4, "dynamic,4"},
        {omp_sched_dynamic, 100, "dynamic,100"},
        {omp_sched_dynamic, N / num_threads[4], "dynamic,N/T"},
        {omp_sched_guided, 1, "guided,1"},
        {omp_sched_guided, 4, "guided,4"},
        {omp_sched_guided, 100, "guided,100"},
        {omp_sched_guided, N / num_threads[4], "guided,N/T"},
    };

    std::cout << std::string(80, '-') << "\n";
    printf("%20s | %15s | %15s | %12s\n",
           "Расписание", "Время (с)", "T1 (с)", "Ускорение");
    std::cout << std::string(80, '-') << "\n";

    double T1_schedule = run_benchmark_averaged(solve_serial, N, 1, NUM_RUNS);

    for (const auto &cfg : configs)
    {
        omp_set_schedule(cfg.kind, cfg.chunk);
        double time = run_benchmark_averaged(solve_parallel_2, N, num_threads[4], NUM_RUNS);
        double speedup = T1_schedule / time;

        printf("%20s | %15.6f | %15.6f | %12.4f\n",
               cfg.name, time, T1_schedule, speedup);
    }

    std::cout << "\n=== ЗАПИСЬ РЕЗУЛЬТАТОВ В CSV ФАЙЛЫ ===\n";

    std::ofstream iter_file("iterationData.csv");
    std::ofstream summ_file("summary.csv");

    if (iter_file && summ_file)
    {
        iter_file << "num_threads,run,time_serial,time_parallel_1,time_parallel_2\n";
        summ_file << "num_threads,time_serial,time_parallel_1,time_parallel_2,speedup_1,speedup_2\n";

        for (size_t t_idx = 0; t_idx < num_threads.size(); t_idx++)
        {
            int threads = num_threads[t_idx];

            std::vector<double> serial_times(NUM_RUNS);
            std::vector<double> parallel1_times(NUM_RUNS);
            std::vector<double> parallel2_times(NUM_RUNS);

            for (int run = 0; run < NUM_RUNS; run++)
            {
                serial_times[run] = run_single_measurement(solve_serial, N, 1);
                parallel1_times[run] = run_single_measurement(solve_parallel_1, N, threads);
                parallel2_times[run] = run_single_measurement(solve_parallel_2, N, threads);

                iter_file << threads << "," << (run + 1) << ","
                          << serial_times[run] << ","
                          << parallel1_times[run] << ","
                          << parallel2_times[run] << "\n";
            }

            double avg_serial = std::accumulate(serial_times.begin(), serial_times.end(), 0.0) / NUM_RUNS;
            double avg_parallel1 = std::accumulate(parallel1_times.begin(), parallel1_times.end(), 0.0) / NUM_RUNS;
            double avg_parallel2 = std::accumulate(parallel2_times.begin(), parallel2_times.end(), 0.0) / NUM_RUNS;

            double speedup1 = avg_serial / avg_parallel1;
            double speedup2 = avg_serial / avg_parallel2;

            summ_file << threads << ","
                      << avg_serial << ","
                      << avg_parallel1 << ","
                      << avg_parallel2 << ","
                      << speedup1 << ","
                      << speedup2 << "\n";
        }
        std::cout << "✓ Файлы iterationData.csv и summary.csv созданы\n";
    }

    std::ofstream iter_sc("iterationData_sc.csv");
    std::ofstream summ_sc("summary_sc.csv");

    if (iter_sc && summ_sc)
    {
        iter_sc << "config_description,run,time_serial,time_parallel\n";
        summ_sc << "config_description,time_serial,time_parallel,speedup\n";

        for (const auto &cfg : configs)
        {
            omp_set_schedule(cfg.kind, cfg.chunk);

            std::vector<double> serial_times(NUM_RUNS);
            std::vector<double> parallel_times(NUM_RUNS);

            for (int run = 0; run < NUM_RUNS; run++)
            {
                serial_times[run] = run_single_measurement(solve_serial, N, 1);
                parallel_times[run] = run_single_measurement(solve_parallel_2, N, num_threads[4]);

                iter_sc << cfg.name << "," << (run + 1) << ","
                        << serial_times[run] << ","
                        << parallel_times[run] << "\n";
            }

            double avg_serial = std::accumulate(serial_times.begin(), serial_times.end(), 0.0) / NUM_RUNS;
            double avg_parallel = std::accumulate(parallel_times.begin(), parallel_times.end(), 0.0) / NUM_RUNS;
            double speedup = avg_serial / avg_parallel;

            summ_sc << cfg.name << ","
                    << avg_serial << ","
                    << avg_parallel << ","
                    << speedup << "\n";
        }
        std::cout << "✓ Файлы iterationData_sc.csv и summary_sc.csv созданы\n";
    }

    std::cout << "\n=== ВЫВОДЫ ===\n";

    auto max_speedup1 = *std::max_element(speedups1.begin(), speedups1.end());
    auto max_speedup2 = *std::max_element(speedups2.begin(), speedups2.end());

    std::cout << "Лучшее ускорение (простое среднее):\n";
    printf("  Алгоритм 1: %.2f\n", max_speedup1);
    printf("  Алгоритм 2: %.2f\n", max_speedup2);

    double efficiency2 = speedups2.back() / num_threads.back() * 100;
    printf("\nЭффективность алгоритма 2 при %d потоках: %.1f%%\n",
           num_threads.back(), efficiency2);

    if (efficiency2 > 80)
        std::cout << "✓ Отличная масштабируемость\n";
    else if (efficiency2 > 50)
        std::cout << "✓ Хорошая масштабируемость\n";
    else
        std::cout << "✓ Удовлетворительная масштабируемость\n";

    return 0;
}