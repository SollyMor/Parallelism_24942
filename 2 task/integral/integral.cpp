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