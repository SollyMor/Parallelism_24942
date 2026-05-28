#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
    int grid_size = 128;
    double precision = 1e-6;
    int max_iterations = 1000000;
    int num_threads = omp_get_max_threads();

    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "Show help")
        ("size,s", po::value<int>(&grid_size)->default_value(128), "Grid size (N for NxN)")
        ("precision,p", po::value<double>(&precision)->default_value(1e-6), "Target precision (error)")
        ("iterations,i", po::value<int>(&max_iterations)->default_value(1000000), "Maximum iterations")
        ("threads,t", po::value<int>(&num_threads)->default_value(omp_get_max_threads()), "Number of threads");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << desc << "\n";
        return 1;
    }

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    omp_set_num_threads(num_threads);
    std::cout << "Using " << num_threads << " threads" << std::endl;

    const double corner_tl = 10.0;
    const double corner_tr = 20.0;
    const double corner_br = 30.0;
    const double corner_bl = 20.0;

    int N = grid_size;
    std::vector<double> grid(N * N, 0.0);
    std::vector<double> new_grid(N * N, 0.0);

    // Установка граничных условий
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double t = static_cast<double>(i) / (N - 1);
        grid[i * N + 0] = corner_tl + t * (corner_bl - corner_tl);
        grid[i * N + (N - 1)] = corner_tr + t * (corner_br - corner_tr);
    }
    
    #pragma omp parallel for
    for (int j = 0; j < N; ++j) {
        double t = static_cast<double>(j) / (N - 1);
        grid[0 * N + j] = corner_tl + t * (corner_tr - corner_tl);
        grid[(N - 1) * N + j] = corner_bl + t * (corner_br - corner_bl);
    }

    #pragma omp parallel for
    for (int i = 0; i < N * N; ++i) {
        new_grid[i] = grid[i];
    }

    int iteration = 0;
    double max_error = precision + 1.0;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Основной цикл с оптимизированным распараллеливанием
    while (iteration < max_iterations && max_error > precision) {
        max_error = 0.0;

        // Объединённый параллельный регион
        #pragma omp parallel
        {
            double local_max_error = 0.0;
            
            // Вычисление новых значений
            #pragma omp for schedule(static) nowait
            for (int i = 1; i < N - 1; ++i) {
                #pragma omp simd
                for (int j = 1; j < N - 1; ++j) {
                    int idx = i * N + j;
                    new_grid[idx] = 0.25 * (grid[idx - 1] + grid[idx + 1] + grid[idx - N] + grid[idx + N]);
                    
                    double error = std::fabs(new_grid[idx] - grid[idx]);
                    if (error > local_max_error) {
                        local_max_error = error;
                    }
                }
            }
            
            // Обновление глобальной ошибки
            #pragma omp critical
            {
                if (local_max_error > max_error) {
                    max_error = local_max_error;
                }
            }
            
            // Копирование new_grid в grid
            #pragma omp for schedule(static)
            for (int i = 1; i < N - 1; ++i) {
                for (int j = 1; j < N - 1; ++j) {
                    int idx = i * N + j;
                    grid[idx] = new_grid[idx];
                }
            }
        }
        
        ++iteration;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Grid size: " << N << "x" << N << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "Iterations: " << iteration << std::endl;
    std::cout << "Final error: " << max_error << std::endl;
    std::cout << "Time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}