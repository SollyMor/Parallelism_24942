#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
    // Параметры по умолчанию
    int grid_size = 128;
    double precision = 1e-6;
    int max_iterations = 1000000;

    // Настройка параметров командной строки
    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "Show help")
        ("size,s", po::value<int>(&grid_size)->default_value(128), "Grid size (N for NxN)")
        ("precision,p", po::value<double>(&precision)->default_value(1e-6), "Target precision (error)")
        ("iterations,i", po::value<int>(&max_iterations)->default_value(1000000), "Maximum iterations");

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

    // Инициализация сетки и граничных условий
    // Углы: [0][0]=10, [0][N-1]=20, [N-1][N-1]=30, [N-1][0]=20 (по часовой стрелке, начиная с левого верхнего)
    const double corner_tl = 10.0;
    const double corner_tr = 20.0;
    const double corner_br = 30.0;
    const double corner_bl = 20.0;

    int N = grid_size;
    // Используем одномерный массив для лучшего кэширования (flat array)
    std::vector<double> grid(N * N, 0.0);
    std::vector<double> new_grid(N * N, 0.0);

    // Установка граничных условий линейной интерполяцией между углами
    for (int i = 0; i < N; ++i) {
        // Левая граница: между (0,0)=10 и (N-1,0)=20
        double t = static_cast<double>(i) / (N - 1);
        grid[i * N + 0] = corner_tl + t * (corner_bl - corner_tl);
        // Правая граница: между (0,N-1)=20 и (N-1,N-1)=30
        grid[i * N + (N - 1)] = corner_tr + t * (corner_br - corner_tr);
    }
    for (int j = 0; j < N; ++j) {
        // Верхняя граница: между (0,0)=10 и (0,N-1)=20
        double t = static_cast<double>(j) / (N - 1);
        grid[0 * N + j] = corner_tl + t * (corner_tr - corner_tl);
        // Нижняя граница: между (N-1,0)=20 и (N-1,N-1)=30
        grid[(N - 1) * N + j] = corner_bl + t * (corner_br - corner_bl);
    }

    // Копируем граничные условия в new_grid (они не меняются)
    new_grid = grid;

    int iteration = 0;
    double max_error = precision + 1.0;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Основной цикл метода Якоби
    while (iteration < max_iterations && max_error > precision) {
        max_error = 0.0;

        // Обработка внутренних точек (i от 1 до N-2, j от 1 до N-2)
        #pragma GCC ivdep // Подсказка компилятору об отсутствии зависимостей (для GCC)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                int idx = i * N + j;
                // Пятиточечный шаблон: среднее арифметическое соседей
                new_grid[idx] = 0.25 * (grid[idx - 1] + grid[idx + 1] + grid[idx - N] + grid[idx + N]);
                
                double error = std::fabs(new_grid[idx] - grid[idx]);
                if (error > max_error) {
                    max_error = error;
                }
            }
        }

        // Обновляем сетку для следующей итерации
        std::swap(grid, new_grid);
        ++iteration;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Вывод результатов
    std::cout << "Grid size: " << N << "x" << N << std::endl;
    std::cout << "Iterations: " << iteration << std::endl;
    std::cout << "Final error: " << max_error << std::endl;
    std::cout << "Time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}