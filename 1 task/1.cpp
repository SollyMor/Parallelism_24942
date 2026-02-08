#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef USE_DOUBLE_TYPE
    using value_type = double;
    const char* type_name = "double";
#else
    using value_type = float;
    const char* type_name = "float";
#endif

int main() {
    const size_t N = 10'000'000;
    const value_type two_pi = 2.0 * M_PI;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<value_type> sine_values(N);
    value_type sum = 0.0;
    
    for (size_t i = 0; i < N; ++i) {
        value_type x = (static_cast<value_type>(i) * two_pi) / static_cast<value_type>(N);
        sine_values[i] = std::sin(x);
        sum += sine_values[i];
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Data type: " << type_name << std::endl;
    std::cout << "Sum: " << std::scientific << std::setprecision(10) << sum << std::endl;
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
    
    return 0;
}