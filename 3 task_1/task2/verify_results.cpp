#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> parts;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        parts.push_back(token);
    }
    return parts;
}

bool nearly_equal(double a, double b, double abs_tol = 1e-9, double rel_tol = 1e-10) {
    const double diff = std::abs(a - b);
    if (diff <= abs_tol) {
        return true;
    }
    const double scale = std::max(std::abs(a), std::abs(b));
    return diff <= rel_tol * scale;
}

int verify_sin_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Cannot open " << path << "\n";
        return 1;
    }
    std::string line;
    std::getline(in, line);
    int checked = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto parts = split_csv_line(line);
        if (parts.size() != 4) {
            std::cerr << "Bad format in " << path << ": " << line << "\n";
            return 1;
        }
        const double x = std::stod(parts[2]);
        const double got = std::stod(parts[3]);
        const double expected = std::sin(x);
        if (!nearly_equal(got, expected)) {
            std::cerr << "Mismatch in " << path << ", line: " << line << "\n";
            return 1;
        }
        ++checked;
    }
    std::cout << path << " verified, rows: " << checked << "\n";
    return 0;
}

int verify_sqrt_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Cannot open " << path << "\n";
        return 1;
    }
    std::string line;
    std::getline(in, line);
    int checked = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto parts = split_csv_line(line);
        if (parts.size() != 4) {
            std::cerr << "Bad format in " << path << ": " << line << "\n";
            return 1;
        }
        const double x = std::stod(parts[2]);
        const double got = std::stod(parts[3]);
        const double expected = std::sqrt(x);
        if (!nearly_equal(got, expected)) {
            std::cerr << "Mismatch in " << path << ", line: " << line << "\n";
            return 1;
        }
        ++checked;
    }
    std::cout << path << " verified, rows: " << checked << "\n";
    return 0;
}

int verify_pow_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Cannot open " << path << "\n";
        return 1;
    }
    std::string line;
    std::getline(in, line);
    int checked = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto parts = split_csv_line(line);
        if (parts.size() != 5) {
            std::cerr << "Bad format in " << path << ": " << line << "\n";
            return 1;
        }
        const double base = std::stod(parts[2]);
        const int exp = std::stoi(parts[3]);
        const double got = std::stod(parts[4]);
        const double expected = std::pow(base, exp);
        if (!nearly_equal(got, expected, 1e-8, 1e-9)) {
            std::cerr << "Mismatch in " << path << ", line: " << line << "\n";
            return 1;
        }
        ++checked;
    }
    std::cout << path << " verified, rows: " << checked << "\n";
    return 0;
}

}  // namespace

int main() {
    int rc = 0;
    rc |= verify_sin_file("client_sin.csv");
    rc |= verify_sqrt_file("client_sqrt.csv");
    rc |= verify_pow_file("client_pow.csv");

    if (rc == 0) {
        std::cout << "All result files are valid.\n";
    }
    return rc;
}

