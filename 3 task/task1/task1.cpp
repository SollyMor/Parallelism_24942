#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double seconds_now() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

std::vector<int> parse_int_list(const std::string& text) {
    std::vector<int> values;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            values.push_back(std::stoi(token));
        }
    }
    return values;
}

std::vector<int> make_range(int from, int to_inclusive) {
    std::vector<int> values;
    if (from > to_inclusive) {
        return values;
    }
    values.reserve(static_cast<std::size_t>(to_inclusive - from + 1));
    for (int v = from; v <= to_inclusive; ++v) {
        values.push_back(v);
    }
    return values;
}

double mean_drop_max(std::vector<double> samples, int drop_max) {
    if (samples.empty()) {
        return -1.0;
    }
    std::sort(samples.begin(), samples.end());
    if (drop_max > 0 && static_cast<int>(samples.size()) > drop_max) {
        samples.resize(samples.size() - static_cast<std::size_t>(drop_max));
    }
    const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    return sum / static_cast<double>(samples.size());
}

void initialize_parallel(std::vector<double>& a, std::vector<double>& b, std::size_t n, int threads) {
    const std::size_t rows_per_thread = (n + static_cast<std::size_t>(threads) - 1U) / static_cast<std::size_t>(threads);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));

    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&, t]() {
            const std::size_t begin = static_cast<std::size_t>(t) * rows_per_thread;
            const std::size_t end = std::min(n, begin + rows_per_thread);
            for (std::size_t i = begin; i < end; ++i) {
                const std::size_t row_offset = i * n;
                for (std::size_t j = 0; j < n; ++j) {
                    a[row_offset + j] = static_cast<double>(i + j);
                }
            }
        });
    }
    for (auto& w : workers) {
        w.join();
    }

    workers.clear();
    const std::size_t cols_per_thread = (n + static_cast<std::size_t>(threads) - 1U) / static_cast<std::size_t>(threads);
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&, t]() {
            const std::size_t begin = static_cast<std::size_t>(t) * cols_per_thread;
            const std::size_t end = std::min(n, begin + cols_per_thread);
            for (std::size_t j = begin; j < end; ++j) {
                b[j] = static_cast<double>(j);
            }
        });
    }
    for (auto& w : workers) {
        w.join();
    }
}

void matrix_vector_serial(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& c, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t row_offset = i * n;
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            sum += a[row_offset + j] * b[j];
        }
        c[i] = sum;
    }
}

void matrix_vector_parallel(
    const std::vector<double>& a,
    const std::vector<double>& b,
    std::vector<double>& c,
    std::size_t n,
    int threads
) {
    const std::size_t rows_per_thread = (n + static_cast<std::size_t>(threads) - 1U) / static_cast<std::size_t>(threads);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));

    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&, t]() {
            const std::size_t begin = static_cast<std::size_t>(t) * rows_per_thread;
            const std::size_t end = std::min(n, begin + rows_per_thread);
            for (std::size_t i = begin; i < end; ++i) {
                const std::size_t row_offset = i * n;
                double sum = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    sum += a[row_offset + j] * b[j];
                }
                c[i] = sum;
            }
        });
    }
    for (auto& w : workers) {
        w.join();
    }
}

struct ResultRow {
    int threads{};
    double t_serial{};
    double t_parallel{};
    double speedup{};
    double efficiency{};
};

bool verify_equal_vectors(
    const std::vector<double>& ref,
    const std::vector<double>& got,
    double abs_tol,
    double rel_tol,
    double& out_max_abs,
    double& out_max_rel
) {
    if (ref.size() != got.size()) {
        return false;
    }
    out_max_abs = 0.0;
    out_max_rel = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double diff = std::abs(ref[i] - got[i]);
        const double scale = std::max(std::abs(ref[i]), std::abs(got[i]));
        const double rel = scale > 0.0 ? diff / scale : diff;
        out_max_abs = std::max(out_max_abs, diff);
        out_max_rel = std::max(out_max_rel, rel);
        if (diff > abs_tol && rel > rel_tol) {
            return false;
        }
    }
    return true;
}

std::vector<ResultRow> benchmark_size(
    std::size_t n,
    const std::vector<int>& threads_list,
    int repeats,
    int drop_max,
    bool verify_results
) {
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> c;
    try {
        a.resize(n * n);
        b.resize(n);
        c.resize(n);
    } catch (const std::bad_alloc&) {
        std::cerr << "ERROR: not enough memory for N=" << n << "\n";
        return {};
    }

    const int init_threads = std::max(1, threads_list.empty() ? 1 : *std::max_element(threads_list.begin(), threads_list.end()));
    initialize_parallel(a, b, n, init_threads);

    std::vector<double> c_serial(n, 0.0);
    std::vector<double> c_parallel(n, 0.0);
    std::vector<double> serial_samples;
    serial_samples.reserve(static_cast<std::size_t>(repeats));
    for (int r = 0; r < repeats; ++r) {
        const double t0 = seconds_now();
        matrix_vector_serial(a, b, c_serial, n);
        serial_samples.push_back(seconds_now() - t0);
    }
    const double t_serial = mean_drop_max(serial_samples, drop_max);

    std::vector<ResultRow> rows;
    rows.reserve(threads_list.size());

    for (int threads : threads_list) {
        std::vector<double> par_samples;
        par_samples.reserve(static_cast<std::size_t>(repeats));
        if (verify_results) {
            matrix_vector_parallel(a, b, c_parallel, n, threads);
            double max_abs = 0.0;
            double max_rel = 0.0;
            const bool ok = verify_equal_vectors(c_serial, c_parallel, 1e-9, 1e-12, max_abs, max_rel);
            if (!ok) {
                std::cerr << "VERIFY FAILED: N=" << n << ", threads=" << threads
                          << ", max_abs=" << max_abs << ", max_rel=" << max_rel << "\n";
                return {};
            }
            std::cout << "VERIFY OK: N=" << n << ", threads=" << threads
                      << ", max_abs=" << max_abs << ", max_rel=" << max_rel << "\n";
        }
        for (int r = 0; r < repeats; ++r) {
            const double t0 = seconds_now();
            matrix_vector_parallel(a, b, c_parallel, n, threads);
            par_samples.push_back(seconds_now() - t0);
        }

        const double t_parallel = mean_drop_max(par_samples, drop_max);
        const double speedup = t_parallel > 0.0 ? (t_serial / t_parallel) : 0.0;
        const double efficiency = threads > 0 ? (speedup / static_cast<double>(threads)) : 0.0;
        rows.push_back(ResultRow{threads, t_serial, t_parallel, speedup, efficiency});
    }

    volatile double checksum = 0.0;
    for (double v : c_parallel) {
        checksum += v;
    }
    std::cout << "Checksum (N=" << n << "): " << checksum << "\n";
    return rows;
}

bool write_csv(const std::string& file_path, const std::vector<ResultRow>& rows) {
    std::ofstream out(file_path);
    if (!out) {
        return false;
    }
    out << "threads,T_serial,T_parallel,speedup,efficiency\n";
    out << std::fixed << std::setprecision(9);
    for (const auto& row : rows) {
        out << row.threads << "," << row.t_serial << "," << row.t_parallel << "," << row.speedup << "," << row.efficiency << "\n";
    }
    return true;
}

void print_table(std::size_t n, const std::vector<ResultRow>& rows) {
    std::cout << "\n=== N=" << n << " ===\n";
    std::cout << std::setw(8) << "threads"
              << std::setw(15) << "T_serial"
              << std::setw(15) << "T_parallel"
              << std::setw(12) << "speedup"
              << std::setw(12) << "eff\n";
    for (const auto& row : rows) {
        std::cout << std::setw(8) << row.threads
                  << std::setw(15) << row.t_serial
                  << std::setw(15) << row.t_parallel
                  << std::setw(12) << row.speedup
                  << std::setw(12) << row.efficiency << "\n";
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    std::vector<int> sizes = {20000, 40000};
    std::vector<int> threads_list = {1, 2, 4, 7, 8, 16, 20, 40};
    int repeats = 5;
    int drop_max = 1;
    std::string out_prefix = "speedup_dgemv_stdthread";
    bool verify_results = true;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--sizes" && i + 1 < argc) {
            sizes = parse_int_list(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads_list = parse_int_list(argv[++i]);
        } else if (arg == "--repeats" && i + 1 < argc) {
            repeats = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--drop-max" && i + 1 < argc) {
            drop_max = std::max(0, std::stoi(argv[++i]));
        } else if (arg == "--out-prefix" && i + 1 < argc) {
            out_prefix = argv[++i];
        } else if (arg == "--no-verify") {
            verify_results = false;
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            std::cerr << "Usage: " << argv[0]
                      << " [--sizes 20000,40000]"
                      << " [--threads 1,2,...,40]"
                      << " [--repeats 5]"
                      << " [--drop-max 1]"
                      << " [--out-prefix speedup_dgemv_stdthread]"
                      << " [--no-verify]\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(9);
    for (int n : sizes) {
        if (n <= 0) {
            std::cerr << "Skip invalid size: " << n << "\n";
            continue;
        }
        const auto rows = benchmark_size(static_cast<std::size_t>(n), threads_list, repeats, drop_max, verify_results);
        if (rows.empty()) {
            continue;
        }
        print_table(static_cast<std::size_t>(n), rows);

        const std::string csv_name = out_prefix + "_" + std::to_string(n) + ".csv";
        if (!write_csv(csv_name, rows)) {
            std::cerr << "ERROR: cannot write " << csv_name << "\n";
            return 1;
        }
        std::cout << "Wrote " << csv_name << "\n";

    }

    return 0;
}

