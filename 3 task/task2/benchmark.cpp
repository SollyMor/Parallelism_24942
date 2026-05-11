#include "server.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct PowArgs {
    double base;
    int exponent;
};

template <typename Server>
void workload(Server& server, int n) {
    std::mt19937 rng(999U);
    std::uniform_real_distribution<double> angle(-50.0, 50.0);
    std::uniform_real_distribution<double> value(0.0, 5000.0);
    std::uniform_real_distribution<double> base(0.1, 20.0);
    std::uniform_int_distribution<int> expv(-3, 6);

    for (int i = 0; i < n; ++i) {
        const double x1 = angle(rng);
        server.request_result(server.add_task([x1]() { return std::sin(x1); }));
        const double x2 = value(rng);
        server.request_result(server.add_task([x2]() { return std::sqrt(x2); }));
        const PowArgs a{base(rng), expv(rng)};
        server.request_result(server.add_task([a]() { return std::pow(a.base, a.exponent); }));
    }
}

template <typename Server>
double measure_seconds(int n_per_kind, int repeats) {
    using clock = std::chrono::steady_clock;
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));

    for (int r = 0; r < repeats; ++r) {
        Server server;
        server.start();
        const auto t0 = clock::now();
        workload(server, n_per_kind);
        const auto t1 = clock::now();
        server.stop();
        samples.push_back(std::chrono::duration<double>(t1 - t0).count());
    }

    double sum = 0.0;
    for (double s : samples) {
        sum += s;
    }
    return sum / static_cast<double>(samples.size());
}

struct Row {
    const char* delivery;
    const char* container;
    double seconds;
};

}  // namespace

int main(int argc, char* argv[]) {
    int n_per_kind = 200;
    int repeats = 5;
    if (argc > 1) {
        n_per_kind = std::stoi(argv[1]);
    }
    if (argc > 2) {
        repeats = std::stoi(argv[2]);
    }
    if (n_per_kind < 1 || repeats < 1) {
        std::cerr << "Usage: " << argv[0] << " [n_per_client_kind] [repeats]\n";
        return 2;
    }

    std::vector<Row> rows;
    rows.push_back({"slot", "unordered_map", measure_seconds<TaskServerSlotUnordered<double>>(n_per_kind, repeats)});
    rows.push_back({"slot", "std::map", measure_seconds<TaskServerSlotOrdered<double>>(n_per_kind, repeats)});
    rows.push_back(
        {"promise", "unordered_map", measure_seconds<TaskServerPromiseUnordered<double>>(n_per_kind, repeats)});
    rows.push_back({"promise", "std::map", measure_seconds<TaskServerPromiseOrdered<double>>(n_per_kind, repeats)});

    const char* out_name = "task2_benchmark_results.csv";
    if (argc > 3) {
        out_name = argv[3];
    }

    std::ofstream out(out_name);
    if (!out) {
        std::cerr << "Cannot write " << out_name << "\n";
        return 1;
    }
    out << "delivery,container,seconds_per_run_mean,n_per_kind,repeats\n";
    out.setf(std::ios::fixed);
    out << std::setprecision(6);
    for (const auto& row : rows) {
        out << row.delivery << "," << row.container << "," << row.seconds << "," << n_per_kind << "," << repeats << "\n";
    }

    std::size_t best = 0;
    for (std::size_t i = 1; i < rows.size(); ++i) {
        if (rows[i].seconds < rows[best].seconds) {
            best = i;
        }
    }

    std::cout << "Wrote " << out_name << "\n";
    std::cout << "Fastest (mean of " << repeats << " runs): " << rows[best].delivery << " + " << rows[best].container
              << " -> " << rows[best].seconds << " s\n";
    std::cout << "Use matching flag in task2_client_server: ";
    const bool is_slot = std::string(rows[best].delivery) == "slot";
    const bool is_unordered = std::string(rows[best].container) == "unordered_map";
    if (is_slot) {
        std::cout << (is_unordered ? "slot-u\n" : "slot-o\n");
    } else {
        std::cout << (is_unordered ? "promise-u\n" : "promise-o\n");
    }

    return 0;
}

