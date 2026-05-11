#include "server.hpp"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

struct PowArgs {
    double base;
    int exponent;
};

template <typename Server>
void run_client_sin(Server& server, int n, const std::string& file_name, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> angle_dist(-100.0, 100.0);

    std::ofstream out(file_name);
    if (!out) {
        throw std::runtime_error("cannot open " + file_name);
    }
    out << std::fixed << std::setprecision(12);
    out << "index,id,x,sin(x)\n";

    for (int i = 0; i < n; ++i) {
        const double x = angle_dist(rng);
        const std::size_t id = server.add_task([x]() { return std::sin(x); });
        const double result = server.request_result(id);
        out << i << "," << id << "," << x << "," << result << "\n";
    }
}

template <typename Server>
void run_client_sqrt(Server& server, int n, const std::string& file_name, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> value_dist(0.0, 10000.0);

    std::ofstream out(file_name);
    if (!out) {
        throw std::runtime_error("cannot open " + file_name);
    }
    out << std::fixed << std::setprecision(12);
    out << "index,id,x,sqrt(x)\n";

    for (int i = 0; i < n; ++i) {
        const double x = value_dist(rng);
        const std::size_t id = server.add_task([x]() { return std::sqrt(x); });
        const double result = server.request_result(id);
        out << i << "," << id << "," << x << "," << result << "\n";
    }
}

template <typename Server>
void run_client_pow(Server& server, int n, const std::string& file_name, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> base_dist(0.1, 25.0);
    std::uniform_int_distribution<int> exp_dist(-5, 8);

    std::ofstream out(file_name);
    if (!out) {
        throw std::runtime_error("cannot open " + file_name);
    }
    out << std::fixed << std::setprecision(12);
    out << "index,id,base,exp,pow(base,exp)\n";

    for (int i = 0; i < n; ++i) {
        const PowArgs args{base_dist(rng), exp_dist(rng)};
        const std::size_t id = server.add_task([args]() { return std::pow(args.base, args.exponent); });
        const double result = server.request_result(id);
        out << i << "," << id << "," << args.base << "," << args.exponent << "," << result << "\n";
    }
}

template <typename Server>
void run_all_clients(Server& server, int n) {
    std::thread client_sin(run_client_sin<Server>, std::ref(server), n, "client_sin.csv", 12345U);
    std::thread client_sqrt(run_client_sqrt<Server>, std::ref(server), n, "client_sqrt.csv", 54321U);
    std::thread client_pow(run_client_pow<Server>, std::ref(server), n, "client_pow.csv", 2026U);

    client_sin.join();
    client_sqrt.join();
    client_pow.join();
}

template <typename Server>
void run_with_server(int n) {
    Server server;
    server.start();
    try {
        run_all_clients(server, n);
    } catch (...) {
        server.stop();
        throw;
    }
    server.stop();
}

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <N> [implementation]\n"
              << "  N: number of tasks per client (must satisfy 5 < N < 10000)\n"
              << "  implementation (optional, default promise-u):\n"
              << "    slot-u     mutex+cv + unordered_map\n"
              << "    slot-o     mutex+cv + std::map\n"
              << "    promise-u  promise/shared_future + unordered_map\n"
              << "    promise-o  promise/shared_future + std::map\n"
              << "Run task2_benchmark to compare timings and pick the fastest on your machine.\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    int n = 100;
    if (argc > 1) {
        n = std::stoi(argv[1]);
    }
    if (n <= 5 || n >= 10000) {
        print_usage(argv[0]);
        return 2;
    }

    std::string impl = "promise-u";
    if (argc > 2) {
        impl = argv[2];
    }

    try {
        if (impl == "slot-u") {
            run_with_server<TaskServerSlotUnordered<double>>(n);
        } else if (impl == "slot-o") {
            run_with_server<TaskServerSlotOrdered<double>>(n);
        } else if (impl == "promise-u") {
            run_with_server<TaskServerPromiseUnordered<double>>(n);
        } else if (impl == "promise-o") {
            run_with_server<TaskServerPromiseOrdered<double>>(n);
        } else {
            print_usage(argv[0]);
            return 2;
        }
    } catch (...) {
        std::cerr << "Server run failed.\n";
        return 1;
    }

    std::cout << "Generated files: client_sin.csv, client_sqrt.csv, client_pow.csv\n";
    std::cout << "Implementation: " << impl << "\n";
    return 0;
}

