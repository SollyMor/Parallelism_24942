#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{
template <typename T>
class TaskServer
{
public:
  using Task = std::function<T()>;

  TaskServer() = default;

  ~TaskServer()
  {
    stop();
  }

  void start()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_)
    {
      return;
    }
    running_ = true;
    worker_ = std::thread(&TaskServer::run, this);
  }

  void stop()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!running_ && !worker_.joinable())
      {
        return;
      }
      running_ = false;
    }

    tasks_cv_.notify_all();
    if (worker_.joinable())
    {
      worker_.join();
    }
  }

  std::size_t add_task(Task task)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t id = next_id_++;
    tasks_.push_back({id, std::move(task)});
    tasks_cv_.notify_one();
    return id;
  }

  T request_result(std::size_t id)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    results_cv_.wait(lock, [&] { return results_.find(id) != results_.end(); });

    T result = results_.at(id);
    results_.erase(id);
    return result;
  }

private:
  struct TaskItem
  {
    std::size_t id = 0;
    Task task;
  };

  void run()
  {
    while (true)
    {
      TaskItem item;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        tasks_cv_.wait(lock, [&] { return !running_ || !tasks_.empty(); });
        if (!running_ && tasks_.empty())
        {
          break;
        }

        item = std::move(tasks_.front());
        tasks_.pop_front();
      }

      const T value = item.task();

      {
        std::lock_guard<std::mutex> lock(mutex_);
        results_.emplace(item.id, value);
      }
      results_cv_.notify_all();
    }
  }

  std::mutex mutex_;
  std::condition_variable tasks_cv_;
  std::condition_variable results_cv_;
  std::deque<TaskItem> tasks_;
  std::unordered_map<std::size_t, T> results_;
  std::thread worker_;
  std::size_t next_id_ = 1;
  bool running_ = false;
};

struct TaskRecord
{
  std::size_t id = 0;
  std::string type;
  double x = 0.0;
  double y = 0.0;
  double result = 0.0;
};

void write_header(std::ofstream &file)
{
  file << "id,type,x,y,result\n";
}

void write_record(std::ofstream &file, const TaskRecord &record)
{
  file << record.id << "," << record.type << ","
       << std::setprecision(17) << record.x << ","
       << record.y << "," << record.result << "\n";
}

void client_sin(TaskServer<double> &server, int count)
{
  std::ofstream file("client_sin_results.csv");
  write_header(file);

  std::mt19937 gen(1001);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);

  for (int i = 0; i < count; ++i)
  {
    const double x = dist(gen);
    const std::size_t id = server.add_task([x] { return std::sin(x); });
    const double result = server.request_result(id);
    write_record(file, {id, "sin", x, 0.0, result});
  }
}

void client_sqrt(TaskServer<double> &server, int count)
{
  std::ofstream file("client_sqrt_results.csv");
  write_header(file);

  std::mt19937 gen(2002);
  std::uniform_real_distribution<double> dist(0.0, 100000.0);

  for (int i = 0; i < count; ++i)
  {
    const double x = dist(gen);
    const std::size_t id = server.add_task([x] { return std::sqrt(x); });
    const double result = server.request_result(id);
    write_record(file, {id, "sqrt", x, 0.0, result});
  }
}

void client_pow(TaskServer<double> &server, int count)
{
  std::ofstream file("client_pow_results.csv");
  write_header(file);

  std::mt19937 gen(3003);
  std::uniform_real_distribution<double> base_dist(0.1, 10.0);
  std::uniform_real_distribution<double> exp_dist(0.5, 5.0);

  for (int i = 0; i < count; ++i)
  {
    const double x = base_dist(gen);
    const double y = exp_dist(gen);
    const std::size_t id = server.add_task([x, y] { return std::pow(x, y); });
    const double result = server.request_result(id);
    write_record(file, {id, "pow", x, y, result});
  }
}
} // namespace

int main(int argc, char **argv)
{
  const int tasks_per_client = (argc > 1) ? std::stoi(argv[1]) : 1000;
  if (tasks_per_client <= 5 || tasks_per_client >= 10000)
  {
    std::cerr << "N must be in range 5 < N < 10000\n";
    return 1;
  }

  TaskServer<double> server;
  server.start();

  std::thread sin_client(client_sin, std::ref(server), tasks_per_client);
  std::thread sqrt_client(client_sqrt, std::ref(server), tasks_per_client);
  std::thread pow_client(client_pow, std::ref(server), tasks_per_client);

  sin_client.join();
  sqrt_client.join();
  pow_client.join();

  server.stop();

  std::cout << "Saved files:\n";
  std::cout << "  client_sin_results.csv\n";
  std::cout << "  client_sqrt_results.csv\n";
  std::cout << "  client_pow_results.csv\n";
  return 0;
}
