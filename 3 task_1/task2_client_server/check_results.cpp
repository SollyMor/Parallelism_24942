#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
struct Row
{
  std::size_t id = 0;
  std::string type;
  double x = 0.0;
  double y = 0.0;
  double result = 0.0;
};

std::vector<std::string> split_csv_line(const std::string &line)
{
  std::vector<std::string> parts;
  std::stringstream stream(line);
  std::string item;
  while (std::getline(stream, item, ','))
  {
    parts.push_back(item);
  }
  return parts;
}

bool parse_row(const std::string &line, Row &row)
{
  const std::vector<std::string> parts = split_csv_line(line);
  if (parts.size() != 5)
  {
    return false;
  }

  row.id = static_cast<std::size_t>(std::stoull(parts[0]));
  row.type = parts[1];
  row.x = std::stod(parts[2]);
  row.y = std::stod(parts[3]);
  row.result = std::stod(parts[4]);
  return true;
}

double expected_value(const Row &row)
{
  if (row.type == "sin")
  {
    return std::sin(row.x);
  }
  if (row.type == "sqrt")
  {
    return std::sqrt(row.x);
  }
  if (row.type == "pow")
  {
    return std::pow(row.x, row.y);
  }

  throw std::runtime_error("Unknown task type: " + row.type);
}

bool check_file(const std::string &path, int &rows_checked)
{
  std::ifstream file(path);
  if (!file)
  {
    std::cerr << "Cannot open " << path << "\n";
    return false;
  }

  std::string line;
  std::getline(file, line);

  bool ok = true;
  while (std::getline(file, line))
  {
    if (line.empty())
    {
      continue;
    }

    Row row;
    if (!parse_row(line, row))
    {
      std::cerr << "Bad CSV line in " << path << ": " << line << "\n";
      ok = false;
      continue;
    }

    const double expected = expected_value(row);
    const double diff = std::abs(expected - row.result);
    const double limit = 1e-10 * std::max(1.0, std::abs(expected));
    if (diff > limit)
    {
      std::cerr << "Mismatch in " << path << ", id=" << row.id
                << ", expected=" << expected
                << ", got=" << row.result << "\n";
      ok = false;
    }

    ++rows_checked;
  }

  return ok;
}
} // namespace

int main()
{
  const std::vector<std::string> files = {
      "client_sin_results.csv",
      "client_sqrt_results.csv",
      "client_pow_results.csv"};

  bool ok = true;
  int rows_checked = 0;
  for (const std::string &file : files)
  {
    ok = check_file(file, rows_checked) && ok;
  }

  if (!ok)
  {
    std::cerr << "Result check failed\n";
    return 1;
  }

  std::cout << "All results are correct. Rows checked: " << rows_checked << "\n";
  return 0;
}
