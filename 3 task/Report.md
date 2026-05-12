# Лабораторная работа №3 (адаптированная версия)

Этот каталог приведен к структуре, похожей на `lab3`, с сохранением ключевого функционала:

- `task1` — DGEMV на `std::thread` (последовательная/параллельная версии, верификация, CSV, графики).
- `task2` — клиент-сервер с очередью задач и 4 реализациями доставки результата:
  - `slot-u`, `slot-o`, `promise-u`, `promise-o`;
  - генерация клиентских CSV;
  - проверка корректности;
  - бенчмарк и график сравнения.

## Сборка

```bash
cd "3 task"
cmake -S . -B build
cmake --build build -j
```

## Task 1

- Исходники: `task1/task1.cpp`
- Усреднение прогонов: `task1/run_task1_average.py`
- Графики: `task1/plot_speedup_task1.py`

Пример запуска:

```bash
./build/task1/task1_stdthread --sizes 20000,40000 --threads 1,2,4,7,8,16,20,40 --repeats 5 --drop-max 1
python3 task1/plot_speedup_task1.py
```

## Task 2

- Основной запуск клиентов: `task2/main.cpp` (`task2_client_server`)
- Проверка: `task2/verify_results.cpp` (`task2_verify`)
- Бенчмарк: `task2/benchmark.cpp` (`task2_benchmark`)
- График бенчмарка: `task2/plot_banchmark.py`

Пример запуска:

```bash
./build/task2/task2_benchmark 200 5 task2_benchmark_results.csv
python3 task2/plot_banchmark.py --csv task2_benchmark_results.csv --out server_benchmark_task2.png
./build/task2/task2_client_server 500 promise-o
./build/task2/task2_verify
```

