# Задание 6 — уравнение теплопроводности (2D, OpenACC)

Стационарное уравнение Лапласа на равномерной сетке: **пятиточечный шаблон**, метод Якоби.  
Углы: **10, 20, 30, 20** (нижний левый → нижний правый → верхний правый → верхний левый).  
Края — линейная интерполяция между углами, внутренность — **0**, тип `double`.

## Требования

- `eps = 1e-6`, `max_iters = 1e6` (по умолчанию, задаются из CLI)
- Параметры: **boost::program_options**
- Сборка: **pgc++/nvc**++, флаги `**-acc`**, `**-Minfo=all**`
- Режимы: `**-acc=host**`, `**-acc=multicore**`, `**-acc=gpu**`
- Профилирование: **Nsight Systems**, `max-iters` **30–100**
- Проверка: печать сетки для **10×10** и **13×13**

## Сборка (Linux / WSL + NVIDIA HPC SDK)

```bash
cd "6 task"
sudo apt install libboost-program-options-dev   # при необходимости

make ACC=host
make ACC=multicore
make ACC=gpu
```

4 режима для отчёта:

1. `CPU-onecore`: `make cpu-onecore` и запуск с `OMP_NUM_THREADS=1 NVCOMPILER_ACC_NUM_CORES=1`
2. `CPU-multicore`: `make cpu-multicore` и запуск с `--optimized`
3. `GPU baseline`: `make gpu-baseline` и запуск без `--optimized`
4. `GPU optimized`: `make gpu-optimized` и запуск с `--optimized`

Или CMake:

```bash
cmake -S . -B build -DACC_MODE=gpu
cmake --build build -j
./build/bin/heat_conduction --help
```

## Запуск

```bash
./build/bin/heat_conduction --size 256 --eps 1e-6 --max-iters 1000000
./build/bin/heat_conduction --size 256 --optimized          # версия после оптимизации
./build/bin/heat_conduction --size 10                       # печать 10×10
./scripts/verify_small.sh ./build/bin/heat_conduction       # 10×10 и 13×13 для отчёта
```

Вывод:

- `iterations=` — число итераций  
- `error=` — достигнутая невязка (max |u_new − u_old| на внутренности)  
- `time_sec=` — время решения

Разбор `**-Minfo=all**`: ищите строки `Generating acc...`, `loop` / `gang` / `vector`, предупреждения о data transfer.

## Бенчмарк и графики

```bash
chmod +x scripts/*.sh
./scripts/benchmark.sh          # CSV: host, multicore, gpu × baseline/optimized
pip install -r requirements_plot.txt
python3 plot_benchmark.py       # plots/time_*.png, plots/speedup_all.png
```

## Nsight Systems

```bash
./scripts/profile_nsight.sh build/heat_conduction 50 host
./scripts/profile_nsight.sh build/heat_conduction 80 gpu
nsys-ui profile_host_n256_it50.nsys-rep
```

## Отчёт

См. [Report.md](Report.md) — постановка, анализ `-Minfo`, ответы на вопросы, места для скриншотов сеток 10×10 / 13×13.