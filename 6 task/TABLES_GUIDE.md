# Как заполнить таблицы отчёта (задание 6)

Замеры делаются на **Linux / WSL** с **NVIDIA HPC SDK** (`nvc++`). На Windows без WSL программа не соберётся с OpenACC.

## Соответствие таблиц и сборки

| Таблица в отчёте | Сборка | Запуск | Что писать в столбцы |
|------------------|--------|--------|----------------------|
| **CPU-onecore** | `make ACC=host` | без `--optimized` | **Время** → `time_sec`, **Точность** → `error`, **Итерации** → `iterations` |
| **CPU-multicore** | `make ACC=multicore` | `--optimized` | то же |
| **GPU — оптимизированный** | `make ACC=gpu` | `--optimized` | то же |

Перед onecore ограничьте потоки:

```bash
export OMP_NUM_THREADS=1
export NVCOMPILER_ACC_NUM_CORES=1
```

Параметры по заданию: `--eps 1e-6 --max-iters 1000000` (это значения по умолчанию).

## Автоматический прогон

```bash
cd "6 task"
chmod +x scripts/fill_report_tables.sh
./scripts/fill_report_tables.sh
```

Результат: `report_tables.txt` (готовые строки для Markdown) и `report_tables.csv`.

## Ручной запуск (одна ячейка таблицы)

Пример для сетки **256×256**, CPU-multicore:

```bash
make ACC=multicore
./build/bin/heat_conduction -s 256 --optimized -e 1e-6 -m 1000000
```

Пример вывода:

```text
mode=optimized
grid=256x256
iterations=…
error=…          ← столбец «Точность»
time_sec=…       ← столбец «Время выполнения»
```

Число **итераций** для одной и той же сетки и `eps=1e-6` должно совпадать во всех режимах (меняется только время).

## Таблица «Этапы оптимизации на 512×512»

| Этап | Сборка | Флаг | Комментарий для отчёта |
|------|--------|------|------------------------|
| 1 | `ACC=host`, 1 поток | без `-o` | Baseline: один `acc data`, невязка в ядре, без tile |
| 2 | `ACC=multicore` | без `-o` | Multicore baseline |
| 3 | `ACC=multicore` | `--optimized` | `copyin`/`create`, `tile(32,32)` |
| 4 | `ACC=gpu` | `--optimized` | GPU, оптимизированный вариант |

**max_iters** в таблице: **1_000_000** (как в шаблоне).

## Проверка маленьких сеток (10×10, 13×13)

```bash
make ACC=host
./build/bin/heat_conduction -s 10
./build/bin/heat_conduction -s 13
```

Скриншоты углов 10/20/30/20 — для раздела проверки, не для таблиц времени.

## Если нет GPU

Таблицу **GPU** заполняют на машине кафедры / в WSL с драйвером NVIDIA. Локально можно оставить пустой и указать в отчёте: «замер на стенде …».

## Зависимости

```bash
sudo apt install libboost-program-options-dev   # Ubuntu/WSL
```
