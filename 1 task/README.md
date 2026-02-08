# Вычисление суммы синусов на полном периоде

Проект вычисляет сумму значений функции синуса на одном полном периоде (0-2π) для массива из 10⁷ элементов с возможностью выбора типа данных (float/double) во время сборки.

# Как выбрать тип массива

```
# Для типа FLOAT (по умолчанию):
cmake -S . -B build_float

# Для типа DOUBLE:
cmake -S . -B build_double -DUSE_DOUBLE=ON
```

# Сборка с float:
```
cmake -S . -B build_float
cmake --build build_float
./build_float/sin
```

# Сборка с double:
```
cmake -S . -B build_double -DUSE_DOUBLE=ON
cmake --build build_double
./build_double/sin
```
