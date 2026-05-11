# Task 2 (кратко)

Основной текст по Task 2 находится в общем отчете: `../Report.md`.

Флаги реализации:

| Флаг | Доставка | Контейнер по `id` |
|------|----------|-------------------|
| `slot-u` | mutex + cv | `unordered_map` |
| `slot-o` | mutex + cv | `std::map` |
| `promise-u` | promise / shared_future | `unordered_map` |
| `promise-o` | promise / shared_future | `std::map` |

