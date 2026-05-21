# Task 4

В папке лежат два учебных варианта задания:

- `main.py` - вариант с датчиками `SensorCam` и `SensorX`;
- `pose_video.py` - обработка видео моделью `yolov8s-pose`;
- `pose_realtime.py` - дополнительный realtime-режим с камеры.

## Установка

```bash
python -m pip install -r requirements.txt
```

## Инференс видео

```bash
python pose_video.py --video input.mp4 --mode single --output output_single.mp4
```

Многопоточный режим:

```bash
python pose_video.py --video input.mp4 --mode thread --workers 4 --output output_thread.mp4
```

Многопроцессный режим:

```bash
python pose_video.py --video input.mp4 --mode process --workers 4 --output output_process.mp4
```

Подбор количества потоков или процессов:

```bash
python pose_video.py --video input.mp4 --mode process --workers 8 --benchmark
```

Скрипт выводит время обработки всех кадров и сохраняет видео с keypoints.
В многопоточном и многопроцессном режимах кадры сначала попадают во входной
буфер, потом обрабатываются рабочими потоками/процессами, возвращаются в
выходной буфер и записываются в исходном порядке по номеру кадра.

## Realtime с камеры

```bash
python pose_realtime.py --camera 0 --width 640 --height 480 --workers 2
```

Выход из окна - клавиша `q`.

## Старый вариант с датчиками

```bash
python main.py --camera 0 --resolution 1280x720 --fps 30
```
