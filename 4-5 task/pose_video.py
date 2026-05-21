import argparse
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


MODEL_NAME = "yolov8s-pose.pt"


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


class VideoReader:
    def __init__(self, path: str):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        self.info = VideoInfo(
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self._cap.get(cv2.CAP_PROP_FPS)) or 25.0,
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def read(self):
        return self._cap.read()

    def __del__(self):
        cap = getattr(self, "_cap", None)
        if cap is not None:
            cap.release()


class VideoWriter:
    def __init__(self, path: str, info: VideoInfo):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, info.fps, (info.width, info.height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {path}")

    def write(self, frame) -> None:
        self._writer.write(frame)

    def __del__(self):
        writer = getattr(self, "_writer", None)
        if writer is not None:
            writer.release()


def process_frame(model: YOLO, frame):
    result = model(frame, verbose=False, device="cpu")[0]
    return result.plot()


def run_single(video_path: str, output_path: str) -> float:
    reader = VideoReader(video_path)
    writer = VideoWriter(output_path, reader.info)
    model = YOLO(MODEL_NAME)

    start = time.perf_counter()
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        writer.write(process_frame(model, frame))

    return time.perf_counter() - start


def thread_worker(input_queue: queue.Queue, output_queue: queue.Queue) -> None:
    model = YOLO(MODEL_NAME)
    while True:
        item = input_queue.get()
        if item is None:
            break

        frame_id, frame = item
        output_queue.put((frame_id, process_frame(model, frame)))


def process_worker(input_queue: mp.Queue, output_queue: mp.Queue) -> None:
    model = YOLO(MODEL_NAME)
    while True:
        item = input_queue.get()
        if item is None:
            break

        frame_id, frame = item
        output_queue.put((frame_id, process_frame(model, frame)))


def write_ordered_results(output_queue: Any, writer: VideoWriter, total_frames: int) -> None:
    next_frame = 0
    ready_frames = {}

    while next_frame < total_frames:
        frame_id, frame = output_queue.get()
        ready_frames[frame_id] = frame

        while next_frame in ready_frames:
            writer.write(ready_frames.pop(next_frame))
            next_frame += 1


def run_threads(video_path: str, output_path: str, workers_count: int) -> float:
    reader = VideoReader(video_path)
    writer = VideoWriter(output_path, reader.info)
    input_queue: queue.Queue = queue.Queue(maxsize=workers_count * 2)
    output_queue: queue.Queue = queue.Queue()

    workers = [
        threading.Thread(target=thread_worker, args=(input_queue, output_queue))
        for _ in range(workers_count)
    ]

    start = time.perf_counter()
    for worker in workers:
        worker.start()

    frames_read = 0
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        input_queue.put((frames_read, frame))
        frames_read += 1

    for _ in workers:
        input_queue.put(None)

    write_ordered_results(output_queue, writer, frames_read)

    for worker in workers:
        worker.join()

    return time.perf_counter() - start


def run_processes(video_path: str, output_path: str, workers_count: int) -> float:
    reader = VideoReader(video_path)
    writer = VideoWriter(output_path, reader.info)
    input_queue: mp.Queue = mp.Queue(maxsize=workers_count * 2)
    output_queue: mp.Queue = mp.Queue()

    workers = [
        mp.Process(target=process_worker, args=(input_queue, output_queue))
        for _ in range(workers_count)
    ]

    start = time.perf_counter()
    for worker in workers:
        worker.start()

    frames_read = 0
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        input_queue.put((frames_read, frame))
        frames_read += 1

    for _ in workers:
        input_queue.put(None)

    write_ordered_results(output_queue, writer, frames_read)

    for worker in workers:
        worker.join()

    return time.perf_counter() - start


def run_benchmark(video_path: str, mode: str, max_workers: int) -> None:
    print("workers,time_sec,speedup")
    single_time = run_single(video_path, "benchmark_single.mp4")
    print(f"1,{single_time:.6f},1.000000")

    for workers in range(2, max_workers + 1):
        output = f"benchmark_{mode}_{workers}.mp4"
        if mode == "thread":
            elapsed = run_threads(video_path, output, workers)
        elif mode == "process":
            elapsed = run_processes(video_path, output, workers)
        else:
            raise ValueError("Benchmark mode must be thread or process")

        print(f"{workers},{elapsed:.6f},{single_time / elapsed:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8s-pose inference for video on CPU.")
    parser.add_argument("--video", required=True, help="Input video path, recommended resolution 640x480")
    parser.add_argument("--mode", choices=["single", "thread", "process"], default="single")
    parser.add_argument("--output", default="output_pose.mp4", help="Output video path")
    parser.add_argument("--workers", type=int, default=2, help="Number of threads or processes")
    parser.add_argument("--benchmark", action="store_true", help="Try worker counts from 1 to --workers")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not Path(args.video).exists():
        raise FileNotFoundError(args.video)
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    if args.benchmark:
        run_benchmark(args.video, args.mode, args.workers)
        return 0

    if args.mode == "single":
        elapsed = run_single(args.video, args.output)
    elif args.mode == "thread":
        elapsed = run_threads(args.video, args.output, args.workers)
    else:
        elapsed = run_processes(args.video, args.output, args.workers)

    print(f"mode={args.mode}")
    print(f"workers={args.workers}")
    print(f"time_sec={elapsed:.6f}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
