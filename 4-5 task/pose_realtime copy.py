from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import time
from multiprocessing import Event, Queue, Value

import cv2
from ultralytics import YOLO

MODEL_NAME = "yolov8n-pose.pt"


class Camera:
    """Захват кадров с веб-камеры."""

    def __init__(self, camera_name: str, width: int, height: int) -> None:
        """Открыть камеру и задать размер кадра."""
        camera_id = int(camera_name) if camera_name.isdigit() else camera_name
        self._cap = cv2.VideoCapture(camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_name}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        """Считать кадр; (ok, frame)."""
        return self._cap.read()

    def release(self) -> None:
        """Освободить камеру."""
        cap = getattr(self, "_cap", None)
        if cap is not None:
            cap.release()
            self._cap = None


class Window:
    """Окно отображения результата."""

    def __init__(self, name: str = "YOLOv8 pose realtime") -> None:
        """Создать именованное окно OpenCV."""
        self._name = name
        self._closed = False
        cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)

    def show(self, frame) -> bool:
        """Показать кадр; False при нажатии q."""
        cv2.imshow(self._name, frame)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def close(self) -> None:
        """Закрыть окно."""
        if self._closed:
            return
        try:
            cv2.destroyWindow(self._name)
        except cv2.error:
            pass
        self._closed = True


def put_latest(out_queue: Queue, value) -> None:
    """Положить значение в очередь, отбрасывая старые при переполнении."""
    for _ in range(32):
        try:
            out_queue.put_nowait(value)
            return
        except queue.Full:
            try:
                out_queue.get_nowait()
            except queue.Empty:
                return

def resolve_device(choice: str) -> tuple[int | str, bool]:
    """Выбрать устройство и флаг half precision для YOLO."""
    if choice == "cpu":
        return "cpu", False

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch required for CUDA. Install cu118 wheels.") from exc

    if choice in ("auto", "cuda"):
        if not torch.cuda.is_available():
            if choice == "cuda":
                raise RuntimeError(
                    "CUDA requested but unavailable. "
                    "Use Python env with torch+cu118 (CUDA 11.8)."
                )
            print("CUDA unavailable, using CPU multiprocessing")
            return "cpu", False

        cuda_ver = torch.version.cuda or "unknown"
        print(f"Using GPU 0: {torch.cuda.get_device_name(0)} (CUDA {cuda_ver})")
        if cuda_ver != "11.8" and not cuda_ver.startswith("11."):
            print(f"Note: this script is tuned for CUDA 11.8; detected {cuda_ver}")
        return 0, True

    return choice, choice != "cpu"


def default_workers(device: int | str, requested: int | None) -> int:
    """Число процессов-воркеров с учётом CPU/GPU."""
    on_gpu = device != "cpu"
    if requested is not None:
        if on_gpu and requested > 1:
            print(f"GPU mode: clamping workers {requested} -> 1 (one model per GPU)")
        return 1 if on_gpu else max(1, requested)

    if on_gpu:
        return 1
    return min(4, mp.cpu_count() or 4)


def _predict_kwargs(device: int | str, imgsz: int, use_half: bool) -> dict:
    """Собрать kwargs для вызова model()."""
    kw: dict = {"verbose": False, "device": device, "imgsz": imgsz}
    if use_half and device != "cpu":
        kw["half"] = True
    return kw


def signal_workers_stop(input_queue: Queue, count: int) -> None:
    """Отправить каждому воркеру сигнал завершения (None)."""
    for _ in range(count):
        while True:
            try:
                input_queue.put_nowait(None)
                break
            except queue.Full:
                try:
                    input_queue.get_nowait()
                except queue.Empty:
                    time.sleep(0.01)


def worker_process(
    worker_id: int,
    input_queue: Queue,
    output_queue: Queue,
    stop_event: Event,
    device: int | str,
    imgsz: int,
    use_half: bool,
    infer_fps: Value,
    worker_ready: Event,
) -> None:
    """Процесс: инференс pose по одному кадру из очереди."""
    if device == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", "1")

    model = YOLO(MODEL_NAME)
    predict_kw = _predict_kwargs(device, imgsz, use_half)
    label = "cuda:0" if device == 0 else str(device)
    print(f"Worker {worker_id}: {MODEL_NAME} on {label}, imgsz={imgsz}, half={predict_kw.get('half', False)}")

    # Warmup (first CUDA kernel compile)
    import numpy as np

    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    model(dummy, **predict_kw)
    worker_ready.set()

    infer_frames = 0
    infer_t0 = time.perf_counter()

    while not stop_event.is_set():
        try:
            frame_data = input_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if frame_data is None:
            break

        frame_id, frame = frame_data
        try:
            result = model(frame, **predict_kw)[0]
            put_latest(output_queue, (frame_id, result.plot()))
        except Exception as exc:
            print(f"Worker {worker_id} inference error: {exc}")
            continue

        infer_frames += 1
        now = time.perf_counter()
        if now - infer_t0 >= 1.0:
            with infer_fps.get_lock():
                infer_fps.value = int(infer_frames / (now - infer_t0))
            infer_frames = 0
            infer_t0 = now

    print(f"Worker {worker_id} stopped")


def batched_worker_process(
    worker_id: int,
    input_queue: Queue,
    output_queue: Queue,
    stop_event: Event,
    batch_size: int,
    device: int | str,
    imgsz: int,
    use_half: bool,
    infer_fps: Value,
    worker_ready: Event,
) -> None:
    """Процесс: инференс pose пакетами кадров (для CPU)."""
    if device == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", "1")

    model = YOLO(MODEL_NAME)
    predict_kw = _predict_kwargs(device, imgsz, use_half)
    print(f"Batched worker {worker_id}: batch_size={batch_size}, device={device}")

    import numpy as np

    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    model(dummy, **predict_kw)
    worker_ready.set()

    batch_frames: list = []
    batch_ids: list[int] = []
    last_batch_time = time.perf_counter()
    infer_frames = 0
    infer_t0 = time.perf_counter()

    def flush_batch() -> None:
        """Обработать накопленный батч и отправить результаты."""
        nonlocal batch_frames, batch_ids, last_batch_time, infer_frames, infer_t0
        if not batch_frames:
            return
        results = model(batch_frames, **predict_kw)
        for fid, res in zip(batch_ids, results):
            put_latest(output_queue, (fid, res.plot()))
        infer_frames += len(batch_frames)
        batch_frames = []
        batch_ids = []
        last_batch_time = time.perf_counter()

        now = time.perf_counter()
        if now - infer_t0 >= 1.0:
            with infer_fps.get_lock():
                infer_fps.value = int(infer_frames / (now - infer_t0))
            infer_frames = 0
            infer_t0 = now

    while not stop_event.is_set():
        try:
            frame_data = input_queue.get(timeout=0.05)
        except queue.Empty:
            if batch_frames and time.perf_counter() - last_batch_time > 0.15:
                flush_batch()
            continue

        if frame_data is None:
            flush_batch()
            break

        frame_id, frame = frame_data
        batch_frames.append(frame)
        batch_ids.append(frame_id)

        if len(batch_frames) >= batch_size or time.perf_counter() - last_batch_time > 0.1:
            flush_batch()

    print(f"Batched worker {worker_id} stopped")


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    cpu_default = min(4, mp.cpu_count() or 4)
    parser = argparse.ArgumentParser(
        description="Realtime yolov8n-pose (CUDA 11.8 / CPU multiprocessing)"
    )
    parser.add_argument("--camera", default="0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--workers", type=int, default=None, help=f"Worker processes (default: 1 on GPU, {cpu_default} on CPU)")
    parser.add_argument("--imgsz", type=int, default=320, help="YOLO inference size")
    parser.add_argument("--batch-size", type=int, default=2, help="CPU batching only")
    parser.add_argument("--use-batching", action="store_true", help="Batch frames (recommended for CPU, not for low-latency GPU)")
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="auto: GPU if cu118 torch sees CUDA, else CPU")
    return parser.parse_args()


def main() -> int:
    """Запустить камеру, воркеры и цикл отображения pose."""
    args = parse_args()
    device, use_half = resolve_device(args.device)
    workers_count = default_workers(device, args.workers)

    if device != "cpu" and args.use_batching:
        print("GPU mode: ignoring --use-batching (single-frame is faster)")
        args.use_batching = False

    print(f"Model: {MODEL_NAME}, device={device}, workers={workers_count}, imgsz={args.imgsz}")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    camera = Camera(args.camera, args.width, args.height)
    window = Window()
    stop_event = Event()
    infer_fps = Value("i", 0)

    input_queue: Queue = Queue(maxsize=args.queue_size)
    output_queue: Queue = Queue(maxsize=2)

    ready_events = [Event() for _ in range(workers_count)]
    workers: list[mp.Process] = []

    for i in range(workers_count):
        common_tail = (device, args.imgsz, use_half, infer_fps, ready_events[i])
        if args.use_batching:
            proc = mp.Process(
                target=batched_worker_process,
                args=(i, input_queue, output_queue, stop_event, args.batch_size, *common_tail),
            )
        else:
            proc = mp.Process(
                target=worker_process,
                args=(i, input_queue, output_queue, stop_event, *common_tail),
            )
        workers.append(proc)
        proc.start()

    print("Loading model in worker(s)...")
    for i, ready in enumerate(ready_events):
        if not ready.wait(timeout=180):
            raise RuntimeError(f"Worker {i} did not become ready in time")

    frame_id = 0
    latest_by_id: dict[int, object] = {}
    display_frames = 0
    display_t0 = time.perf_counter()
    print("Ready. Press q in the window to quit.")

    try:
        while not stop_event.is_set():
            ok, frame = camera.read()
            if not ok:
                print("Camera read error")
                break

            frame_id += 1
            put_latest(input_queue, (frame_id, frame))

            while True:
                try:
                    rid, result_frame = output_queue.get_nowait()
                except queue.Empty:
                    break
                latest_by_id[rid] = result_frame
                if len(latest_by_id) > 8:
                    del latest_by_id[min(latest_by_id)]

            shown = latest_by_id[max(latest_by_id)] if latest_by_id else frame
            display_frames += 1
            now = time.perf_counter()
            overlay = shown.copy()
            if now - display_t0 >= 1.0:
                disp_fps = display_frames / (now - display_t0)
                display_frames = 0
                display_t0 = now
                with infer_fps.get_lock():
                    inf = infer_fps.value
                cv2.putText(
                    overlay,
                    f"display {disp_fps:.1f} | infer {inf}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

            if not window.show(overlay):
                break
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print("Shutting down...")
        stop_event.set()
        signal_workers_stop(input_queue, workers_count)

        for proc in workers:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()

        window.close()
        camera.release()
        print("Done")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
