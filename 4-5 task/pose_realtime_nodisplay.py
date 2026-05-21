import argparse
import queue
import threading
import time
import cv2
from ultralytics import YOLO

MODEL_NAME = "yolov8s-pose.pt"

class Camera:
    def __init__(self, camera_name: str, width: int, height: int):
        camera_id = int(camera_name) if camera_name.isdigit() else camera_name
        self._cap = cv2.VideoCapture(camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_name}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Получаем FPS исходного видео
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        return self._cap.read()

    def __del__(self):
        cap = getattr(self, "_cap", None)
        if cap is not None:
            cap.release()

def put_latest(out_queue: queue.Queue, value) -> None:
    try:
        out_queue.put_nowait(value)
    except queue.Full:
        try:
            out_queue.get_nowait()
        except queue.Empty:
            pass
        out_queue.put_nowait(value)

def worker(input_queue: queue.Queue, output_queue: queue.Queue, stop_event: threading.Event) -> None:
    model = YOLO(MODEL_NAME)
    while not stop_event.is_set():
        try:
            frame = input_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        result = model(frame, verbose=False, device="cpu")[0]
        put_latest(output_queue, result.plot())

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime YOLOv8s-pose inference from camera.")
    parser.add_argument("--camera", default="0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", default="output.mp4", help="Output video file")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    camera = Camera(args.camera, args.width, args.height)
    stop_event = threading.Event()
    input_queue: queue.Queue = queue.Queue(maxsize=args.workers * 2)
    output_queue: queue.Queue = queue.Queue(maxsize=1)

    workers = [
        threading.Thread(target=worker, args=(input_queue, output_queue, stop_event))
        for _ in range(args.workers)
    ]
    for thread in workers:
        thread.start()

    # Для сохранения видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, camera.fps, (camera.width, camera.height))

    last_frame = None
    frames = 0
    fps_time = time.perf_counter()

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break  # Видео закончилось

            put_latest(input_queue, frame)

            try:
                last_frame = output_queue.get_nowait()
            except queue.Empty:
                pass

            shown = last_frame if last_frame is not None else frame
            
            # Сохраняем кадр
            out.write(shown)
            
            frames += 1
            now = time.perf_counter()
            if now - fps_time >= 1.0:
                fps = frames / (now - fps_time)
                print(f"FPS: {fps:.1f}", end="\r")
                frames = 0
                fps_time = now
    finally:
        print()  # Новая строка
        stop_event.set()
        for thread in workers:
            thread.join()
        out.release()
        print(f"Video saved to {args.output}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
