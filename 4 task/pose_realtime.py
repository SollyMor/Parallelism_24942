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

    def read(self):
        return self._cap.read()

    def __del__(self):
        cap = getattr(self, "_cap", None)
        if cap is not None:
            cap.release()


class Window:
    def __init__(self):
        self._name = "YOLOv8 pose realtime"
        cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)

    def show(self, frame) -> bool:
        cv2.imshow(self._name, frame)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def __del__(self):
        cv2.destroyWindow(getattr(self, "_name", "YOLOv8 pose realtime"))


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    camera = Camera(args.camera, args.width, args.height)
    window = Window()
    stop_event = threading.Event()
    input_queue: queue.Queue = queue.Queue(maxsize=args.workers * 2)
    output_queue: queue.Queue = queue.Queue(maxsize=1)

    workers = [
        threading.Thread(target=worker, args=(input_queue, output_queue, stop_event))
        for _ in range(args.workers)
    ]
    for thread in workers:
        thread.start()

    last_frame = None
    frames = 0
    fps_time = time.perf_counter()

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera read error")

            put_latest(input_queue, frame)

            try:
                last_frame = output_queue.get_nowait()
            except queue.Empty:
                pass

            shown = last_frame if last_frame is not None else frame
            frames += 1
            now = time.perf_counter()
            if now - fps_time >= 1.0:
                fps = frames / (now - fps_time)
                frames = 0
                fps_time = now
                cv2.putText(shown, f"FPS: {fps:.1f}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            if not window.show(shown):
                break
    finally:
        stop_event.set()
        for thread in workers:
            thread.join()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
