import argparse
import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")


class SensorX(Sensor):
    """Sensor X"""

    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data


class SensorCam(Sensor):
    def __init__(self, camera_name: str, resolution: tuple[int, int]):
        self._camera_name = camera_name
        self._resolution = resolution
        self._cam = cv2.VideoCapture(self._camera_index(camera_name))

        if not self._cam.isOpened():
            logging.error("Camera '%s' was not found or cannot be opened", camera_name)
            raise RuntimeError(f"Cannot open camera {camera_name}")

        width, height = resolution
        self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get(self):
        ok, frame = self._cam.read()
        if not ok or frame is None:
            logging.error("Camera '%s' read error", self._camera_name)
            raise RuntimeError("Camera read error")
        return frame

    def __del__(self):
        cam = getattr(self, "_cam", None)
        if cam is not None:
            cam.release()

    @staticmethod
    def _camera_index(camera_name: str) -> int | str:
        return int(camera_name) if camera_name.isdigit() else camera_name


class WindowImage:
    def __init__(self, fps: float):
        if fps <= 0:
            logging.error("Display fps must be positive, got %s", fps)
            raise ValueError("Display fps must be positive")

        self._delay_ms = max(1, int(1000 / fps))
        self._name = "Sensors"
        cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)

    def show(self, img) -> bool:
        cv2.imshow(self._name, img)
        key = cv2.waitKey(self._delay_ms) & 0xFF
        return key != ord("q")

    def __del__(self):
        cv2.destroyWindow(getattr(self, "_name", "Sensors"))


def setup_logging() -> None:
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "task4.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def parse_resolution(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Resolution must be like 1280x720") from exc

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Resolution values must be positive")
    return width, height


def put_latest(out_queue: queue.Queue, value: Any) -> None:
    try:
        out_queue.put_nowait(value)
    except queue.Full:
        try:
            out_queue.get_nowait()
        except queue.Empty:
            pass
        out_queue.put_nowait(value)


def sensor_worker(name: str, sensor: Sensor, out_queue: queue.Queue, stop_event: threading.Event) -> None:
    logging.info("Sensor worker '%s' started", name)
    while not stop_event.is_set():
        try:
            put_latest(out_queue, sensor.get())
        except Exception:
            logging.exception("Sensor worker '%s' failed", name)
            stop_event.set()
            break
    logging.info("Sensor worker '%s' stopped", name)


def get_latest(in_queue: queue.Queue, previous: Any) -> Any:
    value = previous
    while True:
        try:
            value = in_queue.get_nowait()
        except queue.Empty:
            return value


def make_image(frame, sensor_values: dict[str, Any], resolution: tuple[int, int]):
    width, height = resolution
    if frame is None:
        image = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        image = cv2.resize(frame, (width, height))

    y = 35
    for name, value in sensor_values.items():
        text = f"{name}: {value}"
        cv2.putText(image, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        y += 35

    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read SensorCam and three SensorX sensors in separate threads.")
    parser.add_argument("--camera", default="0", help="Camera name in system, for example 0 or /dev/video0")
    parser.add_argument("--resolution", type=parse_resolution, default=parse_resolution("1280x720"))
    parser.add_argument("--fps", type=float, default=30.0, help="Display frequency")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    stop_event = threading.Event()
    sensor_queues = {
        "camera": queue.Queue(maxsize=1),
        "sensor0_100Hz": queue.Queue(maxsize=1),
        "sensor1_10Hz": queue.Queue(maxsize=1),
        "sensor2_1Hz": queue.Queue(maxsize=1),
    }

    try:
        sensors = {
            "camera": SensorCam(args.camera, args.resolution),
            "sensor0_100Hz": SensorX(0.01),
            "sensor1_10Hz": SensorX(0.1),
            "sensor2_1Hz": SensorX(1.0),
        }
        window = WindowImage(args.fps)
    except Exception as exc:
        logging.exception("Initialization failed")
        print(f"Initialization failed: {exc}")
        return 1

    threads: list[threading.Thread] = []
    for name, sensor in sensors.items():
        thread = threading.Thread(
            target=sensor_worker,
            args=(name, sensor, sensor_queues[name], stop_event),
            daemon=False,
        )
        thread.start()
        threads.append(thread)

    latest = {name: None for name in sensor_queues}

    try:
        while not stop_event.is_set():
            for name, data_queue in sensor_queues.items():
                latest[name] = get_latest(data_queue, latest[name])

            frame = latest["camera"]
            image = make_image(
                frame,
                {
                    "SensorX 100 Hz": latest["sensor0_100Hz"],
                    "SensorX 10 Hz": latest["sensor1_10Hz"],
                    "SensorX 1 Hz": latest["sensor2_1Hz"],
                },
                args.resolution,
            )

            if not window.show(image):
                stop_event.set()
                break
    except KeyboardInterrupt:
        stop_event.set()
    except Exception:
        logging.exception("Main loop failed")
        stop_event.set()
        return 1
    finally:
        stop_event.set()
        for thread in threads:
            thread.join()

    return 0


if __name__ == "__main__":
    sys.exit(main())
