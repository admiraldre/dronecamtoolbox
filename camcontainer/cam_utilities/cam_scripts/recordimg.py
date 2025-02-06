import os
import copy
import queue
import threading
import time
from typing import Optional

import cv2
import numpy

from vmbpy import *

FRAME_QUEUE_SIZE = 10
FRAME_HEIGHT = 3036
FRAME_WIDTH = 4024
DISPLAY_WIDTH = 720
DISPLAY_HEIGHT = 720

cv2.setUseOptimized(True)

SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def print_preamble():
    print('////////////////////////////////////////')
    print('/// DroneCam Multicam View ///////')
    print('////////////////////////////////////////\n')
    print(flush=True)


def resize_if_required(frame: Frame) -> numpy.ndarray:
    cv_frame = frame.as_opencv_image()
    if (frame.get_height() != FRAME_HEIGHT) or (frame.get_width() != FRAME_WIDTH):
        return cv2.resize(cv_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
    return cv_frame


def resize_for_display(frame: numpy.ndarray) -> numpy.ndarray:
    return cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)


def create_dummy_frame() -> numpy.ndarray:
    cv_frame = numpy.zeros((50, 640, 1), numpy.uint8)
    cv_frame[:] = 0

    cv2.putText(cv_frame, 'No Stream available. Please connect a Camera.', org=(30, 30),
                fontScale=1, color=255, thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)

    return cv_frame


def try_put_frame(q: queue.Queue, cam: Camera, frame: Optional[Frame]):
    try:
        q.put_nowait((cam.get_id(), frame))
    except queue.Full:
        pass


class FrameProducer(threading.Thread):
    def __init__(self, cam: Camera, frame_queue: queue.Queue):
        super().__init__()
        self.cam = cam
        self.frame_queue = frame_queue
        self.killswitch = threading.Event()

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            try_put_frame(self.frame_queue, cam, frame)
        cam.queue_frame(frame)

    def stop(self):
        self.killswitch.set()

    def setup_camera(self):
        try:
            self.cam.set_pixel_format(PixelFormat.Mono8)
        except VmbFeatureError:
            pass

    def run(self):
        with self.cam:
            self.setup_camera()
            try:
                self.cam.start_streaming(self)
                self.killswitch.wait()
            finally:
                self.cam.stop_streaming()
        try_put_frame(self.frame_queue, self.cam, None)


class FrameConsumer:
    def __init__(self, frame_queue: queue.Queue):
        self.frame_queue = frame_queue

    def save_snapshot(self, frame: numpy.ndarray, cam_id: str):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SNAPSHOT_DIR, f"{cam_id}_{timestamp}.png")
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved: {filename}")

    def run(self):
        IMAGE_CAPTION = 'DroneCam Multicam View: Press <Enter> to exit, <S> to take snapshot'
        KEY_CODE_ENTER = 13
        KEY_CODE_SNAPSHOT = ord('s')  # 'S' key to take a snapshot

        frames = {}
        alive = True
        cv2.namedWindow(IMAGE_CAPTION, cv2.WINDOW_AUTOSIZE)

        while alive:
            try:
                cam_id, frame = self.frame_queue.get(timeout=0.01)
                if frame:
                    frames[cam_id] = frame
                else:
                    frames.pop(cam_id, None)
            except queue.Empty:
                pass

            if frames:
                cv_images = [resize_for_display(resize_if_required(frames[cam_id])) for cam_id in sorted(frames.keys())]
                display_frame = numpy.concatenate(cv_images, axis=1)
                cv2.imshow(IMAGE_CAPTION, display_frame)
            else:
                cv2.imshow(IMAGE_CAPTION, create_dummy_frame())

            key = cv2.waitKey(1)
            if key == KEY_CODE_ENTER:
                cv2.destroyAllWindows()
                alive = False
            elif key == KEY_CODE_SNAPSHOT:
                for cam_id, frame in frames.items():
                    cv_frame = resize_if_required(frame)
                    self.save_snapshot(cv_frame, cam_id)


class Application:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.producers = {}
        self.producers_lock = threading.Lock()

    def __call__(self, cam: Camera, event: CameraEvent):
        if event == CameraEvent.Detected:
            with self.producers_lock:
                producer = FrameProducer(cam, self.frame_queue)
                self.producers[cam.get_id()] = producer
                producer.start()
        elif event == CameraEvent.Missing:
            with self.producers_lock:
                producer = self.producers.pop(cam.get_id(), None)
                if producer:
                    producer.stop()
                    producer.join()

    def run(self):
        consumer = FrameConsumer(self.frame_queue)
        with VmbSystem.get_instance() as vmb:
            for cam in vmb.get_all_cameras():
                producer = FrameProducer(cam, self.frame_queue)
                self.producers[cam.get_id()] = producer

            with self.producers_lock:
                for producer in self.producers.values():
                    producer.start()

            vmb.register_camera_change_handler(self)
            consumer.run()
            vmb.unregister_camera_change_handler(self)

            with self.producers_lock:
                for producer in self.producers.values():
                    producer.stop()
                for producer in self.producers.values():
                    producer.join()


if __name__ == '__main__':
    print_preamble()
    app = Application()
    app.run()
