import copy
import queue
import threading
import time
from typing import Optional

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "enhanced_python_aprilgrid")))
from src.aprilgrid import Detector

import cv2
import numpy

from vmbpy import *

FRAME_QUEUE_SIZE = 10
FRAME_HEIGHT = 3036
FRAME_WIDTH = 4024
DISPLAY_WIDTH = 720
DISPLAY_HEIGHT = 720

detector = Detector('t36h11')  # Initialize AprilGrid detector

cv2.setUseOptimized(True)

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

def set_nearest_value(cam: Camera, feat_name: str, feat_value: int):
    feat = cam.get_feature_by_name(feat_name)

    try:
        feat.set(feat_value)
    except VmbFeatureError:
        min_, max_ = feat.get_range()
        inc = feat.get_increment()

        val = max(min_, min(max_, (((feat_value - min_) // inc) * inc) + min_))
        feat.set(val)

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
        set_nearest_value(self.cam, 'Height', FRAME_HEIGHT)
        set_nearest_value(self.cam, 'Width', FRAME_WIDTH)

        try:
            self.cam.ExposureAuto.set('Off')
            self.cam.ExposureTime.set(20000)
            self.cam.Gain.set(16)
            self.cam.BinningHorizontal.set(1)
            self.cam.BinningVertical.set(1)
        except (AttributeError, VmbFeatureError):
            pass

        self.cam.set_pixel_format(PixelFormat.Mono8)

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
        self.camera_data = {}

    def calculate_fps(self, cam_id: str):
        current_time = time.time()
        if cam_id in self.camera_data:
            last_time = self.camera_data[cam_id]["last_time"]
            elapsed_time = current_time - last_time
            if elapsed_time > 0:
                self.camera_data[cam_id]["fps"] = 1.0 / elapsed_time
            self.camera_data[cam_id]["last_time"] = current_time
        else:
            self.camera_data[cam_id] = {"last_time": current_time, "fps": 0.0}

    def log_frame_info(self, cam_id: str, frame: Frame):
        # Log FPS and resolution to the console
        fps = self.camera_data[cam_id].get("fps", 0.0)
        resolution = (frame.get_width(), frame.get_height())
        print(f"Camera {cam_id} - FPS: {fps:.2f}, Resolution: {resolution}")

    def process_aprilgrid(self, frame: numpy.ndarray):
        detections = detector.detect(frame)
        for detection in detections:
            center = numpy.round(numpy.average(detection.corners, axis=0)).astype(int)
            cv2.putText(frame, f"ID: {detection.tag_id}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            for corner in detection.corners:
                corner = numpy.round(corner).astype(int)
                cv2.circle(frame, tuple(corner), 5, (0, 255, 0), -1)

    def run(self):
        IMAGE_CAPTION = 'DroneCam Multicam View: Press <Enter> to exit'
        KEY_CODE_ENTER = 13

        frames = {}
        alive = True

        cv2.namedWindow(IMAGE_CAPTION, cv2.WINDOW_AUTOSIZE)

        while alive:
            try:
                cam_id, frame = self.frame_queue.get(timeout=0.01)
                if frame:
                    frames[cam_id] = frame
                    self.calculate_fps(cam_id)
                    self.log_frame_info(cam_id, frame)  # Log frame info here

                    cv_frame = resize_if_required(frame)
                    self.process_aprilgrid(cv_frame)
                    frames[cam_id] = cv_frame
                else:
                    frames.pop(cam_id, None)
            except queue.Empty:
                pass

            if frames:
                cv_images = [resize_for_display(frames[cam_id]) for cam_id in sorted(frames.keys())]
                display_frame = numpy.concatenate(cv_images, axis=1)
                cv2.imshow(IMAGE_CAPTION, display_frame)
            else:
                cv2.imshow(IMAGE_CAPTION, create_dummy_frame())

            if KEY_CODE_ENTER == cv2.waitKey(1):
                cv2.destroyAllWindows()
                alive = False

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