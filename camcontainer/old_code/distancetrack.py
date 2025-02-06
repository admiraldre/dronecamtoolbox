import threading
import queue
import cv2
import numpy as np
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "enhanced_python_aprilgrid")))
from src.aprilgrid import Detector
from vmbpy import *

# Constants
FRAME_QUEUE_SIZE = 10
FRAME_HEIGHT = 1518
FRAME_WIDTH = 2008
TAG_SIZE = 0.02  # Tag size in meters
FOCAL_LENGTH = 3940
MAX_WORKERS = 4  # Number of threads for processing

def calculate_distance(tag_size, focal_length, tag_width_pixels):
    return (tag_size * focal_length) / tag_width_pixels

def resize_if_required(frame: Frame) -> np.ndarray:
    cv_frame = frame.as_numpy_ndarray()
    if (frame.get_height() != FRAME_HEIGHT) or (frame.get_width() != FRAME_WIDTH):
        cv_frame = cv2.resize(cv_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_NEAREST)
    return cv_frame

class FrameProducer(threading.Thread):
    def __init__(self, cam: Camera, frame_queue: queue.Queue):
        threading.Thread.__init__(self)
        self.cam = cam
        self.frame_queue = frame_queue
        self.killswitch = threading.Event()

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            try:
                self.frame_queue.put_nowait((cam.get_id(), frame))
            except queue.Full:
                pass
        cam.queue_frame(frame)

    def stop(self):
        self.killswitch.set()

    def setup_camera(self):
        try:
            self.cam.ExposureTime.set(20000)
            self.cam.Gain.set(20)
        except (AttributeError, VmbFeatureError):
            pass
        self.cam.set_pixel_format(PixelFormat.Mono8)

    def run(self):
        try:
            vmb = VmbSystem.get_instance()
            with vmb:
                with self.cam:
                    self.setup_camera()
                    self.cam.start_streaming(self)
                    self.killswitch.wait()
                    self.cam.stop_streaming()
        finally:
            try:
                self.frame_queue.put_nowait((self.cam.get_id(), None))
            except queue.Full:
                pass

class FrameConsumer:
    def __init__(self, frame_queue: queue.Queue):
        self.frame_queue = frame_queue
        self.detector = Detector("t16h5b1")
        self.running = True
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    def run(self):
        IMAGE_CAPTION = 'AprilTag Detection: Press <Enter> to exit'
        KEY_CODE_ENTER = 13
        frames = {}

        cv2.namedWindow(IMAGE_CAPTION, cv2.WINDOW_NORMAL)

        while self.running:
            try:
                cam_id, frame = self.frame_queue.get(timeout=0.01)
                if frame:
                    frames[cam_id] = frame
                else:
                    frames.pop(cam_id, None)
            except queue.Empty:
                pass

            if frames:
                cv_images = {}
                futures = {}
                for cam_id, frame in frames.items():
                    gray_image = resize_if_required(frame)
                    futures[cam_id] = self.executor.submit(self.detect_and_draw_apriltags, gray_image, cam_id)

                for cam_id, future in futures.items():
                    cv_images[cam_id] = future.result()

                combined_image = np.concatenate(list(cv_images.values()), axis=1)
                combined_image = self.display_fps(combined_image)

                max_display_height, max_display_width = 1080, 1920
                height, width = combined_image.shape[:2]
                scaling_factor = min(max_display_width / width, max_display_height / height)

                if scaling_factor < 1:
                    new_size = (int(width * scaling_factor), int(height * scaling_factor))
                    combined_image = cv2.resize(combined_image, new_size, interpolation=cv2.INTER_NEAREST)

                cv2.imshow(IMAGE_CAPTION, combined_image)
            else:
                dummy_frame = np.zeros((50, 640, 1), np.uint8)
                cv2.putText(dummy_frame, 'No Stream available. Please connect a Camera.',
                            org=(30, 30), fontScale=1, color=255, thickness=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
                cv2.imshow(IMAGE_CAPTION, dummy_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == KEY_CODE_ENTER:
                self.running = False

        self.executor.shutdown()
        cv2.destroyAllWindows()

    def detect_and_draw_apriltags(self, gray_image, cam_id):
        try:
            detections = self.detector.detect(gray_image)
            image_display = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

            for detection in detections:
                corners = detection.corners.astype(int)
                cv2.polylines(image_display, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
                center = tuple(detection.center.astype(int))
                cv2.circle(image_display, center, radius=5, color=(0, 0, 255), thickness=-1)

                tag_width_pixels = np.linalg.norm(corners[0] - corners[1])
                distance = calculate_distance(TAG_SIZE, FOCAL_LENGTH, tag_width_pixels)

                cv2.putText(image_display, f'ID: {detection.tag_id}, Dist: {distance:.2f}m',
                            (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            return image_display
        except Exception as e:
            print(f"Detection Error: {e}")
            return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    def display_fps(self, image):
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.last_frame_time

        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.last_frame_time = current_time
            self.frame_count = 0

        cv2.putText(image, f'FPS: {self.fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return image

class Application:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)

    def run(self):
        vmb = VmbSystem.get_instance()
        with vmb:
            cameras = vmb.get_all_cameras()
            if not cameras:
                print("No cameras detected.")
                return

            producers = [FrameProducer(cam, self.frame_queue) for cam in cameras]
            for producer in producers:
                producer.start()

            consumer = FrameConsumer(self.frame_queue)
            consumer.run()

            for producer in producers:
                producer.stop()
                producer.join()

if __name__ == '__main__':
    app = Application()
    app.run()