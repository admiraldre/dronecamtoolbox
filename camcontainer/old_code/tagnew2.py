import copy
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np
from vmbpy import *

# -----------------------------------------------------------------------------
# --- ADDED FOR APRILTAG DETECTION ---
import sys
import os

# If aprilgrid is under ../src:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "enhanced_python_aprilgrid", "src")))
from aprilgrid import Detector  
import numpy as np
# -----------------------------------------------------------------------------

# ------------- Constants -------------
FRAME_QUEUE_SIZE = 10
FRAME_HEIGHT = 3036
FRAME_WIDTH = 4024
DISPLAY_WIDTH = 720
DISPLAY_HEIGHT = 720

# Number of parallel threads for AprilTag detection
NUM_DETECTOR_THREADS = 4

cv2.setUseOptimized(True)

# ------------- Helper Functions -------------
def print_preamble():
    print('////////////////////////////////////////')
    print('/// DroneCam Multicam View ///////')
    print('////////////////////////////////////////\n')
    print(flush=True)

def resize_if_required(frame: Frame) -> np.ndarray:
    """
    Convert a Vmb Frame to a NumPy grayscale image,
    and resize it to the target resolution if necessary.
    """
    cv_frame = frame.as_opencv_image()  # Mono8 (grayscale) as NumPy array
    if (frame.get_height() != FRAME_HEIGHT) or (frame.get_width() != FRAME_WIDTH):
        return cv2.resize(cv_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
    return cv_frame

def resize_for_display(frame: np.ndarray) -> np.ndarray:
    """Downscale for display in an OpenCV window."""
    return cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)

def create_dummy_frame() -> np.ndarray:
    """
    If no cameras are streaming, show a dummy image.
    """
    cv_frame = np.zeros((50, 640, 1), np.uint8)
    cv2.putText(
        cv_frame,
        'No Stream available. Please connect a Camera.',
        org=(30, 30),
        fontScale=1,
        color=255,
        thickness=1,
        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL
    )
    return cv_frame

def try_put_frame(q: queue.Queue, cam: Camera, frame: Frame):
    """
    Attempt to put a (camera_id, frame) tuple into the queue without blocking.
    If the queue is full, discard the frame.
    """
    try:
        q.put_nowait((cam.get_id(), frame))
    except queue.Full:
        pass

def set_nearest_value(cam: Camera, feat_name: str, feat_value: int):
    """
    Set a camera feature to a given value, snapping to the nearest valid increment if needed.
    """
    feat = cam.get_feature_by_name(feat_name)
    try:
        feat.set(feat_value)
    except VmbFeatureError:
        min_, max_ = feat.get_range()
        inc = feat.get_increment()
        val = max(min_, min(max_, (((feat_value - min_) // inc) * inc) + min_))
        feat.set(val)

# ------------- FrameProducer -------------
class FrameProducer(threading.Thread):
    """
    Captures frames from a camera in a separate thread.
    For each completed frame, adds it to the shared frame_queue.
    """

    def __init__(self, cam: Camera, frame_queue: queue.Queue):
        super().__init__()
        self.cam = cam
        self.frame_queue = frame_queue
        self.killswitch = threading.Event()

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        """Callback for camera frames."""
        if frame.get_status() == FrameStatus.Complete:
            try_put_frame(self.frame_queue, cam, frame)
        cam.queue_frame(frame)

    def stop(self):
        """Stop this thread's loop."""
        self.killswitch.set()

    def setup_camera(self):
        """Configure camera parameters."""
        set_nearest_value(self.cam, 'Height', FRAME_HEIGHT)
        set_nearest_value(self.cam, 'Width', FRAME_WIDTH)

        try:
            self.cam.ExposureAuto.set('Off')
            self.cam.ExposureTime.set(20000)
            self.cam.Gain.set(16)
            self.cam.BinningHorizontal.set(1)
            self.cam.BinningVertical.set(1)

            self.cam.AcquisitionFrameRateEnable.set(True)
            self.cam.AcquisitionFrameRate.set(30.0)

        except (AttributeError, VmbFeatureError):
            pass

        # Force Mono8 pixel format
        self.cam.set_pixel_format(PixelFormat.Mono8)

    def run(self):
        """Main thread loop: open camera, start streaming until killswitch is set."""
        with self.cam:
            self.setup_camera()
            try:
                self.cam.start_streaming(self)
                self.killswitch.wait()
            finally:
                self.cam.stop_streaming()
        # Signal to consumer that camera is missing by sending None
        try_put_frame(self.frame_queue, self.cam, None)

# ------------- DetectorThread -------------
class DetectorThread(threading.Thread):
    """
    A single worker thread that pulls frames from a shared detection_queue,
    runs AprilTag detection, and puts results into a shared detection_result_queue.
    """

    def __init__(self, detection_queue: queue.Queue, detection_result_queue: queue.Queue):
        super().__init__()
        self.detection_queue = detection_queue
        self.detection_result_queue = detection_result_queue
        self.detector = Detector('t16h5b1')
        self.killswitch = threading.Event()

    def stop(self):
        self.killswitch.set()

    def run(self):
        while not self.killswitch.is_set():
            try:
                # Get a (cam_id, frame) pair from the detection queue
                cam_id, frame = self.detection_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            # If frame is None, indicates camera or streaming ended
            if frame is None:
                # Pass this info along to the result queue if needed
                self.detection_result_queue.put((cam_id, None, None))
                continue

            # Convert to grayscale & resize if needed
            gray = resize_if_required(frame)

            # Perform detection
            detections = self.detector.detect(gray)

            # Push results back
            self.detection_result_queue.put((cam_id, gray, detections))

# ------------- FrameConsumer -------------
class FrameConsumer:
    """
    Main consumer class that:
      1) Pulls frames from the frame_queue.
      2) Hands them off to the parallel detector threads (detection_queue).
      3) Receives detection results (detection_result_queue).
      4) Draws detection overlays and displays in a window.
    """

    def __init__(self, frame_queue: queue.Queue, num_detector_threads: int = 4):
        # The main queue from the producers
        self.frame_queue = frame_queue

        # Queues for handing off frames to the detector threads
        self.detection_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.detection_result_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)

        # Create multiple parallel detector threads
        self.detector_threads = []
        for i in range(num_detector_threads):
            dt = DetectorThread(self.detection_queue, self.detection_result_queue)
            dt.start()
            self.detector_threads.append(dt)

        # For tracking FPS, etc.
        self.camera_data = {}

        # Latest frames and detection results by camera ID
        self.last_frames = {}
        self.latest_results = {}

    def calculate_fps(self, cam_id: str):
        """Simple FPS calculation per camera."""
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
        """Print FPS and resolution to console."""
        fps = self.camera_data[cam_id].get("fps", 0.0)
        resolution = (frame.get_width(), frame.get_height())
        print(f"Camera {cam_id} - FPS: {fps:.2f}, Resolution: {resolution}")

    def run(self):
        """Main loop: fetch frames, pass to detector, fetch results, draw and display."""
        IMAGE_CAPTION = 'DroneCam Multicam View: Press <Enter> to exit'
        KEY_CODE_ENTER = 13

        alive = True
        cv2.namedWindow(IMAGE_CAPTION, cv2.WINDOW_AUTOSIZE)

        while alive:
            # 1. Retrieve new frames from producers
            try:
                cam_id, frame = self.frame_queue.get(timeout=0.01)
                if frame:
                    # We have a valid frame
                    self.calculate_fps(cam_id)
                    self.log_frame_info(cam_id, frame)
                    self.last_frames[cam_id] = frame

                    # Offload to one of the detector threads
                    try:
                        self.detection_queue.put_nowait((cam_id, frame))
                    except queue.Full:
                        # If detection queue is full, consider dropping or handling differently
                        pass
                else:
                    # Frame is None => camera missing or ended
                    self.last_frames.pop(cam_id, None)
                    self.latest_results.pop(cam_id, None)
                    # Notify detector threads that this camera ended
                    try:
                        self.detection_queue.put_nowait((cam_id, None))
                    except queue.Full:
                        pass

            except queue.Empty:
                # No new frames at the moment
                pass

            # 2. Collect detection results from any detector thread
            while True:
                try:
                    cam_id_res, gray_res, detections_res = self.detection_result_queue.get_nowait()
                    if gray_res is None or detections_res is None:
                        # If the detection result is None, the camera is missing or ended
                        self.latest_results.pop(cam_id_res, None)
                    else:
                        # Store the latest detection results
                        self.latest_results[cam_id_res] = (gray_res, detections_res)
                except queue.Empty:
                    # No more results to process
                    break

            # 3. Display results
            if self.latest_results:
                processed_frames = []
                # Sort camera IDs for a consistent layout
                for cid in sorted(self.latest_results.keys()):
                    gray, detections = self.latest_results[cid]

                    # Convert to BGR to draw color overlays
                    color_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                    # Draw AprilTag corners and IDs
                    for detection in detections:
                        tag_id = detection.tag_id
                        corners = np.array(detection.corners)

                        # Draw corners in different colors
                        for i, corner in enumerate(corners):
                            corner = corner.flatten().astype(int)
                            if i == 0:
                                color = (0, 0, 255)      # Red
                            elif i == 1:
                                color = (0, 255, 0)      # Green
                            elif i == 2:
                                color = (255, 0, 0)      # Blue
                            else:
                                color = (0, 255, 255)    # Yellow
                            cv2.circle(color_frame, tuple(corner), 8, color, -1)

                        # Draw lines between corners
                        for i in range(4):
                            start = tuple(corners[i].flatten().astype(int))
                            end = tuple(corners[(i + 1) % 4].flatten().astype(int))
                            cv2.line(color_frame, start, end, (255, 255, 255), 2)

                        # Put ID text at the center
                        center = np.mean(corners, axis=0).flatten().astype(int)
                        cv2.putText(color_frame, str(tag_id), tuple(center),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # Resize for display
                    display_frame = resize_for_display(color_frame)
                    processed_frames.append(display_frame)

                # Concatenate side by side
                if processed_frames:
                    display_frame = np.concatenate(processed_frames, axis=1)
                    cv2.imshow(IMAGE_CAPTION, display_frame)
            else:
                # If no detections, show the dummy frame
                cv2.imshow(IMAGE_CAPTION, create_dummy_frame())

            # 4. Check for user exit
            if KEY_CODE_ENTER == cv2.waitKey(1):
                cv2.destroyAllWindows()
                alive = False

        # Stop the parallel detector threads gracefully
        for dt in self.detector_threads:
            dt.stop()
        for dt in self.detector_threads:
            dt.join()

# ------------- Application -------------
class Application:
    """
    The main application orchestrates camera detection, production, and consumption.
    """

    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.producers = {}
        self.producers_lock = threading.Lock()

    def __call__(self, cam: Camera, event: CameraEvent):
        """
        Camera event callback for hotplug events (Detected / Missing).
        """
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
        """
        Start all cameras currently available, register event handlers, and launch the consumer.
        """
        consumer = FrameConsumer(self.frame_queue, num_detector_threads=NUM_DETECTOR_THREADS)
        with VmbSystem.get_instance() as vmb:
            # Start producers for all currently detected cameras
            for cam in vmb.get_all_cameras():
                producer = FrameProducer(cam, self.frame_queue)
                self.producers[cam.get_id()] = producer

            # Start all producers
            with self.producers_lock:
                for producer in self.producers.values():
                    producer.start()

            # Register camera hotplug callback
            vmb.register_camera_change_handler(self)

            # Run the consumer loop (blocks until user quits)
            consumer.run()

            # Unregister callback once done
            vmb.unregister_camera_change_handler(self)

            # Stop all producers
            with self.producers_lock:
                for producer in self.producers.values():
                    producer.stop()
                for producer in self.producers.values():
                    producer.join()

# ------------- Main Entrypoint -------------
if __name__ == '__main__':
    print_preamble()
    app = Application()
    app.run()