import math
import copy
import queue
import threading
import json
import time
from typing import Optional
from scipy.spatial.transform import Rotation as R

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
NUM_DETECTOR_THREADS = 4

cv2.setUseOptimized(True)

# --- MODIFIED: Additional constants for skipping & downscaling ---
DETECTION_FRAME_SKIP = 2  # Only detect on every Nth frame
DETECTION_DOWNSCALE = 2   # Downscale factor for detection

def print_preamble():
    print('////////////////////////////////////////')
    print('/// DroneCam Multicam View ///////')
    print('////////////////////////////////////////\n')
    print(flush=True)

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

def try_put_frame(q: queue.Queue, cam: 'Camera', frame: 'Frame'):
    """
    Attempt to put a (camera_id, frame) tuple into the queue without blocking.
    If the queue is full, discard the frame.
    """
    try:
        q.put_nowait((cam.get_id(), frame))
    except queue.Full:
        pass

def set_nearest_value(cam: 'Camera', feat_name: str, feat_value: int):
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

def resize_for_display(frame: np.ndarray) -> np.ndarray:
    """Downscale for display in an OpenCV window."""
    return cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)

# --- MODIFIED: Convert Frame to np.ndarray without extra resizing ---
def frame_to_gray_np(frame: 'Frame') -> np.ndarray:
    """
    Convert a Vmb Frame to a 2D NumPy grayscale image. 
    We assume the camera is set to 4024×3036, Mono8.
    """
    return frame.as_opencv_image()  # Already a grayscale NumPy array

# --- MODIFIED: Downscale for detection (optional) ---
def downscale_for_detection(gray_full: np.ndarray) -> np.ndarray:
    """
    Creates a smaller grayscale image for AprilTag detection to reduce CPU load.
    The ratio is controlled by DETECTION_DOWNSCALE.
    """
    if DETECTION_DOWNSCALE <= 1:
        return gray_full  # No downscaling
    h, w = gray_full.shape[:2]
    new_w = w // DETECTION_DOWNSCALE
    new_h = h // DETECTION_DOWNSCALE
    return cv2.resize(gray_full, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

# --- MODIFIED: Scale corners back up to full resolution after detection ---
def upscale_corners(corners, scale_factor_x, scale_factor_y):
    """
    Scale AprilTag corner coordinates back to the full-sized image if detection was done on
    a downscaled version.
    """
    return [corner * np.array([scale_factor_x, scale_factor_y]) for corner in corners]


# ------------- FrameProducer -------------
class FrameProducer(threading.Thread):
    """
    Captures frames from a camera in a separate thread.
    For each completed frame, adds it to the shared frame_queue.
    """

    def __init__(self, cam: 'Camera', frame_queue: queue.Queue):
        super().__init__()
        self.cam = cam
        self.frame_queue = frame_queue
        self.killswitch = threading.Event()

    def __call__(self, cam: 'Camera', stream: 'Stream', frame: 'Frame'):
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
            self.cam.Gain.set(0.000000001)
            self.cam.BinningHorizontal.set(2)
            self.cam.BinningVertical.set(2)

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
    runs AprilTag detection on a downscaled image (if enabled),
    and puts results into a shared detection_result_queue.
    """

    def __init__(self, detection_queue: queue.Queue, detection_result_queue: queue.Queue):
        super().__init__()
        self.detection_queue = detection_queue
        self.detection_result_queue = detection_result_queue
        self.detector = Detector('t16h5b1')
        self.killswitch = threading.Event()

        self.intrinsics_cam0 = np.array([3431.6333053510584, 3425.4233661235844, 1037.0631103881203, 814.2748160563247])

        self.intrinsics_cam1 = np.array([3450.443379487295, 3442.6715206818894, 1000.8126349532237, 768.292008152334])

        self.T_cn_cnm1 = np.array([
            [0.9954706102064841, -0.002938725639798721, 0.09502435533453286, -0.1898177118318758],
            [0.002702420629833145, 0.999992928337077, 0.0026153773296546125, -0.001517588623992596],
            [-0.09503136923073889, -0.002346735488078846, 0.9954715122466737, 0.01757994327060201],
            [0.0, 0.0, 0.0, 1.0]
        ])

        self.distortion_coeffs_cam0 = np.array([-0.29763392127051924, 0.8835065755916732, -0.0023637256530460084, -0.0007121489154046612])
        self.distortion_coeffs_cam1 = np.array([-0.2760132481960459, 0.7026370095608109, -0.0015314823046884355, -0.001072650889485783])
        
        R = self.T_cn_cnm1[:3, :3]  # Rotation matrix (3x3)
        T = self.T_cn_cnm1[:3, 3]   # Translation vector (3x1)
        
        K1 = np.array([[self.intrinsics_cam0[0], 0, self.intrinsics_cam0[2]], [0, self.intrinsics_cam0[1], self.intrinsics_cam0[3]], [0, 0, 1]])
        K2 = np.array([[self.intrinsics_cam1[0], 0, self.intrinsics_cam1[2]], [0, self.intrinsics_cam1[1], self.intrinsics_cam1[3]], [0, 0, 1]])
        
        dist_coeffs1 = self.distortion_coeffs_cam0
        dist_coeffs2 = self.distortion_coeffs_cam1
        
        image_size = (2012, 1518)
        
        R1_rect, R2_rect, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, dist_coeffs1, K2, dist_coeffs2, image_size, R, T
        )
        
        map1x, map1y = cv2.initUndistortRectifyMap(K1, dist_coeffs1, R1_rect, P1, image_size, cv2.CV_32F)
        map2x, map2y = cv2.initUndistortRectifyMap(K2, dist_coeffs2, R2_rect, P2, image_size, cv2.CV_32F)



        # Camera intrinsics for distance calculation
        self.camera_intrinsics = {
            "DEV_1AB22C00E123": [7061.970133146324, 7055.239263382366, 2129.3033141629267, 1627.5503251130253],
            "DEV_1AB22C00E588": [7074.560676188498, 7062.053568655379, 2037.2857092571062, 1549.725255281428]
        }

        # cam dis_coefficient
        self.camera_dis = {
            "DEV_1AB22C00E123": [-0.29254001960563103, 0.8273391822433487, -0.0019010343025743829, -7.179588729872542e-05],
            "DEV_1AB22C00E588": [-0.2706765105538779, 0.6911079417360364, -0.0008490143333884144, -0.0005189366405546061]
        }

        # Physical size of the AprilTag (in meters)
        self.tag_size = 0.018  # Example: 16.2 cm

    def stop(self):
        self.killswitch.set()

    def run(self):
        while not self.killswitch.is_set():
            try:
                # Get a (cam_id, frame) pair from the detection queue
                cam_id, frame = self.detection_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            if frame is None:
                # Camera or stream ended
                self.detection_result_queue.put((cam_id, None, None))
                continue

            # Convert to grayscale full resolution
            gray_full = frame_to_gray_np(frame)

            # Downscale for detection
            gray_small = downscale_for_detection(gray_full)

            # Perform detection at lower res
            detections_small = self.detector.detect(gray_small)

            # Scale corners back to full resolution
            scale_x = gray_full.shape[1] / gray_small.shape[1]
            scale_y = gray_full.shape[0] / gray_small.shape[0]
            detections_fullres = []

            for det in detections_small:
                scaled_corners = upscale_corners(det.corners, scale_x, scale_y)

                # when detection
                
                
                # Append detection result
                detections_fullres.append({
                    'tag_id': det.tag_id,
                    'corners': scaled_corners
                })

            # Push results back
            self.detection_result_queue.put((cam_id, gray_full, detections_fullres))

# ------------- FrameConsumer -------------
class FrameConsumer:
    """
    Main consumer class that:
      1) Pulls frames from the frame_queue.
      2) Hands them off to the parallel detector threads (detection_queue),
         optionally skipping frames to reduce CPU load.
      3) Receives detection results (detection_result_queue).
      4) Draws detection overlays and displays in a window.
    """

    def __init__(self, frame_queue: queue.Queue, num_detector_threads: int = 4):
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

        # --- MODIFIED: We’ll track how many frames we’ve handled per camera for skipping ---
        self.frame_counts = {}

    # def log_frame_info(self, cam_id: str, frame: 'Frame'):
    #     """Log frame resolution."""
    #     resolution = (frame.get_width(), frame.get_height())
    #     print(f"Camera {cam_id} - Resolution: {resolution}")

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
                    # Valid frame
                    # self.log_frame_info(cam_id, frame)
                    self.last_frames[cam_id] = frame

                    # --- MODIFIED: Counting frames per camera to skip if needed ---
                    if cam_id not in self.frame_counts:
                        self.frame_counts[cam_id] = 0
                    self.frame_counts[cam_id] += 1

                    # Only offload to detector on every Nth frame to reduce CPU load
                    if self.frame_counts[cam_id] % DETECTION_FRAME_SKIP == 0:
                        try:
                            self.detection_queue.put_nowait((cam_id, frame))
                        except queue.Full:
                            pass
                else:
                    # Frame is None => camera missing or ended
                    self.last_frames.pop(cam_id, None)
                    self.latest_results.pop(cam_id, None)
                    try:
                        self.detection_queue.put_nowait((cam_id, None))
                    except queue.Full:
                        pass

            except queue.Empty:
                # No new frames at the moment
                pass

            # 2. Collect detection results
            while True:
                try:
                    cam_id_res, gray_res, detections_res = self.detection_result_queue.get_nowait()
                    if gray_res is None or detections_res is None:
                        # If detection result is None, the camera ended
                        self.latest_results.pop(cam_id_res, None)
                    else:
                        # Store the latest detection results
                        self.latest_results[cam_id_res] = (gray_res, detections_res)
                except queue.Empty:
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
                    for det in detections:
                        tag_id = det['tag_id']
                        corners = det['corners']  # Already scaled to full res

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
                # If no detections or no cameras, show dummy frame
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

    def __call__(self, cam: 'Camera', event: CameraEvent):
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