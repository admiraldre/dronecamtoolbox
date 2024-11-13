import threading
import queue
import copy
import cv2
import numpy as np
import apriltag
from vmbpy import *
import time  # To add delay in frame production

# Constants
FRAME_QUEUE_SIZE = 20
FRAME_HEIGHT = 1800
FRAME_WIDTH = 1800
TAG_SIZE = 0.127  # Tag size in meters (e.g., 10 cm)
FOCAL_LENGTH = 3940  # Example focal length in pixels

def calculate_distance(tag_size, focal_length, tag_width_pixels):
    return (tag_size * focal_length) / tag_width_pixels

def set_nearest_value(cam: Camera, feat_name: str, feat_value: int):
    feat = cam.get_feature_by_name(feat_name)
    try:
        feat.set(feat_value)
    except VmbFeatureError:
        min_, max_ = feat.get_range()
        inc = feat.get_increment()
        if feat_value <= min_:
            val = min_
        elif feat_value >= max_:
            val = (((feat_value - min_) // inc) * inc) + min_
        feat.set(val)
        print(f"Camera {cam.get_id()}: Using nearest valid value {val} for feature '{feat_name}'.")

def resize_if_required(frame: Frame) -> np.ndarray:
    cv_frame = frame.as_numpy_ndarray()
    if (frame.get_height() != FRAME_HEIGHT) or (frame.get_width() != FRAME_WIDTH):
        cv_frame = cv2.resize(cv_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
    # Ensure the image is two-dimensional (grayscale)
    if cv_frame.ndim == 3 and cv_frame.shape[2] == 1:
        cv_frame = cv_frame.squeeze(axis=2)
    return cv_frame

class FrameProducer(threading.Thread):
    def __init__(self, cam: Camera, frame_queue: queue.Queue):
        threading.Thread.__init__(self)
        self.cam = cam
        self.frame_queue = frame_queue
        self.killswitch = threading.Event()

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            frame_cpy = copy.deepcopy(frame)
            try:
                self.frame_queue.put_nowait((cam.get_id(), frame_cpy))
                print(f"Frame added to queue for camera {cam.get_id()}")
                time.sleep(0.05)  # Adjust delay as needed
            except queue.Full:
                print(f"Frame queue is full for camera {cam.get_id()}")
        cam.queue_frame(frame)

    def stop(self):
        self.killswitch.set()

    def setup_camera(self):
        set_nearest_value(self.cam, 'Height', FRAME_HEIGHT)
        set_nearest_value(self.cam, 'Width', FRAME_WIDTH)
        try:
            self.cam.ExposureTime.set(20000)
            self.cam.Gain.set(20)
        except (AttributeError, VmbFeatureError):
            print(f"Camera {self.cam.get_id()}: Failed to set exposure or gain.")
        self.cam.set_pixel_format(PixelFormat.Mono8)
        print(f"Camera {self.cam.get_id()}: Pixel format set to {self.cam.get_pixel_format()}")

    def run(self):
        print(f"Starting FrameProducer for camera {self.cam.get_id()}")
        try:
            vmb = VmbSystem.get_instance()
            with vmb:
                with self.cam:
                    self.setup_camera()
                    self.cam.start_streaming(self)
                    self.killswitch.wait()
                    self.cam.stop_streaming()
        except VmbCameraError as e:
            print(f"Camera {self.cam.get_id()}: {e}")
        finally:
            try:
                self.frame_queue.put_nowait((self.cam.get_id(), None))
            except queue.Full:
                pass

class FrameConsumer:
    def __init__(self, frame_queue: queue.Queue):
        self.frame_queue = frame_queue
        self.detector = apriltag.Detector(
            apriltag.DetectorOptions(
                families='tag36h11',
                nthreads=4,
                quad_decimate=1.0,
                refine_edges=True,
            )
        )
        self.running = True

    def run(self):
        IMAGE_CAPTION = 'AprilTag Detection: Press <Enter> to exit'
        KEY_CODE_ENTER = 13
        frames = {}

        cv2.namedWindow(IMAGE_CAPTION, cv2.WINDOW_NORMAL)
        print("Starting FrameConsumer")

        while self.running:
            try:
                while not self.frame_queue.empty():
                    cam_id, frame = self.frame_queue.get_nowait()
                    if frame:
                        frames[cam_id] = frame
                        print(f"Frame received from queue for camera {cam_id}")
                    else:
                        frames.pop(cam_id, None)
            except queue.Empty:
                pass

            if frames:
                cv_images = {}
                for cam_id in sorted(frames.keys()):
                    frame = frames[cam_id]
                    gray_image = resize_if_required(frame)
                    gray_image = self.detect_and_draw_apriltags(gray_image, cam_id)
                    cv_images[cam_id] = gray_image

                combined_image = np.concatenate(list(cv_images.values()), axis=1)

                max_display_height = 1080
                max_display_width = 1920
                height, width = combined_image.shape[:2]
                scaling_factor = min(max_display_width / width, max_display_height / height)

                if scaling_factor < 1:
                    new_width = int(width * scaling_factor)
                    new_height = int(height * scaling_factor)
                    resized_combined_image = cv2.resize(combined_image, (new_width, new_height),
                                                        interpolation=cv2.INTER_AREA)
                else:
                    resized_combined_image = combined_image

                cv2.imshow(IMAGE_CAPTION, resized_combined_image)
            else:
                dummy_frame = np.zeros((50, 640, 1), np.uint8)
                dummy_frame[:] = 0
                cv2.putText(dummy_frame, 'No Stream available. Please connect a Camera.', org=(30, 30),
                            fontScale=1, color=255, thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)
                cv2.imshow(IMAGE_CAPTION, dummy_frame)

            key = cv2.waitKey(10) & 0xFF
            if key == KEY_CODE_ENTER:
                self.running = False
                cv2.destroyAllWindows()

    def detect_and_draw_apriltags(self, gray_image, cam_id):
        try:
            print(f"Processing frame for camera {cam_id}")
            cv2.imshow(f'Raw Camera Image {cam_id}', gray_image)
            cv2.waitKey(1)

            gray_image_blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
            print("Applied Gaussian blur.")

            detections = self.detector.detect(gray_image_blurred)
            print(f"Number of detections for camera {cam_id}: {len(detections)}")

            image_display = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

            for detection in detections:
                corners = detection.corners.astype(int)
                cv2.polylines(image_display, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
                center = tuple(detection.center.astype(int))
                cv2.circle(image_display, center, radius=5, color=(0, 0, 255), thickness=-1)
                
                tag_width_pixels = np.linalg.norm(corners[0] - corners[1])
                distance = calculate_distance(TAG_SIZE, FOCAL_LENGTH, tag_width_pixels)
                print(f"Camera {cam_id} - Tag ID: {detection.tag_id}, Distance: {distance:.2f} meters")
                
                cv2.putText(image_display, f'ID: {detection.tag_id}, Dist: {distance:.2f}m',
                            (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            return image_display
        except Exception as e:
            print(f"Error during AprilTag detection for camera {cam_id}: {e}")
            return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

class Application:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.producers = {}
        self.producers_lock = threading.Lock()

    def run(self):
        vmb = VmbSystem.get_instance()
        vmb.enable_log(LOG_CONFIG_INFO_CONSOLE_ONLY)
        with vmb:
            cameras = vmb.get_all_cameras()
            if not cameras:
                print("No cameras detected. Please connect at least one camera.")
                return
            else:
                print(f"Number of cameras detected: {len(cameras)}")

            for cam in cameras:
                print(f"Starting producer for camera {cam.get_id()}")
                producer = FrameProducer(cam, self.frame_queue)
                self.producers[cam.get_id()] = producer
                producer.start()

            consumer = FrameConsumer(self.frame_queue)
            consumer.run()

            for producer in self.producers.values():
                print(f"Stopping producer for camera {producer.cam.get_id()}")
                producer.stop()
                producer.join()

if __name__ == '__main__':
    print("Starting Application")
    app = Application()
    app.run()