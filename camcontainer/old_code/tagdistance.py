import threading
import queue
import copy
import cv2
import numpy as np
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "enhanced_python_aprilgrid")))
from src.aprilgrid import Detector
from vmbpy import *

# Constants
FRAME_QUEUE_SIZE = 20
TAG_SIZE = 0.02  # Tag size in meters (e.g., 2 cm)
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


def set_max_resolution_and_fps(cam: Camera):
    """
    Sets the camera's resolution and frame rate to their maximum supported values.
    """
    height_feat = cam.get_feature_by_name('Height')
    width_feat = cam.get_feature_by_name('Width')
    framerate_feat = cam.get_feature_by_name('AcquisitionFrameRate')

    max_height = height_feat.get_range()[1]
    max_width = width_feat.get_range()[1]

    set_nearest_value(cam, 'Height', max_height)
    set_nearest_value(cam, 'Width', max_width)

    if framerate_feat.is_writeable():
        max_framerate = framerate_feat.get_range()[1]
        set_nearest_value(cam, 'AcquisitionFrameRate', int(max_framerate))

    print(f"Camera {cam.get_id()}: Resolution set to {max_width}x{max_height}, FPS maximized.")


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
            except queue.Full:
                print(f"Frame queue is full for camera {cam.get_id()}")
        cam.queue_frame(frame)

    def stop(self):
        self.killswitch.set()

    def setup_camera(self):
        set_max_resolution_and_fps(self.cam)
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
        self.detector = Detector("t16h5b1")
        self.running = True
        self.last_time = {}
        self.frame_counts = {}

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
                        # Initialize FPS tracking for new cameras
                        if cam_id not in self.last_time:
                            self.last_time[cam_id] = time.time()
                            self.frame_counts[cam_id] = 0
                    else:
                        frames.pop(cam_id, None)
            except queue.Empty:
                pass

            if frames:
                # Combine frames into a single image for side-by-side display
                combined_image = None
                for cam_id, frame in frames.items():
                    gray_image = frame.as_numpy_ndarray()
                    gray_image = self.detect_and_draw_apriltags(gray_image, cam_id)

                    # Combine the images horizontally (side by side)
                    if combined_image is None:
                        combined_image = gray_image
                    else:
                        combined_image = np.hstack((combined_image, gray_image))

                cv2.imshow(IMAGE_CAPTION, combined_image)
            else:
                dummy_frame = np.zeros((50, 640, 1), np.uint8)
                cv2.putText(dummy_frame, 'No Stream available. Please connect a Camera.', org=(30, 30),
                           fontScale=1, color=255, thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)
                cv2.imshow(IMAGE_CAPTION, dummy_frame)

            key = cv2.waitKey(10) & 0xFF
            if key == KEY_CODE_ENTER:
                self.running = False
                cv2.destroyAllWindows()

    def detect_and_draw_apriltags(self, gray_image, cam_id):
        try:
            detections = self.detector.detect(gray_image)

            # Convert grayscale image to color for displaying
            image_display = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

            # Draw detected AprilTags on the image
            for detection in detections:
                corners = detection.corners.astype(int)
                cv2.polylines(image_display, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
                center = tuple(detection.center.astype(int))
                cv2.circle(image_display, center, radius=5, color=(0, 0, 255), thickness=-1)

                tag_width_pixels = np.linalg.norm(corners[0] - corners[1])
                distance = calculate_distance(TAG_SIZE, FOCAL_LENGTH, tag_width_pixels)

                # Increase the font scale and adjust position for better visibility
                font_scale = 1.5  # Larger font size
                thickness = 3  # Make the text thicker for better visibility
                color = (0, 255, 0)  # Change text color (green for example)

                text_position = (center[0] + 20, center[1] - 20)  # Change this to suit your needs

                cv2.putText(image_display, f'ID: {detection.tag_id}, Dist: {distance:.2f}m',
                            text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

            # Calculate and display FPS
            self.frame_counts[cam_id] += 1
            current_time = time.time()
            elapsed_time = current_time - self.last_time[cam_id]

            if elapsed_time >= 1.0:  # Update FPS every second
                fps = self.frame_counts[cam_id] / elapsed_time
                self.last_time[cam_id] = current_time
                self.frame_counts[cam_id] = 0

                # Display FPS on the image
                cv2.putText(image_display, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            12, (255, 255, 0), 2)

            return image_display
        except Exception as e:
            print(f"Error during AprilTag detection for camera {cam_id}: {e}")
            return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)


class Application:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.producers = {}

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
                producer = FrameProducer(cam, self.frame_queue)
                self.producers[cam.get_id()] = producer
                producer.start()

            consumer = FrameConsumer(self.frame_queue)
            consumer.run()

            for producer in self.producers.values():
                producer.stop()
                producer.join()


if __name__ == '__main__':
    app = Application()
    app.run()