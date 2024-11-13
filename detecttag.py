import threading
import queue
import copy
import cv2
import numpy as np
import apriltag
from vmbpy import *

# Constants
FRAME_QUEUE_SIZE = 10
FRAME_HEIGHT = 1800  # Adjusted frame height
FRAME_WIDTH = 1800   # Adjusted frame width

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
            val = max_
        else:
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
            except queue.Full:
                pass
        cam.queue_frame(frame)

    def stop(self):
        self.killswitch.set()

    def setup_camera(self):
        set_nearest_value(self.cam, 'Height', FRAME_HEIGHT)
        set_nearest_value(self.cam, 'Width', FRAME_WIDTH)
        try:
            # Experiment with different exposure times and gains
            self.cam.ExposureTime.set(20000)  # Increase exposure time
            self.cam.Gain.set(20)             # Increase gain
        except (AttributeError, VmbFeatureError):
            print(f"Camera {self.cam.get_id()}: Failed to set exposure or gain.")
        self.cam.set_pixel_format(PixelFormat.Mono8)
        print(f"Camera {self.cam.get_id()}: Pixel format set to {self.cam.get_pixel_format()}")

    def run(self):
        try:
            # Initialize Vimba in this thread
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
        # Detect both t36h11 and t36h11b1 AprilTag families
        self.detector = apriltag.Detector(
            apriltag.DetectorOptions(
                families='tag36h11,tag36h11b1',
                nthreads=4,
                quad_decimate=1.0,   # Process at full resolution
                refine_edges=True,   # Enable edge refinement
            )
        )
        self.running = True

    def run(self):
        IMAGE_CAPTION = 'AprilTag Detection: Press <Enter> to exit'
        KEY_CODE_ENTER = 13

        frames = {}

        cv2.namedWindow(IMAGE_CAPTION, cv2.WINDOW_NORMAL)

        while self.running:
            try:
                while not self.frame_queue.empty():
                    cam_id, frame = self.frame_queue.get_nowait()
                    if frame:
                        frames[cam_id] = frame
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

                # Resize combined_image to fit within desired dimensions
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
            print(f"Original image shape: {gray_image.shape}, dtype: {gray_image.dtype}")

            # Display the raw grayscale image
            cv2.imshow(f'Raw Camera Image {cam_id}', gray_image)
            cv2.waitKey(1)

            # Apply Gaussian blur to reduce noise and contours
            gray_image_blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
            print("Applied Gaussian blur.")

            # Check pixel values
            print(f"Image min pixel value: {gray_image_blurred.min()}, max pixel value: {gray_image_blurred.max()}")

            detections = self.detector.detect(gray_image_blurred)
            print(f"Number of detections: {len(detections)}")

            # Convert grayscale image to BGR for display
            image_display = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

            for detection in detections:
                corners = detection.corners.astype(int)
                cv2.polylines(image_display, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
                center = tuple(detection.center.astype(int))
                cv2.circle(image_display, center, radius=5, color=(0, 0, 255), thickness=-1)
                tag_id = detection.tag_id
                cv2.putText(image_display, f'ID: {tag_id}', (center[0] + 10, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Optionally save the image for inspection
            # cv2.imwrite(f'captured_frame_{cam_id}.png', image_display)

            return image_display
        except Exception as e:
            print(f"Error during AprilTag detection: {e}")
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

            # Start FrameProducers for all cameras
            for cam in cameras:
                producer = FrameProducer(cam, self.frame_queue)
                self.producers[cam.get_id()] = producer
                producer.start()

            consumer = FrameConsumer(self.frame_queue)
            consumer.run()

            # Stop all producers
            for producer in self.producers.values():
                producer.stop()
                producer.join()

if __name__ == '__main__':
    app = Application()
    app.run()
