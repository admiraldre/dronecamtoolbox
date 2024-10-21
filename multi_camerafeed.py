#This script is for showing live camera feed for multiple cameras using Multithreading
import copy
import queue
import threading
from typing import Optional

import cv2
import numpy

from vmbpy import *

# Import ROS modules
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rosbag
import time  # For timestamping the bag files

FRAME_QUEUE_SIZE = 10
FRAME_HEIGHT = 2100
FRAME_WIDTH = 2100


def print_preamble():
    print('////////////////////////////////////////')
    print('/// VmbPy Multithreading Example ///////')
    print('////////////////////////////////////////\n')
    print(flush=True)


def add_camera_id(frame: Frame, cam_id: str) -> Frame:
    cv2.putText(frame.as_opencv_image(), 'Cam: {}'.format(cam_id), org=(0, 30), fontScale=1,
                color=255, thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)
    return frame


def resize_if_required(frame: Frame) -> numpy.ndarray:
    cv_frame = frame.as_opencv_image()

    if (frame.get_height() != FRAME_HEIGHT) or (frame.get_width() != FRAME_WIDTH):
        cv_frame = cv2.resize(cv_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        cv_frame = cv_frame[..., numpy.newaxis]

    return cv_frame


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

        if feat_value <= min_:
            val = min_
        elif feat_value >= max_:
            val = max_
        else:
            val = (((feat_value - min_) // inc) * inc) + min_

        feat.set(val)

        msg = ('Camera {}: Failed to set value of Feature \'{}\' to \'{}\': '
               'Using nearest valid value \'{}\'. Note that, this causes resizing '
               'during processing, reducing the frame rate.')
        Log.get_instance().info(msg.format(cam.get_id(), feat_name, feat_value, val))


class FrameProducer(threading.Thread):
    def __init__(self, cam: Camera, frame_queue: queue.Queue):
        threading.Thread.__init__(self)

        self.log = Log.get_instance()
        self.cam = cam
        self.frame_queue = frame_queue
        self.killswitch = threading.Event()

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            if not self.frame_queue.full():
                frame_cpy = copy.deepcopy(frame)
                try_put_frame(self.frame_queue, cam, frame_cpy)

        cam.queue_frame(frame)

    def stop(self):
        self.killswitch.set()

    def setup_camera(self):
        set_nearest_value(self.cam, 'Height', FRAME_HEIGHT)
        set_nearest_value(self.cam, 'Width', FRAME_WIDTH)

        # Try to enable automatic exposure time setting
        try:
            self.cam.ExposureAuto.set('Once')
            self.cam.ExposureTime.set(5000)
            self.cam.Gain.set(25)
            self.cam.BinningHorizontal.set(1)
            self.cam.BinningVertical.set(1)
        except (AttributeError, VmbFeatureError):
            self.log.info('Camera {}: Failed to set Feature \'ExposureAuto\'.'.format(
                          self.cam.get_id()))

        self.cam.set_pixel_format(PixelFormat.Mono8)

    def run(self):
        self.log.info('Thread \'FrameProducer({})\' started.'.format(self.cam.get_id()))

        try:
            with self.cam:
                self.setup_camera()
                try:
                    self.cam.start_streaming(self)
                    self.killswitch.wait()
                finally:
                    self.cam.stop_streaming()
        except VmbCameraError:
            pass
        finally:
            try_put_frame(self.frame_queue, self.cam, None)

        self.log.info('Thread \'FrameProducer({})\' terminated.'.format(self.cam.get_id()))


class FrameConsumer:
    def __init__(self, frame_queue: queue.Queue):
        self.log = Log.get_instance()
        self.frame_queue = frame_queue
        self.bridge = CvBridge()
        self.recording = False
        self.bag = None  # rosbag.Bag instance

    def run(self):
        IMAGE_CAPTION = 'Multithreading Example: Press <Enter> to exit, R to record'
        KEY_CODE_ENTER = 13
        KEY_CODE_R = ord('r')

        frames = {}
        alive = True

        self.log.info('\'FrameConsumer\' started.')

        # Make the OpenCV window resizable
        cv2.namedWindow(IMAGE_CAPTION, cv2.WINDOW_NORMAL)

        while alive:
            # Update current state by dequeuing all currently available frames.
            frames_left = self.frame_queue.qsize()
            while frames_left:
                try:
                    cam_id, frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break

                # Add/Remove frame from current state.
                if frame:
                    frames[cam_id] = frame
                else:
                    frames.pop(cam_id, None)

                frames_left -= 1

            # Construct image by stitching frames together.
            if len(frames) >= 1:
                # Create a dictionary of resized images with camera IDs
                cv_images = {cam_id: resize_if_required(frames[cam_id]) for cam_id in sorted(frames.keys())}

                # Combine images for display
                combined_image = numpy.concatenate(list(cv_images.values()), axis=1)

                # Resize combined_image to fit within desired dimensions
                max_display_height = 1080  # Desired maximum display height
                max_display_width = 1920   # Desired maximum display width

                height, width = combined_image.shape[:2]
                scaling_factor = min(max_display_width / width, max_display_height / height)

                if scaling_factor < 1:
                    new_width = int(width * scaling_factor)
                    new_height = int(height * scaling_factor)
                    resized_combined_image = cv2.resize(combined_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                else:
                    resized_combined_image = combined_image

                cv2.imshow(IMAGE_CAPTION, resized_combined_image)

                # If recording, write each frame to rosbag
                if self.recording:
                    for cam_id, cv_image in cv_images.items():
                        # Convert to ROS Image message
                        ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="mono8")
                        ros_image.header.stamp = rospy.Time.now()
                        topic_name = f'/camera/{cam_id}/image'
                        self.bag.write(topic_name, ros_image, ros_image.header.stamp)
            else:
                cv2.imshow(IMAGE_CAPTION, create_dummy_frame())

            # Check for shutdown condition or key press
            key = cv2.waitKey(10) & 0xFF  # Mask for compatibility
            if key == KEY_CODE_ENTER:
                cv2.destroyAllWindows()
                alive = False
            elif key == KEY_CODE_R:
                if not self.recording:
                    # Start recording
                    self.recording = True
                    bag_filename = time.strftime("%Y%m%d-%H%M%S") + '.bag'
                    self.bag = rosbag.Bag(bag_filename, 'w')
                    print('Started recording to {}'.format(bag_filename))
                else:
                    # Stop recording
                    self.recording = False
                    self.bag.close()
                    self.bag = None
                    print('Stopped recording')

        # Close bag file if still open
        if self.bag:
            self.bag.close()

        self.log.info('\'FrameConsumer\' terminated.')


class Application:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.producers = {}
        self.producers_lock = threading.Lock()

    def __call__(self, cam: Camera, event: CameraEvent):
        # New camera was detected. Create FrameProducer, add it to active FrameProducers
        if event == CameraEvent.Detected:
            with self.producers_lock:
                self.producers[cam.get_id()] = FrameProducer(cam, self.frame_queue)
                self.producers[cam.get_id()].start()

        # An existing camera was disconnected, stop associated FrameProducer.
        elif event == CameraEvent.Missing:
            with self.producers_lock:
                if cam.get_id() in self.producers:
                    producer = self.producers.pop(cam.get_id())
                    producer.stop()
                    producer.join()

    def run(self):
        log = Log.get_instance()
        consumer = FrameConsumer(self.frame_queue)

        vmb = VmbSystem.get_instance()
        vmb.enable_log(LOG_CONFIG_INFO_CONSOLE_ONLY)

        log.info('\'Application\' started.')

        with vmb:
            # Construct FrameProducer threads for all detected cameras
            cameras = vmb.get_all_cameras()
            if len(cameras) < 2:
                print("Please connect at least two cameras.")
                return  # Or handle the error as appropriate

            for cam in cameras:
                self.producers[cam.get_id()] = FrameProducer(cam, self.frame_queue)

            # Start FrameProducer threads
            with self.producers_lock:
                for producer in self.producers.values():
                    producer.start()

            # Run the frame consumer to display the recorded images
            vmb.register_camera_change_handler(self)
            consumer.run()
            vmb.unregister_camera_change_handler(self)

            # Stop all FrameProducer threads
            with self.producers_lock:
                # Initiate concurrent shutdown
                for producer in self.producers.values():
                    producer.stop()

                # Wait for shutdown to complete
                for producer in self.producers.values():
                    producer.join()

        log.info('\'Application\' terminated.')


if __name__ == '__main__':
    print_preamble()
    rospy.init_node('multi_camera_recorder', anonymous=True)
    app = Application()
    app.run()