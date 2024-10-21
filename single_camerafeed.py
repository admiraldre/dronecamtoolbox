# This is for a single camera_feed

import sys
from typing import Optional
from queue import Queue
from vmbpy import *
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# All frames will either be recorded in this format, or transformed to it before being displayed
opencv_display_format = PixelFormat.Bgr8

def print_usage():
    print('Usage:')
    print('    python asynchronous_grab_opencv.py [camera_id]')
    print('    python asynchronous_grab_opencv.py [/h] [-h]')
    print()
    print('Parameters:')
    print('    camera_id   ID of the camera to use (using first camera if not specified)')
    print()

def abort(reason: str, return_code: int = 1, usage: bool = False):
    print(reason + '\n')
    if usage:
        print_usage()
    sys.exit(return_code)

def parse_args() -> Optional[str]:
    args = sys.argv[1:]
    argc = len(args)

    for arg in args:
        if arg in ('/h', '-h'):
            print_usage()
            sys.exit(0)

    if argc > 1:
        abort(reason="Invalid number of arguments. Abort.", return_code=2, usage=True)

    return None if argc == 0 else args[0]

def get_camera(camera_id: Optional[str]) -> Camera:
    with VmbSystem.get_instance() as vmb:
        if camera_id:
            try:
                return vmb.get_camera_by_id(camera_id)
            except VmbCameraError:
                abort('Failed to access Camera \'{}\'. Abort.'.format(camera_id))
        else:
            cams = vmb.get_all_cameras()
            if not cams:
                abort('No Cameras accessible. Abort.')
            return cams[0]

def setup_camera(cam: Camera):
    with cam:
        try:
            cam.ExposureAuto.set('Off')  # Turn off auto exposure
            cam.ExposureTime.set(5000)  # Adjust for your lighting
            cam.Gain.set(20)
            cam.BinningHorizontal.set(1)
            cam.BinningVertical.set(1)
        except (AttributeError, VmbFeatureError):
            pass

def setup_pixel_format(cam: Camera):
    cam_formats = cam.get_pixel_formats()
    cam_color_formats = intersect_pixel_formats(cam_formats, COLOR_PIXEL_FORMATS)
    convertible_color_formats = tuple(f for f in cam_color_formats
                                      if opencv_display_format in f.get_convertible_formats())

    cam_mono_formats = intersect_pixel_formats(cam_formats, MONO_PIXEL_FORMATS)
    convertible_mono_formats = tuple(f for f in cam_mono_formats
                                     if opencv_display_format in f.get_convertible_formats())

    if opencv_display_format in cam_formats:
        cam.set_pixel_format(opencv_display_format)
    elif convertible_color_formats:
        cam.set_pixel_format(convertible_color_formats[0])
    elif convertible_mono_formats:
        cam.set_pixel_format(convertible_mono_formats[0])
    else:
        abort('Camera does not support an OpenCV compatible format. Abort.')

class Handler:
    def __init__(self, publisher: rospy.Publisher):
        self.display_queue = Queue(10)
        self.publisher = publisher
        self.bridge = CvBridge()

    def get_image(self):
        return self.display_queue.get(True)

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            if frame.get_pixel_format() == opencv_display_format:
                display = frame
            else:
                display = frame.convert_pixel_format(opencv_display_format)

            # Convert the frame to a ROS image message and publish it
            ros_image = self.bridge.cv2_to_imgmsg(display.as_opencv_image(), encoding="bgr8")
            self.publisher.publish(ros_image)

            self.display_queue.put(display.as_opencv_image(), True)
        cam.queue_frame(frame)

def main():
    cam_id = parse_args()
    rospy.init_node('camera_publisher', anonymous=True)
    image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=50)

    with VmbSystem.get_instance():
        with get_camera(cam_id) as cam:
            setup_camera(cam)
            setup_pixel_format(cam)
            handler = Handler(image_pub)

            try:
                cam.start_streaming(handler=handler, buffer_count=20)

                msg = 'Stream from \'{}\'. Press <Enter> to stop stream.'.format(cam.get_name())
                import cv2
                ENTER_KEY_CODE = 13
                while True:
                    key = cv2.waitKey(1)
                    if key == ENTER_KEY_CODE:
                        cv2.destroyWindow(msg)
                        break

                    display = handler.get_image()
                    display = cv2.resize(display, (640, 480))  # Resize for display
                    cv2.imshow(msg, display)

            finally:
                cam.stop_streaming()

if __name__ == '__main__':
    main()
